@trajectory_forge/run_generate.py @trajectory_forge/README.md 阅读当前的项目代码，当前的代码效果并不好。我让gpt-pro给我出了一个方案，请你阅读下面的方案，然后结合代码和方案，给出你的看法。不需要考虑API的成本，一切为了效果！
```
有。你现在的问题，不是 prompt 再润色一下就能解决，而是**控制架构本身错位了**：你让 MLLM 同时负责“选工具、定方向、猜精确连续参数、承担回滚责任”，这对它来说太重了。

我建议把整个系统改成一句话：

**让 MLLM 负责“离散决策与解释”，让搜索器/渲染器负责“连续参数求解”，让提交器负责“只接受变好的状态”。**

这也是近期图像编辑系统更稳定的方向：有工作已经把图像编辑视作长时程决策并用 tree search / MCTS 做规划；也有工作发现单次差异判断很不稳，因此用**多候选**而不是单候选；还有工作在黑盒 photo finishing tuning 里直接把 **current image + goal image + photo statistics + history** 作为状态去做连续参数搜索，而不是纯靠一步步试探。([arXiv][1])

---

## 我建议的主方案：隐藏搜索 + 单调提交 + 参数响应探针

你的训练数据里，最终仍然是一条干净的长链：

`src -> step1 -> step2 -> ... -> final`

但是**生成这条链的时候**，内部不要再是单分支串行对话，而要允许：

* 多候选并行试探
* 更差结果直接丢弃
* 局部数值搜索
* 回滚到 best-so-far
* 最后只导出“被接受的步骤”

也就是：

**导出轨迹 = 单链**

**内部生成 = 树搜索 / beam search / probe-and-select**

这一步改完，1 和 2 基本就被解决了一大半。

---

## 一、先解决问题 1：避免在 exposure 上来回震荡

你现在是“每轮只走一个动作、且动作一旦输出就推进状态”，所以非常容易出现：

* 模型判断方向对了，但步长错了
* 下一轮看到新图后又反向纠偏
* 然后再反向纠偏
* 整条链浪费在一个参数上

### 改法 1：把“单候选一步走”改成“每步内部多候选搜索”

每个外部 step，不是让模型只报一个动作，而是这样做：

1. **工具候选 shortlist**
   先根据 residual stats 和优先级，选 2 个最可能的工具，而不是只给 1 个。

2. **每个工具生成多个候选**
   不要让模型只输出一个精确值，改成输出：

   * 工具名
   * 方向（increase / decrease）
   * 幅度区间（例如 `+8~+16`，不是精确 `+12`）
   * 简短理由

   每个工具出 2~3 个候选区间。

3. **内部渲染 probe**
   对每个候选区间，渲染一个小梯子，比如 exposure 候选 `+8~+16`，内部实际试：

   * `+8`
   * `+12`
   * `+16`

4. **选最优 probe，再局部 refine**
   只保留 score 最好的，再在附近做一次 local refine。

RetouchLLM 已经观察到“单次差异判断不稳定”，因此用多候选来提高命中正确方向的概率；他们在一个亮度区间预测实验里，多候选设置明显优于单候选。([arXiv][2])

### 改法 2：加“反向震荡约束”

给每个参数加简单但很有效的 anti-oscillation 规则：

* 同一参数连续两步出现**符号翻转**时，若收益没有显著提升，则禁止再次翻转
* 同一工具连续被接受超过 2 次，必须满足“仍有足够增益”才能继续
* 某工具连续 2 次 proposal 被 reject，就进入 cooldown 1 步
* 某维 residual 低于阈值后，该工具“锁定”，除非全局 score 明显恶化才重新开放

这会直接干掉你举的 `-10 -> +12 -> -10 -> +12` 这种循环。

---

## 二、解决问题 2：更差结果绝不能推进状态

这个问题其实最关键。

你现在的状态推进逻辑是：

> 只要模型给了下一步，就把 current 推到新图

这个在优化问题里是致命的。你应该维护三个概念：

* `incumbent_best`：当前**最佳已接受状态**
* `trial_state`：这次试出来的候选状态
* `frontier`：如果用 beam search，就是当前保留的若干最佳节点

### 正确规则应该是：

**只有 trial 比 incumbent_best 更好，才允许提交。**

否则：

* 不推进 current
* 不把它放进 history
* 不把这张更差图喂给下一轮
* 只把它记到 `rejected_candidates` 里

### 建议的 commit gate

定义一个真正用于提交的目标分数，而不是只看某一个指标。比如：

```python
score = (
    w1 * norm_delta_e +
    w2 * lpips +
    w3 * (1 - ssim) +
    w4 * stat_residual +
    w5 * edit_cost
)
# 越小越好
```

提交条件：

```python
accept if score_new < score_best - epsilon
```

如果不满足：

* 直接 reject
* 缩小 trust region
* 或切换工具
* 或停止

### 最重要的一句

**下一轮看到的“当前图”，应该是 best accepted image，而不是 last rendered image。**

这会立刻修正你第 2 个问题。

---

## 三、解决问题 3：不要再让模型直接猜精确参数值

这是根因中的根因。

模型知道“当前图比目标暗”，不代表它知道：

* exposure `+8` 在这张图上会发生什么
* exposure `+8` 和 brightness `+8` 的区别
* tone_tool 里 5 个参数联动后会怎样
* HSL 某个 band 调 10 和调 20 的边际效应差多少

### 核心改法：每一步先做“参数响应探针”

也就是先问引擎，不是先问模型。

对当前 accepted state，在每个候选工具附近做一个**局部有限差分**：

#### 例子：exposure_tool

当前状态下试：

* exposure `+4`
* exposure `+8`
* exposure `+12`

测出真实变化：

* `Δbrightness`
* `ΔL`
* `Δp10/p90`
* `ΔDeltaE`
* `ΔLPIPS`

然后把这个结果组织成一张 effect table：

```text
probe_1: exposure +4   -> brightness +3.1, L +2.9, DeltaE 12.4 -> 10.8
probe_2: exposure +8   -> brightness +6.5, L +6.0, DeltaE 12.4 -> 8.9
probe_3: exposure +12  -> brightness +10.8, L +9.9, DeltaE 12.4 -> 9.6
```

这样模型就不再是“盲猜 +8”，而是“在当前状态下，从真实 probe 里选择最合适的一个”。

### 更强一点：做局部 Jacobian

因为你的引擎是可调用黑盒，完全可以估计：

[
J_{\text{tool}} = \frac{\partial \text{stat residual}}{\partial \text{params}}
]

然后对每个工具解一个受约束的小优化：

[
\Delta p^* = \arg\min_{\Delta p}
| W(r + J\Delta p) |^2 + \lambda |\Delta p|^2
]

其中：

* `r` 是当前 residual stat
* `J` 是这个工具在当前状态下的局部响应
* `Δp` 是这一步参数改变量
* 加 bounds / trust region

这一步做完，**参数大小就不再由 MLLM 猜，而是由当前图像状态下的真实响应决定。**

### 角色重新分工

所以我建议明确分层：

* **MLLM**：决定“现在先修什么问题、选哪个工具、方向大概是什么”
* **probe/searcher**：决定“精确数值到底是多少”
* **evaluator**：决定“这个候选能不能提交”

---

## 四、推荐你直接改成 beam search，而不是单链贪心

如果你说“API 成本不考虑，效果优先”，那我不会再坚持单链在线生成。

我会直接做：

### 外层：beam search / 小规模树搜索

每一层保留 `B=3~5` 个最佳 accepted state。

### 每个节点扩展时

对 1~2 个最可能工具，各生成 2~3 个 magnitude bucket，再通过 probe + local search 得到若干 child。

### 保留 top-B

按 score 和多样性保留 top-B。

### 最终导出 best leaf 的路径

最后只导出从 root 到 best leaf 的这条链。

PhotoAgent 之所以用 tree search / MCTS，本质上就是因为图像编辑是一个长时程规划问题，单步贪心很容易早期走偏。([arXiv][1])

对你这个 paired 数据设定来说，这种做法尤其合适，因为你有 target，可以直接算 reward。
换句话说：

**生成数据阶段可以“作弊”使用 target-aware oracle evaluator；部署时再要求模型学会不作弊。**

这完全合理。

---

## 五、你现在的 prompt 也要改：别再让它直接报 final scalar

我建议改成两段式。

### 阶段 A：规划 prompt

只让模型输出：

* unresolved issue
* tool
* direction
* magnitude bucket
* 为什么现在该用这个工具

例如：

```xml
<thinking>
The image is still darker than target, especially in midtones.
Exposure is still unresolved and has higher priority than white balance.
A moderate positive exposure change is likely needed.
</thinking>
<proposal>
tool: exposure_tool
direction: increase
magnitude_bucket: [8, 16]
reason: global luminance remains low while contrast is already close
</proposal>
```

### 阶段 B：选择 prompt

把 probe table / 缩略图喂给模型，让它只做**候选选择**，而不是发明数值：

```xml
<thinking>
Probe 2 improves brightness and DeltaE the most without clipping highlights.
</thinking>
<select>
candidate_id: probe_2
</select>
```

这样模型就回到它擅长的地方：比较、解释、排序。

---

## 六、为了生成“人类感”的长链，不要只优化 final score，还要加 trajectory prior

如果你只追最终误差最小，系统可能会：

* 一步跳很大
* 用不自然的工具顺序
* 或者很短链

而你要的是可训练的“摄影师式”长链，所以目标函数里要显式加入轨迹先验：

* 优先级违例惩罚（global before local）
* 单步改动过大惩罚
* 参数符号翻转惩罚
* 同一工具过度重复惩罚
* 步数先验（鼓励 3~8 步）
* edit smoothness 惩罚

也就是说，你不要把“3~8 步”放到最后过滤，而应该**放进搜索目标里**。

---

## 七、我会怎么改你的代码结构

最小但有效的改法是新增 4 个模块。

### 1. `candidate_generator.py`

职责：

* 基于 delta stat + 历史，输出 tool shortlist
* 调 MLLM 生成 top-K `{tool, direction, magnitude_bucket}`

### 2. `probe_engine.py`

职责：

* 对每个候选做低分辨率 probe ladder
* 计算每个 probe 的真实 effect table
* 可选估计局部 Jacobian

### 3. `local_optimizer.py`

职责：

* 从最佳 probe 出发局部优化
* `exposure/white_balance/saturation`：1D/2D line search / BO
* `tone`：trust-region coordinate search / Nelder-Mead / CMA-ES
* `hsl`：先根据颜色残差选 1~2 个 band，再优化

### 4. `state_manager.py`

职责：

* 维护 `accepted_state`, `rejected_candidates`, `beam`
* 执行 monotonic accept/reject
* 管理 cooldown / lock / rollback

---

## 八、生成流程我建议改成这样

```python
def generate_trajectory(src, tgt, max_steps=8, beam_size=3):
    root = State.from_source(src)
    beam = [root]

    for step in range(max_steps):
        expanded = []

        for node in beam:
            open_tools = shortlist_tools(node, tgt)

            for tool in open_tools[:2]:
                proposals = mllm_propose_ranges(node, tgt, tool, k=3)

                for prop in proposals:
                    probes = render_probe_ladder(node, tool, prop)
                    best_probe = select_best_probe(probes, tgt)

                    cand = local_optimize(node, tool, best_probe, tgt)

                    if accept(cand, node):
                        expanded.append(cand)
                    else:
                        log_reject(node, cand)

        if not expanded:
            break

        beam = topk_diverse(expanded, k=beam_size)

    best_leaf = select_best_leaf(beam)
    return export_accepted_path(best_leaf)
```

注意最后一句：
**`export_accepted_path`**
也就是说，训练数据里只保留被接受的 steps。

---

## 九、如果你要“效果最强版”，我甚至建议再进一步

最强版不是“让 MLLM 一步步找最终参数”，而是：

### 先求最终参数，再合成轨迹

1. 用黑盒优化器先找一个高质量最终参数向量 `p*`
2. 再按摄影师顺序把 `p*` 分解成 3~8 个单工具子步
3. 每个子步都必须单调改善
4. 最后让 MLLM根据“已验证的真实步骤”来生成 CoT

这样做的好处是：

* 最终图质量显著更稳
* 轨迹不会循环
* CoT 和参数是一致的
* MLLM 不再承担连续控制的数值负担

如果你把“效果优先”放到最高，这个版本其实是我最推荐的。

---

## 十、最后给你一个非常明确的结论

我会把你当前方案从：

**“MLLM 单链贪心地直接输出精确参数”**

改成：

**“MLLM 提多候选工具/区间 → probe 引擎估计真实响应 → 局部优化求精确值 → 只提交更好的状态 → beam search 保留多条潜力链 → 最终只导出被接受的单链轨迹”**

这样对应你的三个问题，分别会变成：

1. **不会再在 exposure 上无效来回**
   因为有多候选、probe、局部搜索、cooldown、反向震荡约束

2. **更差结果不会污染后续状态**
   因为只有 accepted state 才能推进轨迹

3. **参数值和真实图像效果能对齐**
   因为数值来自当前状态下的真实 probe / 局部优化，而不是 MLLM 硬猜

如果你愿意，我下一步可以直接把这个方案落成一版新的 `trajectory_generator.py` / `state_manager.py` / `probe_engine.py` 伪代码接口设计。

[1]: https://arxiv.org/html/2602.22809 "https://arxiv.org/html/2602.22809"
[2]: https://arxiv.org/html/2510.08054v2 "https://arxiv.org/html/2510.08054v2"

```
