# trajectory_forge

用于从 paired retouch 数据中自动构造多步修图轨迹，供后续 SFT / agentic 训练使用。

当前版本已经从“单链贪心、模型直接猜精确参数”的方案，重构为“模型做离散规划，程序做试探搜索和接受门”的方案。

## 给新接手同学的 5 分钟说明

如果你第一次接手这个项目，先记住这 4 句话：

1. `trajectory_forge` 不是一个“让模型直接输出最终参数”的项目，而是一个“模型给方向，程序验证并决定是否接受”的轨迹构造器。
2. 模型输出的是**相对当前图的参数增量**，不是绝对最终值。
3. 渲染器每次都从 `src` 重新回放累计参数，所以不会在 JPEG 中间图上继续叠加误差。
4. 正式轨迹里只保留**被接受的步骤**，被拒绝的候选只存在于内部搜索过程，不会写入训练链。

如果你只想快速建立全局认识，建议按这个顺序阅读代码：

1. `config/pipeline.yaml`
2. `agents/prompts.py`
3. `pipeline/trajectory_generator.py`
4. `pipeline/candidate_generator.py`
5. `pipeline/probe_engine.py`
6. `pipeline/scoring.py`
7. `tools/image_engine_adapter.py`

## 为什么这次要重构

旧方案的问题不是“prompt 再润色一下”就能解决，而是控制闭环本身不稳定。之前的单链贪心方案有几个结构性问题：

- 模型一次只给一个动作，程序直接执行，没有内部比较多个候选。
- 只要模型给了参数，状态就会推进，即使客观指标已经变差。
- 模型既要决定工具，又要猜精确连续参数，这部分负担太重。
- 过滤器只在最后做体检，无法阻止坏步骤污染后续轨迹。

这次重构的目标很明确：

- 模型负责“现在该修什么、先用哪个工具、方向和幅度大概是多少”
- 程序负责“真实渲染、试探、比较、细化、接受或拒绝”
- 正式轨迹只保留客观上确实变好的步骤

## 当前实现摘要

### 目标

- 输入：`source image`、`target image`
- 输出：一条只包含**被接受步骤**的修图轨迹
- 每一步都要求：
  - 模型先决定现在修什么、用哪个工具、方向和大致幅度
  - 程序在真实渲染器上试多个候选参数
  - 只有客观指标变好，才把这一步加入正式轨迹

### 本次代码改动

- `merge_tool_call()` 改成了**增量累加语义**
  - 模型每轮输出的是相对当前图的 `delta`
  - 程序把这个 `delta` 累加到全局参数状态
  - 渲染时始终从 `src` 重新回放累计参数
- 生成主循环从单链贪心改成了 accepted-state search
  - 当前状态只会指向“已经被接受”的图
  - 被拒绝的候选不会污染下一轮
- 新增 hidden search 组件
  - `candidate_generator.py`
  - `probe_engine.py`
  - `scoring.py`
  - `state_manager.py`
- Prompt 改成英文 JSON planner prompt
  - 模型推理时看到的 prompt 全部为英文
  - 模型输出的是 `tool / direction / magnitude_bucket / reason`
  - 不再让模型直接输出最终精确参数
- 新增 accept gate
  - 结合 `DeltaE / SSIM / LPIPS / residual stats / edit cost`
  - 只有变好才提交
- 新增工具冷却与锁定规则
  - 连续失败的工具会暂时 cooldown
  - 已基本修好的问题会被 lock，避免来回震荡
- 过滤器保留，但定位变成“最终体检”
  - 生成阶段先保证坏状态不推进
  - 过滤阶段再做质量筛查

## 核心设计

### 1. 参数语义

当前代码约定如下：

- 模型看到的是 `CURRENT image`
- 模型输出的是相对 `CURRENT image` 的参数增量
- `merge_tool_call(accumulated, tool, delta_params)` 负责把本轮增量累加到累计参数
- `render(src_img, accumulated_params)` 始终从原图重新渲染

这意味着：

- 轨迹里的 `delta_parameters` 表示本轮增量
- `params_accumulated` 表示到当前为止的总参数

### 2. hidden search

导出的训练轨迹仍然是一条单链，但生成过程内部允许：

- 多候选 proposal
- probe ladder 试探渲染
- 局部细化
- 被拒候选直接丢弃
- 多条 accepted state 并行保留

最终只导出 best path。

### 3. 模型与程序的分工

模型负责：

- 现在主要剩余问题是什么
- 现在优先用哪个工具
- 方向是 increase / decrease / mixed
- 幅度属于 small / medium / large

程序负责：

- 真实参数数值搜索
- probe 渲染
- 局部优化
- 客观打分
- accept / reject
- cooldown / lock / beam 保留

### 4. Prompt 约束

模型实际推理时的 prompt 必须全部是英文。

当前代码中：

- planner prompt 是英文
- explanation prompt 是英文
- 历史摘要、统计摘要、结构化字段名全部是英文

中文只用于：

- README
- 注释
- 本地开发说明

## 一条样本在系统里如何流转

下面这张图是当前版本最重要的心智模型：

```text
source / target
    ↓
current accepted state
    ↓
residual stats
    ↓
shortlist tools
    ↓
planner prompt (English)
    ↓
coarse proposals
    ↓
probe ladder on renderer
    ↓
local refine
    ↓
objective score
    ↓
accept / reject
    ├─ accept -> enter next accepted frontier
    └─ reject -> discard candidate
    ↓
best accepted path
    ↓
export trajectory
```

注意两点：

- `CURRENT image` 永远指向“当前最佳已接受状态”
- 下一轮不会看到被拒绝候选生成的图

## 当前项目结构

```text
trajectory_forge/
├── run_generate.py
├── run_filter.py
├── config/
│   └── pipeline.yaml
├── agents/
│   ├── mllm_agent.py
│   └── prompts.py
├── tools/
│   ├── tool_registry.py
│   └── image_engine_adapter.py
├── pipeline/
│   ├── candidate_generator.py
│   ├── probe_engine.py
│   ├── scoring.py
│   ├── state_manager.py
│   ├── trajectory_generator.py
│   └── quality_filter.py
├── utils/
│   ├── image_utils.py
│   ├── metrics.py
│   └── stat_utils.py
└── tests/
    └── test_search_pipeline.py
```

## 生成流程

单张图对的主流程现在是：

1. 读取 `src / tgt`
2. 计算当前 residual stats
3. 根据 residual、priority、cooldown、lock 选出 shortlist tools
4. 调 planner prompt，让模型输出 1 到 3 个 coarse proposals
5. 对每个 proposal 做 probe ladder
6. 从 probe 结果中选最优，再做局部细化
7. 用统一 objective score 做 accept / reject
8. 只保留 accepted children 进入下一轮
9. 每轮保留 top-K accepted states
10. 最终只导出 best accepted path

## 模块职责总表

如果你要改代码，先判断你改的是哪一层：

### `agents/`

- 负责和模型交互
- 不负责决定候选是否接受
- 这里的改动通常是：
  - prompt 结构
  - 输出 JSON 字段
  - explanation 文本风格

### `pipeline/candidate_generator.py`

- 决定本轮允许哪些工具进入 shortlist
- 决定 planner 输出解析失败后的启发式 proposal
- 这里的改动通常影响：
  - 哪些工具更容易被尝试
  - 工具优先级是否足够稳定

### `pipeline/probe_engine.py`

- 把 coarse proposal 变成真实可比较的参数候选
- 定义各个工具的 probe ladder 和局部细化策略
- 这里的改动通常影响：
  - 模型建议是否能被转成有效数值
  - 搜索是否太慢
  - 参数是否容易震荡

### `pipeline/scoring.py`

- 定义 objective score
- 定义 accept gate
- 定义 beam 排序
- 这里的改动通常影响：
  - 哪些候选会被接受
  - 轨迹是否更保守或更激进

### `pipeline/state_manager.py`

- 管理 accepted streak / reject streak / cooldown / lock
- 这里的改动通常影响：
  - 是否容易出现工具来回震荡
  - 某个工具是否被过早禁用或过早锁死

### `tools/image_engine_adapter.py`

- 定义参数语义
- 负责把 delta 合并到累计参数
- 负责从 `src` 回放渲染
- 这里是最容易把“delta”和“absolute total”搞混的地方，修改时必须非常谨慎

## 关键文件说明

### `agents/prompts.py`

- 定义 planner prompt 和 explainer prompt
- 都是英文 prompt
- planner 输出 JSON

### `agents/mllm_agent.py`

- 使用 OpenAI Python SDK 调用兼容接口
- 现在不再提供“缺少依赖时自动降级”的冗余兼容逻辑
- 运行前请确保依赖环境已经安装完整

### `tools/image_engine_adapter.py`

- `merge_tool_call()` 现在执行**累加**
- `get_tool_params()` / `diff_tool_params()` 用于记录累计参数和实际生效增量
- 渲染仍然从 `src` 回放总参数

### `pipeline/candidate_generator.py`

- 根据 residual 和工具状态选 shortlist
- 调模型生成 coarse proposals
- planner 失败时会退回到**启发式 proposal**，但这不是依赖缺失兜底，只是解析失败时的控制兜底

### `pipeline/probe_engine.py`

- 对每个 coarse proposal 生成 probe ladder
- 在低分辨率图上做试探
- 再做局部细化
- 输出候选的 metrics / score / probe_summary

### `pipeline/scoring.py`

- 统一 objective score
- 接受门
- tool residual
- beam ranking

### `pipeline/state_manager.py`

- accepted state 数据结构
- tool cooldown / lock / accepted streak / reject streak

### `pipeline/trajectory_generator.py`

- 主搜索循环
- accepted-only 状态推进
- 最终导出训练轨迹

## 新人最容易踩坑的地方

### 1. 把 `delta_parameters` 当成绝对值

不是。

当前实现中：

- `delta_parameters` 是本轮增量
- `params_accumulated` 才是累计总参数

### 2. 以为 `current_img` 是上一张导出的中间图

不完全是。

语义上它是“当前已接受状态对应的图”。虽然这张图也可能会被导出成中间图，但内部逻辑看的是 accepted state，不是“上一轮模型刚产生的任意图”。

### 3. 以为 planner 说了算

不是。

planner 只决定 coarse direction，真正是否提交由程序根据真实渲染结果决定。

### 4. 以为过滤器会修正生成错误

不会。

过滤器现在只是最终体检。真正防止坏步骤污染轨迹的是生成阶段的 accept gate。

## 如果你要调效果，优先改哪里

### 场景 1：同一个工具反复震荡

优先检查：

- `state_manager.py` 里的 cooldown / lock / accepted streak
- `scoring.py` 里的 sign flip penalty
- `probe_engine.py` 里的 refine step 是否太大

### 场景 2：模型总选错工具

优先检查：

- `candidate_generator.py` 的 shortlist 逻辑
- `prompts.py` 的 planner prompt
- `stat_utils.py` 的 residual 定义是否和工具语义一致

### 场景 3：模型方向对，但数值总是不准

优先检查：

- `probe_engine.py` 的 ladder 设计
- `probe_engine.py` 的 local refine
- `image_engine_adapter.py` 的参数合并语义

### 场景 4：轨迹太短或太长

优先检查：

- `search.stop_residual_threshold`
- `generation.max_turns`
- `search.accept_margin`
- `filter.min_steps`

## 配置文件

`config/pipeline.yaml` 现在除了原有配置，还增加了几组新配置：

- `planner`
- `search`
- `probe`
- `scoring`
- `debug`

其中重点参数如下：

- `search.beam_size`
- `search.shortlist_tools`
- `search.max_proposals`
- `search.accept_margin`
- `search.hard_delta_e_tolerance`
- `probe.render_size`
- `probe.refine_steps`
- `scoring.weights.*`

推荐理解方式：

- `generation.*` 控整体生成边界
- `search.*` 控搜索深度和接受门
- `probe.*` 控数值试探方式
- `scoring.*` 控候选排序标准
- `filter.*` 控最终入库门槛

## 依赖要求

请在完整虚拟环境中运行，至少保证以下依赖已经安装：

- `openai`
- `lpips`
- `torch`
- `opencv-python`
- `scikit-image`
- `Pillow`
- `PyYAML`

当前代码不再对这些缺失依赖做自动降级。

## 运行命令

### 1. 单样本 smoke test

```bash
export OPENAI_API_KEY=你的key

python trajectory_forge/run_generate.py \
  --config trajectory_forge/config/pipeline.yaml \
  --pairs trajectory_forge/data/ppr10k_pairs.json \
  --output trajectories_smoke \
  --max-samples 1
```

### 2. 真实数据生成

下面这条就是当前 README 推荐的真实数据运行命令：

```bash
export OPENAI_API_KEY=你的key

python trajectory_forge/run_generate.py \
  --config trajectory_forge/config/pipeline.yaml \
  --pairs trajectory_forge/data/ppr10k_pairs.json \
  --output trajectories_ppr10k_run
```

如果你只想先跑一部分数据：

```bash
export OPENAI_API_KEY=你的key

python trajectory_forge/run_generate.py \
  --config trajectory_forge/config/pipeline.yaml \
  --pairs trajectory_forge/data/ppr10k_pairs.json \
  --output trajectories_ppr10k_run \
  --max-samples 100 \
  --start-idx 0
```

### 3. 过滤导出

```bash
python trajectory_forge/run_filter.py \
  --input trajectories_ppr10k_run/trajectories_raw.json \
  --output trajectories_ppr10k_run/training_data.json \
  --config trajectory_forge/config/pipeline.yaml \
  --stats
```

## 运行产物

生成阶段输出：

- `trajectories_raw.json`
- `trajectories_brief.json`
- 每条轨迹的中间图目录

其中：

- `trajectories_raw.json` 是完整版本，保留全部搜索结果导出字段
- `trajectories_brief.json` 是精简版本，便于快速收集训练数据
  - 每条轨迹只保留 `initial_quality`
  - 每个 step 只保留 `output_image / tool / parameters / cot / step_quality`

每个 step 现在会包含：

- `parameters`
- `delta_parameters`
- `params_accumulated`
- `params_accumulated_tool`
- `proposal`
- `probe_summary`
- `score_before`
- `score_after`
- `accepted`

其中：

- `parameters` 与 `delta_parameters` 当前等价，都是本轮增量
- `params_accumulated` 是当前总参数状态

如果你想快速人工看一条轨迹，最有用的字段通常是：

- `tool`
- `delta_parameters`
- `step_quality`
- `score_before`
- `score_after`
- `proposal`

如果你想快速整理训练样本，优先看：

- `trajectories_brief.json`

## 已完成的本地验证

- `python -m py_compile $(find trajectory_forge -name '*.py')`
- `python -m unittest trajectory_forge.tests.test_search_pipeline`

当前离线测试覆盖了：

- `merge_tool_call()` 的增量累加语义
- 错误候选不会污染 accepted state
- 正确候选可以被接受并写入正式轨迹

## 下一步

你在完整依赖环境里运行真实数据后，把这些内容发回来：

- `trajectory_forge.log`
- `trajectories_raw.json` 中 1 到 3 条代表性轨迹
- `run_filter.py --stats` 的输出

我会继续基于真实运行结果做第二轮收敛，包括：

- score 权重调参
- probe ladder 调整
- 工具锁定/冷却阈值调整
- planner prompt 的进一步收紧
