# trajectory_forge

用于从 paired retouch 数据中自动构造多步修图轨迹，供后续 SFT / agentic MLLM 训练使用。

当前分支是 **MCTS-only** 版本：

- 不再保留旧的 beam / probe ladder / shortlist 兼容路径
- 每一层都允许模型从全工具集合中自由提出多个候选动作
- 搜索器通过 MCTS 在树上选择长链
- 导出的轨迹只保留最终 best path 上被接受的步骤

## 5 分钟认识项目

先记住这 5 句话：

1. 这是一个“模型提候选，程序做树搜索和客观评估”的轨迹构造器，不是一步一步直接执行模型输出的脚本。
2. 模型每次看到的是 `CURRENT image` 和 `TARGET image`，输出的是相对当前图的单步参数增量。
3. 渲染器每次都从 `src` 重新回放累计参数，不在中间 JPG 上继续叠加误差。
4. `parameters` 保留模型原始提案，`delta_parameters` 才是经过规范化后真正执行的参数。
5. 最终训练链只导出被接受的单链，不导出被剪枝或被拒绝的分支。

建议按这个顺序读代码：

1. `config/pipeline.yaml`
2. `agents/prompts.py`
3. `pipeline/mcts_search.py`
4. `pipeline/trajectory_generator.py`
5. `pipeline/candidate_generator.py`
6. `pipeline/scoring.py`
7. `tools/image_engine_adapter.py`

## 核心设计

### 1. 模型负责什么

模型负责为当前节点提出多个候选动作：

- 选哪个工具
- 这一步的参数是多少
- 为什么这一步值得尝试

当前 planner 输出固定为英文 JSON：

```json
{
  "should_stop": false,
  "main_issue": "exposure",
  "candidates": [
    {
      "tool": "exposure_tool",
      "parameters": {
        "exposure": -6,
        "brightness": -2
      },
      "reason": "The image is still brighter than the target."
    }
  ]
}
```

约束：

- 每个 candidate 只调用一个工具
- `hsl_tool` 一步只允许一个 band
- prompt 会鼓励模型输出整数风格参数，但不会强制导出时改写原始值

### 2. 程序负责什么

程序负责：

- 对模型候选做参数规范化
- 渲染候选结果
- 计算 DeltaE / SSIM / LPIPS / residual / edit cost
- 在树上执行 selection / expansion / backpropagation
- 剪掉明显恶化、重复或无效动作
- 最终选择 best leaf 并导出路径

### 3. 参数语义

当前代码里 4 个字段的语义必须分清：

- `parameters`
  模型原始输出，原样保留
- `delta_parameters`
  实际执行的参数，会经过 numeric 解析、clamp、round、去零值
- `params_accumulated`
  当前累计总参数
- `params_accumulated_tool`
  当前工具维度上的累计总参数

这意味着：

- 训练时可以保留模型原始表述
- 回放时永远以 `delta_parameters` 和累计状态为准

### 4. 搜索语义

现在导出的训练轨迹仍然是一条单链，但内部生成过程是树搜索：

```text
source / target
    ↓
root accepted state
    ↓
planner proposes multiple tool+parameter actions
    ↓
normalize + render + score
    ↓
MCTS select / expand / backpropagate
    ↓
best leaf
    ↓
export root-to-leaf accepted path
```

注意两点：

- `CURRENT image` 永远是当前树节点对应的 accepted state
- 模型不再被程序硬性限制在 shortlist 工具集合里

## 项目结构

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
│   ├── mcts_search.py
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

## 主要模块职责

### `agents/prompts.py`

- 定义英文 planner prompt 和 explainer prompt
- planner 现在只支持 `candidates` 结构

### `agents/mllm_agent.py`

- 调 OpenAI-compatible vision API
- 解析 planner JSON
- 不再兼容旧的 `proposals / direction / magnitude_bucket` 结构

### `pipeline/candidate_generator.py`

- 负责调用 planner
- 合并多次采样得到的候选
- 做去重和 soft prior 赋值
- 可选启发式 fallback

### `pipeline/mcts_search.py`

- 当前搜索主线
- 定义 MCTS 节点
- 做 PUCT 选择、扩展、回传
- 做动作规范化、重复检测和回归剪枝

### `pipeline/scoring.py`

- 统一 objective score
- tool residual
- edit cost
- repeated tool / sign flip penalty

### `pipeline/state_manager.py`

- SearchState 数据结构
- 工具状态记录
- accepted 后的工具状态更新

### `pipeline/trajectory_generator.py`

- 单张图对的主入口
- 初始化 root state
- 调 MCTS 搜索
- 补全 accepted-step explanation
- 导出训练轨迹

### `tools/image_engine_adapter.py`

- 负责把单步 delta 合并进累计参数
- 负责从 `src` 回放渲染
- 是整个系统里参数语义最敏感的地方

## 单张样本的生成流程

1. 读取 `src / tgt`
2. 计算初始 metrics 和 residual
3. 构造 root accepted state
4. 对当前节点调用 planner，拿到多组 `tool + parameters`
5. 规范化动作并渲染候选结果
6. 用统一 objective score 做评估
7. 用 MCTS 选择下一个要扩展的节点
8. 达到停止条件后，选出 best leaf
9. 只导出 best leaf 对应的 accepted path

## 配置说明

重点看这几组配置：

### `planner`

- `candidates_per_call`
- `diversity_calls`
- `temperatures`
- `allow_heuristic_fallback`

### `search`

- `lock_threshold`
- `unlock_threshold`
- `stop_residual_threshold`

### `mcts`

- `num_simulations`
- `c_puct`
- `max_actions_per_node`
- `rollout_horizon`
- `min_step_gain`
- `target_chain_len`
- `length_bonus`
- `regression_tolerance`

### `scoring.weights`

- `delta_e`
- `lpips`
- `ssim_error`
- `stat_residual`
- `edit_cost`
- `repeat_penalty`
- `sign_flip_penalty`

## 运行命令

### 单样本 smoke test

```bash
export OPENAI_API_KEY=你的key

python trajectory_forge/run_generate.py \
  --config trajectory_forge/config/pipeline.yaml \
  --pairs trajectory_forge/data/ppr10k_pairs.json \
  --output trajectories_smoke \
  --max-samples 1
```

### 批量生成

```bash
export OPENAI_API_KEY=你的key

python trajectory_forge/run_generate.py \
  --config trajectory_forge/config/pipeline.yaml \
  --pairs trajectory_forge/data/ppr10k_pairs.json \
  --output trajectories_ppr10k_run
```

### 过滤导出

```bash
python trajectory_forge/run_filter.py \
  --input trajectories_ppr10k_run/trajectories_raw.json \
  --output trajectories_ppr10k_run/training_data.json \
  --config trajectory_forge/config/pipeline.yaml \
  --stats
```

## 输出文件

生成阶段输出：

- `trajectories_raw.json`
- `trajectories_brief.json`
- 每条轨迹的中间图目录

其中：

- `trajectories_raw.json`
  完整调试字段，保留搜索元信息
- `trajectories_brief.json`
  精简训练视图，保留 `input_image / output_image / tool / parameters / delta_parameters / cot / step_quality`

每个 step 现在会包含：

- `parameters`
- `delta_parameters`
- `params_accumulated`
- `params_accumulated_tool`
- `proposal`
- `probe_summary`
- `score_before`
- `score_after`
- `action_signature`
- `planner_call_id`
- `planner_temperature`
- `mcts_summary`
- `accepted`

说明：

- `probe_summary` 为兼容下游结构保留，MCTS 模式下通常为空
- `action_signature` 是规范化动作的唯一标识
- `mcts_summary` 记录该导出节点的访问次数、价值和先验

## 本地验证

当前代码最基本的本地验证命令：

```bash
python -m py_compile $(find trajectory_forge -name '*.py')
python -m unittest trajectory_forge.tests.test_search_pipeline
```

覆盖内容：

- `merge_tool_call()` 的累计语义
- 模型原始 `parameters` 保留
- 执行参数 `delta_parameters` 的规范化
- MCTS 会剪掉明显恶化的候选

## 当前限制

- 搜索仍然是 target-aware 的，适合数据构造，不适合直接部署
- `hsl_tool` 每步只支持一个 band
- 当前默认鼓励整数风格动作，但真实导出的 `parameters` 不强制为整数
- README 只描述当前 MCTS-only 分支，不再覆盖旧版本历史
