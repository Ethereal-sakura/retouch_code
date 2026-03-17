# trajectory_forge

**修图长链轨迹构造框架** — 用于生成训练 Agentic MLLM 修图模型的多轮轨迹数据集。

---

## 概述

`trajectory_forge` 通过调用前沿 MLLM（如 GPT-4o）驱动图像处理引擎，自动为 paired 修图数据集（MIT-FiveK / PPR10K）生成**多轮修图轨迹**，每条轨迹形如：

```
原图 → [CoT + 工具1 + 中间图1] → [CoT + 工具2 + 中间图2] → ... → 最终图
```

生成的轨迹经质量过滤后，可作为训练数据用于：
- 监督微调（SFT）Agentic MLLM 修图模型
- 构建多轮工具调用的推理链数据集

---

## 特性

- **Training-free**：无需任何标注，由前沿 MLLM 自动生成轨迹
- **5 个语义工具**：对应摄影师调色认知顺序（曝光 → 色调 → 色温 → 饱和度 → HSL）
- **定量统计锚点**：每轮 prompt 包含 delta 统计数值，引导模型做出有依据的决策
- **从原图累积渲染**：每步从原图重新渲染，避免误差累积，参数耦合关系正确
- **多候选 + VLM Judge**：每轮生成 N 个多样化候选方案，由 VLM 视觉比较选出最佳
- **质量门控回退**：VLM Judge 可选择保留当前状态（不采纳任何候选），防止质量倒退
- **振荡检测与工具锁定**：自动检测参数正负交替振荡，锁定振荡工具强制模型探索其他维度
- **完整质量过滤**：DeltaE、PSNR、单调性、工具多样性等多维度过滤

---

## 项目结构

```
trajectory_forge/
├── run_generate.py              # 入口：批量生成轨迹
├── run_filter.py                # 入口：质量过滤 + 导出训练数据
│
├── tools/
│   ├── tool_registry.py         # 工具 schema 定义 + 参数校验
│   └── image_engine_adapter.py  # BasicColorParams 构造 + 渲染封装
│
├── agents/
│   ├── mllm_agent.py            # MLLM API 调用 + 多候选/Judge 响应解析
│   └── prompts.py               # 多候选/Judge/单候选 prompt 模板
│
├── pipeline/
│   ├── trajectory_generator.py  # 单图对的轨迹生成主循环
│   └── quality_filter.py        # 轨迹质量过滤
│
├── utils/
│   ├── stat_utils.py            # get_stat + get_delta_stat
│   ├── metrics.py               # PSNR / SSIM / LPIPS / DeltaE
│   └── image_utils.py           # 图像 I/O + base64 编码
│
├── config/
│   ├── tools.json               # 工具 schema（参数名、范围、描述）
│   └── pipeline.yaml            # 超参配置
│
└── data/
    ├── mit5k_pairs.json         # MIT-FiveK paired 列表（示例）
    └── ppr10k_pairs.json        # PPR10K paired 列表（示例）
```

---

## 完整工作流程

### 总览

```
数据集 (pairs.json)
    ↓
run_generate.py          ← 批量入口
    ↓
trajectory_generator.py  ← 单图对主循环
    ├── stat_utils.py        计算 delta 统计
    ├── prompts.py           构造 MLLM 输入
    ├── mllm_agent.py        调用 GPT-4o
    ├── tool_registry.py     校验工具调用
    └── image_engine_adapter.py  渲染图像
    ↓
trajectories_raw.json
    ↓
run_filter.py            ← 过滤入口
    ↓
quality_filter.py        ← 多维度过滤
    ↓
training_data.json       ← 最终训练数据
```

---

### 第一阶段：生成轨迹（`run_generate.py`）

批量入口脚本，负责：

1. 读取 `config/pipeline.yaml` → 所有超参数
2. 读取 `data/mit5k_pairs.json` → 图像路径列表
3. 初始化 `MLLMAgent` → 连接 GPT-4o API
4. 逐个图像对调用 `generate_trajectory()`
5. 每完成一条轨迹立即追加写入 `trajectories_raw.json`（crash-safe，不会因中途失败丢失已生成数据）

---

### 第二阶段：单图对轨迹生成主循环（`trajectory_generator.py`）

对每一对 `(src, tgt)` 图像运行最多 `max_turns=8` 轮对话，采用 **多候选生成 + VLM Judge 选择** 机制：

```
初始化
  src_img = load_image(source_path)        # float32 [0,1]
  tgt_img = load_image(target_path)
  accumulated_params = BasicColorParams()  # 全零，即 identity
  initial_quality = compute_metrics(src, tgt)
  locked_tools = {}                        # 因振荡被锁定的工具集合
  rollback_count = 0                       # 连续回退计数

┌─────────────────────── 主循环（每轮）────────────────────────────┐
│                                                                    │
│  [1] 计算 delta 统计 + 收敛检查                                   │
│      get_delta_stat(current_img, tgt_img)                         │
│      if DeltaE < 4.0: break                                      │
│                                                                    │
│  [2] 构造多候选 Prompt (prompts.py)                               │
│      • 当前图 + 目标图缩略图                                      │
│      • 定量 delta 统计 + 质量变化趋势                             │
│      • 锁定工具警告 + 回退警告                                    │
│      • 要求输出 N=3 个多样化候选方案                              │
│                                                                    │
│  [3] 调用 VLM 生成候选 (mllm_agent.py)                           │
│      → 返回 <candidate_1>...<candidate_3> 各含                    │
│        <thinking> + <tool_call>                                   │
│                                                                    │
│  [4] 解析 + 校验 + 渲染各候选                                     │
│      parse_multi_tool_calls() → N 个 (tool, params)              │
│      跳过使用锁定工具 / 校验失败的候选                            │
│      每个有效候选: merge → render → compute_metrics               │
│                                                                    │
│  [5] VLM Judge 选择最佳候选                                       │
│      将 [当前图, 候选1, 候选2, ..., 目标图] 发送给 VLM           │
│      VLM 输出 <choice>X</choice>                                  │
│      X=0 表示保持当前状态（回退）                                  │
│      X>0 表示采纳对应候选                                          │
│                                                                    │
│  [6] 质量门控                                                      │
│      if X == 0:                                                    │
│          rollback_count++                                          │
│          if rollback_count >= 3: break (提前终止)                 │
│          continue (不更新状态)                                     │
│      else:                                                         │
│          rollback_count = 0                                        │
│          采纳候选 X (更新 accumulated_params 和 current_img)      │
│                                                                    │
│  [7] 振荡检测                                                      │
│      检查最近 window 步中同工具参数方向是否交替                    │
│      如检测到振荡 → 锁定该工具，下轮 prompt 中排除                │
│                                                                    │
│  [8] 记录步骤                                                      │
│      step = {round, action, cot, tool, parameters,                │
│              params_accumulated, step_quality, delta_stat,         │
│              judge_choice, candidates[...]}                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

最终
  final_quality = compute_metrics(current_img, tgt_img)
  return trajectory_dict
```

---

### 关键子模块详解

#### `stat_utils.py` — 定量统计

`get_delta_stat(current_img, target_img)` 对两张图分别提取像素统计、HSV 饱和度、LAB 通道均值，然后做差，映射为工具信号：

| delta 字段 | 对应工具信号 |
|-----------|------------|
| `brightness_delta` / `l_channel_delta` | → `exposure_tool` |
| `contrast_delta` / `highlight_delta` / `shadow_delta` | → `tone_tool` |
| `temperature_delta` / `tint_delta` | → `white_balance_tool` |
| `saturation_delta` | → `saturation_tool` |
| `dominant_issue` | 归一化后取最大维度，显式告知模型主要问题 |

---

#### `prompts.py` — Prompt 构造

包含三套 prompt 模板：

**1. Multi-Candidate System Prompt**（主要使用）
- 要求模型每轮输出 N 个多样化候选，格式为 `<candidate_1>...<candidate_N>`
- 每个候选含独立的 `<thinking>` + `<tool_call>`
- 明确鼓励候选间的多样性：不同工具、不同方向、不同力度
- 优先级从严格规则降级为引导建议，允许候选探索不同优先级

**2. VLM Judge System Prompt**
- 让 VLM 对比当前图（Image 0）+ N 个候选渲染图 + 目标图
- 从曝光、对比度、色温、饱和度、纹理 5 个维度评估
- 输出 `<choice>X</choice>`，X=0 表示保留当前状态

**3. Single-Candidate System Prompt**（保留向后兼容）
- 原始的单候选模式

**User Prompt（每轮）** 包含：
```
[当前图] [目标图]

Brightness delta: +28.3 (target is brighter)
Contrast delta: +5.1 (target has more contrast)
Temperature signal: -12.0 (target is cooler)
Saturation delta: +0.042 (target is more saturated)
Dominant issue: exposure

Quality trend: DeltaE improved from 12.3 to 9.5 (decreased by 2.8). Keep refining.
LOCKED TOOLS (do NOT use in any candidate): exposure_tool
WARNING: The previous round was ROLLED BACK...

Adjustment history:
  Round 1: exposure_tool(exposure=35.0, brightness=10.0) -> DeltaE=12.1
  Round 2: tone_tool(contrast=20.0, highlights=-30.0)    -> DeltaE=8.4

Analyze the remaining differences and propose 3 diverse candidate strategies.
```

User Prompt 新增的动态信息（仅在条件满足时出现）：
- **Quality trend**：上轮 DeltaE 的改善/恶化方向和幅度
- **LOCKED TOOLS**：因振荡被锁定的工具列表
- **WARNING**：上轮回退的警告，提示模型尝试不同方向

---

#### `image_engine_adapter.py` — 渲染核心

`merge_tool_call()` 根据工具名更新 `BasicColorParams` 中对应的字段，其余字段保持不变：

| 工具 | 更新的字段 |
|------|-----------|
| `exposure_tool` | `params.exposure`, `params.brightness` |
| `tone_tool` | `params.contrast/highlights/shadows/whites/blacks` |
| `white_balance_tool` | `params.temperature`, `params.tint` |
| `saturation_tool` | `params.saturation`, `params.vibrance` |
| `hsl_tool` | `params.hsl.{band}.hue/saturation/luminance`（只更新指定色带）|

之后调用 `render(src_img, accumulated_params)` **始终从原始源图出发**重新渲染，不在中间图上叠加。

---

#### 关于图像格式转换

磁盘上的 `.tif` 文件在发送给 MLLM 之前会经历如下转换，**不存在格式冲突**：

```
.tif 文件（磁盘）
    ↓  load_image()       用 imageio 读取，转为 float32 numpy array [0,1]
    ↓  make_thumbnail()   缩放至 512×512，仍为 numpy array
    ↓  encode_image_base64()  numpy → PIL → JPEG 字节流 → base64 字符串
    ↓  build_image_content()  拼接 data:image/jpeg;base64,...
                              media_type="image/jpeg" 与实际编码完全匹配
```

---

### 第三阶段：质量过滤（`quality_filter.py`）

对每条轨迹依次检查 5 项，全部通过才写入训练集：

```
① 最终质量门槛
   DeltaE_final ≤ 10.0  AND  PSNR_final ≥ 20.0 dB

② 改善幅度
   DeltaE_initial − DeltaE_final ≥ 2.0

③ 单调性（每步不能大幅变差）
   DeltaE_step[i] ≤ DeltaE_step[i−1] × 1.3

④ 工具多样性
   每个工具最多使用 2 次

⑤ 轨迹长度
   3 ≤ num_steps ≤ 8
```

---

### 数据流汇总

```
source.tif ──────────────────────────────────────────────────────────┐
                                                                       │
target.tif ──┐                                                         │
             │                                                         │
             ├─ get_delta_stat() ──→ 数值锚点                          │
             │                          │                              │
             │                          ↓                              │
             │                    build_user_prompt()                  │
             │                    (含锁定工具 + 质量反馈)               │
             │                          │                              │
             │                          ↓                              │
             │                    VLM API call #1                      │
             │                    (输出 N 个候选方案)                    │
             │                          │                              │
             │                    parse_multi_tool_calls()             │
             │                          │                              │
             │                    ┌─────┼──────┐                       │
             │                    ↓     ↓      ↓                       │
             │                候选1  候选2   候选3                      │
             │                    │     │      │                       │
             └──────────────→ render(src, params) ×N ──→ N 张渲染图    │
                                        │                              │
                                  VLM API call #2 (Judge)              │
                                  (当前图 + N 候选图 + 目标图)          │
                                        │                              │
                                  parse_judge_choice()                 │
                                        │                              │
                                  ┌─────┴──────┐                       │
                                  ↓            ↓                       │
                              choice=0    choice=K                     │
                              (回退)      (采纳候选K)                   │
                                  │            │                       │
                                  │     振荡检测 → 锁定工具?           │
                                  │            │                       │
                                  └─────┬──────┘                       │
                                        │                              │
                                  [循环 max_turns 次]                   │
                                        │                              │
                                  trajectory_dict ──────────────────── ┘
                                        │
                                  filter_trajectory()
                                        │
                                  training_data.json
```

---

## 工具设计

共 5 个工具，按摄影师调色优先级排序：

| 优先级 | 工具名 | 参数 | 作用 |
|--------|--------|------|------|
| 1 | `exposure_tool` | exposure [-100,100], brightness [-100,100] | 全局曝光亮度 |
| 2 | `tone_tool` | contrast [-100,100], highlights [-100,100], shadows [-100,100], whites [-30,30], blacks [-70,70] | 色调曲线塑形 |
| 3 | `white_balance_tool` | temperature [-100,100], tint [-100,100] | 色温/色调校正 |
| 4 | `saturation_tool` | saturation [-100,100], vibrance [-100,100] | 全局饱和度 |
| 5 | `hsl_tool` | adjustments: [{color, hue, saturation, luminance}] | 选择性色相调整 |

`hsl_tool` 示例：
```json
{
  "tool": "hsl_tool",
  "adjustments": [
    {"color": "reds",    "hue": 5,  "saturation": 10, "luminance": 0},
    {"color": "oranges", "hue": -3, "saturation": 8,  "luminance": 4}
  ]
}
```

---

## 快速开始

### 安装依赖

```bash
pip install openai pyyaml numpy pillow opencv-python scikit-image torch lpips imageio
```

确保 `image_engine` 可被导入（已包含在本仓库 `image_engine/` 目录中）。

### 配置

编辑 `trajectory_forge/config/pipeline.yaml`，主要配置项：

```yaml
api:
  model: "gpt-4o"          # 使用的模型
  api_key_env: "OPENAI_API_KEY"

generation:
  max_turns: 8             # 每条轨迹最多修图轮数
  output_dir: "trajectories"
  num_candidates: 3        # 每轮生成的候选方案数
  max_rollbacks: 3         # 连续回退次数上限（VLM Judge 选 0 的次数）
  oscillation_window: 3    # 振荡检测滑动窗口大小

dataset:
  pairs_file: "data/mit5k_pairs.json"
  max_samples: 10          # 调试时先跑少量样本
```

准备 paired 数据的 JSON 文件（参考 `trajectory_forge/data/mit5k_pairs.json`）：
```json
[
  {"id": "a0001", "source": "path/to/src/a0001.tif", "target": "path/to/gt/a0001.tif"},
  ...
]
```

### 生成轨迹

所有路径均相对于**执行命令的当前目录**：

```bash
cd /path/to/retouch_code

# 设置 API Key
export OPENAI_API_KEY="sk-..."

# 运行生成（--max-samples 用于快速测试）
python trajectory_forge/run_generate.py \
    --config trajectory_forge/config/pipeline.yaml \
    --pairs trajectory_forge/data/mit5k_pairs.json \
    --output trajectories/ \
    --max-samples 5

# Dry run（不调用 API，仅验证配置和数据路径）
python trajectory_forge/run_generate.py \
    --config trajectory_forge/config/pipeline.yaml \
    --dry-run --max-samples 3
```

### 质量过滤

```bash
python trajectory_forge/run_filter.py \
    --input trajectories/trajectories_raw.json \
    --output trajectories/training_data.json \
    --stats
```

输出示例：
```
=== Filter Summary ===
Total:     100
Passed:    62  (62.0%)
Failed:    38
Fail reasons:
  final DeltaE: 15
  improvement: 12
  regression: 8
  too few steps: 3
```

---

## 轨迹数据格式

生成的训练数据为 JSON 数组，每条轨迹结构：

```json
{
  "id": "a0001_0000",
  "source": "path/to/src/a0001.tif",
  "target": "path/to/gt/a0001.tif",
  "initial_quality": {"psnr": 22.3, "ssim": 0.84, "delta_e": 16.2},
  "final_quality":   {"psnr": 27.8, "ssim": 0.91, "delta_e": 6.5},
  "num_steps": 4,
  "steps": [
    {
      "round": 0,
      "action": "adopt",
      "input_image": "trajectories/a0001_0000/step_0_input.jpg",
      "cot": "The image is underexposed. L-channel delta is +28...",
      "tool": "exposure_tool",
      "parameters": {"exposure": 35, "brightness": 10},
      "params_accumulated": {"exposure": 35, "brightness": 10, "contrast": 0, "...": "..."},
      "output_image": "trajectories/a0001_0000/step_0_output.jpg",
      "step_quality": {"psnr": 24.8, "delta_e": 12.1},
      "delta_stat": {"brightness_delta": 28.3, "dominant_issue": "exposure"},
      "judge_choice": 2,
      "candidates": [
        {"tool": "exposure_tool", "parameters": {"exposure": 20}, "quality": {"delta_e": 13.5}, "cot": "..."},
        {"tool": "exposure_tool", "parameters": {"exposure": 35, "brightness": 10}, "quality": {"delta_e": 12.1}, "cot": "..."},
        {"tool": "tone_tool", "parameters": {"contrast": 15}, "quality": {"delta_e": 14.2}, "cot": "..."}
      ]
    },
    {
      "round": 1,
      "action": "rollback",
      "candidates": ["..."],
      "judge_choice": 0,
      "delta_stat": {"...": "..."},
      "current_quality": {"psnr": 24.8, "delta_e": 12.1}
    }
  ]
}
```

每个 step 的 `action` 字段区分两种情况：
- `"adopt"`：VLM Judge 选中了某个候选，`judge_choice` 为 1-based 索引
- `"rollback"`：VLM Judge 认为所有候选都不如当前状态，`judge_choice` 为 0

`candidates` 数组记录了该轮所有有效候选的工具、参数、质量指标和推理过程。
```

---

## 质量过滤标准

| 标准 | 默认阈值 | 说明 |
|------|----------|------|
| 最终 DeltaE | ≤ 10.0 | 颜色差距不能太大 |
| 最终 PSNR | ≥ 20.0 dB | 结构质量保证 |
| 改善幅度 | ≥ 2.0 DeltaE | 相对原图必须有显著改善 |
| 单调性 | 每步退化 ≤ 30% | 避免大幅变差的步骤 |
| 工具多样性 | 同一工具 ≤ 2 次 | 避免重复调整 |
| 轨迹长度 | 3–8 步 | 过短或过长均过滤 |

所有阈值可在 `config/pipeline.yaml` 的 `filter:` 节中调整。

---

## 核心设计：从原图累积渲染

每步不是在前一步结果上叠加，而是将参数累积后**从原图重新渲染**：

```
Step 1: params₁ = exposure(+35)          → render(src, params₁) → img₁
Step 2: params₂ = params₁ + tone(...)    → render(src, params₂) → img₂
Step 3: params₃ = params₂ + wb(...)      → render(src, params₃) → img₃
```

优点：
1. 避免 JPEG 压缩误差累积
2. 参数之间的物理耦合关系正确（由 image_engine 统一处理）
3. 可随时回退或重新渲染任意步骤

---

## 复用的已有代码

| 代码 | 来源 | 用途 |
|------|------|------|
| `get_stat()` | `iclr_retouchllm/diff_tools.py` | 图像统计基础（扩展为 `get_delta_stat()`）|
| `BasicColorRenderer` | `image_engine/rapidraw_basic_color/engine.py` | 图像渲染引擎 |
| `BasicColorParams` | `image_engine/rapidraw_basic_color/params.py` | 参数容器 |
| `load_image/save_image` | `image_engine/rapidraw_basic_color/io.py` | 图像 I/O |

---

## 更新日志

### v2.1 — 移除 VLM Judge，改为 DeltaE 最优选择（2026-03-17）

**问题背景**：v2 的 VLM Judge 在实测中存在系统性判断偏差——Judge 多次选择了 DeltaE 实际更差的候选（视觉感知与像素级指标不一致），且每轮额外增加一次 API 调用。

**解决方案**：取消 VLM Judge，改为直接用 DeltaE 指标选出最优候选；若最优候选仍不如当前状态则触发回退。

| 机制 | 说明 |
|------|------|
| DeltaE 最优选择 | 所有候选渲染后，直接选 DeltaE 最低的候选，无需额外 API 调用 |
| 质量门控回退 | 若最优候选 DeltaE ≥ 当前状态，则回退（不更新状态），连续 `max_rollbacks` 次则提前终止 |

**修改的文件**：
- `pipeline/trajectory_generator.py` — 移除 Judge API 调用，改为 `_pick_best_delta_e()` 直接选优，质量门控内置
- `config/pipeline.yaml` — 更新 `max_rollbacks` 注释语义

**API 调用开销**：每轮仅 1 次调用（生成候选），比 v2 减少 50%。

---

### v2 — 多候选 + VLM Judge 机制（2026-03-17）

**问题背景**：v1 的串行单候选模式存在三个核心缺陷：
1. **参数振荡**：模型在同一工具上反复正负交替调整（如 exposure +20 → -18 → +16），浪费大量轮次
2. **无条件前推**：即使某步导致质量大幅下降，系统仍继续使用恶化的结果
3. **参数-效果盲区**：模型无法预知某个参数值在当前状态下的实际视觉效果

**解决方案**：借鉴 `iclr_retouchllm` 的多候选 + 选择思路，结合本项目工具化参数系统特点，引入以下机制：

| 机制 | 说明 |
|------|------|
| Prompt 层多候选生成 | 每轮 VLM 一次输出 N=3 个多样化候选方案（不同工具/方向/力度），而非依赖温度采样 |
| VLM Judge 视觉选择 | N 个候选渲染后，将当前图 + N 张候选图 + 目标图发给 VLM，由其视觉对比选出最佳 |
| 质量门控回退 | VLM Judge 可选择 0（保持当前状态），连续 3 次回退则提前终止 |
| 振荡检测与工具锁定 | 自动检测同一工具参数方向交替，锁定后从 prompt 中排除该工具 |
| 增强 Prompt 反馈 | 每轮 prompt 包含质量变化趋势、锁定工具列表、回退警告等动态信息 |

**修改的文件**：
- `agents/prompts.py` — 新增 `MULTI_CANDIDATE_SYSTEM_PROMPT`、`JUDGE_SYSTEM_PROMPT`，增强 `build_user_prompt`
- `agents/mllm_agent.py` — 新增 `parse_multi_tool_calls()`、`parse_judge_choice()`
- `pipeline/trajectory_generator.py` — 主循环重构为多候选渲染 → VLM Judge → 回退/采纳 → 振荡检测
- `config/pipeline.yaml` — 新增 `num_candidates`、`max_rollbacks`、`oscillation_window`
- `run_generate.py` — 传递新配置参数

**API 调用开销**：每轮 2 次调用（1 次生成 + 1 次 Judge），预计 4-6 轮收敛，总计 8-12 次，与 v1 的 8 次相当但质量大幅提升。

---

## 许可证

本项目代码遵循 MIT License。`image_engine/` 和 `iclr_retouchllm/` 的许可证见各自目录。
