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
│   ├── mllm_agent.py            # MLLM API 调用（OpenAI 兼容格式）
│   └── prompts.py               # System / user prompt 模板
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

对每一对 `(src, tgt)` 图像运行最多 `max_turns=8` 轮对话：

```
初始化
  src_img = load_image(source_path)        # float32 [0,1]
  tgt_img = load_image(target_path)
  accumulated_params = BasicColorParams()  # 全零，即 identity
  initial_quality = compute_metrics(src, tgt)

┌─────────────────────── 主循环（每轮）────────────────────────────┐
│                                                                    │
│  [1] 计算 delta 统计 (stat_utils.py)                              │
│      get_delta_stat(current_img, tgt_img)                         │
│      → brightness_delta, contrast_delta,                          │
│        temperature_delta, saturation_delta, dominant_issue        │
│                                                                    │
│  [2] 检查是否已收敛                                                │
│      if DeltaE < 4.0: break（提前结束）                           │
│                                                                    │
│  [3] 构造 MLLM 输入 (prompts.py)                                  │
│      • 当前图缩略图（base64 JPEG）                                 │
│      • 目标图缩略图（base64 JPEG）                                 │
│      • 定量 delta 统计文字                                         │
│      • 历史操作记录（已用工具 + 每步 DeltaE）                     │
│                                                                    │
│  [4] 调用 GPT-4o (mllm_agent.py)                                  │
│      → 返回包含 <thinking> 和 <tool_call> 的文本                  │
│                                                                    │
│  [5] 解析响应 (mllm_agent.py)                                     │
│      parse_thinking() → CoT 推理文本                              │
│      parse_tool_call() → (tool_name, params)                      │
│      is_stop()         → 模型主动停止？                           │
│                                                                    │
│  [6] 校验参数 (tool_registry.py)                                   │
│      validate_tool_call() → 检查范围合法性                        │
│                                                                    │
│  [7] 合并参数 + 渲染 (image_engine_adapter.py)                    │
│      accumulated_params = merge_tool_call(params, tool, args)     │
│      new_img = render(src_img, accumulated_params)                │
│      ↑ 始终从原图渲染！                                           │
│                                                                    │
│  [8] 记录步骤                                                      │
│      step = {round, cot, tool, parameters,                        │
│              params_accumulated, step_quality, delta_stat}        │
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

**System Prompt** 告知模型：
- 每轮只能用**一个工具**
- 必须遵循优先级：`exposure → tone → white_balance → saturation → hsl`
- 输出格式严格为 `<thinking>...</thinking><tool_call>...</tool_call>` 或 `<stop>`

**User Prompt（每轮）** 包含：
```
[当前图] [目标图]

Brightness delta: +28.3 (target is brighter)
Contrast delta: +5.1 (target has more contrast)
Temperature signal: -12.0 (target is cooler)
Saturation delta: +0.042 (target is more saturated)
Dominant issue: exposure

Adjustment history:
  Round 1: exposure_tool(exposure=35.0, brightness=10.0) → DeltaE=12.1
  Round 2: tone_tool(contrast=20.0, highlights=-30.0)    → DeltaE=8.4
```

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
             │                          │                              │
             │                          ↓                              │
             │                    GPT-4o API call                      │
             │                          │                              │
             │                    parse_tool_call()                    │
             │                          │                              │
             │                    merge_tool_call()                    │
             │                    (更新 BasicColorParams)              │
             │                          │                              │
             └──────────────────→ render(src, params) ──→ new_img      │
                                        │                              │
                                  compute_metrics() ──→ step_quality   │
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
      "input_image": "trajectories/a0001_0000/step_0_input.jpg",
      "cot": "The image is underexposed. L-channel delta is +28...",
      "tool": "exposure_tool",
      "parameters": {"exposure": 35, "brightness": 10},
      "params_accumulated": {"exposure": 35, "brightness": 10, "contrast": 0, "...": "..."},
      "output_image": "trajectories/a0001_0000/step_0_output.jpg",
      "step_quality": {"psnr": 24.8, "delta_e": 12.1},
      "delta_stat": {"brightness_delta": 28.3, "dominant_issue": "exposure"}
    }
  ]
}
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

## 许可证

本项目代码遵循 MIT License。`image_engine/` 和 `iclr_retouchllm/` 的许可证见各自目录。
