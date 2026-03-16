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

## 工具设计

共 5 个工具，按摄影师调色优先级排序：

| 优先级 | 工具名 | 参数 | 作用 |
|--------|--------|------|------|
| 1 | `exposure_tool` | exposure [-100,100], brightness [-100,100] | 全局曝光亮度 |
| 2 | `tone_tool` | contrast [-100,100], highlights [-120,120], shadows [-120,120], whites [-30,30], blacks [-70,70] | 色调曲线塑形 |
| 3 | `white_balance_tool` | temperature [-500,500], tint [-100,100] | 色温/色调校正 |
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

编辑 `config/pipeline.yaml`，主要配置项：

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

准备 paired 数据的 JSON 文件（参考 `data/mit5k_pairs.json`）：
```json
[
  {"id": "a0001", "source": "path/to/src/a0001.tif", "target": "path/to/gt/a0001.tif"},
  ...
]
```

### 生成轨迹

```bash
cd /path/to/retouch

# 设置 API Key
export OPENAI_API_KEY="sk-..."

# 运行生成（--max-samples 用于快速测试）
python trajectory_forge/run_generate.py \
    --config trajectory_forge/config/pipeline.yaml \
    --pairs trajectory_forge/data/mit5k_pairs.json \
    --output trajectories/ \
    --max-samples 5

# Dry run（不调用 API，仅验证配置）
python trajectory_forge/run_generate.py --dry-run --max-samples 3
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
      "params_accumulated": {"exposure": 35, "brightness": 10, "contrast": 0, ...},
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
