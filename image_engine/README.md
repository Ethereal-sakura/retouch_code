# RapidRAW Basic + Color 抽取版（非 RAW）

这是从当前 Python 重构版里**单独抽出来**的一个精简版本，只保留 RapidRAW 里你现在需要的两块：

- **Basic 面板**
- **Color 面板**

并且明确假设：

- **输入不是 RAW**
- 输入格式一般是 **TIF / PNG / JPG**
- 不做 RAW 解码，不依赖 `rawpy`

---

## 保留的功能

### Basic
- Exposure
- Brightness
- Contrast
- Highlights
- Shadows
- Whites
- Blacks
- Tone Mapper: `basic` / `agx`

### Color
- Temperature
- Tint
- Saturation
- Vibrance
- HSL Mixer（8 色带）
- Color Grading（Shadows / Midtones / Highlights）

---

## 不包含的功能

以下内容在这个抽取版里**全部去掉**：

- RAW 解码
- Crop / Rotation / Geometry
- Curves
- LUT
- Sharpen / Clarity / Structure
- Grain / Vignette
- Mask / AI Patch
- Lens correction
- Glow / Halation / Flare

---

## 目录

```text
main_basic_color.py
rapidraw_basic_color/
  __init__.py
  params.py
  io.py
  colors.py
  basic.py
  engine.py
  cli.py
examples/
  demo_input.png
  params.basic_color.json
tests/
  test_basic_color_only.py
```

---

## 安装依赖

这个子集只需要很少的依赖：

```bash
python3 -m pip install -r requirements_basic_color.txt
```

---

## 使用方法

```bash
python3 main_basic_color.py \
  --input examples/demo_input.png \
  --params examples/params.basic_color.json \
  --output examples/demo_basic_color.jpg
```

如果你想保存中间结果：

```bash
python3 main_basic_color.py \
  --input examples/demo_input.png \
  --params examples/params.basic_color.json \
  --output examples/demo_basic_color.jpg \
  --save-intermediates-dir examples/basic_color_intermediates
```

会输出：

- `01_input.png`
- `02_after_basic.png`
- `03_after_color.png`
- `04_output.png`

---

## 参数文件格式

示例：`examples/params.basic_color.json`

支持字段：

```json
{
  "inputColorSpace": "srgb",
  "toneMapper": "agx",
  "exposure": 0.35,
  "brightness": 0.15,
  "contrast": 18,
  "highlights": -32,
  "shadows": 22,
  "whites": 8,
  "blacks": -6,
  "temperature": 12,
  "tint": 4,
  "saturation": 8,
  "vibrance": 16,
  "hsl": { "...": "..." },
  "colorGrading": { "...": "..." },
}
```

---

## 说明

这个版本依然尽量沿用 RapidRAW 的 Basic / Color 数学思路：

- Basic 的 tone / highlight / shadow 逻辑来自 RapidRAW 的 `shader.wgsl`
- Color 的 white balance / vibrance / HSL / color grading / color calibration 也尽量按原逻辑实现

但是它是一个 **CPU 离线版**，不是 GPU shader 逐像素 1:1 复刻。

因此它适合：

- 服务端离线处理
- 批量套 Basic + Color 调色参数
- 做后端 API
- 做最小渲染内核

不适合：

- 追求与 RapidRAW 桌面版完全逐像素一致
- 复刻 RapidRAW 全功能编辑器


---

## License / Attribution

This repository contains a simplified non-RAW Basic + Color renderer extracted from a RapidRAW-inspired Python refactor.
Because the implementation logic was developed by studying the AGPL-licensed RapidRAW project, keep the upstream `LICENSE` file together with this extracted version when redistributing it.
