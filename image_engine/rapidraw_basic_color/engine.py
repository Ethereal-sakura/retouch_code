from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .basic import (
    SCALES,
    apply_filmic_exposure,
    apply_highlights_adjustment,
    apply_linear_exposure,
    apply_tonal_adjustments,
    gaussian_blur_rgb,
)
from .colors import (
    agx_full_transform,
    apply_color_grading,
    apply_creative_color,
    apply_hsl_mixer,
    apply_white_balance,
    linear_to_srgb,
    srgb_to_linear,
)
from .io import load_image, save_image
from .params import BasicColorParams, ColorGrading, HslSettings


@dataclass
class StageRecorder:
    """中间结果记录器，用于 debug 模式下保存各阶段的图像。
    
    对应 engine.render_file 的 debug_dir 功能，输出 4 个中间 PNG：
      01_input / 02_after_basic / 03_after_color / 04_output
    
    线性图像（add_linear）在保存前先转换为 sRGB，便于直接查看。
    """
    stages: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.stages = {}

    def add_linear(self, name: str, image: np.ndarray) -> None:
        """记录线性光空间图像（转 sRGB 后存储，便于预览）。"""
        self.stages[name] = np.clip(linear_to_srgb(np.clip(image, 0.0, None)), 0.0, 1.0)

    def add_srgb(self, name: str, image: np.ndarray) -> None:
        """记录已在 sRGB 空间的图像（直接 clip 后存储）。"""
        self.stages[name] = np.clip(image.astype(np.float32), 0.0, 1.0)


@dataclass
class RenderOutput:
    """渲染结果容器。
    
    image_srgb: 最终输出图像，sRGB 空间，float32，值域 [0, 1]
    recorder: debug 模式下的中间结果记录器，非 debug 时为 None
    """
    image_srgb: np.ndarray
    recorder: Optional[StageRecorder] = None


class BasicColorRenderer:
    """RapidRAW Basic + Color 面板的非 RAW 离线渲染器。
    
    将 shader.wgsl 中 apply_all_adjustments 的 Basic + Color 部分移植为 CPU NumPy 实现。
    处理流程与 shader 中 fn apply_all_adjustments 的顺序一致：
    
      sRGB 输入
        → srgb_to_linear（色彩空间转换）
        → apply_linear_exposure（线性曝光）
        → apply_white_balance（白平衡）
        → apply_filmic_exposure（胶片式亮度）
        → apply_tonal_adjustments（白色/阴影/黑色/对比度）
        → apply_highlights_adjustment（高光）
        → apply_hsl_mixer（HSL 混合器）
        → apply_color_grading（色调分级）
        → apply_creative_color（饱和度/自然饱和度）
        → agx_full_transform 或 linear_to_srgb（色调映射）
    
    与 GPU 版本的差异：
    - CPU 版不支持 Mask、Curves、LUT、Sharpen、Clarity、Grain、Vignette 等功能
    - tonal_blur 用 scipy gaussian_filter 模拟（GPU 版用专用模糊纹理）
    - AgX 矩阵由 Python 实时计算（GPU 版由 Rust 预计算后传入 uniform）
    """

    def _normalize_basic(self, p: BasicColorParams) -> dict[str, float | str]:
        """将 Basic + Color 面板的 UI 参数除以对应 SCALES，得到 shader 内部使用的归一化值。
        
        对应 Rust 端 image_processing.rs 中 build_global_adjustments 函数的参数归一化逻辑。
        """
        return {
            "tone_mapper": p.tone_mapper,
            "exposure": p.exposure / SCALES["exposure"],
            "brightness": p.brightness / SCALES["brightness"],
            "contrast": p.contrast / SCALES["contrast"],
            "highlights": p.highlights / SCALES["highlights"],
            "shadows": p.shadows / SCALES["shadows"],
            "whites": p.whites / SCALES["whites"],
            "blacks": p.blacks / SCALES["blacks"],
            "temperature": p.temperature / SCALES["temperature"],
            "tint": p.tint / SCALES["tint"],
            "saturation": p.saturation / SCALES["saturation"],
            "vibrance": p.vibrance / SCALES["vibrance"],
        }

    def _normalize_hsl(self, hsl: HslSettings) -> HslSettings:
        """将 HSL 面板的 UI 参数归一化：
        
        - hue: 乘以 hsl_hue_multiplier(0.3)，对应 shader 中 hsl_adjustments[i].hue * 2.0 的前半部分
          （apply_hsl_mixer 中再乘 2，合计 *0.6，与 Rust 端 hue / SCALES * 2 等价）
        - saturation: 除以 hsl_saturation(100)
        - luminance: 除以 hsl_luminance(100)
        """
        payload = {}
        for name in hsl.__dataclass_fields__.keys():
            band = getattr(hsl, name)
            payload[name] = {
                "hue": band.hue * SCALES["hsl_hue_multiplier"],
                "saturation": band.saturation / SCALES["hsl_saturation"],
                "luminance": band.luminance / SCALES["hsl_luminance"],
            }
        return HslSettings.from_dict(payload)

    def _normalize_grading(self, grading: ColorGrading) -> ColorGrading:
        """将色调分级面板的 UI 参数归一化：
        
        - saturation: 除以 color_grading_saturation(500)
        - luminance: 除以 color_grading_luminance(500)
        - blending/balance: 保持原始 UI 值，由 apply_color_grading 内部再除以 100/200
          （与 Rust 端 blending/SCALES.color_grading_blending 等价，但此处分两步完成）
        - hue: 不归一化，直接以角度（0~360°）传入 hsv_to_rgb
        """
        data = {
            "shadows": {
                "hue": grading.shadows.hue,
                "saturation": grading.shadows.saturation / SCALES["color_grading_saturation"],
                "luminance": grading.shadows.luminance / SCALES["color_grading_luminance"],
            },
            "midtones": {
                "hue": grading.midtones.hue,
                "saturation": grading.midtones.saturation / SCALES["color_grading_saturation"],
                "luminance": grading.midtones.luminance / SCALES["color_grading_luminance"],
            },
            "highlights": {
                "hue": grading.highlights.hue,
                "saturation": grading.highlights.saturation / SCALES["color_grading_saturation"],
                "luminance": grading.highlights.luminance / SCALES["color_grading_luminance"],
            },
            "blending": grading.blending,
            "balance": grading.balance,
        }
        return ColorGrading.from_dict(data)

    def render_array(self, image_srgb_or_linear: np.ndarray, params: BasicColorParams, *, debug: bool = False) -> RenderOutput:
        """对 NumPy 图像数组执行完整的 Basic + Color 渲染流程。
        
        参数：
          image_srgb_or_linear: 输入图像，float32，shape=(H,W,3)，值域 [0,1]
          params: Basic + Color 参数
          debug: 若为 True，记录 4 个中间阶段的图像
        
        返回：
          RenderOutput，包含最终 sRGB 图像和（可选的）中间结果记录器
        """
        recorder = StageRecorder() if debug else None

        # 步骤 0：色彩空间转换 — 确保后续所有处理在线性光空间中进行
        if params.input_color_space == "linear":
            linear = image_srgb_or_linear.astype(np.float32)
        elif params.input_color_space == "srgb":
            linear = srgb_to_linear(image_srgb_or_linear.astype(np.float32))
        else:
            raise ValueError(f"Unsupported inputColorSpace: {params.input_color_space}")

        if recorder:
            recorder.add_linear("01_input", linear)

        norm = self._normalize_basic(params)

        # 预计算空间模糊图（对应 shader 中的 tonal_blur_texture）
        # 用于 apply_tonal_adjustments 的防晕圈阴影/黑色调整
        tonal_blur = gaussian_blur_rgb(linear, sigma=3.5)

        # ── Basic 面板处理流程 ──────────────────────────────────────────────────
        # 步骤 1：线性曝光（对应 shader apply_linear_exposure，在 apply_all_adjustments 之前）
        working = apply_linear_exposure(linear, float(norm["exposure"]))
        # 步骤 2：白平衡（对应 shader apply_white_balance）
        working = apply_white_balance(working, float(norm["temperature"]), float(norm["tint"]))
        # 步骤 3：胶片式亮度（对应 shader apply_filmic_exposure）
        working = apply_filmic_exposure(working, float(norm["brightness"]))
        # 步骤 4：白色/阴影/黑色/对比度（对应 shader apply_tonal_adjustments）
        working = apply_tonal_adjustments(
            working,
            tonal_blur,
            float(norm["contrast"]),
            float(norm["shadows"]),
            float(norm["whites"]),
            float(norm["blacks"]),
        )
        # 步骤 5：高光（对应 shader apply_highlights_adjustment）
        working = apply_highlights_adjustment(working, float(norm["highlights"]))
        if recorder:
            recorder.add_linear("02_after_basic", working)

        # ── Color 面板处理流程 ──────────────────────────────────────────────────
        # 步骤 6：HSL 混合器（对应 shader apply_hsl_panel）
        working = apply_hsl_mixer(working, self._normalize_hsl(params.hsl))
        # 步骤 7：色调分级（对应 shader apply_color_grading）
        working = apply_color_grading(working, self._normalize_grading(params.color_grading))
        # 步骤 8：饱和度 + 自然饱和度（对应 shader apply_creative_color）
        working = apply_creative_color(working, float(norm["saturation"]), float(norm["vibrance"]))
        if recorder:
            recorder.add_linear("03_after_color", working)

        # ── 色调映射 ────────────────────────────────────────────────────────────
        # 步骤 9：色调映射（对应 shader 中 tonemapper_mode 的选择）
        if str(norm["tone_mapper"]) == "agx":
            # AgX 色调映射：输出已在 sRGB 线性空间（agx_full_transform 内部不做 linear_to_srgb）
            out = np.clip(agx_full_transform(working), 0.0, 1.0).astype(np.float32)
        else:
            # basic 模式：直接做 linear → sRGB 伽马编码（对应 shader legacy_tonemap 后的 linear_to_srgb）
            out = linear_to_srgb(np.clip(working, 0.0, None)).astype(np.float32)
        if recorder:
            recorder.add_srgb("04_output", out)
        return RenderOutput(image_srgb=out, recorder=recorder)

    def render_file(
        self,
        input_path: str | Path,
        params: BasicColorParams,
        output_path: str | Path,
        *,
        debug_dir: Optional[str | Path] = None,
        quality: int = 95,
    ) -> RenderOutput:
        """从文件加载图像，执行渲染，并将结果保存到文件。
        
        参数：
          input_path: 输入图像路径（支持 JPG/PNG/TIF）
          params: Basic + Color 参数
          output_path: 输出图像路径
          debug_dir: 若指定，将 4 个中间阶段图像保存到该目录
          quality: JPEG 输出质量（1~95）
        
        返回：RenderOutput（同 render_array）
        """
        image = load_image(input_path)
        result = self.render_array(image, params, debug=debug_dir is not None)
        save_image(output_path, result.image_srgb, quality=quality)

        if debug_dir and result.recorder:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            for name, img in result.recorder.stages.items():
                save_image(debug_path / f"{name}.png", img, quality=95)
        return result
