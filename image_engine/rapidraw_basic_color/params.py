from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


# HSL 面板的 8 个色带名称，顺序与 shader.wgsl 中 HSL_RANGES 数组对应
HSL_BAND_NAMES = [
    "reds",
    "oranges",
    "yellows",
    "greens",
    "aquas",
    "blues",
    "purples",
    "magentas",
]


@dataclass
class HslBand:
    """单个 HSL 色带的调整值（对应 shader.wgsl 中 HslColor 结构体）。
    
    hue: 色相偏移量，UI 单位（归一化前），范围约 -100~100
    saturation: 饱和度调整量，UI 单位
    luminance: 明度调整量，UI 单位
    """
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "HslBand":
        data = data or {}
        return HslBand(
            hue=float(data.get("hue", 0.0)),
            saturation=float(data.get("saturation", 0.0)),
            luminance=float(data.get("luminance", 0.0)),
        )


@dataclass
class HslSettings:
    """HSL 面板全部 8 个色带的设置集合（对应 shader.wgsl 中 hsl: array<HslColor, 8>）。"""
    reds: HslBand = field(default_factory=HslBand)
    oranges: HslBand = field(default_factory=HslBand)
    yellows: HslBand = field(default_factory=HslBand)
    greens: HslBand = field(default_factory=HslBand)
    aquas: HslBand = field(default_factory=HslBand)
    blues: HslBand = field(default_factory=HslBand)
    purples: HslBand = field(default_factory=HslBand)
    magentas: HslBand = field(default_factory=HslBand)

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "HslSettings":
        data = data or {}
        return HslSettings(**{name: HslBand.from_dict(data.get(name)) for name in HSL_BAND_NAMES})


@dataclass
class ColorGradeBand:
    """单个色调分级区域（暗部/中间调/高光）的设置（对应 shader.wgsl 中 ColorGradeSettings 结构体）。
    
    hue: 色相角度，0~360 度
    saturation: 饱和度强度，UI 单位（归一化前）
    luminance: 亮度偏移量，UI 单位（归一化前）
    """
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "ColorGradeBand":
        data = data or {}
        return ColorGradeBand(
            hue=float(data.get("hue", 0.0)),
            saturation=float(data.get("saturation", 0.0)),
            luminance=float(data.get("luminance", 0.0)),
        )


@dataclass
class ColorGrading:
    """色调分级面板完整设置（对应 shader.wgsl 中 color_grading_* 系列字段）。
    
    shadows/midtones/highlights: 三个色调区域的 HSL 设置
    blending: 区域边界羽化程度，UI 单位 0~100，归一化后除以 100
    balance: 暗部/高光区域平衡，UI 单位 -100~100，归一化后除以 200
    """
    shadows: ColorGradeBand = field(default_factory=ColorGradeBand)
    midtones: ColorGradeBand = field(default_factory=ColorGradeBand)
    highlights: ColorGradeBand = field(default_factory=ColorGradeBand)
    blending: float = 50.0
    balance: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "ColorGrading":
        data = data or {}
        return ColorGrading(
            shadows=ColorGradeBand.from_dict(data.get("shadows")),
            midtones=ColorGradeBand.from_dict(data.get("midtones")),
            highlights=ColorGradeBand.from_dict(data.get("highlights")),
            blending=float(data.get("blending", 50.0)),
            balance=float(data.get("balance", 0.0)),
        )


@dataclass
class BasicColorParams:
    """Basic + Color 面板的完整参数集合（对应 shader.wgsl GlobalAdjustments 中 Basic/Color 相关字段）。
    
    所有数值均为 UI 原始单位，engine.py 中的 _normalize_* 方法负责将其除以对应 SCALES 后
    传给各处理函数，与 Rust 端 image_processing.rs 中的 SCALES 常量保持一致。
    """

    # 输入色彩空间："srgb" 或 "linear"
    input_color_space: str = "srgb"
    # 色调映射器："basic"（linear_to_srgb）或 "agx"（AgX 色调映射）
    tone_mapper: str = "basic"

    # Basic 面板
    exposure: float = 0.0      # 曝光，UI 单位，SCALE=0.8
    brightness: float = 0.0    # 亮度（filmic），UI 单位，SCALE=0.8
    contrast: float = 0.0      # 对比度，UI 单位，SCALE=100
    highlights: float = 0.0    # 高光，UI 单位，SCALE=120
    shadows: float = 0.0       # 阴影，UI 单位，SCALE=120
    whites: float = 0.0        # 白色，UI 单位，SCALE=30
    blacks: float = 0.0        # 黑色，UI 单位，SCALE=70

    # Color 面板
    temperature: float = 0.0   # 色温，UI 单位，SCALE=25
    tint: float = 0.0          # 色调，UI 单位，SCALE=100
    saturation: float = 0.0    # 饱和度，UI 单位，SCALE=100
    vibrance: float = 0.0      # 自然饱和度，UI 单位，SCALE=100
    hsl: HslSettings = field(default_factory=HslSettings)
    color_grading: ColorGrading = field(default_factory=ColorGrading)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasicColorParams":
        """从 JSON 字典构造参数对象，同时兼容 camelCase（前端格式）和 snake_case 两种键名。"""
        data = dict(data or {})
        return cls(
            input_color_space=str(data.get("inputColorSpace", data.get("input_color_space", "srgb"))).lower(),
            tone_mapper=str(data.get("toneMapper", data.get("tone_mapper", "basic"))).lower(),
            exposure=float(data.get("exposure", 0.0)),
            brightness=float(data.get("brightness", 0.0)),
            contrast=float(data.get("contrast", 0.0)),
            highlights=float(data.get("highlights", 0.0)),
            shadows=float(data.get("shadows", 0.0)),
            whites=float(data.get("whites", 0.0)),
            blacks=float(data.get("blacks", 0.0)),
            temperature=float(data.get("temperature", 0.0)),
            tint=float(data.get("tint", 0.0)),
            saturation=float(data.get("saturation", 0.0)),
            vibrance=float(data.get("vibrance", 0.0)),
            hsl=HslSettings.from_dict(data.get("hsl")),
            color_grading=ColorGrading.from_dict(data.get("colorGrading", data.get("color_grading"))),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BasicColorParams":
        """从 JSON 文件加载参数，文件格式见 examples/params.basic_color.json。"""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)
