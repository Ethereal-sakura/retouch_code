from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict

import numpy as np

from .params import ColorGrading, HSL_BAND_NAMES, HslSettings


# ITU-R BT.709 亮度系数，与 shader.wgsl 中 LUMA_COEFF 完全一致
LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# CIE D65 白点 xy 色度坐标（标准日光）
WP_D65 = np.array([0.3127, 0.3290], dtype=np.float32)

# sRGB 色域三原色 xy 色度坐标（IEC 61966-2-1）
PRIMARIES_SRGB = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], dtype=np.float32)

# Rec.2020 色域三原色 xy 色度坐标（AgX 工作色域）
PRIMARIES_REC2020 = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], dtype=np.float32)

# HSL 面板 8 个色带的中心色相角度和宽度（与 shader.wgsl HSL_RANGES 数组完全一致）
HSL_RANGES = {
    "reds":     (358.0, 35.0),
    "oranges":  (25.0,  45.0),
    "yellows":  (60.0,  40.0),
    "greens":   (115.0, 90.0),
    "aquas":    (180.0, 60.0),
    "blues":    (225.0, 60.0),
    "purples":  (280.0, 55.0),
    "magentas": (330.0, 50.0),
}


def get_luma(rgb: np.ndarray) -> np.ndarray:
    """计算线性 RGB 图像的感知亮度（BT.709 加权）。
    
    对应 shader.wgsl fn get_luma()：dot(c, LUMA_COEFF)
    返回 shape=(H, W) 的 float32 数组。
    """
    return np.tensordot(rgb.astype(np.float32), LUMA_COEFF, axes=([-1], [0]))


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB 伽马编码 → 线性光强度（IEC 61966-2-1 标准）。
    
    对应 shader.wgsl fn srgb_to_linear()：
      c <= 0.04045: c / 12.92
      c >  0.04045: ((c + 0.055) / 1.055) ^ 2.4
    """
    rgb = rgb.astype(np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """线性光强度 → sRGB 伽马编码（IEC 61966-2-1 标准）。
    
    对应 shader.wgsl fn linear_to_srgb()：
      c <= 0.0031308: c * 12.92
      c >  0.0031308: 1.055 * c^(1/2.4) - 0.055
    输入先 clip 到 [0, 1]。
    """
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055).astype(np.float32)


def smoothstep(edge0: float | np.ndarray, edge1: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    """GLSL smoothstep 函数：在 [edge0, edge1] 区间内做三次 Hermite 插值。
    
    对应 shader.wgsl 内置 smoothstep()：
      t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
      return t*t*(3 - 2*t)
    广泛用于各处理函数中生成平滑遮罩。
    """
    edge0_arr = np.asarray(edge0, dtype=np.float32)
    edge1_arr = np.asarray(edge1, dtype=np.float32)
    denom = np.maximum(edge1_arr - edge0_arr, 1e-8)
    t = np.clip((x - edge0_arr) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def mix(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """GLSL mix 函数：线性插值 a*(1-t) + b*t。
    
    对应 shader.wgsl 内置 mix()，广泛用于各处理函数中的混合操作。
    """
    return (a * (1.0 - t) + b * t).astype(np.float32)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV 色彩空间转换。
    
    对应 shader.wgsl fn rgb_to_hsv()：
      H: 0~360 度，S: 0~1（饱和度），V: 0~∞（明度，允许超过 1 以保留 HDR 信息）
    注意：Python 版用 % 360 处理红色区域的负角度，与 shader 中 h < 0 时 h += 360 等价。
    返回 shape=(H, W, 3) 的 float32 数组，通道顺序为 [H, S, V]。
    """
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax, dtype=np.float32)
    nz = delta > 1e-8
    rmask = nz & (cmax == r)
    gmask = nz & (cmax == g)
    bmask = nz & (cmax == b)
    hue[rmask] = (60.0 * ((g[rmask] - b[rmask]) / delta[rmask])) % 360.0
    hue[gmask] = 60.0 * (((b[gmask] - r[gmask]) / delta[gmask]) + 2.0)
    hue[bmask] = 60.0 * (((r[bmask] - g[bmask]) / delta[bmask]) + 4.0)
    sat = np.where(cmax > 1e-8, delta / np.maximum(cmax, 1e-8), 0.0).astype(np.float32)
    return np.stack([hue, sat, cmax], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV → RGB 色彩空间转换。
    
    对应 shader.wgsl fn hsv_to_rgb()：
      C = V*S，X = C*(1 - |H/60 mod 2 - 1|)，m = V - C
      按 H 所在 60° 扇区分配 R/G/B 分量
    V 允许超过 1（HDR），S clip 到 [0,1]。
    返回 shape=(H, W, 3) 的 float32 数组。
    """
    hsv = hsv.astype(np.float32)
    h = hsv[..., 0] % 360.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, None)
    c = v * s
    x = c * (1.0 - np.abs(((h / 60.0) % 2.0) - 1.0))
    m = v - c

    rgb = np.zeros_like(hsv, dtype=np.float32)
    conds = [
        h < 60.0,
        (h >= 60.0) & (h < 120.0),
        (h >= 120.0) & (h < 180.0),
        (h >= 180.0) & (h < 240.0),
        (h >= 240.0) & (h < 300.0),
        h >= 300.0,
    ]
    vals = [
        (c, x, 0.0),
        (x, c, 0.0),
        (0.0, c, x),
        (0.0, x, c),
        (x, 0.0, c),
        (c, 0.0, x),
    ]
    for cond, (rv, gv, bv) in zip(conds, vals):
        rgb[..., 0] = np.where(cond, rv, rgb[..., 0])
        rgb[..., 1] = np.where(cond, gv, rgb[..., 1])
        rgb[..., 2] = np.where(cond, bv, rgb[..., 2])
    return (rgb + m[..., None]).astype(np.float32)


def apply_white_balance(rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """白平衡调整：对 RGB 通道分别施加色温和色调乘数。
    
    对应 shader.wgsl fn apply_white_balance()（归一化后的参数）：
      temp_mult = [1 + temp*0.2, 1 + temp*0.05, 1 - temp*0.2]  (暖色调增红减蓝)
      tint_mult = [1 + tnt*0.25, 1 - tnt*0.25, 1 + tnt*0.25]  (品红/绿偏移)
      rgb *= temp_mult * tint_mult
    
    参数 temperature/tint 为归一化后的值（已除以 SCALES["temperature"/"tint"]）。
    在线性光空间中操作，在 apply_linear_exposure 之后、apply_filmic_exposure 之前调用。
    """
    temp_mult = np.array([1.0 + temperature * 0.2, 1.0 + temperature * 0.05, 1.0 - temperature * 0.2], dtype=np.float32)
    tint_mult = np.array([1.0 + tint * 0.25, 1.0 - tint * 0.25, 1.0 + tint * 0.25], dtype=np.float32)
    return (rgb * temp_mult * tint_mult).astype(np.float32)


def apply_creative_color(rgb: np.ndarray, saturation: float, vibrance: float) -> np.ndarray:
    """饱和度 + 自然饱和度调整。
    
    对应 shader.wgsl fn apply_creative_color()（归一化后的参数）：
    
    饱和度（saturation）：
      mix(luma, color, 1 + sat)  — 在灰色和原色之间插值，sat>0 增饱和，sat<0 去饱和
    
    自然饱和度（vibrance）：
      正值：优先提升低饱和像素，同时对肤色（色相约 25°）施加抑制
        amount = vib * sat_mask * skin_dampener * 3.0
      负值：优先降低低饱和像素的饱和度
        amount = vib * desat_mask
      最终：mix(gray, color, 1 + amount)
    
    注意：shader 中有 delta < 0.02 的早退出（纯灰像素跳过 vibrance），
    Python 版未实现此优化，功能等价但对纯灰像素会多做一次无效计算。
    """
    rgb = rgb.astype(np.float32)
    luma = get_luma(rgb)[..., None]
    out = rgb.copy()

    if abs(saturation) > 1e-8:
        out = mix(luma, out, 1.0 + saturation)
    if abs(vibrance) <= 1e-8:
        return out

    cmax = np.max(out, axis=-1)
    cmin = np.min(out, axis=-1)
    delta = cmax - cmin
    current_sat = delta / np.maximum(cmax, 1e-3)
    hsv = rgb_to_hsv(np.clip(out, 0.0, None))
    hue = hsv[..., 0]

    if vibrance > 0.0:
        # 低饱和遮罩：已饱和的像素受影响较小
        sat_mask = 1.0 - smoothstep(0.4, 0.9, current_sat)
        # 肤色保护：色相接近 25°（橙色）的区域受抑制
        hue_dist = np.minimum(np.abs(hue - 25.0), 360.0 - np.abs(hue - 25.0))
        skin_mask = smoothstep(35.0, 10.0, hue_dist)
        skin_dampener = 1.0 + (0.6 - 1.0) * skin_mask
        amount = vibrance * sat_mask * skin_dampener * 3.0
    else:
        # 负 vibrance：低饱和像素优先去饱和
        desat_mask = 1.0 - smoothstep(0.2, 0.8, current_sat)
        amount = vibrance * desat_mask

    gray = np.repeat(get_luma(out)[..., None], 3, axis=-1)
    return mix(gray, out, (1.0 + amount)[..., None])


def _raw_hsl_influence(hue: np.ndarray, center: float, width: float) -> np.ndarray:
    """计算像素色相对某个 HSL 色带的影响权重（高斯衰减）。
    
    对应 shader.wgsl fn get_raw_hsl_influence()：
      dist = min(|hue - center|, 360 - |hue - center|)  (环形距离)
      falloff = dist / (width * 0.5)
      return exp(-1.5 * falloff^2)  (高斯核，sharpness=1.5)
    """
    dist = np.minimum(np.abs(hue - center), 360.0 - np.abs(hue - center))
    falloff = dist / max(width * 0.5, 1e-6)
    return np.exp(-1.5 * falloff * falloff).astype(np.float32)


def apply_hsl_mixer(rgb: np.ndarray, settings: HslSettings) -> np.ndarray:
    """HSL 混合器：对 8 个色带分别调整色相、饱和度、明度。
    
    对应 shader.wgsl fn apply_hsl_panel()（归一化后的参数）：
    
    1. 转换到 HSV 空间，计算各色带的高斯影响权重并归一化（softmax 式）
    2. 饱和度遮罩（saturation_mask）：低饱和像素（灰色）减少影响，避免色偏
    3. 明度权重（lum_weight）：按饱和度加权，避免对灰色像素做明度调整
    4. 累加各色带的色相偏移、饱和度乘数、明度调整
    5. 先修改 HSV 中的 H/S，转回 RGB，再按目标亮度缩放（保持亮度一致性）
    
    参数 settings 为归一化后的 HslSettings（hue 已乘 0.3，sat/lum 已除以 100）。
    """
    rgb = rgb.astype(np.float32)
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, None))
    original_luma = get_luma(rgb)
    # 低饱和遮罩：饱和度 < 0.05 的像素（接近灰色）几乎不受 HSL 影响
    sat_mask = smoothstep(0.05, 0.20, hsv[..., 1])
    # 明度权重：按饱和度加权，避免对灰色像素做明度调整
    lum_weight = smoothstep(0.0, 1.0, hsv[..., 1])
    hue = hsv[..., 0]

    # 计算每个像素对 8 个色带的原始高斯影响权重
    infs = []
    for name in HSL_BAND_NAMES:
        center, width = HSL_RANGES[name]
        infs.append(_raw_hsl_influence(hue, center, width))
    inf = np.stack(infs, axis=-1)
    # 归一化：使各色带权重之和为 1（等价于 shader 中的 normalized_influence）
    inf_sum = np.maximum(np.sum(inf, axis=-1, keepdims=True), 1e-8)
    inf = inf / inf_sum

    total_hue = np.zeros_like(original_luma)
    total_sat = np.zeros_like(original_luma)
    total_lum = np.zeros_like(original_luma)
    for i, name in enumerate(HSL_BAND_NAMES):
        band = getattr(settings, name)
        hs_inf = inf[..., i] * sat_mask    # 色相/饱和度影响（受饱和度遮罩调制）
        l_inf = inf[..., i] * lum_weight   # 明度影响（受饱和度权重调制）
        # hue 已在 _normalize_hsl 中乘以 hsl_hue_multiplier(0.3)，此处再乘 2 还原 shader 中的 *2
        total_hue += band.hue * 2.0 * hs_inf
        total_sat += band.saturation * hs_inf
        total_lum += band.luminance * l_inf

    # 修改 H/S 分量并转回 RGB
    new_hsv = hsv.copy()
    new_hsv[..., 0] = (new_hsv[..., 0] + total_hue + 360.0) % 360.0
    new_hsv[..., 1] = np.clip(new_hsv[..., 1] * (1.0 + total_sat), 0.0, 1.0)
    hs_rgb = hsv_to_rgb(new_hsv)
    # 按目标亮度缩放 RGB，保持亮度一致性
    new_luma = get_luma(hs_rgb)
    target_luma = original_luma * (1.0 + total_lum)
    scale = target_luma / np.maximum(new_luma, 1e-4)
    return (hs_rgb * scale[..., None]).astype(np.float32)


def apply_color_grading(rgb: np.ndarray, grading: ColorGrading) -> np.ndarray:
    """色调分级：对暗部、中间调、高光分别施加色相/饱和度/亮度偏移。
    
    对应 shader.wgsl fn apply_color_grading()（归一化后的参数）：
    
    1. 根据 balance 参数调整暗部/高光的交叉点位置：
       balance > 0 → 高光区域扩大（highlight_crossover 减小）
       balance < 0 → 暗部区域扩大（shadow_crossover 增大）
    2. 根据 blending 参数控制区域边界的羽化宽度（feather = 0.2 * blending）
    3. 三个区域的遮罩：shadow_mask + highlight_mask + midtone_mask ≈ 1
    4. 色彩偏移：将目标色相转为 RGB（HSV 饱和度=1，明度=1），
       与 0.5 的差值乘以饱和度强度和区域遮罩叠加到图像上
    5. 亮度偏移：luminance 值直接乘以区域遮罩叠加
    
    参数 grading 为归一化后的 ColorGrading（sat/lum 已除以 500，blending/100，balance/200）。
    """
    rgb = rgb.astype(np.float32)
    luma = get_luma(np.maximum(rgb, 0.0))
    balance = grading.balance / 200.0
    blending = grading.blending / 100.0

    # 计算三个区域的交叉点（受 balance 调整）
    base_shadow_crossover = 0.1
    base_highlight_crossover = 0.5
    balance_range = 0.5
    shadow_crossover = base_shadow_crossover + max(0.0, -balance) * balance_range
    highlight_crossover = base_highlight_crossover - max(0.0, balance) * balance_range
    feather = 0.2 * blending
    final_shadow_crossover = min(shadow_crossover, highlight_crossover - 0.01)

    # 生成三个区域的平滑遮罩
    shadow_mask = 1.0 - smoothstep(final_shadow_crossover - feather, final_shadow_crossover + feather, luma)
    highlight_mask = smoothstep(highlight_crossover - feather, highlight_crossover + feather, luma)
    midtone_mask = np.maximum(0.0, 1.0 - shadow_mask - highlight_mask)

    graded = rgb.copy()
    # 各区域的色彩/亮度影响强度（与 shader.wgsl 中常量完全一致）
    strengths = {
        "shadow_sat": 0.3,
        "shadow_lum": 0.5,
        "midtone_sat": 0.6,
        "midtone_lum": 0.8,
        "highlight_sat": 0.8,
        "highlight_lum": 1.0,
    }

    # 暗部色彩偏移：将目标色相（全饱和、全亮度）转为 RGB，以 0.5 为中心叠加
    if grading.shadows.saturation > 1e-3:
        tint_rgb = hsv_to_rgb(np.dstack([
            np.full_like(luma, grading.shadows.hue),
            np.ones_like(luma),
            np.ones_like(luma),
        ]))
        graded += (tint_rgb - 0.5) * grading.shadows.saturation * shadow_mask[..., None] * strengths["shadow_sat"]
    graded += grading.shadows.luminance * shadow_mask[..., None] * strengths["shadow_lum"]

    # 中间调色彩偏移
    if grading.midtones.saturation > 1e-3:
        tint_rgb = hsv_to_rgb(np.dstack([
            np.full_like(luma, grading.midtones.hue),
            np.ones_like(luma),
            np.ones_like(luma),
        ]))
        graded += (tint_rgb - 0.5) * grading.midtones.saturation * midtone_mask[..., None] * strengths["midtone_sat"]
    graded += grading.midtones.luminance * midtone_mask[..., None] * strengths["midtone_lum"]

    # 高光色彩偏移
    if grading.highlights.saturation > 1e-3:
        tint_rgb = hsv_to_rgb(np.dstack([
            np.full_like(luma, grading.highlights.hue),
            np.ones_like(luma),
            np.ones_like(luma),
        ]))
        graded += (tint_rgb - 0.5) * grading.highlights.saturation * highlight_mask[..., None] * strengths["highlight_sat"]
    graded += grading.highlights.luminance * highlight_mask[..., None] * strengths["highlight_lum"]

    return graded.astype(np.float32)


# ─── AgX 色调映射相关函数 ───────────────────────────────────────────────────────


def _xy_to_xyz(xy: np.ndarray) -> np.ndarray:
    """将 CIE xy 色度坐标转换为 XYZ 三刺激值（Y=1 归一化）。"""
    if xy[1] < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]], dtype=np.float32)


def _primaries_to_xyz_matrix(primaries: np.ndarray, white_point: np.ndarray) -> np.ndarray:
    """根据三原色 xy 坐标和白点构建 RGB→XYZ 转换矩阵（Bradford 适应法）。
    
    用于构建 sRGB→XYZ 和 Rec.2020→XYZ 的转换矩阵，
    进而推导出 AgX 的 pipe_to_rendering 和 rendering_to_pipe 矩阵。
    """
    r_xyz = _xy_to_xyz(primaries[0])
    g_xyz = _xy_to_xyz(primaries[1])
    b_xyz = _xy_to_xyz(primaries[2])
    pm = np.stack([r_xyz, g_xyz, b_xyz], axis=1)
    white_xyz = _xy_to_xyz(white_point)
    s = np.linalg.inv(pm) @ white_xyz
    return np.stack([r_xyz * s[0], g_xyz * s[1], b_xyz * s[2]], axis=1).astype(np.float32)


def _rotate_and_scale_primary(primary: np.ndarray, white_point: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    """对色域三原色进行缩放和旋转（用于 AgX inset/outset 色域变换）。
    
    以白点为原点，对三原色的 xy 坐标进行极坐标变换：
      先缩放（scale），再旋转（rotation 弧度）
    """
    p_rel = primary - white_point
    p_scaled = p_rel * scale
    sin_r = math.sin(rotation)
    cos_r = math.cos(rotation)
    return np.array([
        white_point[0] + p_scaled[0] * cos_r - p_scaled[1] * sin_r,
        white_point[1] + p_scaled[0] * sin_r + p_scaled[1] * cos_r,
    ], dtype=np.float32)


@lru_cache(maxsize=1)
def agx_matrices() -> Dict[str, np.ndarray]:
    """计算并缓存 AgX 色调映射所需的两个 3×3 色彩空间转换矩阵。
    
    对应 shader.wgsl 中 GlobalAdjustments 的 agx_pipe_to_rendering_matrix 和
    agx_rendering_to_pipe_matrix（由 Rust 端 image_processing.rs 预计算后传入 GPU）。
    
    AgX 色调映射流程：
      1. pipe_to_rendering：sRGB 线性 → AgX 渲染色域（inset + 旋转的 Rec.2020）
      2. 在 AgX 色域中做对数编码 + S 形曲线（agx_tonemap）
      3. rendering_to_pipe：AgX 渲染色域 → sRGB 线性（outset Rec.2020 的逆）
    
    inset 参数：将 Rec.2020 三原色向白点内缩，避免极端色彩溢出
    rotation 参数：轻微旋转，使色调映射后的色调更自然
    outset 参数：输出时向外扩展，恢复部分色域范围
    """
    pipe_work_profile_to_xyz = _primaries_to_xyz_matrix(PRIMARIES_SRGB, WP_D65)
    base_profile_to_xyz = _primaries_to_xyz_matrix(PRIMARIES_REC2020, WP_D65)
    xyz_to_base = np.linalg.inv(base_profile_to_xyz)
    pipe_to_base = xyz_to_base @ pipe_work_profile_to_xyz

    # AgX inset：将 Rec.2020 三原色向白点内缩的比例
    inset = [0.29462451, 0.25861925, 0.14641371]
    # AgX rotation：三原色的轻微旋转角度（弧度）
    rotation = [0.03540329, -0.02108586, -0.06305724]
    # AgX outset：输出色域向外扩展的比例
    outset = [0.290776401758, 0.263155400753, 0.045810721815]

    inset_rot = np.stack([
        _rotate_and_scale_primary(PRIMARIES_REC2020[i], WP_D65, 1.0 - inset[i], rotation[i])
        for i in range(3)
    ], axis=0)
    rendering_to_xyz = _primaries_to_xyz_matrix(inset_rot, WP_D65)
    base_to_rendering = xyz_to_base @ rendering_to_xyz

    outset_unrot = np.stack([
        _rotate_and_scale_primary(PRIMARIES_REC2020[i], WP_D65, 1.0 - outset[i], 0.0)
        for i in range(3)
    ], axis=0)
    outset_to_xyz = _primaries_to_xyz_matrix(outset_unrot, WP_D65)
    rendering_to_base = np.linalg.inv(xyz_to_base @ outset_to_xyz)

    return {
        "pipe_to_rendering": (base_to_rendering @ pipe_to_base).astype(np.float32),
        "rendering_to_pipe": (np.linalg.inv(pipe_to_base) @ rendering_to_base).astype(np.float32),
    }


def agx_full_transform(rgb: np.ndarray) -> np.ndarray:
    """AgX 完整色调映射流程（对应 shader.wgsl fn agx_full_transform()）。
    
    流程：
    1. agx_compress_gamut：将负值通道压缩（加上最小负值），避免色域外颜色
    2. pipe_to_rendering 矩阵变换：sRGB 线性 → AgX 渲染色域
    3. 对数编码：x_relative = max(c / 0.18, ε)，映射到 [min_ev, max_ev]（-15.2 ~ 5.0 EV）
    4. AgX S 形曲线（分三段：toe、linear、shoulder）：
       - toe（x < 0.606）：scaled_sigmoid(x, -1.0359, 1.5)
       - linear（x ≈ 0.606）：slope*x + intercept（slope=2.3843，intercept=-1.0112）
       - shoulder（x > 0.606）：scaled_sigmoid(x, 1.3475, 1.5)
    5. gamma 编码：pow(curved, 2.4)
    6. rendering_to_pipe 矩阵变换：AgX 渲染色域 → sRGB 线性
    
    输出为 sRGB 线性空间（未做 linear_to_srgb），由 engine.py 直接输出。
    """
    mats = agx_matrices()
    rgb = rgb.astype(np.float32)
    # 步骤 1：色域压缩（对应 shader agx_compress_gamut）
    min_c = np.min(rgb, axis=-1, keepdims=True)
    compressed = np.where(min_c < 0.0, rgb - min_c, rgb)
    flat = compressed.reshape(-1, 3)
    # 步骤 2：变换到 AgX 渲染色域
    agx_space = flat @ mats["pipe_to_rendering"].T

    # 步骤 3：对数编码，映射到 [0, 1]
    epsilon = 1.0e-6
    min_ev = -15.2
    max_ev = 5.0
    range_ev = max_ev - min_ev
    x_relative = np.maximum(agx_space / 0.18, epsilon)
    mapped = np.clip((np.log2(x_relative) - min_ev) / range_ev, 0.0, 1.0)

    # 步骤 4：AgX S 形曲线参数（与 shader.wgsl 中常量完全一致）
    slope = 2.3843
    toe_power = 1.5
    shoulder_power = 1.5
    tx = 0.6060606      # toe/shoulder 过渡点 x
    ty = 0.43446        # toe/shoulder 过渡点 y
    intercept = -1.0112
    toe_scale = -1.0359
    shoulder_scale = 1.3475

    def sigmoid(v: np.ndarray, power: float) -> np.ndarray:
        """AgX sigmoid：v / (1 + v^power)^(1/power)"""
        return v / np.power(1.0 + np.power(v, power), 1.0 / power)

    def scaled_sigmoid(v: np.ndarray, scale: float, power: float) -> np.ndarray:
        """AgX scaled sigmoid：scale * sigmoid(slope*(v-tx)/scale, power) + ty"""
        return scale * sigmoid(slope * (v - tx) / scale, power) + ty

    curved = np.empty_like(mapped, dtype=np.float32)
    toe_mask = mapped < tx
    shoulder_mask = mapped > tx
    linear_mask = ~(toe_mask | shoulder_mask)
    curved[toe_mask] = scaled_sigmoid(mapped[toe_mask], toe_scale, toe_power)
    curved[linear_mask] = slope * mapped[linear_mask] + intercept
    curved[shoulder_mask] = scaled_sigmoid(mapped[shoulder_mask], shoulder_scale, shoulder_power)
    curved = np.clip(curved, 0.0, 1.0)
    # 步骤 5：gamma 编码（AgX 内部 gamma，与 sRGB gamma 不同）
    curved = np.power(curved, 2.4)
    # 步骤 6：变换回 sRGB 线性色域
    out = curved @ mats["rendering_to_pipe"].T
    return out.reshape(rgb.shape).astype(np.float32)
