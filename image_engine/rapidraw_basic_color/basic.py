from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .colors import get_luma, mix, smoothstep


# UI 参数到归一化值的缩放因子，与 Rust 端 image_processing.rs 中 SCALES 常量完全一致。
# engine.py 中的 _normalize_basic/_normalize_hsl/_normalize_grading 将 UI 值除以对应 SCALE
# 后传给各处理函数，使函数内部的数值范围与 shader.wgsl 一致。
SCALES = {
    "exposure": 0.8,
    "brightness": 0.8,
    "contrast": 100.0,
    "highlights": 120.0,
    "shadows": 120.0,
    "whites": 30.0,
    "blacks": 70.0,
    "temperature": 25.0,
    "tint": 100.0,
    "saturation": 100.0,
    "vibrance": 100.0,
    "color_grading_saturation": 500.0,
    "color_grading_luminance": 500.0,
    "color_grading_balance": 200.0,
    "hsl_hue_multiplier": 0.3,
    "hsl_saturation": 100.0,
    "hsl_luminance": 100.0,
}


def gaussian_blur_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    """对 RGB 图像的每个通道分别做高斯模糊，生成空间低频版本。
    
    对应 shader.wgsl 中的 tonal_blur_texture（由 Rust 端 GPU 模糊预计算后传入 shader）。
    模糊结果用于 apply_tonal_adjustments 中的空间自适应阴影/黑色调整（防晕圈）。
    
    sigma=3.5 对应约 7×7 像素的模糊半径，mode="nearest" 边界填充。
    """
    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    return np.stack([gaussian_filter(image[..., i], sigma=sigma, mode="nearest") for i in range(3)], axis=-1).astype(np.float32)


def apply_linear_exposure(image: np.ndarray, exposure_adj: float) -> np.ndarray:
    """线性曝光调整：乘以 2^exposure_adj。
    
    对应 shader.wgsl fn apply_linear_exposure()：
      color * pow(2.0, exposure_adj)
    
    在线性光空间中操作，是所有调整中最先执行的步骤。
    参数 exposure_adj 为归一化后的值（UI 值 / SCALES["exposure"] = UI值 / 0.8）。
    """
    if abs(exposure_adj) <= 1e-8:
        return image.astype(np.float32)
    return (image * (2.0 ** exposure_adj)).astype(np.float32)


def apply_filmic_exposure(image: np.ndarray, brightness_adj: float) -> np.ndarray:
    """胶片式亮度调整：保护高光和阴影的非线性亮度变换。
    
    对应 shader.wgsl fn apply_filmic_exposure()：
    
    将亮度调整分为两部分：
      - 直接调整（5%）：scale = 2^(brightness_adj * 0.05)，全局线性缩放
      - 有理曲线调整（95%）：对亮度分数部分做有理函数变换，保护高光/阴影
        k = 2^(-rational_adj * 1.2)
        shaped_fract = luma_fract / (luma_fract + (1 - luma_fract) * k)
    
    色度（chroma = color - luma）按亮度变化比例的 0.8 次方缩放，
    避免饱和度随亮度变化过大（chroma_scale = total_scale^0.8）。
    
    参数 brightness_adj 为归一化后的值（UI 值 / SCALES["brightness"] = UI值 / 0.8）。
    """
    if abs(brightness_adj) <= 1e-8:
        return image.astype(np.float32)

    luma = get_luma(image)
    mask = np.abs(luma) > 1e-5
    if not np.any(mask):
        return image.astype(np.float32)

    # RATIONAL_CURVE_MIX = 0.95（与 shader 常量一致）
    direct_adj = brightness_adj * (1.0 - 0.95)
    rational_adj = brightness_adj * 0.95
    scale = 2.0 ** direct_adj
    # MIDTONE_STRENGTH = 1.2（与 shader 常量一致）
    k = 2.0 ** (-rational_adj * 1.2)

    # 对亮度的整数部分和小数部分分别处理（支持 HDR 超过 1.0 的情况）
    luma_abs = np.abs(luma)
    luma_floor = np.floor(luma_abs)
    luma_fract = luma_abs - luma_floor
    # 有理函数变换：压缩/扩展小数部分
    shaped_fract = luma_fract / np.maximum(luma_fract + (1.0 - luma_fract) * k, 1e-6)
    shaped_luma_abs = luma_floor + shaped_fract
    new_luma = np.sign(luma) * shaped_luma_abs * scale

    # 色度缩放：按亮度变化比例的 0.8 次方，避免饱和度突变
    chroma = image - luma[..., None]
    total_scale = np.ones_like(luma, dtype=np.float32)
    total_scale[mask] = new_luma[mask] / luma[mask]
    chroma_scale = np.power(np.maximum(total_scale, 1e-6), 0.8)
    return (new_luma[..., None] + chroma * chroma_scale[..., None]).astype(np.float32)


def get_shadow_mult(luma: np.ndarray, shadows_adj: float, blacks_adj: float) -> np.ndarray:
    """计算阴影/黑色调整的亮度乘数。
    
    对应 shader.wgsl fn get_shadow_mult()：
    
    黑色（blacks）：影响范围 luma < 0.05
      x = luma / 0.05，mask = (1-x)^2（二次衰减）
      factor = min(2^(blacks_adj * 0.75), 3.9)
      mult *= mix(1.0, factor, mask)  等价于 1 + (factor-1)*mask
    
    阴影（shadows）：影响范围 luma < 0.1
      x = luma / 0.1，mask = (1-x)^2
      factor = min(2^(shadows_adj * 1.5), 3.9)
      mult *= mix(1.0, factor, mask)
    
    注意：Python 版用 1 + (factor-1)*mask 代替 mix(1, factor, mask)，数学等价。
    参数为归一化后的值（UI 值 / SCALES["shadows"/"blacks"]）。
    """
    mult = np.ones_like(luma, dtype=np.float32)
    safe = np.maximum(luma, 1e-4)

    if abs(blacks_adj) > 1e-8:
        limit = 0.05
        x = np.clip(safe / limit, 0.0, 1.0)
        mask = (1.0 - x) ** 2
        factor = min(2.0 ** (blacks_adj * 0.75), 3.9)
        mult *= 1.0 + (factor - 1.0) * mask

    if abs(shadows_adj) > 1e-8:
        limit = 0.1
        x = np.clip(safe / limit, 0.0, 1.0)
        mask = (1.0 - x) ** 2
        factor = min(2.0 ** (shadows_adj * 1.5), 3.9)
        mult *= 1.0 + (factor - 1.0) * mask

    return mult.astype(np.float32)


def apply_tonal_adjustments(image: np.ndarray, tonal_blur: np.ndarray, contrast_adj: float, shadows_adj: float, whites_adj: float, blacks_adj: float) -> np.ndarray:
    """色调调整：白色、阴影/黑色、对比度的综合处理。
    
    对应 shader.wgsl fn apply_tonal_adjustments()（归一化后的参数）：
    
    1. 白色（whites）：
       white_level = 1 - whites_adj * 0.25
       gain = 1 / max(white_level, 0.01)
       rgb *= gain（同时缩放模糊图，保持防晕圈一致性）
    
    2. 阴影/黑色（shadows/blacks）：空间自适应调整（防晕圈）
       - 用高斯模糊图（tonal_blur）计算空间亮度乘数（spatial_mult）
       - 用像素自身亮度计算像素乘数（pixel_mult）
       - 晕圈保护（halo_protection）：当像素亮度与模糊亮度差异大时，偏向像素乘数
         halo_protection = smoothstep(0.05, 0.25, |sqrt(pixel_luma) - sqrt(blur_luma)|)
       - 最终乘数 = mix(spatial_mult, pixel_mult, halo_protection)
    
    3. 对比度（contrast）：在感知伽马空间（gamma=2.2）中做 S 形曲线
       strength = 2^(contrast_adj * 1.25)
       low:  0.5 * (2*x)^strength
       high: 1 - 0.5 * (2*(1-x))^strength
       超过 1.0 的 HDR 像素通过 smoothstep(1.0, 1.01) 混合保护
    
    注意：Python 版的 tonal_blur 已在 engine.py 中预先转换到线性空间，
    而 shader 中 tonal_blur_texture 存储的是 sRGB 值，需在 shader 内部转换。
    """
    rgb = image.astype(np.float32, copy=True)
    blur = tonal_blur.astype(np.float32, copy=True)

    # 白色调整：缩放整体亮度上限
    if abs(whites_adj) > 1e-8:
        white_level = 1.0 - whites_adj * 0.25
        gain = 1.0 / max(white_level, 0.01)
        rgb *= gain
        blur *= gain

    pixel_luma = get_luma(np.maximum(rgb, 0.0))
    blur_luma = get_luma(np.maximum(blur, 0.0))
    safe_pixel = np.maximum(pixel_luma, 1e-4)
    safe_blur = np.maximum(blur_luma, 1e-4)

    # 晕圈保护：在感知空间（sqrt）中比较像素与模糊的差异
    halo_protection = smoothstep(0.05, 0.25, np.abs(np.sqrt(safe_pixel) - np.sqrt(safe_blur)))

    # 阴影/黑色：空间自适应调整
    if abs(shadows_adj) > 1e-8 or abs(blacks_adj) > 1e-8:
        spatial = get_shadow_mult(safe_blur, shadows_adj, blacks_adj)
        pixel = get_shadow_mult(safe_pixel, shadows_adj, blacks_adj)
        # 混合空间乘数和像素乘数，晕圈区域偏向像素乘数
        mult = spatial * (1.0 - halo_protection) + pixel * halo_protection
        rgb *= mult[..., None]

    # 对比度：在感知伽马空间中做 S 形曲线
    if abs(contrast_adj) > 1e-8:
        safe_rgb = np.maximum(rgb, 0.0)
        perceptual = np.power(safe_rgb, 1.0 / 2.2)
        perceptual = np.clip(perceptual, 0.0, 1.0)
        strength = 2.0 ** (contrast_adj * 1.25)
        low = 0.5 * np.power(2.0 * perceptual, strength)
        high = 1.0 - 0.5 * np.power(2.0 * (1.0 - perceptual), strength)
        curved = np.where(perceptual < 0.5, low, high)
        contrast_rgb = np.power(np.maximum(curved, 0.0), 2.2)
        # HDR 像素保护：超过 1.0 的部分不做对比度调整
        mix_factor = smoothstep(1.0, 1.01, safe_rgb)
        rgb = contrast_rgb * (1.0 - mix_factor) + rgb * mix_factor

    return rgb.astype(np.float32)


def apply_highlights_adjustment(image: np.ndarray, highlights_adj: float) -> np.ndarray:
    """高光调整：对高亮区域进行非线性压缩或提升。
    
    对应 shader.wgsl fn apply_highlights_adjustment()（归一化后的参数）：
    
    高光遮罩：
      pixel_mask_input = tanh(luma * 1.5)
      highlight_mask = smoothstep(0.3, 0.95, pixel_mask_input)
      （仅对亮度较高的区域生效）
    
    负值（高光恢复）：
      luma <= 1.0：new_luma = luma^(1 - highlights_adj * 1.75)（gamma 提亮）
      luma > 1.0：compressed_excess = (luma-1) / (1 + (luma-1) * (-highlights_adj*6))
                  new_luma = 1 + compressed_excess（压缩 HDR 溢出）
      去饱和：smoothstep(1.0, 10.0, luma) 控制高亮区域向白色混合
    
    正值（高光提亮）：
      final = color * 2^(highlights_adj * 1.75)
    
    最终：mix(color, final, highlight_mask)
    
    参数 highlights_adj 为归一化后的值（UI 值 / SCALES["highlights"] = UI值 / 120）。
    """
    if abs(highlights_adj) <= 1e-8:
        return image.astype(np.float32)

    luma = get_luma(np.maximum(image, 0.0))
    safe_luma = np.maximum(luma, 1e-4)
    # 高光遮罩：tanh 压缩后做 smoothstep，聚焦于高亮区域
    highlight_mask = smoothstep(0.3, 0.95, np.tanh(safe_luma * 1.5))

    if highlights_adj < 0.0:
        new_luma = luma.copy()
        # 正常范围内：gamma 曲线提亮（gamma > 1 → 压暗高光）
        low_mask = luma <= 1.0
        gamma = 1.0 - highlights_adj * 1.75
        new_luma[low_mask] = np.power(np.maximum(luma[low_mask], 0.0), gamma)
        # HDR 溢出部分：有理函数压缩
        high_mask = ~low_mask
        excess = luma[high_mask] - 1.0
        compression = -highlights_adj * 6.0
        compressed_excess = excess / (1.0 + excess * compression)
        new_luma[high_mask] = 1.0 + compressed_excess
        # 按新亮度缩放颜色
        toned = image * (new_luma / np.maximum(luma, 1e-4))[..., None]
        # 极高亮区域向白色混合（去饱和）
        desat = smoothstep(1.0, 10.0, luma)[..., None]
        white_point = np.repeat(new_luma[..., None], 3, axis=-1)
        final = mix(toned, white_point, desat)
    else:
        # 正值：直接提亮高光区域
        final = image * (2.0 ** (highlights_adj * 1.75))

    return (image * (1.0 - highlight_mask[..., None]) + final * highlight_mask[..., None]).astype(np.float32)
