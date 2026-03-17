"""Image statistics utilities.

Provides get_stat() and get_delta_stat() for computing per-image and
differential statistics used as quantitative anchors in the MLLM prompt.

Based on iclr_retouchllm/diff_tools.py get_stat(), extended with delta
computation and dominant-issue detection.
"""

import cv2
import numpy as np


HUE_BAND_RANGES = {
    "reds": [(0, 10), (170, 180)],
    "oranges": [(10, 20)],
    "yellows": [(20, 35)],
    "greens": [(35, 85)],
    "aquas": [(85, 100)],
    "blues": [(100, 130)],
    "purples": [(130, 150)],
    "magentas": [(150, 170)],
}


def get_stat(img: np.ndarray) -> dict:
    """Compute pixel-level statistics for an image.

    Parameters
    ----------
    img : np.ndarray
        Float32 image, shape (H, W, 3), values in [0, 1].

    Returns
    -------
    dict with keys:
        pixel mean, pixel median, pixel std,
        pixel percentail 10%, pixel percentail 90%,
        rgb mean, laplacian variance,
        saturation mean, saturation std, saturation min, saturation max,
        l-channel mean, b-channel mean
    """
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # RGB stats
    pixels = img_u8.ravel()
    img_mean = round(float(np.mean(pixels)), 2)
    img_median = round(float(np.median(pixels)), 2)
    img_std = round(float(np.std(pixels)), 2)
    img_p10 = round(float(np.percentile(pixels, 10)), 2)
    img_p90 = round(float(np.percentile(pixels, 90)), 2)
    rgb_mean = [round(float(v), 2) for v in np.mean(img_u8, axis=(0, 1))]

    # Sharpness via Laplacian variance
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_var = round(float(laplacian.var()), 2)

    # HSV saturation stats
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    s_mean = round(float(np.mean(hsv[..., 1])), 2)
    s_std = round(float(np.std(hsv[..., 1])), 2)
    s_min = round(float(np.min(hsv[..., 1])), 2)
    s_max = round(float(np.max(hsv[..., 1])), 2)

    # LAB channel stats
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l_mean = round(float(np.mean(lab[..., 0])), 2)
    b_mean = round(float(np.mean(lab[..., 2])), 2)

    return {
        "pixel mean": img_mean,
        "pixel median": img_median,
        "pixel std": img_std,
        "pixel percentail 10%": img_p10,
        "pixel percentail 90%": img_p90,
        "rgb mean": rgb_mean,
        "laplacian variance": laplacian_var,
        "saturation mean": s_mean,
        "saturation std": s_std,
        "saturation min": s_min,
        "saturation max": s_max,
        "l-channel mean": l_mean,
        "b-channel mean": b_mean,
    }


def _get_hue_band_stats(img_u8: np.ndarray) -> dict[str, dict[str, float]]:
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    stats: dict[str, dict[str, float]] = {}
    total = max(float(hue.size), 1.0)
    for band, ranges in HUE_BAND_RANGES.items():
        mask = np.zeros_like(hue, dtype=bool)
        for lo, hi in ranges:
            mask |= (hue >= lo) & (hue < hi)
        mask &= sat > 12

        if not np.any(mask):
            stats[band] = {
                "area_ratio": 0.0,
                "saturation_mean": 0.0,
                "value_mean": 0.0,
            }
            continue

        stats[band] = {
            "area_ratio": float(mask.sum()) / total,
            "saturation_mean": float(np.mean(sat[mask])),
            "value_mean": float(np.mean(val[mask])),
        }
    return stats


def get_delta_stat(current_img: np.ndarray, target_img: np.ndarray) -> dict:
    """Compute difference statistics between current and target images.

    Maps image differences to tool-relevant signals and identifies the most
    dominant issue for guiding tool selection.

    Parameters
    ----------
    current_img : np.ndarray
        Current (in-progress) image, float32 [0, 1].
    target_img : np.ndarray
        Target (ground truth) image, float32 [0, 1].

    Returns
    -------
    dict with delta values and 'dominant_issue' key.
    """
    s_cur = get_stat(current_img)
    s_tgt = get_stat(target_img)
    cur_u8 = np.clip(current_img * 255.0, 0, 255).astype(np.uint8)
    tgt_u8 = np.clip(target_img * 255.0, 0, 255).astype(np.uint8)
    cur_band_stats = _get_hue_band_stats(cur_u8)
    tgt_band_stats = _get_hue_band_stats(tgt_u8)

    delta = {
        # exposure_tool signals
        "brightness_delta": round(s_tgt["pixel mean"] - s_cur["pixel mean"], 2),
        "l_channel_delta": round(s_tgt["l-channel mean"] - s_cur["l-channel mean"], 2),
        # tone_tool signals
        "contrast_delta": round(s_tgt["pixel std"] - s_cur["pixel std"], 2),
        "highlight_delta": round(
            s_tgt["pixel percentail 90%"] - s_cur["pixel percentail 90%"], 2
        ),
        "shadow_delta": round(
            s_tgt["pixel percentail 10%"] - s_cur["pixel percentail 10%"], 2
        ),
        # white_balance_tool signals
        "temperature_delta": round(
            s_tgt["b-channel mean"] - s_cur["b-channel mean"], 2
        ),
        "tint_delta": round(
            s_tgt["rgb mean"][1] - s_cur["rgb mean"][1], 2
        ),
        # saturation_tool signal
        "saturation_delta": round(
            s_tgt["saturation mean"] - s_cur["saturation mean"], 2
        ),
        "hue_band_residuals": {},
        "dominant_hsl_band": None,
        "dominant_issue": None,
    }

    for band in HUE_BAND_RANGES:
        cur_band = cur_band_stats[band]
        tgt_band = tgt_band_stats[band]
        residual = (
            abs(tgt_band["area_ratio"] - cur_band["area_ratio"]) * 2.0
            + abs(tgt_band["saturation_mean"] - cur_band["saturation_mean"]) / 255.0
            + abs(tgt_band["value_mean"] - cur_band["value_mean"]) / 255.0
        )
        delta["hue_band_residuals"][band] = round(float(residual), 4)

    delta["dominant_hsl_band"] = max(
        delta["hue_band_residuals"],
        key=delta["hue_band_residuals"].get,
    )

    # Identify the most dominant issue (normalized to [0, 1] scale)
    issues = {
        "exposure": abs(delta["brightness_delta"]) / 255.0,
        "tone": abs(delta["contrast_delta"]) / 255.0,
        "white_balance": abs(delta["temperature_delta"]) / 255.0,
        "saturation": abs(delta["saturation_delta"]) / 255.0,
        "hsl": delta["hue_band_residuals"][delta["dominant_hsl_band"]],
    }
    delta["dominant_issue"] = max(issues, key=issues.get)
    return delta
