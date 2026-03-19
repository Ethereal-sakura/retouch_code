"""Helpers for snapping tool deltas onto a discrete action grid."""

from __future__ import annotations

import json
import math
from typing import Any

DEFAULT_ACTION_STEP = 1.0
HSL_NUMERIC_KEYS = ("hue", "saturation", "luminance")


def quantize_tool_delta(
    tool_name: str,
    delta_params: dict[str, Any],
    *,
    step: float = DEFAULT_ACTION_STEP,
) -> dict[str, Any]:
    """Snap a tool delta to the nearest value on the action grid."""
    if step <= 0:
        raise ValueError(f"Quantization step must be positive, got {step}")

    if tool_name == "hsl_tool":
        adjustments = []
        for item in delta_params.get("adjustments", []):
            quantized = {"color": item.get("color", "reds")}
            for key in HSL_NUMERIC_KEYS:
                quantized[key] = _snap_scalar(float(item.get(key, 0.0)), step)
            if any(quantized[key] != 0 for key in HSL_NUMERIC_KEYS):
                adjustments.append(quantized)
        return {"adjustments": adjustments}

    return {
        key: _snap_scalar(float(value), step)
        for key, value in delta_params.items()
    }


def make_quantized_delta_signature(
    tool_name: str,
    delta_params: dict[str, Any],
    *,
    step: float = DEFAULT_ACTION_STEP,
) -> str:
    """Return a stable signature for the snapped action."""
    quantized = quantize_tool_delta(tool_name, delta_params, step=step)
    if tool_name == "hsl_tool":
        canonical = {
            "adjustments": sorted(
                quantized.get("adjustments", []),
                key=lambda item: item.get("color", ""),
            )
        }
    else:
        canonical = {
            key: quantized[key]
            for key in sorted(quantized)
        }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _snap_scalar(value: float, step: float) -> int | float:
    if abs(value) <= 1e-12:
        return 0

    units = math.floor(abs(value) / step + 0.5)
    snapped = math.copysign(units * step, value)
    if abs(snapped) <= 1e-12:
        return 0

    rounded_int = round(snapped)
    if abs(snapped - rounded_int) <= 1e-12:
        return int(rounded_int)
    return float(snapped)
