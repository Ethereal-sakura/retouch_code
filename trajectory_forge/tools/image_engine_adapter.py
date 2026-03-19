"""Image engine adapter: translates tool calls into BasicColorParams and renders images.

Key design:
- each step proposes parameter deltas relative to the CURRENT image
- deltas are accumulated into a global parameter state
- rendering always replays the accumulated state from the original source image
"""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# Add image_engine to sys.path
_engine_path = Path(__file__).parent.parent.parent / "image_engine"
if str(_engine_path) not in sys.path:
    sys.path.insert(0, str(_engine_path))

from rapidraw_basic_color import BasicColorRenderer, BasicColorParams
from rapidraw_basic_color.params import HslSettings, HSL_BAND_NAMES

from .tool_registry import TOOL_SCHEMAS, clamp_params

_renderer = BasicColorRenderer()


def get_renderer() -> BasicColorRenderer:
    return _renderer


def make_default_params() -> BasicColorParams:
    """Return a default (identity) BasicColorParams."""
    return BasicColorParams()


def params_to_dict(params: BasicColorParams) -> dict:
    """Serialize BasicColorParams to a plain JSON-serializable dict."""
    d = asdict(params)
    return d


def merge_tool_call(
    accumulated: BasicColorParams,
    tool_name: str,
    tool_params: dict,
) -> BasicColorParams:
    """Accumulate a tool call's parameter deltas onto the existing params.

    Parameters
    ----------
    accumulated : BasicColorParams
        Current accumulated parameters.
    tool_name : str
        Name of the tool being applied.
    tool_params : dict
        Parameters from the tool call (pre-validated and clamped).

    Returns
    -------
    BasicColorParams
        New accumulated parameter object.
    """
    tool_params = clamp_params(tool_name, tool_params)
    new = copy.deepcopy(accumulated)

    if tool_name == "exposure_tool":
        new.exposure = _clamp_total("exposure_tool", "exposure", new.exposure + float(tool_params.get("exposure", 0.0)))
        new.brightness = _clamp_total(
            "exposure_tool", "brightness", new.brightness + float(tool_params.get("brightness", 0.0))
        )

    elif tool_name == "tone_tool":
        new.contrast = _clamp_total("tone_tool", "contrast", new.contrast + float(tool_params.get("contrast", 0.0)))
        new.highlights = _clamp_total(
            "tone_tool", "highlights", new.highlights + float(tool_params.get("highlights", 0.0))
        )
        new.shadows = _clamp_total("tone_tool", "shadows", new.shadows + float(tool_params.get("shadows", 0.0)))
        new.whites = _clamp_total("tone_tool", "whites", new.whites + float(tool_params.get("whites", 0.0)))
        new.blacks = _clamp_total("tone_tool", "blacks", new.blacks + float(tool_params.get("blacks", 0.0)))

    elif tool_name == "white_balance_tool":
        new.temperature = _clamp_total(
            "white_balance_tool", "temperature", new.temperature + float(tool_params.get("temperature", 0.0))
        )
        new.tint = _clamp_total("white_balance_tool", "tint", new.tint + float(tool_params.get("tint", 0.0)))

    elif tool_name == "saturation_tool":
        new.saturation = _clamp_total(
            "saturation_tool", "saturation", new.saturation + float(tool_params.get("saturation", 0.0))
        )
        new.vibrance = _clamp_total(
            "saturation_tool", "vibrance", new.vibrance + float(tool_params.get("vibrance", 0.0))
        )

    elif tool_name == "hsl_tool":
        # Update only the specified bands; leave others unchanged
        hsl_dict: dict[str, dict] = {}
        for band_name in HSL_BAND_NAMES:
            band = getattr(new.hsl, band_name)
            hsl_dict[band_name] = {
                "hue": band.hue,
                "saturation": band.saturation,
                "luminance": band.luminance,
            }
        for adj in tool_params.get("adjustments", []):
            color = adj["color"]
            if color in hsl_dict:
                hsl_dict[color] = {
                    "hue": max(-100.0, min(100.0, hsl_dict[color]["hue"] + float(adj.get("hue", 0.0)))),
                    "saturation": max(
                        -100.0,
                        min(100.0, hsl_dict[color]["saturation"] + float(adj.get("saturation", 0.0))),
                    ),
                    "luminance": max(
                        -100.0,
                        min(100.0, hsl_dict[color]["luminance"] + float(adj.get("luminance", 0.0))),
                    ),
                }
        new.hsl = HslSettings.from_dict(hsl_dict)

    return new


def get_tool_params(params: BasicColorParams, tool_name: str) -> dict[str, Any]:
    """Return the accumulated totals for a single tool."""
    if tool_name == "exposure_tool":
        return {
            "exposure": float(params.exposure),
            "brightness": float(params.brightness),
        }
    if tool_name == "tone_tool":
        return {
            "contrast": float(params.contrast),
            "highlights": float(params.highlights),
            "shadows": float(params.shadows),
            "whites": float(params.whites),
            "blacks": float(params.blacks),
        }
    if tool_name == "white_balance_tool":
        return {
            "temperature": float(params.temperature),
            "tint": float(params.tint),
        }
    if tool_name == "saturation_tool":
        return {
            "saturation": float(params.saturation),
            "vibrance": float(params.vibrance),
        }
    if tool_name == "hsl_tool":
        adjustments = []
        for band_name in HSL_BAND_NAMES:
            band = getattr(params.hsl, band_name)
            adjustments.append(
                {
                    "color": band_name,
                    "hue": float(band.hue),
                    "saturation": float(band.saturation),
                    "luminance": float(band.luminance),
                }
            )
        return {"adjustments": adjustments}
    raise ValueError(f"Unknown tool: {tool_name}")


def diff_tool_params(
    before: BasicColorParams,
    after: BasicColorParams,
    tool_name: str,
) -> dict[str, Any]:
    """Return the actual applied delta for a tool after clamping."""
    before_params = get_tool_params(before, tool_name)
    after_params = get_tool_params(after, tool_name)

    if tool_name == "hsl_tool":
        before_map = {item["color"]: item for item in before_params["adjustments"]}
        delta_items = []
        for item in after_params["adjustments"]:
            color = item["color"]
            prev = before_map[color]
            delta = {
                "color": color,
                "hue": _normalize_delta_number(item["hue"] - prev["hue"]),
                "saturation": _normalize_delta_number(item["saturation"] - prev["saturation"]),
                "luminance": _normalize_delta_number(item["luminance"] - prev["luminance"]),
            }
            if any(abs(delta[key]) > 1e-6 for key in ("hue", "saturation", "luminance")):
                delta_items.append(delta)
        return {"adjustments": delta_items}

    return {
        key: _normalize_delta_number(float(after_params.get(key, 0.0)) - float(before_params.get(key, 0.0)))
        for key in after_params
    }


def render(source_img: np.ndarray, params: BasicColorParams) -> np.ndarray:
    """Render source image with the given parameters.

    Always renders from the original source to avoid error accumulation.

    Parameters
    ----------
    source_img : np.ndarray
        Original source image, float32 [0, 1].
    params : BasicColorParams
        Accumulated rendering parameters.

    Returns
    -------
    np.ndarray
        Rendered image, float32 [0, 1].
    """
    result = _renderer.render_array(source_img, params)
    return result.image_srgb


def _clamp_total(tool_name: str, param_name: str, value: float) -> float:
    lo, hi = TOOL_SCHEMAS[tool_name]["parameters"][param_name]["range"]
    return max(float(lo), min(float(hi), float(value)))


def _normalize_delta_number(value: float) -> int | float:
    if abs(value) <= 1e-9:
        return 0
    rounded = round(value)
    if abs(value - rounded) <= 1e-9:
        return int(rounded)
    return float(value)
