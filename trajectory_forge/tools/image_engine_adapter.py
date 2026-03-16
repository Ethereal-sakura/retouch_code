"""Image engine adapter: translates tool calls into BasicColorParams and renders images.

Key design: parameters are accumulated from the original image at each step to avoid
compression error buildup and correctly model parameter interactions.
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
from rapidraw_basic_color.params import HslSettings, HslBand, HSL_BAND_NAMES

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
    """Apply a tool call's parameters on top of the accumulated params.

    Parameters from the new tool call are added (not replaced) to the
    corresponding fields in accumulated. This allows the model to make
    incremental adjustments.

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
        new.exposure = float(tool_params.get("exposure", 0.0))
        new.brightness = float(tool_params.get("brightness", 0.0))

    elif tool_name == "tone_tool":
        new.contrast = float(tool_params.get("contrast", 0.0))
        new.highlights = float(tool_params.get("highlights", 0.0))
        new.shadows = float(tool_params.get("shadows", 0.0))
        new.whites = float(tool_params.get("whites", 0.0))
        new.blacks = float(tool_params.get("blacks", 0.0))

    elif tool_name == "white_balance_tool":
        new.temperature = float(tool_params.get("temperature", 0.0))
        new.tint = float(tool_params.get("tint", 0.0))

    elif tool_name == "saturation_tool":
        new.saturation = float(tool_params.get("saturation", 0.0))
        new.vibrance = float(tool_params.get("vibrance", 0.0))

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
                    "hue": float(adj.get("hue", 0.0)),
                    "saturation": float(adj.get("saturation", 0.0)),
                    "luminance": float(adj.get("luminance", 0.0)),
                }
        new.hsl = HslSettings.from_dict(hsl_dict)

    return new


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
