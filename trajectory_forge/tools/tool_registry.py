"""Tool registry: schema definitions and parameter validation.

Each tool corresponds to a semantically coherent group of image_engine parameters,
following the photographer's adjustment priority order:
  exposure → tone → white_balance → saturation → hsl
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Valid HSL color band names (must match image_engine params.py HSL_BAND_NAMES)
HSL_BAND_NAMES = [
    "reds", "oranges", "yellows", "greens",
    "aquas", "blues", "purples", "magentas",
]

# Tool priority order (lower = higher priority)
TOOL_PRIORITY = {
    "exposure_tool": 1,
    "tone_tool": 2,
    "white_balance_tool": 3,
    "saturation_tool": 4,
    "hsl_tool": 5,
}

# Parameter ranges for each tool (used for validation and prompt generation)
TOOL_SCHEMAS: dict[str, dict] = {
    "exposure_tool": {
        "description": "Adjust global exposure and brightness.",
        "parameters": {
            "exposure":   {"type": "float", "range": (-100, 100)},
            "brightness": {"type": "float", "range": (-100, 100)},
        },
    },
    "tone_tool": {
        "description": "Shape the tonal curve (contrast, highlights, shadows, whites, blacks).",
        "parameters": {
            "contrast":   {"type": "float", "range": (-100, 100)},
            "highlights": {"type": "float", "range": (-120, 120)},
            "shadows":    {"type": "float", "range": (-120, 120)},
            "whites":     {"type": "float", "range": (-30, 30)},
            "blacks":     {"type": "float", "range": (-70, 70)},
        },
    },
    "white_balance_tool": {
        "description": "Correct color temperature (warm/cool) and tint (green/magenta).",
        "parameters": {
            "temperature": {"type": "float", "range": (-500, 500)},
            "tint":        {"type": "float", "range": (-100, 100)},
        },
    },
    "saturation_tool": {
        "description": "Adjust global color saturation and vibrance.",
        "parameters": {
            "saturation": {"type": "float", "range": (-100, 100)},
            "vibrance":   {"type": "float", "range": (-100, 100)},
        },
    },
    "hsl_tool": {
        "description": (
            "Fine-tune individual color bands. "
            "Accepts a list of adjustments: [{color, hue, saturation, luminance}]."
        ),
        "parameters": {
            "adjustments": {
                "type": "array",
                "items": {
                    "color":      {"type": "string", "enum": HSL_BAND_NAMES},
                    "hue":        {"type": "float", "range": (-100, 100)},
                    "saturation": {"type": "float", "range": (-100, 100)},
                    "luminance":  {"type": "float", "range": (-100, 100)},
                },
            }
        },
    },
}


def get_tool_schema_text() -> str:
    """Return a human-readable text description of all tools for the system prompt."""
    lines = []
    for tool_name, schema in TOOL_SCHEMAS.items():
        priority = TOOL_PRIORITY[tool_name]
        lines.append(f"[Priority {priority}] {tool_name}: {schema['description']}")
        for param_name, param in schema["parameters"].items():
            if param["type"] == "array":
                lines.append(
                    f"  - adjustments: list of {{color ∈ {HSL_BAND_NAMES}, "
                    f"hue [-100,100], saturation [-100,100], luminance [-100,100]}}"
                )
            else:
                lo, hi = param["range"]
                lines.append(f"  - {param_name}: float [{lo}, {hi}]")
    return "\n".join(lines)


def validate_tool_call(tool_name: str, params: dict) -> tuple[bool, str]:
    """Validate tool name and parameters against the schema.

    Returns (ok: bool, error_message: str).
    """
    if tool_name not in TOOL_SCHEMAS:
        return False, f"Unknown tool: '{tool_name}'. Valid tools: {list(TOOL_SCHEMAS.keys())}"

    schema = TOOL_SCHEMAS[tool_name]

    if tool_name == "hsl_tool":
        adjustments = params.get("adjustments")
        if not isinstance(adjustments, list) or len(adjustments) == 0:
            return False, "hsl_tool requires a non-empty 'adjustments' list."
        for adj in adjustments:
            color = adj.get("color", "")
            if color not in HSL_BAND_NAMES:
                return False, f"Invalid HSL color band: '{color}'. Valid: {HSL_BAND_NAMES}"
            for key in ("hue", "saturation", "luminance"):
                val = adj.get(key, 0.0)
                lo, hi = (-100, 100)
                if not (lo <= float(val) <= hi):
                    return False, f"HSL {key} value {val} out of range [{lo}, {hi}]"
        return True, ""

    for param_name, param_schema in schema["parameters"].items():
        if param_name not in params:
            continue  # Optional — default to 0
        val = params[param_name]
        lo, hi = param_schema["range"]
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False, f"Parameter '{param_name}' must be numeric, got: {val!r}"
        if not (lo <= fval <= hi):
            return False, f"Parameter '{param_name}' = {fval} out of range [{lo}, {hi}]"

    return True, ""


def clamp_params(tool_name: str, params: dict) -> dict:
    """Clamp all parameter values to their valid ranges."""
    schema = TOOL_SCHEMAS[tool_name]
    result = dict(params)

    if tool_name == "hsl_tool":
        clamped = []
        for adj in result.get("adjustments", []):
            clamped.append({
                "color":      adj.get("color", "reds"),
                "hue":        max(-100.0, min(100.0, float(adj.get("hue", 0.0)))),
                "saturation": max(-100.0, min(100.0, float(adj.get("saturation", 0.0)))),
                "luminance":  max(-100.0, min(100.0, float(adj.get("luminance", 0.0)))),
            })
        result["adjustments"] = clamped
        return result

    for param_name, param_schema in schema["parameters"].items():
        if param_name in result:
            lo, hi = param_schema["range"]
            result[param_name] = max(float(lo), min(float(hi), float(result[param_name])))
    return result
