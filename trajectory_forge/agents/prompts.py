"""Prompt templates for the trajectory generation agent.

All model-facing prompts in this module are intentionally English-only.
"""

from __future__ import annotations

import json
from typing import Any

from ..tools.tool_registry import get_tool_schema_text


MCTS_PLANNER_SYSTEM_PROMPT = """\
You are a professional photo retouching planner for long-horizon search.

You are given:
- a CURRENT image
- a TARGET image
- residual statistics
- accepted edit history
- the full tool schema

Your job is to propose multiple diverse NEXT actions.
Each candidate must use exactly one tool and provide concrete parameters.
You are NOT restricted to any shortlist. Any tool in the schema may be used.
Do NOT assume a fixed tool order. Pick whichever unresolved issue is most promising now.

Important semantics:
- The renderer always re-renders from the original source image.
- The search tree compares many candidate edits and keeps only strong paths.
- Every numeric parameter you output is a DELTA relative to the current accumulated slider state.
- Omitted parameters mean no change for that field.
- After your delta is applied, the system re-renders from the ORIGINAL source image using the updated accumulated settings.
- The final accumulated slider state is clamped to the renderer's supported absolute limits.
- Think like a human editor nudging sliders over multiple turns, not like a one-shot absolute predictor.
- Prefer values that are easy for a vision-language model to imitate.
- Prefer integer-looking values when possible.
- If you use hsl_tool, edit exactly one color band in this step.

Available tools:
{tool_schema}

## Renderer Response Guide
- exposure: +16 ≈ about +1 stop brighter, -16 ≈ about -1 stop darker
- brightness: gentler than exposure; mainly shifts midtones with a filmic response
- contrast: ±10 subtle, ±20 moderate
- highlights / shadows: mostly affect bright / dark regions; ±10 subtle, ±20 moderate
- whites / blacks: endpoint controls; usually use smaller moves than highlights / shadows
- temperature: +25 noticeably warmer, -25 noticeably cooler
- tint: +20 mild magenta shift, -20 mild green shift
- saturation: +10 subtle, +25 moderate, +50 strong
- vibrance: gentler than saturation and affects dull colors more
- HSL hue: 1 unit is only a small shift (roughly 0.6 degrees after rendering), so hue changes should usually stay modest
- Prefer small or moderate deltas first, then refine based on the new rendered result

Return ONLY valid JSON with this schema:
{{
  "candidates": [
    {{
      "tool": "exposure_tool",
      "parameters": {{
        "exposure": -6,
        "brightness": -2
      }},
      "reason": "The image is still brighter than the target."
    }},
    {{
      "tool": "tone_tool",
      "parameters": {{
        "highlights": -8,
        "whites": -4
      }},
      "reason": "Highlights still clip relative to the target."
    }}
  ]
}}

Rules:
- "candidates" must contain 1 to {candidate_limit} items.
- Each candidate must contain exactly one tool call.
- "tool" must be a valid tool name from the schema.
- "parameters" must be a JSON object for that tool.
- Prefer diverse tools and step sizes across candidates.
- Openly explore different tool orders when plausible.
- Prefer natural editor-like increments instead of extreme jumps unless the mismatch is obviously large.
- Do not use Markdown fences.
"""


def build_planner_system_prompt(candidate_limit: int) -> str:
    """Build the MCTS planner system prompt with the current tool schema."""
    return MCTS_PLANNER_SYSTEM_PROMPT.format(
        tool_schema=get_tool_schema_text(),
        candidate_limit=int(candidate_limit),
    )


def _format_history(history: list[dict], max_items: int = 8) -> str:
    if not history:
        return "Accepted history: none yet."

    lines = ["Accepted history:"]
    for step in history[-max_items:]:
        metrics = step.get("step_quality", {})
        delta_params = step.get("delta_parameters", step.get("parameters", {}))
        param_text = _format_params(delta_params)
        lines.append(
            (
                f"- Round {step['round'] + 1}: {step['tool']} "
                f"(delta: {param_text}) "
                f"-> DeltaE {metrics.get('delta_e', 0.0):.2f}, "
                f"PSNR {metrics.get('psnr', 0.0):.2f}, "
                f"SSIM {metrics.get('ssim', 0.0):.3f}"
            )
        )
    return "\n".join(lines)


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return "{}"

    if "adjustments" in params:
        return json.dumps(params, ensure_ascii=True, separators=(",", ":"))

    pieces = []
    for key, value in params.items():
        try:
            pieces.append(f"{key}={float(value):+.2f}")
        except (TypeError, ValueError):
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


def _format_delta_stats(delta_stat: dict[str, Any]) -> str:
    band_text = ""
    band_residuals = delta_stat.get("hue_band_residuals", {})
    if band_residuals:
        top_items = sorted(
            band_residuals.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        band_text = (
            "\nTop HSL band residuals: "
            + ", ".join(f"{name}={value:.3f}" for name, value in top_items)
        )

    return (
        "Residual statistics (target minus current):\n"
        f"- brightness_delta: {delta_stat.get('brightness_delta', 0.0):+.2f}\n"
        f"- l_channel_delta: {delta_stat.get('l_channel_delta', 0.0):+.2f}\n"
        f"- contrast_delta: {delta_stat.get('contrast_delta', 0.0):+.2f}\n"
        f"- highlight_delta: {delta_stat.get('highlight_delta', 0.0):+.2f}\n"
        f"- shadow_delta: {delta_stat.get('shadow_delta', 0.0):+.2f}\n"
        f"- temperature_delta: {delta_stat.get('temperature_delta', 0.0):+.2f}\n"
        f"- tint_delta: {delta_stat.get('tint_delta', 0.0):+.2f}\n"
        f"- saturation_delta: {delta_stat.get('saturation_delta', 0.0):+.2f}\n"
        f"- dominant_issue: {delta_stat.get('dominant_issue', 'unknown')}"
        f"{band_text}"
    )


def build_planner_user_prompt(
    *,
    delta_stat: dict[str, Any],
    history: list[dict],
    turn: int,
    current_metrics: dict[str, Any],
) -> list[dict]:
    """Build the open-tool planner prompt used by MCTS."""
    text = (
        f"Turn {turn + 1}\n\n"
        f"{_format_delta_stats(delta_stat)}\n\n"
        "Current quality:\n"
        f"- DeltaE: {current_metrics.get('delta_e', 0.0):.2f}\n"
        f"- PSNR: {current_metrics.get('psnr', 0.0):.2f}\n"
        f"- SSIM: {current_metrics.get('ssim', 0.0):.3f}\n"
        f"- LPIPS: {current_metrics.get('lpips', 0.0):.3f}\n\n"
        f"{_format_history(history)}\n\n"
        "Return only JSON."
    )
    return [{"type": "text", "text": text}]


def build_image_content(
    current_img_b64: str,
    target_img_b64: str,
    media_type: str = "image/jpeg",
) -> list[dict]:
    """Build image content blocks for the OpenAI message."""
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{current_img_b64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{target_img_b64}"},
        },
    ]
