"""Prompt templates for the trajectory generation agent.

All model-facing prompts in this module are intentionally English-only.
"""

from __future__ import annotations

import json
from typing import Any

from ..tools.tool_registry import get_tool_schema_text


PLANNER_SYSTEM_PROMPT = """\
You are a professional photo retouching planner.

You are given:
- a CURRENT image
- a TARGET image
- quantitative residual statistics
- accepted edit history
- the subset of tools that are currently allowed

Your job is NOT to output exact final parameter values.
Your job is to choose the next edit direction:
- which tool to try now
- whether to increase or decrease the main effect
- a coarse magnitude bucket: small, medium, or large
- a short reason

Important semantics:
- The renderer always re-renders from the original source image.
- The editing parameters are accumulated totals built from per-step DELTAS.
- Think in terms of parameter deltas relative to the CURRENT image.
- Only choose from the allowed tools in the shortlist.
- Prefer higher-priority unresolved issues before lower-priority ones.

Available tools:
{tool_schema}

Return ONLY valid JSON with this schema:
{{
  "should_stop": false,
  "main_issue": "exposure",
  "proposals": [
    {{
      "tool": "exposure_tool",
      "direction": "decrease",
      "magnitude_bucket": "small",
      "reason": "The current image is still brighter than the target."
    }}
  ]
}}

Rules:
- "should_stop" must be true only when the current image is already close enough
  or no allowed tool is likely to improve the score.
- "proposals" must contain 1 to 3 items when "should_stop" is false.
- "tool" must be one of the allowed shortlist tools.
- "direction" must be one of: increase, decrease, mixed.
- "magnitude_bucket" must be one of: small, medium, large.
- Do not use Markdown fences.
"""


EXPLAIN_SYSTEM_PROMPT = """\
You are an expert photo retoucher writing short training rationales.

You will be given:
- the accepted editing step
- the current residual statistics before the step
- the accepted parameter delta
- the resulting improvement summary

Write a concise English explanation inside a <thinking>...</thinking> block.
The explanation must describe why this accepted step was the right next move.
Do not mention hidden search, probes, or rejected candidates.
"""


def build_planner_system_prompt() -> str:
    """Build the planner system prompt with the current tool schema."""
    return PLANNER_SYSTEM_PROMPT.format(tool_schema=get_tool_schema_text())


def build_explainer_system_prompt() -> str:
    """Return the system prompt used for accepted-step explanations."""
    return EXPLAIN_SYSTEM_PROMPT


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
    shortlist_tools: list[str],
    current_metrics: dict[str, Any],
    current_score: float,
    locked_tools: list[str],
    cooldown_tools: list[str],
) -> list[dict]:
    """Build the planner user message content."""
    text = (
        f"Turn {turn + 1}\n\n"
        f"{_format_delta_stats(delta_stat)}\n\n"
        "Current quality:\n"
        f"- DeltaE: {current_metrics.get('delta_e', 0.0):.2f}\n"
        f"- PSNR: {current_metrics.get('psnr', 0.0):.2f}\n"
        f"- SSIM: {current_metrics.get('ssim', 0.0):.3f}\n"
        f"- LPIPS: {current_metrics.get('lpips', 0.0):.3f}\n"
        f"- objective_score: {current_score:.4f}\n\n"
        f"Allowed shortlist tools: {', '.join(shortlist_tools) if shortlist_tools else 'none'}\n"
        f"Locked tools: {', '.join(locked_tools) if locked_tools else 'none'}\n"
        f"Cooldown tools: {', '.join(cooldown_tools) if cooldown_tools else 'none'}\n\n"
        f"{_format_history(history)}\n\n"
        "Return only JSON."
    )
    return [{"type": "text", "text": text}]


def build_explainer_user_prompt(
    *,
    step_record: dict[str, Any],
    delta_stat: dict[str, Any],
    score_before: float,
    score_after: float,
) -> list[dict]:
    """Build the explanation prompt for an accepted step."""
    metrics = step_record.get("step_quality", {})
    text = (
        f"Accepted step summary:\n"
        f"- tool: {step_record.get('tool')}\n"
        f"- delta_parameters: {_format_params(step_record.get('delta_parameters', {}))}\n"
        f"- accumulated_parameters: {_format_params(step_record.get('params_accumulated_tool', {}))}\n"
        f"- score_before: {score_before:.4f}\n"
        f"- score_after: {score_after:.4f}\n"
        f"- DeltaE_after: {metrics.get('delta_e', 0.0):.2f}\n"
        f"- PSNR_after: {metrics.get('psnr', 0.0):.2f}\n"
        f"- SSIM_after: {metrics.get('ssim', 0.0):.3f}\n\n"
        f"{_format_delta_stats(delta_stat)}\n\n"
        "Write a short rationale for why this accepted step was the best next move."
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
