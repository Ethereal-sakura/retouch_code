"""Prompt templates for the trajectory generation agent."""

from __future__ import annotations

from ..tools.tool_registry import get_tool_schema_text


SYSTEM_PROMPT = """\
You are an expert photo editor who retouches images step by step, like a professional photographer.

## Task
You will be shown a CURRENT image (your work in progress) and a TARGET image (the desired result).
Each turn, you select EXACTLY ONE tool and output its parameters to bring the current image closer to the target.

## Priority Order
You MUST follow this priority order and always fix the highest-priority unresolved issue first:
  1. exposure_tool   — Fix global over/underexposure first
  2. tone_tool       — Then shape contrast, highlights, and shadows
  3. white_balance_tool — Then correct color temperature and tint
  4. saturation_tool — Then adjust overall color intensity
  5. hsl_tool        — Finally, fine-tune individual color bands

## Available Tools
{tool_schema}

## Output Format
Always respond in this EXACT format:

<thinking>
[Carefully analyze remaining visual differences between current and target.
Identify the dominant issue based on the quantitative statistics provided.
Justify your tool choice and parameter values.]
</thinking>
<tool_call>
tool: {{tool_name}}
{{param_name}}: {{value}}
...
</tool_call>

For hsl_tool, output parameters as JSON:
<tool_call>
tool: hsl_tool
adjustments: [{{"color": "reds", "hue": 5, "saturation": 10, "luminance": 0}}]
</tool_call>

When the current image matches the target well enough (DeltaE < 4 or no significant remaining differences), output:
<thinking>
[Explain that the image has converged and no further significant improvements are needed.]
</thinking>
<stop>Retouching complete.</stop>

## Important Rules
- Select EXACTLY ONE tool per turn
- Do NOT repeat a tool you have already used unless there is strong justification
- Parameter values must be within the specified ranges
- Base your decisions on BOTH visual inspection AND the quantitative statistics
"""


def build_system_prompt() -> str:
    """Build the system prompt with the current tool schema."""
    return SYSTEM_PROMPT.format(tool_schema=get_tool_schema_text())


def build_user_prompt(
    delta_stat: dict,
    history: list[dict],
    turn: int,
) -> list[dict]:
    """Build the per-turn user message content (text part only).

    The caller is responsible for prepending image content blocks (current + target).

    Parameters
    ----------
    delta_stat : dict
        Output of get_delta_stat() for current vs target.
    history : list[dict]
        List of previous steps: [{round, tool, parameters}, ...].
    turn : int
        Current turn number (0-indexed).

    Returns
    -------
    list[dict]
        OpenAI content blocks (text only; images added by caller).
    """
    # Build stats text
    bd = delta_stat.get("brightness_delta", 0.0)
    cd = delta_stat.get("contrast_delta", 0.0)
    td = delta_stat.get("temperature_delta", 0.0)
    sd = delta_stat.get("saturation_delta", 0.0)
    dominant = delta_stat.get("dominant_issue", "unknown")

    brightness_dir = "brighter" if bd > 0 else "darker"
    contrast_dir = "more contrast" if cd > 0 else "less contrast"
    temperature_dir = "warmer" if td > 0 else "cooler"
    saturation_dir = "more saturated" if sd > 0 else "less saturated"

    stats_text = (
        f"Quantitative Statistics (Current vs Target):\n"
        f"- Brightness delta: {bd:+.1f} (target is {brightness_dir})\n"
        f"- Contrast delta: {cd:+.1f} (target has {contrast_dir})\n"
        f"- Temperature signal: {td:+.1f} (target is {temperature_dir})\n"
        f"- Saturation delta: {sd:+.1f} (target is {saturation_dir})\n"
        f"- Dominant issue: {dominant}\n"
    )

    # Build history text
    if history:
        history_lines = ["Adjustment history:"]
        for step in history:
            r = step["round"]
            t = step["tool"]
            p = step["parameters"]
            q = step.get("step_quality", {})
            de_str = f" → DeltaE={q.get('delta_e', '?'):.1f}" if q else ""
            # Format params concisely
            if t == "hsl_tool":
                param_str = f"adjustments={p.get('adjustments', [])}"
            else:
                param_str = ", ".join(f"{k}={v}" for k, v in p.items())
            history_lines.append(f"  Round {r + 1}: {t}({param_str}){de_str}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "Adjustment history: (none yet — this is the first step)"

    text = (
        f"Turn {turn + 1}:\n"
        f"[LEFT IMAGE = CURRENT] [RIGHT IMAGE = TARGET]\n\n"
        f"{stats_text}\n"
        f"{history_text}\n\n"
        "Analyze the remaining differences and apply the single most important adjustment "
        "following the priority order."
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
