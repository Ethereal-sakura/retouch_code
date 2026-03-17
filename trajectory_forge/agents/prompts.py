"""Prompt templates for the trajectory generation agent.

Provides three prompt families:
1. Single-candidate system prompt (original, kept for backward compatibility)
2. Multi-candidate system prompt (asks model to output N diverse strategies)
3. VLM Judge system prompt (selects the best candidate by visual comparison)
"""

from __future__ import annotations

from ..tools.tool_registry import get_tool_schema_text


# ── Original single-candidate system prompt (backward compatible) ─────────────

SYSTEM_PROMPT = """\
You are an expert photo editor who retouches images step by step, like a professional photographer.

## Task
You will be shown a CURRENT image (your work in progress) and a TARGET image (the desired result).
Each turn, you select EXACTLY ONE tool and output parameter DELTAS to bring the current image closer to the target.

## Parameter Semantics
- Every numeric parameter you output is an INCREMENT relative to the current accumulated slider state.
- Omitted parameters mean "no change" for that field.
- After your delta is applied, the system re-renders from the ORIGINAL source image using the updated accumulated settings.
- The final accumulated slider state is clamped to the renderer's absolute supported limits.
- Think like a human editor nudging sliders over multiple turns, not like a one-shot absolute parameter predictor.

## Priority Order
You MUST follow this priority order and always fix the highest-priority unresolved issue first:
  1. exposure_tool   — Fix global over/underexposure first
  2. tone_tool       — Then shape contrast, highlights, and shadows
  3. white_balance_tool — Then correct color temperature and tint
  4. saturation_tool — Then adjust overall color intensity
  5. hsl_tool        — Finally, fine-tune individual color bands

## Available Tools
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
- Repeating the same tool across multiple turns is allowed when refinement is needed
- Parameter values must respect the hard limits; recommended single-step ranges are guidance for natural editing
- Base your decisions on BOTH visual inspection AND the quantitative statistics
"""


# ── Multi-candidate system prompt ─────────────────────────────────────────────

MULTI_CANDIDATE_SYSTEM_PROMPT = """\
You are an expert photo editor who retouches images step by step, like a professional photographer.

## Task
You will be shown a CURRENT image (your work in progress) and a TARGET image (the desired result).
Each turn, you propose {num_candidates} DIVERSE candidate editing strategies.
The system will render all candidates and a visual judge will select the best one.

## Parameter Semantics
- Every numeric parameter you output is an INCREMENT relative to the current accumulated slider state.
- Omitted parameters mean "no change" for that field.
- After your delta is applied, the system re-renders from the ORIGINAL source image using the updated accumulated settings.
- The final accumulated slider state is clamped to the renderer's absolute supported limits.

## Priority Order (guidance — candidates may explore different priorities)
  1. exposure_tool   — Fix global over/underexposure first
  2. tone_tool       — Then shape contrast, highlights, and shadows
  3. white_balance_tool — Then correct color temperature and tint
  4. saturation_tool — Then adjust overall color intensity
  5. hsl_tool        — Finally, fine-tune individual color bands

## Available Tools
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
- HSL hue: 1 unit is only a small shift (roughly 0.6 degrees after rendering)

## Output Format
Output exactly {num_candidates} candidates wrapped in <candidate_N> tags.
Each candidate MUST contain its own <thinking> and <tool_call> block.

<candidate_1>
<thinking>
[Analyze the dominant visual difference and explain this editing strategy.]
</thinking>
<tool_call>
tool: {{tool_name}}
{{param_name}}: {{value}}
</tool_call>
</candidate_1>

<candidate_2>
<thinking>
[A DIFFERENT strategy — vary the tool, direction, or magnitude.]
</thinking>
<tool_call>
tool: {{tool_name}}
{{param_name}}: {{value}}
</tool_call>
</candidate_2>

<candidate_3>
<thinking>
[Yet another distinct approach.]
</thinking>
<tool_call>
tool: {{tool_name}}
{{param_name}}: {{value}}
</tool_call>
</candidate_3>

For hsl_tool, output parameters as JSON inside <tool_call>:
<tool_call>
tool: hsl_tool
adjustments: [{{"color": "reds", "hue": 5, "saturation": 10, "luminance": 0}}]
</tool_call>

## Diversity Requirements
- Candidates MUST differ meaningfully from each other
- Vary across these dimensions:
  * Different tools (e.g., one candidate adjusts exposure, another adjusts tone)
  * Different parameter directions (e.g., brightness +12 vs brightness -5)
  * Different magnitudes (e.g., a conservative nudge vs a bolder correction)
- At least one candidate should address the dominant issue from the statistics
- Avoid repeating the exact same tool with only trivially different values

## Important Rules
- Each candidate selects EXACTLY ONE tool
- Parameter values must respect the hard limits; recommended ranges are guidance
- Base your decisions on BOTH visual inspection AND the quantitative statistics
"""


# ── VLM Judge system prompt ──────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert image quality judge for photo retouching.

## Task
You will be shown multiple images in this order:
- Image 0: the CURRENT image (the status quo before this editing round)
- Images 1 to {num_candidates}: candidate results from different editing strategies
- Last image: the TARGET image (the desired final result)

## Instructions
Compare each candidate (Images 1 to {num_candidates}) against the TARGET image.
Also compare them against Image 0 (current state) to assess improvement.

Evaluate these aspects in order of importance:
1. Overall brightness and exposure accuracy
2. Contrast and tonal range fidelity
3. Color temperature and white balance correctness
4. Saturation and color vibrancy match
5. Fine detail and texture preservation

## Decision Rules
- Select the candidate that is CLOSEST to the TARGET image overall
- If NONE of the candidates is closer to the TARGET than Image 0, select 0 (keep current state)
- When candidates are very similar, prefer the one with more natural appearance

## Output Format
Output ONLY your selection:

<choice>X</choice>

where X is the image number (0 = keep current, 1 to {num_candidates} = adopt that candidate).
"""


# ── Builder functions ─────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """Build the original single-candidate system prompt."""
    return SYSTEM_PROMPT.format(tool_schema=get_tool_schema_text())


def build_multi_candidate_system_prompt(num_candidates: int = 3) -> str:
    """Build the multi-candidate system prompt."""
    return MULTI_CANDIDATE_SYSTEM_PROMPT.format(
        num_candidates=num_candidates,
        tool_schema=get_tool_schema_text(),
    )


def build_judge_system_prompt(num_candidates: int = 3) -> str:
    """Build the VLM judge system prompt."""
    return JUDGE_SYSTEM_PROMPT.format(num_candidates=num_candidates)


def build_user_prompt(
    delta_stat: dict,
    history: list[dict],
    current_params: dict,
    turn: int,
    *,
    locked_tools: set[str] | None = None,
    prev_delta_e: float | None = None,
    current_delta_e: float | None = None,
    was_rollback: bool = False,
    num_candidates: int = 1,
) -> list[dict]:
    """Build the per-turn user message content (text part only).

    The caller is responsible for prepending image content blocks (current + target).

    Parameters
    ----------
    delta_stat : dict
        Output of get_delta_stat() for current vs target.
    history : list[dict]
        List of previous steps: [{round, tool, parameters}, ...].
    current_params : dict
        Current accumulated absolute slider state before this turn.
    turn : int
        Current turn number (0-indexed).
    locked_tools : set[str], optional
        Tools locked due to oscillation detection.
    prev_delta_e : float, optional
        DeltaE at start of the previous turn (for quality trend display).
    current_delta_e : float, optional
        DeltaE at start of this turn.
    was_rollback : bool
        Whether the previous turn was rolled back.
    num_candidates : int
        Number of candidate strategies to request (1 = original behavior).

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

    brightness_dir = _direction_phrase(bd, "brighter", "darker", "similar brightness")
    contrast_dir = _direction_phrase(cd, "more contrast", "less contrast", "similar contrast")
    temperature_dir = _direction_phrase(td, "warmer", "cooler", "similar white balance")
    saturation_dir = _direction_phrase(sd, "more saturated", "less saturated", "similar saturation")

    stats_text = (
        f"Quantitative Statistics (Current vs Target):\n"
        f"- Brightness delta: {bd:+.1f} (target is {brightness_dir})\n"
        f"- Contrast delta: {cd:+.1f} (target has {contrast_dir})\n"
        f"- Temperature signal: {td:+.1f} (target is {temperature_dir})\n"
        f"- Saturation delta: {sd:+.1f} (target is {saturation_dir})\n"
        f"- Dominant issue: {dominant}\n"
        "Note: these are image-difference statistics, not direct slider values.\n"
    )

    # Quality trend feedback
    quality_text = ""
    if prev_delta_e is not None and current_delta_e is not None:
        diff = prev_delta_e - current_delta_e
        if diff > 0.1:
            quality_text = (
                f"Quality trend: DeltaE improved from {prev_delta_e:.1f} to "
                f"{current_delta_e:.1f} (decreased by {diff:.1f}). Keep refining.\n"
            )
        elif diff < -0.1:
            quality_text = (
                f"Quality trend: DeltaE worsened from {prev_delta_e:.1f} to "
                f"{current_delta_e:.1f} (increased by {-diff:.1f}). "
                "The last adjustment was suboptimal.\n"
            )

    # Rollback warning
    rollback_text = ""
    if was_rollback:
        rollback_text = (
            "WARNING: The previous round was ROLLED BACK because no candidate "
            "improved the image. Try a fundamentally different approach — "
            "different tool, different direction, or smaller magnitude.\n"
        )

    # Locked tools warning
    locked_text = ""
    if locked_tools:
        locked_list = ", ".join(sorted(locked_tools))
        locked_text = (
            f"LOCKED TOOLS (do NOT use in any candidate): {locked_list}\n"
            "These tools were locked due to parameter oscillation. "
            "Focus on other tools.\n"
        )

    # Build history text
    if history:
        history_lines = ["Adjustment history:"]
        for step in history:
            r = step["round"]
            t = step["tool"]
            p = step["parameters"]
            q = step.get("step_quality", {})
            de_str = f" -> DeltaE={q['delta_e']:.1f}" if q and "delta_e" in q else ""
            if t == "hsl_tool":
                param_str = _format_hsl_adjustments(p.get("adjustments", []), prefix="d")
            else:
                param_str = _format_scalar_params(p, prefix="d")
            history_lines.append(f"  Round {r + 1}: {t}({param_str}){de_str}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "Adjustment history: (none yet — this is the first step)"

    current_params_text = _format_current_params(current_params)

    # Final instruction
    if num_candidates > 1:
        instruction = (
            f"Analyze the remaining differences and propose {num_candidates} "
            "diverse candidate strategies. Each candidate should use exactly one "
            "tool with parameter deltas. Ensure meaningful diversity between candidates."
        )
    else:
        instruction = (
            "Analyze the remaining differences and apply the single most important "
            "adjustment following the priority order. Output parameter deltas for "
            "exactly one tool."
        )

    text = (
        f"Turn {turn + 1}:\n"
        f"[LEFT IMAGE = CURRENT] [RIGHT IMAGE = TARGET]\n\n"
        f"{stats_text}\n"
        f"{quality_text}"
        f"{rollback_text}"
        f"{locked_text}"
        f"{current_params_text}\n\n"
        f"{history_text}\n\n"
        f"{instruction}"
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


def build_judge_content(
    current_b64: str,
    candidate_b64s: list[str],
    target_b64: str,
    media_type: str = "image/jpeg",
) -> list[dict]:
    """Build image + text content blocks for the VLM judge message.

    Image order: current (0) -> candidates (1..N) -> target (last).
    """
    content: list[dict] = []

    content.append({"type": "text", "text": "Image 0 (CURRENT — keep if no candidate improves):"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{current_b64}"},
    })

    for i, b64 in enumerate(candidate_b64s, 1):
        content.append({"type": "text", "text": f"Image {i} (Candidate {i}):"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        })

    content.append({"type": "text", "text": "TARGET image (the desired result):"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{target_b64}"},
    })

    content.append({
        "type": "text",
        "text": "Select the candidate closest to the TARGET, or 0 to keep current.",
    })

    return content


# ── Helper functions ──────────────────────────────────────────────────────────

def _direction_phrase(value: float, positive: str, negative: str, neutral: str) -> str:
    """Map the sign of a statistic to a human-readable phrase."""
    if value > 1e-6:
        return positive
    if value < -1e-6:
        return negative
    return neutral


def _fmt_number(value: float | int) -> str:
    """Format a scalar value with a sign for prompt readability."""
    return f"{float(value):+.1f}"


def _format_scalar_params(params: dict, prefix: str = "") -> str:
    """Format a flat scalar parameter dict."""
    if not params:
        return "(no change)"
    return ", ".join(f"{prefix}{k}={_fmt_number(v)}" for k, v in params.items())


def _format_hsl_adjustments(adjustments: list[dict], prefix: str = "") -> str:
    """Format HSL adjustment list compactly."""
    if not adjustments:
        return "(no change)"
    parts = []
    for adj in adjustments:
        color = adj.get("color", "unknown")
        parts.append(
            f"{color}(h={prefix}{_fmt_number(adj.get('hue', 0.0))}, "
            f"s={prefix}{_fmt_number(adj.get('saturation', 0.0))}, "
            f"l={prefix}{_fmt_number(adj.get('luminance', 0.0))})"
        )
    return "; ".join(parts)


def _format_current_params(current_params: dict) -> str:
    """Render the current accumulated absolute slider state for the model."""
    hsl = current_params.get("hsl", {})
    active_hsl = []
    for band_name, band in hsl.items():
        hue = float(band.get("hue", 0.0))
        sat = float(band.get("saturation", 0.0))
        lum = float(band.get("luminance", 0.0))
        if any(abs(v) > 1e-6 for v in (hue, sat, lum)):
            active_hsl.append(
                f"{band_name}(h={_fmt_number(hue)}, s={_fmt_number(sat)}, l={_fmt_number(lum)})"
            )

    hsl_text = "; ".join(active_hsl) if active_hsl else "(all zero)"

    return (
        "Current accumulated settings (absolute slider positions before this turn):\n"
        f"- exposure_tool: exposure={_fmt_number(current_params.get('exposure', 0.0))}, "
        f"brightness={_fmt_number(current_params.get('brightness', 0.0))}\n"
        f"- tone_tool: contrast={_fmt_number(current_params.get('contrast', 0.0))}, "
        f"highlights={_fmt_number(current_params.get('highlights', 0.0))}, "
        f"shadows={_fmt_number(current_params.get('shadows', 0.0))}, "
        f"whites={_fmt_number(current_params.get('whites', 0.0))}, "
        f"blacks={_fmt_number(current_params.get('blacks', 0.0))}\n"
        f"- white_balance_tool: temperature={_fmt_number(current_params.get('temperature', 0.0))}, "
        f"tint={_fmt_number(current_params.get('tint', 0.0))}\n"
        f"- saturation_tool: saturation={_fmt_number(current_params.get('saturation', 0.0))}, "
        f"vibrance={_fmt_number(current_params.get('vibrance', 0.0))}\n"
        f"- hsl_tool: {hsl_text}"
    )
