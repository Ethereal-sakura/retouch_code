"""Scoring helpers for accepted-state search."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


DEFAULT_SCORE_WEIGHTS = {
    "delta_e": 1.0,
    "lpips": 12.0,
    "ssim_error": 8.0,
    "stat_residual": 8.0,
    "edit_cost": 2.0,
    "repeat_penalty": 0.08,
    "sign_flip_penalty": 0.10,
}


def compute_tool_residuals(delta_stat: dict[str, Any]) -> dict[str, float]:
    """Map residual statistics to normalized tool-level residual magnitudes."""
    hsl_residuals = delta_stat.get("hue_band_residuals", {})
    return {
        "exposure_tool": max(
            abs(delta_stat.get("brightness_delta", 0.0)),
            abs(delta_stat.get("l_channel_delta", 0.0)),
        ) / 255.0,
        "tone_tool": max(
            abs(delta_stat.get("contrast_delta", 0.0)),
            abs(delta_stat.get("highlight_delta", 0.0)),
            abs(delta_stat.get("shadow_delta", 0.0)),
        ) / 255.0,
        "white_balance_tool": max(
            abs(delta_stat.get("temperature_delta", 0.0)),
            abs(delta_stat.get("tint_delta", 0.0)),
        ) / 255.0,
        "saturation_tool": abs(delta_stat.get("saturation_delta", 0.0)) / 255.0,
        "hsl_tool": max(hsl_residuals.values()) if hsl_residuals else 0.0,
    }


def compute_stat_residual(delta_stat: dict[str, Any]) -> float:
    """Aggregate normalized residual statistics into a single scalar."""
    residuals = compute_tool_residuals(delta_stat)
    return sum(residuals.values()) / max(len(residuals), 1)


def compute_edit_cost(delta_params: dict[str, Any]) -> float:
    """Compute a normalized edit cost for a proposed parameter delta."""
    if not delta_params:
        return 0.0

    if "adjustments" in delta_params:
        items = delta_params.get("adjustments", [])
        if not items:
            return 0.0
        cost = 0.0
        for item in items:
            cost += abs(float(item.get("hue", 0.0))) / 100.0
            cost += abs(float(item.get("saturation", 0.0))) / 100.0
            cost += abs(float(item.get("luminance", 0.0))) / 100.0
        return cost / max(len(items), 1)

    values = [abs(float(value)) / 100.0 for value in delta_params.values()]
    return sum(values) / max(len(values), 1)


def compute_trajectory_penalty(
    *,
    tool_name: str,
    delta_params: dict[str, Any],
    tool_state: dict[str, Any] | None,
    weights: dict[str, float],
) -> float:
    """Penalize repeated tools and immediate sign flips."""
    if not tool_state:
        return 0.0

    state = tool_state.get(tool_name, {})
    penalty = 0.0
    accepted_streak = int(state.get("accepted_streak", 0))
    if accepted_streak > 1:
        penalty += weights["repeat_penalty"] * float(accepted_streak - 1)

    previous_delta = state.get("last_delta", {})
    if not previous_delta:
        return penalty

    if "adjustments" in delta_params:
        prev_map = {item["color"]: item for item in previous_delta.get("adjustments", [])}
        for item in delta_params.get("adjustments", []):
            prev = prev_map.get(item.get("color"))
            if not prev:
                continue
            if _has_sign_flip(prev.get("saturation", 0.0), item.get("saturation", 0.0)):
                penalty += weights["sign_flip_penalty"]
            if _has_sign_flip(prev.get("luminance", 0.0), item.get("luminance", 0.0)):
                penalty += weights["sign_flip_penalty"]
        return penalty

    for key, value in delta_params.items():
        if _has_sign_flip(previous_delta.get(key, 0.0), value):
            penalty += weights["sign_flip_penalty"]
    return penalty


def compute_objective_score(
    *,
    metrics: dict[str, Any],
    delta_stat: dict[str, Any],
    delta_params: dict[str, Any] | None = None,
    tool_name: str | None = None,
    tool_state: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the search objective score. Lower is better."""
    weights = {**DEFAULT_SCORE_WEIGHTS, **(weights or {})}
    stat_residual = compute_stat_residual(delta_stat)
    edit_cost = compute_edit_cost(delta_params or {})
    penalty = 0.0
    if tool_name:
        penalty = compute_trajectory_penalty(
            tool_name=tool_name,
            delta_params=delta_params or {},
            tool_state=tool_state,
            weights=weights,
        )

    return (
        weights["delta_e"] * float(metrics.get("delta_e", 0.0))
        + weights["lpips"] * float(metrics.get("lpips", 0.0))
        + weights["ssim_error"] * (1.0 - float(metrics.get("ssim", 1.0)))
        + weights["stat_residual"] * stat_residual
        + weights["edit_cost"] * edit_cost
        + penalty
    )


def should_accept_candidate(
    *,
    current_score: float,
    candidate_score: float,
    current_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    accept_margin: float,
    hard_delta_e_tolerance: float,
) -> tuple[bool, str]:
    """Return whether a candidate should become the next accepted state."""
    current_delta_e = float(current_metrics.get("delta_e", 0.0))
    candidate_delta_e = float(candidate_metrics.get("delta_e", 0.0))

    if candidate_delta_e > current_delta_e + hard_delta_e_tolerance:
        return False, "delta_e_regression"

    if candidate_score > current_score - accept_margin:
        return False, "insufficient_gain"

    return True, "accepted"


def rank_states_diverse(states: list[Any], beam_size: int) -> list[Any]:
    """Rank candidate states by score while keeping tool diversity when possible."""
    ranked = sorted(states, key=lambda state: (state.score, len(state.steps)))
    if len(ranked) <= beam_size:
        return ranked

    selected = []
    used_last_tools: set[str] = set()

    for state in ranked:
        if len(selected) >= beam_size:
            break
        last_tool = state.steps[-1]["tool"] if state.steps else ""
        if last_tool and last_tool in used_last_tools:
            continue
        selected.append(state)
        if last_tool:
            used_last_tools.add(last_tool)

    for state in ranked:
        if len(selected) >= beam_size:
            break
        if state not in selected:
            selected.append(state)

    return selected[:beam_size]


def _has_sign_flip(previous: float, current: float) -> bool:
    previous = float(previous)
    current = float(current)
    return abs(previous) > 1e-6 and abs(current) > 1e-6 and previous * current < 0.0
