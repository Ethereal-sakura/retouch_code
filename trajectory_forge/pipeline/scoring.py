"""Scoring helpers for accepted-state search."""

from __future__ import annotations

from typing import Any


DEFAULT_SCORE_CFG = {
    "mode": "hybrid",
    "weights": {
        "delta_e": 1.0,
        "l1": 24.0,
        "l2": 18.0,
        "lpips": 8.0,
        "ssim_error": 6.0,
        "stat_residual": 4.0,
        "edit_cost": 0.8,
        "repeat_penalty": 0.08,
        "sign_flip_penalty": 0.10,
    },
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


def compute_stat_residual(delta_stat: dict[str, Any] | None) -> float:
    """Aggregate normalized residual statistics into a single scalar."""
    if not delta_stat:
        return 0.0
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
    tool_name: str | None,
    delta_params: dict[str, Any],
    tool_state: dict[str, Any] | None,
    weights: dict[str, float],
) -> float:
    """Penalize repeated tools and immediate sign flips."""
    if not tool_name or not tool_state:
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
    delta_stat: dict[str, Any] | None = None,
    delta_params: dict[str, Any] | None = None,
    tool_name: str | None = None,
    tool_state: dict[str, Any] | None = None,
    scoring_cfg: dict[str, Any] | None = None,
) -> float:
    """Compute the search objective score. Lower is better."""
    cfg = {
        "mode": DEFAULT_SCORE_CFG["mode"],
        "weights": {**DEFAULT_SCORE_CFG["weights"]},
    }
    if scoring_cfg:
        cfg["mode"] = str(scoring_cfg.get("mode", cfg["mode"]))
        cfg["weights"].update(scoring_cfg.get("weights", {}))
    weights = cfg["weights"]
    mode = str(cfg["mode"]).lower()

    l1 = float(metrics.get("l1", 0.0))
    l2 = float(metrics.get("l2", 0.0))
    delta_e = float(metrics.get("delta_e", 0.0))
    lpips = float(metrics.get("lpips", 0.0))
    ssim_error = 1.0 - float(metrics.get("ssim", 1.0))
    stat_residual = compute_stat_residual(delta_stat)
    edit_cost = compute_edit_cost(delta_params or {})
    penalty = compute_trajectory_penalty(
        tool_name=tool_name,
        delta_params=delta_params or {},
        tool_state=tool_state,
        weights=weights,
    )

    if mode == "delta_e":
        return delta_e + penalty
    if mode == "l1":
        return weights["l1"] * l1 + penalty
    if mode == "l2":
        return weights["l2"] * l2 + penalty
    if mode == "pixel":
        return weights["l1"] * l1 + weights["l2"] * l2 + penalty

    return (
        weights["delta_e"] * delta_e
        + weights["l1"] * l1
        + weights["l2"] * l2
        + weights["lpips"] * lpips
        + weights["ssim_error"] * ssim_error
        + weights["stat_residual"] * stat_residual
        + weights["edit_cost"] * edit_cost
        + penalty
    )


def _has_sign_flip(previous: float, current: float) -> bool:
    previous = float(previous)
    current = float(current)
    return abs(previous) > 1e-6 and abs(current) > 1e-6 and previous * current < 0.0
