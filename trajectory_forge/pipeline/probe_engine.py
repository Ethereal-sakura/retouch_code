"""Probe and local-refinement helpers for accepted-state search."""

from __future__ import annotations

import copy
from typing import Any

from ..tools.image_engine_adapter import (
    diff_tool_params,
    get_tool_params,
    merge_tool_call,
    params_to_dict,
    render,
)
from ..utils.metrics import compute_metrics
from ..utils.stat_utils import get_delta_stat
from .scoring import compute_objective_score


BASE_MAGNITUDES = {
    "exposure_tool": {"small": 4.0, "medium": 8.0, "large": 14.0},
    "tone_tool": {"small": 8.0, "medium": 16.0, "large": 28.0},
    "white_balance_tool": {"small": 4.0, "medium": 8.0, "large": 14.0},
    "saturation_tool": {"small": 6.0, "medium": 12.0, "large": 20.0},
    "hsl_tool": {"small": 6.0, "medium": 12.0, "large": 18.0},
}


def probe_and_refine(
    *,
    state: Any,
    proposal: dict[str, Any],
    delta_stat: dict[str, Any],
    src_probe_img,
    tgt_probe_img,
    src_full_img,
    tgt_full_img,
    metrics_size: tuple[int, int],
    use_lpips: bool,
    metrics_device: str,
    scoring_cfg: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Probe candidate deltas, refine locally, and evaluate the best candidate."""
    ladder = build_probe_ladder(
        tool_name=proposal["tool"],
        direction=proposal.get("direction", "mixed"),
        magnitude_bucket=proposal.get("magnitude_bucket", "medium"),
        delta_stat=delta_stat,
    )
    if not ladder:
        return None

    probe_results = []
    for delta_params in ladder:
        result = evaluate_candidate(
            state=state,
            tool_name=proposal["tool"],
            delta_params=delta_params,
            src_img=src_probe_img,
            tgt_img=tgt_probe_img,
            metrics_size=metrics_size,
            use_lpips=False,
            metrics_device=metrics_device,
            scoring_cfg=scoring_cfg,
        )
        probe_results.append(result)

    probe_results.sort(key=lambda item: item["score"])
    best_probe = probe_results[0]
    refined = refine_candidate(
        state=state,
        tool_name=proposal["tool"],
        seed_delta=best_probe["delta_parameters"],
        seed_score=best_probe["score"],
        delta_stat=delta_stat,
        src_probe_img=src_probe_img,
        tgt_probe_img=tgt_probe_img,
        metrics_size=metrics_size,
        metrics_device=metrics_device,
        scoring_cfg=scoring_cfg,
        step_sizes=probe_cfg.get("refine_steps", [3.0, 1.5]),
    )

    final_delta = refined["delta_parameters"]
    full_eval = evaluate_candidate(
        state=state,
        tool_name=proposal["tool"],
        delta_params=final_delta,
        src_img=src_full_img,
        tgt_img=tgt_full_img,
        metrics_size=metrics_size,
        use_lpips=use_lpips,
        metrics_device=metrics_device,
        scoring_cfg=scoring_cfg,
    )

    full_eval["proposal"] = copy.deepcopy(proposal)
    full_eval["probe_summary"] = [
        {
            "delta_parameters": item["delta_parameters"],
            "score": round(float(item["score"]), 4),
            "metrics": item["metrics"],
        }
        for item in probe_results[:3]
    ]
    return full_eval


def evaluate_candidate(
    *,
    state: Any,
    tool_name: str,
    delta_params: dict[str, Any],
    src_img,
    tgt_img,
    metrics_size: tuple[int, int],
    use_lpips: bool,
    metrics_device: str,
    scoring_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Render and score a candidate delta."""
    new_params = merge_tool_call(state.params, tool_name, delta_params)
    actual_delta = diff_tool_params(state.params, new_params, tool_name)
    rendered = render(src_img, new_params)
    metrics = compute_metrics(
        rendered,
        tgt_img,
        size=metrics_size,
        use_lpips=use_lpips,
        device=metrics_device,
    )
    new_delta_stat = get_delta_stat(rendered, tgt_img)
    score = compute_objective_score(
        metrics=metrics,
        delta_stat=new_delta_stat,
        delta_params=actual_delta,
        tool_name=tool_name,
        tool_state=state.tool_status,
        weights=scoring_cfg.get("weights", {}),
    )
    return {
        "tool": tool_name,
        "delta_parameters": actual_delta,
        "full_params": new_params,
        "metrics": metrics,
        "step_quality": metrics,
        "score": score,
        "output_image": rendered,
        "output_delta_stat": new_delta_stat,
        "params_accumulated": params_to_dict(new_params),
        "params_accumulated_tool": get_tool_params(new_params, tool_name),
    }


def refine_candidate(
    *,
    state: Any,
    tool_name: str,
    seed_delta: dict[str, Any],
    seed_score: float,
    delta_stat: dict[str, Any],
    src_probe_img,
    tgt_probe_img,
    metrics_size: tuple[int, int],
    metrics_device: str,
    scoring_cfg: dict[str, Any],
    step_sizes: list[float],
) -> dict[str, Any]:
    """Run a small local search around the best probe."""
    best = {
        "delta_parameters": copy.deepcopy(seed_delta),
        "score": float(seed_score),
    }

    for step_size in step_sizes:
        improved = True
        while improved:
            improved = False
            for neighbor in generate_neighbors(tool_name, best["delta_parameters"], step_size):
                result = evaluate_candidate(
                    state=state,
                    tool_name=tool_name,
                    delta_params=neighbor,
                    src_img=src_probe_img,
                    tgt_img=tgt_probe_img,
                    metrics_size=metrics_size,
                    use_lpips=False,
                    metrics_device=metrics_device,
                    scoring_cfg=scoring_cfg,
                )
                if result["score"] + 1e-8 < best["score"]:
                    best = {
                        "delta_parameters": result["delta_parameters"],
                        "score": result["score"],
                    }
                    improved = True
    return best


def build_probe_ladder(
    *,
    tool_name: str,
    direction: str,
    magnitude_bucket: str,
    delta_stat: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a ladder of delta candidates for a proposal."""
    magnitude = BASE_MAGNITUDES[tool_name][magnitude_bucket]

    if tool_name == "exposure_tool":
        sign = 1.0 if direction == "increase" else -1.0
        return [
            {"exposure": sign * magnitude * 0.6, "brightness": sign * magnitude * 0.2},
            {"exposure": sign * magnitude * 0.8, "brightness": sign * magnitude * 0.4},
            {"exposure": sign * magnitude, "brightness": sign * magnitude * 0.6},
        ]

    if tool_name == "tone_tool":
        contrast_sign = _sign(delta_stat.get("contrast_delta", 0.0))
        highlight_sign = _sign(delta_stat.get("highlight_delta", 0.0))
        shadow_sign = _sign(delta_stat.get("shadow_delta", 0.0))
        return [
            {"contrast": contrast_sign * magnitude},
            {"highlights": highlight_sign * magnitude, "whites": highlight_sign * magnitude * 0.35},
            {"shadows": shadow_sign * magnitude, "blacks": shadow_sign * magnitude * 0.4},
            {
                "contrast": contrast_sign * magnitude * 0.7,
                "highlights": highlight_sign * magnitude * 0.8,
                "shadows": shadow_sign * magnitude * 0.8,
                "whites": highlight_sign * magnitude * 0.25,
                "blacks": shadow_sign * magnitude * 0.3,
            },
        ]

    if tool_name == "white_balance_tool":
        temp_sign = _sign(delta_stat.get("temperature_delta", 0.0))
        tint_sign = _sign(delta_stat.get("tint_delta", 0.0))
        return [
            {"temperature": temp_sign * magnitude, "tint": tint_sign * magnitude * 0.35},
            {"temperature": temp_sign * magnitude * 0.7, "tint": tint_sign * magnitude * 0.7},
            {"temperature": temp_sign * magnitude * 0.4, "tint": tint_sign * magnitude},
        ]

    if tool_name == "saturation_tool":
        sign = 1.0 if direction == "increase" else -1.0
        return [
            {"saturation": sign * magnitude, "vibrance": sign * magnitude * 0.6},
            {"saturation": sign * magnitude * 0.5, "vibrance": sign * magnitude},
            {"saturation": sign * magnitude * 0.85, "vibrance": sign * magnitude * 0.85},
        ]

    if tool_name == "hsl_tool":
        band = delta_stat.get("dominant_hsl_band", "reds")
        sat_sign = _sign(delta_stat.get("saturation_delta", 0.0))
        lum_sign = _sign(delta_stat.get("brightness_delta", 0.0))
        return [
            {
                "adjustments": [
                    {
                        "color": band,
                        "hue": 0.0,
                        "saturation": sat_sign * magnitude,
                        "luminance": lum_sign * magnitude * 0.35,
                    }
                ]
            },
            {
                "adjustments": [
                    {
                        "color": band,
                        "hue": 0.0,
                        "saturation": sat_sign * magnitude * 0.7,
                        "luminance": lum_sign * magnitude * 0.7,
                    }
                ]
            },
        ]

    return []


def generate_neighbors(
    tool_name: str,
    delta_params: dict[str, Any],
    step_size: float,
) -> list[dict[str, Any]]:
    """Generate local neighbors around a candidate delta."""
    neighbors = []
    if tool_name == "hsl_tool":
        adjustments = delta_params.get("adjustments", [])
        for idx, item in enumerate(adjustments):
            for key in ("saturation", "luminance", "hue"):
                for sign in (-1.0, 1.0):
                    new_item = copy.deepcopy(item)
                    original_value = float(new_item.get(key, 0.0))
                    candidate_value = original_value + sign * step_size
                    if _crosses_zero(original_value, candidate_value):
                        continue
                    new_item[key] = candidate_value
                    new_params = copy.deepcopy(delta_params)
                    new_params["adjustments"][idx] = new_item
                    neighbors.append(new_params)
        return neighbors

    for key, value in delta_params.items():
        for sign in (-1.0, 1.0):
            new_params = copy.deepcopy(delta_params)
            original_value = float(value)
            candidate_value = original_value + sign * step_size
            if _crosses_zero(original_value, candidate_value):
                continue
            new_params[key] = candidate_value
            neighbors.append(new_params)
    return neighbors


def _sign(value: float) -> float:
    return 1.0 if float(value) >= 0.0 else -1.0


def _crosses_zero(original_value: float, candidate_value: float) -> bool:
    if abs(original_value) <= 1e-6:
        return False
    return original_value * candidate_value < 0.0
