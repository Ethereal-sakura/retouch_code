"""Candidate planning utilities for the hidden-search trajectory generator."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..agents.mllm_agent import parse_planner_response
from ..agents.prompts import (
    build_image_content,
    build_planner_system_prompt,
    build_planner_user_prompt,
)
from ..tools.tool_registry import TOOL_PRIORITY

logger = logging.getLogger(__name__)

DEFAULT_MCTS_TEMPERATURES = [0.4, 0.9]


def generate_mcts_candidates(
    *,
    agent: Any,
    current_img_b64: str,
    target_img_b64: str,
    delta_stat: dict[str, Any],
    history: list[dict[str, Any]],
    turn: int,
    current_metrics: dict[str, Any],
    planner_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ask the planner for open-ended tool+parameter candidates for MCTS."""
    candidate_limit = int(planner_cfg.get("candidates_per_call", 6))
    diversity_calls = int(planner_cfg.get("diversity_calls", 2))
    temperatures = list(planner_cfg.get("temperatures", DEFAULT_MCTS_TEMPERATURES))
    allow_heuristic_fallback = bool(planner_cfg.get("allow_heuristic_fallback", False))
    if not temperatures:
        temperatures = list(DEFAULT_MCTS_TEMPERATURES)

    system_prompt = build_planner_system_prompt(candidate_limit)
    user_content = build_image_content(current_img_b64, target_img_b64) + build_planner_user_prompt(
        delta_stat=delta_stat,
        history=history,
        turn=turn,
        current_metrics=current_metrics,
    )

    traces: list[dict[str, Any]] = []
    merged_candidates: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    if agent is not None:
        for call_idx in range(max(diversity_calls, 1)):
            temperature = float(temperatures[min(call_idx, len(temperatures) - 1)])
            raw_response = ""
            parsed = None
            try:
                raw_response = agent.call(
                    system_prompt,
                    [{"role": "user", "content": user_content}],
                    temperature=temperature,
                )
                parsed = parse_planner_response(raw_response)
            except Exception as exc:
                logger.warning("MCTS planner call failed on turn %s call %s: %s", turn, call_idx, exc)

            traces.append(
                {
                    "call_index": call_idx,
                    "temperature": temperature,
                    "raw_response": raw_response,
                    "parsed": parsed,
                }
            )
            if not parsed:
                continue
            for rank, item in enumerate(parsed.get("candidates", [])[:candidate_limit]):
                tool_name = item.get("tool", "")
                if tool_name not in TOOL_PRIORITY:
                    continue
                raw_parameters = item.get("parameters", {})
                if not isinstance(raw_parameters, dict):
                    continue
                signature = _candidate_signature(tool_name, raw_parameters)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                merged_candidates.append(
                    {
                        "tool": tool_name,
                        "parameters": raw_parameters,
                        "raw_parameters": raw_parameters,
                        "reason": item.get("reason", ""),
                        "prior": 1.0 / float(rank + 1 + call_idx),
                        "planner_call_id": call_idx,
                        "planner_temperature": temperature,
                    }
                )

    if not merged_candidates and allow_heuristic_fallback:
        merged_candidates = heuristic_mcts_candidates(delta_stat=delta_stat)
        source = "heuristic_fallback"
    elif merged_candidates:
        max_actions = int(planner_cfg.get("max_actions_per_node", 8))
        merged_candidates = merged_candidates[:max_actions]
        source = "planner"
    else:
        source = "none"

    return merged_candidates, {
        "source": source,
        "traces": traces,
    }


def heuristic_mcts_candidates(*, delta_stat: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate a small deterministic candidate set when the planner is unavailable."""
    candidates: list[dict[str, Any]] = []
    brightness = delta_stat.get("brightness_delta", 0.0)
    contrast = delta_stat.get("contrast_delta", 0.0)
    saturation = delta_stat.get("saturation_delta", 0.0)
    temperature = delta_stat.get("temperature_delta", 0.0)
    dominant_hsl_band = delta_stat.get("dominant_hsl_band", "reds")

    candidates.append(
        {
            "tool": "exposure_tool",
            "parameters": {
                "exposure": int(round(brightness / 12.0)),
                "brightness": int(round(brightness / 24.0)),
            },
            "raw_parameters": {
                "exposure": int(round(brightness / 12.0)),
                "brightness": int(round(brightness / 24.0)),
            },
            "reason": "Heuristic fallback for exposure mismatch.",
            "prior": 1.0,
            "planner_call_id": -1,
            "planner_temperature": None,
        }
    )
    candidates.append(
        {
            "tool": "tone_tool",
            "parameters": {
                "contrast": int(round(contrast / 8.0)),
                "highlights": int(round(delta_stat.get("highlight_delta", 0.0) / 8.0)),
                "shadows": int(round(delta_stat.get("shadow_delta", 0.0) / 8.0)),
            },
            "raw_parameters": {
                "contrast": int(round(contrast / 8.0)),
                "highlights": int(round(delta_stat.get("highlight_delta", 0.0) / 8.0)),
                "shadows": int(round(delta_stat.get("shadow_delta", 0.0) / 8.0)),
            },
            "reason": "Heuristic fallback for tone mismatch.",
            "prior": 0.5,
            "planner_call_id": -1,
            "planner_temperature": None,
        }
    )
    candidates.append(
        {
            "tool": "white_balance_tool",
            "parameters": {
                "temperature": int(round(temperature / 8.0)),
                "tint": int(round(delta_stat.get("tint_delta", 0.0) / 8.0)),
            },
            "raw_parameters": {
                "temperature": int(round(temperature / 8.0)),
                "tint": int(round(delta_stat.get("tint_delta", 0.0) / 8.0)),
            },
            "reason": "Heuristic fallback for white balance mismatch.",
            "prior": 0.33,
            "planner_call_id": -1,
            "planner_temperature": None,
        }
    )
    candidates.append(
        {
            "tool": "saturation_tool",
            "parameters": {
                "saturation": int(round(saturation / 8.0)),
                "vibrance": int(round(saturation / 12.0)),
            },
            "raw_parameters": {
                "saturation": int(round(saturation / 8.0)),
                "vibrance": int(round(saturation / 12.0)),
            },
            "reason": "Heuristic fallback for saturation mismatch.",
            "prior": 0.25,
            "planner_call_id": -1,
            "planner_temperature": None,
        }
    )
    candidates.append(
        {
            "tool": "hsl_tool",
            "parameters": {
                "adjustments": [
                    {
                        "color": dominant_hsl_band,
                        "hue": 0,
                        "saturation": int(round(saturation / 10.0)),
                        "luminance": int(round(brightness / 14.0)),
                    }
                ]
            },
            "raw_parameters": {
                "adjustments": [
                    {
                        "color": dominant_hsl_band,
                        "hue": 0,
                        "saturation": int(round(saturation / 10.0)),
                        "luminance": int(round(brightness / 14.0)),
                    }
                ]
            },
            "reason": "Heuristic fallback for dominant HSL mismatch.",
            "prior": 0.2,
            "planner_call_id": -1,
            "planner_temperature": None,
        }
    )
    return candidates


def _candidate_signature(tool_name: str, parameters: dict[str, Any]) -> str:
    payload = {"tool": tool_name, "parameters": parameters}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
