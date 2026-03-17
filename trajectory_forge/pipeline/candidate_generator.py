"""Candidate planning utilities for the hidden-search trajectory generator."""

from __future__ import annotations

import logging
from typing import Any

from ..agents.mllm_agent import parse_planner_response
from ..agents.prompts import (
    build_image_content,
    build_planner_system_prompt,
    build_planner_user_prompt,
)
from ..tools.tool_registry import TOOL_PRIORITY
from .scoring import compute_tool_residuals
from .state_manager import get_cooldown_tools, get_locked_tools, is_tool_available

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = {"increase", "decrease", "mixed"}
VALID_BUCKETS = {"small", "medium", "large"}


def shortlist_tools(
    *,
    delta_stat: dict[str, Any],
    tool_status: dict[str, dict[str, Any]],
    turn: int,
    shortlist_size: int,
    max_accept_streak: int,
    residual_floor: float,
) -> tuple[list[str], dict[str, float], list[str], list[str]]:
    """Return the allowed tools for the current state."""
    residuals = compute_tool_residuals(delta_stat)
    locked_tools = get_locked_tools(tool_status)
    cooldown_tools = get_cooldown_tools(tool_status, turn)

    ranked = sorted(
        residuals.items(),
        key=lambda item: (item[1], -TOOL_PRIORITY[item[0]]),
        reverse=True,
    )

    shortlist = []
    for tool_name, residual in ranked:
        if residual < residual_floor and shortlist:
            continue
        if not is_tool_available(
            tool_status,
            tool_name=tool_name,
            turn=turn,
            max_accept_streak=max_accept_streak,
        ):
            continue
        shortlist.append(tool_name)
        if len(shortlist) >= shortlist_size:
            break

    if not shortlist:
        for tool_name, _ in ranked:
            if tool_name in cooldown_tools:
                continue
            shortlist.append(tool_name)
            if len(shortlist) >= shortlist_size:
                break

    return shortlist, residuals, locked_tools, cooldown_tools


def generate_tool_proposals(
    *,
    agent: Any,
    current_img_b64: str,
    target_img_b64: str,
    delta_stat: dict[str, Any],
    history: list[dict[str, Any]],
    turn: int,
    shortlist_tools: list[str],
    locked_tools: list[str],
    cooldown_tools: list[str],
    current_metrics: dict[str, Any],
    current_score: float,
    max_proposals: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ask the planner for coarse proposals, with heuristic fallback."""
    if not shortlist_tools:
        return [], {"source": "none", "raw_response": "", "parsed": None}

    system_prompt = build_planner_system_prompt()
    user_content = build_image_content(current_img_b64, target_img_b64) + build_planner_user_prompt(
        delta_stat=delta_stat,
        history=history,
        turn=turn,
        shortlist_tools=shortlist_tools,
        current_metrics=current_metrics,
        current_score=current_score,
        locked_tools=locked_tools,
        cooldown_tools=cooldown_tools,
    )

    raw_response = ""
    parsed = None
    if agent is not None:
        try:
            raw_response = agent.call(system_prompt, [{"role": "user", "content": user_content}])
            parsed = parse_planner_response(raw_response)
        except Exception as exc:
            logger.warning("Planner call failed on turn %s: %s", turn, exc)

    proposals: list[dict[str, Any]] = []
    if parsed and not parsed.get("should_stop", False):
        for proposal in parsed.get("proposals", []):
            tool_name = proposal.get("tool", "")
            if tool_name not in shortlist_tools:
                continue
            direction = proposal.get("direction", "mixed")
            magnitude_bucket = proposal.get("magnitude_bucket", "medium")
            if direction not in VALID_DIRECTIONS or magnitude_bucket not in VALID_BUCKETS:
                continue
            proposals.append(
                {
                    "tool": tool_name,
                    "direction": direction,
                    "magnitude_bucket": magnitude_bucket,
                    "reason": proposal.get("reason", ""),
                }
            )

    if not proposals:
        proposals = heuristic_proposals(
            delta_stat=delta_stat,
            shortlist_tools=shortlist_tools,
            max_proposals=max_proposals,
        )
        source = "heuristic_fallback"
    else:
        proposals = proposals[:max_proposals]
        source = "planner"

    return proposals, {
        "source": source,
        "raw_response": raw_response,
        "parsed": parsed,
        "shortlist_tools": shortlist_tools,
    }


def heuristic_proposals(
    *,
    delta_stat: dict[str, Any],
    shortlist_tools: list[str],
    max_proposals: int,
) -> list[dict[str, Any]]:
    """Generate deterministic fallback proposals when planner output is unavailable."""
    proposals = []
    for tool_name in shortlist_tools[:max_proposals]:
        proposals.append(
            {
                "tool": tool_name,
                "direction": infer_direction(tool_name, delta_stat),
                "magnitude_bucket": infer_bucket(tool_name, delta_stat),
                "reason": f"Heuristic fallback proposal for {tool_name}.",
            }
        )
    return proposals


def infer_direction(tool_name: str, delta_stat: dict[str, Any]) -> str:
    """Infer a coarse direction from residual statistics."""
    if tool_name == "exposure_tool":
        return "increase" if delta_stat.get("brightness_delta", 0.0) > 0 else "decrease"
    if tool_name == "tone_tool":
        return "increase" if delta_stat.get("contrast_delta", 0.0) > 0 else "mixed"
    if tool_name == "white_balance_tool":
        combined = delta_stat.get("temperature_delta", 0.0) + delta_stat.get("tint_delta", 0.0)
        return "increase" if combined >= 0 else "decrease"
    if tool_name == "saturation_tool":
        return "increase" if delta_stat.get("saturation_delta", 0.0) > 0 else "decrease"
    return "mixed"


def infer_bucket(tool_name: str, delta_stat: dict[str, Any]) -> str:
    """Infer a coarse magnitude bucket from residual statistics."""
    residuals = compute_tool_residuals(delta_stat)
    residual = residuals.get(tool_name, 0.0)
    if residual < 0.05:
        return "small"
    if residual < 0.12:
        return "medium"
    return "large"
