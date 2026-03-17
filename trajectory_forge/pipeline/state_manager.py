"""Search-state helpers for accepted-state trajectory generation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..tools.tool_registry import TOOL_PRIORITY


TOOL_NAMES = list(TOOL_PRIORITY.keys())


@dataclass
class SearchState:
    """A single accepted state in the search frontier."""

    params: Any
    current_img: np.ndarray
    metrics: dict[str, Any]
    score: float
    steps: list[dict[str, Any]] = field(default_factory=list)
    step_history: list[dict[str, Any]] = field(default_factory=list)
    tool_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    debug_trace: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False


def make_initial_tool_status() -> dict[str, dict[str, Any]]:
    """Create the initial per-tool runtime state."""
    return {
        tool: {
            "accepted_streak": 0,
            "reject_streak": 0,
            "cooldown_until": -1,
            "locked": False,
            "last_delta": {},
        }
        for tool in TOOL_NAMES
    }


def clone_state(state: SearchState) -> SearchState:
    """Return a deep copy of a search state."""
    return SearchState(
        params=copy.deepcopy(state.params),
        current_img=state.current_img.copy(),
        metrics=copy.deepcopy(state.metrics),
        score=float(state.score),
        steps=copy.deepcopy(state.steps),
        step_history=copy.deepcopy(state.step_history),
        tool_status=copy.deepcopy(state.tool_status),
        debug_trace=copy.deepcopy(state.debug_trace),
        completed=bool(state.completed),
    )


def get_locked_tools(tool_status: dict[str, dict[str, Any]]) -> list[str]:
    """Return the currently locked tools."""
    return [tool for tool, status in tool_status.items() if status.get("locked")]


def get_cooldown_tools(tool_status: dict[str, dict[str, Any]], turn: int) -> list[str]:
    """Return the tools that are currently in cooldown."""
    return [
        tool
        for tool, status in tool_status.items()
        if int(status.get("cooldown_until", -1)) >= turn
    ]


def apply_accept(
    tool_status: dict[str, dict[str, Any]],
    *,
    tool_name: str,
    delta_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Update tool runtime state after accepting a candidate."""
    new_status = copy.deepcopy(tool_status)
    for tool, status in new_status.items():
        if tool == tool_name:
            status["accepted_streak"] = int(status.get("accepted_streak", 0)) + 1
            status["reject_streak"] = 0
            status["cooldown_until"] = -1
            status["last_delta"] = copy.deepcopy(delta_params)
        else:
            status["accepted_streak"] = 0
    return new_status


def apply_reject(
    tool_status: dict[str, dict[str, Any]],
    *,
    tool_name: str,
    turn: int,
    reject_limit: int,
    cooldown_steps: int,
) -> dict[str, dict[str, Any]]:
    """Update tool runtime state after rejecting a candidate."""
    new_status = copy.deepcopy(tool_status)
    status = new_status[tool_name]
    status["reject_streak"] = int(status.get("reject_streak", 0)) + 1
    if status["reject_streak"] >= reject_limit:
        status["cooldown_until"] = turn + cooldown_steps
        status["reject_streak"] = 0
    return new_status


def sync_tool_locks(
    tool_status: dict[str, dict[str, Any]],
    residuals: dict[str, float],
    *,
    lock_threshold: float,
    unlock_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Update lock state based on residual magnitudes."""
    new_status = copy.deepcopy(tool_status)
    for tool, residual in residuals.items():
        if residual <= lock_threshold:
            new_status[tool]["locked"] = True
        elif residual >= unlock_threshold:
            new_status[tool]["locked"] = False
    return new_status


def is_tool_available(
    tool_status: dict[str, dict[str, Any]],
    *,
    tool_name: str,
    turn: int,
    max_accept_streak: int,
) -> bool:
    """Return whether a tool is available for the current turn."""
    status = tool_status[tool_name]
    if status.get("locked"):
        return False
    if int(status.get("cooldown_until", -1)) >= turn:
        return False
    if int(status.get("accepted_streak", 0)) >= max_accept_streak:
        return False
    return True
