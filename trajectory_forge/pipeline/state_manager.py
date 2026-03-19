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


def apply_accept(
    tool_status: dict[str, dict[str, Any]],
    *,
    tool_name: str,
    delta_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Update tool runtime state after accepting a candidate."""
    new_status = copy.deepcopy(tool_status)
    if tool_name not in new_status:
        return new_status

    for tool, status in new_status.items():
        if tool == tool_name:
            status["accepted_streak"] = int(status.get("accepted_streak", 0)) + 1
            status["reject_streak"] = 0
            status["cooldown_until"] = -1
            status["last_delta"] = copy.deepcopy(delta_params)
        else:
            status["accepted_streak"] = 0
    return new_status
