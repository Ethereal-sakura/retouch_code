"""Monte Carlo Tree Search for long-horizon retouch trajectories."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from ..tools.image_engine_adapter import (
    diff_tool_params,
    get_tool_params,
    merge_tool_call,
    params_to_dict,
    render,
)
from ..tools.tool_registry import HSL_BAND_NAMES, TOOL_SCHEMAS
from ..utils.image_utils import encode_image_base64, make_thumbnail
from ..utils.metrics import compute_metrics
from ..utils.stat_utils import get_delta_stat
from .candidate_generator import generate_mcts_candidates
from .scoring import compute_objective_score
from .state_manager import SearchState, apply_accept, clone_state


DEFAULT_MCTS_CFG = {
    "num_simulations": 48,
    "c_puct": 1.4,
    "max_actions_per_node": 8,
    "rollout_horizon": 8,
    "min_step_gain": 0.01,
    "regression_tolerance": 0.05,
    "prune_regression_tolerance": None,
}

STEP_DELTA_LIMITS: dict[str, dict[str, int]] = {
    "exposure_tool": {"exposure": 16, "brightness": 16},
    "tone_tool": {"contrast": 24, "highlights": 24, "shadows": 24, "whites": 12, "blacks": 18},
    "white_balance_tool": {"temperature": 16, "tint": 16},
    "saturation_tool": {"saturation": 20, "vibrance": 20},
    "hsl_tool": {"hue": 18, "saturation": 18, "luminance": 18},
}


@dataclass
class MCTSNode:
    """One node in the search tree."""

    node_id: int
    state: SearchState
    parent: MCTSNode | None = None
    action: dict[str, Any] | None = None
    prior: float = 1.0
    depth: int = 0
    children: list[MCTSNode] = field(default_factory=list)
    unexpanded_actions: list[dict[str, Any]] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    terminal_reason: str = ""
    planner_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / float(self.visits)


def run_mcts_search(
    *,
    root_state: SearchState,
    src_img,
    tgt_img,
    agent: Any,
    max_turns: int,
    convergence_delta_e: float,
    thumbnail_size: int,
    image_quality: int,
    metrics_size: tuple[int, int],
    use_lpips: bool,
    metrics_device: str,
    planner_cfg: dict[str, Any],
    scoring_cfg: dict[str, Any] | None = None,
    mcts_cfg: dict[str, Any] | None = None,
) -> tuple[MCTSNode, MCTSNode, dict[str, Any]]:
    """Run MCTS over retouch states and return the root and best node."""
    mcts_cfg = {**DEFAULT_MCTS_CFG, **(mcts_cfg or {})}
    scoring_cfg = scoring_cfg or {}

    root = MCTSNode(node_id=0, state=clone_state(root_state))
    all_nodes = [root]
    next_node_id = 1
    root_score = float(root.state.score)
    target_b64 = encode_image_base64(make_thumbnail(tgt_img, thumbnail_size), quality=image_quality)

    search_meta = {
        "planner_calls": 0,
        "candidate_evaluations": 0,
        "expanded_nodes": 0,
        "invalid_actions": 0,
        "repeated_actions": 0,
        "regression_nodes": 0,
        "low_gain_nodes": 0,
        "num_simulations": int(mcts_cfg["num_simulations"]),
    }

    for _ in range(int(mcts_cfg["num_simulations"])):
        node = _select_leaf(root, c_puct=float(mcts_cfg["c_puct"]))
        _refresh_node_runtime_state(
            node,
            convergence_delta_e=convergence_delta_e,
            max_turns=max_turns,
            rollout_horizon=int(mcts_cfg["rollout_horizon"]),
        )

        if node.terminal_reason:
            _backpropagate(node, _evaluate_leaf_value(node=node, root_score=root_score))
            continue

        if not node.unexpanded_actions:
            current_b64 = encode_image_base64(
                make_thumbnail(node.state.current_img, thumbnail_size),
                quality=image_quality,
            )
            delta_stat = get_delta_stat(node.state.current_img, tgt_img)
            candidates, planner_trace = generate_mcts_candidates(
                agent=agent,
                current_img_b64=current_b64,
                target_img_b64=target_b64,
                delta_stat=delta_stat,
                history=node.state.step_history,
                turn=node.depth,
                current_metrics=node.state.metrics,
                planner_cfg=planner_cfg,
            )
            search_meta["planner_calls"] += max(len(planner_trace.get("traces", [])), 1)
            node.planner_meta = planner_trace
            node.unexpanded_actions = candidates[: int(mcts_cfg["max_actions_per_node"])]
            if not node.unexpanded_actions:
                node.terminal_reason = "no_actions"
                _backpropagate(node, _evaluate_leaf_value(node=node, root_score=root_score))
                continue

        action = node.unexpanded_actions.pop(0)
        child, outcome = _expand_action(
            node=node,
            node_id=next_node_id,
            action=action,
            src_img=src_img,
            tgt_img=tgt_img,
            metrics_size=metrics_size,
            use_lpips=use_lpips,
            metrics_device=metrics_device,
            scoring_cfg=scoring_cfg,
            regression_tolerance=float(mcts_cfg["regression_tolerance"]),
            prune_regression_tolerance=mcts_cfg.get("prune_regression_tolerance"),
            min_step_gain=float(mcts_cfg["min_step_gain"]),
            convergence_delta_e=convergence_delta_e,
            max_turns=max_turns,
            rollout_horizon=int(mcts_cfg["rollout_horizon"]),
        )
        search_meta["candidate_evaluations"] += 1

        if child is None:
            search_meta[f"{outcome}_actions"] = search_meta.get(f"{outcome}_actions", 0) + 1
            _backpropagate(node, _evaluate_leaf_value(node=node, root_score=root_score))
            continue

        next_node_id += 1
        node.children.append(child)
        all_nodes.append(child)
        search_meta["expanded_nodes"] += 1
        if outcome == "regression":
            search_meta["regression_nodes"] += 1
        if outcome == "low_gain":
            search_meta["low_gain_nodes"] += 1

        _backpropagate(child, _evaluate_leaf_value(node=child, root_score=root_score))

    best_node = _select_best_node(all_nodes)
    _attach_mcts_summaries(best_node)
    search_meta["total_nodes"] = len(all_nodes)
    search_meta["best_node_id"] = best_node.node_id
    search_meta["best_visits"] = best_node.visits
    search_meta["best_value"] = round(best_node.q_value, 4)
    return root, best_node, search_meta


def _select_leaf(root: MCTSNode, *, c_puct: float) -> MCTSNode:
    node = root
    while node.children and not node.unexpanded_actions and not node.terminal_reason:
        parent_scale = max(float(node.visits), 1.0) ** 0.5
        node = max(
            node.children,
            key=lambda child: child.q_value + c_puct * child.prior * parent_scale / (1.0 + float(child.visits)),
        )
    return node


def _refresh_node_runtime_state(
    node: MCTSNode,
    *,
    convergence_delta_e: float,
    max_turns: int,
    rollout_horizon: int,
) -> None:
    if node.depth >= max_turns:
        node.terminal_reason = "max_turns"
        return
    if node.depth >= rollout_horizon:
        node.terminal_reason = "rollout_horizon"
        return
    if float(node.state.metrics.get("delta_e", 0.0)) < convergence_delta_e:
        node.terminal_reason = "delta_e_converged"


def _expand_action(
    *,
    node: MCTSNode,
    node_id: int,
    action: dict[str, Any],
    src_img,
    tgt_img,
    metrics_size: tuple[int, int],
    use_lpips: bool,
    metrics_device: str,
    scoring_cfg: dict[str, Any],
    regression_tolerance: float,
    prune_regression_tolerance: float | None,
    min_step_gain: float,
    convergence_delta_e: float,
    max_turns: int,
    rollout_horizon: int,
) -> tuple[MCTSNode | None, str]:
    normalized = normalize_action(action.get("tool", ""), action.get("parameters", {}))
    if normalized is None:
        return None, "invalid"

    signature = make_action_signature(normalized["tool"], normalized["delta_parameters"])
    if _is_repeated_in_ancestors(node, signature):
        return None, "repeated"

    delta_stat_before = get_delta_stat(node.state.current_img, tgt_img)
    evaluation = _evaluate_action(
        state=node.state,
        tool_name=normalized["tool"],
        raw_parameters=copy.deepcopy(action.get("raw_parameters", action.get("parameters", {}))),
        normalized_delta=normalized["delta_parameters"],
        action_signature=signature,
        src_img=src_img,
        tgt_img=tgt_img,
        metrics_size=metrics_size,
        use_lpips=use_lpips,
        metrics_device=metrics_device,
        scoring_cfg=scoring_cfg,
    )
    if evaluation is None:
        return None, "invalid"

    parent_delta_e = float(node.state.metrics.get("delta_e", 0.0))
    child_delta_e = float(evaluation["metrics"].get("delta_e", 0.0))
    if prune_regression_tolerance is not None and child_delta_e > parent_delta_e + float(prune_regression_tolerance):
        return None, "invalid"

    step_gain = float(node.state.score) - float(evaluation["score"])
    child = _build_child_node(
        node=node,
        node_id=node_id,
        action=action,
        evaluation=evaluation,
        step_gain=step_gain,
        delta_stat_before=delta_stat_before,
    )

    outcome = "expanded"
    _refresh_node_runtime_state(
        child,
        convergence_delta_e=convergence_delta_e,
        max_turns=max_turns,
        rollout_horizon=rollout_horizon,
    )
    if child_delta_e > parent_delta_e + regression_tolerance:
        child.terminal_reason = child.terminal_reason or "regression"
        outcome = "regression"
    elif step_gain < min_step_gain and child.depth >= 1:
        child.terminal_reason = child.terminal_reason or "low_gain"
        outcome = "low_gain"
    return child, outcome


def _evaluate_action(
    *,
    state: SearchState,
    tool_name: str,
    raw_parameters: dict[str, Any],
    normalized_delta: dict[str, Any],
    action_signature: str,
    src_img,
    tgt_img,
    metrics_size: tuple[int, int],
    use_lpips: bool,
    metrics_device: str,
    scoring_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    new_params = merge_tool_call(state.params, tool_name, normalized_delta)
    actual_delta = diff_tool_params(state.params, new_params, tool_name)
    actual_delta = _serialize_numeric_tree(actual_delta)
    if _is_zero_delta(actual_delta):
        return None

    rendered = render(src_img, new_params)
    metrics = compute_metrics(
        rendered,
        tgt_img,
        size=metrics_size,
        use_lpips=use_lpips,
        device=metrics_device,
    )
    output_delta_stat = get_delta_stat(rendered, tgt_img)
    score = compute_objective_score(
        metrics=metrics,
        delta_stat=output_delta_stat,
        delta_params=actual_delta,
        tool_name=tool_name,
        tool_state=state.tool_status,
        scoring_cfg=scoring_cfg,
    )
    return {
        "tool": tool_name,
        "parameters": raw_parameters,
        "delta_parameters": actual_delta,
        "full_params": new_params,
        "metrics": metrics,
        "score": float(score),
        "output_image": rendered,
        "output_delta_stat": output_delta_stat,
        "params_accumulated": _serialize_numeric_tree(params_to_dict(new_params)),
        "params_accumulated_tool": _serialize_numeric_tree(get_tool_params(new_params, tool_name)),
        "action_signature": action_signature,
    }


def _build_child_node(
    *,
    node: MCTSNode,
    node_id: int,
    action: dict[str, Any],
    evaluation: dict[str, Any],
    step_gain: float,
    delta_stat_before: dict[str, Any],
) -> MCTSNode:
    child_state = clone_state(node.state)
    child_state.params = evaluation["full_params"]
    child_state.current_img = evaluation["output_image"]
    child_state.metrics = evaluation["metrics"]
    child_state.score = float(evaluation["score"])
    child_state.tool_status = apply_accept(
        child_state.tool_status,
        tool_name=evaluation["tool"],
        delta_params=evaluation["delta_parameters"],
    )

    step_record = {
        "round": node.depth,
        "cot": action.get("reason", ""),
        "tool": evaluation["tool"],
        "parameters": copy.deepcopy(action.get("raw_parameters", action.get("parameters", {}))),
        "delta_parameters": evaluation["delta_parameters"],
        "params_accumulated": evaluation["params_accumulated"],
        "params_accumulated_tool": evaluation["params_accumulated_tool"],
        "output_image": "",
        "step_quality": evaluation["metrics"],
        "delta_stat": delta_stat_before,
        "proposal": {
            "tool": evaluation["tool"],
            "parameters": copy.deepcopy(action.get("raw_parameters", action.get("parameters", {}))),
            "reason": action.get("reason", ""),
        },
        "probe_summary": [],
        "score_before": node.state.score,
        "score_after": evaluation["score"],
        "accepted": True,
        "score_gain": step_gain,
        "action_signature": evaluation["action_signature"],
        "planner_call_id": action.get("planner_call_id"),
        "planner_temperature": action.get("planner_temperature"),
        "_input_img": node.state.current_img.copy(),
        "_output_img": evaluation["output_image"].copy(),
    }
    child_state.steps.append(step_record)
    child_state.step_history.append(
        {
            "round": node.depth,
            "tool": evaluation["tool"],
            "parameters": copy.deepcopy(action.get("raw_parameters", action.get("parameters", {}))),
            "delta_parameters": evaluation["delta_parameters"],
            "params_accumulated_tool": evaluation["params_accumulated_tool"],
            "step_quality": evaluation["metrics"],
            "action_signature": evaluation["action_signature"],
        }
    )
    child_state.debug_trace.append(
        {
            "turn": node.depth,
            "proposal": action,
            "score_before": node.state.score,
            "score_after": evaluation["score"],
            "action_signature": evaluation["action_signature"],
        }
    )

    return MCTSNode(
        node_id=node_id,
        state=child_state,
        parent=node,
        action=copy.deepcopy(action),
        prior=float(action.get("prior", 0.0)),
        depth=node.depth + 1,
    )


def _evaluate_leaf_value(*, node: MCTSNode, root_score: float) -> float:
    return float(root_score) - float(node.state.score)


def _backpropagate(node: MCTSNode, value: float) -> None:
    current: MCTSNode | None = node
    while current is not None:
        current.visits += 1
        current.value_sum += float(value)
        current = current.parent


def _select_best_node(nodes: list[MCTSNode]) -> MCTSNode:
    candidates = [node for node in nodes if node.depth > 0]
    if not candidates:
        return nodes[0]
    return min(
        candidates,
        key=lambda node: (
            float(node.state.score),
            -float(node.state.metrics.get("psnr", 0.0)),
            -node.depth,
            -node.visits,
        ),
    )


def _attach_mcts_summaries(best_node: MCTSNode) -> None:
    path_nodes = _path_nodes(best_node)[1:]
    if not path_nodes:
        return
    for idx, node in enumerate(path_nodes):
        best_node.state.steps[idx]["mcts_summary"] = {
            "node_id": node.node_id,
            "visit_count": node.visits,
            "value": round(node.q_value, 4),
            "prior": round(float(node.prior), 4),
            "candidate_rank": idx,
            "terminal_reason": node.terminal_reason,
        }


def _path_nodes(node: MCTSNode) -> list[MCTSNode]:
    nodes = []
    current: MCTSNode | None = node
    while current is not None:
        nodes.append(current)
        current = current.parent
    return list(reversed(nodes))


def normalize_action(tool_name: str, raw_parameters: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize model-proposed parameters into executable integer deltas."""
    if tool_name not in TOOL_SCHEMAS or not isinstance(raw_parameters, dict):
        return None

    if tool_name == "hsl_tool":
        adjustments = raw_parameters.get("adjustments")
        if not isinstance(adjustments, list) or len(adjustments) != 1:
            return None
        item = adjustments[0]
        if not isinstance(item, dict):
            return None
        color = item.get("color")
        if color not in HSL_BAND_NAMES:
            return None
        normalized_item = {"color": color}
        for key in ("hue", "saturation", "luminance"):
            value = _to_numeric(item.get(key, 0))
            if value is None:
                continue
            limit = STEP_DELTA_LIMITS["hsl_tool"][key]
            schema_lo, schema_hi = (-100, 100)
            normalized_item[key] = _clamp_rounded(value, lo=max(schema_lo, -limit), hi=min(schema_hi, limit))
        normalized_item.setdefault("hue", 0)
        normalized_item.setdefault("saturation", 0)
        normalized_item.setdefault("luminance", 0)
        if all(int(normalized_item[key]) == 0 for key in ("hue", "saturation", "luminance")):
            return None
        return {"tool": tool_name, "delta_parameters": {"adjustments": [normalized_item]}}

    normalized: dict[str, Any] = {}
    for param_name, param_schema in TOOL_SCHEMAS[tool_name]["parameters"].items():
        if param_name not in raw_parameters:
            continue
        value = _to_numeric(raw_parameters.get(param_name))
        if value is None:
            continue
        lo, hi = param_schema["range"]
        step_limit = STEP_DELTA_LIMITS[tool_name][param_name]
        normalized[param_name] = _clamp_rounded(
            value,
            lo=max(int(lo), -step_limit),
            hi=min(int(hi), step_limit),
        )
    normalized = {key: value for key, value in normalized.items() if int(value) != 0}
    if not normalized:
        return None
    return {"tool": tool_name, "delta_parameters": normalized}


def make_action_signature(tool_name: str, delta_parameters: dict[str, Any]) -> str:
    """Build a canonical action signature for normalized tool deltas."""
    payload = {"tool": tool_name, "delta_parameters": delta_parameters}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _is_repeated_in_ancestors(node: MCTSNode, action_signature: str) -> bool:
    current: MCTSNode | None = node
    while current is not None:
        if current.action is not None and current.state.steps:
            if current.state.steps[-1].get("action_signature") == action_signature:
                return True
        current = current.parent
    return False


def _is_zero_delta(delta_parameters: dict[str, Any]) -> bool:
    if "adjustments" in delta_parameters:
        for item in delta_parameters.get("adjustments", []):
            if any(abs(float(item.get(key, 0))) > 1e-6 for key in ("hue", "saturation", "luminance")):
                return False
        return True
    return all(abs(float(value)) <= 1e-6 for value in delta_parameters.values())


def _serialize_numeric_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_numeric_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_numeric_tree(item) for item in value]
    if isinstance(value, float):
        rounded = round(value)
        if abs(value - rounded) < 1e-6:
            return int(rounded)
        return float(value)
    return value


def _to_numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_rounded(value: float, *, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(float(value)))))
