"""Trajectory generation pipeline using full-tree MCTS search."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.mllm_agent import MLLMAgent
from ..pipeline.mcts_search import MCTSNode, run_mcts_search
from ..pipeline.state_manager import SearchState, make_initial_tool_status
from ..tools.image_engine_adapter import make_default_params, params_to_dict
from ..utils.image_utils import save_image
from ..utils.metrics import compute_metrics
from ..utils.stat_utils import get_delta_stat
from .scoring import compute_objective_score

logger = logging.getLogger(__name__)


def generate_trajectory(
    src_img: np.ndarray,
    tgt_img: np.ndarray,
    agent: MLLMAgent,
    *,
    src_img_path: str = "",
    tgt_img_path: str = "",
    trajectory_id: str | None = None,
    output_dir: Path | None = None,
    max_turns: int = 8,
    convergence_delta_e: float = 4.0,
    thumbnail_size: int = 512,
    image_quality: int = 85,
    metrics_size: tuple[int, int] = (64, 64),
    use_lpips: bool = False,
    metrics_device: str = "cpu",
    save_images: bool = True,
    planner_cfg: dict[str, Any] | None = None,
    scoring_cfg: dict[str, Any] | None = None,
    mcts_cfg: dict[str, Any] | None = None,
    debug_cfg: dict[str, Any] | None = None,
) -> dict:
    """Generate a retouching trajectory for a single image pair."""
    planner_cfg = planner_cfg or {}
    scoring_cfg = scoring_cfg or {}
    mcts_cfg = mcts_cfg or {}
    debug_cfg = debug_cfg or {}

    trajectory_id = trajectory_id or str(uuid.uuid4())[:8]
    initial_quality = compute_metrics(
        src_img,
        tgt_img,
        size=metrics_size,
        use_lpips=use_lpips,
        device=metrics_device,
    )
    initial_score = compute_objective_score(
        metrics=initial_quality,
        delta_stat=get_delta_stat(src_img, tgt_img),
        scoring_cfg=scoring_cfg,
    )

    root_state = SearchState(
        params=make_default_params(),
        current_img=src_img.copy(),
        metrics=initial_quality,
        score=initial_score,
        tool_status=make_initial_tool_status(),
    )

    root_node, best_node, search_meta = run_mcts_search(
        root_state=root_state,
        src_img=src_img,
        tgt_img=tgt_img,
        agent=agent,
        max_turns=max_turns,
        convergence_delta_e=convergence_delta_e,
        thumbnail_size=thumbnail_size,
        image_quality=image_quality,
        metrics_size=metrics_size,
        use_lpips=use_lpips,
        metrics_device=metrics_device,
        planner_cfg=planner_cfg,
        scoring_cfg=scoring_cfg,
        mcts_cfg=mcts_cfg,
    )
    best_state = best_node.state

    trajectory_dir = Path(output_dir) / trajectory_id if output_dir else None
    tree_json_path = ""
    best_path_json_path = ""
    best_steps = _export_best_steps(best_state.steps, best_path_dir=None, image_quality=image_quality)

    if trajectory_dir and save_images:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        tree_json_path = str(trajectory_dir / "search_tree.json")
        best_path_json_path = str(trajectory_dir / "best_path.json")
        best_path_dir = trajectory_dir / "best_path_images"
        tree_payload = _export_tree(
            root_node=root_node,
            tree_dir=trajectory_dir / "tree_nodes",
            image_quality=image_quality,
        )
        with open(tree_json_path, "w", encoding="utf-8") as f:
            json.dump(tree_payload, f, ensure_ascii=False, indent=2)

        best_steps = _export_best_steps(
            best_state.steps,
            best_path_dir=best_path_dir,
            image_quality=image_quality,
        )
        best_path_payload = {
            "id": trajectory_id,
            "source": src_img_path,
            "target": tgt_img_path,
            "initial_quality": initial_quality,
            "final_quality": best_state.metrics,
            "final_score": round(float(best_state.score), 4),
            "best_node_id": best_node.node_id,
            "node_path": [node.node_id for node in _path_nodes(best_node)],
            "steps": best_steps,
        }
        with open(best_path_json_path, "w", encoding="utf-8") as f:
            json.dump(best_path_payload, f, ensure_ascii=False, indent=2)

    trajectory = {
        "id": trajectory_id,
        "source": src_img_path,
        "target": tgt_img_path,
        "initial_quality": initial_quality,
        "steps": best_steps,
        "final_quality": best_state.metrics,
        "num_steps": len(best_steps),
        "artifacts": {
            "trajectory_dir": str(trajectory_dir) if trajectory_dir else "",
            "tree_json": tree_json_path,
            "best_path_json": best_path_json_path,
        },
        "search_meta": {
            **search_meta,
            "final_score": round(float(best_state.score), 4),
            "planner_model_role": planner_cfg.get("role", "planner"),
            "score_mode": scoring_cfg.get("mode", "hybrid"),
            "search_mode": "mcts",
        },
    }

    logger.info(
        "[%s] Completed with %s best-path steps across %s tree nodes -> DeltaE=%.2f, PSNR=%.1f",
        trajectory_id,
        len(best_steps),
        search_meta.get("total_nodes", 1),
        best_state.metrics.get("delta_e", 0.0),
        best_state.metrics.get("psnr", 0.0),
    )
    return trajectory


def _export_tree(root_node: MCTSNode, *, tree_dir: Path, image_quality: int) -> dict[str, Any]:
    tree_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    stack = [root_node]
    while stack:
        node = stack.pop()
        image_path = _save_step_image(node.state.current_img, tree_dir, f"node_{node.node_id:06d}", image_quality)
        last_step = node.state.steps[-1] if node.depth > 0 and node.state.steps else None
        nodes.append(
            {
                "node_id": node.node_id,
                "parent_id": node.parent.node_id if node.parent else None,
                "depth": node.depth,
                "image_path": image_path,
                "metrics": node.state.metrics,
                "score": round(float(node.state.score), 4),
                "params_accumulated": params_to_dict(node.state.params),
                "visits": node.visits,
                "value_sum": round(float(node.value_sum), 4),
                "q_value": round(node.q_value, 4),
                "prior": round(float(node.prior), 4),
                "terminal_reason": node.terminal_reason,
                "children": [child.node_id for child in node.children],
                "planner_meta": node.planner_meta,
                "action": _serialize_tree_action(last_step),
            }
        )
        stack.extend(reversed(node.children))
    nodes.sort(key=lambda item: item["node_id"])
    return {
        "root_node_id": root_node.node_id,
        "num_nodes": len(nodes),
        "nodes": nodes,
    }


def _export_best_steps(
    steps: list[dict[str, Any]],
    *,
    best_path_dir: Path | None,
    image_quality: int,
) -> list[dict[str, Any]]:
    exported = []
    for idx, step in enumerate(steps):
        output_image_path = ""
        input_image_path = ""
        if best_path_dir is not None and "_input_img" in step and "_output_img" in step:
            input_image_path = _save_step_image(
                step["_input_img"],
                best_path_dir,
                f"step_{idx:02d}_input",
                image_quality,
            )
            output_image_path = _save_step_image(
                step["_output_img"],
                best_path_dir,
                f"step_{idx:02d}_output",
                image_quality,
            )

        exported.append(
            {
                "round": step["round"],
                "input_image": input_image_path,
                "cot": step.get("cot", ""),
                "tool": step["tool"],
                "parameters": step["parameters"],
                "delta_parameters": step["delta_parameters"],
                "params_accumulated": step["params_accumulated"],
                "params_accumulated_tool": step["params_accumulated_tool"],
                "output_image": output_image_path,
                "step_quality": step["step_quality"],
                "delta_stat": step["delta_stat"],
                "proposal": step["proposal"],
                "score_before": step["score_before"],
                "score_after": step["score_after"],
                "score_gain": step.get("score_gain"),
                "accepted": True,
                "action_signature": step.get("action_signature", ""),
                "planner_call_id": step.get("planner_call_id"),
                "planner_temperature": step.get("planner_temperature"),
                "mcts_summary": step.get("mcts_summary", {}),
            }
        )
    return exported


def _serialize_tree_action(step: dict[str, Any] | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "round": step.get("round"),
        "tool": step.get("tool"),
        "parameters": step.get("parameters", {}),
        "delta_parameters": step.get("delta_parameters", {}),
        "proposal": step.get("proposal", {}),
        "step_quality": step.get("step_quality", {}),
        "score_before": step.get("score_before"),
        "score_after": step.get("score_after"),
        "score_gain": step.get("score_gain"),
        "action_signature": step.get("action_signature", ""),
        "planner_call_id": step.get("planner_call_id"),
        "planner_temperature": step.get("planner_temperature"),
    }


def _path_nodes(node: MCTSNode) -> list[MCTSNode]:
    nodes = []
    current: MCTSNode | None = node
    while current is not None:
        nodes.append(current)
        current = current.parent
    return list(reversed(nodes))


def _save_step_image(img: np.ndarray, step_dir: Path, name: str, quality: int) -> str:
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / f"{name}.jpg"
    save_image(path, img, quality=quality)
    return str(path)
