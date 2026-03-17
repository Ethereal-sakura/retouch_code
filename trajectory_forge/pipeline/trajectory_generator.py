"""Trajectory generation pipeline using accepted-state hidden search."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.mllm_agent import MLLMAgent, parse_thinking
from ..agents.prompts import (
    build_explainer_system_prompt,
    build_explainer_user_prompt,
)
from ..pipeline.candidate_generator import generate_tool_proposals, shortlist_tools
from ..pipeline.probe_engine import probe_and_refine
from ..pipeline.scoring import (
    compute_objective_score,
    compute_tool_residuals,
    rank_states_diverse,
    should_accept_candidate,
)
from ..pipeline.state_manager import (
    SearchState,
    apply_accept,
    apply_reject,
    clone_state,
    make_initial_tool_status,
    sync_tool_locks,
)
from ..tools.image_engine_adapter import make_default_params
from ..utils.image_utils import encode_image_base64, make_thumbnail, save_image
from ..utils.metrics import compute_metrics
from ..utils.stat_utils import get_delta_stat

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_CFG = {
    "beam_size": 4,
    "shortlist_tools": 2,
    "max_proposals": 3,
    "accept_margin": 0.02,
    "hard_delta_e_tolerance": 0.05,
    "residual_floor": 0.015,
    "lock_threshold": 0.02,
    "unlock_threshold": 0.06,
    "max_accept_streak": 2,
    "reject_limit": 2,
    "cooldown_steps": 1,
    "stop_residual_threshold": 0.015,
}


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
    search_cfg: dict[str, Any] | None = None,
    probe_cfg: dict[str, Any] | None = None,
    scoring_cfg: dict[str, Any] | None = None,
    planner_cfg: dict[str, Any] | None = None,
    debug_cfg: dict[str, Any] | None = None,
) -> dict:
    """Generate a retouching trajectory for a single image pair."""
    search_cfg = {**DEFAULT_SEARCH_CFG, **(search_cfg or {})}
    probe_cfg = probe_cfg or {}
    scoring_cfg = scoring_cfg or {}
    planner_cfg = planner_cfg or {}
    debug_cfg = debug_cfg or {}

    trajectory_id = trajectory_id or str(uuid.uuid4())[:8]
    probe_max_size = int(probe_cfg.get("render_size", 320))
    src_probe_img = make_thumbnail(src_img, probe_max_size)
    tgt_probe_img = make_thumbnail(tgt_img, probe_max_size)

    initial_quality = compute_metrics(
        src_img,
        tgt_img,
        size=metrics_size,
        use_lpips=use_lpips,
        device=metrics_device,
    )
    initial_delta_stat = get_delta_stat(src_img, tgt_img)
    initial_score = compute_objective_score(
        metrics=initial_quality,
        delta_stat=initial_delta_stat,
        weights=scoring_cfg.get("weights", {}),
    )

    root_state = SearchState(
        params=make_default_params(),
        current_img=src_img.copy(),
        metrics=initial_quality,
        score=initial_score,
        steps=[],
        step_history=[],
        tool_status=make_initial_tool_status(),
        debug_trace=[],
        completed=False,
    )

    beam = [root_state]
    completed_states: list[SearchState] = []
    search_meta = {
        "planner_calls": 0,
        "probe_evaluations": 0,
        "accepted_candidates": 0,
        "rejected_candidates": 0,
    }

    for turn in range(max_turns):
        expanded_states: list[SearchState] = []
        frontier_stalled = True

        for state_idx, state in enumerate(beam):
            delta_stat = get_delta_stat(state.current_img, tgt_img)
            residuals = compute_tool_residuals(delta_stat)
            state.tool_status = sync_tool_locks(
                state.tool_status,
                residuals,
                lock_threshold=float(search_cfg["lock_threshold"]),
                unlock_threshold=float(search_cfg["unlock_threshold"]),
            )

            if _should_stop_state(
                state=state,
                delta_stat=delta_stat,
                convergence_delta_e=convergence_delta_e,
                stop_residual_threshold=float(search_cfg["stop_residual_threshold"]),
            ):
                state.completed = True
                completed_states.append(state)
                continue

            shortlist, residuals, locked_tools, cooldown_tools = shortlist_tools(
                delta_stat=delta_stat,
                tool_status=state.tool_status,
                turn=turn,
                shortlist_size=int(search_cfg["shortlist_tools"]),
                max_accept_streak=int(search_cfg["max_accept_streak"]),
                residual_floor=float(search_cfg["residual_floor"]),
            )
            if not shortlist:
                state.completed = True
                completed_states.append(state)
                continue

            current_b64 = encode_image_base64(
                make_thumbnail(state.current_img, thumbnail_size),
                quality=image_quality,
            )
            target_b64 = encode_image_base64(
                make_thumbnail(tgt_img, thumbnail_size),
                quality=image_quality,
            )
            proposals, planner_trace = generate_tool_proposals(
                agent=agent,
                current_img_b64=current_b64,
                target_img_b64=target_b64,
                delta_stat=delta_stat,
                history=state.step_history,
                turn=turn,
                shortlist_tools=shortlist,
                locked_tools=locked_tools,
                cooldown_tools=cooldown_tools,
                current_metrics=state.metrics,
                current_score=state.score,
                max_proposals=int(search_cfg["max_proposals"]),
            )
            search_meta["planner_calls"] += 1
            working_tool_status = state.tool_status

            parsed_planner = planner_trace.get("parsed") or {}
            if parsed_planner.get("should_stop"):
                state.completed = True
                completed_states.append(state)
                continue

            for proposal in proposals:
                candidate = probe_and_refine(
                    state=state,
                    proposal=proposal,
                    delta_stat=delta_stat,
                    src_probe_img=src_probe_img,
                    tgt_probe_img=tgt_probe_img,
                    src_full_img=src_img,
                    tgt_full_img=tgt_img,
                    metrics_size=metrics_size,
                    use_lpips=use_lpips,
                    metrics_device=metrics_device,
                    scoring_cfg=scoring_cfg,
                    probe_cfg=probe_cfg,
                )
                if candidate is None:
                    continue

                search_meta["probe_evaluations"] += len(candidate.get("probe_summary", []))
                accepted, reason = should_accept_candidate(
                    current_score=state.score,
                    candidate_score=float(candidate["score"]),
                    current_metrics=state.metrics,
                    candidate_metrics=candidate["metrics"],
                    accept_margin=float(search_cfg["accept_margin"]),
                    hard_delta_e_tolerance=float(search_cfg["hard_delta_e_tolerance"]),
                )

                if not accepted:
                    search_meta["rejected_candidates"] += 1
                    working_tool_status = apply_reject(
                        working_tool_status,
                        tool_name=proposal["tool"],
                        turn=turn,
                        reject_limit=int(search_cfg["reject_limit"]),
                        cooldown_steps=int(search_cfg["cooldown_steps"]),
                    )
                    state.debug_trace.append(
                        {
                            "turn": turn,
                            "state_index": state_idx,
                            "proposal": proposal,
                            "reason": reason,
                            "score_before": state.score,
                            "score_after": candidate["score"],
                            "metrics_after": candidate["metrics"],
                        }
                    )
                    continue

                child = clone_state(state)
                child.params = candidate["full_params"]
                child.current_img = candidate["output_image"]
                child.metrics = candidate["metrics"]
                child.score = float(candidate["score"])
                child.tool_status = apply_accept(
                    working_tool_status,
                    tool_name=proposal["tool"],
                    delta_params=candidate["delta_parameters"],
                )

                step_record = {
                    "round": turn,
                    "cot": proposal.get("reason", ""),
                    "tool": proposal["tool"],
                    "parameters": candidate["delta_parameters"],
                    "delta_parameters": candidate["delta_parameters"],
                    "params_accumulated": candidate["params_accumulated"],
                    "params_accumulated_tool": candidate["params_accumulated_tool"],
                    "step_quality": candidate["metrics"],
                    "delta_stat": delta_stat,
                    "proposal": proposal,
                    "probe_summary": candidate["probe_summary"],
                    "score_before": state.score,
                    "score_after": candidate["score"],
                    "accepted": True,
                    "_input_img": state.current_img.copy(),
                    "_output_img": candidate["output_image"].copy(),
                    "_delta_stat_before": delta_stat,
                }
                child.steps.append(step_record)
                child.step_history.append(
                    {
                        "round": turn,
                        "tool": proposal["tool"],
                        "parameters": candidate["delta_parameters"],
                        "delta_parameters": candidate["delta_parameters"],
                        "params_accumulated_tool": candidate["params_accumulated_tool"],
                        "step_quality": candidate["metrics"],
                    }
                )
                child.debug_trace.append(
                    {
                        "turn": turn,
                        "proposal": proposal,
                        "accepted": True,
                        "score_before": state.score,
                        "score_after": candidate["score"],
                    }
                )
                expanded_states.append(child)
                search_meta["accepted_candidates"] += 1
                frontier_stalled = False

        if not expanded_states:
            if frontier_stalled:
                completed_states.extend(beam)
            break

        beam = rank_states_diverse(expanded_states, int(search_cfg["beam_size"]))
        _populate_explanations(agent, beam)

    if not completed_states:
        completed_states = beam

    best_state = min(completed_states, key=lambda state: (state.score, len(state.steps)))
    _populate_explanations(agent, [best_state], force=True)

    if output_dir and save_images:
        step_dir = Path(output_dir) / trajectory_id
        step_dir.mkdir(parents=True, exist_ok=True)
    else:
        step_dir = None

    trajectory_steps = _export_steps(
        best_state.steps,
        step_dir=step_dir,
        image_quality=image_quality,
    )
    trajectory = {
        "id": trajectory_id,
        "source": src_img_path,
        "target": tgt_img_path,
        "initial_quality": initial_quality,
        "steps": trajectory_steps,
        "final_quality": best_state.metrics,
        "num_steps": len(trajectory_steps),
        "search_meta": {
            **search_meta,
            "final_score": round(float(best_state.score), 4),
            "planner_model_role": planner_cfg.get("role", "planner"),
            "beam_size": int(search_cfg["beam_size"]),
        },
    }

    logger.info(
        "[%s] Completed with %s accepted steps -> DeltaE=%.2f, PSNR=%.1f",
        trajectory_id,
        len(trajectory_steps),
        best_state.metrics.get("delta_e", 0.0),
        best_state.metrics.get("psnr", 0.0),
    )
    return trajectory


def _should_stop_state(
    *,
    state: SearchState,
    delta_stat: dict[str, Any],
    convergence_delta_e: float,
    stop_residual_threshold: float,
) -> bool:
    if float(state.metrics.get("delta_e", 0.0)) < convergence_delta_e:
        return True
    residuals = compute_tool_residuals(delta_stat)
    return max(residuals.values()) < stop_residual_threshold


def _populate_explanations(agent: MLLMAgent | None, states: list[SearchState], force: bool = False) -> None:
    """Populate accepted-step rationales after control has already selected the step."""
    if agent is None:
        return

    system_prompt = build_explainer_system_prompt()
    for state in states:
        if not state.steps:
            continue
        last_step = state.steps[-1]
        if last_step.get("_cot_finalized") and not force:
            continue

        try:
            response = agent.call(
                system_prompt,
                [
                    {
                        "role": "user",
                        "content": build_explainer_user_prompt(
                            step_record=last_step,
                            delta_stat=last_step.get("_delta_stat_before", last_step.get("delta_stat", {})),
                            score_before=float(last_step.get("score_before", 0.0)),
                            score_after=float(last_step.get("score_after", 0.0)),
                        ),
                    }
                ],
            )
            cot = parse_thinking(response) or response.strip()
            if cot:
                last_step["cot"] = cot
                state.step_history[-1]["cot"] = cot
        except Exception as exc:
            logger.warning("Explanation call failed: %s", exc)
        last_step["_cot_finalized"] = True


def _export_steps(
    steps: list[dict[str, Any]],
    *,
    step_dir: Path | None,
    image_quality: int,
) -> list[dict[str, Any]]:
    exported = []
    for step in steps:
        input_image_path = ""
        output_image_path = ""
        if step_dir is not None:
            input_image_path = _save_step_image(
                step["_input_img"],
                step_dir,
                f"step_{step['round']}_input",
                image_quality,
            )
            output_image_path = _save_step_image(
                step["_output_img"],
                step_dir,
                f"step_{step['round']}_output",
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
                "probe_summary": step["probe_summary"],
                "score_before": step["score_before"],
                "score_after": step["score_after"],
                "accepted": True,
            }
        )
    return exported


def _save_step_image(img: np.ndarray, step_dir: Path, name: str, quality: int) -> str:
    path = step_dir / f"{name}.jpg"
    save_image(path, img, quality=quality)
    return str(path)
