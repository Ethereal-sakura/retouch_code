"""Trajectory generation pipeline: single image-pair main loop.

Design principles:
- Render from original source at each step (no error accumulation)
- Accumulate parameters across turns
- Multi-candidate generation with DeltaE-best selection
- Quality-gated rollback prevents regression (no-improvement turns skipped)
- Oscillation detection prevents parameter cycling
"""

from __future__ import annotations

import copy
import logging
import uuid
from pathlib import Path

import numpy as np

from ..agents.mllm_agent import (
    MLLMAgent,
    is_stop,
    parse_multi_tool_calls,
)
from ..agents.prompts import (
    build_image_content,
    build_multi_candidate_system_prompt,
    build_user_prompt,
)
from ..tools.image_engine_adapter import (
    make_default_params,
    merge_tool_call,
    params_to_dict,
    render,
)
from ..tools.tool_registry import validate_tool_call
from ..utils.image_utils import encode_image_base64, make_thumbnail, save_image
from ..utils.metrics import compute_metrics
from ..utils.stat_utils import get_delta_stat

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
    num_candidates: int = 3,
    max_rollbacks: int = 3,
    oscillation_window: int = 3,
) -> dict:
    """Generate a retouching trajectory for a single (source, target) image pair.

    Uses multi-candidate generation with DeltaE-best selection, quality-gated
    rollback, and oscillation detection for robust trajectory quality.

    Each turn the VLM proposes num_candidates diverse strategies; all are
    rendered and evaluated, then the one with the lowest DeltaE is adopted.
    If no candidate improves on the current state, the turn is rolled back.

    Parameters
    ----------
    src_img : np.ndarray
        Source image, float32 [0, 1].
    tgt_img : np.ndarray
        Target image, float32 [0, 1].
    agent : MLLMAgent
        Initialized MLLM agent.
    src_img_path, tgt_img_path : str
        Path strings for metadata recording.
    trajectory_id : str, optional
        Unique ID; auto-generated if None.
    output_dir : Path, optional
        Directory for saving intermediate images.
    max_turns : int
        Maximum number of editing rounds.
    convergence_delta_e : float
        Early stop if DeltaE falls below this value.
    thumbnail_size : int
        Max dimension for images sent to MLLM.
    image_quality : int
        JPEG quality for saved images.
    metrics_size : tuple[int, int]
        Image resize for metrics computation.
    use_lpips : bool
        Whether to compute LPIPS (slow on CPU).
    metrics_device : str
        Device for metrics computation.
    save_images : bool
        Whether to save intermediate images to disk.
    num_candidates : int
        Number of candidate strategies per turn.
    max_rollbacks : int
        Stop after this many consecutive no-improvement turns.
    oscillation_window : int
        Number of recent adopted steps to check for oscillation.

    Returns
    -------
    dict
        Full trajectory record with steps, quality metrics, and metadata.
    """
    trajectory_id = trajectory_id or str(uuid.uuid4())[:8]
    system_prompt = build_multi_candidate_system_prompt(num_candidates)

    if output_dir and save_images:
        step_dir = Path(output_dir) / trajectory_id
        step_dir.mkdir(parents=True, exist_ok=True)
    else:
        step_dir = None

    initial_quality = compute_metrics(
        src_img, tgt_img, size=metrics_size,
        use_lpips=use_lpips, device=metrics_device,
    )

    trajectory = {
        "id": trajectory_id,
        "source": src_img_path,
        "target": tgt_img_path,
        "parameter_semantics": "delta_per_turn_accumulated_to_absolute_state",
        "initial_quality": initial_quality,
        "steps": [],
        "final_quality": None,
    }

    accumulated_params = make_default_params()
    current_img = src_img.copy()
    step_history: list[dict] = []
    locked_tools: set[str] = set()
    rollback_count = 0
    prev_delta_e: float | None = None
    last_was_rollback = False

    for turn in range(max_turns):
        # ── Convergence check ─────────────────────────────────────────────
        delta_stat = get_delta_stat(current_img, tgt_img)
        current_metrics = compute_metrics(
            current_img, tgt_img, size=metrics_size, use_lpips=False,
        )
        current_delta_e = current_metrics["delta_e"]

        if current_delta_e < convergence_delta_e:
            logger.info(
                f"[{trajectory_id}] Turn {turn}: converged "
                f"(DeltaE={current_delta_e:.2f} < {convergence_delta_e})"
            )
            break

        # ── Encode thumbnails ─────────────────────────────────────────────
        thumb_current = make_thumbnail(current_img, thumbnail_size)
        thumb_target = make_thumbnail(tgt_img, thumbnail_size)
        current_b64 = encode_image_base64(thumb_current, quality=image_quality)
        target_b64 = encode_image_base64(thumb_target, quality=image_quality)

        # ── Build multi-candidate prompt ──────────────────────────────────
        image_blocks = build_image_content(current_b64, target_b64)
        current_params_dict = params_to_dict(accumulated_params)
        text_blocks = build_user_prompt(
            delta_stat, step_history, current_params_dict, turn,
            locked_tools=locked_tools,
            prev_delta_e=prev_delta_e,
            current_delta_e=current_delta_e,
            was_rollback=last_was_rollback,
            num_candidates=num_candidates,
        )
        user_content = image_blocks + text_blocks
        user_msg = {"role": "user", "content": user_content}

        # ── Call VLM for candidates ───────────────────────────────────────
        try:
            response = agent.call(system_prompt, [user_msg])
        except Exception as e:
            logger.error(f"[{trajectory_id}] Turn {turn}: API call failed: {e}")
            break

        candidates_raw = parse_multi_tool_calls(response, num_candidates)

        if not candidates_raw:
            if is_stop(response):
                logger.info(f"[{trajectory_id}] Turn {turn}: model issued <stop>")
            else:
                logger.warning(
                    f"[{trajectory_id}] Turn {turn}: "
                    "no candidates parsed from response"
                )
            break

        # ── Validate, render, evaluate each candidate ─────────────────────
        valid_candidates = _build_valid_candidates(
            candidates_raw, accumulated_params, locked_tools,
            src_img, tgt_img, metrics_size,
        )

        if not valid_candidates:
            logger.warning(
                f"[{trajectory_id}] Turn {turn}: "
                "no valid candidates after filtering"
            )
            break

        # ── Select best candidate by DeltaE (quality-gated) ─────────────
        candidates_summary = [
            {
                "tool": vc["tool_name"],
                "parameters": vc["tool_params"],
                "quality": vc["trial_quality"],
                "cot": vc["cot"],
            }
            for vc in valid_candidates
        ]

        best_idx = _pick_best_delta_e(valid_candidates)   # 1-based
        best_candidate = valid_candidates[best_idx - 1]
        best_delta_e = best_candidate["trial_quality"]["delta_e"]

        # ── Quality gate: rollback if no candidate improves current state ─
        if best_delta_e >= current_delta_e:
            rollback_count += 1
            last_was_rollback = True

            logger.info(
                f"[{trajectory_id}] Turn {turn}: ROLLBACK "
                f"(best DeltaE={best_delta_e:.2f} >= current={current_delta_e:.2f}) "
                f"({rollback_count}/{max_rollbacks})"
            )

            trajectory["steps"].append({
                "round": turn,
                "action": "rollback",
                "candidates": candidates_summary,
                "best_candidate_idx": best_idx,
                "delta_stat": delta_stat,
                "current_quality": current_metrics,
            })

            if rollback_count >= max_rollbacks:
                logger.info(
                    f"[{trajectory_id}] Turn {turn}: max rollbacks reached"
                )
                break

            prev_delta_e = current_delta_e
            continue

        # ── Adopt the best candidate ──────────────────────────────────────
        rollback_count = 0
        last_was_rollback = False
        chosen = best_candidate
        chosen_idx = best_idx

        params_before_step = current_params_dict
        accumulated_params = chosen["trial_params"]
        new_img = chosen["trial_img"]

        if use_lpips:
            step_quality = compute_metrics(
                new_img, tgt_img, size=metrics_size,
                use_lpips=True, device=metrics_device,
            )
        else:
            step_quality = chosen["trial_quality"]

        output_image_path = ""
        input_image_path = ""
        if step_dir and save_images:
            input_image_path = _save_step_image(
                current_img, step_dir, f"step_{turn}_input", image_quality,
            )
            output_image_path = _save_step_image(
                new_img, step_dir, f"step_{turn}_output", image_quality,
            )

        step_record = {
            "round": turn,
            "action": "adopt",
            "input_image": input_image_path,
            "cot": chosen["cot"],
            "tool": chosen["tool_name"],
            "parameters": chosen["tool_params"],
            "params_before_step": params_before_step,
            "params_accumulated": params_to_dict(accumulated_params),
            "output_image": output_image_path,
            "step_quality": step_quality,
            "delta_stat": delta_stat,
            "selected_candidate_idx": chosen_idx,
            "candidates": candidates_summary,
        }
        trajectory["steps"].append(step_record)

        step_history.append({
            "round": turn,
            "tool": chosen["tool_name"],
            "parameters": chosen["tool_params"],
            "params_accumulated": step_record["params_accumulated"],
            "step_quality": step_quality,
        })

        logger.info(
            f"[{trajectory_id}] Turn {turn}: adopted #{chosen_idx} "
            f"({chosen['tool_name']}) -> "
            f"DeltaE={step_quality['delta_e']:.2f}, "
            f"PSNR={step_quality['psnr']:.1f}"
        )

        # ── Oscillation detection ─────────────────────────────────────────
        newly_locked = _detect_and_lock_oscillations(
            step_history, locked_tools, oscillation_window,
        )
        if newly_locked:
            logger.info(f"  Locked tools due to oscillation: {newly_locked}")

        prev_delta_e = current_delta_e
        current_img = new_img

    # ── Final quality ─────────────────────────────────────────────────────────
    trajectory["final_quality"] = compute_metrics(
        current_img, tgt_img, size=metrics_size,
        use_lpips=use_lpips, device=metrics_device,
    )
    trajectory["num_steps"] = len(trajectory["steps"])

    return trajectory


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_valid_candidates(
    candidates_raw: list[dict],
    accumulated_params,
    locked_tools: set[str],
    src_img: np.ndarray,
    tgt_img: np.ndarray,
    metrics_size: tuple[int, int],
) -> list[dict]:
    """Validate, merge, render, and evaluate each raw candidate."""
    valid = []
    for i, cand in enumerate(candidates_raw):
        tool_name = cand["tool_name"]
        tool_params = cand["tool_params"]

        if tool_name is None or tool_params is None:
            logger.debug(f"  Candidate {i+1}: skipped (parse failed)")
            continue

        if tool_name in locked_tools:
            logger.debug(
                f"  Candidate {i+1}: skipped (locked tool {tool_name})"
            )
            continue

        ok, err = validate_tool_call(tool_name, tool_params)
        if not ok:
            logger.debug(f"  Candidate {i+1}: invalid ({err})")
            continue

        trial_params = merge_tool_call(
            copy.deepcopy(accumulated_params), tool_name, tool_params,
        )
        trial_img = render(src_img, trial_params)
        trial_quality = compute_metrics(
            trial_img, tgt_img, size=metrics_size, use_lpips=False,
        )

        valid.append({
            "index": i,
            "cot": cand["cot"],
            "tool_name": tool_name,
            "tool_params": tool_params,
            "trial_params": trial_params,
            "trial_img": trial_img,
            "trial_quality": trial_quality,
        })

    return valid


def _pick_best_delta_e(candidates: list[dict]) -> int:
    """Return 1-based index of candidate with lowest DeltaE (fallback selector)."""
    best_i, best_de = 0, float("inf")
    for i, c in enumerate(candidates):
        de = c["trial_quality"]["delta_e"]
        if de < best_de:
            best_i, best_de = i, de
    return best_i + 1


def _detect_and_lock_oscillations(
    history: list[dict],
    locked_tools: set[str],
    window: int,
) -> set[str]:
    """Check recent adopted steps for parameter oscillation; lock affected tools.

    Oscillation is detected when the same tool's parameter deltas alternate in
    sign (e.g., brightness +20, then -18, then +16) within the recent window.
    """
    newly_locked: set[str] = set()
    recent = history[-window:]

    tool_params_seq: dict[str, list[dict]] = {}
    for step in recent:
        t = step["tool"]
        tool_params_seq.setdefault(t, []).append(step["parameters"])

    for tool_name, param_list in tool_params_seq.items():
        if tool_name in locked_tools or len(param_list) < 2:
            continue
        if tool_name == "hsl_tool":
            continue

        all_keys: set[str] = set()
        for p in param_list:
            all_keys.update(p.keys())

        for key in all_keys:
            vals = [p.get(key, 0.0) for p in param_list]
            signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
            nonzero = [s for s in signs if s != 0]
            if len(nonzero) >= 2 and any(
                nonzero[i] != nonzero[i + 1] for i in range(len(nonzero) - 1)
            ):
                locked_tools.add(tool_name)
                newly_locked.add(tool_name)
                break

    return newly_locked


def _save_step_image(
    img: np.ndarray, step_dir: Path, name: str, quality: int,
) -> str:
    """Save an intermediate step image and return its path string."""
    path = step_dir / f"{name}.jpg"
    save_image(path, img, quality=quality)
    return str(path)
