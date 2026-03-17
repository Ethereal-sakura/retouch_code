"""Quality filter for generated trajectories.

Filters trajectories to retain only high-quality, monotonically improving
training examples. See plan Section VI for the filtering criteria.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def filter_trajectory(traj: dict, cfg: dict | None = None) -> tuple[bool, str]:
    """Determine whether a trajectory meets quality criteria for training data.

    Parameters
    ----------
    traj : dict
        Trajectory record as produced by generate_trajectory().
    cfg : dict, optional
        Filter configuration (keys match pipeline.yaml filter section).
        Falls back to defaults if not provided.

    Returns
    -------
    (passed: bool, reason: str)
        passed=True means the trajectory is retained; reason describes why it
        passed or which criterion it failed.
    """
    cfg = cfg or {}
    max_final_de = float(cfg.get("max_final_delta_e", 10.0))
    min_final_psnr = float(cfg.get("min_final_psnr", 20.0))
    min_improvement = float(cfg.get("min_improvement", 2.0))
    max_regression = float(cfg.get("max_regression_ratio", 1.3))
    max_repeats = int(cfg.get("max_tool_repeats", 2))
    min_steps = int(cfg.get("min_steps", 3))
    max_steps = int(cfg.get("max_steps", 8))
    allow_short_below = float(cfg.get("allow_short_if_initial_delta_e_below", 0.0))

    steps = traj.get("steps", [])
    initial_q = traj.get("initial_quality", {})
    final_q = traj.get("final_quality", {})

    # 1. Final quality thresholds
    final_de = final_q.get("delta_e", float("inf"))
    final_psnr = final_q.get("psnr", 0.0)

    if final_de > max_final_de:
        return False, f"final DeltaE={final_de:.2f} > {max_final_de}"
    if final_psnr < min_final_psnr:
        return False, f"final PSNR={final_psnr:.1f} < {min_final_psnr}"

    # 2. Improvement over initial
    initial_de = initial_q.get("delta_e", 0.0)
    improvement = initial_de - final_de
    if improvement < min_improvement:
        return False, (
            f"improvement={improvement:.2f} < {min_improvement} "
            f"(initial={initial_de:.2f}, final={final_de:.2f})"
        )

    # 3. Monotonicity check: no step should cause a large quality regression
    prev_de = initial_de
    for step in steps:
        cur_de = step.get("step_quality", {}).get("delta_e", float("inf"))
        if cur_de > prev_de * max_regression:
            return False, (
                f"regression at round {step['round']}: "
                f"DeltaE {prev_de:.2f} → {cur_de:.2f} (ratio={cur_de/max(prev_de,1e-6):.2f})"
            )
        prev_de = cur_de

    # 4. Tool diversity: no single tool used more than max_repeats times
    tool_counts = Counter(s["tool"] for s in steps)
    for tool, count in tool_counts.items():
        if count > max_repeats:
            return False, f"tool '{tool}' used {count} times > {max_repeats}"

    # 5. Trajectory length
    n = len(steps)
    if n < min_steps and initial_de > allow_short_below:
        return False, f"too few steps: {n} < {min_steps}"
    if n > max_steps:
        return False, f"too many steps: {n} > {max_steps}"

    return True, "passed"


def filter_and_export(
    trajectories: list[dict],
    output_path: str | Path,
    cfg: dict | None = None,
) -> dict:
    """Filter a list of trajectories and export passing ones to a JSON file.

    Parameters
    ----------
    trajectories : list[dict]
        List of trajectory records.
    output_path : str | Path
        Path to write the filtered training data JSON.
    cfg : dict, optional
        Filter configuration dict.

    Returns
    -------
    dict
        Summary statistics: total, passed, failed, pass_rate.
    """
    passed = []
    fail_reasons: Counter = Counter()

    for traj in trajectories:
        ok, reason = filter_trajectory(traj, cfg)
        if ok:
            passed.append(traj)
        else:
            fail_reasons[reason.split(":")[0].split("=")[0].strip()] += 1
            logger.debug(f"[{traj.get('id', '?')}] filtered out: {reason}")

    total = len(trajectories)
    n_passed = len(passed)
    pass_rate = n_passed / total if total > 0 else 0.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)

    summary = {
        "total": total,
        "passed": n_passed,
        "failed": total - n_passed,
        "pass_rate": round(pass_rate, 3),
        "fail_reasons": dict(fail_reasons),
    }
    logger.info(
        f"Filter complete: {n_passed}/{total} passed ({pass_rate:.1%}). "
        f"Saved to {output_path}"
    )
    return summary
