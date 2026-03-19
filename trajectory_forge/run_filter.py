#!/usr/bin/env python3
"""Quality filtering and training data export entry point.

Usage:
    python run_filter.py [--input trajectories/trajectories_raw.json]
                         [--output trajectories/training_data.json]
                         [--config config/pipeline.yaml]

The filtered output is a JSON array of trajectories that passed all
quality criteria, ready for use as training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from trajectory_forge/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_forge.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter generated trajectories for training data")
    p.add_argument(
        "--input", default="trajectories/trajectories_raw.json",
        help="Input trajectories JSON (output of run_generate.py)"
    )
    p.add_argument(
        "--output", default="trajectories/training_data.json",
        help="Output filtered training data JSON"
    )
    p.add_argument(
        "--config", default="config/pipeline.yaml",
        help="Pipeline config YAML"
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print detailed statistics about the filtered trajectories"
    )
    return p.parse_args()



def print_statistics(trajectories: list[dict]) -> None:
    """Print detailed statistics about filtered trajectories."""
    import statistics
    from collections import Counter

    if not trajectories:
        print("No trajectories to analyze.")
        return

    step_counts = [t.get("num_steps", len(t.get("steps", []))) for t in trajectories]
    final_des = [t["final_quality"]["delta_e"] for t in trajectories if "final_quality" in t]
    initial_des = [t["initial_quality"]["delta_e"] for t in trajectories if "initial_quality" in t]
    improvements = [i - f for i, f in zip(initial_des, final_des)]

    tool_usage: Counter = Counter()
    for traj in trajectories:
        for step in traj.get("steps", []):
            tool_usage[step["tool"]] += 1

    print("\n=== Trajectory Statistics ===")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"\nStep counts:")
    print(f"  Mean:   {statistics.mean(step_counts):.1f}")
    print(f"  Median: {statistics.median(step_counts):.1f}")
    print(f"  Min:    {min(step_counts)}")
    print(f"  Max:    {max(step_counts)}")

    if final_des:
        print(f"\nFinal DeltaE:")
        print(f"  Mean:   {statistics.mean(final_des):.2f}")
        print(f"  Median: {statistics.median(final_des):.2f}")
        print(f"  Min:    {min(final_des):.2f}")
        print(f"  Max:    {max(final_des):.2f}")

    if improvements:
        print(f"\nDeltaE improvement:")
        print(f"  Mean:   {statistics.mean(improvements):.2f}")
        print(f"  Median: {statistics.median(improvements):.2f}")

    print(f"\nTool usage frequency:")
    total_steps = sum(tool_usage.values())
    for tool, count in tool_usage.most_common():
        pct = count / total_steps * 100 if total_steps > 0 else 0
        print(f"  {tool}: {count} ({pct:.1f}%)")


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent))

    from trajectory_forge.pipeline.quality_filter import filter_and_export

    # CLI-provided paths (args.config, args.input, args.output) are relative to cwd.
    # script_dir is only used for internal fallbacks if needed in the future.
    config_path = Path(args.config)          # relative to cwd
    cfg = load_config(str(config_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("run_filter")

    # Paths are relative to cwd (where the user runs the command)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading trajectories from {input_path}")
    with open(input_path, "r") as f:
        trajectories = json.load(f)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Filter and export
    filter_cfg = cfg.get("filter", {})
    summary = filter_and_export(trajectories, output_path, filter_cfg)

    print(f"\n=== Filter Summary ===")
    print(f"Total:     {summary['total']}")
    print(f"Passed:    {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed:    {summary['failed']}")
    if summary["fail_reasons"]:
        print("Fail reasons:")
        for reason, count in sorted(summary["fail_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"\nTraining data saved to: {output_path}")

    # Optional detailed stats on passing trajectories
    if args.stats and summary["passed"] > 0:
        with open(output_path, "r") as f:
            filtered = json.load(f)
        print_statistics(filtered)


if __name__ == "__main__":
    main()
