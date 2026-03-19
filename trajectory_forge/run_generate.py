#!/usr/bin/env python3
"""Batch trajectory generation entry point.

Usage:
    python run_generate.py [--config config/pipeline.yaml] [--pairs data/mit5k_pairs.json]
                           [--output trajectories/] [--max-samples 10] [--api-key KEY]

Example (single test run):
    python run_generate.py --max-samples 1 --output trajectories/test/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Allow running from trajectory_forge/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_forge.agents.mllm_agent import MLLMAgent
from trajectory_forge.pipeline.trajectory_generator import generate_trajectory
from trajectory_forge.utils.image_utils import load_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate retouching trajectories")
    p.add_argument(
        "--config", default="config/pipeline.yaml",
        help="Pipeline config YAML (default: config/pipeline.yaml)"
    )
    p.add_argument(
        "--pairs", default=None,
        help="Override pairs JSON file from config"
    )
    p.add_argument(
        "--output", default=None,
        help="Override output directory from config"
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max_samples from config"
    )
    p.add_argument(
        "--start-idx", type=int, default=None,
        help="Override start_idx from config"
    )
    p.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)"
    )
    p.add_argument(
        "--model", default=None,
        help="Override model name from config"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Load config and pairs but do not call the API"
    )
    return p.parse_args()


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        # Use defaults if config file not found
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def make_trajectory_id(pair: dict, idx: int) -> str:
    src = Path(pair.get("source", f"img_{idx}")).stem
    return f"{src}_{idx:04d}"


def make_brief_trajectory(traj: dict) -> dict:
    """Build a compact trajectory record for training-data collection."""
    return {
        "id": traj.get("id", ""),
        "source": traj.get("source", ""),
        "target": traj.get("target", ""),
        "initial_quality": traj.get("initial_quality", {}),
        "steps": [
            {
                "round": step.get("round"),
                "input_image": step.get("input_image", ""),
                "output_image": step.get("output_image", ""),
                "tool": step.get("tool", ""),
                "parameters": step.get("parameters", {}),
                "delta_parameters": step.get("delta_parameters", {}),
                "cot": step.get("cot", ""),
                "step_quality": step.get("step_quality", {}),
            }
            for step in traj.get("steps", [])
        ],
    }


def main() -> None:
    args = parse_args()

    # script_dir is used only for config-derived default paths (e.g. dataset.pairs_file).
    # CLI-provided paths (args.config, args.pairs, args.output) are relative to cwd.
    script_dir = Path(__file__).parent
    config_path = Path(args.config)          # relative to cwd, as the user typed it
    cfg = load_config(str(config_path))
    setup_logging(cfg)
    logger = logging.getLogger("run_generate")

    # Extract config sections
    api_cfg = cfg.get("api", {})
    gen_cfg = cfg.get("generation", {})
    dataset_cfg = cfg.get("dataset", {})
    metrics_cfg = cfg.get("metrics", {})
    scoring_cfg = cfg.get("scoring", {})
    planner_cfg = cfg.get("planner", {})
    mcts_cfg = cfg.get("mcts", {})
    debug_cfg = cfg.get("debug", {})

    # Resolve parameters (CLI overrides config)
    pairs_file = args.pairs or (script_dir / dataset_cfg.get("pairs_file", "data/mit5k_pairs.json"))
    output_dir = Path(args.output or (script_dir / gen_cfg.get("output_dir", "trajectories")))
    max_samples = args.max_samples if args.max_samples is not None else dataset_cfg.get("max_samples")
    start_idx = args.start_idx if args.start_idx is not None else int(dataset_cfg.get("start_idx", 0))
    model = args.model or api_cfg.get("model", "gpt-4o")
    api_key = args.api_key or os.environ.get(api_cfg.get("api_key_env", "OPENAI_API_KEY"))
    base_url = api_cfg.get("base_url") or None

    # Load pairs
    pairs_file = Path(pairs_file)
    if not pairs_file.exists():
        logger.error(f"Pairs file not found: {pairs_file}")
        sys.exit(1)
    with open(pairs_file, "r") as f:
        pairs = json.load(f)

    # Slice pairs
    pairs = pairs[start_idx:]
    if max_samples is not None:
        pairs = pairs[:max_samples]
    logger.info(f"Processing {len(pairs)} image pairs from {pairs_file}")

    if args.dry_run:
        logger.info("Dry run mode: exiting without generating trajectories.")
        for i, pair in enumerate(pairs[:5]):
            logger.info(f"  [{i}] {pair.get('source')} → {pair.get('target')}")
        return

    # Initialize agent
    agent = MLLMAgent(
        model=model,
        api_key=api_key,
        base_url=base_url if base_url != "https://api.openai.com/v1" else None,
        max_tokens=int(api_cfg.get("max_tokens", 1024)),
        temperature=float(api_cfg.get("temperature", 0.2)),
        request_timeout=int(api_cfg.get("request_timeout", 60)),
    )

    # Generation loop
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "trajectories_raw.json"
    brief_results_file = output_dir / "trajectories_brief.json"
    best_paths_file = output_dir / "best_paths.json"
    all_trajectories = []
    brief_trajectories = []
    best_paths = []

    log_every_n = cfg.get("logging", {}).get("log_every_n", 10)

    for idx, pair in enumerate(pairs):
        src_path = pair.get("source", "")
        tgt_path = pair.get("target", "")
        traj_id = make_trajectory_id(pair, start_idx + idx)

        if idx % log_every_n == 0:
            logger.info(f"Progress: {idx}/{len(pairs)} — {traj_id}")

        try:
            src_img = load_image(src_path)
            tgt_img = load_image(tgt_path)
        except Exception as e:
            logger.error(f"[{traj_id}] Failed to load images: {e}")
            continue

        try:
            traj = generate_trajectory(
                src_img=src_img,
                tgt_img=tgt_img,
                agent=agent,
                src_img_path=src_path,
                tgt_img_path=tgt_path,
                trajectory_id=traj_id,
                output_dir=output_dir,
                max_turns=int(gen_cfg.get("max_turns", 8)),
                convergence_delta_e=float(gen_cfg.get("convergence_delta_e", 4.0)),
                thumbnail_size=int(gen_cfg.get("thumbnail_size", [512, 512])[0]),
                image_quality=int(gen_cfg.get("image_quality", 85)),
                metrics_size=tuple(metrics_cfg.get("eval_size", [64, 64])),
                use_lpips=bool(metrics_cfg.get("use_lpips", False)),
                metrics_device=str(metrics_cfg.get("device", "cpu")),
                save_images=bool(gen_cfg.get("save_intermediate_images", True)),
                planner_cfg=planner_cfg,
                scoring_cfg=scoring_cfg,
                mcts_cfg=mcts_cfg,
                debug_cfg=debug_cfg,
            )
            all_trajectories.append(traj)
            brief_trajectories.append(make_brief_trajectory(traj))
            best_paths.append(
                {
                    "id": traj.get("id", ""),
                    "source": traj.get("source", ""),
                    "target": traj.get("target", ""),
                    "initial_quality": traj.get("initial_quality", {}),
                    "final_quality": traj.get("final_quality", {}),
                    "num_steps": traj.get("num_steps", 0),
                    "steps": traj.get("steps", []),
                    "artifacts": traj.get("artifacts", {}),
                    "search_meta": traj.get("search_meta", {}),
                }
            )
        except Exception as e:
            logger.error(f"[{traj_id}] Trajectory generation failed: {e}", exc_info=True)
            continue

        # Append to results file incrementally (crash-safe)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
        with open(brief_results_file, "w", encoding="utf-8") as f:
            json.dump(brief_trajectories, f, ensure_ascii=False, indent=2)
        with open(best_paths_file, "w", encoding="utf-8") as f:
            json.dump(best_paths, f, ensure_ascii=False, indent=2)

    logger.info(
        "Generation complete: %s/%s trajectories saved to %s, %s, and %s",
        len(all_trajectories),
        len(pairs),
        results_file,
        brief_results_file,
        best_paths_file,
    )


if __name__ == "__main__":
    main()
