"""Trajectory generation pipeline: single image-pair main loop.

Design principles:
- Render from original source at each step (no error accumulation)
- Accumulate parameters across turns
- Record full step metadata for training data
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.mllm_agent import MLLMAgent, is_stop, parse_thinking, parse_tool_call
from ..agents.prompts import build_image_content, build_system_prompt, build_user_prompt
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
) -> dict:
    """Generate a retouching trajectory for a single (source, target) image pair.

    Parameters
    ----------
    src_img : np.ndarray
        Source image, float32 [0, 1].
    tgt_img : np.ndarray
        Target image, float32 [0, 1].
    agent : MLLMAgent
        Initialized MLLM agent.
    src_img_path : str
        Path string for metadata recording.
    tgt_img_path : str
        Path string for metadata recording.
    trajectory_id : str, optional
        Unique ID for this trajectory; auto-generated if None.
    output_dir : Path, optional
        Directory for saving intermediate images.
    max_turns : int
        Maximum number of editing steps.
    convergence_delta_e : float
        Early stop if DeltaE falls below this value.
    thumbnail_size : int
        Max dimension for intermediate images sent to MLLM.
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

    Returns
    -------
    dict
        Full trajectory record with steps, quality metrics, and metadata.
    """
    trajectory_id = trajectory_id or str(uuid.uuid4())[:8]
    system_prompt = build_system_prompt()

    if output_dir and save_images:
        step_dir = Path(output_dir) / trajectory_id
        step_dir.mkdir(parents=True, exist_ok=True)
    else:
        step_dir = None

    # Compute initial metrics
    initial_quality = compute_metrics(
        src_img, tgt_img, size=metrics_size,
        use_lpips=use_lpips, device=metrics_device
    )

    trajectory = {
        "id": trajectory_id,
        "source": src_img_path,
        "target": tgt_img_path,
        "initial_quality": initial_quality,
        "steps": [],
        "final_quality": None,
    }

    accumulated_params = make_default_params()
    current_img = src_img.copy()
    conv_history: list[dict] = []  # OpenAI message history
    step_history: list[dict] = []  # Human-readable history for prompt

    # Save initial input image
    if step_dir and save_images:
        _save_step_image(src_img, step_dir, "step_0_input", image_quality)

    for turn in range(max_turns):
        # Early convergence check
        delta_stat = get_delta_stat(current_img, tgt_img)
        current_metrics = compute_metrics(
            current_img, tgt_img, size=metrics_size, use_lpips=False
        )
        if current_metrics["delta_e"] < convergence_delta_e:
            logger.info(
                f"[{trajectory_id}] Turn {turn}: converged "
                f"(DeltaE={current_metrics['delta_e']:.2f} < {convergence_delta_e})"
            )
            break

        # Encode current and target as thumbnails
        thumb_current = make_thumbnail(current_img, thumbnail_size)
        thumb_target = make_thumbnail(tgt_img, thumbnail_size)
        current_b64 = encode_image_base64(thumb_current, quality=image_quality)
        target_b64 = encode_image_base64(thumb_target, quality=image_quality)

        # Build user message
        image_blocks = build_image_content(current_b64, target_b64)
        text_blocks = build_user_prompt(delta_stat, step_history, turn)
        user_content = image_blocks + text_blocks
        user_msg = {"role": "user", "content": user_content}

        # Call MLLM
        try:
            response = agent.call(system_prompt, conv_history + [user_msg])
        except Exception as e:
            logger.error(f"[{trajectory_id}] Turn {turn}: API call failed: {e}")
            break

        # Update conversation history
        conv_history.append(user_msg)
        conv_history.append({"role": "assistant", "content": response})

        # Parse response
        cot = parse_thinking(response)

        if is_stop(response):
            logger.info(f"[{trajectory_id}] Turn {turn}: model issued <stop>")
            break

        tool_name, tool_params = parse_tool_call(response)
        if tool_name is None or tool_params is None:
            logger.warning(
                f"[{trajectory_id}] Turn {turn}: could not parse tool call from response"
            )
            logger.debug(f"Response was:\n{response}")
            break

        # Validate parameters
        ok, err = validate_tool_call(tool_name, tool_params)
        if not ok:
            logger.warning(
                f"[{trajectory_id}] Turn {turn}: invalid tool call ({err}). Skipping."
            )
            break

        # Apply tool: merge into accumulated params, render from source
        accumulated_params = merge_tool_call(accumulated_params, tool_name, tool_params)
        new_img = render(src_img, accumulated_params)

        # Compute step quality
        step_quality = compute_metrics(
            new_img, tgt_img, size=metrics_size,
            use_lpips=use_lpips, device=metrics_device
        )

        # Save intermediate output image
        output_image_path = ""
        input_image_path = ""
        if step_dir and save_images:
            input_image_path = _save_step_image(
                current_img, step_dir, f"step_{turn}_input", image_quality
            )
            output_image_path = _save_step_image(
                new_img, step_dir, f"step_{turn}_output", image_quality
            )

        # Record step
        step_record = {
            "round": turn,
            "input_image": input_image_path,
            "cot": cot,
            "tool": tool_name,
            "parameters": tool_params,
            "params_accumulated": params_to_dict(accumulated_params),
            "output_image": output_image_path,
            "step_quality": step_quality,
            "delta_stat": delta_stat,
        }
        trajectory["steps"].append(step_record)

        # Update step history for next-turn prompt
        step_history.append({
            "round": turn,
            "tool": tool_name,
            "parameters": tool_params,
            "step_quality": step_quality,
        })

        logger.info(
            f"[{trajectory_id}] Turn {turn}: {tool_name} → "
            f"DeltaE={step_quality['delta_e']:.2f}, PSNR={step_quality['psnr']:.1f}"
        )

        current_img = new_img

    # Final quality
    trajectory["final_quality"] = compute_metrics(
        current_img, tgt_img, size=metrics_size,
        use_lpips=use_lpips, device=metrics_device
    )
    trajectory["num_steps"] = len(trajectory["steps"])

    return trajectory


def _save_step_image(img: np.ndarray, step_dir: Path, name: str, quality: int) -> str:
    """Save an intermediate step image and return its path string."""
    path = step_dir / f"{name}.jpg"
    save_image(path, img, quality=quality)
    return str(path)
