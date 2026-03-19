from __future__ import annotations

import unittest

import numpy as np

from trajectory_forge.pipeline.quantization import quantize_tool_delta
from trajectory_forge.pipeline.trajectory_generator import generate_trajectory
from trajectory_forge.run_generate import make_brief_trajectory
from trajectory_forge.tools.image_engine_adapter import make_default_params, merge_tool_call


class FakeAgent:
    def __init__(self, planner_responses: list[str]):
        self._planner_responses = list(planner_responses)

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        if "professional photo retouching planner" in system_prompt:
            if self._planner_responses:
                return self._planner_responses.pop(0)
            return '{"should_stop": true, "main_issue": "none", "proposals": []}'
        if "writing short training rationales" in system_prompt:
            return "<thinking>Accepted because it improves the remaining exposure mismatch.</thinking>"
        raise AssertionError(f"Unexpected prompt: {system_prompt[:80]}")


class SearchPipelineTests(unittest.TestCase):
    def test_quantize_tool_delta_rounds_half_away_from_zero(self) -> None:
        quantized = quantize_tool_delta(
            "exposure_tool",
            {"exposure": -2.5, "brightness": 1.49},
        )

        self.assertEqual(quantized["exposure"], -3)
        self.assertEqual(quantized["brightness"], 1)

    def test_merge_tool_call_accumulates_numeric_deltas(self) -> None:
        params = make_default_params()
        params = merge_tool_call(params, "exposure_tool", {"exposure": 5, "brightness": 2})
        params = merge_tool_call(params, "exposure_tool", {"exposure": 3, "brightness": -1})

        self.assertAlmostEqual(params.exposure, 8.0)
        self.assertAlmostEqual(params.brightness, 1.0)

    def test_generate_trajectory_rejects_regression_candidates(self) -> None:
        src = np.full((48, 48, 3), 0.50, dtype=np.float32)
        tgt = np.full((48, 48, 3), 0.35, dtype=np.float32)
        agent = FakeAgent(
            [
                (
                    '{"should_stop": false, "main_issue": "exposure", "proposals": '
                    '[{"tool": "exposure_tool", "direction": "increase", '
                    '"magnitude_bucket": "small", "reason": "Wrong direction on purpose."}]}'
                )
            ]
        )

        trajectory = generate_trajectory(
            src,
            tgt,
            agent,  # type: ignore[arg-type]
            max_turns=2,
            metrics_size=(32, 32),
            use_lpips=False,
            search_cfg={
                "beam_size": 2,
                "shortlist_tools": 1,
                "max_proposals": 1,
                "accept_margin": 0.01,
                "hard_delta_e_tolerance": 0.01,
            },
            probe_cfg={"render_size": 64, "refine_steps": [2.0]},
        )

        self.assertEqual(trajectory["num_steps"], 0)
        self.assertAlmostEqual(
            trajectory["final_quality"]["delta_e"],
            trajectory["initial_quality"]["delta_e"],
            places=4,
        )

    def test_generate_trajectory_accepts_improving_candidate(self) -> None:
        src = np.full((48, 48, 3), 0.35, dtype=np.float32)
        tgt = np.full((48, 48, 3), 0.50, dtype=np.float32)
        agent = FakeAgent(
            [
                (
                    '{"should_stop": false, "main_issue": "exposure", "proposals": '
                    '[{"tool": "exposure_tool", "direction": "increase", '
                    '"magnitude_bucket": "small", "reason": "Raise the exposure slightly."}]}'
                ),
                '{"should_stop": true, "main_issue": "none", "proposals": []}',
            ]
        )

        trajectory = generate_trajectory(
            src,
            tgt,
            agent,  # type: ignore[arg-type]
            max_turns=3,
            metrics_size=(32, 32),
            use_lpips=False,
            convergence_delta_e=0.5,
            search_cfg={
                "beam_size": 2,
                "shortlist_tools": 1,
                "max_proposals": 1,
                "accept_margin": 0.01,
                "hard_delta_e_tolerance": 0.2,
            },
            probe_cfg={"render_size": 64, "refine_steps": [2.0]},
        )

        self.assertGreaterEqual(trajectory["num_steps"], 1)
        self.assertLess(
            trajectory["final_quality"]["delta_e"],
            trajectory["initial_quality"]["delta_e"],
        )
        self.assertTrue(trajectory["steps"][0]["accepted"])
        self.assertIn("delta_parameters", trajectory["steps"][0])
        self.assertTrue(
            all(isinstance(value, int) for value in trajectory["steps"][0]["delta_parameters"].values())
        )

        brief = make_brief_trajectory(trajectory)
        self.assertEqual(
            brief["steps"][0]["parameters"],
            trajectory["steps"][0]["delta_parameters"],
        )


if __name__ == "__main__":
    unittest.main()
