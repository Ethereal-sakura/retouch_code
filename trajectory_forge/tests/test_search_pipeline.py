from __future__ import annotations

import unittest

import numpy as np

from trajectory_forge.agents.mllm_agent import parse_planner_response
from trajectory_forge.pipeline.mcts_search import normalize_action
from trajectory_forge.pipeline.trajectory_generator import generate_trajectory
from trajectory_forge.tools.image_engine_adapter import make_default_params, merge_tool_call


class FakeAgent:
    def __init__(
        self,
        *,
        mcts_responses: list[str] | None = None,
    ):
        self._mcts_responses = list(mcts_responses or [])

    def call(self, system_prompt: str, messages: list[dict], temperature: float | None = None) -> str:
        if "long-horizon search" in system_prompt:
            if self._mcts_responses:
                return self._mcts_responses.pop(0)
            return '{"should_stop": true, "main_issue": "none", "candidates": []}'
        if "writing short training rationales" in system_prompt:
            return "<thinking>Accepted because it improves the remaining exposure mismatch.</thinking>"
        raise AssertionError(f"Unexpected prompt: {system_prompt[:80]}")


class SearchPipelineTests(unittest.TestCase):
    def test_merge_tool_call_accumulates_numeric_deltas(self) -> None:
        params = make_default_params()
        params = merge_tool_call(params, "exposure_tool", {"exposure": 5, "brightness": 2})
        params = merge_tool_call(params, "exposure_tool", {"exposure": 3, "brightness": -1})

        self.assertAlmostEqual(params.exposure, 8.0)
        self.assertAlmostEqual(params.brightness, 1.0)

    def test_normalize_action_rounds_execute_values(self) -> None:
        normalized = normalize_action(
            "exposure_tool",
            {"exposure": 7.6, "brightness": "2.2", "ignored": 5},
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(
            normalized["delta_parameters"],
            {"exposure": 8, "brightness": 2},
        )

    def test_generate_trajectory_mcts_preserves_raw_parameters(self) -> None:
        src = np.full((48, 48, 3), 0.35, dtype=np.float32)
        tgt = np.full((48, 48, 3), 0.50, dtype=np.float32)
        agent = FakeAgent(
            mcts_responses=[
                (
                    '{"should_stop": false, "main_issue": "exposure", "candidates": '
                    '[{"tool": "exposure_tool", "parameters": {"exposure": 7.6, "brightness": 2.2}, '
                    '"reason": "Raise exposure with an integer-like step."}]}'
                ),
                '{"should_stop": true, "main_issue": "none", "candidates": []}',
            ]
        )

        trajectory = generate_trajectory(
            src,
            tgt,
            agent,  # type: ignore[arg-type]
            max_turns=3,
            metrics_size=(32, 32),
            use_lpips=False,
            planner_cfg={"candidates_per_call": 1, "diversity_calls": 1, "temperatures": [0.4]},
            mcts_cfg={"num_simulations": 6, "max_actions_per_node": 1, "target_chain_len": 2},
        )

        self.assertGreaterEqual(trajectory["num_steps"], 1)
        step = trajectory["steps"][0]
        self.assertEqual(step["parameters"]["exposure"], 7.6)
        self.assertEqual(step["parameters"]["brightness"], 2.2)
        self.assertEqual(step["delta_parameters"]["exposure"], 8)
        self.assertEqual(step["delta_parameters"]["brightness"], 2)
        self.assertEqual(trajectory["search_meta"]["search_mode"], "mcts")
        self.assertIn("mcts_summary", step)
        self.assertTrue(step["accepted"])

    def test_generate_trajectory_mcts_prunes_regression_candidates(self) -> None:
        src = np.full((48, 48, 3), 0.50, dtype=np.float32)
        tgt = np.full((48, 48, 3), 0.35, dtype=np.float32)
        agent = FakeAgent(
            mcts_responses=[
                (
                    '{"should_stop": false, "main_issue": "exposure", "candidates": '
                    '[{"tool": "exposure_tool", "parameters": {"exposure": 8, "brightness": 4}, '
                    '"reason": "Wrong direction on purpose."}]}'
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
            planner_cfg={"candidates_per_call": 1, "diversity_calls": 1, "temperatures": [0.4]},
            mcts_cfg={
                "num_simulations": 4,
                "max_actions_per_node": 1,
                "regression_tolerance": 0.01,
                "prune_regression_tolerance": 0.01,
            },
        )

        self.assertEqual(trajectory["num_steps"], 0)
        self.assertAlmostEqual(
            trajectory["final_quality"]["delta_e"],
            trajectory["initial_quality"]["delta_e"],
            places=4,
        )

    def test_parse_planner_response_tolerates_leading_plus_numbers(self) -> None:
        parsed = parse_planner_response(
            """
            ```json
            {
              "candidates": [
                {
                  "tool": "white_balance_tool",
                  "parameters": {"temperature": -8, "tint": +12},
                  "reason": "Repair slightly invalid JSON."
                }
              ]
            }
            ```
            """
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["candidates"][0]["parameters"]["tint"], 12)


if __name__ == "__main__":
    unittest.main()
