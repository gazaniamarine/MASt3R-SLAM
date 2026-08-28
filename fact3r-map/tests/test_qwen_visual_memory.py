from __future__ import annotations

import unittest

import numpy as np

from fact3r.semantics.qwen_visual_memory import visual_link_diagnostics


class QwenVisualMemoryTests(unittest.TestCase):
    def test_link_diagnostic_separates_tracks(self) -> None:
        observations = [
            {"proposal_id": "a0", "frame_id": 0, "track_id": "a"},
            {"proposal_id": "b0", "frame_id": 0, "track_id": "b"},
            {"proposal_id": "a1", "frame_id": 1, "track_id": "a"},
            {"proposal_id": "b1", "frame_id": 1, "track_id": "b"},
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]],
            dtype=np.float32,
        )
        result = visual_link_diagnostics(embeddings, observations)
        self.assertEqual(result["linked_pair_count"], 2)
        self.assertEqual(result["top1_trial_count"], 2)
        self.assertEqual(result["previous_frame_top1_accuracy"], 1.0)
        self.assertGreater(result["median_link_margin"], 0.9)

    def test_link_diagnostic_handles_first_frame_only(self) -> None:
        result = visual_link_diagnostics(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [{"proposal_id": "a0", "frame_id": 0, "track_id": "a"}],
        )
        self.assertEqual(result["linked_pair_count"], 0)
        self.assertIsNone(result["previous_frame_top1_accuracy"])


if __name__ == "__main__":
    unittest.main()
