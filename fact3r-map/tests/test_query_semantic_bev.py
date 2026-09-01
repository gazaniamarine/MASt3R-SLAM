from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "query_semantic_bev.py"
SPEC = importlib.util.spec_from_file_location("query_semantic_bev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
query_semantic_bev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(query_semantic_bev)


class QuerySemanticBEVTests(unittest.TestCase):
    def test_query_ensemble_keeps_exact_phrase_and_head_noun(self) -> None:
        prompts = query_semantic_bev._query_prompts("computer monitor")
        self.assertIn("computer monitor", prompts)
        self.assertIn("monitor", prompts)
        self.assertIn("a monitor", prompts)

    def test_shared_vlm_prompts_also_include_head_noun(self) -> None:
        from fact3r.semantics.observation_index import default_positive_prompts

        prompts = default_positive_prompts("computer monitor")
        self.assertIn("monitor", prompts)
        self.assertIn("a monitor", prompts)

    def test_two_prompt_agreement_recovers_head_noun_signal(self) -> None:
        observation = np.asarray([[1.0, 0.0]], dtype=np.float32)
        prompts = np.asarray(
            [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]],
            dtype=np.float32,
        )
        fused, winner, raw = query_semantic_bev._fuse_prompt_scores(
            observation, prompts, agreement_prompts=2
        )
        self.assertGreater(float(fused[0]), 0.9)
        self.assertEqual(int(winner[0]), 2)
        self.assertEqual(raw.shape, (1, 4))

    def test_ranking_retains_best_source_observation(self) -> None:
        observations = [
            {"group_id": "a"},
            {"group_id": "a"},
            {"group_id": "b"},
        ]
        ranked = query_semantic_bev._rank_groups(
            np.asarray([0.2, 0.8, 0.5]),
            observations,
            {"a", "b"},
            top_views=2,
        )
        self.assertEqual(ranked[0]["group_id"], "a")
        self.assertEqual(ranked[0]["best_observation_index"], 1)
        self.assertAlmostEqual(float(ranked[0]["best_view_score"]), 0.8)

    def test_observed_frame_renderer_writes_highlighted_image(self) -> None:
        rgb = np.full((10, 12, 3), 80, dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=bool)
        mask[2:8, 3:9] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.jpg"
            query_semantic_bev._render_observed_frame(
                rgb,
                mask,
                query="chair",
                group_id="entity-1",
                frame_id=4,
                score=0.7,
                rank=1,
                output=output,
            )
            self.assertTrue(output.exists())
            with Image.open(output) as rendered:
                self.assertEqual(rendered.size, (12, 68))


if __name__ == "__main__":
    unittest.main()
