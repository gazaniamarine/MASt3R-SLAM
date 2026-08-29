from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_image_uot_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_image_uot_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)

RETRIEVAL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_semantic_retrieval_ablation.py"
)
RETRIEVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_semantic_retrieval_ablation", RETRIEVAL_SCRIPT
)
assert RETRIEVAL_SPEC is not None and RETRIEVAL_SPEC.loader is not None
RETRIEVAL = importlib.util.module_from_spec(RETRIEVAL_SPEC)
RETRIEVAL_SPEC.loader.exec_module(RETRIEVAL)


class ImageUOTAblationTests(unittest.TestCase):
    def test_sparse_point_annotation_identifies_covering_entity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((10, 10), dtype=bool)
            mask[2:7, 3:8] = True
            np.savez_compressed(root / "mask.npz", mask=mask)
            relevant = RETRIEVAL.relevant_entities_for_target(
                [
                    {
                        "frame_id": 4,
                        "proposal_id": "p0",
                        "entity_id": "e0",
                    }
                ],
                {"p0": {"mask_file": "mask.npz"}},
                root,
                [{"frame_id": 4, "xy": [5, 5]}],
            )
            self.assertEqual(relevant, {"e0"})

    def test_evaluate_variant_measures_continuity_and_coherence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {
                "appearance_model": "test-model",
                "association_cues": ["appearance", "sam2"],
                "source_appearance_index": str(root / "appearance"),
                "timing": {"total_seconds": 0.5, "frames_per_second": 4.0},
                "frames": [
                    {
                        "frame_id": 0,
                        "matches": [],
                        "unmatched_proposals": [
                            {
                                "proposal_id": "p0",
                                "created_entity_id": "e0",
                                "track_id": "t0",
                            }
                        ],
                    },
                    {
                        "frame_id": 1,
                        "matches": [
                            {
                                "proposal_id": "p1",
                                "entity_id": "e0",
                                "tracklet": {"track_id": "t0"},
                                "appearance_similarity": 0.98,
                                "sam2_link_iou": 0.90,
                                "mast3r_mask_support": 0.0,
                            }
                        ],
                        "unmatched_proposals": [],
                    },
                ],
            }
            (root / "manifest.json").write_text(json.dumps(mapping))
            appearance_manifest = {
                "observations": [
                    {"proposal_id": "p0", "index": 0},
                    {"proposal_id": "p1", "index": 1},
                ]
            }
            embeddings = np.asarray([[1.0, 0.0], [0.98, 0.02]], dtype=np.float32)
            with mock.patch.object(
                EVALUATE,
                "load_observation_index",
                return_value=(root / "appearance" / "manifest.json", appearance_manifest, embeddings),
            ):
                metrics = EVALUATE.evaluate_variant("A+S", root)

            self.assertEqual(metrics["entities"], 1)
            self.assertEqual(metrics["matched_rate"], 0.5)
            self.assertEqual(metrics["birth_rate"], 0.5)
            self.assertEqual(metrics["fragmented_track_fraction"], 0.0)
            self.assertGreater(metrics["semantic_within_cosine"], 0.99)
            self.assertEqual(metrics["mapping_fps"], 4.0)


if __name__ == "__main__":
    unittest.main()
