from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_uot_mapping.py"
SPEC = importlib.util.spec_from_file_location("check_uot_mapping", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class CheckUOTMappingTests(unittest.TestCase):
    def test_summary_exposes_solver_and_track_failures(self) -> None:
        manifest = {
            "format": "fact3r-visibility-residual-transport",
            "mode": "image_uot_no_dense_reconstruction",
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
                    "uot": {
                        "converged": True,
                        "fixed_point_error": 1e-8,
                        "unmatched_reason_counts": {"empty_map": 1},
                    },
                },
                {
                    "frame_id": 1,
                    "matches": [
                        {
                            "proposal_id": "p1",
                            "entity_id": "e0",
                            "tracklet": {"track_id": "t0"},
                            "conditional_probability": 0.9,
                            "retained_ratio": 0.8,
                            "cost": 0.1,
                        }
                    ],
                    "unmatched_proposals": [
                        {
                            "proposal_id": "p2",
                            "resolved_entity_id": "e0",
                            "track_id": "t1",
                        }
                    ],
                    "uot": {
                        "converged": True,
                        "fixed_point_error": 2e-8,
                        "unmatched_reason_counts": {},
                    },
                },
                {
                    "frame_id": 2,
                    "matches": [
                        {
                            "proposal_id": "p3",
                            "entity_id": "e1",
                            "tracklet": {"track_id": "t0"},
                            "conditional_probability": 0.7,
                            "retained_ratio": 0.6,
                            "cost": 0.2,
                        }
                    ],
                    "unmatched_proposals": [],
                    "uot": {
                        "converged": False,
                        "fixed_point_error": 0.1,
                        "unmatched_reason_counts": {"ambiguous_transport": 1},
                    },
                },
            ],
        }

        summary = CHECK.summarize_uot_mapping(manifest)

        self.assertEqual(summary["numerical_health"], "inspect")
        self.assertAlmostEqual(summary["uot_convergence_rate"], 2 / 3)
        self.assertAlmostEqual(summary["identity_reuse_fraction"], 0.75)
        self.assertEqual(summary["resolved_duplicate_observations"], 1)
        self.assertEqual(summary["fragmented_repeat_track_fraction"], 1.0)
        self.assertEqual(summary["track_identity_switch_rate"], 0.5)
        self.assertEqual(summary["entities_joining_multiple_tracklets"], 1)


if __name__ == "__main__":
    unittest.main()
