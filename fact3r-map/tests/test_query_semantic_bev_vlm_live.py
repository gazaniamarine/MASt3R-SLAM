from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "query_semantic_bev_vlm_live.py"
)
SPEC = importlib.util.spec_from_file_location("query_semantic_bev_vlm_live", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
query_live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(query_live)


class QuerySemanticBEVVLMTests(unittest.TestCase):
    def test_verified_entity_is_rendered_on_bev(self) -> None:
        occupancy = np.asarray([[0, 70, -1], [0, 70, -1]], dtype=np.int8)
        semantic_ids = np.asarray([[-1, 4, -1], [-1, 4, -1]], dtype=np.int32)
        result = {
            "query": "monitor",
            "entities": [
                {
                    "candidate_id": "entity-monitor",
                    "best_revisit_view": {"frame_id": 12},
                    "vlm": {
                        "predicted_object": "computer monitor",
                        "confidence": 0.91,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "verified.png"
            query_live._render_verified_bev(
                occupancy=occupancy,
                semantic_ids=semantic_ids,
                map_groups={
                    "entity-monitor": {
                        "group_id": "entity-monitor",
                        "semantic_id": 4,
                    }
                },
                result=result,
                output=output,
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.size, (433, 2))

    def test_best_frame_image_uses_verified_revisit_frame(self) -> None:
        entity = {
            "best_revisit_view": {"frame_id": 8},
            "observations": [
                {"frame_id": 3, "image": "frames/three.jpg"},
                {"frame_id": 8, "image": "frames/eight.jpg"},
            ],
        }
        self.assertEqual(
            query_live._best_frame_image(Path("query"), entity),
            Path("query/frames/eight.jpg"),
        )


if __name__ == "__main__":
    unittest.main()
