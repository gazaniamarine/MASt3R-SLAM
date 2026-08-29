from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "query_qwen_memory_live.py"
SPEC = importlib.util.spec_from_file_location("query_qwen_memory_live", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUERY)


class QwenMemoryLiveQueryTests(unittest.TestCase):
    def test_best_view_gallery_contains_masked_image_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposals = root / "proposals"
            proposals.mkdir()
            mask = np.zeros((12, 16), dtype=bool)
            mask[3:10, 5:13] = True
            np.savez_compressed(proposals / "mask.npz", mask=mask)
            rgb = np.full((12, 16, 3), 80, dtype=np.uint8)
            observations = [
                {
                    "index": 0,
                    "proposal_id": "proposal-0",
                    "frame_id": 7,
                    "timestamp": 3.5,
                    "mask_file": "mask.npz",
                }
            ]
            results = [
                {
                    "rank": 1,
                    "group_id": "image-entity-000060",
                    "score": 0.369,
                    "best_view": 0.458,
                    "frame_id": 7,
                    "views": 4,
                    "observation_index": 0,
                }
            ]
            output = root / "query"
            keyframe = SimpleNamespace(frame_id=7, rgb=rgb)
            with mock.patch.object(
                QUERY, "iter_exported_keyframes", return_value=[keyframe]
            ):
                QUERY._render_results(
                    "3D printer",
                    results,
                    manifest={
                        "source_keyframes": str(root / "frames"),
                        "source_proposals": str(proposals),
                    },
                    observations=observations,
                    output=output,
                    render_top_k=1,
                )

            self.assertTrue((output / "contact_sheet.jpg").is_file())
            self.assertTrue((output / "index.html").is_file())
            rendered = list(output.glob("rank_*.jpg"))
            self.assertEqual(len(rendered), 1)
            payload = json.loads((output / "results.json").read_text())
            self.assertEqual(payload["query"], "3D printer")
            self.assertEqual(payload["results"][0]["frame_id"], 7)
            self.assertEqual(
                payload["results"][0]["group_id"], "image-entity-000060"
            )


if __name__ == "__main__":
    unittest.main()
