from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fact3r.association.tracklets import (
    TRACKLET_FORMAT,
    TRACKLET_VERSION,
    link_propagated_masks,
    load_tracklet_run,
)
from fact3r.proposals.sam2_video_tracker import SAM2OfficialVideoTracker


def _mask(row_start: int, column_start: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[row_start : row_start + 3, column_start : column_start + 3] = True
    return mask


class _FakeVideoPredictor:
    def __init__(self) -> None:
        self.inputs: dict[int, np.ndarray] = {}

    def init_state(self, **kwargs):
        return {"video_path": kwargs["video_path"]}

    def reset_state(self, state) -> None:
        self.inputs.clear()

    def add_new_mask(self, *, obj_id, mask, **kwargs) -> None:
        self.inputs[int(obj_id)] = np.asarray(mask, dtype=bool)

    def propagate_in_video(self, *, start_frame_idx, **kwargs):
        object_ids = tuple(sorted(self.inputs))
        logits = np.stack(
            [
                np.where(self.inputs[obj_id], 1.0, -1.0)[None]
                for obj_id in object_ids
            ],
            axis=0,
        )
        yield start_frame_idx, object_ids, logits
        yield start_frame_idx + 1, object_ids, logits


class Sam2TrackletTests(unittest.TestCase):
    def test_propagated_masks_are_linked_jointly_by_iou(self) -> None:
        links = link_propagated_masks(
            ["source-left", "source-right"],
            [_mask(1, 1), _mask(1, 4)],
            ["target-right", "target-left"],
            [_mask(1, 4), _mask(1, 1)],
            min_mask_iou=0.5,
        )
        self.assertEqual(
            {(link.source_proposal_id, link.target_proposal_id) for link in links},
            {
                ("source-left", "target-left"),
                ("source-right", "target-right"),
            },
        )
        self.assertTrue(all(link.mask_iou == 1.0 for link in links))

    def test_video_adapter_preserves_source_order_across_batches(self) -> None:
        tracker = SAM2OfficialVideoTracker(
            predictor_instance=_FakeVideoPredictor(), device="cpu"
        )
        state = tracker.initialize("unused")
        source_masks = (_mask(1, 1), _mask(1, 4))
        propagated = tracker.propagate_one_step(
            state,
            source_frame_index=0,
            source_masks=source_masks,
            max_seeds_per_batch=1,
        )
        np.testing.assert_array_equal(propagated[0], source_masks[0])
        np.testing.assert_array_equal(propagated[1], source_masks[1])

    def test_tracklet_manifest_loads_adjacent_evidence(self) -> None:
        payload = {
            "format": TRACKLET_FORMAT,
            "version": TRACKLET_VERSION,
            "source_proposals": "/tmp/proposals",
            "model": "mock",
            "frames": [
                {
                    "frame_id": 0,
                    "observations": [
                        {
                            "proposal_id": "p0",
                            "track_id": "track-000000",
                            "source_proposal_id": None,
                            "link_iou": None,
                        }
                    ],
                },
                {
                    "frame_id": 1,
                    "observations": [
                        {
                            "proposal_id": "p1",
                            "track_id": "track-000000",
                            "source_proposal_id": "p0",
                            "link_iou": 0.75,
                        }
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            run = load_tracklet_run(path)
        observation = run.observations_by_frame[1][0]
        self.assertEqual(observation.source_proposal_id, "p0")
        self.assertEqual(observation.track_id, "track-000000")
        self.assertAlmostEqual(observation.link_iou, 0.75)


if __name__ == "__main__":
    unittest.main()
