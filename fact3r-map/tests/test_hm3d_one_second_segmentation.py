from __future__ import annotations

import unittest

import numpy as np

from fact3r.experiments import AdjacentMaskTracker, select_frame_window
from fact3r.proposals.mask_filter import (
    MaskFilterConfig,
    filter_image_mask_proposals,
)
from fact3r.proposals.mask_generator import MaskProposal2D


def _proposal(
    proposal_id: str,
    frame_id: int,
    rows: slice,
    columns: slice,
    *,
    score: float = 0.95,
) -> MaskProposal2D:
    mask = np.zeros((10, 10), dtype=bool)
    mask[rows, columns] = True
    return MaskProposal2D(
        proposal_id=proposal_id,
        frame_id=frame_id,
        mask=mask,
        score=score,
        source="test",
    )


class FrameWindowTests(unittest.TestCase):
    def test_one_second_at_30_fps_contains_every_captured_frame(self) -> None:
        indices = select_frame_window(
            total_frames=592,
            fps=30.0,
            duration_seconds=1.0,
            start_frame=240,
        )
        self.assertEqual(len(indices), 30)
        self.assertEqual(indices[0], 240)
        self.assertEqual(indices[-1], 269)
        self.assertIn(248, indices)

    def test_window_is_clipped_at_sequence_end(self) -> None:
        indices = select_frame_window(
            total_frames=10,
            fps=30.0,
            duration_seconds=1.0,
            start_frame=8,
        )
        self.assertEqual(indices, (8, 9))


class DenseMaskTrackingTests(unittest.TestCase):
    def test_adjacent_iou_preserves_tracks_when_proposal_order_changes(self) -> None:
        tracker = AdjacentMaskTracker(min_mask_iou=0.5)
        first = tracker.update(
            0,
            (
                _proposal("left-0", 0, slice(0, 4), slice(0, 4)),
                _proposal("right-0", 0, slice(0, 4), slice(6, 10)),
            ),
        )
        self.assertEqual(first.linked_count, 0)
        self.assertEqual(first.new_track_count, 2)

        second = tracker.update(
            1,
            (
                _proposal("right-1", 1, slice(0, 4), slice(6, 10)),
                _proposal("left-1", 1, slice(0, 4), slice(0, 4)),
                _proposal("new-1", 1, slice(6, 10), slice(3, 7)),
            ),
        )
        by_proposal = {
            item.proposal_id: item for item in second.observations
        }
        first_tracks = {
            item.proposal_id: item.track_id for item in first.observations
        }
        self.assertEqual(second.linked_count, 2)
        self.assertEqual(second.new_track_count, 1)
        self.assertEqual(
            by_proposal["right-1"].track_id, first_tracks["right-0"]
        )
        self.assertEqual(
            by_proposal["left-1"].track_id, first_tracks["left-0"]
        )
        self.assertIsNone(by_proposal["new-1"].source_proposal_id)

    def test_image_only_filter_reuses_shared_cleanup(self) -> None:
        kept = filter_image_mask_proposals(
            [
                _proposal("good", 4, slice(1, 9), slice(1, 9)),
                _proposal(
                    "low-score",
                    4,
                    slice(0, 2),
                    slice(0, 2),
                    score=0.2,
                ),
            ],
            (10, 10),
            MaskFilterConfig(
                min_score=0.8,
                min_area_pixels=4,
                min_area_fraction=0.0,
                max_area_fraction=0.8,
                erosion_pixels=0,
                min_component_pixels=4,
            ),
            frame_id=4,
        )
        self.assertEqual(tuple(item.proposal_id for item in kept), ("good",))


if __name__ == "__main__":
    unittest.main()
