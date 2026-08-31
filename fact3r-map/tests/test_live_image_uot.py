from __future__ import annotations

import unittest

import numpy as np

from fact3r.association.live_image_uot import (
    LiveImageUOTMapper,
    LiveTrackletEvidence,
)
from fact3r.proposals.mask_generator import MaskProposal2D


def _proposal(proposal_id: str, frame_id: int) -> MaskProposal2D:
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:10, 4:13] = True
    return MaskProposal2D(
        proposal_id=proposal_id,
        frame_id=frame_id,
        mask=mask,
        score=0.95,
    )


class LiveImageUOTTests(unittest.TestCase):
    def test_temporally_linked_observation_keeps_entity_identity(self) -> None:
        mapper = LiveImageUOTMapper()
        first, _ = mapper.update(
            frame_id=0,
            proposals=[_proposal("p0", 0)],
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            tracklets={"p0": LiveTrackletEvidence("track-0", None, None)},
        )
        second, frame = mapper.update(
            frame_id=1,
            proposals=[_proposal("p1", 1)],
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            tracklets={"p1": LiveTrackletEvidence("track-0", "p0", 1.0)},
        )
        self.assertEqual(first["p0"].entity_id, second["p1"].entity_id)
        self.assertEqual(second["p1"].status, "matched")
        self.assertEqual(mapper.entity_count, 1)
        self.assertEqual(mapper.total_matches, 1)
        self.assertEqual(len(frame["matches"]), 1)
        self.assertEqual(frame["entity_count_after"], 1)
        self.assertEqual(frame["created_entity_ids"], [])

    def test_empty_frame_is_a_valid_live_update(self) -> None:
        mapper = LiveImageUOTMapper()
        assignments, frame = mapper.update(
            frame_id=0,
            proposals=[],
            embeddings=np.empty((0, 8), dtype=np.float32),
            tracklets={},
        )
        self.assertEqual(assignments, {})
        self.assertEqual(frame["matches"], [])
        self.assertEqual(mapper.entity_count, 0)
        self.assertEqual(frame["entity_count_after"], 0)


if __name__ == "__main__":
    unittest.main()
