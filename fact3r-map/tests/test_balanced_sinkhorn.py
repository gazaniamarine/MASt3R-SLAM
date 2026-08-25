from __future__ import annotations

import unittest

import numpy as np

from fact3r.association import (
    BalancedSinkhornConfig,
    BalancedSinkhornEntityMapper,
    PairwiseCostMatrix,
    UnmatchedReason,
    solve_balanced_sinkhorn,
)
from fact3r.proposals.lift_to_3d import LiftedProposal


def _matrix(costs: list[list[float]]) -> PairwiseCostMatrix:
    values = np.asarray(costs, dtype=np.float64)
    candidates = np.isfinite(values)
    return PairwiseCostMatrix(
        proposal_ids=tuple(f"p{index}" for index in range(values.shape[0])),
        entity_ids=tuple(f"e{index}" for index in range(values.shape[1])),
        costs=values,
        candidate_mask=candidates,
        components={},
    )


def _proposal(proposal_id: str, frame_id: int) -> LiftedProposal:
    points = np.asarray(
        [
            [-0.01, -0.01, 1.0],
            [0.01, -0.01, 1.0],
            [-0.01, 0.01, 1.0],
            [0.01, 0.01, 1.0],
        ],
        dtype=np.float32,
    )
    return LiftedProposal(
        proposal_id=proposal_id,
        frame_id=frame_id,
        timestamp=float(frame_id),
        pixel_rc=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        points_world=points,
        colours_rgb=np.full((4, 3), 128, dtype=np.uint8),
        geometry_confidence=np.ones(4, dtype=np.float32),
        mast3r_descriptors=None,
        descriptor_confidence=None,
        source_mask_area=4,
    )


class BalancedSinkhornTests(unittest.TestCase):
    def test_plan_satisfies_uniform_marginals_and_prefers_low_cost(self) -> None:
        result = solve_balanced_sinkhorn(
            _matrix([[0.0, 1.0], [1.0, 0.0]]),
            config=BalancedSinkhornConfig(
                entropy_temperature=0.1,
                marginal_tolerance=1e-9,
            ),
            max_match_cost=1.0,
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.transport_plan.sum(axis=1), [0.5, 0.5])
        np.testing.assert_allclose(result.transport_plan.sum(axis=0), [0.5, 0.5])
        self.assertGreater(result.transport_plan[0, 0], result.transport_plan[0, 1])
        self.assertEqual(
            {(match.proposal_index, match.entity_index) for match in result.matches},
            {(0, 0), (1, 1)},
        )

    def test_row_commitment_allows_multiple_fragments_to_one_entity(self) -> None:
        result = solve_balanced_sinkhorn(_matrix([[0.1], [0.2]]))
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(
            tuple(match.entity_id for match in result.matches), ("e0", "e0")
        )
        self.assertTrue(all(match.row_probability == 1.0 for match in result.matches))

    def test_hard_matches_never_use_noncandidate_transport_edges(self) -> None:
        result = solve_balanced_sinkhorn(
            _matrix([[0.1, np.inf], [np.inf, 0.1]])
        )
        self.assertEqual(
            {(match.proposal_index, match.entity_index) for match in result.matches},
            {(0, 0), (1, 1)},
        )
        self.assertGreaterEqual(result.noncandidate_mass, 0.0)

    def test_isolated_and_above_threshold_rows_remain_unmatched(self) -> None:
        result = solve_balanced_sinkhorn(
            _matrix([[0.8], [np.inf]]), max_match_cost=0.65
        )
        self.assertEqual(result.matches, ())
        self.assertEqual(
            tuple(item.reason for item in result.unmatched_proposals),
            (
                UnmatchedReason.COST_ABOVE_THRESHOLD,
                UnmatchedReason.NO_SPATIAL_CANDIDATE,
            ),
        )

    def test_mapper_updates_one_entity_from_multiple_frame_fragments(self) -> None:
        mapper = BalancedSinkhornEntityMapper()
        mapper.process_frame([_proposal("seed", 0)], frame_id=0)
        result = mapper.process_frame(
            [_proposal("fragment-a", 1), _proposal("fragment-b", 1)],
            frame_id=1,
        )
        self.assertEqual(len(result.assignment.matches), 2)
        self.assertEqual(result.created_entity_ids, ())
        self.assertEqual(len(mapper.entities), 1)
        self.assertEqual(mapper.entities[0].observation_count, 3)


if __name__ == "__main__":
    unittest.main()
