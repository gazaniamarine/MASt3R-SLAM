from __future__ import annotations

import unittest

import numpy as np

from fact3r.association.costs import (
    PairwiseCostConfig,
    PairwiseCostMatrix,
    TemporalEntityHint,
    build_pairwise_cost_matrix,
)
from fact3r.association.hungarian import (
    UnmatchedReason,
    associate_hungarian,
    solve_hungarian,
)
from fact3r.entities.entity import Entity, EntityStatus
from fact3r.proposals.lift_to_3d import LiftedProposal


def _proposal(
    proposal_id: str,
    centre: tuple[float, float, float],
    *,
    colour: tuple[int, int, int] = (128, 128, 128),
    descriptor: tuple[float, ...] | None = None,
) -> LiftedProposal:
    offsets = np.asarray(
        [
            [-0.01, -0.01, 0.0],
            [0.01, -0.01, 0.0],
            [-0.01, 0.01, 0.0],
            [0.01, 0.01, 0.0],
        ],
        dtype=np.float32,
    )
    points = offsets + np.asarray(centre, dtype=np.float32)
    descriptors = (
        None
        if descriptor is None
        else np.tile(np.asarray(descriptor, dtype=np.float32), (len(points), 1))
    )
    return LiftedProposal(
        proposal_id=proposal_id,
        frame_id=1,
        timestamp=1.0,
        pixel_rc=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        points_world=points,
        colours_rgb=np.tile(np.asarray(colour, dtype=np.uint8), (len(points), 1)),
        geometry_confidence=np.ones(len(points), dtype=np.float32),
        mast3r_descriptors=descriptors,
        descriptor_confidence=(
            None if descriptors is None else np.ones(len(points), dtype=np.float32)
        ),
        source_mask_area=len(points),
    )


def _entity(
    entity_id: str,
    centre: tuple[float, float, float],
    *,
    colour: tuple[int, int, int] = (128, 128, 128),
    descriptor: tuple[float, ...] | None = None,
) -> Entity:
    proposal = _proposal("geometry", centre, colour=colour, descriptor=descriptor)
    return Entity(
        id=entity_id,
        status=EntityStatus.CONFIRMED,
        centroid_xyz=proposal.centroid_xyz,
        bounding_box_xyz=proposal.bounding_box_xyz,
        surfel_or_voxel_geometry=proposal.points_world,
        colour_statistics={"median_rgb": colour},
        mast3r_descriptor_bank=proposal.mast3r_descriptors,
        descriptor_confidence=proposal.descriptor_confidence,
        observation_count=2,
        persistence_probability=0.8,
    )


class PairwiseCostTests(unittest.TestCase):
    def test_identical_geometry_colour_and_descriptor_has_zero_cost(self) -> None:
        proposal = _proposal("p0", (0.0, 0.0, 1.0), descriptor=(1.0, 0.0))
        entity = _entity("e0", (0.0, 0.0, 1.0), descriptor=(1.0, 0.0))
        matrix = build_pairwise_cost_matrix([proposal], [entity])
        self.assertTrue(matrix.candidate_mask[0, 0])
        self.assertAlmostEqual(matrix.costs[0, 0], 0.0, places=7)
        self.assertAlmostEqual(matrix.components["geometry"][0, 0], 0.0)

    def test_far_pair_is_removed_by_spatial_gating(self) -> None:
        proposal = _proposal("p0", (0.0, 0.0, 1.0))
        entity = _entity("e0", (5.0, 0.0, 1.0))
        matrix = build_pairwise_cost_matrix(
            [proposal],
            [entity],
            PairwiseCostConfig(max_centroid_distance_m=0.5),
        )
        self.assertFalse(matrix.candidate_mask[0, 0])
        self.assertTrue(np.isinf(matrix.costs[0, 0]))

    def test_missing_optional_cues_are_renormalized_not_penalized(self) -> None:
        proposal = _proposal("p0", (0.0, 0.0, 1.0))
        entity = _entity("e0", (0.0, 0.0, 1.0))
        entity.colour_statistics.clear()
        matrix = build_pairwise_cost_matrix([proposal], [entity])
        self.assertTrue(np.isnan(matrix.components["colour"][0, 0]))
        self.assertTrue(np.isnan(matrix.components["descriptor"][0, 0]))
        self.assertAlmostEqual(matrix.costs[0, 0], 0.0, places=7)

    def test_temporal_hint_biases_but_does_not_bypass_spatial_gating(self) -> None:
        proposal = _proposal("p0", (0.0, 0.0, 1.0))
        entities = [
            _entity("other", (0.0, 0.0, 1.0)),
            _entity("tracked", (0.0, 0.0, 1.0)),
            _entity("far", (5.0, 0.0, 1.0)),
        ]
        hint = TemporalEntityHint(entity_id="tracked", confidence=0.8)
        matrix = build_pairwise_cost_matrix(
            [proposal], entities, temporal_hints={"p0": hint}
        )
        self.assertGreater(matrix.costs[0, 0], matrix.costs[0, 1])
        self.assertEqual(matrix.components["temporal"][0, 0], 1.0)
        self.assertEqual(matrix.components["temporal"][0, 1], 0.0)
        self.assertFalse(matrix.candidate_mask[0, 2])
        self.assertTrue(np.isinf(matrix.costs[0, 2]))


class HungarianTests(unittest.TestCase):
    @staticmethod
    def _manual_matrix(costs: list[list[float]]) -> PairwiseCostMatrix:
        array = np.asarray(costs, dtype=np.float64)
        candidate_mask = np.isfinite(array)
        return PairwiseCostMatrix(
            proposal_ids=tuple(f"p{index}" for index in range(len(array))),
            entity_ids=tuple(f"e{index}" for index in range(array.shape[1])),
            costs=array,
            candidate_mask=candidate_mask,
            components={},
        )

    def test_hungarian_finds_global_not_row_greedy_assignment(self) -> None:
        matrix = self._manual_matrix([[0.10, 0.20], [0.11, 0.90]])
        result = solve_hungarian(matrix, max_match_cost=0.95)
        assignments = {
            match.proposal_index: match.entity_index for match in result.matches
        }
        self.assertEqual(assignments, {0: 1, 1: 0})

    def test_invalid_and_expensive_pairs_remain_unmatched(self) -> None:
        matrix = self._manual_matrix([[0.10, np.inf], [0.20, 0.90]])
        result = solve_hungarian(matrix, max_match_cost=0.65)
        self.assertEqual(
            [(match.proposal_index, match.entity_index) for match in result.matches],
            [(0, 0)],
        )
        self.assertEqual(result.unmatched_proposal_ids, ("p1",))
        self.assertEqual(result.unmatched_entity_ids, ("e1",))
        self.assertEqual(
            result.unmatched_proposals[0].reason,
            UnmatchedReason.ASSIGNMENT_COMPETITION,
        )
        self.assertEqual(result.unmatched_proposals[0].best_entity_id, "e0")
        self.assertAlmostEqual(result.unmatched_proposals[0].best_cost, 0.20)

    def test_unmatched_reasons_distinguish_gating_and_cost_threshold(self) -> None:
        matrix = self._manual_matrix([[np.inf, np.inf], [0.80, np.inf]])
        result = solve_hungarian(matrix, max_match_cost=0.65)
        self.assertEqual(
            tuple(item.reason for item in result.unmatched_proposals),
            (
                UnmatchedReason.NO_SPATIAL_CANDIDATE,
                UnmatchedReason.COST_ABOVE_THRESHOLD,
            ),
        )
        self.assertEqual(
            result.unmatched_reason_counts,
            {
                "empty_map": 0,
                "no_spatial_candidate": 1,
                "cost_above_threshold": 1,
                "assignment_competition": 0,
            },
        )

    def test_end_to_end_association_is_one_to_one(self) -> None:
        proposals = [
            _proposal("left", (0.0, 0.0, 1.0)),
            _proposal("right", (0.5, 0.0, 1.0)),
        ]
        entities = [
            _entity("right-entity", (0.5, 0.0, 1.0)),
            _entity("left-entity", (0.0, 0.0, 1.0)),
        ]
        result = associate_hungarian(proposals, entities)
        pairs = {(match.proposal_id, match.entity_id) for match in result.matches}
        self.assertEqual(
            pairs,
            {("left", "left-entity"), ("right", "right-entity")},
        )
        self.assertEqual(result.unmatched_proposal_indices, ())
        self.assertEqual(result.unmatched_entity_indices, ())

    def test_empty_side_returns_all_items_unmatched(self) -> None:
        result = associate_hungarian([_proposal("p0", (0.0, 0.0, 1.0))], [])
        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_proposal_ids, ("p0",))
        self.assertEqual(
            result.unmatched_proposals[0].reason, UnmatchedReason.EMPTY_MAP
        )


if __name__ == "__main__":
    unittest.main()
