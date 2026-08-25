from __future__ import annotations

import unittest

import numpy as np

from fact3r.association import (
    EntityVisibility,
    PairwiseCostMatrix,
    ResidualTransportConfig,
    ResidualUnmatchedReason,
    VisibilityConfig,
    VisibilityResidualEntityMapper,
    estimate_entity_visibility,
    solve_residual_transport,
)
from fact3r.entities.entity import Entity, EntityStatus
from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.reconstruction.keyframes import KeyframeRecord


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


def _visibility(entity_id: str, score: float) -> EntityVisibility:
    return EntityVisibility(
        entity_id=entity_id,
        score=score,
        in_frustum_fraction=score,
        unoccluded_fraction=score,
        sampled_point_count=1,
        projected_point_count=int(score > 0.0),
        visible_point_count=int(score > 0.0),
    )


def _entity(entity_id: str, point: list[float]) -> Entity:
    geometry = np.asarray([point], dtype=np.float32)
    return Entity(
        id=entity_id,
        status=EntityStatus.PROVISIONAL,
        centroid_xyz=geometry[0],
        bounding_box_xyz=np.stack((geometry[0], geometry[0])),
        surfel_or_voxel_geometry=geometry,
    )


def _small_keyframe(frame_id: int) -> KeyframeRecord:
    pointmap = np.asarray(
        [
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0]],
            [[-0.5, 0.5, 1.0], [0.5, 0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    return KeyframeRecord(
        frame_id=frame_id,
        timestamp=float(frame_id),
        rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        pointmap_camera=pointmap,
        geometry_confidence=np.ones((2, 2), dtype=np.float32),
        pose_world_from_camera=np.eye(4, dtype=np.float32),
        intrinsics=np.asarray(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )


def _proposal(proposal_id: str, keyframe: KeyframeRecord) -> LiftedProposal:
    return LiftedProposal(
        proposal_id=proposal_id,
        frame_id=keyframe.frame_id,
        timestamp=keyframe.timestamp,
        pixel_rc=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        points_world=keyframe.points_world().reshape(-1, 3),
        colours_rgb=keyframe.rgb.reshape(-1, 3),
        geometry_confidence=np.ones(4, dtype=np.float32),
        mast3r_descriptors=None,
        descriptor_confidence=None,
        source_mask_area=4,
    )


class VisibilityTests(unittest.TestCase):
    def test_projection_distinguishes_visible_occluded_and_out_of_view(self) -> None:
        pointmap = np.zeros((3, 3, 3), dtype=np.float32)
        pointmap[..., 2] = 2.0
        keyframe = KeyframeRecord(
            frame_id=0,
            timestamp=0.0,
            rgb=np.zeros((3, 3, 3), dtype=np.uint8),
            pointmap_camera=pointmap,
            geometry_confidence=np.ones((3, 3), dtype=np.float32),
            pose_world_from_camera=np.eye(4, dtype=np.float32),
            intrinsics=np.asarray(
                [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )
        visibility = estimate_entity_visibility(
            (
                _entity("visible", [0.0, 0.0, 1.0]),
                _entity("occluded", [0.0, 0.0, 3.0]),
                _entity("outside", [10.0, 0.0, 1.0]),
            ),
            keyframe,
            VisibilityConfig(depth_tolerance_m=0.05),
        )
        self.assertEqual(tuple(item.score for item in visibility), (1.0, 0.0, 0.0))
        self.assertEqual(visibility[0].visible_point_count, 1)
        self.assertEqual(visibility[1].projected_point_count, 1)
        self.assertEqual(visibility[2].projected_point_count, 0)


class ResidualTransportTests(unittest.TestCase):
    def test_mapper_reuses_entity_after_visibility_conditioned_transport(self) -> None:
        mapper = VisibilityResidualEntityMapper()
        first_keyframe = _small_keyframe(0)
        first = mapper.process_frame(
            (_proposal("first", first_keyframe),), keyframe=first_keyframe
        )
        self.assertEqual(first.created_entity_ids, ("entity-000000",))

        second_keyframe = _small_keyframe(1)
        second = mapper.process_frame(
            (_proposal("second", second_keyframe),), keyframe=second_keyframe
        )
        self.assertEqual(len(second.assignment.matches), 1)
        self.assertEqual(second.assignment.matches[0].entity_id, "entity-000000")
        self.assertEqual(second.created_entity_ids, ())
        self.assertEqual(len(mapper.entities), 1)

    def test_strict_support_keeps_forbidden_mass_exactly_zero(self) -> None:
        result = solve_residual_transport(
            _matrix([[0.1, np.inf], [0.2, np.inf], [np.inf, 0.1]]),
            np.ones(3),
            (_visibility("e0", 1.0), _visibility("e1", 1.0)),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.forbidden_mass, 0.0)
        self.assertTrue(
            np.all(result.transport_plan[~result.cost_matrix.candidate_mask] == 0.0)
        )
        self.assertEqual(
            {(match.proposal_id, match.entity_id) for match in result.matches},
            {("p0", "e0"), ("p1", "e0"), ("p2", "e1")},
        )

    def test_visibility_changes_entity_demand_and_miss_residual(self) -> None:
        visible = solve_residual_transport(
            _matrix([[1.0]]),
            np.ones(1),
            (_visibility("e0", 1.0),),
            max_match_cost=1.0,
        )
        mostly_occluded = solve_residual_transport(
            _matrix([[1.0]]),
            np.ones(1),
            (_visibility("e0", 0.1),),
            max_match_cost=1.0,
        )
        self.assertEqual(visible.entity_masses[0], 1.0)
        self.assertEqual(mostly_occluded.entity_masses[0], 0.1)
        self.assertGreater(
            visible.entity_miss_residuals[0],
            mostly_occluded.entity_miss_residuals[0],
        )

    def test_low_mass_and_ambiguity_have_separate_reasons(self) -> None:
        low_mass = solve_residual_transport(
            _matrix([[0.6]]),
            np.ones(1),
            (_visibility("e0", 1.0),),
            config=ResidualTransportConfig(min_retained_ratio=0.9),
        )
        self.assertEqual(
            low_mass.unmatched_proposals[0].reason,
            ResidualUnmatchedReason.LOW_RETAINED_MASS,
        )
        ambiguous = solve_residual_transport(
            _matrix([[0.1, 0.1]]),
            np.ones(1),
            (_visibility("e0", 1.0), _visibility("e1", 1.0)),
            config=ResidualTransportConfig(min_conditional_probability=0.6),
        )
        self.assertEqual(
            ambiguous.unmatched_proposals[0].reason,
            ResidualUnmatchedReason.AMBIGUOUS_TRANSPORT,
        )

    def test_no_proposal_leaves_only_visible_entity_miss_mass(self) -> None:
        matrix = PairwiseCostMatrix(
            proposal_ids=(),
            entity_ids=("e0", "e1"),
            costs=np.empty((0, 2), dtype=np.float64),
            candidate_mask=np.empty((0, 2), dtype=bool),
            components={},
        )
        result = solve_residual_transport(
            matrix,
            np.empty(0),
            (_visibility("e0", 1.0), _visibility("e1", 0.0)),
        )
        self.assertEqual(result.transport_plan.shape, (0, 2))
        np.testing.assert_allclose(result.entity_miss_residuals, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
