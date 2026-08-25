from __future__ import annotations

import unittest

import numpy as np

from fact3r.association import (
    BirthCommitmentStatus,
    DelayedCommitmentConfig,
    EntityVisibility,
    PairwiseCostMatrix,
    ResidualTransportConfig,
    ResidualUnmatchedReason,
    VisibilityConfig,
    VisibilityResidualEntityMapper,
    estimate_entity_visibility,
    solve_residual_transport,
)
from fact3r.association.tracklets import TrackletObservation
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


def _shifted_proposal(
    proposal_id: str,
    keyframe: KeyframeRecord,
    shift_xyz: list[float],
) -> LiftedProposal:
    proposal = _proposal(proposal_id, keyframe)
    return LiftedProposal(
        proposal_id=proposal.proposal_id,
        frame_id=proposal.frame_id,
        timestamp=proposal.timestamp,
        pixel_rc=proposal.pixel_rc,
        points_world=proposal.points_world + np.asarray(shift_xyz),
        colours_rgb=proposal.colours_rgb,
        geometry_confidence=proposal.geometry_confidence,
        mast3r_descriptors=proposal.mast3r_descriptors,
        descriptor_confidence=proposal.descriptor_confidence,
        source_mask_area=proposal.source_mask_area,
    )


def _tracklet(
    proposal_id: str,
    frame_id: int,
    track_id: str,
    *,
    source_proposal_id: str | None = None,
    link_iou: float | None = None,
) -> TrackletObservation:
    return TrackletObservation(
        frame_id=frame_id,
        proposal_id=proposal_id,
        track_id=track_id,
        source_proposal_id=source_proposal_id,
        link_iou=link_iou,
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

    def test_delayed_birth_requires_repeated_tracklet_evidence(self) -> None:
        mapper = VisibilityResidualEntityMapper(
            delayed_commitment_config=DelayedCommitmentConfig(
                min_observations=3,
                min_mean_birth_residual_ratio=0.5,
                min_median_link_iou=0.6,
                max_centroid_step_m=0.1,
            )
        )
        previous_id = None
        results = []
        for frame_id in range(3):
            keyframe = _small_keyframe(frame_id)
            proposal_id = f"proposal-{frame_id}"
            observation = _tracklet(
                proposal_id,
                frame_id,
                "track-stable",
                source_proposal_id=previous_id,
                link_iou=None if previous_id is None else 0.9,
            )
            results.append(
                mapper.process_frame(
                    (_proposal(proposal_id, keyframe),),
                    keyframe=keyframe,
                    tracklet_observations={proposal_id: observation},
                )
            )
            previous_id = proposal_id

        self.assertEqual(results[0].created_entity_ids, ())
        self.assertEqual(
            results[0].birth_decisions[0].status,
            BirthCommitmentStatus.DEFERRED,
        )
        self.assertEqual(results[1].created_entity_ids, ())
        self.assertEqual(results[2].created_entity_ids, ("entity-000000",))
        self.assertEqual(
            results[2].birth_decisions[0].status,
            BirthCommitmentStatus.CONFIRMED,
        )
        self.assertEqual(results[2].birth_decisions[0].observation_count, 3)
        self.assertAlmostEqual(
            results[2].birth_decisions[0].median_link_iou, 0.9
        )
        self.assertEqual(results[2].pending_track_count_after, 0)
        self.assertEqual(len(mapper.entities), 1)
        self.assertEqual(mapper.entities[0].first_seen_timestamp, 0.0)
        self.assertEqual(mapper.entities[0].last_seen_timestamp, 2.0)

    def test_single_frame_pending_birth_expires_without_entity(self) -> None:
        mapper = VisibilityResidualEntityMapper(
            delayed_commitment_config=DelayedCommitmentConfig(
                min_observations=3,
                max_missed_frames=0,
            )
        )
        first_keyframe = _small_keyframe(0)
        mapper.process_frame(
            (_proposal("first", first_keyframe),),
            keyframe=first_keyframe,
            tracklet_observations={
                "first": _tracklet("first", 0, "track-one-frame")
            },
        )
        second_keyframe = _small_keyframe(1)
        second = mapper.process_frame(
            (_proposal("second", second_keyframe),),
            keyframe=second_keyframe,
            tracklet_observations={
                "second": _tracklet("second", 1, "track-new")
            },
        )
        self.assertEqual(
            second.expired_pending_track_ids, ("track-one-frame",)
        )
        self.assertEqual(len(mapper.entities), 0)
        self.assertEqual(second.pending_track_count_after, 1)

    def test_known_track_cannot_spawn_duplicate_after_uot_rejection(self) -> None:
        mapper = VisibilityResidualEntityMapper(
            delayed_commitment_config=DelayedCommitmentConfig(
                min_observations=2,
                min_mean_birth_residual_ratio=0.5,
                min_median_link_iou=0.6,
                max_centroid_step_m=0.1,
            )
        )
        first_keyframe = _small_keyframe(0)
        mapper.process_frame(
            (_proposal("p0", first_keyframe),),
            keyframe=first_keyframe,
            tracklet_observations={"p0": _tracklet("p0", 0, "track-a")},
        )
        second_keyframe = _small_keyframe(1)
        confirmed = mapper.process_frame(
            (_proposal("p1", second_keyframe),),
            keyframe=second_keyframe,
            tracklet_observations={
                "p1": _tracklet(
                    "p1",
                    1,
                    "track-a",
                    source_proposal_id="p0",
                    link_iou=0.9,
                )
            },
        )
        self.assertEqual(confirmed.created_entity_ids, ("entity-000000",))

        third_keyframe = _small_keyframe(2)
        rejected = mapper.process_frame(
            (_shifted_proposal("p2", third_keyframe, [5.0, 0.0, 0.0]),),
            keyframe=third_keyframe,
            tracklet_observations={
                "p2": _tracklet(
                    "p2",
                    2,
                    "track-a",
                    source_proposal_id="p1",
                    link_iou=0.9,
                )
            },
        )
        self.assertEqual(len(rejected.assignment.matches), 0)
        self.assertEqual(rejected.created_entity_ids, ())
        self.assertEqual(
            rejected.birth_decisions[0].status,
            BirthCommitmentStatus.HELD_EXISTING,
        )
        self.assertEqual(
            rejected.birth_decisions[0].resolved_entity_id,
            "entity-000000",
        )
        self.assertEqual(len(mapper.entities), 1)

    def test_delayed_commitment_requires_complete_tracklet_observations(self) -> None:
        mapper = VisibilityResidualEntityMapper(
            delayed_commitment_config=DelayedCommitmentConfig()
        )
        keyframe = _small_keyframe(0)
        with self.assertRaisesRegex(ValueError, "every proposal"):
            mapper.process_frame(
                (_proposal("missing", keyframe),), keyframe=keyframe
            )

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
