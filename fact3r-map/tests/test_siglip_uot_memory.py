from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fact3r.association import (
    AppearanceMemoryConfig,
    HungarianEntityMapper,
    HungarianMapConfig,
    PairwiseCostConfig,
    VisibilityResidualEntityMapper,
    build_pairwise_cost_matrix,
)
from fact3r.entities.entity import Entity, EntityStatus
from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.proposals.storage import SavedProposalFrame
from fact3r.reconstruction.keyframes import KeyframeRecord
from fact3r.semantics.appearance_memory import (
    AppearanceReliabilityConfig,
    load_siglip_appearance_index,
    proposal_appearance_reliability,
)
from fact3r.semantics.observation_index import (
    attach_mapping_to_observation_index,
    load_observation_index,
    map_derived_hard_negative_scores,
)
from fact3r.semantics.vlm_verification import rank_vlm_candidates


def _keyframe(frame_id: int) -> KeyframeRecord:
    points = np.asarray(
        [
            [[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0]],
            [[-0.1, 0.1, 1.0], [0.1, 0.1, 1.0]],
        ],
        dtype=np.float32,
    )
    return KeyframeRecord(
        frame_id=frame_id,
        timestamp=float(frame_id),
        rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        pointmap_camera=points,
        geometry_confidence=np.ones((2, 2), dtype=np.float32),
        pose_world_from_camera=np.eye(4, dtype=np.float32),
        intrinsics=np.asarray(
            [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )


def _proposal(
    frame_id: int,
    proposal_id: str,
    descriptor: list[float],
    reliability: float,
) -> LiftedProposal:
    keyframe = _keyframe(frame_id)
    return LiftedProposal(
        proposal_id=proposal_id,
        frame_id=frame_id,
        timestamp=float(frame_id),
        pixel_rc=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        points_world=keyframe.points_world().reshape(-1, 3),
        colours_rgb=keyframe.rgb.reshape(-1, 3),
        geometry_confidence=np.ones(4, dtype=np.float32),
        mast3r_descriptors=None,
        descriptor_confidence=None,
        source_mask_area=4,
        appearance_descriptor=np.asarray(descriptor, dtype=np.float32),
        appearance_reliability=reliability,
    )


def _entity(entity_id: str, descriptor: list[float]) -> Entity:
    proposal = _proposal(0, "seed", descriptor, 1.0)
    return Entity(
        id=entity_id,
        status=EntityStatus.CONFIRMED,
        centroid_xyz=proposal.centroid_xyz,
        bounding_box_xyz=proposal.bounding_box_xyz,
        surfel_or_voxel_geometry=proposal.points_world,
        colour_statistics={"mean_rgb": [128, 128, 128]},
        appearance_descriptor_bank=np.asarray([descriptor], dtype=np.float32),
        appearance_reliability=np.asarray([1.0], dtype=np.float32),
    )


class AppearanceAssociationTests(unittest.TestCase):
    def test_siglip_best_view_separates_geometrically_identical_entities(self) -> None:
        entities = (_entity("same", [1.0, 0.0]), _entity("other", [0.0, 1.0]))
        reliable = build_pairwise_cost_matrix(
            (_proposal(0, "p", [1.0, 0.0], 1.0),),
            entities,
            PairwiseCostConfig(appearance_weight=1.0),
        )
        weak = build_pairwise_cost_matrix(
            (_proposal(0, "p", [1.0, 0.0], 0.01),),
            entities,
            PairwiseCostConfig(appearance_weight=1.0),
        )
        self.assertLess(reliable.costs[0, 0], reliable.costs[0, 1])
        self.assertGreater(
            reliable.costs[0, 1] - reliable.costs[0, 0],
            weak.costs[0, 1] - weak.costs[0, 0],
        )

    def test_memory_keeps_diverse_views_and_replaces_redundant_lower_quality(self) -> None:
        mapper = HungarianEntityMapper(
            HungarianMapConfig(
                appearance_memory=AppearanceMemoryConfig(
                    max_views=2, max_redundant_similarity=0.95
                )
            )
        )
        mapper.process_frame((_proposal(0, "p0", [1.0, 0.0], 0.6),), frame_id=0)
        mapper.process_frame((_proposal(1, "p1", [0.0, 1.0], 0.8),), frame_id=1)
        mapper.process_frame((_proposal(2, "p2", [0.999, 0.02], 0.4),), frame_id=2)
        entity = mapper.entities[0]
        self.assertEqual(len(entity.appearance_descriptor_bank), 2)
        np.testing.assert_allclose(
            np.sort(entity.appearance_reliability), [0.6, 0.8], atol=1e-6
        )

    def test_residual_mapper_blocks_low_reliability_memory_poisoning(self) -> None:
        mapper = VisibilityResidualEntityMapper()
        mapper.process_frame(
            (_proposal(0, "p0", [1.0, 0.0], 0.9),),
            keyframe=_keyframe(0),
        )
        second = mapper.process_frame(
            (_proposal(1, "p1", [0.0, 1.0], 0.2),),
            keyframe=_keyframe(1),
        )
        self.assertEqual(len(second.assignment.matches), 1)
        decision = second.appearance_memory_decisions[0]
        self.assertFalse(decision.updated)
        self.assertIn("low_appearance_reliability", decision.blocking_reasons)
        self.assertEqual(len(mapper.entities[0].appearance_descriptor_bank), 1)


class AppearanceIndexTests(unittest.TestCase):
    def test_reliability_is_geometric_mean_of_existing_evidence(self) -> None:
        proposal = _proposal(0, "p", [1.0, 0.0], 1.0)
        proposal = replace(proposal, source_mask_area=16)
        evidence = proposal_appearance_reliability(
            proposal,
            {"proposal_score": 0.81, "mask_area": 1024},
            None,
            AppearanceReliabilityConfig(
                reference_mask_area=4096,
                missing_track_quality=1.0,
            ),
        )
        expected = (0.81 * 0.25 * 1.0 * 0.5) ** 0.25
        self.assertAlmostEqual(evidence.reliability, expected)

    def test_pre_uot_index_enriches_and_mapping_is_attached_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "pre"
            mapping = root / "mapping"
            output = root / "mapped"
            index.mkdir()
            mapping.mkdir()
            embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
            np.save(index / "embeddings.npy", embeddings)
            pre_manifest = {
                "format": "fact3r-siglip-observation-index",
                "version": 1,
                "model": "test",
                "source_keyframes": str(root / "keyframes"),
                "source_proposals": str(root / "proposals"),
                "source_mapping": None,
                "embedding_file": "embeddings.npy",
                "observation_count": 1,
                "timing": {},
                "observations": [
                    {
                        "index": 0,
                        "frame_id": 0,
                        "proposal_id": "p",
                        "proposal_score": 1.0,
                        "mask_area": 4,
                        "group_id": "observation-000000",
                    }
                ],
            }
            (index / "manifest.json").write_text(json.dumps(pre_manifest))
            mapping_manifest = {
                "format": "fact3r-visibility-residual-transport",
                "version": 2,
                "frames": [
                    {
                        "frame_id": 0,
                        "matches": [
                            {
                                "proposal_id": "p",
                                "entity_id": "entity-0",
                                "conditional_probability": 0.9,
                            }
                        ],
                    }
                ],
            }
            (mapping / "manifest.json").write_text(json.dumps(mapping_manifest))
            appearance = load_siglip_appearance_index(index)
            enriched, _ = appearance.enrich_frame(
                SavedProposalFrame(0, 0.0, (_proposal(0, "p", [0, 1], 1.0),))
            )
            np.testing.assert_allclose(
                enriched.proposals[0].appearance_descriptor, embeddings[0]
            )
            attach_mapping_to_observation_index(
                index=index, mapping=mapping, output=output
            )
            _, mapped_manifest, mapped_embeddings = load_observation_index(output)
            np.testing.assert_array_equal(mapped_embeddings, embeddings)
            self.assertEqual(
                mapped_manifest["observations"][0]["entity_id"], "entity-0"
            )
            self.assertTrue(
                mapped_manifest["timing"]["reused_pre_uot_embeddings"]
            )


class MapHardNegativeTests(unittest.TestCase):
    def test_nearest_competing_entity_becomes_automatic_negative(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]], dtype=np.float32
        )
        observations = [
            {"entity_id": "clock", "group_id": "clock"},
            {"entity_id": "frame", "group_id": "frame"},
            {"entity_id": "chair", "group_id": "chair"},
        ]
        scores, diagnostics = map_derived_hard_negative_scores(
            embeddings,
            observations,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            neighbors=1,
        )
        self.assertGreater(scores[0], 0.9)
        self.assertEqual(diagnostics["clock"][0]["group_id"], "frame")

    def test_absent_competitor_does_not_boost_vlm_candidate(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        observations = [
            {
                "entity_id": "only-entity",
                "group_id": "only-entity",
                "proposal_score": 1.0,
                "association_confidence": 1.0,
                "mask_area": 4096,
            },
            {
                "entity_id": "only-entity",
                "group_id": "only-entity",
                "proposal_score": 1.0,
                "association_confidence": 1.0,
                "mask_area": 4096,
            },
        ]
        scores, _, groups = rank_vlm_candidates(
            embeddings,
            observations,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(groups[0]["map_hard_negative_score"], 0.0)
        self.assertAlmostEqual(
            groups[0]["candidate_score"], float(np.mean(scores)), places=6
        )


if __name__ == "__main__":
    unittest.main()
