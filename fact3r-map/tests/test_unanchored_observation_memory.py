from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fact3r.proposals.mask_filter import MaskFilterConfig
from fact3r.proposals.mask_generator import MaskProposal2D
from fact3r.proposals.proposal_pipeline import (
    GeometryStatus,
    generate_lifted_proposals,
)
from fact3r.proposals.storage import (
    iter_saved_proposal_frames,
    save_frame_proposals,
)
from fact3r.reconstruction.keyframes import KeyframeRecord
from fact3r.semantics.observation_index import (
    attach_mapping_to_observation_index,
    build_observation_index,
    load_observation_index,
    rank_semantic_entity_groups,
)


class _Generator:
    def generate(self, rgb, *, frame_id):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[1:5, 1:5] = True
        return [
            MaskProposal2D(
                proposal_id=f"object-{frame_id}",
                frame_id=frame_id,
                mask=mask,
                score=0.98,
                source="test",
            )
        ]


class _Encoder:
    model_name = "test-siglip"
    device_name = "cpu"
    load_seconds = 0.0

    def encode_images(self, images):
        return np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (len(images), 1),
        )

    def encode_text(self, texts):
        return np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )


def _keyframe(frame_id: int, *, geometry: bool) -> KeyframeRecord:
    rows, columns = np.indices((6, 6))
    pointmap = np.stack(
        (columns * 0.01, rows * 0.01, np.ones((6, 6))), axis=-1
    ).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = frame_id * 0.1
    return KeyframeRecord(
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        rgb=np.full((6, 6, 3), 120, dtype=np.uint8),
        pointmap_camera=pointmap,
        geometry_confidence=np.full(
            (6, 6), 1.0 if geometry else 0.0, dtype=np.float32
        ),
        pose_world_from_camera=pose,
        intrinsics=np.asarray(
            [[4.0, 0.0, 2.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )


def _write_keyframes(directory: Path, keyframes: list[KeyframeRecord]) -> None:
    directory.mkdir()
    entries = []
    for index, keyframe in enumerate(keyframes):
        filename = f"keyframe_{index:06d}_frame_{keyframe.frame_id:06d}.npz"
        np.savez_compressed(
            directory / filename,
            frame_id=np.asarray(keyframe.frame_id),
            rgb=keyframe.rgb,
            pointmap_camera=keyframe.pointmap_camera,
            geometry_confidence=keyframe.geometry_confidence,
            pose_world_from_camera=keyframe.pose_world_from_camera,
            intrinsics=keyframe.intrinsics,
        )
        entries.append(
            {
                "keyframe_index": index,
                "frame_id": keyframe.frame_id,
                "timestamp": keyframe.timestamp,
                "file": filename,
                "image_shape": [6, 6],
                "has_mast3r_descriptors": False,
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "fact3r-mast3r-keyframes",
                "version": 1,
                "coordinate_convention": (
                    "pointmap_camera + pose_world_from_camera"
                ),
                "keyframes": entries,
            }
        ),
        encoding="utf-8",
    )


def _write_proposals(
    directory: Path, keyframes: list[KeyframeRecord]
) -> None:
    directory.mkdir()
    frames = []
    config = MaskFilterConfig(
        min_score=0.5,
        min_area_pixels=1,
        min_area_fraction=0.0,
        max_area_fraction=1.0,
        erosion_pixels=0,
        min_component_pixels=1,
        min_lifted_points=4,
    )
    for keyframe in keyframes:
        generated = generate_lifted_proposals(keyframe, _Generator(), config)
        summary = save_frame_proposals(directory, keyframe, generated)
        frames.append(
            {
                "frame_id": keyframe.frame_id,
                "proposal_count": summary["proposal_count"],
                "lifted_proposal_count": summary["lifted_proposal_count"],
                "unanchored_proposal_count": summary[
                    "unanchored_proposal_count"
                ],
                "manifest": (
                    f"frame_{keyframe.frame_id:06d}/manifest.json"
                ),
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "fact3r-sam2-proposals",
                "version": 2,
                "backend": "official",
                "model": "test",
                "frame_count": len(frames),
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def _write_tracklets(directory: Path, proposals: Path) -> None:
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "fact3r-sam2-tracklets",
                "version": 1,
                "source_proposals": str(proposals.resolve()),
                "model": "test",
                "frames": [
                    {
                        "frame_id": 0,
                        "observations": [
                            {
                                "proposal_id": "object-0",
                                "track_id": "track-object",
                                "source_proposal_id": None,
                                "link_iou": None,
                            }
                        ],
                    },
                    {
                        "frame_id": 1,
                        "observations": [
                            {
                                "proposal_id": "object-1",
                                "track_id": "track-object",
                                "source_proposal_id": "object-0",
                                "link_iou": 0.9,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class UnanchoredObservationTests(unittest.TestCase):
    def test_partial_geometry_keeps_raw_mask_area_and_measured_coverage(self) -> None:
        keyframe = _keyframe(0, geometry=False)
        confidence = np.zeros((6, 6), dtype=np.float32)
        confidence[1:3, 1:3] = 2.0
        keyframe = replace(keyframe, geometry_confidence=confidence)
        generated = generate_lifted_proposals(
            keyframe,
            _Generator(),
            MaskFilterConfig(
                min_score=0.5,
                min_area_pixels=1,
                min_area_fraction=0.0,
                max_area_fraction=1.0,
                erosion_pixels=0,
                min_component_pixels=1,
                min_geometry_confidence=1.0,
                min_lifted_points=4,
                full_anchor_coverage=0.5,
            ),
        )
        self.assertEqual(generated[0].geometry_status, GeometryStatus.PARTIAL_3D)
        self.assertAlmostEqual(generated[0].geometry_coverage, 0.25)
        self.assertEqual(generated[0].lifted_3d.source_mask_area, 16)
        self.assertEqual(len(generated[0].lifted_3d.points_world), 4)

    def test_sam_mask_survives_complete_geometry_failure(self) -> None:
        keyframe = _keyframe(0, geometry=False)
        generated = generate_lifted_proposals(
            keyframe,
            _Generator(),
            MaskFilterConfig(
                min_score=0.5,
                min_area_pixels=1,
                min_area_fraction=0.0,
                max_area_fraction=1.0,
                erosion_pixels=0,
                min_component_pixels=1,
                min_lifted_points=4,
            ),
        )
        self.assertEqual(len(generated), 1)
        self.assertIsNone(generated[0].lifted_3d)
        self.assertEqual(
            generated[0].geometry_status, GeometryStatus.UNANCHORED_2D
        )
        self.assertEqual(generated[0].geometry_coverage, 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = save_frame_proposals(temporary, keyframe, generated)
            with np.load(
                Path(temporary) / "frame_000000" / "proposal_0000.npz",
                allow_pickle=False,
            ) as payload:
                self.assertIn("mask", payload.files)
                self.assertNotIn("points_world", payload.files)
        self.assertEqual(manifest["proposal_count"], 1)
        self.assertEqual(manifest["lifted_proposal_count"], 0)

    def test_track_only_memory_is_searchable_and_has_revisit_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes"
            proposals = root / "proposals"
            tracklets = root / "tracklets"
            index = root / "index"
            records = [
                _keyframe(0, geometry=False),
                _keyframe(1, geometry=False),
            ]
            _write_keyframes(keyframes, records)
            _write_proposals(proposals, records)
            lifted_frames = list(iter_saved_proposal_frames(proposals))
            self.assertEqual(
                [len(frame.proposals) for frame in lifted_frames], [0, 0]
            )
            _write_tracklets(tracklets, proposals)
            build_observation_index(
                keyframes=keyframes,
                proposals=proposals,
                tracklets=tracklets,
                output=index,
                encoder=_Encoder(),
            )
            _, manifest, embeddings = load_observation_index(index)
            self.assertEqual(manifest["unanchored_observation_count"], 2)
            self.assertEqual(manifest["track_only_observation_count"], 2)
            self.assertEqual(
                {item["track_id"] for item in manifest["observations"]},
                {"track-object"},
            )
            self.assertIsNotNone(
                manifest["observations"][0]["view_ray_world"]
            )
            _, _, _, _, groups = rank_semantic_entity_groups(
                embeddings,
                manifest["observations"],
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 1.0]], dtype=np.float32),
                automatic_map_negatives=False,
            )
            self.assertEqual(groups[0]["group_id"], "track-object")
            self.assertIsNone(groups[0]["entity_id"])
            self.assertTrue(groups[0]["accepted"])

    def test_later_committed_track_backfills_earlier_2d_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "pre"
            mapping = root / "mapping"
            output = root / "mapped"
            index.mkdir()
            mapping.mkdir()
            np.save(index / "embeddings.npy", np.eye(2, dtype=np.float32))
            observations = [
                {
                    "index": index_value,
                    "frame_id": index_value,
                    "proposal_id": f"object-{index_value}",
                    "entity_id": None,
                    "track_id": "track-object",
                    "group_id": "track-object",
                    "assignment_status": "unanchored_2d",
                    "association_confidence": 0.8,
                    "geometry_status": "unanchored_2d",
                }
                for index_value in range(2)
            ]
            (index / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "fact3r-siglip-observation-index",
                        "version": 1,
                        "model": "test",
                        "source_proposals": str(root / "proposals"),
                        "source_mapping": None,
                        "embedding_file": "embeddings.npy",
                        "observation_count": 2,
                        "timing": {},
                        "observations": observations,
                    }
                ),
                encoding="utf-8",
            )
            (mapping / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "fact3r-visibility-residual-transport",
                        "version": 2,
                        "source_proposals": str(root / "proposals"),
                        "committed_track_entities": {
                            "track-object": "entity-object"
                        },
                        "frames": [],
                    }
                ),
                encoding="utf-8",
            )
            attach_mapping_to_observation_index(
                index=index, mapping=mapping, output=output
            )
            _, manifest, _ = load_observation_index(output)
            self.assertEqual(
                {item["entity_id"] for item in manifest["observations"]},
                {"entity-object"},
            )
            self.assertEqual(
                {
                    item["assignment_status"]
                    for item in manifest["observations"]
                },
                {"retrospectively_anchored"},
            )


if __name__ == "__main__":
    unittest.main()
