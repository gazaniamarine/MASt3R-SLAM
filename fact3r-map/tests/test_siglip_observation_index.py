from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from fact3r.semantics.observation_index import (
    build_observation_index,
    load_observation_index,
    masked_context_crop,
    _pooled_feature_value,
    query_observation_index,
    rank_semantic_entity_groups,
)


class _ColourEncoder:
    model_name = "test-colour-encoder"
    device_name = "cpu"
    load_seconds = 0.0

    def encode_images(self, images):
        features = []
        for image in images:
            mean = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            features.append([mean[0] + 1.0, mean[2] + 1.0])
        return np.asarray(features, dtype=np.float32)

    def encode_text(self, texts):
        return np.asarray(
            [[1.0, 0.0] if "clock" in text.lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def _write_keyframes(directory: Path) -> None:
    directory.mkdir()
    entries = []
    for index, frame_id in enumerate((0, 1)):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[:, :3, 0] = 255
        rgb[:, 3:, 2] = 255
        filename = f"keyframe_{index:06d}_frame_{frame_id:06d}.npz"
        np.savez_compressed(
            directory / filename,
            frame_id=np.asarray(frame_id, dtype=np.int64),
            rgb=rgb,
            pointmap_camera=np.ones((4, 6, 3), dtype=np.float32),
            geometry_confidence=np.ones((4, 6), dtype=np.float32),
            pose_world_from_camera=np.eye(4, dtype=np.float32),
        )
        entries.append(
            {
                "keyframe_index": index,
                "frame_id": frame_id,
                "timestamp": frame_id / 30.0,
                "file": filename,
                "image_shape": [4, 6],
                "has_mast3r_descriptors": False,
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "fact3r-mast3r-keyframes",
                "version": 1,
                "coordinate_convention": "pointmap_camera + pose_world_from_camera",
                "keyframes": entries,
            }
        ),
        encoding="utf-8",
    )


def _write_proposals(directory: Path) -> None:
    directory.mkdir()
    run_entries = []
    for frame_id in (0, 1):
        frame_directory = directory / f"frame_{frame_id:06d}"
        frame_directory.mkdir()
        proposal_entries = []
        for index, (name, columns) in enumerate(
            (("red", slice(0, 3)), ("blue", slice(3, 6)))
        ):
            mask = np.zeros((4, 6), dtype=bool)
            mask[:, columns] = True
            filename = f"proposal_{index:04d}.npz"
            np.savez_compressed(frame_directory / filename, mask=mask)
            proposal_entries.append(
                {
                    "proposal_id": f"{name}-{frame_id}",
                    "file": filename,
                    "source": "test",
                    "score": 0.99,
                    "mask_area": int(mask.sum()),
                    "lifted_point_count": int(mask.sum()),
                    "centroid_xyz": [0.0, 0.0, 1.0],
                    "bounding_box_xyz": [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                    "bounding_box_xyxy": (
                        [0, 0, 3, 4] if name == "red" else [3, 0, 6, 4]
                    ),
                }
            )
        relative_manifest = f"frame_{frame_id:06d}/manifest.json"
        (directory / relative_manifest).write_text(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "timestamp": frame_id / 30.0,
                    "image_shape": [4, 6],
                    "proposal_count": 2,
                    "visualization": "alignment.ply",
                    "proposals": proposal_entries,
                }
            ),
            encoding="utf-8",
        )
        run_entries.append(
            {
                "frame_id": frame_id,
                "proposal_count": 2,
                "manifest": relative_manifest,
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "fact3r-sam2-proposals",
                "version": 1,
                "backend": "official",
                "model": "test",
                "frame_count": 2,
                "frames": run_entries,
            }
        ),
        encoding="utf-8",
    )


def _write_mapping(directory: Path) -> None:
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "fact3r-visibility-residual-transport",
                "version": 2,
                "committed_track_entities": {"track-red": "entity-clock"},
                "frames": [
                    {
                        "frame_id": 0,
                        "matches": [],
                        "unmatched_proposals": [
                            {
                                "proposal_id": "red-0",
                                "track_id": "track-red",
                                "commitment_status": "deferred",
                                "resolved_entity_id": None,
                                "created_entity_id": None,
                            },
                            {
                                "proposal_id": "blue-0",
                                "created_entity_id": "entity-blue",
                            },
                        ],
                    },
                    {
                        "frame_id": 1,
                        "matches": [
                            {
                                "proposal_id": "blue-1",
                                "entity_id": "entity-blue",
                            }
                        ],
                        "unmatched_proposals": [
                            {
                                "proposal_id": "red-1",
                                "track_id": "track-red",
                                "commitment_status": "confirmed",
                                "resolved_entity_id": "entity-clock",
                                "created_entity_id": "entity-clock",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class SiglipObservationIndexTests(unittest.TestCase):
    def test_transformers_pooled_output_container_is_unwrapped(self) -> None:
        pooled = np.ones((2, 4), dtype=np.float32)
        output = SimpleNamespace(
            last_hidden_state=np.ones((2, 16, 4), dtype=np.float32),
            pooler_output=pooled,
        )
        self.assertIs(_pooled_feature_value(output), pooled)

    def test_masked_crop_dims_context_without_losing_selected_pixels(self) -> None:
        rgb = np.full((5, 5, 3), 200, dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        crop = np.asarray(
            masked_context_crop(
                rgb, mask, context_fraction=1.0, outside_mask_alpha=0.0
            )
        )
        np.testing.assert_array_equal(crop[1, 1], [200, 200, 200])
        np.testing.assert_array_equal(crop[0, 0], [127, 127, 127])

    def test_confirmed_multiview_ranking_excludes_unassigned_singletons(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [1.0, 0.0]], dtype=np.float32
        )
        observations = [
            {
                "group_id": "entity-clock",
                "entity_id": "entity-clock",
                "proposal_score": 1.0,
                "mask_area": 4096,
            },
            {
                "group_id": "entity-clock",
                "entity_id": "entity-clock",
                "proposal_score": 1.0,
                "association_confidence": 1.0,
                "mask_area": 4096,
            },
            {
                "group_id": "observation-fragment",
                "entity_id": None,
                "proposal_score": 1.0,
                "association_confidence": 1.0,
                "mask_area": 4096,
            },
        ]
        _, _, _, _, groups = rank_semantic_entity_groups(
            embeddings,
            observations,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["group_id"], "entity-clock")
        self.assertTrue(groups[0]["accepted"])
        self.assertEqual(groups[0]["supporting_view_count"], 2)

    def test_clock_query_returns_every_view_of_committed_entity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes"
            proposals = root / "proposals"
            mapping = root / "mapping"
            index = root / "index"
            query_output = root / "query"
            _write_keyframes(keyframes)
            _write_proposals(proposals)
            _write_mapping(mapping)
            encoder = _ColourEncoder()

            manifest_path = build_observation_index(
                keyframes=keyframes,
                proposals=proposals,
                mapping=mapping,
                output=index,
                encoder=encoder,
                batch_size=2,
                context_fraction=0.0,
                outside_mask_alpha=0.0,
            )
            manifest_path, manifest, embeddings = load_observation_index(
                manifest_path
            )
            self.assertEqual(manifest["observation_count"], 4)
            self.assertEqual(manifest["assigned_observation_count"], 4)
            self.assertEqual(embeddings.shape, (4, 2))
            red_observations = [
                item for item in manifest["observations"]
                if str(item["proposal_id"]).startswith("red")
            ]
            self.assertEqual(
                {item["entity_id"] for item in red_observations},
                {"entity-clock"},
            )

            results_path = query_observation_index(
                index=manifest_path,
                query="clock",
                output=query_output,
                encoder=encoder,
                max_entities=1,
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results["entities"][0]["entity_id"], "entity-clock")
            self.assertEqual(
                [
                    item["frame_id"]
                    for item in results["entities"][0]["observations"]
                ],
                [0, 1],
            )
            self.assertEqual(results["rendered_observation_count"], 2)
            self.assertTrue((query_output / "matches.gif").is_file())
            self.assertTrue((query_output / "contact_sheet.jpg").is_file())
            self.assertTrue((query_output / "index.html").is_file())

            no_match_output = root / "no-match"
            no_match_path = query_observation_index(
                index=manifest_path,
                query="lamp",
                output=no_match_output,
                encoder=encoder,
                max_entities=1,
            )
            no_match = json.loads(no_match_path.read_text(encoding="utf-8"))
            self.assertFalse(no_match["confident_match_found"])
            self.assertEqual(no_match["entities"], [])
            self.assertIsNone(no_match["gif"])
            self.assertTrue((no_match_output / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
