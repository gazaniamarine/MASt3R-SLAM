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
from fact3r.semantics.vlm_verification import (
    VLMVerification,
    local_image_source,
    parse_listwise_verification_output,
    parse_verification_output,
    pathological_border_sliver,
    prepare_vlm_query,
    rank_vlm_candidates,
    verify_prepared_query,
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


class _EntityVerifier:
    model_name = "test-entity-verifier"
    load_seconds = 0.0
    listwise_batch_size = 3

    def __init__(self):
        self.calls = []

    def _answer(self, entity_id, frame_ids):
        if entity_id == "entity-clock":
            return VLMVerification(
                decision="yes",
                confidence=0.94,
                predicted_object="wall clock",
                confusable_with=("picture frame",),
                supporting_frames=tuple(frame_ids),
                reason="The highlighted object is a clock in both views.",
            )
        return VLMVerification(
            decision="no",
            confidence=0.91,
            predicted_object="blue wall panel",
            confusable_with=("window",),
            supporting_frames=(),
            reason="The highlighted object is not a clock.",
        )

    def verify(self, *, query, entity_id, evidence_images, frame_ids):
        self.calls.append((query, entity_id, tuple(frame_ids)))
        return self._answer(entity_id, frame_ids)

    def verify_many(self, *, query, requests):
        self.calls.append(
            ("many", query, tuple(request.entity_id for request in requests))
        )
        return {
            request.entity_id: self._answer(request.entity_id, request.frame_ids)
            for request in requests
        }


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
    def test_propagated_tracks_reuse_discovery_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes"
            proposals = root / "proposals"
            tracklets = root / "tracklets"
            index = root / "index"
            _write_keyframes(keyframes)
            _write_proposals(proposals)
            second_manifest_path = proposals / "frame_000001/manifest.json"
            second_manifest = json.loads(
                second_manifest_path.read_text(encoding="utf-8")
            )
            for proposal in second_manifest["proposals"]:
                proposal["source"] = "sam2-video-memory"
            second_manifest_path.write_text(
                json.dumps(second_manifest), encoding="utf-8"
            )
            tracklets.mkdir()
            tracklet_frames = []
            for frame_id in (0, 1):
                observations = []
                for colour in ("red", "blue"):
                    observations.append(
                        {
                            "proposal_id": f"{colour}-{frame_id}",
                            "track_id": f"track-{colour}",
                            "source_proposal_id": (
                                None if frame_id == 0 else f"{colour}-0"
                            ),
                            "link_iou": None if frame_id == 0 else 0.9,
                        }
                    )
                tracklet_frames.append(
                    {"frame_id": frame_id, "observations": observations}
                )
            (tracklets / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "fact3r-sam2-tracklets",
                        "version": 1,
                        "source_proposals": str(proposals.resolve()),
                        "model": "test",
                        "frames": tracklet_frames,
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = build_observation_index(
                keyframes=keyframes,
                proposals=proposals,
                tracklets=tracklets,
                output=index,
                encoder=_ColourEncoder(),
                batch_size=2,
                context_fraction=0.0,
                outside_mask_alpha=0.0,
                reuse_propagated_track_embeddings=True,
            )
            _, manifest, embeddings = load_observation_index(manifest_path)
            self.assertEqual(manifest["observation_count"], 4)
            self.assertEqual(manifest["encoded_observation_count"], 2)
            self.assertEqual(manifest["reused_embedding_count"], 2)
            self.assertTrue(np.allclose(embeddings[0], embeddings[2]))
            self.assertTrue(np.allclose(embeddings[1], embeddings[3]))

    def test_pathological_border_sliver_rejects_edge_strip_only(self) -> None:
        strip = np.zeros((480, 640), dtype=bool)
        strip[:, 625:639] = True
        chair_like = np.zeros((480, 640), dtype=bool)
        chair_like[180:470, 20:210] = True
        self.assertTrue(pathological_border_sliver(strip))
        self.assertFalse(pathological_border_sliver(chair_like))

    def test_qwen_local_image_source_is_plain_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "evidence image.jpg"
            image.write_bytes(b"test")
            source = local_image_source(image)
            self.assertEqual(source, str(image.resolve()))
            self.assertFalse(source.startswith("file://"))

    def test_qwen_json_is_extracted_and_validated(self) -> None:
        parsed = parse_verification_output(
            "```json\n"
            '{"decision":"yes","confidence":0.91,'
            '"predicted_object":"clock","confusable_with":["fan"],'
            '"supporting_frames":[2,3],"reason":"visible hands"}'
            "\n```"
        )
        self.assertEqual(parsed.decision, "yes")
        self.assertAlmostEqual(parsed.confidence, 0.91)
        self.assertEqual(parsed.supporting_frames, (2, 3))
        with self.assertRaisesRegex(ValueError, "JSON"):
            parse_verification_output("yes, probably")

    def test_qwen_listwise_json_fails_missing_candidate_closed(self) -> None:
        parsed = parse_listwise_verification_output(
            '{"candidates":[{"entity_id":"entity-clock",'
            '"decision":"yes","confidence":0.92,'
            '"predicted_object":"clock","confusable_with":[],'
            '"supporting_frames":[2,3],"reason":"clock face"}]}',
            ["entity-clock", "entity-fan"],
        )
        self.assertEqual(parsed["entity-clock"].decision, "yes")
        self.assertEqual(parsed["entity-fan"].decision, "uncertain")
        self.assertEqual(parsed["entity-fan"].confidence, 0.0)

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

    def test_vlm_shortlist_is_positive_only_and_requires_persistence(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [1.0, 0.0]], dtype=np.float32
        )
        observations = [
            {"entity_id": "entity-clock", "mask_area": 4096},
            {"entity_id": "entity-clock", "mask_area": 4096},
            {"entity_id": None, "mask_area": 4096},
        ]
        _, _, groups = rank_vlm_candidates(
            embeddings,
            observations,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual([item["entity_id"] for item in groups], ["entity-clock"])
        self.assertEqual(groups[0]["evidence_observation_indices"], [0, 1])

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

    def test_vlm_verifier_filters_candidates_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keyframes = root / "keyframes"
            proposals = root / "proposals"
            mapping = root / "mapping"
            index = root / "index"
            output = root / "vlm-query"
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
            prepared = prepare_vlm_query(
                index=manifest_path,
                query="clock",
                output=output,
                encoder=encoder,
                max_candidates=2,
                min_siglip_score=-1.0,
            )
            verifier = _EntityVerifier()
            result_path = verify_prepared_query(prepared, verifier=verifier)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual([item["entity_id"] for item in result["entities"]], ["entity-clock"])
            self.assertEqual(result["rendered_observation_count"], 2)
            self.assertIn("blue wall panel", result["dynamic_confounders"])
            self.assertTrue((output / "matches.gif").is_file())
            self.assertEqual(
                verifier.calls,
                [("many", "clock", ("entity-clock", "entity-blue"))],
            )

            cached_verifier = _EntityVerifier()
            verify_prepared_query(prepared, verifier=cached_verifier)
            self.assertEqual(cached_verifier.calls, [])


if __name__ == "__main__":
    unittest.main()
