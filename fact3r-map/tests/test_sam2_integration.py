from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from fact3r.integrations.mast3r_slam import (
    export_mast3r_keyframes,
    iter_exported_keyframes,
)
from fact3r.proposals.mask_filter import MaskFilterConfig, filter_mask_proposals
from fact3r.proposals.mask_generator import MaskProposal2D
from fact3r.proposals.proposal_pipeline import generate_lifted_proposals
from fact3r.proposals.sam2_generator import SAM2AutomaticMaskGenerator
from fact3r.proposals.sam2_official_generator import SAM2OfficialMaskGenerator
from fact3r.proposals.storage import save_frame_proposals
from fact3r.reconstruction.keyframes import KeyframeRecord


class _MockOfficialGenerator:
    """Stands in for sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator."""

    def __init__(self, mask_shape: tuple[int, int] = (6, 6)) -> None:
        self.mask_shape = mask_shape

    def generate(self, image: np.ndarray) -> list[dict]:
        annotations = []
        for index in range(2):
            mask = np.zeros(self.mask_shape, dtype=bool)
            mask[index, :] = True
            annotations.append(
                {
                    "segmentation": mask,
                    "predicted_iou": 0.9 - 0.1 * index,
                    "stability_score": 0.97,
                    "area": int(mask.sum()),
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                }
            )
        return annotations


class _FakePose:
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = 2.0
        return matrix[None]


class _FakeFrame:
    frame_id = 0
    uimg = np.full((6, 6, 3), 0.5, dtype=np.float32)
    # meshgrid returns a list on numpy 1.x and a tuple on 2.x; unpack so the
    # concatenation works in either environment.
    X_canon = np.dstack(
        [
            *np.meshgrid(
                np.arange(6, dtype=np.float32),
                np.arange(6, dtype=np.float32),
                indexing="xy",
            ),
            np.ones((6, 6), dtype=np.float32),
        ]
    ).reshape(-1, 3)
    C = np.full((36, 1), 2.0, dtype=np.float32)
    T_WC = _FakePose()
    K = None

    def get_average_conf(self) -> np.ndarray:
        return self.C


def _keyframe() -> KeyframeRecord:
    pointmap = np.zeros((6, 6, 3), dtype=np.float32)
    rows, columns = np.indices((6, 6))
    pointmap[..., 0] = columns
    pointmap[..., 1] = rows
    pointmap[..., 2] = 1.0
    return KeyframeRecord(
        frame_id=3,
        timestamp=0.3,
        rgb=np.zeros((6, 6, 3), dtype=np.uint8),
        pointmap_camera=pointmap,
        geometry_confidence=np.full((6, 6), 2.0, dtype=np.float32),
        pose_world_from_camera=np.eye(4, dtype=np.float32),
    )


class _MockPipeline:
    def __init__(self) -> None:
        self.call_kwargs = None

    def __call__(self, image, **kwargs):
        self.call_kwargs = kwargs
        primary = np.zeros((6, 6), dtype=bool)
        primary[1:5, 1:5] = True
        duplicate = primary.copy()
        full_image = np.ones((6, 6), dtype=bool)
        return {
            "masks": [primary, duplicate, full_image],
            "scores": np.asarray([0.97, 0.91, 0.99], dtype=np.float32),
            "bounding_boxes": np.asarray(
                [[1, 1, 5, 5], [1, 1, 5, 5], [0, 0, 6, 6]],
                dtype=np.float32,
            ),
        }


class _StaticGenerator:
    def generate(self, rgb, *, frame_id):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[1:5, 1:5] = True
        return [
            MaskProposal2D(
                proposal_id="static-0",
                frame_id=frame_id,
                mask=mask,
                score=0.99,
                source="test",
            )
        ]


class Sam2IntegrationTests(unittest.TestCase):
    def test_mast3r_export_round_trip_preserves_geometry_and_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_mast3r_keyframes(directory, [1.25], [_FakeFrame()])
            records = list(iter_exported_keyframes(directory))
        self.assertTrue(manifest.name == "manifest.json")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].timestamp, 1.25)
        self.assertEqual(records[0].rgb.dtype, np.uint8)
        np.testing.assert_allclose(records[0].points_world()[0, 0], [2.0, 0.0, 1.0])

    def test_calibrated_export_constrains_points_to_camera_rays(self) -> None:
        frame = _FakeFrame()
        frame.X_canon = np.full((36, 3), [99.0, 99.0, 1.0], dtype=np.float32)
        frame.K = np.asarray(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            export_mast3r_keyframes(directory, [1.25], [frame])
            record = next(iter_exported_keyframes(directory))
        np.testing.assert_allclose(
            record.pointmap_camera[0, 0], [-0.5, -0.5, 1.0]
        )

    def test_sam2_backend_uses_automatic_mask_generation(self) -> None:
        pipeline = _MockPipeline()
        generator = SAM2AutomaticMaskGenerator(
            pipeline_instance=pipeline,
            points_per_batch=32,
        )
        proposals = generator.generate(np.zeros((6, 6, 3)), frame_id=3)
        self.assertEqual(len(proposals), 3)
        self.assertEqual(proposals[0].mask.shape, (6, 6))
        self.assertEqual(pipeline.call_kwargs["points_per_batch"], 32)
        self.assertNotIn("input_points", pipeline.call_kwargs)

    def test_official_backend_converts_annotations_to_proposals(self) -> None:
        generator = SAM2OfficialMaskGenerator(
            generator_instance=_MockOfficialGenerator(),
            device="cpu",
        )
        proposals = generator.generate(np.zeros((6, 6, 3)), frame_id=3)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0].mask.shape, (6, 6))
        self.assertEqual(proposals[0].frame_id, 3)
        self.assertAlmostEqual(proposals[0].score, 0.9)
        self.assertEqual(proposals[0].metadata["stability_score"], 0.97)
        # sam2 reports xywh; the shared contract is xyxy.
        np.testing.assert_allclose(
            proposals[0].bounding_box_xyxy, [1.0, 2.0, 4.0, 6.0]
        )

    def test_official_backend_rejects_mask_of_the_wrong_size(self) -> None:
        """A mask that does not match the keyframe raster cannot be lifted."""

        generator = SAM2OfficialMaskGenerator(
            generator_instance=_MockOfficialGenerator(mask_shape=(4, 4)),
            device="cpu",
        )
        with self.assertRaises(ValueError):
            generator.generate(np.zeros((6, 6, 3)), frame_id=0)

    def test_filter_removes_full_image_and_duplicate_masks(self) -> None:
        pipeline = _MockPipeline()
        raw = SAM2AutomaticMaskGenerator(
            pipeline_instance=pipeline
        ).generate(np.zeros((6, 6, 3)), frame_id=3)
        filtered = filter_mask_proposals(
            raw,
            _keyframe(),
            MaskFilterConfig(
                min_score=0.8,
                min_area_pixels=1,
                min_area_fraction=0.0,
                max_area_fraction=0.8,
                erosion_pixels=0,
                min_component_pixels=1,
                duplicate_iou_threshold=0.9,
            ),
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].area, 16)

    def test_filtered_mask_lifts_and_saves_world_points(self) -> None:
        keyframe = _keyframe()
        generated = generate_lifted_proposals(
            keyframe,
            _StaticGenerator(),
            MaskFilterConfig(
                min_score=0.8,
                min_area_pixels=1,
                min_area_fraction=0.0,
                max_area_fraction=0.8,
                erosion_pixels=0,
                min_component_pixels=1,
            ),
        )
        self.assertEqual(len(generated), 1)
        self.assertEqual(len(generated[0].lifted_3d.points_world), 16)
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_frame_proposals(directory, keyframe, generated)
            proposal_file = (
                Path(directory) / "frame_000003" / "proposal_0000.npz"
            )
            with np.load(proposal_file, allow_pickle=False) as payload:
                points = payload["points_world"]
        self.assertEqual(manifest["proposal_count"], 1)
        self.assertEqual(points.shape, (16, 3))


if __name__ == "__main__":
    unittest.main()
