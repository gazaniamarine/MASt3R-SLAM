from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from import_vlnce_frames import _cropped_intrinsics, _frame_stride, convert  # noqa: E402


def _render_directory(directory: Path, frames: int = 12, size: int = 64) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        pixels = np.full((size, size, 3), index * 5 % 256, dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f"{index:06d}.png")
    lines = ["# habitat camera poses: timestamp tx ty tz qx qy qz qw"]
    for index in range(frames):
        lines.append(f"{index / 30.0:.6f} {index:.6f} 1.5 0.0 0.0 0.0 0.0 1.0")
    (directory / "groundtruth.txt").write_text("\n".join(lines) + "\n")
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "scene": "zsNo4HB9uLZ",
                "fps": 30.0,
                "intrinsics": {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 32.0},
                "legs": [{"index": 0, "landmark": "rug"}],
                "return_query": "rug",
                "return_leg_index": 0,
                "return_position": [1.0, 0.1, 2.0],
                "return_optimal_geodesic_m": 7.5,
                "success_distance_m": 3.0,
            }
        )
    )
    return directory


class FrameStrideTest(unittest.TestCase):
    def test_downsamples_to_the_requested_rate(self):
        self.assertEqual(_frame_stride(30.0, 2.0), 15)

    def test_keeps_every_frame_when_asked_for_more_than_available(self):
        self.assertEqual(_frame_stride(30.0, 60.0), 1)

    def test_never_returns_zero(self):
        self.assertEqual(_frame_stride(30.0, 29.9), 1)

    def test_rejects_a_non_positive_rate(self):
        with self.assertRaises(ValueError):
            _frame_stride(30.0, 0.0)


class CroppedIntrinsicsTest(unittest.TestCase):
    def test_square_render_crops_to_four_thirds(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            Image.fromarray(np.zeros((512, 512, 3), np.uint8)).save(image)
            intrinsics = _cropped_intrinsics(
                {"fx": 256.0, "fy": 256.0, "cx": 256.0, "cy": 256.0}, image, 512,
                [384, 512],
            )
        # No rescale at 512, so focal is unchanged; the crop moves cy only.
        self.assertAlmostEqual(intrinsics["fx"], 256.0)
        self.assertAlmostEqual(intrinsics["cx"], 256.0)
        self.assertAlmostEqual(intrinsics["cy"], 192.0)

    def test_focal_scales_with_the_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            Image.fromarray(np.zeros((1024, 1024, 3), np.uint8)).save(image)
            intrinsics = _cropped_intrinsics(
                {"fx": 512.0, "fy": 512.0, "cx": 512.0, "cy": 512.0}, image, 512,
                [384, 512],
            )
        self.assertAlmostEqual(intrinsics["fx"], 256.0)

    def test_returns_none_without_source_intrinsics(self):
        self.assertIsNone(_cropped_intrinsics(None, Path("unused"), 512, [384, 512]))


class ConvertTest(unittest.TestCase):
    def test_writes_a_readable_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = _render_directory(Path(tmp) / "render")
            output = Path(tmp) / "frames"
            manifest_path = convert(render, output, sample_fps=2.0, max_frames=None,
                                    size=64)
            manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["format"], "fact3r-mast3r-keyframes")
        self.assertEqual(manifest["vlnce"]["frame_stride"], 15)
        self.assertEqual(manifest["vlnce"]["return_query"], "rug")

    def test_keyframe_ids_are_contiguous_after_subsampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render"
            _render_directory(render, frames=12)
            output = Path(tmp) / "frames"
            manifest = json.loads(
                convert(render, output, sample_fps=15.0, max_frames=None, size=64).read_text()
            )
        ids = [entry["frame_id"] for entry in manifest["keyframes"]]
        self.assertEqual(ids, list(range(len(ids))))
        # The original render index is kept for audit even though ids restart.
        self.assertEqual(manifest["keyframes"][1]["source_frame_id"], 2)

    def test_geometry_stays_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render"
            _render_directory(render, frames=4)
            output = Path(tmp) / "frames"
            manifest = json.loads(
                convert(render, output, sample_fps=30.0, max_frames=None, size=64).read_text()
            )
            payload = np.load(output / manifest["keyframes"][0]["file"])
        self.assertTrue(np.isnan(payload["pointmap_camera"]).all())
        np.testing.assert_allclose(payload["pose_world_from_camera"], np.eye(4))

    def test_groundtruth_pose_is_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render"
            _render_directory(render, frames=4)
            output = Path(tmp) / "frames"
            manifest = json.loads(
                convert(render, output, sample_fps=30.0, max_frames=None, size=64).read_text()
            )
            self.assertTrue((output / "groundtruth.txt").is_file())
        self.assertIn("groundtruth_pose", manifest["keyframes"][0])

    def test_max_frames_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render"
            _render_directory(render, frames=12)
            output = Path(tmp) / "frames"
            manifest = json.loads(
                convert(render, output, sample_fps=30.0, max_frames=3, size=64).read_text()
            )
        self.assertEqual(len(manifest["keyframes"]), 3)

    def test_empty_render_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render"
            render.mkdir(parents=True)
            with self.assertRaises(ValueError):
                convert(render, Path(tmp) / "frames", sample_fps=2.0, max_frames=None,
                        size=64)


if __name__ == "__main__":
    unittest.main()
