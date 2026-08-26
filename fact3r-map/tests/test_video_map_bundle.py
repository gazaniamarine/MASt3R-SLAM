from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fact3r.map_bundle import create_video_map_manifest, load_video_map


def _write(path: Path, payload: dict[str, object]) -> None:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class VideoMapBundleTests(unittest.TestCase):
    def _stages(self, root: Path) -> dict[str, Path]:
        keyframes = root / "fact3r_keyframes" / "room"
        proposals = root / "fact3r_sam2" / "room"
        tracklets = root / "fact3r_sam2_tracklets" / "room"
        mapping = root / "fact3r_delayed_commitment_uot" / "room"
        observations = root / "fact3r_siglip_observations" / "room"
        _write(
            keyframes,
            {
                "format": "fact3r-mast3r-keyframes",
                "version": 1,
                "keyframes": [{"frame_id": 0}, {"frame_id": 1}],
            },
        )
        _write(
            proposals,
            {
                "format": "fact3r-sam2-proposals",
                "version": 1,
                "keyframe_export": str(keyframes.resolve()),
                "frame_count": 2,
            },
        )
        _write(
            tracklets,
            {
                "format": "fact3r-sam2-tracklets",
                "version": 1,
                "keyframe_export": str(keyframes.resolve()),
                "source_proposals": str(proposals.resolve()),
                "frame_count": 2,
            },
        )
        _write(
            mapping,
            {
                "format": "fact3r-visibility-residual-transport",
                "version": 2,
                "source_keyframes": str(keyframes.resolve()),
                "source_proposals": str(proposals.resolve()),
                "source_tracklets": str(tracklets.resolve()),
                "frame_count": 2,
                "entity_count": 1,
            },
        )
        _write(
            observations,
            {
                "format": "fact3r-siglip-observation-index",
                "version": 1,
                "source_keyframes": str(keyframes.resolve()),
                "source_proposals": str(proposals.resolve()),
                "source_mapping": str((mapping / "manifest.json").resolve()),
                "frame_count": 2,
                "observation_count": 4,
            },
        )
        return {
            "keyframes": keyframes,
            "proposals": proposals,
            "tracklets": tracklets,
            "mapping": mapping,
            "observations": observations,
        }

    def test_completed_map_is_validated_and_paths_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "map"
            stages = self._stages(root)
            video = Path(temporary) / "room.mp4"
            video.write_bytes(b"video")
            manifest = create_video_map_manifest(
                output=root,
                video=video,
                map_name="my-room",
                sequence_name="room",
                **stages,
            )
            _, loaded = load_video_map(manifest)
            self.assertEqual(loaded["frame_count"], 2)
            self.assertEqual(loaded["entity_count"], 1)
            self.assertEqual(loaded["observation_count"], 4)
            self.assertEqual(
                Path(loaded["artifacts"]["observations"]["directory"]),
                stages["observations"].resolve(),
            )
            self.assertEqual(
                Path(loaded["query_directory"]), (root / "queries").resolve()
            )

    def test_mismatched_stage_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "map"
            stages = self._stages(root)
            manifest = stages["observations"] / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["frame_count"] = 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different frame counts"):
                create_video_map_manifest(
                    output=root,
                    video=Path(temporary) / "room.mp4",
                    map_name="my-room",
                    sequence_name="room",
                    **stages,
                )


if __name__ == "__main__":
    unittest.main()
