"""Portable manifest for a completed video-to-Fact3R map run."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping


MAP_FORMAT = "fact3r-video-map"
MAP_VERSION = 1

_ARTIFACT_FORMATS = {
    "keyframes": "fact3r-mast3r-keyframes",
    "proposals": "fact3r-sam2-proposals",
    "tracklets": "fact3r-sam2-tracklets",
    "mapping": "fact3r-visibility-residual-transport",
    "observations": "fact3r-siglip-observation-index",
}


def _manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "manifest.json" if candidate.is_dir() else candidate


def _load_stage(name: str, path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = _manifest_path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{name} manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _ARTIFACT_FORMATS[name]
    if payload.get("format") != expected:
        raise ValueError(
            f"{name} manifest has format {payload.get('format')!r}; "
            f"expected {expected!r}"
        )
    return manifest_path, payload


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _frame_count(name: str, payload: Mapping[str, object]) -> int:
    if name == "keyframes":
        return len(payload.get("keyframes", []))
    return int(payload.get("frame_count", 0))


def create_video_map_manifest(
    *,
    output: str | Path,
    video: str | Path,
    map_name: str,
    sequence_name: str,
    keyframes: str | Path,
    proposals: str | Path,
    tracklets: str | Path,
    mapping: str | Path,
    observations: str | Path,
    calibration: str | Path | None = None,
) -> Path:
    """Validate completed stages and write one user-facing map manifest."""

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, tuple[Path, dict[str, object]]] = {
        name: _load_stage(name, path)
        for name, path in {
            "keyframes": keyframes,
            "proposals": proposals,
            "tracklets": tracklets,
            "mapping": mapping,
            "observations": observations,
        }.items()
    }
    counts = {
        name: _frame_count(name, payload)
        for name, (_, payload) in stages.items()
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"map stages contain different frame counts: {counts}")
    frame_count = next(iter(counts.values()))
    if frame_count <= 0:
        raise ValueError("cannot finalize an empty video map")

    keyframe_directory = stages["keyframes"][0].parent
    proposal_directory = stages["proposals"][0].parent
    tracklet_directory = stages["tracklets"][0].parent
    mapping_directory = stages["mapping"][0].parent
    observation_directory = stages["observations"][0].parent
    lineage = {
        "proposal_keyframes": stages["proposals"][1].get("keyframe_export"),
        "tracklet_keyframes": stages["tracklets"][1].get("keyframe_export"),
        "tracklet_proposals": stages["tracklets"][1].get("source_proposals"),
        "mapping_keyframes": stages["mapping"][1].get("source_keyframes"),
        "mapping_proposals": stages["mapping"][1].get("source_proposals"),
        "mapping_tracklets": stages["mapping"][1].get("source_tracklets"),
        "observation_keyframes": stages["observations"][1].get(
            "source_keyframes"
        ),
        "observation_proposals": stages["observations"][1].get(
            "source_proposals"
        ),
    }
    expected_lineage = {
        "proposal_keyframes": keyframe_directory,
        "tracklet_keyframes": keyframe_directory,
        "tracklet_proposals": proposal_directory,
        "mapping_keyframes": keyframe_directory,
        "mapping_proposals": proposal_directory,
        "mapping_tracklets": tracklet_directory,
        "observation_keyframes": keyframe_directory,
        "observation_proposals": proposal_directory,
    }
    for field, expected in expected_lineage.items():
        recorded = lineage[field]
        if recorded is not None and Path(str(recorded)).resolve() != expected.resolve():
            raise ValueError(
                f"artifact lineage mismatch for {field}: {recorded} != {expected}"
            )
    observation_mapping = stages["observations"][1].get("source_mapping")
    if observation_mapping is not None:
        if Path(str(observation_mapping)).resolve() != stages["mapping"][0]:
            raise ValueError("observation index was built from a different mapping")

    artifact_directories = {
        "keyframes": keyframe_directory,
        "proposals": proposal_directory,
        "tracklets": tracklet_directory,
        "mapping": mapping_directory,
        "observations": observation_directory,
    }
    payload = {
        "format": MAP_FORMAT,
        "version": MAP_VERSION,
        "map_name": map_name,
        "sequence_name": sequence_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(Path(video).resolve()),
        "calibration": (
            None if calibration is None else str(Path(calibration).resolve())
        ),
        "frame_count": frame_count,
        "entity_count": int(stages["mapping"][1].get("entity_count", 0)),
        "observation_count": int(
            stages["observations"][1].get("observation_count", 0)
        ),
        "unanchored_observation_count": int(
            stages["observations"][1].get(
                "unanchored_observation_count", 0
            )
        ),
        "track_only_observation_count": int(
            stages["observations"][1].get(
                "track_only_observation_count", 0
            )
        ),
        "artifacts": {
            name: {
                "directory": _relative(directory, root),
                "manifest": _relative(stages[name][0], root),
                "format": _ARTIFACT_FORMATS[name],
            }
            for name, directory in artifact_directories.items()
        },
        "query_directory": "queries",
    }
    manifest_path = root / "map.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_video_map(path: str | Path) -> tuple[Path, dict[str, object]]:
    """Load a bundle and resolve its artifact paths for callers."""

    candidate = Path(path)
    manifest_path = (candidate / "map.json" if candidate.is_dir() else candidate).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != MAP_FORMAT or payload.get("version") != MAP_VERSION:
        raise ValueError(f"unsupported Fact3R video map: {manifest_path}")
    resolved = dict(payload)
    resolved_artifacts: dict[str, dict[str, object]] = {}
    for name, artifact in payload["artifacts"].items():
        entry = dict(artifact)
        for field in ("directory", "manifest"):
            value = Path(str(entry[field]))
            if not value.is_absolute():
                value = manifest_path.parent / value
            entry[field] = str(value.resolve())
        resolved_artifacts[str(name)] = entry
    resolved["artifacts"] = resolved_artifacts
    query_directory = Path(str(payload.get("query_directory", "queries")))
    if not query_directory.is_absolute():
        query_directory = manifest_path.parent / query_directory
    resolved["query_directory"] = str(query_directory.resolve())
    return manifest_path, resolved
