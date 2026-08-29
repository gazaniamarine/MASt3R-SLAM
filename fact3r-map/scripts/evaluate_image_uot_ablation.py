#!/usr/bin/env python3
"""Summarize image-UOT cue/backend ablations without requiring object labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import load_observation_index  # noqa: E402


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must have the form LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("variant label and path cannot be empty")
    return label.strip(), Path(raw_path)


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return None if not materialized else float(statistics.median(materialized))


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return None if not materialized else float(statistics.fmean(materialized))


def _assignment_rows(
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in manifest["frames"]:
        frame_id = int(frame["frame_id"])
        for match in frame.get("matches", []):
            tracklet = match.get("tracklet") or {}
            rows.append(
                {
                    "frame_id": frame_id,
                    "proposal_id": str(match["proposal_id"]),
                    "entity_id": str(match["entity_id"]),
                    "track_id": tracklet.get("track_id"),
                    "status": "matched",
                    "appearance_similarity": match.get("appearance_similarity"),
                    "sam2_link_iou": match.get("sam2_link_iou"),
                    "mast3r_mask_support": match.get("mast3r_mask_support"),
                }
            )
        for unmatched in frame.get("unmatched_proposals", []):
            entity_id = unmatched.get("created_entity_id") or unmatched.get(
                "resolved_entity_id"
            )
            if entity_id is None:
                continue
            tracklet = unmatched.get("tracklet") or {}
            rows.append(
                {
                    "frame_id": frame_id,
                    "proposal_id": str(unmatched["proposal_id"]),
                    "entity_id": str(entity_id),
                    "track_id": unmatched.get("track_id") or tracklet.get("track_id"),
                    "status": (
                        "duplicate"
                        if unmatched.get("resolved_entity_id") is not None
                        else "created"
                    ),
                    "appearance_similarity": None,
                    "sam2_link_iou": None,
                    "mast3r_mask_support": None,
                }
            )
    return rows


def _semantic_coherence(
    rows: list[dict[str, object]], appearance_index: str | Path
) -> dict[str, float | None]:
    _, appearance_manifest, embeddings = load_observation_index(appearance_index)
    vector_by_proposal = {
        str(observation["proposal_id"]): embeddings[int(observation["index"])]
        for observation in appearance_manifest["observations"]
    }
    grouped: dict[str, list[np.ndarray]] = {}
    for row in rows:
        vector = vector_by_proposal.get(str(row["proposal_id"]))
        if vector is not None:
            grouped.setdefault(str(row["entity_id"]), []).append(vector)
    multiview = {key: values for key, values in grouped.items() if len(values) >= 2}
    within_scores: list[float] = []
    prototypes: list[np.ndarray] = []
    for values in multiview.values():
        matrix = np.asarray(values, dtype=np.float64)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        prototype = np.mean(matrix, axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        within_scores.extend((matrix @ prototype).tolist())
        prototypes.append(prototype)
    nearest_scores: list[float] = []
    if len(prototypes) >= 2:
        prototype_matrix = np.stack(prototypes)
        similarities = prototype_matrix @ prototype_matrix.T
        np.fill_diagonal(similarities, -np.inf)
        nearest_scores = np.max(similarities, axis=1).tolist()
    return {
        "semantic_within_cosine": _mean(within_scores),
        "nearest_entity_cosine_median": _median(nearest_scores),
        "multiview_entity_fraction": (
            len(multiview) / len(grouped) if grouped else None
        ),
    }


def evaluate_variant(label: str, path: Path) -> dict[str, object]:
    manifest_path = _manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _assignment_rows(manifest)
    entities: dict[str, list[dict[str, object]]] = {}
    tracks: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        entities.setdefault(str(row["entity_id"]), []).append(row)
        if row["track_id"] is not None:
            tracks.setdefault(str(row["track_id"]), []).append(row)

    track_entity_counts: list[int] = []
    switches = 0
    transitions = 0
    for track_rows in tracks.values():
        ordered = sorted(track_rows, key=lambda item: int(item["frame_id"]))
        sequence: list[str] = []
        for row in ordered:
            entity_id = str(row["entity_id"])
            if not sequence or sequence[-1] != entity_id:
                sequence.append(entity_id)
        track_entity_counts.append(len(set(sequence)))
        switches += max(0, len(sequence) - 1)
        transitions += max(0, len(ordered) - 1)

    entity_track_counts = [
        len({str(row["track_id"]) for row in entity_rows if row["track_id"] is not None})
        for entity_rows in entities.values()
    ]
    matched = sum(row["status"] == "matched" for row in rows)
    created = sum(row["status"] == "created" for row in rows)
    duplicates = sum(row["status"] == "duplicate" for row in rows)
    total = len(rows)
    metrics: dict[str, object] = {
        "variant": label,
        "semantic_model": manifest.get("appearance_model"),
        "cues": "+".join(manifest.get("association_cues", [])) or "legacy-full",
        "frames": len(manifest["frames"]),
        "observations": total,
        "entities": len(entities),
        "matched_rate": matched / total if total else None,
        "birth_rate": created / total if total else None,
        "duplicate_rate": duplicates / total if total else None,
        "fragmented_track_fraction": (
            sum(count > 1 for count in track_entity_counts) / len(track_entity_counts)
            if track_entity_counts
            else None
        ),
        "entities_per_track_median": _median(track_entity_counts),
        "track_switch_rate": switches / transitions if transitions else None,
        "observations_per_entity_median": _median(
            len(entity_rows) for entity_rows in entities.values()
        ),
        "tracklets_per_entity_median": _median(entity_track_counts),
        "appearance_similarity_on_matches": _mean(
            float(row["appearance_similarity"])
            for row in rows
            if row["appearance_similarity"] is not None
        ),
        "sam2_iou_on_matches": _mean(
            float(row["sam2_link_iou"])
            for row in rows
            if row["sam2_link_iou"] is not None
        ),
        "mast3r_support_on_matches": _mean(
            float(row["mast3r_mask_support"])
            for row in rows
            if row["mast3r_mask_support"] is not None
        ),
        "mapping_seconds": manifest.get("timing", {}).get("total_seconds"),
        "mapping_fps": manifest.get("timing", {}).get("frames_per_second"),
    }
    metrics.update(
        _semantic_coherence(rows, str(manifest["source_appearance_index"]))
    )
    return metrics


def _display(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _leader(
    rows: list[dict[str, object]], key: str, *, maximize: bool
) -> tuple[str, object] | None:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return None
    winner = (max if maximize else min)(eligible, key=lambda row: float(row[key]))
    return str(winner["variant"]), winner[key]


def write_report(rows: list[dict[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps({"format": "fact3r-image-uot-ablation", "version": 1, "variants": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table_fields = [
        "variant",
        "cues",
        "entities",
        "matched_rate",
        "birth_rate",
        "fragmented_track_fraction",
        "track_switch_rate",
        "semantic_within_cosine",
        "nearest_entity_cosine_median",
        "mapping_fps",
    ]
    header = "| " + " | ".join(table_fields) + " |"
    separator = "| " + " | ".join("---" for _ in table_fields) + " |"
    body = [
        "| " + " | ".join(_display(row.get(field)) for field in table_fields) + " |"
        for row in rows
    ]
    leaders = [
        ("Lowest SAM2-track fragmentation", "fragmented_track_fraction", False),
        ("Lowest entity-switch rate", "track_switch_rate", False),
        ("Highest within-entity semantic coherence", "semantic_within_cosine", True),
        ("Fastest UOT mapping", "mapping_fps", True),
    ]
    leader_lines = []
    for title, key, maximize in leaders:
        result = _leader(rows, key, maximize=maximize)
        if result is not None:
            leader_lines.append(f"- {title}: **{result[0]}** ({_display(result[1])})")
    report = "\n".join(
        [
            "# Fact3R image-UOT ablation",
            "",
            header,
            separator,
            *body,
            "",
            "## Metric leaders",
            "",
            *leader_lines,
            "",
            "These are label-free diagnostics, not object-retrieval accuracy. A low birth rate can",
            "also indicate harmful over-merging. Use the annotated retrieval evaluation before",
            "claiming one configuration is best overall.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=_parse_variant, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [evaluate_variant(label, path) for label, path in args.variant]
    write_report(rows, args.output)
    print(f"Wrote ablation report to {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
