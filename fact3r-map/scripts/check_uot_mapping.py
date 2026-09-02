#!/usr/bin/env python3
"""Check numerical health and identity continuity of a saved UOT mapping."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return None if not materialized else float(statistics.fmean(materialized))


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return None if not materialized else float(statistics.median(materialized))


def _track_id(item: dict[str, object]) -> str | None:
    value = item.get("track_id")
    if value is None and isinstance(item.get("tracklet"), dict):
        value = item["tracklet"].get("track_id")
    return None if value is None else str(value)


def _assignment_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in manifest.get("frames", []):
        frame_id = int(frame["frame_id"])
        for item in frame.get("matches", []):
            rows.append(
                {
                    "frame_id": frame_id,
                    "proposal_id": str(item["proposal_id"]),
                    "entity_id": str(item["entity_id"]),
                    "track_id": _track_id(item),
                    "status": "matched",
                    "conditional_probability": item.get(
                        "conditional_probability"
                    ),
                    "retained_ratio": item.get("retained_ratio"),
                    "cost": item.get("cost"),
                }
            )
        for item in frame.get("unmatched_proposals", []):
            resolved = item.get("resolved_entity_id")
            created = item.get("created_entity_id")
            entity_id = resolved if resolved is not None else created
            if entity_id is None:
                continue
            rows.append(
                {
                    "frame_id": frame_id,
                    "proposal_id": str(item["proposal_id"]),
                    "entity_id": str(entity_id),
                    "track_id": _track_id(item),
                    "status": "resolved" if resolved is not None else "created",
                    "conditional_probability": item.get(
                        "conditional_probability"
                    ),
                    "retained_ratio": item.get("retained_ratio"),
                    "cost": item.get("cost"),
                }
            )
    return rows


def summarize_uot_mapping(manifest: dict[str, object]) -> dict[str, object]:
    if manifest.get("format") != "fact3r-visibility-residual-transport":
        raise ValueError("input is not a Fact3R UOT mapping manifest")

    frames = list(manifest.get("frames", []))
    rows = _assignment_rows(manifest)
    status_counts = Counter(str(row["status"]) for row in rows)
    entities: dict[str, list[dict[str, object]]] = defaultdict(list)
    tracks: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        entities[str(row["entity_id"])].append(row)
        if row["track_id"] is not None:
            tracks[str(row["track_id"])].append(row)

    repeat_tracks = {
        track_id: track_rows
        for track_id, track_rows in tracks.items()
        if len(track_rows) >= 2
    }
    fragmented_tracks = 0
    switches = 0
    transitions = 0
    for track_rows in repeat_tracks.values():
        ordered = sorted(
            track_rows,
            key=lambda row: (int(row["frame_id"]), str(row["proposal_id"])),
        )
        entity_sequence = [str(row["entity_id"]) for row in ordered]
        fragmented_tracks += len(set(entity_sequence)) > 1
        switches += sum(
            previous != current
            for previous, current in zip(entity_sequence, entity_sequence[1:])
        )
        transitions += max(0, len(entity_sequence) - 1)

    uot_frames = [
        frame for frame in frames if isinstance(frame.get("uot"), dict)
    ]
    convergence_values = [bool(frame["uot"].get("converged")) for frame in uot_frames]
    fixed_errors = [
        float(frame["uot"]["fixed_point_error"])
        for frame in uot_frames
        if frame["uot"].get("fixed_point_error") is not None
    ]
    finite_errors = [value for value in fixed_errors if math.isfinite(value)]
    reason_counts: Counter[str] = Counter()
    for frame in uot_frames:
        reason_counts.update(frame["uot"].get("unmatched_reason_counts", {}))

    assignments = len(rows)
    reused = status_counts["matched"] + status_counts["resolved"]
    repeat_track_count = len(repeat_tracks)
    convergence_rate = (
        sum(convergence_values) / len(convergence_values)
        if convergence_values
        else None
    )
    fragmentation = (
        fragmented_tracks / repeat_track_count if repeat_track_count else None
    )
    switch_rate = switches / transitions if transitions else None
    numerical_health = (
        "not-recorded"
        if not uot_frames
        else (
            "healthy"
            if all(convergence_values) and len(finite_errors) == len(fixed_errors)
            else "inspect"
        )
    )
    reuse_fraction = reused / assignments if assignments else None
    if reuse_fraction is None:
        observed_utility = "not-assessable"
    elif reuse_fraction >= 0.20 or status_counts["resolved"] > 0:
        observed_utility = "material-identity-reuse"
    elif reuse_fraction >= 0.05:
        observed_utility = "limited-identity-reuse"
    else:
        observed_utility = "little-identity-reuse-in-this-clip"

    multi_track_entities = sum(
        len(
            {
                str(row["track_id"])
                for row in entity_rows
                if row["track_id"] is not None
            }
        )
        > 1
        for entity_rows in entities.values()
    )
    summary: dict[str, object] = {
        "format": "fact3r-uot-diagnostic",
        "version": 1,
        "mapping_mode": manifest.get("mode"),
        "frames": len(frames),
        "assignments": assignments,
        "entities": len(entities),
        "matches": status_counts["matched"],
        "births": status_counts["created"],
        "resolved_duplicate_observations": status_counts["resolved"],
        "identity_reuse_fraction": reuse_fraction,
        "observations_per_entity_median": _median(
            len(entity_rows) for entity_rows in entities.values()
        ),
        "tracked_observations": sum(
            row["track_id"] is not None for row in rows
        ),
        "tracks": len(tracks),
        "repeat_tracks": repeat_track_count,
        "fragmented_repeat_tracks": fragmented_tracks,
        "fragmented_repeat_track_fraction": fragmentation,
        "track_identity_switches": switches,
        "track_transition_count": transitions,
        "track_identity_switch_rate": switch_rate,
        "entities_joining_multiple_tracklets": multi_track_entities,
        "match_conditional_probability_median": _median(
            float(row["conditional_probability"])
            for row in rows
            if row["status"] == "matched"
            and row["conditional_probability"] is not None
        ),
        "match_retained_ratio_median": _median(
            float(row["retained_ratio"])
            for row in rows
            if row["status"] == "matched" and row["retained_ratio"] is not None
        ),
        "match_cost_mean": _mean(
            float(row["cost"])
            for row in rows
            if row["status"] == "matched" and row["cost"] is not None
        ),
        "uot_frames_with_diagnostics": len(uot_frames),
        "uot_convergence_rate": convergence_rate,
        "uot_nonconverged_frames": (
            sum(not value for value in convergence_values)
            if convergence_values
            else manifest.get("nonconverged_frames")
        ),
        "uot_fixed_point_error_max": (
            max(finite_errors)
            if finite_errors
            else manifest.get("max_fixed_point_error")
        ),
        "unmatched_reason_totals": dict(sorted(reason_counts.items())),
        "numerical_health": numerical_health,
        "observed_utility": observed_utility,
        "interpretation": (
            "Label-free diagnostics can show solver failure, fragmentation, and "
            "identity reuse, but cannot prove that a merge is semantically correct. "
            "Inspect retrieved evidence or use annotated identities for that."
        ),
    }
    return summary


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def print_summary(summary: dict[str, object]) -> None:
    print("UOT diagnostic")
    print(
        f"  numerical solver: {summary['numerical_health']} "
        f"(converged={_percent(summary['uot_convergence_rate'])}, "
        f"nonconverged={summary['uot_nonconverged_frames']})"
    )
    print(
        f"  identity reuse:   {_percent(summary['identity_reuse_fraction'])} "
        f"({summary['matches']} matches + "
        f"{summary['resolved_duplicate_observations']} resolved duplicates)"
    )
    print(
        f"  entity births:    {summary['births']} from "
        f"{summary['assignments']} assigned observations"
    )
    print(
        f"  track continuity: fragmented="
        f"{_percent(summary['fragmented_repeat_track_fraction'])}, "
        f"switch-rate={_percent(summary['track_identity_switch_rate'])}"
    )
    print(
        f"  association confidence median: "
        f"{summary['match_conditional_probability_median']}"
    )
    print(f"  observed utility: {summary['observed_utility']}")
    print("  Important: these checks cannot detect semantically wrong over-merges.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = _manifest_path(args.mapping)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = summarize_uot_mapping(manifest)
    summary["source_mapping"] = str(manifest_path.resolve())
    print_summary(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"  saved report: {args.output}")


if __name__ == "__main__":
    main()
