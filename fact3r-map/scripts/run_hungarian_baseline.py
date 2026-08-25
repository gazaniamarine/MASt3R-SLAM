#!/usr/bin/env python3
"""Run the frame-level Hungarian baseline over saved official-SAM2 proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.association import (  # noqa: E402
    HungarianEntityMapper,
    HungarianMapConfig,
    PairwiseCostConfig,
    UnmatchedReason,
)
from fact3r.proposals.storage import (  # noqa: E402
    iter_saved_proposal_frames,
    load_proposal_run_manifest,
)


def _default_output(proposal_directory: Path) -> Path:
    return (
        proposal_directory.parent.parent
        / "fact3r_hungarian"
        / proposal_directory.name
    )


def _save_cost_evidence(output: Path, result) -> str:
    frame_directory = output / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    filename = f"frame_{result.frame_id:06d}_costs.npz"
    matrix = result.assignment.cost_matrix
    payload = {
        "costs": matrix.costs,
        "candidate_mask": matrix.candidate_mask,
        "proposal_ids": np.asarray(matrix.proposal_ids, dtype=np.str_),
        "entity_ids": np.asarray(matrix.entity_ids, dtype=np.str_),
    }
    payload.update(
        {f"component_{name}": values for name, values in matrix.components.items()}
    )
    np.savez_compressed(frame_directory / filename, **payload)
    return f"frames/{filename}"


def _save_entities(output: Path, entities) -> list[dict[str, object]]:
    entity_directory = output / "entities"
    entity_directory.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for entity in entities:
        filename = f"{entity.id}.npz"
        payload = {"geometry": entity.surfel_or_voxel_geometry}
        if entity.mast3r_descriptor_bank is not None:
            payload["mast3r_descriptor_bank"] = entity.mast3r_descriptor_bank
        if entity.descriptor_confidence is not None:
            payload["descriptor_confidence"] = entity.descriptor_confidence
        np.savez_compressed(entity_directory / filename, **payload)
        colour = entity.colour_statistics.get(
            "mean_rgb", entity.colour_statistics.get("median_rgb")
        )
        entries.append(
            {
                "entity_id": entity.id,
                "status": entity.status.value,
                "file": f"entities/{filename}",
                "centroid_xyz": entity.centroid_xyz.tolist(),
                "bounding_box_xyz": entity.bounding_box_xyz.tolist(),
                "mean_rgb": None if colour is None else np.asarray(colour).tolist(),
                "observation_count": entity.observation_count,
                "first_seen_timestamp": entity.first_seen_timestamp,
                "last_seen_timestamp": entity.last_seen_timestamp,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-any-backend",
        action="store_true",
        help="allow a non-official SAM2 proposal manifest",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-match-cost", type=float, default=0.65)
    parser.add_argument("--max-centroid-distance", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=float, default=0.05)
    parser.add_argument("--geometry-match-distance", type=float, default=0.08)
    parser.add_argument("--max-geometry-points", type=int, default=256)
    parser.add_argument("--entity-voxel-size", type=float, default=0.04)
    parser.add_argument("--max-entity-points", type=int, default=4096)
    args = parser.parse_args()

    run_manifest = load_proposal_run_manifest(args.proposals)
    backend = run_manifest.get("backend")
    if not args.allow_any_backend and backend != "official":
        raise ValueError(
            f"expected an official-SAM2 proposal run, found backend={backend!r}; "
            "pass --allow-any-backend to override"
        )

    pairwise_config = PairwiseCostConfig(
        max_centroid_distance_m=args.max_centroid_distance,
        bounding_box_padding_m=args.bbox_padding,
        geometry_match_distance_m=args.geometry_match_distance,
        max_geometry_points=args.max_geometry_points,
    )
    map_config = HungarianMapConfig(
        pairwise_cost=pairwise_config,
        max_match_cost=args.max_match_cost,
        entity_voxel_size_m=args.entity_voxel_size,
        max_entity_points=args.max_entity_points,
    )
    mapper = HungarianEntityMapper(map_config)
    output = args.output or _default_output(args.proposals)
    output.mkdir(parents=True, exist_ok=True)

    frame_entries: list[dict[str, object]] = []
    unmatched_reason_totals = {reason.value: 0 for reason in UnmatchedReason}
    for frame_index, frame in enumerate(
        iter_saved_proposal_frames(args.proposals)
    ):
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        result = mapper.process_frame(
            frame.proposals,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )
        evidence_file = _save_cost_evidence(output, result)
        unmatched_reason_counts = result.assignment.unmatched_reason_counts
        for reason, count in unmatched_reason_counts.items():
            unmatched_reason_totals[reason] += count
        unmatched_entries = [
            {
                "proposal_id": unmatched.proposal_id,
                "reason": unmatched.reason.value,
                "best_entity_id": unmatched.best_entity_id,
                "best_cost": unmatched.best_cost,
                "created_entity_id": created_entity_id,
            }
            for unmatched, created_entity_id in zip(
                result.assignment.unmatched_proposals,
                result.created_entity_ids,
                strict=True,
            )
        ]
        frame_entries.append(
            {
                "frame_id": result.frame_id,
                "timestamp": result.timestamp,
                "proposal_count": result.proposal_count,
                "entity_count_before": result.entity_count_before,
                "entity_count_after": result.entity_count_after,
                "cost_evidence": evidence_file,
                "matches": [
                    {
                        "proposal_id": match.proposal_id,
                        "entity_id": match.entity_id,
                        "cost": match.cost,
                    }
                    for match in result.assignment.matches
                ],
                "created_entity_ids": list(result.created_entity_ids),
                "unmatched_reason_counts": unmatched_reason_counts,
                "unmatched_proposals": unmatched_entries,
                "unobserved_entity_ids": list(
                    result.assignment.unmatched_entity_ids
                ),
            }
        )
        reason_summary = ",".join(
            f"{reason}={count}"
            for reason, count in unmatched_reason_counts.items()
            if count
        )
        print(
            f"frame {frame.frame_id}: proposals={len(frame.proposals)} "
            f"matched={len(result.assignment.matches)} "
            f"created={len(result.created_entity_ids)} "
            f"entities={result.entity_count_after} "
            f"unmatched[{reason_summary or 'none'}]"
        )

    entity_entries = _save_entities(output, mapper.entities)
    output_manifest = {
        "format": "fact3r-hungarian-baseline",
        "version": 2,
        "source_proposals": str(args.proposals.resolve()),
        "source_backend": backend,
        "source_model": run_manifest.get("model"),
        "pairwise_cost_config": asdict(pairwise_config),
        "map_config": {
            "max_match_cost": map_config.max_match_cost,
            "entity_voxel_size_m": map_config.entity_voxel_size_m,
            "max_entity_points": map_config.max_entity_points,
            "max_descriptor_samples": map_config.max_descriptor_samples,
        },
        "frame_count": len(frame_entries),
        "entity_count": len(entity_entries),
        "unmatched_reason_totals": unmatched_reason_totals,
        "frames": frame_entries,
        "entities": entity_entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    reason_summary = ", ".join(
        f"{reason}={count}"
        for reason, count in unmatched_reason_totals.items()
    )
    print(f"Unmatched reason totals: {reason_summary}")
    print(f"Wrote Hungarian baseline map to {manifest_path}")


if __name__ == "__main__":
    main()
