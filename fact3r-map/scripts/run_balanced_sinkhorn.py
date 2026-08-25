#!/usr/bin/env python3
"""Run balanced Sinkhorn over saved complete-frame SAM2 proposals."""

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
    BalancedSinkhornConfig,
    BalancedSinkhornEntityMapper,
    HungarianMapConfig,
    PairwiseCostConfig,
    TemporalEntityHint,
    UnmatchedReason,
    load_tracklet_run,
)
from fact3r.proposals.storage import (  # noqa: E402
    iter_saved_proposal_frames,
    load_proposal_run_manifest,
)


def _default_output(proposal_directory: Path) -> Path:
    return (
        proposal_directory.parent.parent
        / "fact3r_balanced_sinkhorn"
        / proposal_directory.name
    )


def _save_transport_evidence(output: Path, result) -> str:
    frame_directory = output / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    filename = f"frame_{result.frame_id:06d}_transport.npz"
    assignment = result.assignment
    matrix = assignment.cost_matrix
    payload = {
        "costs": matrix.costs,
        "candidate_mask": matrix.candidate_mask,
        "proposal_ids": np.asarray(matrix.proposal_ids, dtype=np.str_),
        "entity_ids": np.asarray(matrix.entity_ids, dtype=np.str_),
        "transport_plan": assignment.transport_plan,
        "proposal_marginals": assignment.proposal_marginals,
        "entity_marginals": assignment.entity_marginals,
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
    parser.add_argument("--tracklets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-match-cost", type=float, default=0.65)
    parser.add_argument("--max-centroid-distance", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=float, default=0.05)
    parser.add_argument("--geometry-match-distance", type=float, default=0.08)
    parser.add_argument("--max-geometry-points", type=int, default=256)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--entropy-temperature", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--marginal-tolerance", type=float, default=1e-6)
    parser.add_argument("--noncandidate-cost", type=float, default=2.0)
    parser.add_argument("--entity-voxel-size", type=float, default=0.04)
    parser.add_argument("--max-entity-points", type=int, default=4096)
    args = parser.parse_args()

    proposal_run = load_proposal_run_manifest(args.proposals)
    if proposal_run.get("backend") != "official":
        raise ValueError("balanced comparison expects official-SAM2 proposals")
    pairwise_config = PairwiseCostConfig(
        max_centroid_distance_m=args.max_centroid_distance,
        bounding_box_padding_m=args.bbox_padding,
        geometry_match_distance_m=args.geometry_match_distance,
        max_geometry_points=args.max_geometry_points,
        temporal_weight=args.temporal_weight,
    )
    map_config = HungarianMapConfig(
        pairwise_cost=pairwise_config,
        max_match_cost=args.max_match_cost,
        entity_voxel_size_m=args.entity_voxel_size,
        max_entity_points=args.max_entity_points,
    )
    sinkhorn_config = BalancedSinkhornConfig(
        entropy_temperature=args.entropy_temperature,
        max_iterations=args.max_iterations,
        marginal_tolerance=args.marginal_tolerance,
        noncandidate_cost=args.noncandidate_cost,
    )
    mapper = BalancedSinkhornEntityMapper(map_config, sinkhorn_config)
    tracklet_run = (
        None if args.tracklets is None else load_tracklet_run(args.tracklets)
    )
    output = args.output or _default_output(args.proposals)
    output.mkdir(parents=True, exist_ok=True)

    frame_entries: list[dict[str, object]] = []
    previous_proposal_entities: dict[str, str] = {}
    unmatched_totals = {reason.value: 0 for reason in UnmatchedReason}
    matched_total = 0
    created_total = 0
    temporal_hint_total = 0
    temporal_hint_honored_total = 0
    noncandidate_masses: list[float] = []
    marginal_errors: list[float] = []
    nonconverged_frames = 0

    for frame_index, frame in enumerate(
        iter_saved_proposal_frames(args.proposals)
    ):
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        tracklet_observations = (
            ()
            if tracklet_run is None
            else tracklet_run.observations_by_frame.get(frame.frame_id, ())
        )
        observations_by_proposal = {
            observation.proposal_id: observation
            for observation in tracklet_observations
        }
        temporal_hints: dict[str, TemporalEntityHint] = {}
        for proposal in frame.proposals:
            observation = observations_by_proposal.get(proposal.proposal_id)
            if (
                observation is None
                or observation.source_proposal_id is None
                or observation.link_iou is None
            ):
                continue
            entity_id = previous_proposal_entities.get(
                observation.source_proposal_id
            )
            if entity_id is not None:
                temporal_hints[proposal.proposal_id] = TemporalEntityHint(
                    entity_id=entity_id,
                    confidence=observation.link_iou,
                )

        result = mapper.process_frame(
            frame.proposals,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            temporal_hints=temporal_hints,
        )
        assignment = result.assignment
        current_proposal_entities = {
            match.proposal_id: match.entity_id for match in assignment.matches
        }
        current_proposal_entities.update(
            {
                frame.proposals[proposal_index].proposal_id: entity_id
                for proposal_index, entity_id in zip(
                    assignment.unmatched_proposal_indices,
                    result.created_entity_ids,
                    strict=True,
                )
            }
        )
        honored_hints = {
            match.proposal_id
            for match in assignment.matches
            if match.proposal_id in temporal_hints
            and temporal_hints[match.proposal_id].entity_id == match.entity_id
        }
        reason_counts = assignment.unmatched_reason_counts
        for reason, count in reason_counts.items():
            unmatched_totals[reason] += count
        matched_total += len(assignment.matches)
        created_total += len(result.created_entity_ids)
        temporal_hint_total += len(temporal_hints)
        temporal_hint_honored_total += len(honored_hints)
        if assignment.iterations:
            noncandidate_masses.append(assignment.noncandidate_mass)
            marginal_errors.append(assignment.marginal_error)
        if not assignment.converged:
            nonconverged_frames += 1

        evidence_file = _save_transport_evidence(output, result)
        unmatched_entries = [
            {
                "proposal_id": unmatched.proposal_id,
                "reason": unmatched.reason.value,
                "best_entity_id": unmatched.best_entity_id,
                "best_cost": unmatched.best_cost,
                "created_entity_id": created_entity_id,
            }
            for unmatched, created_entity_id in zip(
                assignment.unmatched_proposals,
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
                "transport_evidence": evidence_file,
                "converged": assignment.converged,
                "iterations": assignment.iterations,
                "marginal_error": assignment.marginal_error,
                "noncandidate_mass": assignment.noncandidate_mass,
                "matches": [
                    {
                        "proposal_id": match.proposal_id,
                        "entity_id": match.entity_id,
                        "cost": match.cost,
                        "transport_mass": match.transport_mass,
                        "row_probability": match.row_probability,
                        "temporal_hint_entity_id": (
                            None
                            if match.proposal_id not in temporal_hints
                            else temporal_hints[match.proposal_id].entity_id
                        ),
                        "temporal_hint_honored": match.proposal_id in honored_hints,
                    }
                    for match in assignment.matches
                ],
                "created_entity_ids": list(result.created_entity_ids),
                "unmatched_reason_counts": reason_counts,
                "unmatched_proposals": unmatched_entries,
                "unobserved_entity_ids": list(assignment.unmatched_entity_ids),
                "temporal_hint_count": len(temporal_hints),
                "temporal_hint_honored_count": len(honored_hints),
            }
        )
        reason_summary = ",".join(
            f"{reason}={count}" for reason, count in reason_counts.items() if count
        )
        print(
            f"frame {frame.frame_id}: proposals={len(frame.proposals)} "
            f"matched={len(assignment.matches)} "
            f"created={len(result.created_entity_ids)} "
            f"entities={result.entity_count_after} "
            f"unmatched[{reason_summary or 'none'}] "
            f"sinkhorn[converged={assignment.converged},"
            f"iterations={assignment.iterations},"
            f"forbidden_mass={assignment.noncandidate_mass:.3e}] "
            f"tracklet_hints={len(temporal_hints)} "
            f"honored={len(honored_hints)}"
        )
        previous_proposal_entities = current_proposal_entities

    entity_entries = _save_entities(output, mapper.entities)
    output_manifest = {
        "format": "fact3r-balanced-sinkhorn",
        "version": 1,
        "source_proposals": str(args.proposals.resolve()),
        "source_tracklets": (
            None if args.tracklets is None else str(args.tracklets.resolve())
        ),
        "source_model": proposal_run.get("model"),
        "pairwise_cost_config": asdict(pairwise_config),
        "map_config": asdict(map_config),
        "sinkhorn_config": asdict(sinkhorn_config),
        "frame_count": len(frame_entries),
        "entity_count": len(entity_entries),
        "matched_total": matched_total,
        "created_total": created_total,
        "unmatched_reason_totals": unmatched_totals,
        "temporal_hint_total": temporal_hint_total,
        "temporal_hint_honored_total": temporal_hint_honored_total,
        "nonconverged_frames": nonconverged_frames,
        "max_marginal_error": max(marginal_errors, default=0.0),
        "mean_noncandidate_mass": (
            float(np.mean(noncandidate_masses)) if noncandidate_masses else 0.0
        ),
        "frames": frame_entries,
        "entities": entity_entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    reason_summary = ", ".join(
        f"{reason}={count}" for reason, count in unmatched_totals.items()
    )
    print(f"Unmatched reason totals: {reason_summary}")
    print(
        f"Balanced Sinkhorn totals: matched={matched_total}, "
        f"created={created_total}, entities={len(entity_entries)}, "
        f"nonconverged_frames={nonconverged_frames}, "
        f"mean_forbidden_mass={output_manifest['mean_noncandidate_mass']:.3e}"
    )
    print(
        f"Tracklet hints: used={temporal_hint_total}, "
        f"honored={temporal_hint_honored_total}"
    )
    print(f"Wrote balanced Sinkhorn map to {manifest_path}")


if __name__ == "__main__":
    main()
