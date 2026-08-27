#!/usr/bin/env python3
"""Run visibility-conditioned unbalanced transport over saved SAM2 proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.association import (  # noqa: E402
    AppearanceMemoryConfig,
    BirthCommitmentStatus,
    DelayedCommitmentConfig,
    HungarianMapConfig,
    PairwiseCostConfig,
    ResidualTransportConfig,
    ResidualUnmatchedReason,
    TemporalEntityHint,
    VisibilityConfig,
    VisibilityResidualEntityMapper,
    load_tracklet_run,
)
from fact3r.integrations.mast3r_slam import (  # noqa: E402
    iter_exported_keyframes,
)
from fact3r.proposals.storage import (  # noqa: E402
    iter_saved_proposal_frames,
    load_proposal_run_manifest,
)
from fact3r.semantics.appearance_memory import (  # noqa: E402
    AppearanceReliabilityConfig,
    load_siglip_appearance_index,
)


def _default_output(
    proposal_directory: Path, *, delayed_commitment: bool
) -> Path:
    stage = (
        "fact3r_delayed_commitment_uot"
        if delayed_commitment
        else "fact3r_visibility_residual_transport"
    )
    return (
        proposal_directory.parent.parent
        / stage
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
        "proposal_masses": assignment.proposal_masses,
        "entity_masses": assignment.entity_masses,
        "transported_proposal_masses": assignment.transported_proposal_masses,
        "transported_entity_masses": assignment.transported_entity_masses,
        "proposal_birth_residuals": assignment.proposal_birth_residuals,
        "entity_miss_residuals": assignment.entity_miss_residuals,
        "proposal_excess_masses": assignment.proposal_excess_masses,
        "entity_excess_masses": assignment.entity_excess_masses,
        "proposal_quality": assignment.proposal_quality,
        "proposal_relaxation": assignment.proposal_relaxation,
        "entity_relaxation": assignment.entity_relaxation,
        "entity_visibility": np.asarray(
            [item.score for item in assignment.entity_visibility],
            dtype=np.float64,
        ),
        "entity_in_frustum_fraction": np.asarray(
            [item.in_frustum_fraction for item in assignment.entity_visibility],
            dtype=np.float64,
        ),
        "entity_unoccluded_fraction": np.asarray(
            [item.unoccluded_fraction for item in assignment.entity_visibility],
            dtype=np.float64,
        ),
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
        if entity.appearance_descriptor_bank is not None:
            payload["appearance_descriptor_bank"] = (
                entity.appearance_descriptor_bank
            )
        if entity.appearance_reliability is not None:
            payload["appearance_reliability"] = entity.appearance_reliability
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
                "appearance_view_count": (
                    0
                    if entity.appearance_descriptor_bank is None
                    else len(entity.appearance_descriptor_bank)
                ),
                "mean_appearance_reliability": (
                    None
                    if entity.appearance_reliability is None
                    else float(np.mean(entity.appearance_reliability))
                ),
                "first_seen_timestamp": entity.first_seen_timestamp,
                "last_seen_timestamp": entity.last_seen_timestamp,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--tracklets", type=Path)
    parser.add_argument(
        "--appearance-index",
        type=Path,
        help="pre-UOT SigLIP observation index built without --mapping",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-match-cost", type=float, default=0.65)
    parser.add_argument("--max-centroid-distance", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=float, default=0.05)
    parser.add_argument("--geometry-match-distance", type=float, default=0.08)
    parser.add_argument("--max-geometry-points", type=int, default=256)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument(
        "--appearance-weight", type=float, default=0.25
    )
    parser.add_argument("--appearance-temperature", type=float, default=0.07)
    parser.add_argument("--entropy-temperature", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--fixed-point-tolerance", type=float, default=1e-7)
    parser.add_argument("--proposal-relaxation-min", type=float, default=0.05)
    parser.add_argument("--proposal-relaxation-max", type=float, default=0.50)
    parser.add_argument("--entity-relaxation-min", type=float, default=0.02)
    parser.add_argument("--entity-relaxation-max", type=float, default=0.50)
    parser.add_argument("--mass-floor", type=float, default=1e-3)
    parser.add_argument("--geometry-retention-weight", type=float, default=0.70)
    parser.add_argument("--min-retained-ratio", type=float, default=0.25)
    parser.add_argument("--min-conditional-probability", type=float, default=0.50)
    parser.add_argument("--visibility-max-points", type=int, default=512)
    parser.add_argument("--visibility-depth-tolerance", type=float, default=0.15)
    parser.add_argument("--unknown-depth-visibility", type=float, default=0.0)
    parser.add_argument("--entity-voxel-size", type=float, default=0.04)
    parser.add_argument("--max-entity-points", type=int, default=4096)
    parser.add_argument(
        "--appearance-max-views", type=int, default=8
    )
    parser.add_argument(
        "--appearance-max-redundant-similarity", type=float, default=0.95
    )
    parser.add_argument(
        "--appearance-min-update-reliability", type=float, default=0.45
    )
    parser.add_argument(
        "--appearance-min-conditional-probability", type=float, default=0.70
    )
    parser.add_argument(
        "--appearance-min-retained-ratio", type=float, default=0.50
    )
    parser.add_argument("--appearance-min-track-iou", type=float, default=0.60)
    parser.add_argument(
        "--appearance-reference-mask-area", type=float, default=4096.0
    )
    parser.add_argument(
        "--appearance-missing-track-quality", type=float, default=0.75
    )
    parser.add_argument(
        "--delayed-commitment",
        action="store_true",
        help=(
            "accumulate unmatched birth residual by SAM2 track ID before "
            "creating an entity"
        ),
    )
    parser.add_argument("--birth-min-observations", type=int, default=3)
    parser.add_argument(
        "--birth-min-mean-residual-ratio", type=float, default=0.55
    )
    parser.add_argument("--birth-min-median-link-iou", type=float, default=0.60)
    parser.add_argument("--birth-max-centroid-step", type=float, default=0.30)
    parser.add_argument("--birth-max-missed-frames", type=int, default=0)
    args = parser.parse_args()

    if args.delayed_commitment and args.tracklets is None:
        raise ValueError("--delayed-commitment requires --tracklets")

    proposal_run = load_proposal_run_manifest(args.proposals)
    if proposal_run.get("backend") != "official":
        raise ValueError("residual transport expects official-SAM2 proposals")
    pairwise_config = PairwiseCostConfig(
        max_centroid_distance_m=args.max_centroid_distance,
        bounding_box_padding_m=args.bbox_padding,
        geometry_match_distance_m=args.geometry_match_distance,
        max_geometry_points=args.max_geometry_points,
        appearance_weight=args.appearance_weight,
        appearance_temperature=args.appearance_temperature,
        temporal_weight=args.temporal_weight,
    )
    appearance_memory_config = AppearanceMemoryConfig(
        max_views=args.appearance_max_views,
        max_redundant_similarity=(
            args.appearance_max_redundant_similarity
        ),
        min_update_reliability=args.appearance_min_update_reliability,
        min_conditional_probability=(
            args.appearance_min_conditional_probability
        ),
        min_retained_ratio=args.appearance_min_retained_ratio,
        min_track_iou=args.appearance_min_track_iou,
    )
    map_config = HungarianMapConfig(
        pairwise_cost=pairwise_config,
        max_match_cost=args.max_match_cost,
        entity_voxel_size_m=args.entity_voxel_size,
        max_entity_points=args.max_entity_points,
        appearance_memory=appearance_memory_config,
    )
    appearance_reliability_config = AppearanceReliabilityConfig(
        reference_mask_area=args.appearance_reference_mask_area,
        missing_track_quality=args.appearance_missing_track_quality,
    )
    transport_config = ResidualTransportConfig(
        entropy_temperature=args.entropy_temperature,
        max_iterations=args.max_iterations,
        fixed_point_tolerance=args.fixed_point_tolerance,
        proposal_relaxation_min=args.proposal_relaxation_min,
        proposal_relaxation_max=args.proposal_relaxation_max,
        entity_relaxation_min=args.entity_relaxation_min,
        entity_relaxation_max=args.entity_relaxation_max,
        mass_floor=args.mass_floor,
        geometry_retention_weight=args.geometry_retention_weight,
        min_retained_ratio=args.min_retained_ratio,
        min_conditional_probability=args.min_conditional_probability,
    )
    visibility_config = VisibilityConfig(
        max_entity_points=args.visibility_max_points,
        depth_tolerance_m=args.visibility_depth_tolerance,
        unknown_depth_visibility=args.unknown_depth_visibility,
    )
    delayed_commitment_config = (
        None
        if not args.delayed_commitment
        else DelayedCommitmentConfig(
            min_observations=args.birth_min_observations,
            min_mean_birth_residual_ratio=(
                args.birth_min_mean_residual_ratio
            ),
            min_median_link_iou=args.birth_min_median_link_iou,
            max_centroid_step_m=args.birth_max_centroid_step,
            max_missed_frames=args.birth_max_missed_frames,
        )
    )
    mapper = VisibilityResidualEntityMapper(
        map_config,
        transport_config,
        visibility_config,
        delayed_commitment_config,
    )
    tracklet_run = (
        None if args.tracklets is None else load_tracklet_run(args.tracklets)
    )
    appearance_index = (
        None
        if args.appearance_index is None
        else load_siglip_appearance_index(args.appearance_index)
    )
    if appearance_index is not None:
        indexed_proposals = Path(
            str(appearance_index.manifest["source_proposals"])
        ).resolve()
        if indexed_proposals != args.proposals.resolve():
            raise ValueError(
                "appearance index was built from a different proposal run"
            )
    output = args.output or _default_output(
        args.proposals, delayed_commitment=args.delayed_commitment
    )
    output.mkdir(parents=True, exist_ok=True)

    frame_entries: list[dict[str, object]] = []
    previous_proposal_entities: dict[str, str] = {}
    unmatched_totals = {reason.value: 0 for reason in ResidualUnmatchedReason}
    matched_total = 0
    created_total = 0
    temporal_hint_total = 0
    temporal_hint_honored_total = 0
    nonconverged_frames = 0
    fixed_point_errors: list[float] = []
    birth_residual_sum = 0.0
    proposal_mass_sum = 0.0
    miss_residual_sum = 0.0
    entity_mass_sum = 0.0
    fallback_frames = 0
    deferred_observation_total = 0
    confirmed_birth_total = 0
    held_existing_total = 0
    expired_pending_total = 0
    resolved_pending_total = 0
    peak_pending_track_count = 0
    appearance_memory_update_total = 0
    appearance_memory_rejection_totals: dict[str, int] = {}
    sentinel = object()

    paired_frames = zip_longest(
        iter_saved_proposal_frames(args.proposals),
        iter_exported_keyframes(args.keyframes),
        fillvalue=sentinel,
    )
    for frame_index, (frame, keyframe) in enumerate(paired_frames):
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        if frame is sentinel or keyframe is sentinel:
            raise ValueError("keyframe and proposal runs have different lengths")
        if frame.frame_id != keyframe.frame_id:
            raise ValueError(
                f"frame mismatch: proposals={frame.frame_id}, "
                f"keyframe={keyframe.frame_id}"
            )
        observations = (
            ()
            if tracklet_run is None
            else tracklet_run.observations_by_frame.get(frame.frame_id, ())
        )
        observations_by_proposal = {
            observation.proposal_id: observation for observation in observations
        }
        appearance_evidence = {}
        if appearance_index is not None:
            frame, appearance_evidence = appearance_index.enrich_frame(
                frame,
                observations_by_proposal,
                appearance_reliability_config,
            )
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
            keyframe=keyframe,
            temporal_hints=temporal_hints,
            tracklet_observations=observations_by_proposal,
        )
        assignment = result.assignment
        current_proposal_entities = {
            match.proposal_id: match.entity_id for match in assignment.matches
        }
        current_proposal_entities.update(
            {
                decision.proposal_id: decision.resolved_entity_id
                for decision in result.birth_decisions
                if decision.resolved_entity_id is not None
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
        if not assignment.converged:
            nonconverged_frames += 1
        if assignment.iterations:
            fixed_point_errors.append(assignment.fixed_point_error)
        birth_residual_sum += float(assignment.proposal_birth_residuals.sum())
        proposal_mass_sum += float(assignment.proposal_masses.sum())
        miss_residual_sum += float(assignment.entity_miss_residuals.sum())
        entity_mass_sum += float(assignment.entity_masses.sum())
        if any(
            item.used_intrinsics_fallback for item in assignment.entity_visibility
        ):
            fallback_frames += 1

        decision_status_counts = {
            status.value: sum(
                decision.status == status
                for decision in result.birth_decisions
            )
            for status in BirthCommitmentStatus
        }
        deferred_observation_total += decision_status_counts[
            BirthCommitmentStatus.DEFERRED.value
        ]
        confirmed_birth_total += decision_status_counts[
            BirthCommitmentStatus.CONFIRMED.value
        ]
        held_existing_total += decision_status_counts[
            BirthCommitmentStatus.HELD_EXISTING.value
        ]
        expired_pending_total += len(result.expired_pending_track_ids)
        resolved_pending_total += len(result.resolved_pending_track_ids)
        peak_pending_track_count = max(
            peak_pending_track_count, result.pending_track_count_after
        )
        appearance_memory_update_total += sum(
            decision.updated
            for decision in result.appearance_memory_decisions
        )
        for decision in result.appearance_memory_decisions:
            for reason in decision.blocking_reasons:
                appearance_memory_rejection_totals[reason] = (
                    appearance_memory_rejection_totals.get(reason, 0) + 1
                )

        evidence_file = _save_transport_evidence(output, result)
        decisions_by_proposal = {
            decision.proposal_id: decision
            for decision in result.birth_decisions
        }
        appearance_decisions_by_proposal = {
            decision.proposal_id: decision
            for decision in result.appearance_memory_decisions
        }
        unmatched_entries = [
            {
                "proposal_id": unmatched.proposal_id,
                "reason": unmatched.reason.value,
                "best_entity_id": unmatched.best_entity_id,
                "best_cost": unmatched.best_cost,
                "retained_ratio": unmatched.retained_ratio,
                "conditional_probability": unmatched.conditional_probability,
                "birth_residual": float(
                    assignment.proposal_birth_residuals[
                        unmatched.proposal_index
                    ]
                ),
                "birth_residual_ratio": (
                    float(
                        assignment.proposal_birth_residuals[
                            unmatched.proposal_index
                        ]
                        / assignment.proposal_masses[
                            unmatched.proposal_index
                        ]
                    )
                ),
                "commitment_status": (
                    decisions_by_proposal[unmatched.proposal_id].status.value
                ),
                "track_id": (
                    decisions_by_proposal[unmatched.proposal_id].track_id
                ),
                "resolved_entity_id": (
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].resolved_entity_id
                ),
                "created_entity_id": (
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].created_entity_id
                ),
                "pending_observation_count": (
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].observation_count
                ),
                "pending_mean_birth_residual_ratio": (
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].mean_birth_residual_ratio
                ),
                "pending_median_link_iou": (
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].median_link_iou
                ),
                "pending_max_centroid_step_m": (
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].max_centroid_step_m
                ),
                "commitment_blocking_reasons": list(
                    decisions_by_proposal[
                        unmatched.proposal_id
                    ].blocking_reasons
                ),
                "appearance_evidence": (
                    None
                    if unmatched.proposal_id not in appearance_evidence
                    else asdict(appearance_evidence[unmatched.proposal_id])
                ),
            }
            for unmatched in assignment.unmatched_proposals
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
                "fixed_point_error": assignment.fixed_point_error,
                "forbidden_mass": assignment.forbidden_mass,
                "proposal_birth_residual": float(
                    assignment.proposal_birth_residuals.sum()
                ),
                "entity_miss_residual": float(
                    assignment.entity_miss_residuals.sum()
                ),
                "matches": [
                    {
                        "proposal_id": match.proposal_id,
                        "entity_id": match.entity_id,
                        "cost": match.cost,
                        "transport_mass": match.transport_mass,
                        "retained_ratio": match.retained_ratio,
                        "conditional_probability": match.conditional_probability,
                        "temporal_hint_entity_id": (
                            None
                            if match.proposal_id not in temporal_hints
                            else temporal_hints[match.proposal_id].entity_id
                        ),
                        "temporal_hint_honored": match.proposal_id in honored_hints,
                        "appearance_evidence": (
                            None
                            if match.proposal_id not in appearance_evidence
                            else asdict(appearance_evidence[match.proposal_id])
                        ),
                        "appearance_memory": asdict(
                            appearance_decisions_by_proposal[match.proposal_id]
                        ),
                    }
                    for match in assignment.matches
                ],
                "created_entity_ids": list(result.created_entity_ids),
                "birth_commitment_status_counts": decision_status_counts,
                "pending_track_count_after": result.pending_track_count_after,
                "expired_pending_track_ids": list(
                    result.expired_pending_track_ids
                ),
                "resolved_pending_track_ids": list(
                    result.resolved_pending_track_ids
                ),
                "unmatched_reason_counts": reason_counts,
                "unmatched_proposals": unmatched_entries,
                "unobserved_entity_ids": list(assignment.unmatched_entity_ids),
                "entity_visibility": [
                    asdict(item) for item in assignment.entity_visibility
                ],
                "temporal_hint_count": len(temporal_hints),
                "temporal_hint_honored_count": len(honored_hints),
                "appearance_memory_update_count": sum(
                    decision.updated
                    for decision in result.appearance_memory_decisions
                ),
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
            f"commitment[pending={result.pending_track_count_after},"
            f"confirmed={decision_status_counts['confirmed']},"
            f"held={decision_status_counts['held_existing']},"
            f"expired={len(result.expired_pending_track_ids)}] "
            f"residual[birth={assignment.proposal_birth_residuals.sum():.3f},"
            f"miss={assignment.entity_miss_residuals.sum():.3f}] "
            f"uot[converged={assignment.converged},"
            f"iterations={assignment.iterations},"
            f"forbidden_mass={assignment.forbidden_mass:.1e}] "
            f"tracklet_hints={len(temporal_hints)} honored={len(honored_hints)}"
            " appearance_updates="
            f"{sum(decision.updated for decision in result.appearance_memory_decisions)}"
        )
        previous_proposal_entities = current_proposal_entities

    entity_entries = _save_entities(output, mapper.entities)
    output_manifest = {
        "format": "fact3r-visibility-residual-transport",
        "version": 2,
        "source_keyframes": str(args.keyframes.resolve()),
        "source_proposals": str(args.proposals.resolve()),
        "source_tracklets": (
            None if args.tracklets is None else str(args.tracklets.resolve())
        ),
        "source_appearance_index": (
            None
            if appearance_index is None
            else str(appearance_index.manifest_path.resolve())
        ),
        "source_model": proposal_run.get("model"),
        "pairwise_cost_config": asdict(pairwise_config),
        "map_config": asdict(map_config),
        "appearance_reliability_config": asdict(
            appearance_reliability_config
        ),
        "transport_config": asdict(transport_config),
        "visibility_config": asdict(visibility_config),
        "delayed_commitment_config": (
            None
            if delayed_commitment_config is None
            else asdict(delayed_commitment_config)
        ),
        "frame_count": len(frame_entries),
        "entity_count": len(entity_entries),
        "matched_total": matched_total,
        "created_total": created_total,
        "unmatched_reason_totals": unmatched_totals,
        "temporal_hint_total": temporal_hint_total,
        "temporal_hint_honored_total": temporal_hint_honored_total,
        "nonconverged_frames": nonconverged_frames,
        "max_fixed_point_error": max(fixed_point_errors, default=0.0),
        "forbidden_mass": 0.0,
        "proposal_birth_residual_fraction": (
            0.0 if proposal_mass_sum == 0.0 else birth_residual_sum / proposal_mass_sum
        ),
        "entity_miss_residual_fraction": (
            0.0 if entity_mass_sum == 0.0 else miss_residual_sum / entity_mass_sum
        ),
        "visibility_fallback_frames": fallback_frames,
        "deferred_birth_observation_total": deferred_observation_total,
        "confirmed_birth_total": confirmed_birth_total,
        "held_existing_observation_total": held_existing_total,
        "expired_pending_track_total": expired_pending_total,
        "resolved_pending_track_total": resolved_pending_total,
        "peak_pending_track_count": peak_pending_track_count,
        "appearance_memory_update_total": appearance_memory_update_total,
        "appearance_memory_rejection_totals": dict(
            sorted(appearance_memory_rejection_totals.items())
        ),
        "final_pending_tracks": [
            asdict(summary) for summary in mapper.pending_births
        ],
        "committed_track_entities": dict(
            sorted(mapper.committed_track_entities.items())
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
        f"Visibility residual transport totals: matched={matched_total}, "
        f"created={created_total}, entities={len(entity_entries)}, "
        f"nonconverged_frames={nonconverged_frames}, "
        f"birth_residual_fraction="
        f"{output_manifest['proposal_birth_residual_fraction']:.3f}, "
        f"miss_residual_fraction="
        f"{output_manifest['entity_miss_residual_fraction']:.3f}, "
        "forbidden_mass=0"
    )
    if delayed_commitment_config is not None:
        print(
            "Delayed commitment totals: "
            f"deferred_observations={deferred_observation_total}, "
            f"confirmed_births={confirmed_birth_total}, "
            f"held_existing={held_existing_total}, "
            f"expired_tracks={expired_pending_total}, "
            f"peak_pending={peak_pending_track_count}, "
            f"final_pending={len(mapper.pending_births)}"
        )
    print(
        f"Tracklet hints: used={temporal_hint_total}, "
        f"honored={temporal_hint_honored_total}"
    )
    print(f"Wrote visibility residual transport map to {manifest_path}")


if __name__ == "__main__":
    main()
