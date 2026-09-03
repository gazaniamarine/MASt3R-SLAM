#!/usr/bin/env python3
"""Associate real-world 2D masks with UOT, without dense reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.association.image_uot import (  # noqa: E402
    ImageTrackEvidence,
    build_image_uot_cost_matrix,
)
from fact3r.association.residual_transport import (  # noqa: E402
    ResidualTransportConfig,
    solve_residual_transport,
)
from fact3r.association.tracklets import load_tracklet_run  # noqa: E402
from fact3r.association.visibility import EntityVisibility  # noqa: E402
from fact3r.semantics.observation_index import load_observation_index  # noqa: E402


@dataclass(slots=True)
class TrackState:
    evidence: ImageTrackEvidence
    observation_count: int


def _load_proposal_frame(
    root: Path, run_entry: dict[str, object]
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    path = root / str(run_entry["manifest"])
    frame = json.loads(path.read_text(encoding="utf-8"))
    masks = []
    for proposal in frame["proposals"]:
        with np.load(path.parent / str(proposal["file"]), allow_pickle=False) as data:
            masks.append(np.asarray(data["mask"], dtype=bool))
    return list(frame["proposals"]), masks


def _load_pairs(root: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for pair in manifest["pairs"]:
        with np.load(root / str(pair["file"]), allow_pickle=False) as data:
            result[(int(pair["source_frame_id"]), int(pair["target_frame_id"]))] = (
                np.asarray(data["source_xy"], dtype=np.int32),
                np.asarray(data["target_xy"], dtype=np.int32),
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--appearance-index", type=Path, required=True)
    parser.add_argument("--mast3r-matches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-track-gap", type=int, default=3)
    parser.add_argument("--max-match-cost", type=float, default=0.70)
    parser.add_argument(
        "--cues",
        default="appearance,sam2,mast3r",
        help=(
            "comma-separated association cues; appearance is required and "
            "sam2/mast3r may be ablated"
        ),
    )
    args = parser.parse_args()
    requested_cues = tuple(
        value.strip().lower() for value in args.cues.split(",") if value.strip()
    )
    unknown_cues = set(requested_cues) - {"appearance", "sam2", "mast3r"}
    if unknown_cues:
        raise ValueError(f"unsupported association cues: {sorted(unknown_cues)}")
    if "appearance" not in requested_cues:
        raise ValueError("image UOT ablations currently require the appearance cue")
    enabled_cues = tuple(
        cue for cue in ("appearance", "sam2", "mast3r") if cue in requested_cues
    )
    cue_weights = {
        "appearance": 0.20,
        "sam2": 0.35 if "sam2" in enabled_cues else 0.0,
        "mast3r": 0.45 if "mast3r" in enabled_cues else 0.0,
    }
    min_sam2_iou = 0.20 if "sam2" in enabled_cues else float("inf")
    min_mast3r_support = 0.08 if "mast3r" in enabled_cues else float("inf")
    started = perf_counter()

    proposal_run = json.loads(
        (args.proposals / "manifest.json").read_text(encoding="utf-8")
    )
    tracklet_run = load_tracklet_run(args.tracklets)
    _, appearance_manifest, embeddings = load_observation_index(
        args.appearance_index
    )
    embedding_by_proposal = {
        str(observation["proposal_id"]): embeddings[int(observation["index"])]
        for observation in appearance_manifest["observations"]
    }
    pairs = _load_pairs(args.mast3r_matches)
    args.output.mkdir(parents=True, exist_ok=True)
    tracks: dict[str, TrackState] = {}
    next_entity = 0
    previous_frame_id: int | None = None
    frames_output: list[dict[str, object]] = []
    committed_track_entities: dict[str, str] = {}
    total_matches = 0
    total_births = 0
    empty_frames = 0
    transport_config = ResidualTransportConfig(
        min_retained_ratio=0.15,
        min_conditional_probability=0.45,
    )

    for run_entry in proposal_run["frames"]:
        frame_id = int(run_entry["frame_id"])
        proposals, masks = _load_proposal_frame(args.proposals, run_entry)
        proposal_ids = [str(item["proposal_id"]) for item in proposals]
        if not proposal_ids:
            # SAM2 returns nothing for a featureless view -- a blank wall during
            # an in-place turn, for instance. There is nothing to associate, but
            # the frame still advances the temporal cue, so record it and move on
            # rather than stacking an empty list.
            empty_frames += 1
            previous_frame_id = frame_id
            continue
        vectors = np.stack([embedding_by_proposal[item] for item in proposal_ids])
        observations = {
            item.proposal_id: item
            for item in tracklet_run.observations_by_frame.get(frame_id, ())
        }
        source_ids = [
            None if observations.get(item) is None else observations[item].source_proposal_id
            for item in proposal_ids
        ]
        link_ious = [
            None if observations.get(item) is None else observations[item].link_iou
            for item in proposal_ids
        ]
        active = [
            state
            for state in tracks.values()
            if 0 < frame_id - state.evidence.last_frame_id <= args.max_track_gap
        ]
        pair = None if previous_frame_id is None else pairs.get((previous_frame_id, frame_id))
        source_xy, target_xy = (None, None) if pair is None else pair
        matrix = build_image_uot_cost_matrix(
            proposal_ids,
            masks,
            vectors,
            source_ids,
            link_ious,
            [state.evidence for state in active],
            frame_id=frame_id,
            pair_source_frame_id=previous_frame_id,
            source_xy=source_xy,
            target_xy=target_xy,
            appearance_weight=cue_weights["appearance"],
            sam2_weight=cue_weights["sam2"],
            mast3r_weight=cue_weights["mast3r"],
            min_sam2_iou=min_sam2_iou,
            min_mast3r_support=min_mast3r_support,
            max_track_gap=args.max_track_gap,
        )
        visibility = tuple(
            EntityVisibility(
                entity_id=state.evidence.entity_id,
                score=float(np.exp(-0.5 * (frame_id - state.evidence.last_frame_id - 1))),
                in_frustum_fraction=1.0,
                unoccluded_fraction=1.0,
                sampled_point_count=0,
                projected_point_count=0,
                visible_point_count=0,
                used_intrinsics_fallback=True,
            )
            for state in active
        )
        quality = np.asarray(
            [np.clip(float(item["score"]), 0.0, 1.0) for item in proposals]
        )
        result = solve_residual_transport(
            matrix,
            quality,
            visibility,
            config=transport_config,
            max_match_cost=args.max_match_cost,
        )
        # UOT may distribute mass from duplicate proposals to one entity. Keep
        # only its strongest observation in this frame; the others become births.
        best_by_entity = {}
        for match in result.matches:
            current = best_by_entity.get(match.entity_id)
            key = (match.conditional_probability, match.transport_mass, -match.cost)
            if current is None or key > current[0]:
                best_by_entity[match.entity_id] = (key, match)
        selected = {
            match.proposal_index: match for _, match in best_by_entity.values()
        }
        duplicate_matches = {
            match.proposal_index: match
            for match in result.matches
            if match.proposal_index not in selected
            and match.entity_id in best_by_entity
        }
        frame_matches = []
        unmatched = []
        for proposal_index, proposal_id in enumerate(proposal_ids):
            observation = observations.get(proposal_id)
            tracklet_payload = None if observation is None else {
                "track_id": observation.track_id,
                "source_proposal_id": observation.source_proposal_id,
                "link_iou": observation.link_iou,
            }
            match = selected.get(proposal_index)
            vector = vectors[proposal_index]
            duplicate = duplicate_matches.get(proposal_index)
            if match is None and duplicate is not None:
                # Several current masks transported mass to the same identity.
                # Preserve their evidence under that identity without creating
                # another entity or replacing the strongest track state.
                entity_id = duplicate.entity_id
                unmatched.append(
                    {
                        "proposal_id": proposal_id,
                        "reason": "duplicate_same_entity_transport",
                        "resolved_entity_id": entity_id,
                        "commitment_status": "held_existing",
                        "track_id": None if observation is None else observation.track_id,
                        "tracklet": tracklet_payload,
                        "cost": duplicate.cost,
                        "conditional_probability": duplicate.conditional_probability,
                    }
                )
                if observation is not None:
                    committed_track_entities[observation.track_id] = entity_id
                continue
            if match is None:
                entity_id = f"image-entity-{next_entity:06d}"
                next_entity += 1
                total_births += 1
                unmatched.append(
                    {
                        "proposal_id": proposal_id,
                        "reason": "uot_birth_residual",
                        "created_entity_id": entity_id,
                        "commitment_status": "confirmed",
                        "track_id": None if observation is None else observation.track_id,
                        "tracklet": tracklet_payload,
                    }
                )
                count = 1
                prototype = vector
            else:
                entity_id = match.entity_id
                total_matches += 1
                state = tracks[entity_id]
                count = state.observation_count + 1
                prototype = (state.evidence.prototype * state.observation_count + vector) / count
                frame_matches.append(
                    {
                        "proposal_id": proposal_id,
                        "entity_id": entity_id,
                        "cost": match.cost,
                        "transport_mass": match.transport_mass,
                        "retained_ratio": match.retained_ratio,
                        "conditional_probability": match.conditional_probability,
                        "tracklet": tracklet_payload,
                        "appearance_similarity": float(
                            matrix.components["appearance_similarity"][proposal_index, match.entity_index]
                        ),
                        "sam2_link_iou": float(
                            matrix.components["sam2_link_iou"][proposal_index, match.entity_index]
                        ),
                        "mast3r_mask_support": float(
                            matrix.components["mast3r_mask_support"][proposal_index, match.entity_index]
                        ),
                    }
                )
            tracks[entity_id] = TrackState(
                ImageTrackEvidence(
                    entity_id=entity_id,
                    last_frame_id=frame_id,
                    last_proposal_id=proposal_id,
                    prototype=np.asarray(prototype, dtype=np.float32),
                    last_mask=masks[proposal_index],
                ),
                observation_count=count,
            )
            if observation is not None:
                committed_track_entities[observation.track_id] = entity_id
        frames_output.append(
            {
                "frame_id": frame_id,
                "matches": frame_matches,
                "unmatched_proposals": unmatched,
                "uot": {
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "fixed_point_error": result.fixed_point_error,
                    "unmatched_reason_counts": result.unmatched_reason_counts,
                },
            }
        )
        previous_frame_id = frame_id
        print(
            f"frame {frame_id}: proposals={len(proposals)} "
            f"uot_matches={len(frame_matches)} births={len(unmatched)}"
        )

    elapsed_seconds = perf_counter() - started
    manifest = {
        "format": "fact3r-visibility-residual-transport",
        "version": 1,
        "mode": "image_uot_no_dense_reconstruction",
        "source_proposals": str(args.proposals.resolve()),
        "source_tracklets": str((args.tracklets / "manifest.json").resolve()),
        "source_appearance_index": str((args.appearance_index / "manifest.json").resolve()),
        "source_mast3r_pair_matches": str((args.mast3r_matches / "manifest.json").resolve()),
        "appearance_model": appearance_manifest.get("model"),
        "association_cues": list(enabled_cues),
        "frames_without_proposals": empty_frames,
        "association_config": {
            "cue_weights": cue_weights,
            "min_sam2_iou": None if not np.isfinite(min_sam2_iou) else min_sam2_iou,
            "min_mast3r_support": (
                None
                if not np.isfinite(min_mast3r_support)
                else min_mast3r_support
            ),
            "min_reidentification_similarity": 0.80,
            "max_track_gap": args.max_track_gap,
            "max_match_cost": args.max_match_cost,
        },
        "entity_count": len(tracks),
        "matched_total": total_matches,
        "created_total": total_births,
        "committed_track_entities": committed_track_entities,
        "timing": {
            "total_seconds": elapsed_seconds,
            "frames_per_second": (
                len(frames_output) / elapsed_seconds if elapsed_seconds > 0 else None
            ),
        },
        "frames": frames_output,
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if empty_frames:
        print(
            f"{empty_frames} frame(s) carried no SAM2 proposals and were skipped; "
            "a large share means the tour spends time facing featureless geometry"
        )
    print(f"Wrote reconstruction-free image UOT map to {path}")


if __name__ == "__main__":
    main()
