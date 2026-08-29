#!/usr/bin/env python3
"""Evaluate semantic backends using sparse point annotations on real video."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
)
from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must have the form LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("variant label and path cannot be empty")
    return label.strip(), Path(path)


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _normalise(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _mapping_assignments(manifest: dict[str, object]) -> list[dict[str, object]]:
    assignments = []
    for frame in manifest["frames"]:
        frame_id = int(frame["frame_id"])
        for match in frame.get("matches", []):
            assignments.append(
                {
                    "frame_id": frame_id,
                    "proposal_id": str(match["proposal_id"]),
                    "entity_id": str(match["entity_id"]),
                }
            )
        for unmatched in frame.get("unmatched_proposals", []):
            entity_id = unmatched.get("created_entity_id") or unmatched.get(
                "resolved_entity_id"
            )
            if entity_id is not None:
                assignments.append(
                    {
                        "frame_id": frame_id,
                        "proposal_id": str(unmatched["proposal_id"]),
                        "entity_id": str(entity_id),
                    }
                )
    return assignments


def relevant_entities_for_target(
    assignments: list[dict[str, object]],
    observations_by_proposal: dict[str, dict[str, object]],
    proposal_directory: Path,
    points: list[dict[str, object]],
) -> set[str]:
    """Return entities whose masks contain at least one annotated target point."""

    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for assignment in assignments:
        by_frame[int(assignment["frame_id"])].append(assignment)
    relevant: set[str] = set()
    for point in points:
        frame_id = int(point["frame_id"])
        xy = point.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError("each annotation point requires xy=[x, y]")
        x, y = (int(xy[0]), int(xy[1]))
        for assignment in by_frame.get(frame_id, []):
            observation = observations_by_proposal.get(str(assignment["proposal_id"]))
            if observation is None:
                continue
            with np.load(
                proposal_directory / str(observation["mask_file"]),
                allow_pickle=False,
            ) as evidence:
                mask = np.asarray(evidence["mask"], dtype=bool)
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                relevant.add(str(assignment["entity_id"]))
    return relevant


def _load_variant(label: str, mapping_path: Path) -> dict[str, object]:
    mapping_manifest = json.loads(
        _manifest_path(mapping_path).read_text(encoding="utf-8")
    )
    appearance_path, appearance_manifest, embeddings = load_observation_index(
        str(mapping_manifest["source_appearance_index"])
    )
    assignments = _mapping_assignments(mapping_manifest)
    observation_by_proposal = {
        str(observation["proposal_id"]): observation
        for observation in appearance_manifest["observations"]
    }
    vector_by_proposal = {
        str(observation["proposal_id"]): embeddings[int(observation["index"])]
        for observation in appearance_manifest["observations"]
    }
    grouped_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for assignment in assignments:
        vector = vector_by_proposal.get(str(assignment["proposal_id"]))
        if vector is not None:
            grouped_vectors[str(assignment["entity_id"])].append(vector)
    entity_ids = sorted(grouped_vectors)
    matrices = [_normalise(np.asarray(grouped_vectors[item])) for item in entity_ids]
    prototypes = _normalise(np.stack([np.mean(matrix, axis=0) for matrix in matrices]))
    return {
        "label": label,
        "mapping": mapping_manifest,
        "appearance_path": appearance_path,
        "appearance": appearance_manifest,
        "assignments": assignments,
        "observation_by_proposal": observation_by_proposal,
        "proposal_directory": Path(str(appearance_manifest["source_proposals"])),
        "entity_ids": entity_ids,
        "matrices": matrices,
        "prototypes": prototypes,
    }


def _encoder(manifest: dict[str, object], args: argparse.Namespace) -> object:
    if manifest["format"] == "fact3r-qwen-embedding-observation-index":
        return Qwen3VLEmbeddingEncoder(
            str(manifest["model"]), device_map=args.device_map, dtype=args.dtype
        )
    if manifest["format"] == "fact3r-siglip-observation-index":
        return Siglip2Encoder(str(manifest["model"]), device=args.device)
    raise ValueError(f"unsupported semantic index {manifest['format']}")


def _rankings(variant: dict[str, object], query_vector: np.ndarray) -> dict[str, list[str]]:
    prototypes = np.asarray(variant["prototypes"])
    entity_ids = list(variant["entity_ids"])
    matrices = list(variant["matrices"])
    prototype_scores = prototypes @ query_vector
    best_view_scores = np.asarray(
        [float(np.max(matrix @ query_vector)) for matrix in matrices]
    )
    hybrid_scores = 0.5 * prototype_scores + 0.5 * best_view_scores
    return {
        "prototype": [entity_ids[index] for index in np.argsort(-prototype_scores)],
        "best_view": [entity_ids[index] for index in np.argsort(-best_view_scores)],
        "hybrid": [entity_ids[index] for index in np.argsort(-hybrid_scores)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=_parse_variant, action="append", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    variants = [_load_variant(label, path) for label, path in args.variant]
    rows: list[dict[str, object]] = []
    encoder_cache: dict[tuple[str, str], object] = {}
    query_cache: dict[tuple[str, str, str], tuple[np.ndarray, float]] = {}
    for variant in variants:
        appearance = variant["appearance"]
        encoder_key = (str(appearance["format"]), str(appearance["model"]))
        encoder = encoder_cache.get(encoder_key)
        if encoder is None:
            encoder = _encoder(appearance, args)
            encoder_cache[encoder_key] = encoder
        for query_entry in annotations["queries"]:
            query = str(query_entry["query"])
            query_key = (*encoder_key, query)
            cached = query_cache.get(query_key)
            if cached is None:
                started = perf_counter()
                vector = np.asarray(encoder.encode_text([query])[0])
                cached = (vector / max(float(np.linalg.norm(vector)), 1e-12), perf_counter() - started)
                query_cache[query_key] = cached
            query_vector, query_seconds = cached
            rankings = _rankings(variant, query_vector)
            for target in query_entry["targets"]:
                relevant = relevant_entities_for_target(
                    variant["assignments"],
                    variant["observation_by_proposal"],
                    variant["proposal_directory"],
                    target["points"],
                )
                for aggregation, ranking in rankings.items():
                    relevant_ranks = [
                        index + 1
                        for index, entity_id in enumerate(ranking)
                        if entity_id in relevant
                    ]
                    rank = min(relevant_ranks) if relevant_ranks else None
                    rows.append(
                        {
                            "variant": variant["label"],
                            "semantic_model": appearance["model"],
                            "query": query,
                            "target_id": target["target_id"],
                            "aggregation": aggregation,
                            "relevant_entities": len(relevant),
                            "rank": rank,
                            "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                            "recall_at_1": int(rank is not None and rank <= 1),
                            "recall_at_5": int(rank is not None and rank <= 5),
                            "recall_at_10": int(rank is not None and rank <= 10),
                            "text_encoding_seconds": query_seconds,
                        }
                    )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "retrieval_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    keys = sorted({(str(row["variant"]), str(row["aggregation"])) for row in rows})
    for variant, aggregation in keys:
        selected = [
            row
            for row in rows
            if row["variant"] == variant and row["aggregation"] == aggregation
        ]
        summaries.append(
            {
                "variant": variant,
                "aggregation": aggregation,
                "cases": len(selected),
                "mrr": float(np.mean([row["reciprocal_rank"] for row in selected])),
                "recall_at_1": float(np.mean([row["recall_at_1"] for row in selected])),
                "recall_at_5": float(np.mean([row["recall_at_5"] for row in selected])),
                "recall_at_10": float(np.mean([row["recall_at_10"] for row in selected])),
            }
        )
    (args.output / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                "format": "fact3r-semantic-retrieval-ablation",
                "version": 1,
                "summaries": summaries,
                "cases": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    table = [
        "| variant | aggregation | MRR | R@1 | R@5 | R@10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    table.extend(
        "| {variant} | {aggregation} | {mrr:.3f} | {recall_at_1:.3f} | "
        "{recall_at_5:.3f} | {recall_at_10:.3f} |".format(**summary)
        for summary in summaries
    )
    (args.output / "retrieval_report.md").write_text(
        "# Fact3R annotated semantic-retrieval ablation\n\n"
        + "\n".join(table)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote retrieval ablation to {args.output / 'retrieval_report.md'}")


if __name__ == "__main__":
    main()
