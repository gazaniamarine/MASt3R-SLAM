"""Shared Qwen3-VL frame features for mask appearance memory.

This module intentionally never calls ``generate``.  One complete frame is
encoded once and every SAM proposal pools from the resulting spatial token
grid.  The descriptors are suitable for testing appearance association in UOT;
they are not contrastive text embeddings and therefore are not used by the
SigLIP query runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fact3r.association.tracklets import load_tracklet_run
from fact3r.integrations.mast3r_slam import iter_exported_keyframes
from fact3r.proposals.storage import load_proposal_run_manifest


FloatArray = NDArray[np.floating]


def _normalise_rows(values: object) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("visual descriptors must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("visual descriptors cannot contain zero rows")
    return np.ascontiguousarray(array / norms)


def _rgb_image(values: object) -> Image.Image:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("keyframe RGB must have shape (height, width, 3)")
    if np.issubdtype(array.dtype, np.floating) and array.size:
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    return Image.fromarray(
        np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8)), mode="RGB"
    )


class Qwen3VLFrameEncoder:
    """Extract spatial Qwen3-VL tokens without running the language decoder."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        attention_implementation: str | None = None,
    ) -> None:
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Qwen visual pixel limits are invalid")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported Qwen dtype")
        started = perf_counter()
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen visual memory requires transformers>=4.57, torch, "
                "accelerate, and Pillow"
            ) from error

        kwargs: dict[str, object] = {
            "device_map": device_map,
            "dtype": dtype if dtype == "auto" else getattr(torch, dtype),
        }
        if attention_implementation is not None:
            kwargs["attn_implementation"] = attention_implementation
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **kwargs
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self._torch = torch
        self._model_name = model_name
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        visual = getattr(getattr(self._model, "model", self._model), "visual", None)
        if visual is None:
            raise RuntimeError("loaded Qwen model does not expose its visual encoder")
        self._visual = visual
        self._merge_size = int(getattr(visual, "spatial_merge_size", 2))
        self._device = next(visual.parameters()).device
        self._load_seconds = perf_counter() - started

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device_name(self) -> str:
        return str(self._device)

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    @property
    def pixel_limits(self) -> tuple[int, int]:
        return self._min_pixels, self._max_pixels

    def _pooler_tokens(self, output: object) -> object:
        value = getattr(output, "pooler_output", output)
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError("frame encoder expected exactly one image")
            value = value[0]
        if not self._torch.is_tensor(value) or value.ndim != 2:
            raise TypeError("Qwen visual encoder returned invalid spatial tokens")
        return value

    def encode_frame_masks(
        self, image: Image.Image, masks: Sequence[object]
    ) -> NDArray[np.float32]:
        """Pool one shared full-frame token grid into one vector per mask."""

        if not masks:
            return np.empty((0, 0), dtype=np.float32)
        inputs = self._processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        grid = inputs["image_grid_thw"].to(self._device)
        with self._torch.inference_mode():
            output = self._model.get_image_features(
                pixel_values=pixel_values, image_grid_thw=grid
            )
        tokens = self._pooler_tokens(output)
        temporal, grid_height, grid_width = (
            int(value) for value in grid[0].detach().cpu().tolist()
        )
        pooled_height = grid_height // self._merge_size
        pooled_width = grid_width // self._merge_size
        expected = temporal * pooled_height * pooled_width
        if len(tokens) != expected:
            raise RuntimeError(
                f"Qwen token/grid mismatch: {len(tokens)} tokens for "
                f"{temporal}x{pooled_height}x{pooled_width}"
            )
        # Still images normally have one temporal cell.  Average defensively if
        # a processor version emits duplicated temporal cells.
        token_grid = tokens.reshape(
            temporal, pooled_height, pooled_width, tokens.shape[-1]
        ).mean(dim=0)

        descriptors = []
        for raw_mask in masks:
            mask = np.asarray(raw_mask, dtype=np.float32)
            if mask.ndim != 2 or not np.any(mask > 0):
                raise ValueError("every Qwen-pooled proposal mask must be nonempty")
            weights = self._torch.from_numpy(mask)[None, None].to(self._device)
            weights = self._torch.nn.functional.interpolate(
                weights,
                size=(pooled_height, pooled_width),
                mode="area",
            )[0, 0]
            denominator = weights.sum()
            if float(denominator) <= 1e-8:
                rows, columns = np.nonzero(mask)
                row = min(
                    pooled_height - 1,
                    int(float(np.mean(rows)) / mask.shape[0] * pooled_height),
                )
                column = min(
                    pooled_width - 1,
                    int(float(np.mean(columns)) / mask.shape[1] * pooled_width),
                )
                descriptor = token_grid[row, column]
            else:
                descriptor = (
                    token_grid * weights[..., None]
                ).sum(dim=(0, 1)) / denominator
            descriptors.append(descriptor.detach().float().cpu().numpy())
        return _normalise_rows(np.stack(descriptors))


def _manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "manifest.json" if candidate.is_dir() else candidate


def build_qwen_visual_index(
    *,
    keyframes: str | Path,
    proposals: str | Path,
    output: str | Path,
    encoder: Qwen3VLFrameEncoder,
    tracklets: str | Path | None = None,
    max_frames: int | None = None,
) -> Path:
    """Write a UOT-compatible pre-association appearance index."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    keyframe_directory = Path(keyframes)
    proposal_directory = Path(proposals)
    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    proposal_run = load_proposal_run_manifest(proposal_directory)
    proposal_frames = {
        int(entry["frame_id"]): entry for entry in proposal_run["frames"]
    }
    tracklet_run = None if tracklets is None else load_tracklet_run(tracklets)

    observations: list[dict[str, object]] = []
    chunks: list[NDArray[np.float32]] = []
    frame_seconds = 0.0
    frame_count = 0
    for keyframe in iter_exported_keyframes(keyframe_directory):
        run_entry = proposal_frames.get(keyframe.frame_id)
        if run_entry is None:
            continue
        if max_frames is not None and frame_count >= max_frames:
            break
        frame_manifest_path = proposal_directory / str(run_entry["manifest"])
        frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
        masks = []
        proposals_in_frame = list(frame_manifest["proposals"])
        for proposal in proposals_in_frame:
            proposal_file = frame_manifest_path.parent / str(proposal["file"])
            with np.load(proposal_file, allow_pickle=False) as evidence:
                masks.append(np.asarray(evidence["mask"], dtype=bool))
        if not masks:
            continue
        started = perf_counter()
        chunks.append(
            encoder.encode_frame_masks(
                _rgb_image(keyframe.rgb), masks
            )
        )
        frame_seconds += perf_counter() - started
        frame_tracklets = {
            item.proposal_id: item
            for item in (
                ()
                if tracklet_run is None
                else tracklet_run.observations_by_frame.get(keyframe.frame_id, ())
            )
        }
        for proposal, mask in zip(proposals_in_frame, masks, strict=True):
            proposal_id = str(proposal["proposal_id"])
            tracklet = frame_tracklets.get(proposal_id)
            index = len(observations)
            observations.append(
                {
                    "index": index,
                    "proposal_id": proposal_id,
                    "frame_id": keyframe.frame_id,
                    "timestamp": keyframe.timestamp,
                    "entity_id": None,
                    "track_id": None if tracklet is None else tracklet.track_id,
                    "group_id": (
                        f"observation-{index:06d}"
                        if tracklet is None
                        else tracklet.track_id
                    ),
                    "assignment_status": "unassigned",
                    "association_confidence": (
                        0.5
                        if tracklet is None or tracklet.link_iou is None
                        else float(tracklet.link_iou)
                    ),
                    "proposal_score": float(proposal["score"]),
                    "mask_area": int(proposal["mask_area"]),
                    "geometry_status": str(
                        proposal.get("geometry_status", "unanchored_2d")
                    ),
                    "geometry_coverage": float(
                        proposal.get("geometry_coverage", 0.0)
                    ),
                    "lifted_point_count": int(
                        proposal.get("lifted_point_count", 0)
                    ),
                    "bounding_box_xyxy": proposal.get("bounding_box_xyxy"),
                    "track_link_iou": (
                        None if tracklet is None else tracklet.link_iou
                    ),
                    "mask_file": str(
                        (frame_manifest_path.parent / str(proposal["file"]))
                        .relative_to(proposal_directory)
                    ),
                }
            )
        frame_count += 1

    if not observations:
        raise ValueError("no common keyframe and proposal observations found")
    embeddings = _normalise_rows(np.concatenate(chunks, axis=0))
    np.save(output_directory / "embeddings.npy", embeddings)
    min_pixels, max_pixels = encoder.pixel_limits
    manifest: dict[str, object] = {
        "format": "fact3r-vla-visual-observation-index",
        "version": 1,
        "semantic_query_capable": False,
        "model": encoder.model_name,
        "device": encoder.device_name,
        "source_keyframes": str(keyframe_directory.resolve()),
        "source_proposals": str(proposal_directory.resolve()),
        "source_tracklets": (
            None if tracklets is None else str(_manifest_path(tracklets).resolve())
        ),
        "source_mapping": None,
        "embedding_file": "embeddings.npy",
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "frame_count": frame_count,
        "observation_count": len(observations),
        "assigned_observation_count": 0,
        "unanchored_observation_count": sum(
            item["geometry_status"] == "unanchored_2d" for item in observations
        ),
        "track_only_observation_count": sum(
            item["track_id"] is not None for item in observations
        ),
        "visual_pooling": {
            "source": "shared_full_frame_qwen_tokens",
            "mask_resampling": "area",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
        "timing": {
            "model_load_seconds": float(encoder.load_seconds),
            "frame_encoding_seconds": frame_seconds,
            "frames_per_encoding_second": (
                0.0 if frame_seconds <= 0 else frame_count / frame_seconds
            ),
            "observations_per_encoding_second": (
                0.0 if frame_seconds <= 0 else len(observations) / frame_seconds
            ),
        },
        "observations": observations,
    }
    manifest["diagnostics"] = visual_link_diagnostics(embeddings, observations)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest_path


def visual_link_diagnostics(
    embeddings: FloatArray,
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Measure whether shared VLM features preserve short-term identity."""

    vectors = _normalise_rows(embeddings)
    linked_scores = []
    negative_scores = []
    paired_margins = []
    top1_hits = 0
    top1_trials = 0
    frame_rows: dict[int, list[int]] = {}
    for index, item in enumerate(observations):
        frame_rows.setdefault(int(item["frame_id"]), []).append(index)
    last_by_track: dict[str, int] = {}
    for current_index, item in enumerate(observations):
        # Recover the previous observation through the persistent SAM track.
        track_id = item.get("track_id")
        if track_id is None:
            continue
        track_key = str(track_id)
        source_index = last_by_track.get(track_key)
        last_by_track[track_key] = current_index
        if source_index is None:
            continue
        linked_scores.append(float(vectors[current_index] @ vectors[source_index]))
        previous_frame = int(observations[source_index]["frame_id"])
        pool = [
            index
            for index in frame_rows.get(previous_frame, [])
            if observations[index].get("track_id") != track_id
        ]
        if pool:
            similarities = vectors[pool] @ vectors[current_index]
            negative_scores.append(float(np.max(similarities)))
            paired_margins.append(linked_scores[-1] - negative_scores[-1])
            top1_hits += int(
                float(linked_scores[-1]) > float(np.max(similarities))
            )
            top1_trials += 1
    linked = np.asarray(linked_scores, dtype=np.float32)
    negatives = np.asarray(negative_scores, dtype=np.float32)
    return {
        "linked_pair_count": len(linked_scores),
        "linked_cosine_median": (
            None if len(linked) == 0 else float(np.median(linked))
        ),
        "strongest_unrelated_cosine_median": (
            None if len(negatives) == 0 else float(np.median(negatives))
        ),
        "median_link_margin": (
            None if not paired_margins else float(np.median(paired_margins))
        ),
        "previous_frame_top1_accuracy": (
            None if top1_trials == 0 else top1_hits / top1_trials
        ),
        "top1_trial_count": top1_trials,
    }
