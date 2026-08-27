"""Query-time Qwen3-VL verification of SigLIP entity candidates.

SigLIP remains the fast, high-recall retriever.  The VLM only inspects a small
set of confirmed persistent entities and cannot change geometric identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from fact3r.integrations.mast3r_slam import iter_exported_keyframes
from fact3r.semantics.observation_index import (
    VisionLanguageEncoder,
    default_positive_prompts,
    load_observation_index,
    map_derived_hard_negative_scores,
)
from fact3r.visualization.association import mask_boundary


FloatArray = NDArray[np.floating]
PROMPT_VERSION = 2


class EntityVerifier(Protocol):
    """Minimal interface used by the query runner and its tests."""

    @property
    def model_name(self) -> str: ...

    @property
    def load_seconds(self) -> float: ...

    def verify(
        self,
        *,
        query: str,
        entity_id: str,
        evidence_images: Sequence[Path],
        frame_ids: Sequence[int],
    ) -> "VLMVerification": ...


@dataclass(frozen=True, slots=True)
class VLMVerification:
    decision: str
    confidence: float
    predicted_object: str
    confusable_with: tuple[str, ...]
    supporting_frames: tuple[int, ...]
    reason: str
    raw_output: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "VLMVerification":
        decision = str(payload.get("decision", "uncertain")).strip().lower()
        if decision not in {"yes", "no", "uncertain"}:
            raise ValueError("decision must be yes, no, or uncertain")
        confidence = float(payload.get("confidence", 0.0))
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        confidence = float(np.clip(confidence, 0.0, 1.0))
        confusable = payload.get("confusable_with", ())
        if isinstance(confusable, str):
            confusable = (confusable,)
        if not isinstance(confusable, Sequence):
            confusable = ()
        frames = payload.get("supporting_frames", ())
        if not isinstance(frames, Sequence) or isinstance(frames, str):
            frames = ()
        return cls(
            decision=decision,
            confidence=confidence,
            predicted_object=str(payload.get("predicted_object", "")).strip(),
            confusable_with=tuple(
                str(item).strip() for item in confusable if str(item).strip()
            ),
            supporting_frames=tuple(int(item) for item in frames),
            reason=str(payload.get("reason", "")).strip(),
            raw_output=str(payload.get("raw_output", "")),
        )


@dataclass(frozen=True, slots=True)
class EntityVerificationRequest:
    entity_id: str
    evidence_images: tuple[Path, ...]
    frame_ids: tuple[int, ...]


def build_verification_prompt(query: str, frame_ids: Sequence[int]) -> str:
    """Create the deliberately strict prompt used for every candidate."""

    return (
        "You are verifying an object-memory retrieval for a mobile robot. "
        "All supplied images are different observations of ONE persistent 3D "
        "entity. Each image contains an overview, a highlighted context crop, "
        "and an isolated target crop. Judge the entity from the isolated target "
        "and highlighted pixels. The overview is only for location; an object "
        "elsewhere in the overview is never evidence for the candidate. Reject "
        "thin borders, image-edge strips, shadows, and incomplete fragments.\n\n"
        f"Target query: {json.dumps(query.strip())}\n"
        f"Frame IDs in image order: {list(frame_ids)}\n\n"
        "Use evidence across the views. Small size alone is not a rejection, "
        "but select uncertain if the highlighted region is too incomplete or "
        "ambiguous. decision=yes means the highlighted entity itself is the "
        "target. decision=no means it is another identifiable object. Name that "
        "object in predicted_object so it can become a query-specific dynamic "
        "confounder. Return ONLY one JSON object with exactly these fields:\n"
        '{"decision":"yes|no|uncertain","confidence":0.0,'
        '"predicted_object":"short noun phrase",'
        '"confusable_with":["short noun phrase"],'
        f'"supporting_frames":{list(frame_ids)},'
        '"reason":"one short sentence"}'
    )


def build_listwise_verification_prompt(
    query: str, requests: Sequence[EntityVerificationRequest]
) -> str:
    """Prompt one VLM call to judge several persistent candidates independently."""

    image_index = 1
    candidate_lines = []
    for request in requests:
        end = image_index + len(request.evidence_images) - 1
        candidate_lines.append(
            f"- {request.entity_id}: images {image_index}-{end}, "
            f"frame IDs {list(request.frame_ids)}"
        )
        image_index = end + 1
    return (
        "You are verifying object-memory retrieval candidates for a mobile robot. "
        "Each candidate is one persistent entity seen in multiple frames. Each "
        "evidence image has an overview, highlighted context, and an isolated "
        "target. Judge the isolated target and highlighted pixels. The overview "
        "is location-only: never use another object elsewhere in it as evidence. "
        "Reject borders, edge strips, shadows, and incomplete fragments.\n\n"
        f"Target query: {json.dumps(query.strip())}\n"
        "Candidate-to-image mapping:\n"
        + "\n".join(candidate_lines)
        + "\n\nJudge every candidate independently. Use uncertain when evidence is "
        "incomplete. supporting_frames must contain only listed frame IDs that "
        "visually support yes. For no, predicted_object must name what the "
        "highlighted entity actually is. Return ONLY this JSON structure:\n"
        '{"candidates":[{"entity_id":"exact supplied ID",'
        '"decision":"yes|no|uncertain","confidence":0.0,'
        '"predicted_object":"short noun phrase",'
        '"confusable_with":["short noun phrase"],'
        '"supporting_frames":[0],"reason":"short sentence"}]}'
    )


def parse_verification_output(text: str) -> VLMVerification:
    """Extract and validate the first JSON object produced by the VLM."""

    source = text.strip()
    start = source.find("{")
    if start < 0:
        raise ValueError("VLM response did not contain a JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(source[start:])
    except json.JSONDecodeError as error:
        raise ValueError("VLM response contained invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("VLM response JSON must be an object")
    enriched = dict(payload)
    enriched["raw_output"] = text
    return VLMVerification.from_mapping(enriched)


def parse_listwise_verification_output(
    text: str, entity_ids: Sequence[str]
) -> dict[str, VLMVerification]:
    """Parse a multi-candidate response and fail missing candidates closed."""

    source = text.strip()
    start = source.find("{")
    if start < 0:
        raise ValueError("VLM response did not contain a JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(source[start:])
    except json.JSONDecodeError as error:
        raise ValueError("VLM response contained invalid JSON") from error
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("candidates"), Sequence
    ):
        raise ValueError("VLM response must contain a candidates array")
    allowed = set(entity_ids)
    parsed: dict[str, VLMVerification] = {}
    for item in payload["candidates"]:
        if not isinstance(item, Mapping):
            continue
        entity_id = str(item.get("entity_id", ""))
        if entity_id not in allowed or entity_id in parsed:
            continue
        enriched = dict(item)
        enriched["raw_output"] = text
        parsed[entity_id] = VLMVerification.from_mapping(enriched)
    for entity_id in entity_ids:
        if entity_id not in parsed:
            parsed[entity_id] = VLMVerification(
                decision="uncertain",
                confidence=0.0,
                predicted_object="",
                confusable_with=(),
                supporting_frames=(),
                reason="candidate missing from structured VLM output",
                raw_output=text,
            )
    return parsed


def local_image_source(path: str | Path) -> str:
    """Return the plain absolute path expected by Transformers image loaders."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Qwen evidence image does not exist: {resolved}")
    return str(resolved)


class Qwen3VLVerifier:
    """Lazy Hugging Face adapter for Qwen3-VL instruction models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        attention_implementation: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported Qwen dtype")
        self._model_name = model_name
        self._device_map = device_map
        self._dtype = dtype
        self._attention_implementation = attention_implementation
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._torch = None
        self._load_seconds = 0.0
        self.listwise_batch_size = 2 if max_new_tokens < 192 else 3

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        started = perf_counter()
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL verification requires transformers>=4.57, torch, "
                "accelerate, and Pillow; install the fact3r-map vlm extras"
            ) from error
        model_kwargs: dict[str, object] = {
            "dtype": (
                self._dtype
                if self._dtype == "auto"
                else getattr(torch, self._dtype)
            ),
            "device_map": self._device_map,
        }
        if self._attention_implementation is not None:
            model_kwargs["attn_implementation"] = self._attention_implementation
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._torch = torch
        self._load_seconds = perf_counter() - started

    def verify(
        self,
        *,
        query: str,
        entity_id: str,
        evidence_images: Sequence[Path],
        frame_ids: Sequence[int],
    ) -> VLMVerification:
        if not evidence_images:
            raise ValueError("at least one evidence image is required")
        if len(evidence_images) != len(frame_ids):
            raise ValueError("evidence images and frame IDs must align")
        content: list[dict[str, str]] = [
            {"type": "image", "image": local_image_source(path)}
            for path in evidence_images
        ]
        content.append(
            {
                "type": "text",
                "text": build_verification_prompt(query, frame_ids),
            }
        )
        output = self._generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Follow the visual grounding instructions exactly and "
                        "emit valid JSON only."
                    ),
                },
                {"role": "user", "content": content},
            ]
        )
        try:
            return parse_verification_output(output)
        except ValueError as error:
            return VLMVerification(
                decision="uncertain",
                confidence=0.0,
                predicted_object="",
                confusable_with=(),
                supporting_frames=(),
                reason=f"invalid structured VLM output: {error}",
                raw_output=output,
            )

    def verify_many(
        self,
        *,
        query: str,
        requests: Sequence[EntityVerificationRequest],
    ) -> dict[str, VLMVerification]:
        """Judge a small candidate set in one multimodal generation."""

        if not requests:
            return {}
        content: list[dict[str, str]] = []
        for request in requests:
            if not request.evidence_images:
                raise ValueError("each candidate needs evidence images")
            if len(request.evidence_images) != len(request.frame_ids):
                raise ValueError("evidence images and frame IDs must align")
            content.extend(
                {"type": "image", "image": local_image_source(path)}
                for path in request.evidence_images
            )
        content.append(
            {
                "type": "text",
                "text": build_listwise_verification_prompt(query, requests),
            }
        )
        output = self._generate(
            [
                {
                    "role": "system",
                    "content": "Ground every verdict visually and emit valid JSON only.",
                },
                {"role": "user", "content": content},
            ]
        )
        try:
            return parse_listwise_verification_output(
                output, [request.entity_id for request in requests]
            )
        except ValueError as error:
            return {
                request.entity_id: VLMVerification(
                    decision="uncertain",
                    confidence=0.0,
                    predicted_object="",
                    confusable_with=(),
                    supporting_frames=(),
                    reason=f"invalid structured VLM output: {error}",
                    raw_output=output,
                )
                for request in requests
            }

    def _generate(self, messages: Sequence[Mapping[str, object]]) -> str:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        inputs = self._processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self._model.device)
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

def _normalise_rows(values: object) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings cannot contain zero-length rows")
    return np.ascontiguousarray(array / norms)


def rank_vlm_candidates(
    embeddings: FloatArray,
    observations: Sequence[Mapping[str, object]],
    positive_embeddings: FloatArray,
    *,
    top_views: int = 3,
    min_observations: int = 2,
    reference_mask_area: float = 4096.0,
    map_negative_neighbors: int = 3,
    map_negative_weight: float = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, object]]]:
    """Build a shortlist using nearest map entities as automatic confounders."""

    if top_views <= 0 or min_observations <= 0:
        raise ValueError("top_views and min_observations must be positive")
    if reference_mask_area <= 0.0:
        raise ValueError("reference_mask_area must be positive")
    if map_negative_neighbors <= 0 or map_negative_weight < 0.0:
        raise ValueError("map-negative settings are invalid")
    vectors = _normalise_rows(embeddings)
    positives = _normalise_rows(positive_embeddings)
    if positives.shape[1] != vectors.shape[1]:
        raise ValueError("positive prompt and observation dimensions differ")
    scores = np.asarray(np.mean(vectors @ positives.T, axis=1), dtype=np.float32)
    map_negative_scores, map_negative_diagnostics = (
        map_derived_hard_negative_scores(
            vectors,
            observations,
            positives,
            neighbors=map_negative_neighbors,
            confirmed_only=True,
        )
    )
    qualities = np.zeros(len(observations), dtype=np.float32)
    grouped: dict[str, list[int]] = {}
    for index, observation in enumerate(observations):
        entity_id = observation.get("entity_id")
        track_id = observation.get("track_id")
        group_id = entity_id or track_id
        if group_id is None:
            continue
        sam = float(np.clip(float(observation.get("proposal_score", 1.0)), 0, 1))
        association = float(
            np.clip(float(observation.get("association_confidence", 1.0)), 0, 1)
        )
        area = max(0.0, float(observation.get("mask_area", 0.0)))
        qualities[index] = sam * association * min(
            1.0, math.sqrt(area / reference_mask_area)
        )
        grouped.setdefault(str(group_id), []).append(index)

    groups: list[dict[str, object]] = []
    for group_id, indices in grouped.items():
        if len(indices) < min_observations:
            continue
        ordered = sorted(indices, key=lambda item: float(scores[item]), reverse=True)
        evidence = ordered[: min(top_views, len(ordered))]
        weights = qualities[evidence].astype(np.float64)
        if float(weights.sum()) <= 1e-12:
            positive_score = float(np.mean(scores[evidence]))
        else:
            positive_score = float(np.average(scores[evidence], weights=weights))
        raw_hard_negative_score = float(map_negative_scores[indices[0]])
        # ``-1`` is the sentinel used when no competing map entity exists.  A
        # missing negative must be neutral rather than accidentally boosting a
        # one-entity map, and an anticorrelated competitor is not a confounder.
        hard_negative_score = max(0.0, raw_hard_negative_score)
        score = positive_score - map_negative_weight * hard_negative_score
        first = observations[indices[0]]
        groups.append(
            {
                "candidate_id": group_id,
                "group_id": group_id,
                "entity_id": first.get("entity_id"),
                "track_id": first.get("track_id"),
                "candidate_score": score,
                "positive_candidate_score": positive_score,
                "map_hard_negative_score": hard_negative_score,
                "map_hard_negatives": map_negative_diagnostics.get(group_id, []),
                "observation_count": len(indices),
                "mean_observation_quality": float(np.mean(qualities[indices])),
                "observation_indices": indices,
                "ranked_observation_indices": ordered,
                "evidence_observation_indices": evidence,
            }
        )
    groups.sort(key=lambda item: float(item["candidate_score"]), reverse=True)
    return scores, qualities, groups


def _rgb_uint8(rgb: object) -> NDArray[np.uint8]:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def _highlight(rgb: object, mask: object) -> Image.Image:
    image = _rgb_uint8(rgb)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != image.shape[:2]:
        raise ValueError("stored mask and keyframe shape do not match")
    canvas = image.astype(np.float32)
    green = np.asarray([20.0, 255.0, 70.0], dtype=np.float32)
    canvas[selected] = 0.48 * canvas[selected] + 0.52 * green
    canvas[mask_boundary(selected)] = green
    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


def _context_crop(
    rgb: object, mask: object, *, context_fraction: float = 0.40
) -> tuple[object, object]:
    image = _rgb_uint8(rgb)
    selected = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(selected)
    if not len(rows):
        raise ValueError("cannot render an empty mask")
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    padding = int(round(context_fraction * max(y1 - y0, x1 - x0)))
    y0, y1 = max(0, y0 - padding), min(image.shape[0], y1 + padding)
    x0, x1 = max(0, x0 - padding), min(image.shape[1], x1 + padding)
    return image[y0:y1, x0:x1], selected[y0:y1, x0:x1]


def pathological_border_sliver(
    mask: object,
    *,
    max_thickness_fraction: float = 0.05,
    min_aspect_ratio: float = 12.0,
) -> bool:
    """Detect SAM fragments that are thin strips attached to an image edge."""

    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2 or not np.any(selected):
        return True
    rows, columns = np.nonzero(selected)
    height, width = selected.shape
    box_height = int(rows.max() - rows.min() + 1)
    box_width = int(columns.max() - columns.min() + 1)
    aspect = max(box_height / max(box_width, 1), box_width / max(box_height, 1))
    edge_tolerance = max(2, int(round(0.005 * max(height, width))))
    touches_edge = (
        rows.min() <= edge_tolerance
        or rows.max() >= height - 1 - edge_tolerance
        or columns.min() <= edge_tolerance
        or columns.max() >= width - 1 - edge_tolerance
    )
    thin = (
        box_width / width <= max_thickness_fraction
        or box_height / height <= max_thickness_fraction
    )
    return bool(touches_edge and thin and aspect >= min_aspect_ratio)


def _isolated_target(rgb: object, mask: object) -> Image.Image:
    """Show target appearance while neutralising all contextual distractors."""

    image = _rgb_uint8(rgb)
    selected = np.asarray(mask, dtype=bool)
    isolated = np.full_like(image, 127)
    isolated[selected] = image[selected]
    isolated[mask_boundary(selected)] = [20, 255, 70]
    crop_rgb, _ = _context_crop(isolated, selected, context_fraction=0.10)
    return Image.fromarray(crop_rgb, mode="RGB")


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (12, 12, 12))
    panel.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return panel


def _render_evidence_view(
    rgb: object,
    mask: object,
    *,
    query: str,
    entity_id: str,
    frame_id: int,
    siglip_score: float,
) -> Image.Image:
    overview = _highlight(rgb, mask)
    crop_rgb, crop_mask = _context_crop(rgb, mask)
    closeup = _highlight(crop_rgb, crop_mask)
    isolated = _isolated_target(rgb, mask)
    rendered = Image.new("RGB", (1000, 420), (20, 20, 20))
    rendered.paste(_fit_panel(overview, 430, 352), (8, 60))
    rendered.paste(_fit_panel(closeup, 270, 352), (446, 60))
    rendered.paste(_fit_panel(isolated, 268, 352), (724, 60))
    draw = ImageDraw.Draw(rendered)
    draw.text(
        (10, 8),
        f'query "{query}" | candidate {entity_id} | frame {frame_id}',
        fill=(245, 245, 245),
    )
    draw.text(
        (10, 33),
        f"overview | highlighted context | isolated target | SigLIP={siglip_score:.3f}",
        fill=(185, 230, 195),
    )
    return rendered


@dataclass(slots=True)
class PreparedVLMQuery:
    manifest_path: Path
    manifest: dict[str, object]
    output: Path
    query: str
    positive_prompts: tuple[str, ...]
    scores: NDArray[np.float32]
    qualities: NDArray[np.float32]
    candidates: list[dict[str, object]]
    siglip_load_seconds: float
    text_encoding_seconds: float
    preparation_seconds: float
    map_negative_neighbors: int
    map_negative_weight: float


def prepare_vlm_query(
    *,
    index: str | Path,
    query: str,
    output: str | Path,
    encoder: VisionLanguageEncoder,
    max_candidates: int = 8,
    top_views: int = 3,
    min_observations: int = 2,
    reference_mask_area: float = 4096.0,
    map_negative_neighbors: int = 3,
    map_negative_weight: float = 1.0,
    min_siglip_score: float = 0.10,
) -> PreparedVLMQuery:
    """Shortlist candidates with SigLIP and render their multi-view evidence."""

    if not query.strip():
        raise ValueError("query cannot be empty")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if not -1.0 <= min_siglip_score <= 1.0:
        raise ValueError("min_siglip_score must be in [-1, 1]")
    started = perf_counter()
    manifest_path, manifest, embeddings = load_observation_index(index)
    if encoder.model_name != manifest["model"]:
        raise ValueError(
            f"query encoder {encoder.model_name!r} does not match index model "
            f"{manifest['model']!r}"
        )
    prompts = default_positive_prompts(query)
    text_started = perf_counter()
    prompt_embeddings = encoder.encode_text(prompts)
    text_seconds = perf_counter() - text_started
    scores, qualities, groups = rank_vlm_candidates(
        embeddings,
        manifest["observations"],
        prompt_embeddings,
        top_views=top_views,
        min_observations=min_observations,
        reference_mask_area=reference_mask_area,
        map_negative_neighbors=map_negative_neighbors,
        map_negative_weight=map_negative_weight,
    )
    proposal_directory = Path(str(manifest["source_proposals"]))
    observations = manifest["observations"]
    candidates: list[dict[str, object]] = []
    for candidate in groups:
        if float(candidate["positive_candidate_score"]) < min_siglip_score:
            continue
        valid_evidence: list[int] = []
        for observation_index in candidate["ranked_observation_indices"]:
            observation = observations[int(observation_index)]
            with np.load(
                proposal_directory / str(observation["mask_file"]),
                allow_pickle=False,
            ) as payload:
                candidate_mask = np.asarray(payload["mask"], dtype=bool)
            if not pathological_border_sliver(candidate_mask):
                valid_evidence.append(int(observation_index))
            if len(valid_evidence) >= top_views:
                break
        if len(valid_evidence) < min_observations:
            continue
        candidate["evidence_observation_indices"] = valid_evidence
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    output_directory = Path(output)
    evidence_directory = output_directory / "evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    needed_frames = {
        int(observations[index_value]["frame_id"])
        for candidate in candidates
        for index_value in candidate["evidence_observation_indices"]
    }
    keyframes = {
        keyframe.frame_id: np.array(keyframe.rgb, copy=True)
        for keyframe in iter_exported_keyframes(manifest["source_keyframes"])
        if keyframe.frame_id in needed_frames
    }
    for rank, candidate in enumerate(candidates, start=1):
        evidence_paths: list[Path] = []
        frame_ids: list[int] = []
        for view_rank, observation_index in enumerate(
            candidate["evidence_observation_indices"], start=1
        ):
            observation = observations[observation_index]
            frame_id = int(observation["frame_id"])
            rgb = keyframes.get(frame_id)
            if rgb is None:
                raise ValueError(f"keyframe {frame_id} is missing")
            with np.load(
                proposal_directory / str(observation["mask_file"]),
                allow_pickle=False,
            ) as payload:
                mask = np.array(payload["mask"], dtype=bool, copy=True)
            evidence = _render_evidence_view(
                rgb,
                mask,
                query=query,
                entity_id=str(candidate["candidate_id"]),
                frame_id=frame_id,
                siglip_score=float(scores[observation_index]),
            )
            path = evidence_directory / (
                f"candidate_{rank:02d}_{str(candidate['candidate_id']).replace('/', '-')}_"
                f"view_{view_rank:02d}_frame_{frame_id:06d}.jpg"
            )
            evidence.save(path, quality=94)
            evidence_paths.append(path)
            frame_ids.append(frame_id)
        candidate["shortlist_rank"] = rank
        candidate["evidence_images"] = evidence_paths
        candidate["evidence_frame_ids"] = frame_ids
        best_observation = observations[
            int(candidate["evidence_observation_indices"][0])
        ]
        candidate["best_revisit_view"] = {
            "frame_id": best_observation["frame_id"],
            "timestamp": best_observation.get("timestamp"),
            "camera_pose_world_from_camera": best_observation.get(
                "camera_pose_world_from_camera"
            ),
            "camera_origin_world": best_observation.get("camera_origin_world"),
            "view_ray_world": best_observation.get("view_ray_world"),
            "mask_center_rc": best_observation.get("mask_center_rc"),
        }
    return PreparedVLMQuery(
        manifest_path=manifest_path,
        manifest=manifest,
        output=output_directory,
        query=query.strip(),
        positive_prompts=prompts,
        scores=scores,
        qualities=qualities,
        candidates=candidates,
        siglip_load_seconds=float(encoder.load_seconds),
        text_encoding_seconds=text_seconds,
        preparation_seconds=perf_counter() - started,
        map_negative_neighbors=map_negative_neighbors,
        map_negative_weight=map_negative_weight,
    )


def _verification_cache_key(
    *, query: str, model_name: str, entity_id: str, evidence_images: Sequence[Path]
) -> str:
    digest = sha256()
    digest.update(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "query": query.strip().casefold(),
                "model": model_name,
                "entity_id": entity_id,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for path in evidence_images:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _render_verified_observation(
    rgb: object,
    mask: object,
    *,
    query: str,
    entity_id: str,
    frame_id: int,
    siglip_score: float,
    confidence: float,
) -> Image.Image:
    highlighted = _highlight(rgb, mask)
    rendered = Image.new(
        "RGB", (highlighted.width, highlighted.height + 62), (20, 20, 20)
    )
    rendered.paste(highlighted, (0, 62))
    draw = ImageDraw.Draw(rendered)
    draw.text(
        (8, 7),
        f'Qwen3-VL verified "{query}" | {entity_id} | frame {frame_id}',
        fill=(245, 245, 245),
    )
    draw.text(
        (8, 33),
        f"VLM confidence={confidence:.2f} | SigLIP view score={siglip_score:.3f}",
        fill=(185, 230, 195),
    )
    return rendered


def _resize_to_width(image: Image.Image, maximum_width: int) -> Image.Image:
    if image.width <= maximum_width:
        return image.copy()
    height = max(1, round(image.height * maximum_width / image.width))
    return image.resize((maximum_width, height), Image.Resampling.LANCZOS)


def _write_media(
    rendered: Sequence[Path], output: Path, *, width: int, duration_ms: int
) -> None:
    if not rendered:
        return
    images: list[Image.Image] = []
    thumbs: list[Image.Image] = []
    for path in rendered:
        with Image.open(path) as source:
            image = source.convert("RGB")
            images.append(_resize_to_width(image, width))
            thumbs.append(_resize_to_width(image, 420))
    images[0].save(
        output / "matches.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    columns = 3
    tile_height = max(image.height for image in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 420, rows * tile_height), (25, 25, 25))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * 420, (index // columns) * tile_height))
    sheet.save(output / "contact_sheet.jpg", quality=90)


def _dynamic_confounders(
    query: str, rejected: Sequence[Mapping[str, object]]
) -> list[str]:
    blocked = query.strip().casefold()
    found: list[str] = []
    for candidate in rejected:
        verification = candidate["verification"]
        values = [verification.predicted_object, *verification.confusable_with]
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned.casefold() != blocked and cleaned not in found:
                found.append(cleaned)
    return found


def verify_prepared_query(
    prepared: PreparedVLMQuery,
    *,
    verifier: EntityVerifier,
    min_confidence: float = 0.75,
    min_supporting_views: int = 2,
    max_entities: int = 3,
    cache_directory: str | Path | None = None,
    force_reverify: bool = False,
    max_observations_per_entity: int | None = None,
    gif_width: int = 1000,
    gif_duration_ms: int = 400,
) -> Path:
    """Verify candidates, then render every observation of accepted entities."""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if min_supporting_views <= 0:
        raise ValueError("min_supporting_views must be positive")
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    if max_observations_per_entity is not None and max_observations_per_entity <= 0:
        raise ValueError("max_observations_per_entity must be positive")
    started = perf_counter()
    cache = (
        Path(cache_directory)
        if cache_directory is not None
        else prepared.manifest_path.parent / "vlm_cache"
    )
    cache.mkdir(parents=True, exist_ok=True)
    verification_seconds = 0.0
    cache_hits = 0
    candidate_records: list[dict[str, object]] = []
    pending_records: list[dict[str, object]] = []
    for candidate in prepared.candidates:
        entity_id = str(candidate["candidate_id"])
        evidence_images = candidate["evidence_images"]
        frame_ids = candidate["evidence_frame_ids"]
        key = _verification_cache_key(
            query=prepared.query,
            model_name=verifier.model_name,
            entity_id=entity_id,
            evidence_images=evidence_images,
        )
        cache_path = cache / f"{key}.json"
        was_cached = cache_path.is_file() and not force_reverify
        if was_cached:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            verification = VLMVerification.from_mapping(payload["verification"])
            cache_hits += 1
        else:
            verification = None
        record = {
            "candidate": candidate,
            "entity_id": entity_id,
            "evidence_images": evidence_images,
            "frame_ids": frame_ids,
            "cache_path": cache_path,
            "cache_hit": was_cached,
            "verification": verification,
        }
        candidate_records.append(record)
        if verification is None:
            pending_records.append(record)

    verify_many = getattr(verifier, "verify_many", None)
    listwise_batch_size = max(1, int(getattr(verifier, "listwise_batch_size", 1)))
    for offset in range(0, len(pending_records), listwise_batch_size):
        chunk = pending_records[offset : offset + listwise_batch_size]
        inference_started = perf_counter()
        if callable(verify_many) and len(chunk) > 1:
            print(
                f"Qwen verifying {len(chunk)} persistent candidates in one request "
                f"({offset + 1}-{offset + len(chunk)} of {len(pending_records)})"
            )
            requests = [
                EntityVerificationRequest(
                    entity_id=str(record["entity_id"]),
                    evidence_images=tuple(record["evidence_images"]),
                    frame_ids=tuple(record["frame_ids"]),
                )
                for record in chunk
            ]
            chunk_verifications = verify_many(
                query=prepared.query, requests=requests
            )
        else:
            chunk_verifications = {}
            for record in chunk:
                print(
                    f"Qwen verifying {record['entity_id']} "
                    f"({offset + 1} of {len(pending_records)})"
                )
                chunk_verifications[str(record["entity_id"])] = verifier.verify(
                    query=prepared.query,
                    entity_id=str(record["entity_id"]),
                    evidence_images=record["evidence_images"],
                    frame_ids=record["frame_ids"],
                )
        verification_seconds += perf_counter() - inference_started
        for record in chunk:
            entity_id = str(record["entity_id"])
            verification = chunk_verifications[entity_id]
            record["verification"] = verification
            Path(record["cache_path"]).write_text(
                json.dumps(
                    {
                        "format": "fact3r-vlm-verification-cache",
                        "version": 1,
                        "prompt_version": PROMPT_VERSION,
                        "query": prepared.query,
                        "model": verifier.model_name,
                        "entity_id": entity_id,
                        "evidence_frames": record["frame_ids"],
                        "verification": asdict(verification),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    checked: list[dict[str, object]] = []
    for record in candidate_records:
        candidate = record["candidate"]
        verification = record["verification"]
        frame_ids = record["frame_ids"]
        assert isinstance(verification, VLMVerification)
        valid_supporting_frames = sorted(
            set(verification.supporting_frames).intersection(frame_ids)
        )
        accepted = (
            verification.decision == "yes"
            and verification.confidence >= min_confidence
            and len(valid_supporting_frames) >= min_supporting_views
        )
        rejection_reasons: list[str] = []
        if verification.decision != "yes":
            rejection_reasons.append(f"vlm_decision_{verification.decision}")
        if verification.confidence < min_confidence:
            rejection_reasons.append("vlm_confidence_below_threshold")
        if len(valid_supporting_frames) < min_supporting_views:
            rejection_reasons.append("insufficient_vlm_supporting_views")
        checked.append(
            {
                **candidate,
                "verification": verification,
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "valid_supporting_frames": valid_supporting_frames,
                "cache_hit": bool(record["cache_hit"]),
            }
        )
    accepted = [candidate for candidate in checked if candidate["accepted"]][
        :max_entities
    ]
    rejected = [candidate for candidate in checked if not candidate["accepted"]]

    observations = prepared.manifest["observations"]
    selected_indices: list[tuple[dict[str, object], int]] = []
    for candidate in accepted:
        indices = list(candidate["ranked_observation_indices"])
        if max_observations_per_entity is not None:
            indices = indices[:max_observations_per_entity]
        indices.sort(key=lambda item: int(observations[item]["frame_id"]))
        selected_indices.extend((candidate, item) for item in indices)
    needed_frames = {int(observations[item]["frame_id"]) for _, item in selected_indices}
    keyframes = {
        keyframe.frame_id: np.array(keyframe.rgb, copy=True)
        for keyframe in iter_exported_keyframes(prepared.manifest["source_keyframes"])
        if keyframe.frame_id in needed_frames
    }
    proposal_directory = Path(str(prepared.manifest["source_proposals"]))
    frames_directory = prepared.output / "frames"
    frames_directory.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []
    entity_results: list[dict[str, object]] = []
    entity_lookup: dict[str, dict[str, object]] = {}
    for rank, candidate in enumerate(accepted, start=1):
        verification = candidate["verification"]
        result = {
            "rank": rank,
            "entity_id": candidate["entity_id"],
            "track_id": candidate["track_id"],
            "candidate_id": candidate["candidate_id"],
            "memory_type": (
                "anchored_3d_entity"
                if candidate["entity_id"] is not None
                else "unanchored_2d_track"
            ),
            "navigation_target_available": candidate["entity_id"] is not None,
            "best_revisit_view": candidate.get("best_revisit_view"),
            "siglip_candidate_score": candidate["candidate_score"],
            "observation_count": candidate["observation_count"],
            "vlm": asdict(verification),
            "observations": [],
        }
        entity_results.append(result)
        entity_lookup[str(candidate["candidate_id"])] = result
    for render_index, (candidate, observation_index) in enumerate(selected_indices):
        observation = observations[observation_index]
        frame_id = int(observation["frame_id"])
        rgb = keyframes.get(frame_id)
        if rgb is None:
            raise ValueError(f"keyframe {frame_id} is missing")
        with np.load(
            proposal_directory / str(observation["mask_file"]), allow_pickle=False
        ) as payload:
            mask = np.array(payload["mask"], dtype=bool, copy=True)
        verification = candidate["verification"]
        image = _render_verified_observation(
            rgb,
            mask,
            query=prepared.query,
            entity_id=str(candidate["candidate_id"]),
            frame_id=frame_id,
            siglip_score=float(prepared.scores[observation_index]),
            confidence=verification.confidence,
        )
        filename = (
            f"{render_index:04d}_{str(candidate['candidate_id']).replace('/', '-')}_"
            f"frame_{frame_id:06d}.jpg"
        )
        path = frames_directory / filename
        image.save(path, quality=92)
        rendered_paths.append(path)
        entity_lookup[str(candidate["candidate_id"])]["observations"].append(
            {
                "proposal_id": observation["proposal_id"],
                "frame_id": frame_id,
                "timestamp": observation.get("timestamp"),
                "siglip_score": float(prepared.scores[observation_index]),
                "observation_quality": float(prepared.qualities[observation_index]),
                "image": str(path.relative_to(prepared.output)),
            }
        )
    _write_media(
        rendered_paths,
        prepared.output,
        width=gif_width,
        duration_ms=gif_duration_ms,
    )

    candidate_cards = []
    for candidate in checked:
        verification = candidate["verification"]
        evidence = "".join(
            f'<img src="{escape(str(path.relative_to(prepared.output)))}">'
            for path in candidate["evidence_images"]
        )
        state = "accepted" if candidate["accepted"] else "rejected"
        candidate_cards.append(
            f'<section class="{state}"><h2>{escape(str(candidate["candidate_id"]))} · '
            f'{state} · {verification.confidence:.2f}</h2><p>'
            f'{escape(verification.reason)}</p><div>{evidence}</div></section>'
        )
    result_cards = "".join(
        f'<figure><img src="{escape(str(path.relative_to(prepared.output)))}"></figure>'
        for path in rendered_paths
    )
    prepared.output.mkdir(parents=True, exist_ok=True)
    (prepared.output / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>VLM-verified Fact3R query</title>"
        "<style>body{background:#151515;color:#eee;font-family:sans-serif;margin:24px}"
        "section{border-left:6px solid #c44;background:#222;padding:10px;margin:12px 0}"
        "section.accepted{border-color:#3c7}section div{display:flex;gap:8px;overflow:auto}"
        "section img{width:31%;min-width:320px;height:auto}main{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}"
        "figure{margin:0}figure img{width:100%;height:auto}</style>"
        f"<h1>Query: {escape(prepared.query)}</h1>"
        f"<p>{len(accepted)} verified entities; {len(rejected)} rejected candidates.</p>"
        f"<h2>Candidate verification</h2>{''.join(candidate_cards)}"
        f"<h2>All observations of accepted entities</h2><main>{result_cards}</main>",
        encoding="utf-8",
    )

    rejected_results = [
        {
            "shortlist_rank": candidate["shortlist_rank"],
            "entity_id": candidate["entity_id"],
            "track_id": candidate["track_id"],
            "candidate_id": candidate["candidate_id"],
            "siglip_candidate_score": candidate["candidate_score"],
            "observation_count": candidate["observation_count"],
            "vlm": asdict(candidate["verification"]),
            "rejection_reasons": candidate["rejection_reasons"],
            "cache_hit": candidate["cache_hit"],
            "evidence_images": [
                str(path.relative_to(prepared.output))
                for path in candidate["evidence_images"]
            ],
        }
        for candidate in rejected
    ]
    result = {
        "format": "fact3r-vlm-verified-query-results",
        "version": 1,
        "query": prepared.query,
        "source_index": str(prepared.manifest_path.resolve()),
        "siglip_model": prepared.manifest["model"],
        "vlm_model": verifier.model_name,
        "positive_prompts": list(prepared.positive_prompts),
        "config": {
            "max_candidates": len(prepared.candidates),
            "min_vlm_confidence": min_confidence,
            "min_vlm_supporting_views": min_supporting_views,
            "max_entities": max_entities,
            "prompt_version": PROMPT_VERSION,
            "map_hard_negative_neighbors": prepared.map_negative_neighbors,
            "map_hard_negative_weight": prepared.map_negative_weight,
        },
        "confident_match_found": bool(entity_results),
        "selected_entity_count": len(entity_results),
        "checked_candidate_count": len(checked),
        "rendered_observation_count": len(rendered_paths),
        "dynamic_confounders": _dynamic_confounders(prepared.query, rejected),
        "entities": entity_results,
        "rejected_candidates": rejected_results,
        "timing": {
            "siglip_model_load_seconds": prepared.siglip_load_seconds,
            "siglip_text_encoding_seconds": prepared.text_encoding_seconds,
            "candidate_preparation_seconds": prepared.preparation_seconds,
            "vlm_model_load_seconds": float(verifier.load_seconds),
            "vlm_inference_seconds": verification_seconds,
            "vlm_cache_hits": cache_hits,
            "verification_and_render_seconds": perf_counter() - started,
        },
        "gallery": "index.html",
        "gif": "matches.gif" if rendered_paths else None,
        "contact_sheet": "contact_sheet.jpg" if rendered_paths else None,
    }
    result_path = prepared.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path
