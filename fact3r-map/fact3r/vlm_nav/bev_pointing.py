"""Qwen points at a pixel on the rendered semantic BEV; A* still plans to it.

This module itself is SigLIP-free: given whatever `groups` it is handed, it
renders each one as a numbered marker at its centroid and asks Qwen which
number (or, failing that, which pixel) is the target. It has no opinion on
how that `groups` list was chosen.

In production that list is not the whole map, though. `resolve_semantic_goal_vlm.py`
(Pipeline B's driver) runs a SigLIP shortlist stage before ever calling
`render_pointing_bev` here, narrowing thousands of manifest entities down to
`--siglip-top-k` (default 8) candidates -- see that script's own docstring for
why: a real scan holds too many entities to render legibly at once, and a
naive size-based cut drops small-but-correct targets before Qwen ever sees
them. So end-to-end, Pipeline B is SigLIP-shortlist-then-Qwen-points, not
SigLIP-free; this module is just the second half of that.

One limitation worth stating up front rather than discovering by surprise:
entities carry no human-readable label anywhere in this codebase's manifests
-- `build_depth_semantic_bev.py` colours a group by hashing its `group_id`,
nothing more. So Qwen is not reading names off the map; it is matching a text
query to a coloured, numbered blob purely by position, size, and shape. That
may simply not work well for some queries -- that is an open question raised
before this was built, not a bug in it, and the fix (if one is needed) is
probably richer visual context per marker, not a change to this module's
contract.

`resolve_pointing` produces the exact candidate-geometry shape
`resolve_semantic_goal.py` already produces, so `project_semantic_goal.py`
and everything after it run completely unmodified against either pipeline's
output.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from fact3r.semantics.semantic_goal import cell_centre_xy, group_cell_counts, weighted_centroid_cell


FloatArray = NDArray[np.floating]

# Free/occupied/unknown thresholds match build_depth_semantic_bev.py's own
# render, so this map looks like the one an operator already knows how to read.
_FREE = 245
_OCCUPIED = 35
_UNKNOWN = 128
_OCCUPANCY_OCCUPIED_THRESHOLD = 65

_ROBOT_MARKER = "R"


def _colour_for_group(group_id: str) -> tuple[int, int, int]:
    """Deterministic per-entity colour, identical to build_depth_semantic_bev.py.

    Reimplemented rather than imported: that function lives in a script, and
    importing a script module for one three-line hash is more coupling than
    the hash is worth. Keeping the formula identical is what matters -- a map
    rendered here should tint the same entity the same colour as
    map_semantic.png already does.
    """

    digest = sha256(group_id.encode("utf-8")).digest()
    return 55 + digest[0] % 190, 55 + digest[1] % 190, 55 + digest[2] % 190


def _cell_to_pixel(row: float, col: float, height: int) -> tuple[int, int]:
    """Grid (row, col) -> image (x, y), undoing the top-down row flip.

    The render flips rows (`canvas[::-1]`) so row 0 -- the low-y edge -- ends
    up at the bottom of the image, matching how a floor plan is normally read.
    Every pixel placed on the image has to invert that, and every pixel read
    back off it (`resolve_pointing`'s freeform fallback) has to invert it the
    same way, or the two silently disagree about which half of the map is
    which -- the exact failure mode semantic_goal.py's own docstring warns
    about for the (y, x) / (row, col) boundary.
    """

    return int(round(col)), int(height - 1 - round(row))


def _draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, *, radius: int = 10) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(250, 235, 40), outline=(20, 20, 20))
    # PIL's default bitmap font is small and legible at this size without
    # bundling a font file, matching every other renderer in this codebase.
    bbox = draw.textbbox((0, 0), text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x - text_w / 2, y - text_h / 2 - 1), text, fill=(10, 10, 10))


def _draw_robot(draw: ImageDraw.ImageDraw, x: int, y: int, *, size: int = 12) -> None:
    draw.polygon(
        [(x, y - size), (x - size, y + size), (x + size, y + size)],
        fill=(230, 30, 30),
        outline=(255, 255, 255),
    )
    draw.text((x - 4, y - size - 14), _ROBOT_MARKER, fill=(230, 30, 30))


def render_pointing_bev(
    occupancy: NDArray[np.integer],
    semantic_ids: NDArray[np.integer],
    groups: Sequence[Mapping[str, object]],
    *,
    origin_xy: Sequence[float],
    resolution: float,
    confidence: FloatArray | None = None,
    robot_cell: tuple[float, float] | None = None,
    min_cells: int = 3,
    max_markers: int = 60,
    output: Path,
) -> tuple[Path, dict[int, dict[str, object]]]:
    """Render the BEV with numbered markers, and return where each one is.

    Only entities holding at least `min_cells` are offered a marker -- a
    single stray cell is more visual noise than signal, and `max_markers`
    keeps a dense map legible instead of the numbers overlapping into a wall
    of text. Both are exactly the kind of thing worth tuning once real runs
    show whether the target is usually being cut from the list.
    """

    if occupancy.shape != semantic_ids.shape:
        raise ValueError("occupancy and semantic_ids must share a shape")
    height, width = occupancy.shape

    canvas = np.full((height, width, 3), _UNKNOWN, dtype=np.uint8)
    canvas[occupancy >= 0] = _FREE
    canvas[occupancy >= _OCCUPANCY_OCCUPIED_THRESHOLD] = _OCCUPIED

    cell_counts = group_cell_counts(semantic_ids, groups)
    markable = [g for g in groups if cell_counts.get(str(g["group_id"]), 0) >= min_cells]
    # Most-supported entities first, so if the cap trims the list it trims the
    # least-observed entities, not an arbitrary manifest-order prefix.
    markable.sort(key=lambda g: cell_counts[str(g["group_id"])], reverse=True)
    markable = markable[:max_markers]

    for group in markable:
        mask = semantic_ids == int(group["semantic_id"])
        colour = np.asarray(_colour_for_group(str(group["group_id"])), dtype=np.float32)
        canvas[mask] = (0.25 * canvas[mask] + 0.75 * colour).astype(np.uint8)

    image = Image.fromarray(canvas[::-1].copy())
    draw = ImageDraw.Draw(image)

    markers: dict[int, dict[str, object]] = {}
    for number, group in enumerate(markable, start=1):
        semantic_id = int(group["semantic_id"])
        rows, cols = np.nonzero(semantic_ids == semantic_id)
        weights = confidence[rows, cols] if confidence is not None else np.ones(len(rows))
        centroid_row, centroid_col = weighted_centroid_cell(rows, cols, weights)
        x, y = _cell_to_pixel(centroid_row, centroid_col, height)
        _draw_marker(draw, x, y, str(number))
        centroid_x, centroid_y = cell_centre_xy(centroid_row, centroid_col, origin_xy, resolution)
        markers[number] = {
            "group_id": str(group["group_id"]),
            "semantic_id": semantic_id,
            "centroid_cell_rc": [float(centroid_row), float(centroid_col)],
            "centroid_yx": [float(centroid_y), float(centroid_x)],
            "cell_count": int(cell_counts[str(group["group_id"])]),
        }

    if robot_cell is not None:
        x, y = _cell_to_pixel(robot_cell[0], robot_cell[1], height)
        _draw_robot(draw, x, y)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output, markers


def build_pointing_prompt(query: str, markers: Mapping[int, Mapping[str, object]]) -> str:
    """The prompt Qwen sees alongside the rendered map image."""

    return (
        "You are locating a target for a mobile robot on its own top-down "
        "semantic occupancy map. Light gray is free space, mid gray is "
        "unexplored, dark is a wall or obstacle. Every yellow numbered circle "
        "marks one distinct object the robot has already mapped, sitting on "
        "top of that object's coloured region; the same object never carries "
        "two numbers. A red marker labelled R, if present, is the robot's own "
        "current position -- never answer with R, it is not a target.\n\n"
        f"Target: {json.dumps(query.strip())}\n"
        f"Marker numbers present on this map: {sorted(markers)}\n\n"
        "There is no name written on the map, so judge each marker by its "
        "region's position, size, and shape relative to the rest of the "
        "floor plan. Pick the single numbered marker that is the target. If "
        "you are confident none of the numbered markers is it, set marker to "
        "null and instead give your best guess as normalized image "
        "coordinates x, y in [0, 1] with (0, 0) at the top-left. Return ONLY "
        "one JSON object with exactly these fields:\n"
        '{"marker": <integer or null>, "x": 0.0, "y": 0.0, '
        '"reason": "one short sentence"}'
    )


@dataclass(frozen=True, slots=True)
class PointingResult:
    marker: int | None
    x: float
    y: float
    reason: str
    raw_output: str = ""


def parse_pointing_output(text: str) -> PointingResult:
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
    marker = payload.get("marker")
    if marker is not None:
        try:
            marker = int(marker)
        except (TypeError, ValueError) as error:
            raise ValueError("marker must be an integer or null") from error
    x = float(payload.get("x", 0.5))
    y = float(payload.get("y", 0.5))
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("x/y must be finite")
    return PointingResult(
        marker=marker,
        x=float(np.clip(x, 0.0, 1.0)),
        y=float(np.clip(y, 0.0, 1.0)),
        reason=str(payload.get("reason", "")).strip(),
        raw_output=text,
    )


def _nearest_labelled_group(
    semantic_ids: NDArray[np.integer],
    groups: Sequence[Mapping[str, object]],
    row: int,
    col: int,
    radius_cells: float,
) -> tuple[str, int]:
    """The mapped entity nearest a freeform pixel, within a search radius.

    The same idea as `nearest_cell_in`'s navigability projection, applied to
    "is there an entity here" instead of "can the robot stand here": a
    freeform point Qwen names is a seed, not automatically a legal answer.
    """

    labelled = semantic_ids >= 0
    if not labelled.any():
        raise ValueError("no entity is mapped anywhere on this grid")
    rows, cols = np.nonzero(labelled)
    distance = np.hypot(rows - row, cols - col)
    winner = int(np.argmin(distance))
    if float(distance[winner]) > radius_cells:
        raise ValueError(
            f"pointed pixel is {float(distance[winner]):.1f} cells from the "
            f"nearest mapped entity, past the {radius_cells:.1f}-cell search radius"
        )
    semantic_id = int(semantic_ids[rows[winner], cols[winner]])
    lookup = {int(g["semantic_id"]): str(g["group_id"]) for g in groups}
    group_id = lookup.get(semantic_id)
    if group_id is None:
        raise ValueError(f"semantic id {semantic_id} has no group in the manifest")
    return group_id, semantic_id


def resolve_pointing(
    result: PointingResult,
    markers: Mapping[int, Mapping[str, object]],
    *,
    semantic_ids: NDArray[np.integer],
    groups: Sequence[Mapping[str, object]],
    confidence: FloatArray | None,
    origin_xy: Sequence[float],
    resolution: float,
    fallback_search_radius_cells: float = 20.0,
) -> dict[str, object]:
    """Turn a parsed pointing answer into `resolve_semantic_goal.py`'s candidate shape.

    Same fields (`group_id`, `cell_count`, `centroid_yx`, `centroid_cell_rc`,
    `cells_yx`, `cell_weights`) as the SigLIP path produces, so
    `project_semantic_goal.py` needs no changes to accept either.
    """

    if result.marker is not None and result.marker in markers:
        marker = markers[result.marker]
        group_id, semantic_id = str(marker["group_id"]), int(marker["semantic_id"])
    else:
        height, width = semantic_ids.shape
        col = int(np.clip(round(result.x * (width - 1)), 0, width - 1))
        row_from_top = int(np.clip(round(result.y * (height - 1)), 0, height - 1))
        row = height - 1 - row_from_top
        group_id, semantic_id = _nearest_labelled_group(
            semantic_ids, groups, row, col, fallback_search_radius_cells
        )

    rows, cols = np.nonzero(semantic_ids == semantic_id)
    if not len(rows):
        raise ValueError(f"group {group_id} holds no BEV cell")
    weights = confidence[rows, cols] if confidence is not None else np.ones(len(rows))
    centroid_row, centroid_col = weighted_centroid_cell(rows, cols, weights)
    centroid_x, centroid_y = cell_centre_xy(centroid_row, centroid_col, origin_xy, resolution)
    cell_x, cell_y = cell_centre_xy(rows, cols, origin_xy, resolution)
    return {
        "group_id": group_id,
        "semantic_id": semantic_id,
        "cell_count": int(len(rows)),
        "centroid_cell_rc": [float(centroid_row), float(centroid_col)],
        "centroid_yx": [float(centroid_y), float(centroid_x)],
        "cells_rc": np.stack([rows, cols], axis=1).astype(int).tolist(),
        "cells_yx": np.stack([cell_y, cell_x], axis=1).tolist(),
        "cell_weights": np.asarray(weights, dtype=float).tolist(),
    }


class Qwen3VLPointer:
    """Lazy Hugging Face adapter for Qwen3-VL, prompted to point at a map pixel.

    A deliberate twin of `fact3r.semantics.vlm_verification.Qwen3VLVerifier`,
    not a subclass or a shared base -- Pipeline A's verifier stays untouched so
    it keeps working exactly as it does today no matter what changes here, and
    this class is free to use a different checkpoint or generation setting
    without risking that.

    Defaults to Qwen3-VL-2B-Instruct, not the 8B verifier default: this is the
    checkpoint meant to end up fine-tuned and shared with the NavDP
    integration later, so it is the one every part of this pipeline should
    exercise now, not a bigger stand-in swapped out afterwards.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        attention_implementation: str | None = None,
        max_new_tokens: int = 200,
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

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def load(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        started = perf_counter()
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL pointing requires transformers>=4.57, torch, "
                "accelerate, and Pillow; install the fact3r-map vlm extras"
            ) from error
        model_kwargs: dict[str, object] = {
            "dtype": self._dtype if self._dtype == "auto" else getattr(torch, self._dtype),
            "device_map": self._device_map,
        }
        if self._attention_implementation is not None:
            model_kwargs["attn_implementation"] = self._attention_implementation
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self._model_name, **model_kwargs)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._torch = torch
        self._load_seconds = perf_counter() - started

    def point(
        self,
        *,
        query: str,
        map_image: Path,
        markers: Mapping[int, Mapping[str, object]],
    ) -> PointingResult:
        resolved = Path(map_image).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"rendered BEV image does not exist: {resolved}")
        content = [
            {"type": "image", "image": str(resolved)},
            {"type": "text", "text": build_pointing_prompt(query, markers)},
        ]
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
            return parse_pointing_output(output)
        except ValueError as error:
            return PointingResult(
                marker=None,
                x=0.5,
                y=0.5,
                reason=f"invalid structured VLM output: {error}",
                raw_output=output,
            )

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
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
