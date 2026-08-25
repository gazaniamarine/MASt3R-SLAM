"""Image overlays for inspecting persistent proposal-to-entity association."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class DisplayAssignment:
    proposal_id: str
    entity_id: str
    status: str


@dataclass(frozen=True, slots=True)
class DisplayFrame:
    frame_id: int
    matched_count: int
    created_count: int
    entity_count: int
    unmatched_reason_counts: Mapping[str, int]
    assignments: tuple[DisplayAssignment, ...]
    pending_count: int = 0
    held_count: int = 0
    converged: bool | None = None
    iterations: int | None = None
    forbidden_mass: float | None = None


def display_frame_from_manifest(entry: Mapping[str, object]) -> DisplayFrame:
    """Normalize Hungarian or Sinkhorn frame JSON for rendering."""

    assignments = [
        DisplayAssignment(
            proposal_id=str(match["proposal_id"]),
            entity_id=str(match["entity_id"]),
            status="matched",
        )
        for match in entry.get("matches", [])
    ]
    for unmatched in entry.get("unmatched_proposals", []):
        entity_id = unmatched.get("created_entity_id")
        if entity_id is not None:
            assignments.append(
                DisplayAssignment(
                    proposal_id=str(unmatched["proposal_id"]),
                    entity_id=str(entity_id),
                    status="created",
                )
            )
            continue
        commitment_status = unmatched.get("commitment_status")
        if commitment_status == "deferred" and unmatched.get("track_id") is not None:
            assignments.append(
                DisplayAssignment(
                    proposal_id=str(unmatched["proposal_id"]),
                    entity_id=str(unmatched["track_id"]),
                    status="pending",
                )
            )
        elif (
            commitment_status == "held_existing"
            and unmatched.get("resolved_entity_id") is not None
        ):
            assignments.append(
                DisplayAssignment(
                    proposal_id=str(unmatched["proposal_id"]),
                    entity_id=str(unmatched["resolved_entity_id"]),
                    status="held",
                )
            )
    return DisplayFrame(
        frame_id=int(entry["frame_id"]),
        matched_count=len(entry.get("matches", [])),
        created_count=len(entry.get("created_entity_ids", [])),
        entity_count=int(entry["entity_count_after"]),
        unmatched_reason_counts={
            str(reason): int(count)
            for reason, count in entry.get("unmatched_reason_counts", {}).items()
        },
        assignments=tuple(assignments),
        pending_count=sum(item.status == "pending" for item in assignments),
        held_count=sum(item.status == "held" for item in assignments),
        converged=(
            None if "converged" not in entry else bool(entry["converged"])
        ),
        iterations=(
            None if "iterations" not in entry else int(entry["iterations"])
        ),
        forbidden_mass=(
            None
            if "noncandidate_mass" not in entry
            and "forbidden_mass" not in entry
            else float(
                entry.get("noncandidate_mass", entry.get("forbidden_mass"))
            )
        ),
    )


def entity_colour(entity_id: str) -> NDArray[np.uint8]:
    """Return a stable, bright RGB colour for one persistent ID."""

    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=3).digest()
    colour = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)
    colour = 70.0 + 185.0 * colour / 255.0
    return colour.astype(np.uint8)


def mask_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return a one-pixel four-connected inner boundary."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape (height, width)")
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def _rgb_uint8(rgb: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def _short_entity_id(entity_id: str) -> str:
    suffix = entity_id.rsplit("-", 1)[-1]
    try:
        prefix = "T" if entity_id.startswith("track-") else "E"
        return f"{prefix}{int(suffix)}"
    except ValueError:
        return entity_id[:12]


def _header_lines(title: str, frame: DisplayFrame) -> tuple[str, str]:
    first = (
        f"{title} | frame {frame.frame_id} | matched {frame.matched_count} | "
        f"new {frame.created_count} | pending {frame.pending_count} | "
        f"held {frame.held_count} | entities {frame.entity_count}"
    )
    reasons = ", ".join(
        f"{name}={count}"
        for name, count in frame.unmatched_reason_counts.items()
        if count
    ) or "unmatched=none"
    if frame.converged is not None:
        reasons += (
            f" | converged={frame.converged} iter={frame.iterations} "
            f"forbidden={frame.forbidden_mass:.3f}"
        )
    return first, reasons


def render_association_panel(
    rgb: NDArray[np.generic],
    masks: Mapping[str, NDArray[np.bool_]],
    frame: DisplayFrame,
    *,
    title: str,
    alpha: float = 0.45,
) -> Image.Image:
    """Render persistent entity colours and match/new boundaries over RGB."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    canvas = _rgb_uint8(rgb).astype(np.float64)
    height, width = canvas.shape[:2]
    assignments = sorted(frame.assignments, key=lambda item: item.proposal_id)
    visible: list[tuple[DisplayAssignment, NDArray[np.bool_]]] = []
    for assignment in assignments:
        if assignment.proposal_id not in masks:
            continue
        mask = np.asarray(masks[assignment.proposal_id], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"mask {assignment.proposal_id!r} does not match RGB shape"
            )
        colour = entity_colour(assignment.entity_id).astype(np.float64)
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * colour
        visible.append((assignment, mask))

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for assignment, mask in visible:
        boundary_colour = {
            "matched": np.asarray([40, 255, 80], dtype=np.uint8),
            "created": np.asarray([255, 55, 55], dtype=np.uint8),
            "pending": np.asarray([255, 210, 40], dtype=np.uint8),
            "held": np.asarray([40, 220, 255], dtype=np.uint8),
        }.get(assignment.status, np.asarray([255, 255, 255], dtype=np.uint8))
        canvas[mask_boundary(mask)] = boundary_colour

    header_height = 48
    panel = Image.new("RGB", (width, height + header_height), (15, 15, 15))
    panel.paste(Image.fromarray(canvas), (0, header_height))
    draw = ImageDraw.Draw(panel)
    first, second = _header_lines(title, frame)
    draw.text((6, 5), first, fill=(255, 255, 255))
    draw.text((6, 25), second, fill=(220, 220, 220))
    for assignment, mask in visible:
        coordinates = np.argwhere(mask)
        if len(coordinates) == 0:
            continue
        row, column = np.median(coordinates, axis=0).astype(int)
        label = _short_entity_id(assignment.entity_id)
        label += " M" if assignment.status == "matched" else " N"
        x = int(np.clip(column, 0, width - 1))
        y = int(np.clip(row + header_height, header_height, height + header_height - 1))
        box = draw.textbbox((x, y), label)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x, y), label, fill=(255, 255, 255))
    return panel


def render_rgb_panel(
    rgb: NDArray[np.generic], *, frame_id: int
) -> Image.Image:
    image = Image.fromarray(_rgb_uint8(rgb))
    panel = Image.new("RGB", (image.width, image.height + 48), (15, 15, 15))
    panel.paste(image, (0, 48))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), f"RGB | frame {frame_id}", fill=(255, 255, 255))
    draw.text(
        (6, 25),
        "green boundary = matched | red boundary = new entity",
        fill=(220, 220, 220),
    )
    return panel


def join_panels(panels: tuple[Image.Image, ...], gap: int = 4) -> Image.Image:
    if not panels:
        raise ValueError("at least one panel is required")
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    montage = Image.new("RGB", (width, height), (35, 35, 35))
    offset = 0
    for panel in panels:
        montage.paste(panel, (offset, 0))
        offset += panel.width + gap
    return montage
