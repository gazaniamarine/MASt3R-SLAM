"""Backend-neutral interface for class-agnostic 2D mask generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MaskProposal2D:
    proposal_id: str
    frame_id: int
    mask: NDArray[np.bool_]
    score: float
    bounding_box_xyxy: NDArray[np.floating] | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must have shape (height, width)")
        if not np.isfinite(self.score):
            raise ValueError("mask score must be finite")
        box = None
        if self.bounding_box_xyxy is not None:
            box = np.asarray(self.bounding_box_xyxy, dtype=np.float32)
            if box.shape != (4,):
                raise ValueError("bounding_box_xyxy must have shape (4,)")
        object.__setattr__(self, "mask", np.ascontiguousarray(mask))
        object.__setattr__(self, "bounding_box_xyxy", box)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class MaskGenerator(Protocol):
    def generate(
        self, rgb: NDArray[np.generic], *, frame_id: int
    ) -> Sequence[MaskProposal2D]: ...

