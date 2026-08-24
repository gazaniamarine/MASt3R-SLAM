"""Point-grounded semantic fact contract; extraction starts in Milestone 4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SupportType(str, Enum):
    ENTIRE_ENTITY = "entire_entity"
    PART_ID = "part_id"
    SPARSE_POINTS = "sparse_points"
    LOCAL_3D_BOX = "local_3d_box"


@dataclass(slots=True)
class SemanticFact:
    id: str
    entity_id: str
    subject: str
    predicate: str
    value: Any
    support_type: SupportType
    support_reference: Any
    posterior_probability: float
    supporting_view_count: int = 0
    contradictory_view_count: int = 0
    first_seen_timestamp: float | str | None = None
    last_seen_timestamp: float | str | None = None
    provenance_summary: str = ""

    def __post_init__(self) -> None:
        self.support_type = SupportType(self.support_type)
        if not 0.0 <= self.posterior_probability <= 1.0:
            raise ValueError("posterior_probability must be in [0, 1]")
        if self.supporting_view_count < 0 or self.contradictory_view_count < 0:
            raise ValueError("fact view counts cannot be negative")

