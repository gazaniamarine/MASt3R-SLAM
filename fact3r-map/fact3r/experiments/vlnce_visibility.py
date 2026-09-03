"""Decide whether a tour's return target was actually seen on the way out.

A landmark mined from an instruction ("...stop near the rug") says the object is
near the goal. It does not say the camera ever looked at it. Without this check
a failed return is ambiguous: the map may have missed the object, or the object
may never have been observable in the first place, and those call for opposite
fixes.

The renderer supplies per-instance visibility from habitat's semantic sensor;
everything here is pure so the thresholds can be tested and tuned offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# MP3D category names are terse and sometimes differ from how people speak.
# Deliberately small: a wrong synonym silently mislabels a tour as usable.
_SYNONYMS: Dict[str, tuple] = {
    "couch": ("sofa", "seating"),
    "sofa": ("couch", "seating"),
    "tv": ("tv_monitor", "television", "monitor"),
    "television": ("tv_monitor", "tv", "monitor"),
    "monitor": ("tv_monitor", "tv"),
    "rug": ("carpet", "mat", "floor mat"),
    "carpet": ("rug", "mat"),
    "painting": ("picture", "artwork"),
    "picture": ("painting", "artwork"),
    "stairs": ("stairway", "staircase", "steps", "stair"),
    "stair": ("stairs", "stairway", "staircase", "step"),
    "step": ("stairs", "stair", "steps"),
    "doorway": ("door", "door frame", "entrance"),
    "door": ("doorway", "door frame"),
    "archway": ("arch", "doorway"),
    "counter": ("countertop", "counter top"),
    "fridge": ("refrigerator",),
    "refrigerator": ("fridge",),
    "cupboard": ("cabinet",),
    "cabinet": ("cupboard", "chest_of_drawers"),
    "bin": ("trash can", "garbage bin"),
    "plant": ("indoor plant", "potted plant"),
}

_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


class Verdict:
    OBSERVED = "observed"
    GLIMPSED = "glimpsed"
    NEVER_OBSERVED = "never_observed"
    NO_CANDIDATE = "no_candidate_instance"


@dataclass(frozen=True)
class InstanceVisibility:
    """How much of the outbound tour one semantic instance appeared in."""

    instance_id: int
    category: str
    center: np.ndarray
    frames_visible: int
    total_pixels: int
    max_pixel_fraction: float
    first_frame: int
    last_frame: int

    @classmethod
    def from_json(cls, record: dict) -> "InstanceVisibility":
        return cls(
            instance_id=int(record["instance_id"]),
            category=str(record.get("category", "")),
            center=np.asarray(record.get("center", [0.0, 0.0, 0.0]), dtype=np.float64),
            frames_visible=int(record.get("frames_visible", 0)),
            total_pixels=int(record.get("total_pixels", 0)),
            max_pixel_fraction=float(record.get("max_pixel_fraction", 0.0)),
            first_frame=int(record.get("first_frame", -1)),
            last_frame=int(record.get("last_frame", -1)),
        )


@dataclass
class TargetVisibility:
    """The verdict for one tour's return target."""

    query: str
    verdict: str
    candidates: List[InstanceVisibility] = field(default_factory=list)
    best: Optional[InstanceVisibility] = None
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.verdict == Verdict.OBSERVED

    def to_json(self) -> dict:
        return {
            "query": self.query,
            "verdict": self.verdict,
            "usable": self.usable,
            "reason": self.reason,
            "candidate_count": len(self.candidates),
            "best": None
            if self.best is None
            else {
                "instance_id": self.best.instance_id,
                "category": self.best.category,
                "frames_visible": self.best.frames_visible,
                "max_pixel_fraction": self.best.max_pixel_fraction,
                "first_frame": self.best.first_frame,
                "last_frame": self.best.last_frame,
            },
        }


def normalise(text: str) -> str:
    text = str(text).replace("_", " ").strip().lower()
    return _ARTICLES.sub("", text).strip()


def _expand(term: str) -> set:
    terms = {term}
    for synonym in _SYNONYMS.get(term, ()):
        terms.add(normalise(synonym))
    head = term.split()[-1] if term.split() else term
    terms.add(head)
    for synonym in _SYNONYMS.get(head, ()):
        terms.add(normalise(synonym))
    return terms


def category_matches(landmark: str, category: str) -> bool:
    """Whether an MP3D category plausibly names the mined landmark.

    Matching is generous on wording but never crosses head nouns: "pool table"
    matches `table`, and "small table" matches `table`, but "table" never
    matches `chair`.
    """

    landmark_text, category_text = normalise(landmark), normalise(category)
    if not landmark_text or not category_text:
        return False
    if landmark_text == category_text:
        return True

    landmark_terms, category_terms = _expand(landmark_text), _expand(category_text)
    if landmark_terms & category_terms:
        return True
    # "glass dining table" vs category "table": the category is a trailing word.
    landmark_words = landmark_text.split()
    if category_text in landmark_words:
        return True
    return landmark_text in category_text.split()


def select_candidates(
    instances: Sequence[InstanceVisibility],
    landmark: str,
    target_position: Sequence[float],
    *,
    radius: float = 3.0,
) -> List[InstanceVisibility]:
    """Instances whose category fits the landmark and that sit near the goal.

    The radius matches the VLN-CE success threshold: an object further than that
    from the goal is not what "stop near the X" referred to.
    """

    target = np.asarray(target_position, dtype=np.float64)
    matched = []
    for instance in instances:
        if not category_matches(landmark, instance.category):
            continue
        if float(np.linalg.norm(instance.center - target)) > radius:
            continue
        matched.append(instance)
    return matched


def score_target(
    instances: Sequence[InstanceVisibility],
    landmark: str,
    target_position: Sequence[float],
    *,
    radius: float = 3.0,
    min_frames: int = 5,
    min_pixel_fraction: float = 0.005,
) -> TargetVisibility:
    """Judge whether the return target was observed well enough to be findable.

    `min_pixel_fraction` of 0.5% of a 512x384 frame is roughly 1000 px, about
    the smallest region SAM2 reliably proposes as its own mask.
    """

    candidates = select_candidates(
        instances, landmark, target_position, radius=radius
    )
    if not candidates:
        return TargetVisibility(
            query=landmark,
            verdict=Verdict.NO_CANDIDATE,
            reason=(
                "no semantic instance within %.1f m of the goal matches %r; the "
                "instruction landmark may not name a mapped object"
                % (radius, landmark)
            ),
        )

    best = max(
        candidates,
        key=lambda instance: (instance.max_pixel_fraction, instance.frames_visible),
    )
    if best.frames_visible == 0 or best.max_pixel_fraction <= 0.0:
        verdict, reason = (
            Verdict.NEVER_OBSERVED,
            "the matching instance never appeared in an outbound frame",
        )
    elif best.frames_visible < min_frames or best.max_pixel_fraction < min_pixel_fraction:
        verdict, reason = (
            Verdict.GLIMPSED,
            "seen in %d frames at most %.3f%% of a frame; below the %d-frame / "
            "%.3f%% bar for a findable object"
            % (
                best.frames_visible,
                100.0 * best.max_pixel_fraction,
                min_frames,
                100.0 * min_pixel_fraction,
            ),
        )
    else:
        verdict, reason = (
            Verdict.OBSERVED,
            "seen in %d frames, peaking at %.2f%% of a frame"
            % (best.frames_visible, 100.0 * best.max_pixel_fraction),
        )
    return TargetVisibility(
        query=landmark,
        verdict=verdict,
        candidates=candidates,
        best=best,
        reason=reason,
    )


def summarise(results: Sequence[TargetVisibility]) -> dict:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.verdict] = counts.get(result.verdict, 0) + 1
    return {
        "tours": len(results),
        "usable": sum(1 for result in results if result.usable),
        "by_verdict": counts,
    }
