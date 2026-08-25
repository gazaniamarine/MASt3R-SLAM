"""Utilities for a dense one-second HM3D mask-stability experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fact3r.association.tracklets import link_propagated_masks
from fact3r.proposals.mask_generator import MaskProposal2D


def select_frame_window(
    *,
    total_frames: int,
    fps: float,
    duration_seconds: float = 1.0,
    start_frame: int | None = None,
    start_second: float | None = None,
) -> tuple[int, ...]:
    """Return every captured frame in a fixed-duration video window."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if start_frame is not None and start_second is not None:
        raise ValueError("provide either start_frame or start_second, not both")
    if start_second is not None:
        if not np.isfinite(start_second) or start_second < 0.0:
            raise ValueError("start_second must be finite and non-negative")
        start = int(round(start_second * fps))
    else:
        start = 0 if start_frame is None else int(start_frame)
    if start < 0 or start >= total_frames:
        raise ValueError("the requested window starts outside the sequence")
    count = max(1, int(round(duration_seconds * fps)))
    return tuple(range(start, min(total_frames, start + count)))


@dataclass(frozen=True, slots=True)
class MaskTrackObservation:
    proposal_id: str
    track_id: str
    source_proposal_id: str | None
    link_iou: float | None


@dataclass(frozen=True, slots=True)
class TrackedMaskFrame:
    frame_id: int
    observations: tuple[MaskTrackObservation, ...]
    linked_count: int
    new_track_count: int


class AdjacentMaskTracker:
    """Link independently generated masks by one-to-one adjacent-frame IoU.

    This intentionally measures raw automatic-mask stability. It does not use
    SAM2 video propagation, 3D geometry, or UOT, so each can be added later as a
    controlled change.
    """

    def __init__(self, min_mask_iou: float = 0.30) -> None:
        if not 0.0 <= min_mask_iou <= 1.0:
            raise ValueError("min_mask_iou must be in [0, 1]")
        self.min_mask_iou = min_mask_iou
        self._next_track_index = 0
        self._source_proposals: tuple[MaskProposal2D, ...] = ()
        self._source_tracks: dict[str, str] = {}

    @property
    def track_count(self) -> int:
        return self._next_track_index

    def _new_track(self) -> str:
        track_id = f"track-{self._next_track_index:06d}"
        self._next_track_index += 1
        return track_id

    def update(
        self,
        frame_id: int,
        proposals: Sequence[MaskProposal2D],
    ) -> TrackedMaskFrame:
        proposals = tuple(proposals)
        if any(proposal.frame_id != frame_id for proposal in proposals):
            raise ValueError("every proposal must belong to the tracked frame")

        if not self._source_proposals:
            observations = tuple(
                MaskTrackObservation(
                    proposal_id=proposal.proposal_id,
                    track_id=self._new_track(),
                    source_proposal_id=None,
                    link_iou=None,
                )
                for proposal in proposals
            )
            linked_count = 0
        else:
            links = link_propagated_masks(
                tuple(item.proposal_id for item in self._source_proposals),
                tuple(item.mask for item in self._source_proposals),
                tuple(item.proposal_id for item in proposals),
                tuple(item.mask for item in proposals),
                min_mask_iou=self.min_mask_iou,
            )
            links_by_target = {link.target_proposal_id: link for link in links}
            built: list[MaskTrackObservation] = []
            for proposal in proposals:
                link = links_by_target.get(proposal.proposal_id)
                if link is None:
                    built.append(
                        MaskTrackObservation(
                            proposal_id=proposal.proposal_id,
                            track_id=self._new_track(),
                            source_proposal_id=None,
                            link_iou=None,
                        )
                    )
                else:
                    built.append(
                        MaskTrackObservation(
                            proposal_id=proposal.proposal_id,
                            track_id=self._source_tracks[
                                link.source_proposal_id
                            ],
                            source_proposal_id=link.source_proposal_id,
                            link_iou=link.mask_iou,
                        )
                    )
            observations = tuple(built)
            linked_count = len(links)

        self._source_proposals = proposals
        self._source_tracks = {
            observation.proposal_id: observation.track_id
            for observation in observations
        }
        return TrackedMaskFrame(
            frame_id=frame_id,
            observations=observations,
            linked_count=linked_count,
            new_track_count=len(proposals) - linked_count,
        )
