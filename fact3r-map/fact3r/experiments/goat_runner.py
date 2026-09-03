"""Sequencing for a GOAT episode: many goals, one map, one continuous agent.

GOAT hands the agent 5-10 goals one after another and never teleports it: the
pose where subtask k ends is where subtask k+1 begins. That carry-over is the
whole point -- it is what makes the map worth keeping -- and it is easy to get
wrong by restarting each subtask from the episode start, which would quietly
turn a lifelong task into a series of independent ones.

Everything here is pure. The stages that need a simulator or a GPU live in
`scripts/run_goat_eval.py`; this decides what to run, in what order, from what
pose, and what the results mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from fact3r.experiments.goat import GoatEpisode, GoatSubtask, SUCCESS_DISTANCE


@dataclass
class SubtaskRequest:
    """Everything the per-subtask pipeline needs to run one goal."""

    episode_id: str
    scene: str
    index: int
    modality: str
    category: str
    instance_id: Optional[str]
    start_position: np.ndarray
    prompt: Optional[str] = None
    image_index: Optional[int] = None
    goal_positions: List[np.ndarray] = field(default_factory=list)
    # One array of view points per acceptable goal, in the same order. GOAT
    # scores against these, not against the centroids in `goal_positions`.
    goal_view_points: List[np.ndarray] = field(default_factory=list)

    @property
    def is_runnable(self) -> bool:
        """Image goals need a rendered goal image; text goals need a prompt."""

        if self.modality == "image":
            return self.image_index is not None
        return bool(self.prompt)

    def image_name(self, scene: str) -> Optional[str]:
        if self.modality != "image" or self.image_index is None:
            return None
        return "%s_%s_%02d.jpg" % (scene, self.instance_id, self.image_index)


def plan_episode(
    episode: GoatEpisode,
    *,
    start_position: Optional[Sequence[float]] = None,
) -> List[SubtaskRequest]:
    """Turn an episode into an ordered list of runnable subtask requests.

    The start pose of each subtask is left as the episode start here; the
    runner overwrites it with where the previous subtask actually stopped,
    because that is only known once the previous one has run.
    """

    origin = np.asarray(
        episode.start_position if start_position is None else start_position,
        dtype=np.float64,
    )
    requests = []
    for subtask in episode.subtasks:
        requests.append(
            SubtaskRequest(
                episode_id=episode.episode_id,
                scene=episode.scene,
                index=subtask.index,
                modality=subtask.modality,
                category=subtask.category,
                instance_id=subtask.instance_id,
                start_position=origin.copy(),
                prompt=subtask.prompt,
                image_index=subtask.image_index,
                goal_positions=[goal.position.copy() for goal in subtask.goals],
                goal_view_points=[
                    np.array(goal.view_points, dtype=np.float64, copy=True)
                    for goal in subtask.goals
                ],
            )
        )
    return requests


def carry_forward(
    requests: Sequence[SubtaskRequest], index: int, final_position: Sequence[float]
) -> None:
    """Start the next subtask where this one stopped, as GOAT does."""

    if index + 1 < len(requests):
        requests[index + 1].start_position = np.asarray(final_position, dtype=np.float64)


def score_request(
    request: SubtaskRequest,
    final_position: Sequence[float],
    *,
    success_distance: float = SUCCESS_DISTANCE,
) -> dict:
    """Distance to the nearest acceptable goal, and whether that is a success.

    Measured to the goal's view points, which is what GOAT counts as arriving.
    Scoring to the object centroid instead makes wall-mounted and shelved
    objects unreachable by definition -- see `GoatGoal.distance_from`.
    """

    if not request.goal_positions:
        raise ValueError("subtask %d has no goal position" % request.index)
    here = np.asarray(final_position, dtype=np.float64)
    distances = []
    for index, centroid in enumerate(request.goal_positions):
        views = (
            request.goal_view_points[index]
            if index < len(request.goal_view_points)
            else np.zeros((0, 3))
        )
        if len(views):
            distances.append(float(np.min(np.linalg.norm(views - here, axis=1))))
        else:
            distances.append(float(np.linalg.norm(centroid - here)))
    error = min(distances)
    return {
        "episode_id": request.episode_id,
        "index": request.index,
        "modality": request.modality,
        "category": request.category,
        "instance_id": request.instance_id,
        "distance_to_goal": error,
        "success": bool(error < success_distance),
    }


def skipped_result(request: SubtaskRequest, reason: str) -> dict:
    """A subtask that could not be attempted, recorded rather than dropped.

    Silently omitting these would inflate the success rate over whatever
    remained, which is the easiest way to publish a wrong number.
    """

    return {
        "episode_id": request.episode_id,
        "index": request.index,
        "modality": request.modality,
        "category": request.category,
        "instance_id": request.instance_id,
        "distance_to_goal": float("nan"),
        "success": False,
        "skipped": True,
        "reason": reason,
    }


def summarise(results: Sequence[dict]) -> Dict[str, object]:
    """Success by modality and overall, counting skips as failures."""

    if not results:
        return {"subtasks": 0}
    attempted = [r for r in results if not r.get("skipped")]
    summary: Dict[str, object] = {
        "subtasks": len(results),
        "attempted": len(attempted),
        "skipped": len(results) - len(attempted),
        "success_rate": float(np.mean([float(r["success"]) for r in results])),
    }
    if attempted:
        summary["success_rate_attempted"] = float(
            np.mean([float(r["success"]) for r in attempted])
        )
        summary["mean_distance_to_goal"] = float(
            np.mean([r["distance_to_goal"] for r in attempted])
        )
    for modality in ("object", "description", "image"):
        subset = [r for r in results if r["modality"] == modality]
        if subset:
            summary["success_rate_%s" % modality] = float(
                np.mean([float(r["success"]) for r in subset])
            )
            summary["count_%s" % modality] = len(subset)
    return summary
