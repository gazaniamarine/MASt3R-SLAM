"""GOAT-Bench episode handling.

GOAT fits Fact3R better than VLN-CE does. Each episode issues 5-10 goals in
sequence in one house, so the map is built once and reused -- which is the
persistent-memory claim, tested natively instead of through the chained
go-there/come-back protocol we had to invent for VLN-CE. Goals arrive as an
object category, an instance-specific language description, or an image, and
the descriptions name a particular instance ("ficus tree located to the right
of the painting"), which is exactly what `doorway` failed to do.

Two goal kinds behave differently and must not be conflated:

* `object` goals name a category with no instance, so **any** instance of that
  category satisfies them, as in ObjectNav;
* `description` and `image` goals name one instance, and only that instance
  counts.

Scenes are HM3D. Episode files live per scene under
`goat_bench/hm3d/v1/<split>/content/<scene>.json.gz`.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SPLITS = ("train", "val_seen", "val_unseen", "val_seen_synonyms")

# GOAT counts a subtask successful when the agent stops within 1 m of the goal.
SUCCESS_DISTANCE = 1.0

MODALITIES = ("object", "description", "image")


@dataclass(frozen=True)
class GoatGoal:
    """One goal object instance in a scene."""

    object_category: str
    object_id: str
    position: np.ndarray
    lang_desc: Optional[str]
    view_points: np.ndarray
    image_goal_count: int

    def distance_from(self, position: Sequence[float]) -> float:
        """Distance to the nearest place the benchmark counts as "at" this goal.

        GOAT scores against the goal's *view points* -- navigable poses from
        which the object is visible -- not against the object's centroid. The
        difference is not cosmetic: a picture's centroid is on the wall, often
        metres above the floor, so a robot standing correctly in front of it is
        never within a metre of the centroid and would score zero forever.
        Goals without view points fall back to the centroid.
        """

        here = np.asarray(position, dtype=np.float64)
        if len(self.view_points):
            return float(np.min(np.linalg.norm(self.view_points - here, axis=1)))
        return float(np.linalg.norm(self.position - here))

    @classmethod
    def from_json(cls, record: dict) -> "GoatGoal":
        points = [
            np.asarray(item["agent_state"]["position"], dtype=np.float64)
            if isinstance(item, dict) and "agent_state" in item
            else np.asarray(item, dtype=np.float64)
            for item in record.get("view_points", [])
        ]
        return cls(
            object_category=str(record["object_category"]),
            object_id=str(record["object_id"]),
            position=np.asarray(record["position"], dtype=np.float64),
            lang_desc=(record.get("lang_desc") or None),
            view_points=np.asarray(points, dtype=np.float64) if points else np.zeros((0, 3)),
            image_goal_count=len(record.get("image_goals", []) or []),
        )


@dataclass(frozen=True)
class GoatSubtask:
    """One goal within an episode, in the order the agent receives it."""

    index: int
    category: str
    modality: str
    instance_id: Optional[str]
    image_index: Optional[int]
    goals: Tuple[GoatGoal, ...]

    @property
    def is_instance_specific(self) -> bool:
        return self.modality in ("description", "image")

    @property
    def prompt(self) -> Optional[str]:
        """The text a retriever should be given, when there is one.

        `object` goals give only the category; `description` goals give the
        instance sentence, which is the one worth querying with.
        """

        if self.modality == "description":
            for goal in self.goals:
                if goal.lang_desc:
                    return goal.lang_desc
            return None
        if self.modality == "object":
            return self.category
        return None

    def distance_to_nearest_goal(self, position: Sequence[float]) -> float:
        here = np.asarray(position, dtype=np.float64)
        if not self.goals:
            raise ValueError("subtask has no resolved goal")
        return min(goal.distance_from(here) for goal in self.goals)


@dataclass(frozen=True)
class GoatEpisode:
    episode_id: str
    scene: str
    scene_id: str
    start_position: np.ndarray
    start_rotation: np.ndarray
    subtasks: Tuple[GoatSubtask, ...]

    @property
    def subtask_count(self) -> int:
        return len(self.subtasks)


def _goal_key(scene: str, category: str) -> str:
    return f"{scene}.basis.glb_{category}"


def _scene_from_id(scene_id: str) -> str:
    """`hm3d/val//00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb` -> `y9hTuugGdiq`."""

    name = Path(scene_id).name
    for suffix in (".basis.glb", ".glb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_scene(path) -> List[GoatEpisode]:
    """Load every episode for one scene file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"episode file not found: {path}")
    with gzip.open(str(path), "rt") as handle:
        payload = json.load(handle)

    goals_by_key: Dict[str, List[GoatGoal]] = {}
    for key, records in payload.get("goals", {}).items():
        goals_by_key[key] = [GoatGoal.from_json(record) for record in records]

    episodes = []
    for record in payload["episodes"]:
        scene = _scene_from_id(record["scene_id"])
        subtasks = []
        for index, task in enumerate(record.get("tasks", [])):
            category = str(task[0])
            modality = str(task[1])
            instance_id = task[2] if len(task) > 2 else None
            image_index = int(task[3]) if len(task) > 3 and task[3] is not None else None
            pool = goals_by_key.get(_goal_key(scene, category), [])
            if instance_id:
                resolved = tuple(g for g in pool if g.object_id == str(instance_id))
            else:
                # An object goal is satisfied by any instance of the category.
                resolved = tuple(pool)
            subtasks.append(
                GoatSubtask(
                    index=index,
                    category=category,
                    modality=modality,
                    instance_id=str(instance_id) if instance_id else None,
                    image_index=image_index,
                    goals=resolved,
                )
            )
        episodes.append(
            GoatEpisode(
                episode_id=str(record["episode_id"]),
                scene=scene,
                scene_id=str(record["scene_id"]),
                start_position=np.asarray(record["start_position"], dtype=np.float64),
                start_rotation=np.asarray(record["start_rotation"], dtype=np.float64),
                subtasks=tuple(subtasks),
            )
        )
    return episodes


def load_split(dataset_root, split: str) -> List[GoatEpisode]:
    """Load every scene in a split.

    `dataset_root` is the directory holding the split folders, i.e.
    `datasets/goat/data/datasets/goat_bench/hm3d/v1`.
    """

    if split not in SPLITS:
        raise ValueError("unknown split: %s" % split)
    content = Path(dataset_root) / split / "content"
    if not content.is_dir():
        raise FileNotFoundError(f"no per-scene content under {content}")
    episodes: List[GoatEpisode] = []
    for path in sorted(content.glob("*.json.gz")):
        episodes.extend(load_scene(path))
    return episodes


def scene_names(dataset_root, split: str) -> List[str]:
    """Scene ids a split needs, without parsing every episode."""

    content = Path(dataset_root) / split / "content"
    return sorted(path.name.replace(".json.gz", "") for path in content.glob("*.json.gz"))


def score_subtask(
    subtask: GoatSubtask,
    final_position: Sequence[float],
    *,
    success_distance: float = SUCCESS_DISTANCE,
) -> dict:
    """Score one subtask by GOAT's criterion: stop within 1 m of the goal."""

    error = subtask.distance_to_nearest_goal(final_position)
    return {
        "index": subtask.index,
        "modality": subtask.modality,
        "category": subtask.category,
        "instance_id": subtask.instance_id,
        "distance_to_goal": error,
        "success": bool(error < success_distance),
    }


def aggregate_subtasks(results: Sequence[dict]) -> dict:
    """Per-modality and overall success, which is how GOAT reports."""

    if not results:
        return {"subtasks": 0}
    summary = {
        "subtasks": len(results),
        "success_rate": float(np.mean([float(r["success"]) for r in results])),
        "mean_distance_to_goal": float(np.mean([r["distance_to_goal"] for r in results])),
    }
    for modality in MODALITIES:
        subset = [r for r in results if r["modality"] == modality]
        if subset:
            summary[f"success_rate_{modality}"] = float(
                np.mean([float(r["success"]) for r in subset])
            )
    return summary
