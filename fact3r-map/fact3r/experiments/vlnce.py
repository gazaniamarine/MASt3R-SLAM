"""R2R-CE episode handling for the go-there/come-back mapping protocol.

VLN-CE is an online instruction-following benchmark; Fact3R is a mapping and
retrieval system. This module supplies the piece that makes the two meet: it
chains short R2R-CE episodes inside one scene into a long outbound tour, then
names a return target that has both a natural-language reference (mined from an
earlier leg's instruction) and an exact ground-truth position (that leg's goal).
Chaining is what buys the long horizon: single R2R-CE episodes average only
8.9 m of geodesic distance, which is far too short to test persistent memory.

The habitat-sim renderer runs under the `habitat-vla` environment on Python
3.9, which cannot import this package, so tours cross that boundary as plain
JSON written by `scripts/build_vlnce_tours.py`.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SPLITS = ("train", "val_seen", "val_unseen", "test")

# The held-out challenge split ships instructions and a start pose only: no
# goals, no reference_path, no geodesic distance. A tour cannot be traversed or
# scored without those, so it is refused rather than half-parsed.
TOUR_SPLITS = ("train", "val_seen", "val_unseen")

# R2R-CE inherits the R2R success criterion: within 3 m of the goal.
SUCCESS_DISTANCE = 3.0


@dataclass(frozen=True)
class VLNCEEpisode:
    """One R2R-CE episode, as stored in `R2R_VLNCE_v1-3/<split>/<split>.json.gz`."""

    episode_id: int
    trajectory_id: int
    scene: str
    scene_id: str
    start_position: np.ndarray
    start_rotation: np.ndarray
    goal_position: np.ndarray
    goal_radius: float
    geodesic_distance: float
    instruction_text: str
    reference_path: np.ndarray

    @property
    def stop_landmark(self) -> Optional[str]:
        return extract_stop_landmark(self.instruction_text)


def load_split(dataset_root, split: str) -> List[VLNCEEpisode]:
    """Load one R2R-CE split.

    `dataset_root` is the directory holding the per-split folders, i.e.
    `datasets/vlnce/R2R_VLNCE_v1-3`.
    """

    if split not in SPLITS:
        raise ValueError("unknown split: %s" % split)
    if split not in TOUR_SPLITS:
        raise ValueError(
            "the %s split withholds goals and reference paths, so no tour can "
            "be built or scored from it; use one of %s"
            % (split, ", ".join(TOUR_SPLITS))
        )
    path = Path(dataset_root) / split / ("%s.json.gz" % split)
    if not path.is_file():
        raise FileNotFoundError("episode file not found: %s" % path)
    with gzip.open(str(path), "rt") as handle:
        payload = json.load(handle)
    return [_parse_episode(record) for record in payload["episodes"]]


def _parse_episode(record: dict) -> VLNCEEpisode:
    scene_id = record["scene_id"]
    # scene_id is "mp3d/<house>/<house>.glb"; the house id is the stable key.
    scene = Path(scene_id).parent.name
    goals = record.get("goals") or []
    if not goals:
        raise ValueError("episode %s has no goal" % record.get("episode_id"))
    instruction = record.get("instruction", {}) or {}
    return VLNCEEpisode(
        episode_id=int(record["episode_id"]),
        trajectory_id=int(record.get("trajectory_id", -1)),
        scene=scene,
        scene_id=scene_id,
        start_position=np.asarray(record["start_position"], dtype=np.float64),
        start_rotation=np.asarray(record["start_rotation"], dtype=np.float64),
        goal_position=np.asarray(goals[0]["position"], dtype=np.float64),
        goal_radius=float(goals[0].get("radius", SUCCESS_DISTANCE)),
        geodesic_distance=float(record.get("info", {}).get("geodesic_distance", float("nan"))),
        instruction_text=str(instruction.get("instruction_text", "")).strip(),
        reference_path=np.asarray(record["reference_path"], dtype=np.float64),
    )


def group_by_scene(episodes: Sequence[VLNCEEpisode]) -> Dict[str, List[VLNCEEpisode]]:
    grouped: Dict[str, List[VLNCEEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.scene, []).append(episode)
    for scene_episodes in grouped.values():
        scene_episodes.sort(key=lambda episode: episode.episode_id)
    return grouped


# --------------------------------------------------------------------------
# Stop-landmark mining
# --------------------------------------------------------------------------

# R2R instructions close with a stop clause naming the terminal landmark, e.g.
# "... and stop near the rug." After the precision filters below this
# yields a landmark for 46% of val_unseen and val_seen episodes alike.
_STOP_CLAUSE = re.compile(
    r"\b(?:stop|stopping|wait|waiting|halt)\b\s*"
    r"(?:there\s+|right\s+|just\s+)?"
    r"(?:near|by|at|beside|next\s+to|in\s+front\s+of|infront\s+of|before|"
    r"on|under|underneath|behind|inside|in)\b\s+"
    r"(?:the|a|an|your)?\s*"
    r"([a-z0-9][a-z0-9\s\-']{0,40}?)"
    r"\s*(?:\.|,|;|\band\b|\bthen\b|\bwhen\b|\bwhich\b|\bthat\b|$)",
    re.IGNORECASE,
)

# "the far end of the bar" names the bar, so the object sits after "of" here.
# This must run before the connective trim, which would otherwise cut it away.
_RELATIONAL_HEAD = re.compile(
    r"^(?:far\s+)?(?:end|edge|middle|centre|center|corner|top|bottom|side|foot|"
    r"head|front|back|base|entrance|entry|nearest|closest|farthest|furthest)"
    r"\s+of\s+(?:the\s+|a\s+|an\s+)?(.+)$",
    re.IGNORECASE,
)

# Everything from a connective onward describes context, not the object:
# "bottom step of the stairs" -> "bottom step", "6th seat from the left" -> "6th seat".
_CONNECTIVE = re.compile(
    r"\s+(?:of|from|with|on|in|at|to|into|for|near|beside|behind|under|across|"
    r"next|once|when|where|that|which|and|then|until|after|before)\s+.*$",
    re.IGNORECASE,
)

# Leading determiners and ordinals carry no visual meaning for a retriever.
_LEADING_FILLER = re.compile(
    r"^(?:the|a|an|your|first|second|third|fourth|fifth|sixth|last|next|"
    r"\d+(?:st|nd|rd|th)?)\s+",
    re.IGNORECASE,
)

# A stop clause that captured a verb phrase rather than an object. R2R has many
# of these ("stop when you reach the table"), and they make nonsense queries.
_NOT_AN_OBJECT = frozenset(
    {
        "you", "your", "we", "they", "it", "them", "there", "here", "this", "that",
        "reach", "reaching", "reached", "go", "going", "goes", "gone", "walk",
        "walking", "enter", "entering", "exit", "exiting", "turn", "turning",
        "look", "looking", "see", "seeing", "facing", "face", "stand", "standing",
        "wait", "waiting", "stop", "stopping", "continue", "head", "heading",
        "way", "top", "bottom", "side", "end", "middle", "centre", "center",
        "front", "back", "left", "right", "outside", "inside", "room", "area",
        "place", "spot", "point", "position", "one", "ones", "part",
    }
)

# "cabinet full of dolls" trims to "cabinet full"; the dangling adjective
# reads worse to a retriever than the bare noun.
_TRAILING_ADJECTIVE = re.compile(
    r"\s+(?:full|filled|covered|made|marked|labelled|labeled)$", re.IGNORECASE
)

_MAX_LANDMARK_WORDS = 3


def extract_stop_landmark(instruction_text: str) -> Optional[str]:
    """Return the object named in the instruction's stop clause, if any.

    This is a deliberately shallow rule over R2R phrasing, not a parser, and it
    is tuned for precision over recall: a landmark becomes a retrieval query, so
    a wrong one silently corrupts the experiment while a missing one only costs
    an episode. Clauses that resolve to a verb phrase, a pronoun, or a bare
    spatial word are rejected rather than guessed.
    """

    if not instruction_text:
        return None
    match = _STOP_CLAUSE.search(instruction_text)
    if match is None:
        return None
    landmark = " ".join(match.group(1).split()).strip(" -'").lower()
    if not landmark:
        return None
    relational = _RELATIONAL_HEAD.match(landmark)
    if relational is not None:
        landmark = relational.group(1).strip()
    landmark = _CONNECTIVE.sub("", landmark).strip()
    landmark = _TRAILING_ADJECTIVE.sub("", landmark).strip()
    previous = None
    while landmark != previous:
        previous = landmark
        landmark = _LEADING_FILLER.sub("", landmark).strip()
    if not landmark:
        return None

    words = landmark.split()
    if len(words) > _MAX_LANDMARK_WORDS:
        return None
    # The head noun decides whether this names a thing at all.
    if words[-1] in _NOT_AN_OBJECT:
        return None
    if any(word in _NOT_AN_OBJECT for word in words):
        return None
    if len(landmark) < 3:
        return None
    return landmark


# --------------------------------------------------------------------------
# Tour chaining
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TourLeg:
    """One outbound leg: a single R2R-CE episode traversed in order."""

    index: int
    episode: VLNCEEpisode

    @property
    def landmark(self) -> Optional[str]:
        return self.episode.stop_landmark


@dataclass
class ChainedTour:
    """An outbound tour plus the return target the agent is later asked for."""

    scene: str
    scene_id: str
    legs: List[TourLeg]
    return_leg_index: int
    return_query: str
    return_position: np.ndarray
    link_tolerance: float
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def outbound_path(self) -> np.ndarray:
        """Concatenated reference paths, with duplicated link points dropped."""

        points: List[np.ndarray] = []
        for leg in self.legs:
            for point in leg.episode.reference_path:
                if points and float(np.linalg.norm(point - points[-1])) < 1e-6:
                    continue
                points.append(np.asarray(point, dtype=np.float64))
        return np.asarray(points, dtype=np.float64)

    @property
    def outbound_length(self) -> float:
        path = self.outbound_path
        if len(path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    @property
    def start_position(self) -> np.ndarray:
        return self.legs[0].episode.start_position

    @property
    def final_position(self) -> np.ndarray:
        return self.legs[-1].episode.goal_position

    def to_json(self) -> dict:
        return {
            "scene": self.scene,
            "scene_id": self.scene_id,
            "legs": [
                {
                    "index": leg.index,
                    "episode_id": leg.episode.episode_id,
                    "trajectory_id": leg.episode.trajectory_id,
                    "instruction": leg.episode.instruction_text,
                    "landmark": leg.landmark,
                    "start_position": leg.episode.start_position.tolist(),
                    "goal_position": leg.episode.goal_position.tolist(),
                    "geodesic_distance": leg.episode.geodesic_distance,
                    # The renderer walks these waypoints; without them it would
                    # have to straight-line between leg endpoints.
                    "reference_path": leg.episode.reference_path.tolist(),
                }
                for leg in self.legs
            ],
            "return_leg_index": self.return_leg_index,
            "return_query": self.return_query,
            "return_position": self.return_position.tolist(),
            "outbound_length": self.outbound_length,
            "link_tolerance": self.link_tolerance,
            "metadata": dict(self.metadata),
        }


def chain_tour(
    episodes: Sequence[VLNCEEpisode],
    *,
    num_legs: int = 3,
    link_tolerance: float = 2.0,
    seed_episode_id: Optional[int] = None,
    require_landmark_on_first_leg: bool = True,
    distance_fn: Optional[Callable[[np.ndarray, np.ndarray], Optional[float]]] = None,
) -> Optional[ChainedTour]:
    """Greedily chain episodes in one scene into a single outbound tour.

    A leg may follow another when its start lies within `link_tolerance` of the
    previous goal. `distance_fn` should be the simulator's geodesic distance
    when a pathfinder is available; without it, straight-line distance is used,
    which can link across a wall and must be re-checked at render time.

    Returns None when no chain of the requested length exists.
    """

    if num_legs < 1:
        raise ValueError("num_legs must be positive")
    if link_tolerance < 0.0:
        raise ValueError("link_tolerance must be non-negative")
    if not episodes:
        return None
    scenes = {episode.scene for episode in episodes}
    if len(scenes) != 1:
        raise ValueError("chain_tour expects episodes from a single scene")

    measure = distance_fn or (lambda a, b: float(np.linalg.norm(a - b)))

    candidates = list(episodes)
    if seed_episode_id is not None:
        seeds = [e for e in candidates if e.episode_id == seed_episode_id]
        if not seeds:
            raise ValueError("seed episode %s is not in this scene" % seed_episode_id)
    else:
        seeds = candidates
    if require_landmark_on_first_leg:
        preferred = [e for e in seeds if e.stop_landmark]
        # Falling back keeps chaining usable when the caller only wants geometry.
        seeds = preferred or seeds
    # Longest first: the return leg should be far from where the tour ends.
    seeds = sorted(seeds, key=lambda e: -e.geodesic_distance)

    for seed in seeds:
        chain = _extend_chain(seed, candidates, num_legs, link_tolerance, measure)
        if chain is None:
            continue
        target = _select_return_target(chain)
        if target is None:
            continue
        leg_index, query = target
        legs = [TourLeg(index=i, episode=e) for i, e in enumerate(chain)]
        return ChainedTour(
            scene=seed.scene,
            scene_id=seed.scene_id,
            legs=legs,
            return_leg_index=leg_index,
            return_query=query,
            return_position=chain[leg_index].goal_position.copy(),
            link_tolerance=link_tolerance,
            metadata={"seed_episode_id": seed.episode_id, "geodesic_links": distance_fn is not None},
        )
    return None


def _extend_chain(
    seed: VLNCEEpisode,
    pool: Sequence[VLNCEEpisode],
    num_legs: int,
    link_tolerance: float,
    measure: Callable[[np.ndarray, np.ndarray], Optional[float]],
) -> Optional[List[VLNCEEpisode]]:
    chain = [seed]
    used_trajectories = {seed.trajectory_id}
    while len(chain) < num_legs:
        tail = chain[-1]
        best = None
        best_distance = None
        for candidate in pool:
            if candidate.trajectory_id in used_trajectories:
                continue
            distance = measure(tail.goal_position, candidate.start_position)
            if distance is None or not np.isfinite(distance) or distance > link_tolerance:
                continue
            # Prefer the longest onward leg, breaking ties by tighter links.
            key = (-candidate.geodesic_distance, distance)
            if best is None or key < best_distance:
                best, best_distance = candidate, key
        if best is None:
            return None
        chain.append(best)
        used_trajectories.add(best.trajectory_id)
    return chain


def _select_return_target(chain: Sequence[VLNCEEpisode]) -> Optional[Tuple[int, str]]:
    """Pick the earliest leg whose instruction names a landmark.

    Earliest maximises the horizon between seeing the object and being asked to
    return to it, which is the property the protocol is meant to stress.
    """

    for index, episode in enumerate(chain[:-1]):
        landmark = episode.stop_landmark
        if landmark:
            return index, landmark
    return None


def build_tours(
    episodes: Sequence[VLNCEEpisode],
    *,
    num_legs: int = 3,
    link_tolerance: float = 2.0,
    max_per_scene: int = 1,
    distance_fn_factory: Optional[Callable[[str], Callable[[np.ndarray, np.ndarray], Optional[float]]]] = None,
) -> List[ChainedTour]:
    """Build up to `max_per_scene` chained tours for every scene in `episodes`."""

    tours: List[ChainedTour] = []
    for scene, scene_episodes in sorted(group_by_scene(episodes).items()):
        distance_fn = distance_fn_factory(scene) if distance_fn_factory else None
        remaining = list(scene_episodes)
        for _ in range(max_per_scene):
            tour = chain_tour(
                remaining,
                num_legs=num_legs,
                link_tolerance=link_tolerance,
                distance_fn=distance_fn,
            )
            if tour is None:
                break
            tours.append(tour)
            consumed = {leg.episode.trajectory_id for leg in tour.legs}
            remaining = [e for e in remaining if e.trajectory_id not in consumed]
    return tours


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnMetrics:
    """Standard VLN-CE metrics applied to the return leg.

    These describe return navigation from a self-built map, which is a
    different task from the VLN-CE benchmark. They are not comparable to
    leaderboard numbers and must not be reported as such.
    """

    navigation_error: float
    success: bool
    oracle_success: bool
    path_length: float
    optimal_length: float
    spl: float

    def to_json(self) -> dict:
        return {
            "navigation_error": self.navigation_error,
            "success": bool(self.success),
            "oracle_success": bool(self.oracle_success),
            "path_length": self.path_length,
            "optimal_length": self.optimal_length,
            "spl": self.spl,
        }


def path_length(path: Sequence[Sequence[float]]) -> float:
    points = np.asarray(path, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def score_return(
    executed_path: Sequence[Sequence[float]],
    target_position: Sequence[float],
    optimal_length: float,
    *,
    success_distance: float = SUCCESS_DISTANCE,
    distance_fn: Optional[Callable[[np.ndarray, np.ndarray], Optional[float]]] = None,
) -> ReturnMetrics:
    """Score one return leg.

    `optimal_length` is the geodesic distance from the tour's end pose to the
    target, taken from the simulator pathfinder rather than from the agent's
    own map, so a distorted map cannot inflate SPL.
    """

    points = np.asarray(executed_path, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("executed_path must contain at least one position")
    target = np.asarray(target_position, dtype=np.float64)
    measure = distance_fn or (lambda a, b: float(np.linalg.norm(a - b)))

    def to_target(point: np.ndarray) -> float:
        distance = measure(point, target)
        if distance is None or not np.isfinite(distance):
            # An unreachable final pose is a failure, not a missing value.
            return float(np.linalg.norm(point - target))
        return float(distance)

    final_error = to_target(points[-1])
    closest_error = min(to_target(point) for point in points)
    travelled = path_length(points)
    success = final_error < success_distance
    optimal = float(max(optimal_length, 0.0))
    if success and travelled > 0.0:
        spl = optimal / max(optimal, travelled)
    else:
        spl = 0.0
    return ReturnMetrics(
        navigation_error=final_error,
        success=success,
        oracle_success=closest_error < success_distance,
        path_length=travelled,
        optimal_length=optimal,
        spl=spl,
    )


def aggregate(metrics: Sequence[ReturnMetrics]) -> dict:
    """Mean of each metric over episodes, matching VLN-CE reporting."""

    if not metrics:
        return {"episodes": 0}
    return {
        "episodes": len(metrics),
        "navigation_error": float(np.mean([m.navigation_error for m in metrics])),
        "success_rate": float(np.mean([float(m.success) for m in metrics])),
        "oracle_success_rate": float(np.mean([float(m.oracle_success) for m in metrics])),
        "spl": float(np.mean([m.spl for m in metrics])),
        "path_length": float(np.mean([m.path_length for m in metrics])),
    }
