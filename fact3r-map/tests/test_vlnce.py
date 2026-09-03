from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fact3r.experiments.vlnce import (
    ChainedTour,
    build_tours,
    chain_tour,
    extract_stop_landmark,
    group_by_scene,
    load_split,
    score_return,
)
from fact3r.experiments.vlnce import aggregate, path_length


def _episode(
    episode_id: int,
    start,
    goal,
    *,
    scene: str = "zsNo4HB9uLZ",
    instruction: str = "Walk forward and stop near the rug.",
    trajectory_id: int | None = None,
    geodesic: float = 8.0,
) -> dict:
    return {
        "episode_id": episode_id,
        "trajectory_id": episode_id if trajectory_id is None else trajectory_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": list(start),
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "info": {"geodesic_distance": geodesic},
        "goals": [{"position": list(goal), "radius": 3.0}],
        "instruction": {"instruction_text": instruction, "instruction_tokens": []},
        "reference_path": [list(start), list(goal)],
    }


def _write_split(directory: Path, split: str, records) -> Path:
    split_dir = directory / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{split}.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"episodes": list(records), "instruction_vocab": {}}, handle)
    return path


class StopLandmarkTest(unittest.TestCase):
    def test_extracts_simple_stop_clause(self):
        self.assertEqual(
            extract_stop_landmark("Exit the bedroom and stop near the rug."), "rug"
        )

    def test_extracts_multiword_landmark(self):
        self.assertEqual(
            extract_stop_landmark("Walk out and wait right by the coffee table."),
            "coffee table",
        )

    def test_reduces_relational_phrase_to_head_noun(self):
        self.assertEqual(
            extract_stop_landmark("Walk across the room and wait at the far end of the bar."),
            "bar",
        )

    def test_trims_trailing_prepositional_clause(self):
        self.assertEqual(
            extract_stop_landmark("Stop near the dining table with pictures on the wall."),
            "dining table",
        )

    def test_resolves_superlative_relational_phrase(self):
        self.assertEqual(
            extract_stop_landmark("Stop next to the nearest of the two beds."),
            "two beds",
        )

    def test_drops_dangling_adjective(self):
        self.assertEqual(
            extract_stop_landmark("Stop in front of the cabinet full of dolls."),
            "cabinet",
        )

    def test_rejects_verb_phrase_clause(self):
        self.assertIsNone(
            extract_stop_landmark("Walk on and stop in front of you reach the table.")
        )

    def test_rejects_overlong_phrase(self):
        self.assertIsNone(
            extract_stop_landmark("Stop by the big old wooden dining table thing.")
        )

    def test_returns_none_without_a_stop_clause(self):
        self.assertIsNone(extract_stop_landmark("Turn left and walk down the hallway."))

    def test_rejects_contentless_landmark(self):
        self.assertIsNone(extract_stop_landmark("Walk ahead and stop in the room."))

    def test_handles_empty_instruction(self):
        self.assertIsNone(extract_stop_landmark(""))


class LoadSplitTest(unittest.TestCase):
    def test_parses_episode_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(root, "val_unseen", [_episode(1, [0, 0, 0], [5, 0, 0])])
            episodes = load_split(root, "val_unseen")
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(episode.scene, "zsNo4HB9uLZ")
        self.assertEqual(episode.episode_id, 1)
        self.assertEqual(episode.stop_landmark, "rug")
        np.testing.assert_allclose(episode.goal_position, [5, 0, 0])

    def test_rejects_unknown_split(self):
        with self.assertRaises(ValueError):
            load_split("/nonexistent", "holdout")

    def test_refuses_the_held_out_test_split(self):
        # It ships no goals and no reference_path, so a tour cannot be scored.
        with self.assertRaises(ValueError) as caught:
            load_split("/nonexistent", "test")
        self.assertIn("withholds goals", str(caught.exception))

    def test_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_split(tmp, "val_unseen")

    def test_groups_by_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(
                root,
                "val_seen",
                [
                    _episode(1, [0, 0, 0], [5, 0, 0], scene="aaa"),
                    _episode(2, [0, 0, 0], [5, 0, 0], scene="bbb"),
                ],
            )
            grouped = group_by_scene(load_split(root, "val_seen"))
        self.assertEqual(sorted(grouped), ["aaa", "bbb"])


class ChainTourTest(unittest.TestCase):
    def _episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(
                root,
                "val_unseen",
                [
                    _episode(1, [0, 0, 0], [5, 0, 0], instruction="Stop near the rug."),
                    _episode(2, [5, 0, 0], [10, 0, 0], instruction="Stop by the sink."),
                    _episode(3, [10, 0, 0], [15, 0, 0], instruction="Stop at the window."),
                    _episode(4, [90, 0, 90], [95, 0, 90], instruction="Stop by the door."),
                ],
            )
            return load_split(root, "val_unseen")

    def test_chains_three_legs(self):
        tour = chain_tour(self._episodes(), num_legs=3, link_tolerance=0.5)
        self.assertIsNotNone(tour)
        self.assertEqual(len(tour.legs), 3)
        self.assertEqual([leg.episode.episode_id for leg in tour.legs], [1, 2, 3])

    def test_return_target_is_first_landmark_leg(self):
        tour = chain_tour(self._episodes(), num_legs=3, link_tolerance=0.5)
        self.assertEqual(tour.return_leg_index, 0)
        self.assertEqual(tour.return_query, "rug")
        np.testing.assert_allclose(tour.return_position, [5, 0, 0])

    def test_return_target_is_never_the_final_leg(self):
        # Returning to where the tour already ended would be a no-op.
        tour = chain_tour(self._episodes(), num_legs=3, link_tolerance=0.5)
        self.assertLess(tour.return_leg_index, len(tour.legs) - 1)

    def test_disconnected_episode_is_not_chained(self):
        tour = chain_tour(self._episodes(), num_legs=4, link_tolerance=0.5)
        self.assertIsNone(tour)

    def test_respects_link_tolerance(self):
        episodes = self._episodes()
        self.assertIsNone(chain_tour(episodes, num_legs=2, link_tolerance=0.0001,
                                     seed_episode_id=4))

    def test_geodesic_distance_fn_can_reject_a_link(self):
        # A wall between two nearby points: euclidean links, geodesic does not.
        blocked = lambda a, b: None  # noqa: E731
        self.assertIsNone(
            chain_tour(self._episodes(), num_legs=2, link_tolerance=5.0,
                       distance_fn=blocked)
        )

    def test_outbound_path_drops_duplicate_link_points(self):
        tour = chain_tour(self._episodes(), num_legs=3, link_tolerance=0.5)
        path = tour.outbound_path
        self.assertEqual(len(path), 4)
        self.assertAlmostEqual(tour.outbound_length, 15.0)

    def test_single_scene_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(
                root,
                "val_unseen",
                [
                    _episode(1, [0, 0, 0], [5, 0, 0], scene="aaa"),
                    _episode(2, [5, 0, 0], [10, 0, 0], scene="bbb"),
                ],
            )
            episodes = load_split(root, "val_unseen")
        with self.assertRaises(ValueError):
            chain_tour(episodes, num_legs=2)

    def test_tour_serializes_to_json(self):
        tour = chain_tour(self._episodes(), num_legs=3, link_tolerance=0.5)
        payload = json.loads(json.dumps(tour.to_json()))
        self.assertEqual(payload["return_query"], "rug")
        self.assertEqual(len(payload["legs"]), 3)

    def test_build_tours_covers_each_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split(
                root,
                "val_unseen",
                [
                    _episode(1, [0, 0, 0], [5, 0, 0], scene="aaa"),
                    _episode(2, [5, 0, 0], [10, 0, 0], scene="aaa"),
                    _episode(3, [0, 0, 0], [5, 0, 0], scene="bbb"),
                    _episode(4, [5, 0, 0], [10, 0, 0], scene="bbb"),
                ],
            )
            tours = build_tours(load_split(root, "val_unseen"), num_legs=2,
                                link_tolerance=0.5)
        self.assertEqual([tour.scene for tour in tours], ["aaa", "bbb"])


class ReturnMetricsTest(unittest.TestCase):
    def test_success_within_three_metres(self):
        metrics = score_return([[0, 0, 0], [4, 0, 0]], [5, 0, 0], optimal_length=5.0)
        self.assertTrue(metrics.success)
        self.assertAlmostEqual(metrics.navigation_error, 1.0)

    def test_failure_beyond_three_metres(self):
        metrics = score_return([[0, 0, 0], [1, 0, 0]], [10, 0, 0], optimal_length=10.0)
        self.assertFalse(metrics.success)
        self.assertEqual(metrics.spl, 0.0)

    def test_spl_is_one_for_an_optimal_path(self):
        metrics = score_return([[0, 0, 0], [5, 0, 0]], [5, 0, 0], optimal_length=5.0)
        self.assertAlmostEqual(metrics.spl, 1.0)

    def test_spl_penalises_a_detour(self):
        metrics = score_return(
            [[0, 0, 0], [0, 0, 5], [5, 0, 5], [5, 0, 0]], [5, 0, 0], optimal_length=5.0
        )
        self.assertTrue(metrics.success)
        self.assertAlmostEqual(metrics.spl, 5.0 / 15.0)

    def test_oracle_success_when_passing_the_target(self):
        # Walks over the goal, then overshoots well past it.
        metrics = score_return([[0, 0, 0], [5, 0, 0], [20, 0, 0]], [5, 0, 0],
                               optimal_length=5.0)
        self.assertFalse(metrics.success)
        self.assertTrue(metrics.oracle_success)

    def test_rejects_empty_path(self):
        with self.assertRaises(ValueError):
            score_return([], [0, 0, 0], optimal_length=1.0)

    def test_unreachable_geodesic_falls_back_to_euclidean(self):
        metrics = score_return([[0, 0, 0]], [1, 0, 0], optimal_length=1.0,
                               distance_fn=lambda a, b: None)
        self.assertAlmostEqual(metrics.navigation_error, 1.0)

    def test_path_length_of_single_point_is_zero(self):
        self.assertEqual(path_length([[1, 2, 3]]), 0.0)

    def test_aggregate_averages_episodes(self):
        good = score_return([[0, 0, 0], [5, 0, 0]], [5, 0, 0], optimal_length=5.0)
        bad = score_return([[0, 0, 0]], [50, 0, 0], optimal_length=50.0)
        summary = aggregate([good, bad])
        self.assertEqual(summary["episodes"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["spl"], 0.5)

    def test_aggregate_of_nothing(self):
        self.assertEqual(aggregate([]), {"episodes": 0})


if __name__ == "__main__":
    unittest.main()
