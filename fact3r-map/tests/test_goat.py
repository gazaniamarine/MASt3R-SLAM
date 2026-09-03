from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fact3r.experiments.goat import (
    GoatGoal,
    aggregate_subtasks,
    load_scene,
    load_split,
    scene_names,
    score_subtask,
)

SCENE = "y9hTuugGdiq"


def _goal(category, object_id, position, desc=None, view_points=2, images=3):
    return {
        "object_category": category,
        "object_id": object_id,
        "position": list(position),
        "lang_desc": desc,
        "children_object_categories": [],
        # Real view points sit a short step away from the object, on the floor;
        # placing them at the origin instead would make every scoring test
        # accidentally measure distance to the origin.
        "view_points": [
            {"agent_state": {"position": [position[0] - 0.5 * (i + 1),
                                          position[1], position[2]]}}
            for i in range(view_points)
        ],
        "image_goals": [{} for _ in range(images)],
    }


def _write_scene(directory: Path, scene: str = SCENE, tasks=None) -> Path:
    content = directory / "val_unseen" / "content"
    content.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": [
            {
                "episode_id": "0",
                "scene_dataset_config": "hm3d.json",
                "scene_id": f"hm3d/val//00808-{scene}/{scene}.basis.glb",
                "start_position": [-0.21, 0.01, 2.74],
                "start_rotation": [0, 0, 0, 1],
                "tasks": tasks if tasks is not None else [
                    ["plant", "image", "plant_14", 20],
                    ["plant", "description", "plant_14"],
                    ["plant", "object", None],
                ],
            }
        ],
        "goals": {
            f"{scene}.basis.glb_plant": [
                _goal("plant", "plant_14", [1.0, 0.0, 0.0], "ficus tree right of the painting."),
                _goal("plant", "plant_99", [9.0, 0.0, 0.0], "small plant on the sill."),
            ]
        },
    }
    path = content / f"{scene}.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)
    return path


class LoadTest(unittest.TestCase):
    def test_parses_episode_and_subtasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_scene(Path(tmp))
            episodes = load_scene(path)
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(episode.scene, SCENE)
        self.assertEqual(episode.subtask_count, 3)
        np.testing.assert_allclose(episode.start_position, [-0.21, 0.01, 2.74])

    def test_scene_id_is_stripped_to_the_house(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        self.assertEqual(episodes[0].scene, SCENE)

    def test_instance_goals_resolve_to_one_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        description = episodes[0].subtasks[1]
        self.assertEqual(description.modality, "description")
        self.assertEqual(len(description.goals), 1)
        self.assertEqual(description.goals[0].object_id, "plant_14")
        self.assertTrue(description.is_instance_specific)

    def test_object_goals_accept_any_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        obj = episodes[0].subtasks[2]
        self.assertEqual(obj.modality, "object")
        self.assertIsNone(obj.instance_id)
        self.assertEqual(len(obj.goals), 2)
        self.assertFalse(obj.is_instance_specific)

    def test_image_index_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        self.assertEqual(episodes[0].subtasks[0].image_index, 20)

    def test_view_points_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        goal = episodes[0].subtasks[1].goals[0]
        self.assertEqual(goal.view_points.shape, (2, 3))
        self.assertEqual(goal.image_goal_count, 3)

    def test_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_scene("/nonexistent/scene.json.gz")

    def test_split_loads_every_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_scene(Path(tmp), scene="aaaaaaa")
            _write_scene(Path(tmp), scene="bbbbbbb")
            episodes = load_split(Path(tmp), "val_unseen")
            names = scene_names(Path(tmp), "val_unseen")
        self.assertEqual(len(episodes), 2)
        self.assertEqual(names, ["aaaaaaa", "bbbbbbb"])

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_split("/tmp", "holdout")


class PromptTest(unittest.TestCase):
    def test_description_goal_uses_the_instance_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        self.assertEqual(
            episodes[0].subtasks[1].prompt, "ficus tree right of the painting."
        )

    def test_object_goal_uses_the_bare_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        self.assertEqual(episodes[0].subtasks[2].prompt, "plant")

    def test_image_goal_has_no_text_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes = load_scene(_write_scene(Path(tmp)))
        self.assertIsNone(episodes[0].subtasks[0].prompt)


class ScoreTest(unittest.TestCase):
    def _subtasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            return load_scene(_write_scene(Path(tmp)))[0].subtasks

    def test_success_within_one_metre(self):
        # plant_14 sits at x=1.0 with view points at x=0.5 and x=0.0.
        result = score_subtask(self._subtasks()[1], [0.8, 0.0, 0.0])
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["distance_to_goal"], 0.3)

    def test_distance_is_measured_to_view_points_not_the_centroid(self):
        """A wall-mounted object is only ever reachable via its view points."""

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_scene(Path(tmp))
            subtask = load_scene(path)[0].subtasks[1]
        # Standing at the nearest view point is an arrival even though the
        # centroid is half a metre further on.
        result = score_subtask(subtask, [0.5, 0.0, 0.0])
        self.assertAlmostEqual(result["distance_to_goal"], 0.0)
        self.assertTrue(result["success"])

    def test_failure_beyond_one_metre(self):
        result = score_subtask(self._subtasks()[1], [4.0, 0.0, 0.0])
        self.assertFalse(result["success"])

    def test_object_goal_scores_against_the_nearest_instance(self):
        # Standing by plant_99 satisfies an object goal but not the instance one.
        obj, description = self._subtasks()[2], self._subtasks()[1]
        self.assertTrue(score_subtask(obj, [9.0, 0.0, 0.0])["success"])
        self.assertFalse(score_subtask(description, [9.2, 0.0, 0.0])["success"])

    def test_subtask_without_a_goal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_scene(Path(tmp), tasks=[["sofa", "description", "sofa_1"]])
            episode = load_scene(path)[0]
        with self.assertRaises(ValueError):
            score_subtask(episode.subtasks[0], [0.0, 0.0, 0.0])

    def test_aggregate_reports_per_modality(self):
        subtasks = self._subtasks()
        results = [
            score_subtask(subtasks[1], [1.2, 0.0, 0.0]),   # description, success
            score_subtask(subtasks[2], [5.0, 0.0, 0.0]),   # object, failure
        ]
        summary = aggregate_subtasks(results)
        self.assertEqual(summary["subtasks"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["success_rate_description"], 1.0)
        self.assertAlmostEqual(summary["success_rate_object"], 0.0)

    def test_aggregate_of_nothing(self):
        self.assertEqual(aggregate_subtasks([]), {"subtasks": 0})


if __name__ == "__main__":
    unittest.main()
