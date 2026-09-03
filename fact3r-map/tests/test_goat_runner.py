from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fact3r.experiments.goat import load_scene
from fact3r.experiments.goat_runner import (
    carry_forward,
    plan_episode,
    score_request,
    skipped_result,
    summarise,
)

SCENE = "y9hTuugGdiq"


def _scene_file(directory: Path) -> Path:
    content = directory / "val_unseen" / "content"
    content.mkdir(parents=True, exist_ok=True)
    goal = lambda oid, pos, desc: {  # noqa: E731
        "object_category": "plant", "object_id": oid, "position": list(pos),
        "lang_desc": desc, "children_object_categories": [],
        "view_points": [], "image_goals": [{}, {}],
    }
    payload = {
        "episodes": [{
            "episode_id": "0",
            "scene_id": f"hm3d/val//00808-{SCENE}/{SCENE}.basis.glb",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0, 0, 0, 1],
            "tasks": [
                ["plant", "description", "plant_14"],
                ["plant", "image", "plant_14", 1],
                ["plant", "object", None],
            ],
        }],
        "goals": {
            f"{SCENE}.basis.glb_plant": [
                goal("plant_14", [3.0, 0.0, 0.0], "ficus by the painting."),
                goal("plant_99", [20.0, 0.0, 0.0], "small plant on the sill."),
            ]
        },
    }
    path = content / f"{SCENE}.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)
    return path


def _episode():
    with tempfile.TemporaryDirectory() as tmp:
        return load_scene(_scene_file(Path(tmp)))[0]


class PlanTest(unittest.TestCase):
    def test_one_request_per_subtask_in_order(self):
        requests = plan_episode(_episode())
        self.assertEqual([r.index for r in requests], [0, 1, 2])
        self.assertEqual([r.modality for r in requests],
                         ["description", "image", "object"])

    def test_first_subtask_starts_at_the_episode_start(self):
        requests = plan_episode(_episode())
        np.testing.assert_allclose(requests[0].start_position, [0.0, 0.0, 0.0])

    def test_explicit_start_overrides(self):
        requests = plan_episode(_episode(), start_position=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(requests[0].start_position, [1.0, 2.0, 3.0])

    def test_text_prompts_are_carried(self):
        requests = plan_episode(_episode())
        self.assertEqual(requests[0].prompt, "ficus by the painting.")
        self.assertEqual(requests[2].prompt, "plant")
        self.assertIsNone(requests[1].prompt)

    def test_runnable_flags(self):
        requests = plan_episode(_episode())
        self.assertTrue(requests[0].is_runnable)     # description with a prompt
        self.assertTrue(requests[1].is_runnable)     # image with an index
        self.assertTrue(requests[2].is_runnable)     # object category

    def test_image_goal_filename(self):
        requests = plan_episode(_episode())
        self.assertEqual(requests[1].image_name(SCENE), f"{SCENE}_plant_14_01.jpg")
        self.assertIsNone(requests[0].image_name(SCENE))


class CarryForwardTest(unittest.TestCase):
    def test_next_subtask_starts_where_this_one_stopped(self):
        requests = plan_episode(_episode())
        carry_forward(requests, 0, [5.0, 0.0, 2.0])
        np.testing.assert_allclose(requests[1].start_position, [5.0, 0.0, 2.0])
        # The one after that is untouched until its turn.
        np.testing.assert_allclose(requests[2].start_position, [0.0, 0.0, 0.0])

    def test_carrying_past_the_last_subtask_is_a_no_op(self):
        requests = plan_episode(_episode())
        carry_forward(requests, len(requests) - 1, [9.0, 9.0, 9.0])
        self.assertEqual(len(requests), 3)

    def test_the_agent_is_never_teleported_back(self):
        requests = plan_episode(_episode())
        for index, position in enumerate([[1, 0, 0], [2, 0, 0]]):
            carry_forward(requests, index, position)
        starts = [tuple(r.start_position) for r in requests]
        self.assertEqual(starts, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])


class ScoreTest(unittest.TestCase):
    def test_success_within_one_metre(self):
        requests = plan_episode(_episode())
        result = score_request(requests[0], [3.5, 0.0, 0.0])
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["distance_to_goal"], 0.5)

    def test_instance_goal_ignores_the_other_instance(self):
        requests = plan_episode(_episode())
        # Standing at plant_99 fails a plant_14 description goal.
        self.assertFalse(score_request(requests[0], [20.0, 0.0, 0.0])["success"])

    def test_object_goal_accepts_either_instance(self):
        requests = plan_episode(_episode())
        self.assertTrue(score_request(requests[2], [20.2, 0.0, 0.0])["success"])
        self.assertTrue(score_request(requests[2], [3.2, 0.0, 0.0])["success"])

    def test_missing_goal_is_rejected(self):
        requests = plan_episode(_episode())
        requests[0].goal_positions = []
        with self.assertRaises(ValueError):
            score_request(requests[0], [0.0, 0.0, 0.0])


class SummariseTest(unittest.TestCase):
    def _results(self):
        requests = plan_episode(_episode())
        return [
            score_request(requests[0], [3.2, 0.0, 0.0]),     # description, success
            skipped_result(requests[1], "no goal image rendered"),
            score_request(requests[2], [9.0, 0.0, 0.0]),     # object, failure
        ]

    def test_skips_count_as_failures_not_omissions(self):
        summary = summarise(self._results())
        self.assertEqual(summary["subtasks"], 3)
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["skipped"], 1)
        self.assertAlmostEqual(summary["success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["success_rate_attempted"], 0.5)

    def test_per_modality_breakdown(self):
        summary = summarise(self._results())
        self.assertAlmostEqual(summary["success_rate_description"], 1.0)
        self.assertAlmostEqual(summary["success_rate_image"], 0.0)
        self.assertAlmostEqual(summary["success_rate_object"], 0.0)
        self.assertEqual(summary["count_image"], 1)

    def test_mean_distance_ignores_skips(self):
        summary = summarise(self._results())
        self.assertFalse(math.isnan(summary["mean_distance_to_goal"]))

    def test_empty(self):
        self.assertEqual(summarise([]), {"subtasks": 0})


if __name__ == "__main__":
    unittest.main()
