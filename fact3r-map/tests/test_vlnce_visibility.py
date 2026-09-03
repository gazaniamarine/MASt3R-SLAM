from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from fact3r.experiments.vlnce_visibility import (
    InstanceVisibility,
    Verdict,
    category_matches,
    normalise,
    score_target,
    select_candidates,
    summarise,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_vlnce_tour import VisibilityAccumulator  # noqa: E402


def _instance(
    instance_id=1,
    category="table",
    center=(0.0, 0.0, 0.0),
    frames_visible=30,
    max_pixel_fraction=0.05,
):
    return InstanceVisibility(
        instance_id=instance_id,
        category=category,
        center=np.asarray(center, dtype=np.float64),
        frames_visible=frames_visible,
        total_pixels=frames_visible * 1000,
        max_pixel_fraction=max_pixel_fraction,
        first_frame=0,
        last_frame=frames_visible,
    )


class CategoryMatchTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(category_matches("table", "table"))

    def test_modifier_is_ignored(self):
        self.assertTrue(category_matches("glass dining table", "table"))
        self.assertTrue(category_matches("pool table", "table"))

    def test_synonym_match(self):
        self.assertTrue(category_matches("couch", "sofa"))
        self.assertTrue(category_matches("tv", "tv_monitor"))

    def test_underscores_and_articles_are_normalised(self):
        self.assertEqual(normalise("the tv_monitor"), "tv monitor")
        self.assertTrue(category_matches("the television", "tv_monitor"))

    def test_does_not_cross_head_nouns(self):
        self.assertFalse(category_matches("table", "chair"))
        self.assertFalse(category_matches("bed", "sink"))

    def test_empty_never_matches(self):
        self.assertFalse(category_matches("", "table"))
        self.assertFalse(category_matches("table", ""))


class SelectCandidatesTest(unittest.TestCase):
    def test_keeps_a_matching_nearby_instance(self):
        instances = [_instance(category="table", center=(1.0, 0.0, 0.0))]
        self.assertEqual(len(select_candidates(instances, "table", (0, 0, 0))), 1)

    def test_drops_a_matching_instance_that_is_far_away(self):
        instances = [_instance(category="table", center=(20.0, 0.0, 0.0))]
        self.assertEqual(select_candidates(instances, "table", (0, 0, 0)), [])

    def test_drops_a_nearby_instance_of_the_wrong_category(self):
        instances = [_instance(category="chair", center=(1.0, 0.0, 0.0))]
        self.assertEqual(select_candidates(instances, "table", (0, 0, 0)), [])


class ScoreTargetTest(unittest.TestCase):
    def test_well_seen_target_is_observed(self):
        result = score_target([_instance()], "table", (0, 0, 0))
        self.assertEqual(result.verdict, Verdict.OBSERVED)
        self.assertTrue(result.usable)

    def test_missing_instance_is_reported_distinctly(self):
        result = score_target([], "table", (0, 0, 0))
        self.assertEqual(result.verdict, Verdict.NO_CANDIDATE)
        self.assertFalse(result.usable)
        self.assertIn("no semantic instance", result.reason)

    def test_never_rendered_target_is_never_observed(self):
        instance = _instance(frames_visible=0, max_pixel_fraction=0.0)
        result = score_target([instance], "table", (0, 0, 0))
        self.assertEqual(result.verdict, Verdict.NEVER_OBSERVED)

    def test_too_few_frames_is_a_glimpse(self):
        result = score_target([_instance(frames_visible=2)], "table", (0, 0, 0))
        self.assertEqual(result.verdict, Verdict.GLIMPSED)

    def test_too_small_on_screen_is_a_glimpse(self):
        result = score_target(
            [_instance(max_pixel_fraction=0.0001)], "table", (0, 0, 0)
        )
        self.assertEqual(result.verdict, Verdict.GLIMPSED)

    def test_best_candidate_wins_on_screen_area(self):
        small = _instance(instance_id=1, max_pixel_fraction=0.01)
        large = _instance(instance_id=2, max_pixel_fraction=0.2)
        result = score_target([small, large], "table", (0, 0, 0))
        self.assertEqual(result.best.instance_id, 2)

    def test_thresholds_are_configurable(self):
        instance = _instance(frames_visible=3)
        self.assertEqual(
            score_target([instance], "table", (0, 0, 0), min_frames=2).verdict,
            Verdict.OBSERVED,
        )

    def test_summary_counts_usable_tours(self):
        good = score_target([_instance()], "table", (0, 0, 0))
        bad = score_target([], "table", (0, 0, 0))
        summary = summarise([good, bad])
        self.assertEqual(summary, {
            "tours": 2,
            "usable": 1,
            "by_verdict": {Verdict.OBSERVED: 1, Verdict.NO_CANDIDATE: 1},
        })


class VisibilityAccumulatorTest(unittest.TestCase):
    def test_counts_pixels_and_frames(self):
        accumulator = VisibilityAccumulator(pixels_per_frame=100)
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[:5, :] = 7  # instance 7 covers half the frame
        accumulator.add_frame(0, frame)
        accumulator.add_frame(1, frame)
        records = {r["instance_id"]: r for r in accumulator.records({})}
        self.assertEqual(records[7]["frames_visible"], 2)
        self.assertAlmostEqual(records[7]["max_pixel_fraction"], 0.5)
        self.assertEqual(records[7]["first_frame"], 0)
        self.assertEqual(records[7]["last_frame"], 1)

    def test_tracks_the_largest_appearance(self):
        accumulator = VisibilityAccumulator(pixels_per_frame=100)
        small = np.zeros((10, 10), dtype=np.int32)
        small[0, :2] = 3
        large = np.zeros((10, 10), dtype=np.int32)
        large[:4, :] = 3
        accumulator.add_frame(0, large)
        accumulator.add_frame(1, small)
        records = {r["instance_id"]: r for r in accumulator.records({})}
        self.assertAlmostEqual(records[3]["max_pixel_fraction"], 0.4)

    def test_unannotated_background_is_dropped(self):
        accumulator = VisibilityAccumulator(pixels_per_frame=100)
        frame = np.zeros((10, 10), dtype=np.int32)
        frame[:5, :] = 7
        accumulator.add_frame(0, frame)
        ids = {record["instance_id"] for record in accumulator.records({})}
        self.assertEqual(ids, {7})

    def test_background_is_kept_when_the_scene_defines_id_zero(self):
        accumulator = VisibilityAccumulator(pixels_per_frame=100)
        accumulator.add_frame(0, np.zeros((10, 10), dtype=np.int32))
        ids = {
            record["instance_id"]
            for record in accumulator.records({0: {"category": "wall", "center": [0, 0, 0]}})
        }
        self.assertEqual(ids, {0})

    def test_attaches_category_metadata(self):
        accumulator = VisibilityAccumulator(pixels_per_frame=100)
        frame = np.full((10, 10), 5, dtype=np.int32)
        accumulator.add_frame(0, frame)
        records = accumulator.records({5: {"category": "sofa", "center": [1, 2, 3]}})
        self.assertEqual(records[0]["category"], "sofa")
        self.assertEqual(records[0]["center"], [1, 2, 3])

    def test_records_feed_the_scorer(self):
        accumulator = VisibilityAccumulator(pixels_per_frame=100)
        frame = np.full((10, 10), 5, dtype=np.int32)
        for index in range(10):
            accumulator.add_frame(index, frame)
        instances = [
            InstanceVisibility.from_json(record)
            for record in accumulator.records(
                {5: {"category": "sofa", "center": [0.0, 0.0, 0.0]}}
            )
        ]
        result = score_target(instances, "couch", (0, 0, 0))
        self.assertEqual(result.verdict, Verdict.OBSERVED)


if __name__ == "__main__":
    unittest.main()
