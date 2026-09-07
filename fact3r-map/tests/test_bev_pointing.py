from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fact3r.vlm_nav.bev_pointing import (
    PointingResult,
    build_pointing_prompt,
    parse_pointing_output,
    render_pointing_bev,
    resolve_pointing,
)


def _two_entity_grid():
    """A 4x6 grid with two entities: a big one (rug-sized) and a small one."""

    occupancy = np.full((4, 6), 0, dtype=np.int8)
    semantic_ids = np.full((4, 6), -1, dtype=np.int32)
    semantic_ids[1:3, 0:3] = 0  # 6 cells: the "big" entity
    semantic_ids[0, 5] = 1  # 1 cell: the "small" entity
    groups = [
        {"semantic_id": 0, "group_id": "big-entity"},
        {"semantic_id": 1, "group_id": "small-entity"},
    ]
    return occupancy, semantic_ids, groups


class RenderPointingBevTests(unittest.TestCase):
    def test_marks_entities_above_the_cell_floor_only(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "map.png"
            path, markers = render_pointing_bev(
                occupancy,
                semantic_ids,
                groups,
                origin_xy=(0.0, 0.0),
                resolution=0.5,
                min_cells=3,
                max_markers=10,
                output=output,
            )
            self.assertTrue(path.is_file())
            # Only "big-entity" clears the min_cells=3 floor.
            self.assertEqual(len(markers), 1)
            self.assertEqual(markers[1]["group_id"], "big-entity")
            with Image.open(path) as image:
                self.assertEqual(image.size, (6, 4))

    def test_max_markers_keeps_the_most_supported_entities(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "map.png"
            _, markers = render_pointing_bev(
                occupancy,
                semantic_ids,
                groups,
                origin_xy=(0.0, 0.0),
                resolution=0.5,
                min_cells=1,
                max_markers=1,
                output=output,
            )
            self.assertEqual(len(markers), 1)
            # The 6-cell entity beats the 1-cell one for the single slot.
            self.assertEqual(markers[1]["group_id"], "big-entity")

    def test_marker_centroid_is_a_real_grid_cell(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "map.png"
            _, markers = render_pointing_bev(
                occupancy,
                semantic_ids,
                groups,
                origin_xy=(0.0, 0.0),
                resolution=0.5,
                min_cells=1,
                max_markers=10,
                output=output,
            )
            big = next(m for m in markers.values() if m["group_id"] == "big-entity")
            row, col = big["centroid_cell_rc"]
            self.assertTrue(1 <= row <= 2)
            self.assertTrue(0 <= col <= 2)


class PromptAndParseTests(unittest.TestCase):
    def test_prompt_lists_every_marker_number(self) -> None:
        markers = {1: {}, 3: {}, 7: {}}
        prompt = build_pointing_prompt("the rug", markers)
        self.assertIn("[1, 3, 7]", prompt)
        self.assertIn("the rug", prompt)

    def test_parses_a_marker_answer(self) -> None:
        result = parse_pointing_output('{"marker": 3, "x": 0.1, "y": 0.9, "reason": "matches"}')
        self.assertEqual(result.marker, 3)
        self.assertAlmostEqual(result.x, 0.1)
        self.assertEqual(result.reason, "matches")

    def test_parses_a_null_marker_with_freeform_pixel(self) -> None:
        result = parse_pointing_output('{"marker": null, "x": 0.4, "y": 0.6, "reason": "no marker fits"}')
        self.assertIsNone(result.marker)

    def test_clips_out_of_range_coordinates_rather_than_raising(self) -> None:
        result = parse_pointing_output('{"marker": null, "x": 1.5, "y": -0.3, "reason": "r"}')
        self.assertEqual(result.x, 1.0)
        self.assertEqual(result.y, 0.0)

    def test_rejects_output_with_no_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_pointing_output("I think it's marker 3.")

    def test_rejects_non_integer_marker(self) -> None:
        with self.assertRaises(ValueError):
            parse_pointing_output('{"marker": "three", "x": 0.5, "y": 0.5, "reason": "r"}')


class ResolvePointingTests(unittest.TestCase):
    def test_marker_answer_resolves_to_that_entity(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        markers = {
            1: {"group_id": "big-entity", "semantic_id": 0},
            2: {"group_id": "small-entity", "semantic_id": 1},
        }
        result = PointingResult(marker=2, x=0.5, y=0.5, reason="matches the small one")
        candidate = resolve_pointing(
            result,
            markers,
            semantic_ids=semantic_ids,
            groups=groups,
            confidence=None,
            origin_xy=(0.0, 0.0),
            resolution=0.5,
        )
        self.assertEqual(candidate["group_id"], "small-entity")
        self.assertEqual(candidate["cell_count"], 1)

    def test_freeform_pixel_snaps_to_the_nearest_entity(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        markers: dict[int, dict[str, object]] = {}
        # (row, col) = (1, 1) is inside "big-entity"; as a display pixel that is
        # x=1, y = height-1-row = 4-1-1 = 2 -> normalized (1/5, 2/3).
        result = PointingResult(marker=None, x=1.0 / 5.0, y=2.0 / 3.0, reason="freeform")
        candidate = resolve_pointing(
            result,
            markers,
            semantic_ids=semantic_ids,
            groups=groups,
            confidence=None,
            origin_xy=(0.0, 0.0),
            resolution=0.5,
            fallback_search_radius_cells=5.0,
        )
        self.assertEqual(candidate["group_id"], "big-entity")

    def test_marker_number_not_on_the_map_falls_back_to_freeform(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        markers = {1: {"group_id": "big-entity", "semantic_id": 0}}
        # marker=9 does not exist; x, y point at the small entity's cell.
        # small entity cell is (row=0, col=5); pixel x=5, y=4-1-0=3 ->
        # normalized (5/5, 3/3).
        result = PointingResult(marker=9, x=1.0, y=1.0, reason="hallucinated marker")
        candidate = resolve_pointing(
            result,
            markers,
            semantic_ids=semantic_ids,
            groups=groups,
            confidence=None,
            origin_xy=(0.0, 0.0),
            resolution=0.5,
            fallback_search_radius_cells=5.0,
        )
        self.assertEqual(candidate["group_id"], "small-entity")

    def test_pixel_far_from_any_entity_raises_rather_than_guessing(self) -> None:
        occupancy, semantic_ids, groups = _two_entity_grid()
        result = PointingResult(marker=None, x=0.0, y=0.0, reason="empty corner")
        with self.assertRaises(ValueError):
            resolve_pointing(
                result,
                {},
                semantic_ids=semantic_ids,
                groups=groups,
                confidence=None,
                origin_xy=(0.0, 0.0),
                resolution=0.5,
                fallback_search_radius_cells=0.5,
            )


if __name__ == "__main__":
    unittest.main()
