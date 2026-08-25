from __future__ import annotations

import unittest

import numpy as np

from fact3r.visualization.association import (
    DisplayAssignment,
    DisplayFrame,
    display_frame_from_manifest,
    entity_colour,
    join_panels,
    mask_boundary,
    render_association_panel,
    render_rgb_panel,
)


class AssociationVisualizationTests(unittest.TestCase):
    def test_residual_transport_manifest_uses_exact_forbidden_mass(self) -> None:
        frame = display_frame_from_manifest(
            {
                "frame_id": 2,
                "entity_count_after": 3,
                "matches": [],
                "created_entity_ids": [],
                "unmatched_proposals": [],
                "converged": True,
                "iterations": 12,
                "forbidden_mass": 0.0,
            }
        )
        self.assertEqual(frame.forbidden_mass, 0.0)

    def test_mapping_manifest_normalizes_matches_and_created_entities(self) -> None:
        frame = display_frame_from_manifest(
            {
                "frame_id": 16,
                "entity_count_after": 4,
                "matches": [{"proposal_id": "p0", "entity_id": "entity-000001"}],
                "created_entity_ids": ["entity-000003"],
                "unmatched_reason_counts": {"cost_above_threshold": 1},
                "unmatched_proposals": [
                    {
                        "proposal_id": "p1",
                        "created_entity_id": "entity-000003",
                    }
                ],
                "converged": True,
                "iterations": 12,
                "noncandidate_mass": 0.125,
            }
        )
        self.assertEqual(frame.matched_count, 1)
        self.assertEqual(frame.created_count, 1)
        self.assertEqual(
            tuple(assignment.status for assignment in frame.assignments),
            ("matched", "created"),
        )
        self.assertTrue(frame.converged)
        self.assertAlmostEqual(frame.forbidden_mass, 0.125)

    def test_delayed_commitment_masks_remain_visible_while_pending(self) -> None:
        frame = display_frame_from_manifest(
            {
                "frame_id": 17,
                "entity_count_after": 1,
                "matches": [],
                "created_entity_ids": [],
                "unmatched_proposals": [
                    {
                        "proposal_id": "pending-mask",
                        "commitment_status": "deferred",
                        "track_id": "track-000005",
                        "created_entity_id": None,
                    },
                    {
                        "proposal_id": "held-mask",
                        "commitment_status": "held_existing",
                        "resolved_entity_id": "entity-000000",
                        "created_entity_id": None,
                    },
                ],
            }
        )
        self.assertEqual(frame.pending_count, 1)
        self.assertEqual(frame.held_count, 1)
        self.assertEqual(
            tuple(assignment.status for assignment in frame.assignments),
            ("pending", "held"),
        )

    def test_mask_boundary_marks_edge_but_not_interior(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:6, 1:6] = True
        boundary = mask_boundary(mask)
        self.assertTrue(boundary[1, 3])
        self.assertFalse(boundary[3, 3])

    def test_entity_colours_are_stable_and_distinct(self) -> None:
        first = entity_colour("entity-000001")
        np.testing.assert_array_equal(first, entity_colour("entity-000001"))
        self.assertFalse(np.array_equal(first, entity_colour("entity-000002")))

    def test_panels_render_at_keyframe_resolution_with_header(self) -> None:
        rgb = np.full((32, 48, 3), 80, dtype=np.uint8)
        mask = np.zeros((32, 48), dtype=bool)
        mask[8:24, 12:36] = True
        frame = DisplayFrame(
            frame_id=1,
            matched_count=1,
            created_count=0,
            entity_count=1,
            unmatched_reason_counts={},
            assignments=(
                DisplayAssignment(
                    proposal_id="p0",
                    entity_id="entity-000000",
                    status="matched",
                ),
            ),
        )
        rgb_panel = render_rgb_panel(rgb, frame_id=1)
        map_panel = render_association_panel(
            rgb, {"p0": mask}, frame, title="balanced"
        )
        montage = join_panels((rgb_panel, map_panel))
        self.assertEqual(rgb_panel.size, (48, 80))
        self.assertEqual(map_panel.size, (48, 80))
        self.assertEqual(montage.size, (100, 80))
        rendered = np.asarray(map_panel)
        np.testing.assert_array_equal(rendered[48 + 8, 20], [40, 255, 80])


if __name__ == "__main__":
    unittest.main()
