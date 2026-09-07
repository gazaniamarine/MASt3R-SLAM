import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'build_visual_subgoals.py'
spec = importlib.util.spec_from_file_location('visual_subgoals', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VisualSubgoalTests(unittest.TestCase):
    def test_full_evidence_export_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (12, 12), 'red').save(root / 'rgb.png')
            mask = np.ones((12, 12), dtype=bool)
            np.savez(root / 'mask.npz', mask=mask)
            (root / 'manifest.json').write_text(json.dumps({'keyframes': [{'frame_id': 1, 'rgb_file': 'rgb.png'}]}))
            (root / 'index.json').write_text(json.dumps({'source_keyframes': str(root), 'source_proposals': str(root), 'observations': [{'frame_id': 1, 'timestamp': 1., 'group_id': 'e0', 'proposal_id': 'p0', 'mask_file': 'mask.npz', 'proposal_score': .9}]}))
            np.savez(root / 'grid.npz', occupancy=np.zeros((8, 8)), semantic_ids=np.zeros((8, 8), int), origin_xy=[0., 0.], resolution=1., floor_u=[1., 0., 0.], floor_v=[0., 0., 1.], floor_origin=[0., 0., 0.])
            np.savetxt(root / 'map.txt', [[1, 1.5, -.5, .5, 0, 0, 0, 1]])
            (root / 'map_semantic.json').write_text(json.dumps({'format': 'fact3r-depth-semantic-bev', 'grid_file': 'grid.npz', 'source_observation_index': str(root / 'index.json'), 'groups': [{'group_id': 'e0', 'semantic_id': 0}]}))
            np.save(root / 'route.npy', [[.5, .5], [1.5, .5], [2.5, .5]])
            args = ['build_visual_subgoals', '--map', str(root / 'map'), '--path', str(root / 'route.npy'), '--coordinate-order', 'xy', '--output', str(root / 'out'), '--landmark-radius', '10']
            with patch('sys.argv', args):
                module.main()
            result = json.loads((root / 'out/subgoals.json').read_text())
            self.assertEqual(result['subgoals'][0]['action_at_waypoint'], 'stop')
            self.assertEqual(result['subgoals'][0]['landmark']['entity_id'], 'e0')
            self.assertTrue((root / 'out/subgoal_001_crop.jpg').exists())
            self.assertIsNone(result['caption_model'])

    def test_turns_keep_original_route_and_long_segments(self):
        path = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], float)
        original = path.copy()
        selected, arc, turns = module.anchors(path, spacing=3)
        self.assertIn(2, selected)
        self.assertAlmostEqual(turns[2], np.pi / 2)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 4)
        np.testing.assert_array_equal(path, original)
        self.assertEqual(arc[-1], 4)

    def test_planner_yx_order(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'plan.npz'
            np.savez(file, centerline=np.array([[[4, 1], [4, 2], [3, 2]]]))
            route = module.load_route(file, 'centerline', 0, 'yx')
            np.testing.assert_array_equal(route, [[1, 4], [2, 4], [2, 3]])
            self.assertLess(module.anchors(route)[2][1], 0)

    def test_pose_uses_floor_projection_and_clock_offset(self):
        trajectory = np.array([[2, 3, -.5, 4, 0, 0, 0, 1]], float)
        basis = np.array([[0, 0, 1], [1, 0, 0]])
        xy, heading = module.observation_pose({'timestamp': 1}, trajectory, 1, basis, np.zeros(3), .05)
        np.testing.assert_array_equal(xy, [4, 3])
        self.assertAlmostEqual(heading, np.pi/2)
        self.assertIsNone(module.observation_pose({'timestamp': 10}, trajectory, 1, basis, np.zeros(3), .05))


if __name__ == '__main__':
    unittest.main()
