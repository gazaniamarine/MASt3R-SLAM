#!/usr/bin/env python3
"""Attach pose-matched landmark evidence and optional SmolVLM captions to a route."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_semantic_bev import _load_observation_image, _map_manifest


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_route(path, key, plan_index, order):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            route = np.array(data[key], dtype=float)
    elif path.suffix == '.npy':
        route = np.load(path, allow_pickle=False).astype(float)
    else:
        data = json.loads(path.read_text())
        route = np.asarray(data[key] if isinstance(data, dict) else data, dtype=float)
    if route.ndim == 3:
        route = route[plan_index]
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise ValueError('Route must be N x 2 (or plans x N x 2), with at least two points')
    if not np.isfinite(route).all():
        raise ValueError('Route contains non-finite coordinates')
    if order == 'yx':
        route = route[:, ::-1]
    route = route[np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-8]]
    if len(route) < 2:
        raise ValueError('Route has no movement')
    return route


def anchors(route, spacing=3.0, turn_degrees=30.0, lookahead=0.6):
    """Select indices only; never replace collision-checked path segments by chords."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    turns = np.zeros(len(route))
    for i in range(1, len(route) - 1):
        a = max(0, np.searchsorted(arc, arc[i] - lookahead, side='right') - 1)
        b = min(len(route) - 1, np.searchsorted(arc, arc[i] + lookahead))
        before, after = route[i] - route[a], route[b] - route[i]
        turns[i] = wrap(np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0]))
    candidates = sorted(range(1, len(route) - 1), key=lambda i: -abs(turns[i]))
    selected = [0, len(route) - 1]
    for i in candidates:
        if abs(turns[i]) >= np.deg2rad(turn_degrees) and all(abs(arc[i] - arc[j]) >= lookahead for j in selected):
            selected.append(i)
    for target in np.arange(spacing, arc[-1], spacing):
        i = int(np.searchsorted(arc, target))
        if all(abs(arc[i] - arc[j]) >= lookahead for j in selected):
            selected.append(i)
    return sorted(set(selected)), arc, turns


def observation_pose(observation, trajectory, offset, basis, floor_origin, tolerance):
    timestamp = observation.get('timestamp')
    if timestamp is None:
        return None
    t = float(timestamp) + offset
    j = int(np.argmin(abs(trajectory[:, 0] - t)))
    if abs(trajectory[j, 0] - t) > tolerance:
        return None
    row = trajectory[j]
    xy = (row[1:4] - floor_origin) @ basis.T
    # build_depth_semantic_bev writes qy=sin(-yaw/2), qw=cos(-yaw/2).
    yaw = -2 * np.arctan2(row[5], row[7])
    direction = np.array([np.cos(yaw), 0., np.sin(yaw)]) @ basis.T
    return xy, float(np.arctan2(direction[1], direction[0]))


class SmolCaptioner:
    def __init__(self, model_id, device):
        self.model_id, self.device = model_id, device
        self.model = None

    def caption(self, image):
        import torch
        import transformers
        if self.model is None:
            loader = getattr(transformers, 'AutoModelForImageTextToText', None)
            if loader is None:
                loader = transformers.AutoModelForVision2Seq
            self.processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            self.model = loader.from_pretrained(self.model_id).to(self.device).eval()
        message = [{'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': 'Name the visible object in this isolated image in a short noun phrase. Use only visible properties. If unclear, answer uncertain. Do not give navigation instructions.'}]}]
        prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        return None if not text or 'uncertain' in text.lower() else text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--path-key', default='centerline')
    p.add_argument('--plan-index', type=int, default=0)
    p.add_argument('--coordinate-order', choices=['xy', 'yx'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=3.)
    p.add_argument('--turn-degrees', type=float, default=30.)
    p.add_argument('--approach-radius', type=float, default=1.5)
    p.add_argument('--heading-tolerance-degrees', type=float, default=45.)
    p.add_argument('--landmark-radius', type=float, default=3.)
    p.add_argument('--timestamp-tolerance', type=float, default=.05)
    p.add_argument('--captioner', choices=['none', 'smolvlm'], default='none')
    p.add_argument('--model', default='HuggingFaceTB/SmolVLM-256M-Instruct')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    for value in (args.spacing, args.approach_radius, args.landmark_radius, args.timestamp_tolerance):
        if value <= 0:
            p.error('Distances and timestamp tolerance must be positive')
    if not 0 < args.turn_degrees < 180 or not 0 < args.heading_tolerance_degrees <= 180:
        p.error('Invalid angle threshold')
    started = perf_counter()
    map_path = _map_manifest(args.map)
    manifest = json.loads(map_path.read_text())
    if manifest.get('format') != 'fact3r-depth-semantic-bev':
        raise ValueError('Expected a depth-semantic BEV map')
    with np.load(map_path.parent / manifest['grid_file'], allow_pickle=False) as grid:
        ids, occupancy = grid['semantic_ids'], grid['occupancy']
        origin, resolution = grid['origin_xy'], float(grid['resolution'])
        basis = np.stack([grid['floor_u'], grid['floor_v']])
        floor_origin = grid['floor_origin']
    route = load_route(args.path, args.path_key, args.plan_index, args.coordinate_order)
    cells = np.floor((route - origin) / resolution).astype(int)
    if np.any(cells < 0) or np.any(cells[:, 0] >= ids.shape[1]) or np.any(cells[:, 1] >= ids.shape[0]):
        raise ValueError('Route is outside this BEV: check coordinate order and map provenance')
    # occupancy is a ROS-style probability grid (-1 unknown, 0..64 free, >=65
    # occupied), not binary -- same threshold build_depth_semantic_bev.py's
    # own renderer and fact3r.vlm_nav.bev_pointing use.
    cell_occupancy = occupancy[cells[:, 1], cells[:, 0]]
    if np.any(cell_occupancy < 0) or np.any(cell_occupancy >= 65):
        raise ValueError('Route vertices enter occupied/unknown cells; validate the planner route first')
    source_index = Path(manifest['source_observation_index'])
    if source_index.is_dir():
        source_index /= 'manifest.json'
    index = json.loads(source_index.read_text())
    trajectory_path = Path(str(map_path).removesuffix('_semantic.json') + '.txt')
    trajectory = np.loadtxt(trajectory_path, ndmin=2) if trajectory_path.exists() else None
    if trajectory is not None and (trajectory.shape[1] != 8 or not np.isfinite(trajectory).all()):
        raise ValueError('Expected finite timestamp xyz quaternion trajectory rows')
    selected, arc, turns = anchors(route, args.spacing, args.turn_degrees)
    entity_positions = {}
    for group in manifest['groups']:
        rr, cc = np.nonzero(ids == int(group['semantic_id']))
        if len(rr):
            entity_positions[str(group['group_id'])] = origin + resolution * np.array([cc.mean() + .5, rr.mean() + .5])
    posed = []
    if trajectory is not None:
        for obs in index['observations']:
            pose = observation_pose(obs, trajectory, float(manifest.get('time_offset_seconds', 0)), basis, floor_origin, args.timestamp_tolerance)
            if pose is not None and str(obs['group_id']) in entity_positions:
                posed.append((obs, *pose))
    args.output.mkdir(parents=True, exist_ok=True)
    captioner = SmolCaptioner(args.model, args.device) if args.captioner == 'smolvlm' else None
    caption_cache = {}
    subgoals = []
    cards = []
    for number, (a, b) in enumerate(zip(selected, selected[1:]), 1):
        approach = max(a, int(np.searchsorted(arc, max(arc[a], arc[b] - .6))))
        approach = min(approach, b - 1)
        direction = route[b] - route[approach]
        heading = float(np.arctan2(direction[1], direction[0]))
        candidates = []
        for obs, xy, yaw in posed:
            distance = float(np.linalg.norm(xy - route[approach]))
            angle = abs(wrap(yaw - heading))
            landmark_distance = np.linalg.norm(entity_positions[str(obs['group_id'])] - route[b])
            if distance <= args.approach_radius and angle <= np.deg2rad(args.heading_tolerance_degrees) and landmark_distance <= args.landmark_radius:
                score = distance / args.approach_radius + angle / np.pi - .1 * float(obs.get('proposal_score', 0))
                candidates.append((score, obs))
        action = 'stop' if b == len(route) - 1 else ('turn_left' if turns[b] >= np.deg2rad(args.turn_degrees) else 'turn_right' if turns[b] <= -np.deg2rad(args.turn_degrees) else 'continue')
        instruction = f'Follow the planned path for {arc[b] - arc[a]:.1f} metres. At the waypoint, {action.replace("_", " ")}.'
        landmark = None
        if candidates:
            obs = min(candidates, key=lambda x: x[0])[1]
            rgb, mask = _load_observation_image(obs, index)
            if mask.any():
                rr, cc = np.nonzero(mask)
                isolated = rgb.copy()
                isolated[~mask] = 128
                crop = Image.fromarray(isolated[rr.min():rr.max()+1, cc.min():cc.max()+1])
                crop.thumbnail((512, 512))
                image_name, mask_name = f'subgoal_{number:03d}.jpg', f'subgoal_{number:03d}_mask.png'
                Image.fromarray(rgb).save(args.output / image_name)
                Image.fromarray(mask.astype('uint8') * 255).save(args.output / mask_name)
                crop_name = f'subgoal_{number:03d}_crop.jpg'
                crop.save(args.output / crop_name)
                key = (int(obs['frame_id']), str(obs['proposal_id']))
                if captioner and key not in caption_cache:
                    caption_cache[key] = captioner.caption(crop)
                caption = caption_cache.get(key)
                landmark = {'entity_id': obs['group_id'], 'frame_id': obs['frame_id'], 'reference_image': image_name, 'reference_mask': mask_name, 'crop': crop_name, 'description': caption, 'description_verified': False, 'visibility': 'historical_approach_pose_match; current visibility unverified'}
                if caption:
                    instruction += f' Reference landmark: {caption}'
                cards.append(f'<h3>Subgoal {number}</h3><p>{html.escape(instruction)}</p><img width="480" src="{image_name}"><img width="200" src="{crop_name}">')
        if landmark is None:
            cards.append(f'<h3>Subgoal {number}</h3><p>{html.escape(instruction)} No aligned landmark evidence.</p>')
        subgoals.append({'id': number, 'path_start_index': a, 'path_end_index': b, 'waypoint_xy_m': route[b].tolist(), 'approach_heading_rad': heading, 'turn_angle_rad': float(turns[b]), 'action_at_waypoint': action, 'instruction': instruction, 'landmark': landmark, 'completion': {'position_tolerance_m': .4, 'heading_tolerance_rad': .2}})
    result = {'format': 'fact3r-visual-subgoals', 'version': 1, 'source_map': str(map_path.resolve()), 'source_path': str(args.path.resolve()), 'path_key': args.path_key, 'plan_index': args.plan_index, 'coordinate_frame': 'semantic BEV floor-plane xy metres', 'route_xy_m': route.tolist(), 'start_xy_m': route[0].tolist(), 'subgoals': subgoals, 'caption_model': args.model if captioner else None, 'elapsed_seconds': perf_counter() - started, 'execution_note': 'Follow original route segments; waypoint chords are not collision checked. Landmark captions are descriptive evidence, not action triggers.'}
    (args.output / 'subgoals.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Visual subgoals</title><h1>Visual subgoals</h1>' + ''.join(cards))
    print(f'Created {len(subgoals)} subgoals; {sum(s["landmark"] is not None for s in subgoals)} with aligned evidence. {args.output / "index.html"}')


if __name__ == '__main__':
    main()
