#!/usr/bin/env python3
"""
Render the outbound leg of a chained R2R-CE tour with habitat-sim.

Consumes the tours.json written by fact3r-map/scripts/build_vlnce_tours.py and
produces, per tour, a MASt3R-SLAM-readable image folder plus the metadata the
return leg needs: per-leg frame ranges, the return target, and the geodesic
optimum used as the SPL reference.

The camera matches scripts/render_hm3d_traj.py (square 512x512 at 90 deg HFOV,
fx = fy = cx = cy = 256), so config/hm3d_intrinsics.yaml applies unchanged.
That deliberately differs from the RxR-Habitat challenge observation spec
(480x640); this protocol is not a challenge submission.

habitat-sim lives in the `habitat-vla` env, so run this as:

    conda run -n habitat-vla python3 scripts/render_vlnce_tour.py \
        --tours logs/vlnce/val_unseen/tours.json \
        --mp3d-root datasets/mp3d
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_hm3d_traj import (  # noqa: E402
    build_poses,
    build_sim,
    geodesic_path,
    resample,
    yaw_to_quat,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TOURS = os.path.join(REPO_ROOT, "logs", "vlnce", "val_unseen", "tours.json")
DEFAULT_MP3D_ROOT = os.path.join(REPO_ROOT, "datasets", "mp3d")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "datasets", "vlnce_seqs")


def scene_glb(mp3d_root, scene):
    """Locate a house mesh under `mp3d_root`.

    The canonical habitat MP3D layout is <root>/<house>/<house>.glb, but dumps
    vary (v1/scans nesting, id-prefixed folders, basis-compressed meshes), so
    fall back to a search keyed on the house id.
    """
    canonical = os.path.join(mp3d_root, scene, "%s.glb" % scene)
    if os.path.isfile(canonical):
        return canonical
    matches = sorted(
        glob.glob(os.path.join(mp3d_root, "**", "%s*.glb" % scene), recursive=True)
    )
    if not matches:
        return canonical
    # Prefer an uncompressed mesh; basis meshes need a scene dataset config.
    exact = [path for path in matches if os.path.basename(path) == "%s.glb" % scene]
    return exact[0] if exact else matches[0]


def build_semantic_sim(scene_glb, dataset_config, resolution, hfov, cam_height):
    """Same camera as build_sim, plus a semantic instance sensor.

    The semantic channel is only used to audit target visibility; it is never
    fed to the mapper, which must work from RGB alone.
    """
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [resolution, resolution]
    rgb_spec.position = [0.0, cam_height, 0.0]
    rgb_spec.hfov = hfov

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic_sensor"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = [resolution, resolution]
    semantic_spec.position = [0.0, cam_height, 0.0]
    semantic_spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, semantic_spec]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def semantic_object_table(sim):
    """Map each instance id used by the sensor to its category and centre."""
    scene = sim.semantic_scene
    if scene is None:
        return {}
    table = {}
    for obj in scene.objects:
        if obj is None:
            continue
        try:
            category = obj.category.name() if obj.category is not None else ""
        except Exception:
            category = ""
        # habitat exposes aabb.center as a bound method here, but as a plain
        # attribute in other builds; accept either.
        centre = None
        if obj.aabb is not None:
            raw = obj.aabb.center
            if callable(raw):
                raw = raw()
            centre = np.asarray(raw, dtype=np.float64)
        table[int(obj.semantic_id)] = {
            "category": category,
            "center": centre.tolist() if centre is not None else [0.0, 0.0, 0.0],
        }
    return table


class VisibilityAccumulator:
    """Fold per-frame instance pixel counts into one record per instance."""

    def __init__(self, pixels_per_frame):
        self.pixels_per_frame = float(pixels_per_frame)
        self.frames_visible = {}
        self.total_pixels = {}
        self.max_pixels = {}
        self.first_frame = {}
        self.last_frame = {}

    def add_frame(self, frame_index, semantic_image):
        counts = np.bincount(np.asarray(semantic_image, dtype=np.int64).ravel())
        for instance_id in np.nonzero(counts)[0]:
            pixels = int(counts[instance_id])
            key = int(instance_id)
            self.frames_visible[key] = self.frames_visible.get(key, 0) + 1
            self.total_pixels[key] = self.total_pixels.get(key, 0) + pixels
            if pixels > self.max_pixels.get(key, 0):
                self.max_pixels[key] = pixels
            self.first_frame.setdefault(key, frame_index)
            self.last_frame[key] = frame_index

    def records(self, object_table):
        out = []
        for key in sorted(self.frames_visible):
            # Habitat paints unannotated geometry with instance id 0. Drop it
            # unless the scene really does define an object with that id.
            if key == 0 and key not in object_table:
                continue
            meta = object_table.get(key, {})
            out.append(
                {
                    "instance_id": key,
                    "category": meta.get("category", ""),
                    "center": meta.get("center", [0.0, 0.0, 0.0]),
                    "frames_visible": self.frames_visible[key],
                    "total_pixels": self.total_pixels[key],
                    "max_pixel_fraction": self.max_pixels[key] / self.pixels_per_frame,
                    "first_frame": self.first_frame[key],
                    "last_frame": self.last_frame[key],
                }
            )
        return out


def snap(pathfinder, point):
    """Nearest navigable point; R2R goals sit on the floor, not on the navmesh."""
    snapped = pathfinder.snap_point(np.asarray(point, dtype=np.float32))
    snapped = np.asarray(snapped, dtype=np.float64)
    if not np.all(np.isfinite(snapped)):
        return None
    return snapped


def geodesic_distance(pathfinder, a, b):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not pathfinder.find_path(path):
        return None
    return float(path.geodesic_distance)


def validate_links(distance_fn, tour, tolerance):
    """Re-check euclidean chain links against the navmesh.

    build_vlnce_tours.py links legs by straight-line distance because it runs
    without the scene. Two points a metre apart can sit on opposite sides of a
    wall, so every link is re-measured here and the tour is dropped if any leg
    is not actually walkable from the previous one. `distance_fn` is injected so
    this stays testable without a loaded simulator.
    """
    legs = tour["legs"]
    for previous, current in zip(legs[:-1], legs[1:]):
        distance = distance_fn(
            previous["goal_position"], current["start_position"]
        )
        if distance is None:
            return "leg %d is unreachable from leg %d" % (current["index"], previous["index"])
        if distance > tolerance:
            return "link %d->%d is %.2f m geodesic (tolerance %.2f)" % (
                previous["index"],
                current["index"],
                distance,
                tolerance,
            )
    return None


def tour_polyline(path_fn, tour):
    """Concatenate every leg's reference path into one walkable polyline.

    Waypoints are the ported MP3D-Sim panorama nodes, so consecutive ones are
    joined by the navmesh shortest path rather than a straight line.
    """
    waypoints = []
    for leg in tour["legs"]:
        for point in leg["reference_path"]:
            point = np.asarray(point, dtype=np.float64)
            if waypoints and float(np.linalg.norm(point - waypoints[-1])) < 1e-6:
                continue
            waypoints.append(point)

    polyline = [waypoints[0]]
    straight_segments = 0
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        points = path_fn(start, end)
        if points is None or len(points) < 2:
            # Fall back to the straight segment rather than dropping the leg.
            polyline.append(end)
            straight_segments += 1
            continue
        polyline.extend(points[1:])
    return np.stack(polyline), straight_segments


def leg_frame_ranges(frames, tour):
    """Map each leg's goal onto the frame index that comes closest to it.

    The return leg needs to know which frames observed the target, and the
    outbound render is a single continuous sweep with no leg boundaries in it.
    """
    positions = np.stack([pose[0] for pose in frames])
    boundaries = []
    cursor = 0
    for leg in tour["legs"]:
        goal = np.asarray(leg["goal_position"], dtype=np.float64)
        # Search forward only. Chained tours double back through earlier rooms,
        # so a global nearest frame can land before the previous leg ended and
        # produce an inverted, unsliceable range.
        distances = np.linalg.norm(positions[cursor:] - goal, axis=1)
        boundary = cursor + int(np.argmin(distances))
        boundaries.append(boundary)
        cursor = boundary
    ranges = []
    previous = 0
    for leg, boundary in zip(tour["legs"], boundaries):
        ranges.append(
            {
                "index": leg["index"],
                "episode_id": leg["episode_id"],
                "instruction": leg["instruction"],
                "landmark": leg["landmark"],
                "first_frame": previous,
                "last_frame": boundary,
                "goal_position": leg["goal_position"],
            }
        )
        previous = boundary
    return ranges


def build_pitched_sim(scene_glb, dataset_config, resolution, hfov, cam_height,
                      pitch_degrees, semantic):
    """build_sim with the camera tilted down, and optionally a semantic sensor.

    A level camera at 1.5 m sees no floor nearer than h / tan(vfov/2) = 2.0 m,
    and the floor is what ray carving turns into free space. Tilting the sensor
    down brings the near floor into frame without moving the agent, which is
    how the rover is mounted (2.75 deg) and why its maps carve.

    Only the sensor is pitched. The agent pose written to groundtruth.txt stays
    a yaw-only body pose, so the odometry converter and the BEV builder's
    camera_to_body mount keep the same contract they have on the rover.
    """
    pitch = math.radians(-abs(float(pitch_degrees)))
    specs = []
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [resolution, resolution]
    rgb_spec.position = [0.0, cam_height, 0.0]
    rgb_spec.orientation = [pitch, 0.0, 0.0]
    rgb_spec.hfov = hfov
    specs.append(rgb_spec)
    if semantic:
        semantic_spec = habitat_sim.CameraSensorSpec()
        semantic_spec.uuid = "semantic_sensor"
        semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_spec.resolution = [resolution, resolution]
        semantic_spec.position = [0.0, cam_height, 0.0]
        semantic_spec.orientation = [pitch, 0.0, 0.0]
        semantic_spec.hfov = hfov
        specs.append(semantic_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def resolve_scene(mp3d_root, scene, dataset_config):
    """The scene id to hand habitat: a mesh path, or a name the config knows.

    MP3D dumps are loose .glb files, but ReplicaCAD and HM3D address scenes by
    name through a scene dataset config. Falling back to the bare name lets one
    renderer serve all three.
    """
    glb = scene_glb(mp3d_root, scene)
    if os.path.isfile(glb):
        return glb
    if dataset_config:
        return scene
    return None


def _near_floor_distance(cam_height, hfov, pitch_degrees):
    """Ground distance at which the floor first enters the frame."""
    # A square render is cropped to 4:3 downstream, so the usable half-angle is
    # atan(0.75 * tan(hfov / 2)), not hfov / 2.
    half = math.atan(0.75 * math.tan(math.radians(hfov) / 2.0))
    depression = half + math.radians(abs(float(pitch_degrees)))
    if depression >= math.pi / 2.0:
        return 0.0
    return float(cam_height / math.tan(depression))


def choose_return_leg(pathfinder, tour, final_position, min_distance):
    """Pick the landmark leg whose goal is furthest from where the tour ends.

    build_vlnce_tours.py picks the earliest landmark leg, which maximises the
    time between seeing the object and being asked for it. That is the right
    idea but the wrong axis: in a small house the tour loops back, and the
    earliest goal can end up metres from the final pose, where the agent scores
    a success without moving. Distance is what the metric actually rewards, so
    it is what the leg is chosen on -- measured geodesically, since a target
    behind a wall is not close.
    """
    best = None
    for leg in tour["legs"][:-1]:
        if not leg.get("landmark"):
            continue
        position = snap(pathfinder, leg["goal_position"])
        if position is None:
            continue
        distance = geodesic_distance(pathfinder, final_position, position)
        if distance is None or not np.isfinite(distance):
            continue
        if best is None or distance > best[3]:
            best = (int(leg["index"]), str(leg["landmark"]), position, float(distance))
    if best is None or best[3] < min_distance:
        return None
    return best


def render_tour(tour, mp3d_root, out_dir, args):
    scene = tour["scene"]
    glb = resolve_scene(mp3d_root, scene, args.scene_dataset_config)
    if glb is None:
        print("  SKIP %s: no mesh found and no scene dataset config given" % scene)
        return None

    if args.pitch or args.semantic:
        sim = build_pitched_sim(
            glb, args.scene_dataset_config, args.resolution, args.hfov,
            args.cam_height, args.pitch, args.semantic,
        )
    else:
        sim = build_sim(
            glb, args.scene_dataset_config, args.resolution, args.hfov, args.cam_height
        )
    try:
        if args.recompute_navmesh or not sim.pathfinder.is_loaded:
            # ReplicaCAD ships no navmesh with the scene, and a recomputed one
            # also reflects any clutter the stage places.
            settings = habitat_sim.nav.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = args.agent_radius
            settings.agent_height = args.cam_height
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                print("  SKIP %s: navmesh could not be built" % scene)
                return None
        if not sim.pathfinder.is_loaded:
            print("  SKIP %s: no navmesh loaded" % scene)
            return None
        agent = sim.initialize_agent(0)

        pathfinder = sim.pathfinder
        measure = lambda a, b: geodesic_distance(pathfinder, a, b)  # noqa: E731
        walk = lambda a, b: geodesic_path(pathfinder, a, b)  # noqa: E731

        problem = validate_links(measure, tour, args.link_tolerance)
        if problem is not None:
            print("  SKIP %s: %s" % (scene, problem))
            return None

        polyline, straight_segments = tour_polyline(walk, tour)
        if straight_segments:
            print(
                "    note: %d waypoint pair(s) had no navmesh path and were "
                "straight-lined" % straight_segments
            )
        positions = resample(polyline, args.step)
        frames = build_poses(positions, math.radians(args.max_turn_deg))
        if args.max_frames and len(frames) > args.max_frames:
            print(
                "  %s: truncating %d frames to %d" % (scene, len(frames), args.max_frames)
            )
            frames = frames[: args.max_frames]

        final_position = frames[-1][0]
        chosen = choose_return_leg(
            sim.pathfinder, tour, final_position, args.min_return_distance
        )
        if chosen is None:
            print(
                "  SKIP %s: no landmark leg sits more than %.1f m from the tour "
                "end, so any return would be trivially successful"
                % (scene, args.min_return_distance)
            )
            return None
        return_leg_index, return_query, return_position, optimal = chosen
        trivial = optimal < args.success_distance

        if os.path.isdir(out_dir):
            for name in os.listdir(out_dir):
                if name.endswith(".png"):
                    os.remove(os.path.join(out_dir, name))
        os.makedirs(out_dir, exist_ok=True)

        visibility = None
        object_table = {}
        if args.semantic:
            object_table = semantic_object_table(sim)
            if not object_table:
                print("    note: scene has no semantic annotations; skipping audit")
            else:
                visibility = VisibilityAccumulator(args.resolution * args.resolution)

        gt_lines = []
        for i, (pos, yaw) in enumerate(frames):
            state = habitat_sim.AgentState()
            state.position = pos.astype(np.float32)
            state.rotation = yaw_to_quat(yaw)
            agent.set_state(state)
            obs = sim.get_sensor_observations()
            Image.fromarray(obs["color_sensor"][:, :, :3], mode="RGB").save(
                os.path.join(out_dir, "%06d.png" % i)
            )
            if visibility is not None:
                visibility.add_frame(i, obs["semantic_sensor"])
            cam = pos + np.array([0.0, args.cam_height, 0.0])
            qx, qy, qz, qw = yaw_to_quat(yaw)
            ts = i / args.fps
            gt_lines.append(
                "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
                % (ts, cam[0], cam[1], cam[2], qx, qy, qz, qw)
            )
            if (i + 1) % 100 == 0:
                print("    %d/%d frames" % (i + 1, len(frames)), flush=True)

        with open(os.path.join(out_dir, "groundtruth.txt"), "w") as handle:
            handle.write("# habitat camera poses: timestamp tx ty tz qx qy qz qw\n")
            handle.write("# frame: habitat world (y-up); camera looks along local -z\n")
            handle.write("\n".join(gt_lines) + "\n")

        focal = (args.resolution / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
        travelled = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        meta = {
            "format": "fact3r-vlnce-outbound",
            "version": 1,
            "scene": scene,
            "scene_glb": glb,
            "num_frames": len(frames),
            "fps": args.fps,
            "step_m": args.step,
            "max_turn_deg": args.max_turn_deg,
            "cam_height_m": args.cam_height,
            "camera_pitch_deg": float(args.pitch),
            "near_floor_visible_m": _near_floor_distance(
                args.cam_height, args.hfov, args.pitch
            ),
            "outbound_length_m": travelled,
            "straight_line_segments": straight_segments,
            "resolution": [args.resolution, args.resolution],
            "hfov_deg": args.hfov,
            "intrinsics": {
                "fx": focal,
                "fy": focal,
                "cx": args.resolution / 2.0,
                "cy": args.resolution / 2.0,
            },
            "legs": leg_frame_ranges(frames, tour),
            "return_query": return_query,
            "return_leg_index": return_leg_index,
            "return_position": return_position.tolist(),
            "return_position_raw": tour["legs"][return_leg_index]["goal_position"],
            "chain_time_return_query": tour["return_query"],
            "return_optimal_geodesic_m": optimal,
            "trivial_return": bool(trivial),
            "tour_start_position": frames[0][0].tolist(),
            "tour_final_position": final_position.tolist(),
            "success_distance_m": args.success_distance,
        }
        if visibility is not None:
            records = visibility.records(object_table)
            with open(os.path.join(out_dir, "semantic_visibility.json"), "w") as handle:
                json.dump(
                    {
                        "format": "fact3r-vlnce-visibility",
                        "version": 1,
                        "scene": scene,
                        "frame_count": len(frames),
                        "pixels_per_frame": args.resolution * args.resolution,
                        "instances": records,
                    },
                    handle,
                    indent=2,
                )
            meta["semantic_visibility"] = "semantic_visibility.json"
            meta["semantic_instance_count"] = len(records)
        with open(os.path.join(out_dir, "meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)

        print(
            "  %s: %d frames, outbound %.1f m, return to %r %.1f m away -> %s"
            % (scene, len(frames), travelled, return_query, optimal, out_dir)
        )
        if trivial:
            print(
                "    WARNING: the target is already within the %.1f m success "
                "radius; standing still scores a success, so this tour cannot "
                "measure return navigation" % args.success_distance
            )
        return meta
    finally:
        sim.close()


def main():
    if habitat_sim is None:
        print("habitat-sim is not importable; run under the habitat-vla env", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="Render chained R2R-CE outbound tours.")
    parser.add_argument("--tours", default=DEFAULT_TOURS)
    parser.add_argument("--mp3d-root", default=DEFAULT_MP3D_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--scene-dataset-config",
        default=None,
        help="Scene dataset config; needed for basis-compressed meshes.",
    )
    parser.add_argument("--scene", default=None, help="Render only this MP3D house.")
    parser.add_argument("--limit", type=int, default=0, help="Render at most N tours.")
    parser.add_argument("--step", type=float, default=0.04, help="Metres per frame.")
    parser.add_argument("--max-turn-deg", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = no cap.")
    parser.add_argument("--cam-height", type=float, default=1.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--link-tolerance",
        type=float,
        default=3.0,
        help="Max geodesic metres allowed between consecutive legs.",
    )
    parser.add_argument("--success-distance", type=float, default=3.0)
    parser.add_argument(
        "--min-return-distance",
        type=float,
        default=6.0,
        help="Least geodesic distance from the tour end to the return target. "
             "Below the success radius the return is trivially successful, so "
             "this defaults to twice it.",
    )
    parser.add_argument(
        "--recompute-navmesh",
        action="store_true",
        help="Rebuild the navmesh from scene geometry (needed for ReplicaCAD).",
    )
    parser.add_argument("--agent-radius", type=float, default=0.20)
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Degrees to tilt the camera below horizontal. A level camera sees "
             "no floor nearer than 2 m, which starves ray carving.",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Also record per-instance visibility, to audit whether the return "
             "target was ever actually seen. Never fed to the mapper.",
    )
    args = parser.parse_args()

    with open(args.tours) as handle:
        payload = json.load(handle)
    tours = payload["tours"]
    if args.scene:
        tours = [tour for tour in tours if tour["scene"] == args.scene]
    if args.limit:
        tours = tours[: args.limit]
    if not tours:
        print("no tours selected", file=sys.stderr)
        return 1

    # A scene dataset config resolves scenes by name, so the mesh directory is
    # only required when there is no config to look them up in.
    if not args.scene_dataset_config and not os.path.isdir(args.mp3d_root):
        print(
            "MP3D scenes not found at %s\n"
            "Obtain them with the Matterport download script "
            "(see VLNCE_RUNBOOK.md)." % args.mp3d_root,
            file=sys.stderr,
        )
        return 1

    print("rendering %d tours from %s" % (len(tours), args.tours))
    rendered = []
    for index, tour in enumerate(tours):
        name = "%s_t%02d" % (tour["scene"], index)
        out_dir = os.path.join(args.out_root, name)
        try:
            meta = render_tour(tour, args.mp3d_root, out_dir, args)
        except Exception as error:  # keep the batch going if one scene fails
            print("  FAILED %s: %s: %s" % (name, type(error).__name__, error), flush=True)
            continue
        if meta is not None:
            meta["name"] = name
            rendered.append(meta)

    if not rendered:
        print("no tour rendered", file=sys.stderr)
        return 1

    index_path = os.path.join(args.out_root, "index.json")
    with open(index_path, "w") as handle:
        json.dump({"format": "fact3r-vlnce-outbound-index", "version": 1,
                   "tours": rendered}, handle, indent=2)
    print("\nrendered %d/%d tours -> %s" % (len(rendered), len(tours), index_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
