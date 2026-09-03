#!/usr/bin/env python3
"""
Render GOAT-Bench image goals from the poses stored in the episode file.

GOAT does not ship goal images -- it ships the camera pose each was taken from
(position, rotation, hfov, dimensions). The image has to be re-rendered in the
simulator, which is why this runs under habitat-vla.

    conda run -n habitat-vla python3 scripts/render_goat_image_goals.py \
        --episodes datasets/goat/data/datasets/goat_bench/hm3d/v1/val_unseen/content/y9hTuugGdiq.json.gz \
        --hm3d-root datasets/hm3d_root \
        --out datasets/goat_image_goals
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_hm3d_traj import find_dataset_config  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scene_glb(hm3d_root, scene):
    """Find `<scene>.basis.glb` under any split of the HM3D tree."""
    matches = sorted(
        glob.glob(os.path.join(hm3d_root, "*", "*-%s" % scene, "%s.basis.glb" % scene))
    )
    return matches[0] if matches else None


def build_sim(scene_glb_path, dataset_config, width, height, hfov):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "color_sensor"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    # habitat wants [height, width]; GOAT stores [width, height].
    spec.resolution = [height, width]
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb_path
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def main():
    if habitat_sim is None:
        print("habitat-sim is not importable; run under habitat-vla", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help="one scene's .json.gz")
    parser.add_argument("--hm3d-root", default=os.path.join(REPO_ROOT, "datasets", "hm3d_root"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="render at most N goals")
    parser.add_argument(
        "--referenced-only",
        action="store_true",
        help="render only the (object, view) pairs the episodes actually ask "
             "for; a scene stores every view of every goal, of which the "
             "episodes use a small fraction",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    with gzip.open(args.episodes, "rt") as handle:
        payload = json.load(handle)
    scene = os.path.basename(args.episodes).replace(".json.gz", "")

    glb = scene_glb(args.hm3d_root, scene)
    if glb is None:
        print("scene not found under %s: %s" % (args.hm3d_root, scene), file=sys.stderr)
        return 1
    config = find_dataset_config(args.hm3d_root)

    # Every goal in a scene shares the same camera intrinsics in practice, but
    # they are stored per goal, so the sim is rebuilt when they differ.
    # Episode tasks are [category, modality, instance_id, image_index]; only the
    # image ones name a view, and only those need rendering.
    wanted = None
    if args.referenced_only:
        wanted = set()
        for episode in payload.get("episodes", []):
            for task in episode.get("tasks", []):
                if len(task) >= 4 and task[1] == "image":
                    wanted.add((task[2], int(task[3])))

    jobs = []
    for key, records in payload.get("goals", {}).items():
        for record in records:
            for index, goal in enumerate(record.get("image_goals", []) or []):
                if wanted is not None and (record["object_id"], index) not in wanted:
                    continue
                jobs.append((record["object_id"], index, goal))
    if args.skip_existing:
        jobs = [
            job for job in jobs
            if not os.path.isfile(
                os.path.join(args.out, "%s_%s_%02d.jpg" % (scene, job[0], job[1]))
            )
        ]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("no image goals in %s" % args.episodes, file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    sim = None
    current = None
    written = 0
    manifest = []
    try:
        for object_id, index, goal in jobs:
            width, height = goal.get("image_dimensions", [512, 512])
            hfov = float(goal.get("hfov", 90))
            key = (width, height, hfov)
            if key != current:
                if sim is not None:
                    sim.close()
                sim = build_sim(glb, config, int(width), int(height), hfov)
                agent = sim.initialize_agent(0)
                current = key
            state = habitat_sim.AgentState()
            state.position = np.asarray(goal["position"], dtype=np.float32)
            state.rotation = np.asarray(goal["rotation"], dtype=np.float32)
            agent.set_state(state)
            rgb = sim.get_sensor_observations()["color_sensor"][:, :, :3]
            name = "%s_%s_%02d.jpg" % (scene, object_id, index)
            Image.fromarray(rgb).save(os.path.join(args.out, name), quality=95)
            manifest.append(
                {
                    "scene": scene,
                    "object_id": object_id,
                    "image_index": index,
                    "file": name,
                    "object_coverage": goal.get("object_coverage"),
                    "frame_coverage": goal.get("frame_coverage"),
                }
            )
            written += 1
            if written % 50 == 0:
                print("  %d/%d image goals" % (written, len(jobs)), flush=True)
    finally:
        if sim is not None:
            sim.close()

    with open(os.path.join(args.out, "manifest.json"), "w") as handle:
        json.dump(
            {"format": "fact3r-goat-image-goals", "version": 1,
             "scene": scene, "episodes": args.episodes, "goals": manifest},
            handle,
            indent=2,
        )
    print("rendered %d image goals -> %s" % (written, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
