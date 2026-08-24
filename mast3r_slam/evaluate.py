import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    confs = []
    kf_ids = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        conf = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        valid = conf > c_conf_threshold
        pointclouds.append(pW[valid])
        colors.append(color[valid])
        # Kept per point, not just used as a cutoff. The threshold above is a
        # single global number inherited from the viewer's slider default, and
        # downstream consumers (the occupancy grid) want to filter on their own
        # criterion; they cannot if the value is discarded here.
        confs.append(conf[valid])
        # Which keyframe saw this point. Known here for free because the loop is
        # per keyframe, and impossible to recover from the merged cloud
        # afterwards -- which is why occupancy_grid.py has to approximate the
        # observing pose by the nearest one when it casts free-space rays.
        kf_ids.append(np.full(int(valid.sum()), i, dtype=np.uint16))
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)
    confs = np.concatenate(confs, axis=0)
    kf_ids = np.concatenate(kf_ids, axis=0)

    save_ply(savedir / filename, pointclouds, colors, confs, kf_ids)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors, confs=None, kf_ids=None):
    """Write XYZ+RGB, plus per-point confidence and source keyframe if given.

    `conf` and `kf_id` are extra named properties. Readers that ask for fields
    by name (plyfile, CloudCompare, Open3D) are unaffected; anything assuming a
    fixed 15-byte stride is not, so parse the header rather than hard-coding it.
    """
    colors = colors.astype(np.uint8)
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if confs is not None:
        fields.append(("conf", "f4"))
    if kf_ids is not None:
        # u2 caps at 65535 keyframes, orders of magnitude above any real run,
        # and costs half what u4 would over a cloud of tens of millions.
        fields.append(("kf_id", "u2"))

    pcd = np.empty(len(points), dtype=fields)
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    if confs is not None:
        pcd["conf"] = confs
    if kf_ids is not None:
        pcd["kf_id"] = kf_ids
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
