"""Chessboard calibration for the rover camera.

Records or reads chessboard views, solves for intrinsics + distortion, and writes a
YAML file in the format main.py expects for --calib (see config/intrinsics.yaml).

Capture views live from a camera:
    python scripts/calibrate_camera.py --camera 0 --out config/rover_intrinsics.yaml
    (press SPACE to grab a view, q when done -- aim for 20+ views)

Or solve from images you already recorded on the rover:
    python scripts/calibrate_camera.py --images 'calib_shots/*.jpg' --out config/rover_intrinsics.yaml

--board is the number of INNER corners, not squares: a 10x7 square board is 9x6.
"""

import argparse
import glob
import pathlib

import cv2
import numpy as np


def find_corners(gray, board):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def collect_from_camera(camera, board, square_size):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")

    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2) * square_size

    obj_points, img_points, size = [], [], None
    print("SPACE captures a view, q finishes. Tilt the board between views.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = find_corners(gray, board)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, board, corners, found)
        cv2.putText(
            view,
            f"captured: {len(obj_points)}  {'BOARD OK' if found else 'no board'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        cv2.imshow("calibration", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"captured view {len(obj_points)}")

    cap.release()
    cv2.destroyAllWindows()
    return obj_points, img_points, size


def collect_from_images(pattern, board, square_size):
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2) * square_size

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no images matched {pattern}")

    obj_points, img_points, size = [], [], None
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = find_corners(gray, board)
        print(f"{'ok  ' if found else 'skip'} {path}")
        if found:
            obj_points.append(objp)
            img_points.append(corners)

    return obj_points, img_points, size


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="camera index to capture from")
    src.add_argument("--images", help="glob of chessboard images, e.g. 'shots/*.jpg'")
    parser.add_argument(
        "--board",
        default="9x6",
        help="inner corners as WxH (a 10x7-square board is 9x6)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="square side length; only affects extrinsics, not the intrinsics we write",
    )
    parser.add_argument("--out", default="config/rover_intrinsics.yaml")
    args = parser.parse_args()

    w, h = (int(v) for v in args.board.lower().split("x"))
    board = (w, h)

    if args.camera is not None:
        obj_points, img_points, size = collect_from_camera(
            args.camera, board, args.square_size
        )
    else:
        obj_points, img_points, size = collect_from_images(
            args.images, board, args.square_size
        )

    if len(obj_points) < 8:
        raise SystemExit(f"only {len(obj_points)} usable views; need 8+, 20 is better")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None
    )

    errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        errors.append(cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected))
    mean_error = float(np.mean(errors))

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.flatten()[:5]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# {len(obj_points)} views, RMS {rms:.4f}px, mean reproj {mean_error:.4f}px\n")
        f.write(f"width: {size[0]}\n")
        f.write(f"height: {size[1]}\n")
        f.write("# With distortion (fx, fy, cx, cy, k1, k2, p1, p2, k3)\n")
        f.write(
            f"calibration: [{fx:.4f}, {fy:.4f}, {cx:.4f}, {cy:.4f}, "
            f"{k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f}]\n"
        )

    print(f"\nviews used:   {len(obj_points)}")
    print(f"resolution:   {size[0]}x{size[1]}")
    print(f"RMS error:    {rms:.4f} px  (want < 0.5)")
    print(f"mean reproj:  {mean_error:.4f} px")
    print(f"fx fy cx cy:  {fx:.2f} {fy:.2f} {cx:.2f} {cy:.2f}")
    print(f"\nwrote {out}")
    print(f"use with:  python main.py --dataset <video> --config config/calib.yaml --calib {out}")


if __name__ == "__main__":
    main()
