# Rover Capture Runbook

From a webcam on the Pi to a point cloud on the workstation.

Every command below is tagged with the machine it runs on — that's the thing most easily got wrong.

| Tag | Machine | Notes |
|---|---|---|
| **[PI]** | The rover's Raspberry Pi | Reached with `ssh pi@diana.local`. Holds the webcam at `/dev/video0`. Needs only `ffmpeg` and `v4l-utils` — none of this repo, no conda. |
| **[I3D]** | Your desktop, RTX 4090 | Prompt reads `nahar4@i3d`. Always with `conda activate mast3r-slam` in `~/Gazania/MASt3R-SLAM`. |

**Hardware, as verified 2026-08-20.** Logitech Brio 100 on `/dev/video0`, fixed focus (no
focus controls exist — focal length physically cannot drift). MJPEG to 1920x1080 @30fps; YUYV
collapses to 5fps above 640x480, so always MJPEG. Record at **1280x720**: `resize_img` forces the
long edge to 512, so 4:3 modes are centre crops that trade 33% of horizontal FOV for finer
sampling — verified by pixel comparison, not assumed. Pi is `diana.local`, key auth installed.

**Steps 0–1 are already done.** The environment is installed and verified — the TUM sequence ran at ~14 FPS and produced 51 keyframes and an 8.96 M-point cloud. Start at Step 2.

---

## Prerequisite — getting onto the Pi

You SSH **from the workstation**. The Pi's physical location is irrelevant — it needs no monitor, keyboard, or mouse. Three conditions:

1. **Same network.** Desktop and Pi must see each other.
2. **SSH enabled on the Pi.** The one thing you can't do over SSH. Either run `sudo raspi-config` → Interface Options → SSH with a monitor attached once, or mount the SD card's boot partition here and create an empty file named `ssh` on it.
3. **Know the address.** Try `ssh pi@diana.local` first. If mDNS doesn't resolve, check the router's client list or scan with `nmap -sn 192.168.1.0/24`.

Save yourself repeated password prompts:

```bash
# [I3D]
ssh-copy-id pi@diana.local
```

> **The rover moves, so it's on Wi-Fi, and Wi-Fi drops.** If the SSH session dies mid-recording, `ffmpeg` is killed and the take is lost. Start any recording inside `tmux`:
>
> ```bash
> # [PI]
> tmux new -s rec
> # ...run the ffmpeg command...
> # Ctrl+B then D to detach; reconnect later with: tmux attach -t rec
> ```

---

## Step 2 — Set up and lock the camera  `[PI]`

SSH into the rover. Everything in this step is typed into that session.

```bash
# [I3D] open a session on the rover
ssh pi@diana.local
```

```bash
# [PI] one-time setup
sudo apt install -y ffmpeg v4l-utils
```

### Find the camera

```bash
# [PI]
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext   # which sizes do MJPEG
v4l2-ctl -d /dev/video0 --list-ctrls         # which controls exist
```

Use **MJPEG**, never YUYV. YUYV at 720p saturates USB 2.0 and collapses to about 5 fps; MJPEG gives a clean 30 and costs the Pi no encoding.

### Free the camera from ROS first

`rover-camera.service` runs a ROS 2 `v4l2_camera_node` that holds `/dev/video0` exclusively.
`ffmpeg` will fail with `Device or resource busy` until it is stopped, and because the unit is
`Restart=always`, killing the process is useless — systemd returns it in 3 seconds.

```bash
# [PI]
sudo systemctl stop rover-camera.service     # ... and start it again when you are done
```

### Lock exposure, white balance, gain

The single biggest quality lever. Run `scripts/lock_camera.sh` — it probes for whichever control
spellings this kernel uses, switches every auto mode off *before* writing the manual value (the
reverse order is silently discarded while auto still owns the control), and prints a readback.

```bash
# [I3D] once
scp scripts/lock_camera.sh pi@diana.local:~/
```

```bash
# [PI] before every recording
./lock_camera.sh /dev/video0 100
```

Verified holding under real capture: 407 frames at 1280x720, `gain`, `exposure_time_absolute`
and `white_balance_temperature` all constant throughout.

**Exposure must match your mains frequency.** Light flickers at twice the supply frequency, so
the exposure has to be a whole number of flicker periods or brightness pulses between frames —
which is poison for a feature matcher. Units are 100 µs; this camera accepts 5–2500.

| Mains | Flicker period | Use |
|---|---|---|
| 50 Hz | 10 ms | **100**, 200 |
| 60 Hz | 8.33 ms | 83, 167, 250 |

The script defaults to 50 Hz. On 60 Hz mains: `MAINS=60 ./lock_camera.sh /dev/video0 167`.

Too dark? Raise gain (`GAIN=96 ./lock_camera.sh …`), never exposure. Noise averages out across a
keyframe; motion blur destroys the corner structure MASt3R matches on and nothing recovers it.

> Settings are lost on replug, reboot, and USB power blips. There is no persistence in v4l2 —
> treat the lock as part of the record command, not as one-time setup.

### Aim the camera

You don't need live video to frame a shot. Grab one still, pull it over, look at it, adjust, repeat.

```bash
# [PI]
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 \
       -i /dev/video0 -frames:v 1 -y /tmp/test.jpg
```

```bash
# [I3D]
scp pi@diana.local:/tmp/test.jpg /tmp/ && xdg-open /tmp/test.jpg
```

Target framing: camera rigid, roughly level, 30–60 cm up, pointed forward and slightly down so the floor fills the lower third. That floor is what the occupancy grid's ground plane gets fitted to in Step 8.

---

## Step 3 — Calibrate: capture on the Pi, solve on the workstation  `[PI]` `[I3D]`

> **DEFERRED — skipping this for now.** MASt3R-SLAM runs uncalibrated; that's its headline
> feature, and `use_calib: False` is already the default in `config/base.yaml`. Jump to Step 4
> and drop the `--calib` flag in Step 6.
>
> Know what you're trading. `Intrinsics.from_calib` returns `None` when `use_calib` is false
> (`mast3r_slam/dataloader.py:300`), so **no undistortion is applied at all** — raw lens
> distortion reaches the network. Wide-FOV webcams bend straight walls into curves, which is
> the one artefact an occupancy grid cannot tolerate. Good enough to prove the pipeline; come
> back here before trusting a grid for navigation.
>
> Step 2 is **not** optional either way — locking focus matters more without calibration, since
> autofocus shifting focal length mid-sequence breaks the constant-intrinsics assumption in both
> modes. Lock the camera now and today's footage stays re-runnable with `--calib` later.

Do this once per camera, per resolution. Print a chessboard and tape it flat to something rigid. A 10×7-square board has **9×6 inner corners** — the script wants inner corners.

### 3a · Record the board  `[PI]`

Same resolution and same locked settings you'll use for real. Re-run the `v4l2-ctl` lock command first.

```bash
# [PI]
mkdir -p ~/calib_shots
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
       -i /dev/video0 -t 40 -c copy ~/calib_shots/calib.mkv
```

During those 40 seconds, present the board at ~20 distinct poses: near and far, all four corners of the frame, tilted left, right, up, down. Tilt matters more than position — a set of head-on views can't separate focal length from distance.

```bash
# [PI] split to stills
ffmpeg -i ~/calib_shots/calib.mkv -vf fps=2 ~/calib_shots/%03d.jpg
```

### 3b · Solve  `[I3D]`

```bash
# [I3D]
cd ~/Gazania/MASt3R-SLAM
conda activate mast3r-slam

scp -r pi@diana.local:~/calib_shots ./

python scripts/calibrate_camera.py \
       --images 'calib_shots/*.jpg' \
       --board 9x6 \
       --out config/rover_intrinsics.yaml
```

You want **RMS below 0.5 px**. Higher means blurry shots or too little variation in board angle — redo the capture. The script needs 8 usable views minimum and refuses below that.

> **Don't use the script's `--camera` mode over SSH.** It opens an OpenCV preview window and needs a display. The `--images` path above is the one that works headless.

---

## Step 4 — Record the run  `[PI]`

### The one-command way — drive and record together  `[I3D]`

Everything in Steps 4 and 5 in a single command, run from the workstation:

```bash
# [I3D]
./scripts/record_rover_run.sh                       # or: ... 172.22.217.52
./scripts/record_rover_run.sh --name lab_loop --slam
```

It stops `rover-camera.service`, locks the camera, starts `ffmpeg` on the Pi,
hands you the keyboard, and when you Ctrl+C out of teleop it ends the take,
pulls the `.mkv` over and converts it to `datasets/rover/<name>/%06d.png` —
ready for Step 6. `--slam` runs `main.py` on it immediately.

| Flag | Meaning |
|---|---|
| `--name NAME` | Dataset folder name. Default is the next free `runN`. |
| `--max SECONDS` | Hard cap on the take (default 900), so a crashed script can't fill the SD card. |
| `--slam` | Run `main.py` on the result once the transfer finishes. |
| `--keep-mkv` | Keep the `.mkv` here as well as on the Pi. |
| `--no-convert` | Pull the video, skip the PNG conversion. |

`PI_PASS=…` makes the `sudo systemctl stop` step non-interactive **and** makes
the script restart `rover-camera.service` on the way out. Without it you get a
password prompt and the service is left stopped with the restore command printed.

**No live preview while recording, by design.** ffmpeg's `tee` muxer cannot
both serve MJPEG-over-HTTP and write a file — `listen` breaks stream mapping —
and only one process can hold `/dev/video0`. Frame the shot with the
single-still trick in Step 2 before you start.

**Verified 2026-08-21** on the Pi: the pidfile names the real `ffmpeg` pid
(`setsid bash -c 'echo $$ …; exec ffmpeg …'`), the readiness poll fires in ~3 s,
SIGINT finalises a valid matroska rather than truncating it, and the converted
PNG folder loads as `RGBFiles` at 1280x720. The drive-and-capture leg itself is
still untested — the ESP32 was unplugged.

> **`diana.local` is unreliable.** It resolved, then stopped resolving minutes
> later while the Pi stayed perfectly reachable at **172.22.217.52**. Pass the IP
> as the first argument to any of these scripts when mDNS lets you down.

### The manual way


```bash
# [PI]
v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0 \
  -c auto_exposure=1 -c exposure_time_absolute=250 \
  -c white_balance_automatic=0

ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
       -i /dev/video0 -t 60 -c copy ~/run1.mkv
```

### Optional — watch while you drive

Record and stream at once, so the preview never costs you the take. Then open `http://diana.local:8090` in a browser on the workstation. Expect a second of lag; fine for framing, useless for judging blur.

```bash
# [PI]
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
       -i /dev/video0 -t 60 -c copy -f tee \
       "[f=matroska]/home/pi/run1.mkv|[f=mpjpeg:onfail=ignore]http://0.0.0.0:8090"
```

### How you drive decides whether this works

Monocular SLAM recovers geometry from parallax. The camera must *translate*. These aren't style preferences — each one is a failure mode.

| | | |
|---|---|---|
| **Speed** | ~0.2 m/s | Slower than feels necessary. |
| **Rotation** | Never spin in place | Zero parallax — the classic way to lose tracking. Turn along an arc. |
| **Scene** | Textured, lit | A blank white wall gives the matcher nothing to match. |
| **Path** | Return to start | Closing a loop lets global optimisation cancel drift. |
| **Duration** | 30–60 s at first | Memory grows with keyframes; keep the debug cycle fast. |
| **Mount** | Rigid, no vibration | Every wobble is motion blur the network can't match through. |

---

## Step 5 — Bring the video over and convert it  `[I3D]`

You pull from the workstation, so this is typed here, not on the Pi.

```bash
# [I3D]
cd ~/Gazania/MASt3R-SLAM
scp pi@diana.local:~/run1.mkv /tmp/
vlc /tmp/run1.mkv        # watch it — sharp when moving? exposure stable?
```

Judge the footage *here*, on a real screen. Sharpness under motion and exposure stability are the two things a laggy network preview will lie to you about.

### Convert to a PNG frame folder

```bash
# [I3D]
mkdir -p datasets/rover/run1
ffmpeg -i /tmp/run1.mkv -vsync 0 datasets/rover/run1/%06d.png
```

Folder input beats MP4 input here: without `torchcodec` installed, the MP4 loader seeks frame-by-frame with OpenCV, which is slow and unreliable on inter-frame-compressed video. A PNG folder is deterministic.

> **Two ways this fails silently.** The folder loader globs `*.png` only — JPEGs give you an empty dataset, not an error. And the loader dispatches on path *components*, so a folder named `tum`, `euroc`, `eth3d`, `7-scenes`, `realsense`, or `webcam` gets routed to the wrong reader. `datasets/rover/run1` is safe.

---

## Step 6 — Run the SLAM  `[I3D]`

```bash
# [I3D]
conda activate mast3r-slam
cd ~/Gazania/MASt3R-SLAM

# uncalibrated (Step 3 deferred) — this is the one to run now
python main.py --dataset datasets/rover/run1 --config config/base.yaml

# once calibrated, same footage, better geometry
python main.py --dataset datasets/rover/run1 \
               --config config/base.yaml \
               --calib config/rover_intrinsics.yaml
```

- Use `python`, not `python3` — the latter can resolve to the system interpreter and hand you a confusing `ModuleNotFoundError`.
- `width` and `height` in `rover_intrinsics.yaml` must match your frames exactly, or the undistortion map is wrong.
- Too slow, or the rover crawled? Raise `dataset.subsample` in `config/base.yaml` to 2 or 3. Fewer frames, wider baselines — often a *better* result, not just a faster one.
- The viewer's `C_conf_threshold` slider (default 1.5) sets the confidence cut applied to the saved cloud, using whatever value is set when you quit. Raise for cleaner and sparser, lower for denser and noisier.
- Add `--no-viz` only for headless runs; normally you want to watch the map build.

---

## Step 6b — Live mode: drive and reconstruct together  `[I3D]`

**Everything that thinks runs on the workstation.** The Pi only serves its
camera as MJPEG and relays `/cmd_vel` to the ESP32 — no SLAM, no teleop, no
processing of any kind.

```bash
# [I3D] terminal 1 — brings up the Pi over SSH, then runs the SLAM here
./scripts/launch_rover_slam.sh
```

```bash
# [I3D] terminal 2 — keyboard driving, also on this machine
./scripts/teleop_rover.sh
```

Both run on the GPU box. Teleop publishes `geometry_msgs/Twist` on `/cmd_vel`
over plain DDS; the Pi's micro-ROS agent picks it up with nothing extra
running there. **Verified**: a Twist published from the workstation arrived on
the Pi as `linear.x: 0.2, angular.z: 0.1`, with no zenoh bridge involved —
both machines sit on `172.22.217.0/24`, so multicast discovery works directly.

### Three things that will bite

**`python3` is the wrong interpreter here.** This shell's `python3` is conda's
3.12 (from `poseE`); ROS Humble's C extensions are built for 3.10 and fail with
a `_rclpy_pybind11` import error. `teleop_rover.sh` calls `/usr/bin/python3.10`
explicitly for exactly this reason.

**Teleop starts at 0.4 m/s.** Tap `z` twice for the ~0.2 m/s this pipeline
wants. Wheel-speed caps also apply: the reference launcher uses 0.15 m/s.

**Rotation below 0.174 rad/s does nothing at all.** The firmware zeroes any
per-wheel target under `VEL_DEADBAND_MS = 0.03` m/s, and a pure rotation gives
`angular_z * TRACK_WIDTH_M/2` with `TRACK_WIDTH_M = 0.345`. Command less than
that and the rover silently sits still — it looks like a dead motor, but the
firmware is discarding the command.

### Why MJPEG-over-HTTP rather than the `/image_raw` topic

`v4l2_camera` publishes **raw** `sensor_msgs/Image`. At 1280x720x3 @ 30 Hz that
is ~83 MB/s, which no WiFi link carries, and this node advertises no
`/compressed` transport — the Pi's topic list is only `/image_raw` and
`/camera_info`. The camera already emits MJPEG, so streaming it costs the Pi
no re-encoding at all.

Live mode **streams only; it does not archive the raw video.** ffmpeg's `tee`
muxer cannot both serve MJPEG-over-HTTP and write a file — the `listen` option
breaks stream mapping (verified, not assumed). MASt3R still saves its `.ply`,
trajectory and keyframes as `logs/rover_live_<timestamp>.*`. Use Steps 4–6 when
you want re-runnable footage.

---

## Step 7 — Check the result before building on it  `[I3D]`

| Output | Contents |
|---|---|
| `logs/run1.txt` | One line per keyframe: `timestamp x y z qx qy qz qw`, world←camera |
| `logs/run1.ply` | Dense coloured point cloud, XYZ + RGB — your obstacles |
| `logs/keyframes/run1/` | The undistorted keyframe images |

Open the `.ply` in CloudCompare or Open3D and check three things:

1. **Are flat things flat?** Banana-curved corridors mean drift.
2. **Did the loop close?** If you drove back to the start, trajectory start and end should coincide.
3. **Is the scale right?** It will not be. MASt3R-SLAM's metric head does not recover real-world scale — measured across TUM, HM3D and ReplicaCAD it scatters from 0.45x to 2.05x, and it scatters on real camera footage just as much as on renders. Do not eyeball a factor: measure your camera's height above the floor once, to the centimetre, and pass it as `--cam-height` in Step 8. That recovers the scale from the reconstruction itself. On HM3D it lands within a median 5% of truth, against 27% uncorrected.

> **Compare against the reference run.** `logs/rgbd_dataset_freiburg1_room.ply` from the verified TUM test is what "good" looks like. Judge your first rover run against it rather than against expectation.

---

## Step 8 — Build the occupancy grid  `[I3D]`

```bash
# [I3D]
python scripts/occupancy_grid.py --ply logs/run1.ply --traj logs/run1.txt \
       --out logs/grid_run1
```

Writes `grid_run1.pgm` + `.yaml` (the ROS `map_server` pair — directly loadable
by nav2) and `.npy` (int8, ROS convention: 0-100 probability, -1 unknown).
Depends only on numpy + plyfile, both already in the env; it deliberately does
NOT use open3d, since installing it risks pulling a numpy 2.x that breaks the
dataset loaders.

> **The world frame is the first camera, not gravity.** Origin is wherever the
> camera sat on frame 0, in camera convention: +x right, **+y down**, +z forward.
> The floor is an arbitrary plane, *not* `z = 0`. The script RANSACs it, using
> +y as a gravity prior and rejecting hypotheses more than `--gravity-tol`
> degrees off it — without that prior, RANSAC happily picks a wall or a desktop
> and silently tilts the entire map.

**Read the inlier fraction it prints.** Below 15% it warns, and you should
believe the warning: it means the floor is poorly observed, and every height
threshold downstream is then measured from a plane that isn't the floor. The
fix is at capture time — pitch the camera down so the floor fills the lower
third of the frame.

`--cam-height` makes this failure mode much less likely: given the mount height
the floor plane is known rather than fitted, so RANSAC cannot pick a table top
instead. On HM3D scene 00801 the plane vote put the floor 0.95 m below the
camera when it was 1.72 m down, which lifted the whole obstacle band by 0.55 m
and reclassified every knee-high obstacle as floor. With `--cam-height` the same
cloud reports 1.51 m, steady to +/-0.03 m across the run.

Worth knowing what these knobs do before turning them:

| Flag | Default | Meaning |
|---|---|---|
| `--voxel` | 0.03 | Downsample size. 9 M points becomes ~240 k; full resolution buys no accuracy. |
| `--res` | 0.05 | Grid cell size. |
| `--min-h` / `--max-h` | 0.10 / 1.50 | Height band counted as obstacle. Below is floor, above is overhang the rover drives under. |
| `--cam-height` | — | **Set this.** Camera height above the floor in metres, measured on the rover. Recovers metric scale and pins the floor plane; without it the grid is in arbitrary units and `--min-h`/`--max-h` are measured from a RANSAC plane that may not be the floor. |
| `--scale` | 1.0 | Extra manual scale, applied after `--cam-height`. Normally leave alone. |
| `--no-floor-support` | off | Clears space on line of sight alone. Raises recall, lowers precision — the wrong trade for navigation, so gating is on by default. |
| `--gravity-tol` | 35 | Max degrees the floor normal may sit off +y. |

Free space is ray-cast from the nearest keyframe pose to each occupied cell,
accumulated in log-odds. Since the `.ply` carries no point-to-keyframe
association, nearest-pose is an approximation — a conservative one, as it only
clears space some pose genuinely had line of sight to.

**Validated end to end on the TUM reference cloud** (8.96 M points, ~5 s), and
the output PGM/YAML/npy are format-checked. The plane fit itself could *not* be
validated there: `fr1/room` is a handheld office sequence whose densest height
slice holds just 1.9% of near-path points — it has no observable floor. Your
rover footage, with a level mount aimed slightly down, is the first real test of
that stage, so check the inlier fraction on your first run.

## Standing caution on the environment

Don't run a bare `pip install` in `mast3r-slam` without checking what it pulls in. Installing mast3r dragged in numpy 2 and OpenCV 5 over the repo's own `numpy==1.26.4` pin, which would have broken the dataset loaders (`np.unicode_` was removed in numpy 2). If an install ever reports `Successfully installed … numpy-2.x`:

```bash
pip install "numpy==1.26.4" "opencv-python==4.10.0.84"
```

## SafeDiffuser + DSTT on the DA2 BEV grid

Planning on `logs/rover/grids/mpl_da2_final.npy` (285x238 @ 0.05 m, free 20.1%,
occupied 23.9%, unknown 55.4%). The planner is vendored at
`thirdparty/safediffuser/` — see its README for provenance — so this runs from
the repo root; `--planner-root` still accepts an external checkout.

```bash
cd thirdparty/safediffuser
conda activate mast3r-slam
G=../../logs/rover/grids/mpl_da2_final.npy

python scripts/plan_hm3d.py --grid $G --n-plans 8 --seed 0 \
  --unknown-slack 0.20 --radius-margin 0.05 --min-separation 3.0

python scripts/plan_hm3d.py --grid $G --n-plans 8 --seed 0 \
  --unknown-slack 0.20 --radius-margin 0.05 --min-separation 3.0 --no-guidance \
  --out ../MASt3R-SLAM/logs/rover/plans

python scripts/plot_tube.py --grids $G --seed 0 \
  --unknown-slack 0.20 --radius-margin 0.05 --min-separation 3.0
```

### Why these three flags differ from the HM3D defaults

**`--unknown-slack 0.20`, not 0.50.** This is the setting that decides whether the
run means anything. `clearance = min(sd_hard, sd_soft + slack) - robot_radius`, so a
plan may penetrate unobserved space by `slack - robot_radius` and still score
`collision_frac = 0`. At the 0.50 default that is 0.30 m of licence: goals were
sampled *inside* never-observed cells and the trajectories ran 20 m around the
outside of the building, skimming the unknown halo. Only 39% / 23% of the first two
trajectories lay in observed free cells, and nothing in the printed metrics said so.
Setting slack to `robot_radius` makes the licence exactly zero. Verified afterwards:
**100.0% of all 8 DSTT trajectories lie in observed free cells, 0.0% in walls.**

**Exterior flood fill stays ON here** — unlike the pilot `mpl` grid, where it claimed
88% of the map. On this grid it costs only 18.6 -> 16.9 m2 of navigable area, and it
is what stops plans escaping into the outdoor halo. Do not carry `--no-exclude-exterior`
over from the pilot run. (Note `mpl_da2_final_view.png` *was* drawn with that flag,
which is why its middle panel reports exterior 0.0% while the planner reports 44.5%.)

**`--radius-margin 0.05`, not 0.15.** Swept at slack 0.20 (`sweep_margin.py`, 8 fixed
problems):

| margin | coll% | min_clear | dev_mean |
|---|---|---|---|
| 0.02 | 0.59% | 0.001 | 0.205 |
| **0.05** | **0.00%** | **0.029** | **0.200** |
| 0.10 | 0.00% | 0.029 | 0.182 |
| 0.15 | 0.00% | 0.029 | 0.164 |
| 0.20 | 0.00% | 0.029 | 0.147 |

`min_clear` saturates at 0.029 m from 0.05 upward — it is pinned by A* pinch points
where the cap `d - margin` is already negative, so a larger margin buys **no** safety
and only closes the tube onto the centerline (`dev_mean` falls monotonically, i.e. the
diffusion prior contributes less). 0.05 is the smallest margin with 0% collision.

**`--min-separation 3.0`**, because the navigable component is 5.8 m across its short
axis at slack 0.20; the 4.0 default is sized for HM3D houses.

`sweep_margin.py` and `plot_tube.py` both hardcoded `min_separation=4.0`, and
`plot_tube.py` had no `--unknown-slack` at all — it would have drawn the tube figure
on the slack-0.50 navigable set, i.e. a different map than the plans. Both now take
the flags.

### Results

| | collision | min clearance | in observed free |
|---|---|---|---|
| DSTT | **0.0%** | +0.031 m | **100.0%** |
| bare prior (`--no-guidance`) | 82.7% | -1.189 m | 27.2% |

`max_tube_excess_m` is 1.2e-6 (float32 zero), so containment holds. Do not score it
with a bare `dev > r` test — guidance is zero strictly inside the tube and an exact
projection outside, so its fixed point *is* the boundary.

Unlike the pilot grid, **the prescribed-time contraction is no longer inert**: the four
`r_j` curves in `safety_tubes_profiles.png` are visibly nested wherever clearance
exceeds ~0.42 m, and collapse onto the cap only at the three pinch points.

### The knob overlap, quantified

`unknown_slack` and `radius_margin` both meter distance to unobserved space, so they
partly do the same job. Measured over 8 centerlines, the binding term in
`min(sd_hard, sd_soft + slack)` is the **unknown boundary on 60.7% of the horizon**
and a real wall on 39.3% (at slack 0.50 it is 71.1% unknown). So on most of the path
`radius_margin` is a *second* buffer against unknown, stacked on `slack`.

They are not redundant — slack governs where the **centerline** may go (it defines the
navigable set A* searches), margin governs the **tube width** around it — but the
overlap is why margin saturates above. Resolution: **let `unknown_slack` be the single
knob for unobserved space and keep `radius_margin` at the smallest 0%-collision value.**
Raising margin to "get more safety" here does nothing but starve the prior.

Consequence for reading the numbers: because unknown binds 61% of the time,
`min_clearance_m` is **not** a wall clearance. It is distance to the nearer of a wall
or the frontier of what the camera actually saw. A plan with 0.03 m "clearance" is
typically 0.03 m from unobserved space, not 0.03 m from an obstacle.

---

## The full pipeline: video + odometry -> a trajectory toward a named object

Everything above is a stage. `scripts/run_rover_pipeline.py` is the flow between them.

```bash
# [I3D]
python3 scripts/run_rover_pipeline.py \
  --run logs/rover/pipeline/mpl_20260826 \
  --video /home/nahar4/Gazania/MPL/manual_drive_20260826_180408.mp4 \
  --odom  /home/nahar4/Gazania/MPL/odom_home_session_20260826_180408.csv \
  --time-offset 26.80 \
  --query "a 3D printer"
```

| stage | env | what it does |
|---|---|---|
| `frontend` | SAM2 + mast3r-slam | `run_fact3r_real_uot.sh`: frames -> SAM2 proposals -> SigLIP index -> MASt3R 2D matches -> UOT association. Resumes each of those itself. |
| `fuse` | SAM2 | Depth-Anything-V2 metric depth + odometry -> occupancy grid **and** semantic grid. One depth pass, so one stage. |
| `locate` | SAM2 | SigLIP text query -> the winning entity's footprint. |
| `goal` | mast3r-slam | that footprint -> a reachable world `(y, x)`. |
| `plan` | mast3r-slam | SafeDiffuser + DSTT to that goal. |

`--stage NAME` runs one; `--from` / `--through` run a range; the five front-end step
names (`frames`, `proposals`, `embed`, `match`, `associate`) are accepted and select
`frontend`. Resume is the default — a stage whose outputs are newer than its inputs is
skipped — and `--force` re-runs the selected stages and nothing upstream. Every run
writes `pipeline.json` with the resolved parameters, per-stage timings, and the git SHA.

Pass `--frontend-dir` to adopt an existing stages 1–5 output rather than spending hours
rebuilding one.

Everything the pipeline needs is in this one repo. The SafeDiffuser + DSTT planner is
vendored at `thirdparty/safediffuser/` (MIT, upstream `Weixy21/SafeDiffuser`); only the
two model checkpoints live outside git — the 2.75 GB MASt3R matcher under `checkpoints/`
and the 29.6 MB maze2d prior under `thirdparty/safediffuser/logs/pretrained/`.

### `--time-offset` is required, and that is deliberate

The keyframe timestamps are on the **video** clock; `_load_odometry` re-bases odometry to
its own first row. On the 2026-08-26 capture those differ by **26.80 s** — logging started
~27 s before the camera. Measure it with `scripts/find_time_offset.py`, never assume it.

It fails *silently*. Video 0–467 s sits entirely inside odometry 0–494.8 s, so the
`timestamp < odom_t[0]` guard never trips: `skipped_frames` stays 0 and the PNG renders
fine while every pose is wrong by a median **2.65 m and 89.2° of yaw**. What it costs:

| grid | free | occupied | unknown |
|---|---|---|---|
| `logs/rover/depth_semantic/map.npy` (offset 0) | **0.9%** | 39.3% | 59.7% |
| `logs/rover/pipeline/mpl_20260826/map.npy` (offset 26.80) | **15.4%** | 26.7% | 57.1% |
| `logs/rover/grids/mpl_da2_final.npy` (reference) | 20.1% | 23.9% | 55.4% |

0.9% free leaves A* nothing to search. `build_depth_semantic_bev.py` now **refuses to
run** when no offset is given and the two stream durations disagree by more than
`--clock-tolerance` (2 s); `--assume-synchronised` overrides it for captures that really
do share a clock. The offset used is recorded in the map manifest as `time_offset_seconds`.

### The query only ranks entities that are on the map

`build_semantic_grid` awards each BEV cell to a single winning entity, so most groups win
none: **1,532 of 14,555** hold even one cell on this map. Ranking all of them returns
matches with no position at all — every one of the top five for `"a 3D printer"` occupied
zero cells. `query_semantic_bev.py` and `resolve_semantic_goal.py` now restrict to
on-map entities by default and report how many were dropped; `--include-unmapped` keeps
the unrestricted behaviour for retrieval evaluation, where recall over the whole memory
is the point.

Expect ties. Consecutive frames of one object that UOT never merged become separate
entities with near-identical prototypes, so eight entities share the top score here.
`--tie-break cells` (the default) takes the one with the most map support.

### The centroid of an entity is not a goal

An object's own footprint is occupied by construction, and the centroid of a U-shaped or
split entity lands in the wall between its parts or in unobserved space. So
`project_semantic_goal.py` treats it as a seed: it projects to the nearest cell that is
navigable **and** in the same component, using `HM3DMap`'s clearance and
`planner.largest_component` rather than a second definition of either. Among cells the
same distance away to within a robot radius it prefers the one with the most clearance —
projecting onto a region always lands on its boundary, which is exactly where clearance
is worst.

`--max-projection` (default 2.0 m) is the honesty check: past it, the goal is no longer
about the entity that was asked for and the stage fails with the distance rather than
returning a point. On this capture `"a 3D printer"` needs **4.0 m**, because the rover
never drove within 2.75 m of the printer — its own track stops at x = 3.81 m and the
printer is at x = 6.46 m, so it was only ever seen from across the room.

### `plan_hm3d.py --start / --goal`

Endpoints in world **`(y, x)`** metres — the order `HM3DMap.cell_to_world` returns, *not*
the `(x, y)` of the grid yaml's `origin`. A swap produces a confident plan to the wrong
room and no error, so both are validated against the navigable set (and against each
other's connected component) before the checkpoint is loaded.

With neither flag the behaviour is exactly as before: `sample_endpoints`, same rng draws.
With one, the other is drawn from the same component via `planner.sample_partner`.

> Note that `--seed` seeds numpy only — endpoints and the PRM. The diffusion noise comes
> from torch's global generator, which nothing here seeds, so path lengths vary by a few
> hundred millimetres between identical runs. Collision, clearance, and goal error do not.
