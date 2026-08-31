# Running MASt3R-SLAM on HM3D

HM3D ships **3D scene meshes**, not RGB video, so it cannot be fed to MASt3R-SLAM
directly. The pipeline here renders a camera trajectory through each scene with
habitat-sim, then runs SLAM on the resulting image folder.

```
HM3D .glb + .navmesh  ──render_hm3d_traj.py──>  PNG sequence + GT poses
                                                        │
                                                  main.py (SLAM)
                                                        │
                                          trajectory .txt + dense .ply
                                                        │
                                        eval_hm3d.py / plot_hm3d_recon.py
```

## Data layout

Source data lives in `/home/nahar4/Gazania/hm3d/`, downloaded as Google-Drive
chunks that are each individually incomplete — scenes are missing their
`.basis.glb` until the three `versioned_data-*` chunks are merged. The merged
tree at `datasets/hm3d_root/` is built from symlinks (no extra disk) and holds
all **10 minival + 100 val** scenes plus the scene-dataset configs:

```
datasets/hm3d_root/
├── hm3d_annotated_basis.scene_dataset_config.json
├── minival/00800-TEEsavR23oF/{TEEsavR23oF.basis.glb,.basis.navmesh}
└── val/...
```

## 1. Render sequences

habitat-sim lives in the `habitat-vla` conda env (habitat-sim 0.3.3, headless
EGL — no display needed).

```bash
conda run -n habitat-vla python3 scripts/render_hm3d_traj.py --split minival
```

The camera walks a continuous geodesic tour over the navmesh: 0.04 m per frame,
heading changes rate-limited to 2°/frame (turned into in-place rotations), and
the tour closes a loop so global optimisation has a real loop closure to find.
Three details matter for robustness:

- **Waypoints are sampled from the largest navmesh island.** HM3D navmeshes are
  multi-storey and fragmented; raw `get_random_navigable_point()` returns points
  that may be unreachable from one another.
- **`min_leg` relaxes adaptively.** Small scenes (e.g. `00802`, 128 m² navigable)
  cannot offer long legs and would otherwise yield a single frame.
- **Tours are screened for mesh holes.** HM3D meshes are reconstructions with
  gaps; a camera passing one renders near-black, textureless frames that lose
  tracking. Candidate tours are probed and resampled if more than `--max-dark`
  (default 5%) of views are empty. `00805` needed 3 attempts.

Output per scene: `000000.png …`, `groundtruth.txt` (TUM format, habitat world
frame), and `meta.json` (intrinsics, tour length, dark fraction).

## 2. Run SLAM

```bash
conda run -n mast3r-slam bash scripts/eval_hm3d.sh              # calibrated
conda run -n mast3r-slam bash scripts/eval_hm3d.sh --no-calib   # intrinsics estimated
```

The render is an exact pinhole camera (square 512×512 at 90° HFOV → fx = fy =
cx = cy = 256, no distortion), written to `config/hm3d_intrinsics.yaml`, so the
calibrated path uses known intrinsics rather than estimating them. Results land
in `logs/hm3d/calib/<scene>.{txt,ply}`.

## 3. Evaluate

```bash
python3 scripts/eval_hm3d.py --run hm3d/calib
python3 scripts/plot_hm3d_recon.py --run hm3d/calib --scene 00809-Qpor2mEya8F
```

ATE is reported under both SE(3) alignment (scale fixed at 1 — meaningful
because the metric MASt3R checkpoint predicts real-world scale) and Sim(3)
alignment (which also reveals the scale actually recovered). Only translation is
compared: habitat's camera axes differ from the SLAM convention by a constant
right-multiplied rotation, which leaves camera positions untouched but would
corrupt a naive rotation comparison.

**`coverage` is the column to check first.** A run that loses tracking early
scores a flatteringly low ATE, because only the tracked prefix is ever compared
— before the mesh-hole fix, `00805` reported 0.034 m ATE over 19.7% of its
sequence. Only full-coverage runs are comparable, and the median row counts
them.

## 4. Occupancy grids (BEV maps)

```bash
conda run -n mast3r-slam  python3 scripts/hm3d_occupancy.py --run hm3d/calib
conda run -n habitat-vla  python3 scripts/eval_hm3d_occupancy.py --run hm3d/calib
```

A 2D occupancy grid *is* the BEV map — an orthographic projection onto the
fitted floor plane. There is no separate BEV stage; you would only add one for
learned semantic BEV, which navigation grids do not need.

`scripts/occupancy_grid.py` does the gridding (floor plane, height-band obstacle
slice, Bresenham free-space carving, ROS `map_server` output).
`hm3d_occupancy.py` drives it per scene and adds two things:

- **Scale correction, without ground truth.** `scripts/metric_scale.py` measures
  the scale from the known 1.5 m render camera height instead of taking it from
  `metrics.json`. This matters beyond tidiness: the ground-truth scale does not
  exist on a rover run, so the old path could never have shipped.

  It is also, unexpectedly, **more accurate than using the ground truth**. The
  three modes below were run under identical gridding — same floor plane, same
  storey splitting, same carving — so the only variable is where scale comes
  from:

  | `--scale-mode` | trajectory scale error | free prec | recall | IoU | occ prec |
  |---|---|---|---|---|---|
  | *(uncorrected)* | 27.1% | — | — | — | — |
  | `anchor` (default, no GT) | 5.4% | 0.483 | 0.620 | **0.380** | **0.858** |
  | `gt-global` | 2.1% | 0.488 | 0.596 | 0.348 | 0.854 |
  | `gt-profile` (oracle) | **0.3%** | 0.443 | 0.480 | 0.294 | 0.681 |

  (Grid columns here were measured before occlusion-aware carving and the
  per-cell evidence threshold, so they are lower than the current headline
  numbers. The comparison between the three modes is still like-for-like —
  all three ran under the same gridding as each other.)

  **The two rankings are inverted: the more metrically perfect the trajectory,
  the worse the map.** Two mechanisms, and both say the same thing.

  `gt-global` fits one Sim(3) scale to the whole trajectory, so it optimises
  *path alignment*. Once a reconstruction is warped that is a different number
  from the scale that makes *local geometry* metric, and an obstacle slice
  referenced to a floor plane depends entirely on the local one. The anchor
  measures the local quantity directly, which is why it wins despite being the
  less accurate estimate of the global scale.

  `gt-profile` is the oracle: a per-keyframe scale read straight off ground
  truth, applied with `metric_scale.deform`. It drives trajectory error to 0.3%
  — essentially perfect — and produces the worst map of the three. Re-scaling
  each point about its nearest keyframe breaks the mutual registration of a
  *fused* cloud: a wall seen from two keyframes at different scale factors is
  split into two walls. Perfect path, shredded geometry.

  The conclusion is the useful part: **scale is no longer what limits this map.**
  Perfect scale, handed over for free, makes it worse. What remains is the
  reconstruction's internal geometric consistency — drift and registration —
  which no post-processing step can repair.

  The anchor also pins the floor plane, which turns out to matter more than the
  scale for obstacle quality: given the camera height the floor is known rather
  than RANSAC-fitted, and RANSAC was getting it badly wrong (on `00801`, 0.95 m
  below the camera against a true 1.72 m, on 3.9% inliers). That shifted the
  whole `--min-h`/`--max-h` band up by 0.55 m.
- **Per-storey splitting.** Four tours climb stairs. Levels come from clustering
  camera heights by *density*, not by gaps — stairs make the height
  distribution continuous, so a gap-based split collapses both floors into one.

Storey heights come from the cameras' own up axis (`metric_scale.camera_up`),
**not** from a plane fit on the cloud. Fitting a global plane to find storeys is
unreliable: RANSAC can lock onto an upper floor or a ceiling, and `plane_basis`'s
"cameras sit above the floor" heuristic then orients up backwards. Measured
against ground truth that gave corr(cam_height, true_height) = **−0.998** on
`00809` and −0.495 on `00800` — the slab selection was running upside-down.
Averaging the cameras' image-down axis cannot invert like that. Every
`--scale-mode` uses it, so the comparison above varies only the scale source.

Output: 14 grids from 10 scenes in `logs/<run>/grids/` — `.pgm`/`.yaml`
(ROS map_server), `.npy` (int8, ROS convention), `.png` preview, and
`_vs_gt.png` (SLAM map / navmesh / agreement, all in habitat's raster).

### What the numbers mean

Read **DANGER** first. It is the fraction of obstacles the camera actually
observed that the map records as *free space* — the cells that drive a rover
into a wall. IoU says nothing about this, and for most of this project's life
nothing measured it at all.

Median over 12 real storeys (ramp artifacts excluded, see below):

| metric | value | |
|---|---|---|
| DANGER — observed obstacles marked free | **0.050** | lower is better |
| occupied recall — observed obstacles marked occupied | **0.947** | |
| occupied precision | 0.718 | |
| free precision (fair) | 0.937 | |
| free recall | 0.609 | |
| free IoU | 0.527 | |

How it got here, measured at each step:

| stage | DANGER | occ recall | prec (fair) | recall | IoU |
|---|---|---|---|---|---|
| ground-truth scale, RANSAC floor | *unmeasured* | *unmeasured* | 0.455 | 0.547 | 0.331 |
| metric anchor + pinned floor | 0.319 | 0.596 | 0.514 | 0.648 | 0.382 |
| + occlusion-aware carving | 0.083 | 0.907 | 0.795 | 0.452 | 0.373 |
| + rays from observing keyframes | 0.136 | 0.856 | 0.889 | 0.651 | 0.578 |
| **+ per-cell evidence (default)** | **0.050** | **0.947** | **0.937** | 0.609 | 0.527 |

IoU is a poor guide here and the table shows why: the single largest safety fix
(occlusion) moved IoU *down* slightly while cutting DANGER fourfold. Steer by
DANGER and occupied recall.

### Choosing the operating point

`--min-cell-points` is the safety/coverage dial, and it is close to monotone.
Measured over the same 12 storeys, with every point kept:

| `--min-cell-points` | DANGER | occ recall | prec (fair) | recall | IoU |
|---|---|---|---|---|---|
| 2 | **0.024** | 0.975 | 0.945 | 0.408 | 0.378 |
| **4 (default)** | 0.050 | 0.947 | 0.937 | 0.609 | 0.527 |
| 8 | 0.164 | 0.824 | 0.926 | 0.720 | **0.652** |

4 is the knee: it keeps DANGER at 5% while recovering most of the coverage.
Drop to 2 for a map that is safer and more timid; raise to 8 to maximise IoU,
which is the wrong trade for navigation.

**Per-point confidence filtering is the wrong lever and is off by default.**
`--min-conf` exists and the .ply carries per-point confidence, but at matched
cell thresholds keeping every point strictly dominates filtering:

| config | DANGER | occ recall | prec (fair) | IoU |
|---|---|---|---|---|
| all points, cell≥4 | **0.050** | **0.947** | **0.937** | 0.527 |
| conf≥2.0, cell≥4 | 0.236 | 0.740 | 0.850 | 0.613 |

Low-confidence points are worth keeping because they act as *occluders*: they
stop rays that would otherwise punch through a thinly-reconstructed wall. What
they must not do is each become an obstacle on their own, which is exactly what
the per-cell threshold prevents. Filtering by confidence throws away the
occlusion and keeps nothing in exchange.

What bounds the remaining error:

- **Free recall, at 0.609, is the weak column.** Rays stop at the first
  obstacle, so genuinely occluded space is correctly left unknown rather than
  guessed free — much of the gap is the fix working as intended.
- **Grid accuracy tracks SLAM drift**, corr(ATE, IoU) = **−0.68** as measured
  before these fixes. At 0.05 m/cell an ATE of 0.3 m is a 6-cell displacement
  and `00802`'s 1.2 m is 24 cells. `00802` and `00806` remain the worst grids,
  and they are the worst reconstructions.
- **Habitat's "navigable" is eroded** by the agent radius and excludes space
  under furniture, while our free space is line-of-sight floor. The `prec_f`
  column erodes ours to match; it is worth only about 3 points, so the rest of
  the precision gap is real error rather than a definitional mismatch.

Grids observing fewer than `--min-gt` (default 2000) navmesh cells are ramp
artifacts rather than storeys — `00803`'s stairwell splits score against a few
hundred navigable cells — and are listed separately below the median.

### On the free-space carving

**Rays are stopped by obstacles** (`--no-occlusion` restores the old
behaviour). They were not, originally: `bresenham_free` marched the whole way
from camera to target, clearing every cell en route. A ray aimed at a far wall
therefore erased every obstacle standing between it and the camera — and
because each ray subtracts `L_FREE` while a cell's obstacle points each add
`L_OCC` only once, a genuine wall crossed by enough rays was voted empty. The
original comment claimed the nearest-pose scheme was "conservative"; the DANGER
column showed it was marking 32% of observed obstacles as drivable.

Building the occupancy mask before casting, and returning from the ray at the
first occupied cell, is the whole fix:

| variant | DANGER | occ recall | prec (fair) | recall | IoU |
|---|---|---|---|---|---|
| rays pass through walls (old) | 0.319 | 0.596 | 0.514 | 0.648 | 0.382 |
| **rays stop at walls** | **0.083** | **0.907** | **0.795** | 0.452 | 0.373 |

The mask must be built up front rather than accumulated during the sweep, or
the result depends on the order cells happen to be visited in.

Floor-support gating, re-measured on top of occlusion, still earns its cost:

| variant | DANGER | occ recall | prec (fair) | recall | IoU |
|---|---|---|---|---|---|
| ungated (`--no-floor-support`) | 0.123 | 0.852 | 0.696 | 0.506 | **0.394** |
| gated (default) | **0.083** | **0.907** | **0.795** | 0.452 | 0.373 |

It is a trade, and it is **on by default** because IoU is the wrong objective
for a navigation map: a false-free cell drives the rover into an obstacle, a
missed-free cell only makes it take the long way round. Gating cuts DANGER by a
third for 2 points of IoU. `--no-floor-support` turns it off if you are
optimising for coverage.

Historical note — the earlier tuning below was done *before* occlusion and
before the floor plane was pinned, when the fans it was fighting were an
artifact of rays passing through walls. Kept because it is why `--max-ray`
defaults to 6 m:

| variant | mean IoU (5 grids) |
|---|---|
| baseline (uncapped, no gating) | 0.423 |
| 6 m ray cap | **0.425** |
| cap + floor-support r=0.3 | 0.419 |
| cap + floor-support r=0.6 | 0.410 |
| floor-support r=0.6, uncapped | 0.409 |

## 5. SAM2 object proposals (2D masks -> 3D points)

Occupancy grids say where the rover can drive. This stage says *what is there*:
SAM2 segments each keyframe, and the masks are lifted into the same world frame
as the MASt3R point cloud.

```bash
# 1. SLAM again, this time exporting finalized keyframes
conda run -n mast3r-slam bash scripts/eval_hm3d.sh --export-fact3r --scene 00800-TEEsavR23oF

# 2. SAM2 in its own env, reading those keyframes off disk
cd fact3r-map
conda run -n SAM2 python3 scripts/build_sam2_proposals.py \
    --keyframes ../logs/hm3d/calib_fact3r/fact3r_keyframes/00800-TEEsavR23oF \
    --device 0 --points-per-batch 32
```

Export goes to `logs/hm3d/calib_fact3r/` rather than `calib/`, because keyframes
cost ~3 MB each and the metrics runs are re-read by everything else.

The two stages are separate processes on purpose: MASt3R and SAM2 never hold GPU
memory at the same time, and thresholds can be retuned without rerunning SLAM.
They also live in different conda envs (`mast3r-slam`, `SAM2`) and only meet at
the NPZ files, so neither has to satisfy the other's dependency pins.

### Why not the point-prompt example

The `facebook/sam2-hiera-large` model card shows `SAM2ImagePredictor.predict()`
with input prompts. That answers "what is at this pixel", so it needs the caller
to already know which objects matter. Mapping needs every candidate object
*before* any query exists, so this stage uses `SAM2AutomaticMaskGenerator`, which
puts a 32x32 grid of prompts over the image and returns overlapping,
class-agnostic masks. `--backend transformers` selects the equivalent Hugging
Face `mask-generation` pipeline instead.

### Masks must be generated on the keyframe raster

HM3D renders 512x512, but `resize_img` centre-crops MASt3R input to **512x384**.
Only the exported keyframe RGB shares a pixel grid with the pointmap, so masks
generated from the original PNG would be vertically offset by 64 px. The
generator therefore takes `keyframe.rgb`, never the source frame.

Lifting is then just indexing: mask pixel `(r, c)` -> `pointmap_camera[r, c]` ->
`pose_world_from_camera`. No registration step exists because none is needed.

### What the numbers looked like on 00800

51 keyframes, 703 proposals kept after filtering, ~14/frame, 62% median pixel
coverage, ~3 s/keyframe on a 4090.

Recomputing world points independently from the NPZ pointmap and pose reproduces
the stored `points_world` to **0.0 m**, so the lift itself is exact.

Cross-keyframe agreement is the number worth understanding, because the naive
version of it looks alarming. Nearest-*centroid* distance between consecutive
keyframes has a median of ~1 m, and 44% of proposals share no 10 cm voxel with
any proposal in the next keyframe. Neither means the masks are wrong:

- A mask is a *partial* view, so its centroid slides along the object as the
  camera moves. Centroid distance is the wrong metric.
- Of the zero-overlap proposals, **76% have literally 0% of their points still
  inside the next keyframe's frustum** — they left the field of view. Matched
  proposals have median 91% visibility. Only **8.4%** of proposals are visible
  but unmatched, and that is the real mask-inconsistency rate.

The control that separates geometry from segmentation: consecutive *raw*
keyframe pointmaps share 37% of 5 cm voxels and 60% of 20 cm voxels. The
reconstruction is co-registered, so low proposal overlap is never the poses.

Run these checks against a fresh export before trusting a new scene — a genuine
regression shows up as matched-proposal visibility falling, or as the raw
pointmap control dropping.

### Known limits

Proposals are not object IDs. SAM2's mask order is meaningless across frames,
masks overlap, and nothing associates a sofa in keyframe 152 with the same sofa
in keyframe 170 yet. That association step is the next milestone; see
`fact3r-map/IMPLEMENTATION_STATUS.md`.

Storage is ~500 MB of proposals per scene, dominated by full-resolution boolean
masks in each proposal NPZ. Bit-packing them is the obvious win if this needs to
scale to all 100 val scenes.

## 6. Diffusion planning on the BEV map (SafeDiffuser + DSTT)

Plans are sampled from the stock **maze2d-large-v1** diffusion prior, zero-shot
— nothing is trained here — and steered at sampling time by a DSTT safety tube
built from the occupancy grid section 4 produced. The code lives in the sibling
checkout `../SafeDiffuser_STT`; only `compute_dstt_tube` is overridden, so the
guidance, the gain schedule and the endpoint conditioning are the paper's.

Two conda envs are involved and they cannot be merged: torch runs in
`mast3r-slam`, `habitat_sim` in `habitat-vla`. The handoff between them is the
`.npz` of planned trajectories, written in the grid's own plane coordinates.

### Prerequisites

Grids from section 4, plus the pretrained weights, which ship inside
`logs.zip` (2.6 GB) rather than on disk:

```bash
cd ../SafeDiffuser_STT
ls logs/pretrained/maze2d-large-v1/diffusion/H384_T256/state_1920000.pt \
  || unzip -j logs.zip \
       'logs/pretrained/maze2d-large-v1/diffusion/H384_T256/*' \
       -d logs/pretrained/maze2d-large-v1/diffusion/H384_T256/
```

### Run

```bash
cd ../SafeDiffuser_STT
conda activate mast3r-slam
G=../MASt3R-SLAM/logs/hm3d/calib/grids
SCENES="00801-HaxA7YrQdEC 00802-wcojb4TFT35 00804-BHXhpBwSMLh \
        00805-SUHsP6z2gcJ 00806-tQ5s4ShP627 00807-rsggHU7g7dh"

# a. what the unknown-space knob does to the navigable set (optional, diagnostic)
python scripts/hm3d_map_diagnostic.py --grid $G/00807-rsggHU7g7dh.npy

# b. plan. writes <stem>_dstt.{png,json,npz} into the run's plans/ folder
for s in $SCENES; do
  python scripts/plan_hm3d.py --grid $G/$s.npy --n-plans 8 --seed 0
done

# c. the ablation that makes the tube's contribution legible
python scripts/plan_hm3d.py --grid $G/00807-rsggHU7g7dh.npy \
  --n-plans 8 --seed 0 --no-guidance

# d. the tube figure + its radius profiles, one panel per scene
python scripts/plot_tube.py --grids $(for s in $SCENES; do echo $G/$s.npy; done)

# e. why radius_margin is 0.15 (slow: 6 margins x 8 plans per scene)
python scripts/sweep_margin.py --grids $G/00807-rsggHU7g7dh.npy
```

Then score the plans against habitat's ground-truth navmesh, from **this**
repo and in the **other** env:

```bash
cd ../MASt3R-SLAM
for s in $SCENES; do
  conda run -n habitat-vla python scripts/eval_plan_gt.py \
    --plans logs/hm3d/calib/plans/${s}_dstt.npz
done
```

Everything lands in `logs/hm3d/calib/plans/`. Output dir is derived from the
grid path, so pointing the planner at `logs/hm3d/oracle/grids/...` sends its
results to that run's own `plans/` instead of overwriting these.

### Reading the figures

`<stem>_dstt.png` — eight independent start/goal problems on one floorplan,
each in its own colour: **triangle = start, cross = goal, numbered** so a start
pairs with its own goal. Dashed grey is the A* centerline; the two shaded bands
per plan are the safety tube at j=255 (widest, dotted edge) and j=0 (final,
solid edge). Black = occupied, white = observed free, grey = enclosed unknown,
orange = undetermined, beige = exterior (hard obstacle).

`safety_tubes.png` — one problem per scene, drawn at four diffusion steps, so
the prescribed-time contraction is visible as nested blue bands.
`safety_tubes_profiles.png` is the same information as radius-versus-horizon
curves against the cap `d(k) - margin`; where a curve rides the cap, the map is
setting the tube width rather than the schedule.

### Knobs that matter

| flag | default | why |
|---|---|---|
| `--unknown-slack` | 0.50 m | how far unobserved space is pushed away before it blocks. 0 = unknown is wall, inf = unknown is free. Dominates everything on an 80%-unknown map. |
| `--radius-margin` | 0.15 m | gap held between tube wall and nearest obstacle. Worst-case plan clearance comes out at roughly this. Past ~0.20 m the tube closes onto the centerline and the prior stops contributing. |
| `--lambda-stt` | 0.5262 | derived, not tuned: `1/(2(1-e^-mu))` with `mu=3` is the exact-projection relaxation. Do not round it to 0.5. |
| `--radius-min` / `--eta` | 0.25 m / 0.6 | the paper's tube parameters. `r_min` acts as a cap here, not a floor — see the module docstring. |
| `--no-guidance` | off | samples the bare prior. On 00807 that is 90.8% collision. |
