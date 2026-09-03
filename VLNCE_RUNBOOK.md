# Testing Fact3R-Map on VLN-CE

This runbook covers the **go-there/come-back** protocol: the agent maps a scene
while it travels an outbound route, then is asked to return to an object it saw
early on, navigating from its own map alone.

```
R2R-CE episodes ──build_vlnce_tours.py──> chained tour (3 legs, ~37 m)
                                                 │
                                    render_vlnce_tour.py (habitat-sim)
                                                 │
                                   PNG sequence + GT poses + return target
                                                 │
                                    import_vlnce_frames.py
                                                 │
                              run_fact3r_real_uot.sh (SAM2 → SigLIP → UOT)
                                                 │
                                  persistent entity map + semantic index
                                                 │
                          return leg: query → BEV goal → plan → execute
                                                 │
                                    NE / SR / OSR / SPL
```

## What this is, and what it is not

VLN-CE is an **online instruction-following** benchmark: no prior map, and the
agent executes a route description step by step. Fact3R is a mapping and
retrieval system with no instruction-following policy.

This protocol therefore measures something different: whether a persistent
entity map, built causally during an outbound traverse, can later resolve a
language reference to an object and support navigation back to it. It reuses
VLN-CE **scenes, episodes, instructions, and metric definitions**, but the agent
has seen the environment.

> Numbers from this protocol are **not comparable to the VLN-CE leaderboard**
> and must never be reported as VLN-CE results. Report them as return
> navigation from a self-built map.

### Why episodes are chained

Single R2R-CE episodes are short — mean geodesic distance 8.9 m in `val_unseen`,
max 21.0 m. That is too short to stress persistent memory, which is the claim
under test. Chaining three episodes inside one scene gives a **~37 m mean
outbound traverse** (max 49 m), and it supplies the return target for free: leg
0's goal has both an exact ground-truth position and an instruction that names
it ("...and stop near the rug" → return query `"rug"`).

## Status

| Stage | State |
|---|---|
| Episode loading, chaining, landmark mining, metrics | **Done**, 35 tests |
| Outbound renderer (`render_vlnce_tour.py`) | **Written**, 16 tests on injected navmesh functions; never executed against a real scene |
| Frames bridge (`import_vlnce_frames.py`) | **Done**, 13 tests, verified end-to-end on real 512×512 habitat renders |
| Fact3R mapping over the outbound leg | Existing pipeline, unmodified |
| Return leg: query → goal → plan → execute | **Not built** — see below |

Everything up to and including the frames bridge runs today. The renderer is
blocked only on scene meshes.

## 0. Data

### Episodes — already present

`datasets/vlnce/R2R_VLNCE_v1-3/` (2.6 MB) is downloaded:
`train` 10819 episodes / 61 scenes, `val_seen` 778 / 53, `val_unseen` 1839 / 11.

### Scenes — the blocker

R2R-CE episodes reference `mp3d/<house>/<house>.glb`. Matterport3D is **not**
freely downloadable: you must sign the Matterport Terms of Use on the
[project webpage](https://niessner.github.io/Matterport/) to be sent
`download_mp.py`. Nobody can do this step for you.

You do **not** need all 90 houses. `val_unseen` uses only these 11:

```
zsNo4HB9uLZ  TbHJrupSAjP  QUCTc6BB5sX  2azQ1b91cZZ  oLBMNvg9in8  Z6MFQCViBuw
X7HyMhZNoso  EU6Fwq7SyZv  x8F5xyUWy9e  8194nk5LbLH  pLe4wQe7qrG
```

```bash
for house in zsNo4HB9uLZ TbHJrupSAjP QUCTc6BB5sX 2azQ1b91cZZ oLBMNvg9in8 \
             Z6MFQCViBuw X7HyMhZNoso EU6Fwq7SyZv x8F5xyUWy9e 8194nk5LbLH pLe4wQe7qrG; do
  python download_mp.py --task habitat -o datasets/mp3d/ --id "$house"
done
```

Extract to `datasets/mp3d/<house>/<house>.glb`. `render_vlnce_tour.py` also
accepts nested dumps (`v1/scans/...`) and basis meshes (pass
`--scene-dataset-config`).

### Environments

- `habitat-vla` — habitat-sim 0.3.3, Python 3.9, headless EGL. Rendering only.
- `SAM2` — Python 3.11. Tours, frames bridge, mapping, queries.

Do **not** install habitat-lab. The original VLN-CE code pins habitat-lab/sim
0.1.7 and would fight the installed 0.3.3. Episodes are plain JSON, so the
renderer drives habitat-sim directly. Tours cross the Python-version boundary as
JSON, never as imports.

## 1. Build tours — runs now, no scenes needed

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/build_vlnce_tours.py \
  --split val_unseen --num-legs 3 --max-per-scene 2
```

Writes `logs/vlnce/val_unseen/tours.json`. Add `--stats-only` to inspect
coverage without writing.

Current output: 20 tours, mean outbound 36.5 m, one return query each —
`pool table`, `red chair`, `footstool`, `console table`, `glass dining table`,
`statue`, `archway`, `bed`, `toilet`, `bar`, …

Chaining links legs by straight-line distance, because this stage has no scene
to consult. The renderer re-measures every link geodesically and drops tours
whose legs turn out to be on opposite sides of a wall.

## 2. Render the outbound tour

```bash
conda run --no-capture-output -n habitat-vla python3 scripts/render_vlnce_tour.py \
  --tours logs/vlnce/val_unseen/tours.json \
  --mp3d-root datasets/mp3d \
  --out-root datasets/vlnce_seqs \
  --semantic
```

Per tour it writes `000000.png …`, `groundtruth.txt` (TUM, habitat world frame),
and `meta.json` carrying per-leg frame ranges, the return query, the navmesh-
snapped return position, and `return_optimal_geodesic_m` — the SPL reference,
taken from the simulator so a distorted map cannot inflate the score.

Camera matches [render_hm3d_traj.py](scripts/render_hm3d_traj.py): square
512×512 at 90° HFOV, so `config/hm3d_intrinsics.yaml` applies unchanged. This
deliberately differs from the RxR-Habitat challenge spec (480×640 RGBD); the
protocol is not a challenge submission.

**Smoke-tested on ReplicaCAD, not on MP3D.** The whole habitat path - render,
frames bridge, odometry, plan, drive, score - was exercised end to end on
`apt_0` (see "Smoke test" below). MP3D-specific surface that remains unproven:
loose-`.glb` scene lookup, and the semantic sensor, since ReplicaCAD carries no
annotations.

Scenes addressed by name through a scene dataset config (ReplicaCAD, HM3D) need
`--scene-dataset-config`; scenes with no shipped navmesh also need
`--recompute-navmesh`.

## 2b. Check the target was actually seen

`--semantic` adds a habitat instance sensor - used only for this audit, never
fed to the mapper - and writes `semantic_visibility.json` per tour.

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/check_vlnce_target_visibility.py \
  --renders datasets/vlnce_seqs \
  --output logs/vlnce/visibility_report.json
```

```text
  [OK ] zsNo4HB9uLZ_t00: 'bed' -> observed
         seen in 48 frames, peaking at 18.00% of a frame
  [DROP] EU6Fwq7SyZv_t01: 'red chair' -> glimpsed
         seen in 2 frames at most 0.090% of a frame
  [DROP] oLBMNvg9in8_t02: 'statue' -> no_candidate_instance
```

Four verdicts, and they are not interchangeable:

| Verdict | Meaning |
|---|---|
| `observed` | seen in >= `--min-frames` at >= `--min-pixel-fraction` of a frame; usable |
| `glimpsed` | present but too briefly or too small to expect a mask |
| `never_observed` | the instance exists near the goal but never rendered |
| `no_candidate_instance` | nothing near the goal matches the landmark at all |

**Drop everything that is not `observed` before mapping.** A return failure on a
never-seen target measures nothing about the map, and a table of such results
would be actively misleading. Loosening the thresholds is a decision that needs
a stated reason, not a default.

Category matching is generous on wording but never crosses head nouns: "glass
dining table" matches `table`, "couch" matches `sofa`, but "table" never matches
`chair`. The synonym table is deliberately small - a wrong synonym silently
marks an unusable tour as usable.

## 3. Bridge into the Fact3R pipeline

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/import_vlnce_frames.py \
  --render datasets/vlnce_seqs/zsNo4HB9uLZ_t00 \
  --output logs/vlnce_runs/zsNo4HB9uLZ_t00/frames \
  --sample-fps 6
```

Pick the rate by **spacing, not fps** - the render's 30 fps is synthetic, and
what matters to SAM2 tracking and UOT association is how far the camera moves
between keyframes. At the renderer's 0.04 m/frame:

| `--sample-fps` | stride | spacing | keyframes on a 37 m tour |
|---|---|---|---|
| 2 | 15 | 0.60 m | 61 |
| 4 | 8 | 0.32 m | 114 |
| 6 | 5 | 0.20 m | 182 |
| 10 | 3 | 0.12 m | 304 |

`2` is the rover default but leaves only ~61 keyframes for a whole tour, far
sparser than the 935-frame runs the association thresholds were tuned on. Start
at `6` and treat it as a calibration knob.

Emits the `fact3r-mast3r-keyframes` layout, so `run_fact3r_real_uot.sh` skips
its own sampling stage and reuses these frames.

Keyframes keep the video path's image-only convention: **identity poses, NaN
pointmaps**. Habitat ground truth is available and real, but writing it into the
keyframes would hand the map geometry the rover runs never had, and the two
would stop being comparable. GT poses ride along in the manifest, for scoring
only.

Note `_mast3r_raster` centre-crops the square render to 4:3 (512×384), losing
25% of vertical field of view. The manifest records both `source_intrinsics`
and the post-crop `keyframe_intrinsics` (cy 256 → 192).

## 3b. Odometry, if the return leg grounds through the BEV

`build_depth_semantic_bev.py` reads `odom_*.csv` (`t,x,y,theta,v`); the renderer
writes 7-DoF TUM poses in habitat's y-up frame. Convert them:

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/vlnce_poses_to_odom.py \
  --poses datasets/vlnce_seqs/zsNo4HB9uLZ_t00
```

The planar frame is `x = -z_habitat`, `y = -x_habitat`, `theta = yaw about +y`,
which puts habitat's forward axis on +x at theta = 0 with standard CCW rotation.
Verified against real habitat poses: a 1200-frame HM3D tour converts to 30.3 m
travelled at a 1.20 m/s peak, exactly the renderer's 0.04 m x 30 fps.

Habitat poses are exact, so this is **odometry without drift** - a stronger
input than the rover's wheel encoders ever provide. Say so when comparing.

## 4. Map the outbound leg

```bash
bash scripts/run_fact3r_real_uot.sh \
  --output logs/vlnce_runs/zsNo4HB9uLZ_t00 \
  --video datasets/vlnce_seqs/zsNo4HB9uLZ_t00 \
  --sample-fps 2 --qwen-semantic
```

The frames directory already holds a manifest, so stage 1 is skipped and
`--video` is only recorded as provenance. Stages 2–7 (SAM2 proposals, tracklets,
MASt3R pair matches, embeddings, residual UOT, observation index) run unchanged.

Then confirm the target is actually in the map before trusting any nav result:

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/query_siglip_observations.py \
  --index logs/vlnce_runs/zsNo4HB9uLZ_t00/siglip_observations \
  --query 'pool table' --device 0
```

If retrieval fails here, the return leg cannot succeed and its metrics say
nothing about navigation.

## 5. Return leg — not built

The remaining piece. Interfaces are fixed by what stages 2–4 already emit:

1. **Resolve** the return query against the observation index → entity id.
2. **Ground** the entity to a metric BEV goal. The rover path
   ([DEPTH_SEMANTIC_BEV.md](fact3r-map/DEPTH_SEMANTIC_BEV.md)) does this with
   Depth Anything V2 + odometry; here habitat gives exact depth and poses, so a
   simpler grounding is available — but using GT depth weakens the claim, so
   prefer the same monocular path the rover uses and report which was used.
3. **Plan** on the occupancy grid ([occupancy_grid.py](scripts/occupancy_grid.py),
   A* centreline, the vendored SafeDiffuser/DSTT planner). Set `unknown_slack`
   to the robot radius or `collision_frac` reads 0 while plans cross unobserved
   space.
4. **Execute** in habitat with the VLN-CE action space (0.25 m forward, 15°
   turn, STOP), logging every agent position.
5. **Score** with `fact3r.experiments.vlnce.score_return`, passing
   `return_optimal_geodesic_m` as `optimal_length` and the simulator pathfinder
   as `distance_fn`.

The planner must consume **only the agent's own map**. Any use of
`pathfinder.find_path` for planning turns this into an oracle-navigation test;
the pathfinder is for scoring only.

## Smoke test without MP3D

ReplicaCAD is already on disk and needs no download, so the habitat path can be
exercised before committing to a dataset:

```bash
CFG=datasets/habitat_data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json
conda run -n habitat-vla python3 scripts/render_vlnce_tour.py \
  --tours <synthetic tours.json> --scene-dataset-config $CFG \
  --recompute-navmesh --semantic
```

A 3-leg tour over `apt_0` rendered 342 frames across 9.2 m, the odometry
converter independently reproduced that 9.2 m, and the return leg planned 256
waypoints and drove them in 25 actions to NE 1.08 m (SPL 1.000), with the
geodesic to target decreasing monotonically from 3.97 m.

What this does **not** cover: `apt_0` is 59 m2, so tours are ~9 m rather than
the ~37 m of a chained R2R-CE tour, and ReplicaCAD has no semantic annotations,
so the visibility audit only exercises its "no annotations" branch. It is a
check that the code runs, not a result.

## Metrics

`score_return` / `aggregate` implement the standard VLN-CE definitions applied
to the return leg, with success at 3 m:

| Metric | Meaning |
|---|---|
| NE | geodesic distance from the final pose to the return target |
| SR | NE < 3 m |
| OSR | any pose along the return path came within 3 m |
| SPL | SR · optimal / max(optimal, travelled) |
| heading error | final heading minus the goal heading |
| **pose success** | within 3 m **and** within one turn increment of the goal heading |

Report `pose_success`, not only `success`. VLN-CE's criterion is position-only,
and a return that lands on the right spot facing 166 degrees the wrong way
scores as a success while the camera re-observes nothing. Measured on this run:
position-only returns scored `success=True` at **-166.2 deg** heading error.

Report alongside them, or the navigation numbers are uninterpretable:
retrieval rank of the target entity, entity count, outbound length, and the
gap between when the target was last seen and when it was requested.

## Known issues and honest limits

- **Landmark mining recovers 46% of episodes** (847/1839 `val_unseen`). The
  extractor is a shallow regex over R2R stop clauses, tuned for precision:
  clauses resolving to a verb phrase, pronoun, or bare spatial word are dropped
  rather than guessed. Episodes without a landmark are simply not used.
- **A mined landmark is not a verified visible object.** "stop near the rug"
  means the rug is near the goal, not that it is the goal or even in frame. A
  handful of manually verified targets would make retrieval claims much
  stronger.
- **Chaining reuses `val_unseen` episodes across tours only by trajectory id.**
  Two tours in one scene may still overlap spatially.
- **`datasets/hm3d_root/` is entirely broken.** All 110 scenes are symlinks into
  `/home/nahar4/Gazania/hm3d/`, which no longer exists. HM3D cannot be
  re-rendered until that source tree is restored. The already-rendered
  `datasets/hm3d_seqs/` PNG sequences (2.5 GB) are intact and unaffected.
- **No depth sensor.** The renderer emits RGB (and optionally semantics), so
  there is no ground-truth-depth upper bound to compare the monocular path
  against, though habitat could supply one cheaply.
- **The visibility audit has never run against MP3D.** Its decision logic is
  tested, but `semantic_object_table` reads `semantic_id`, `category.name()` and
  `aabb.center` off habitat's `semantic_scene`, and that has not been exercised
  on a real MP3D house.
- **The held-out `test` split is unusable** and is refused with an explanation:
  it ships no goals and no reference paths.
- Output directories are named `<scene>_t<NN>` by position in the *filtered*
  tour list, so `--scene`/`--limit` runs can collide with a full run's names.
- `score_return` gives SPL 0 for a successful return of zero path length; the
  return target is far by construction, so this edge case should not arise.
- The renderer's `--link-tolerance` defaults to 3 m geodesic, looser than the
  2 m euclidean used when chaining, so a slightly winding link is kept rather
  than discarding the tour.

## Files

| Path | Role |
|---|---|
| [fact3r/experiments/vlnce.py](fact3r-map/fact3r/experiments/vlnce.py) | episodes, chaining, landmark mining, metrics |
| [scripts/build_vlnce_tours.py](fact3r-map/scripts/build_vlnce_tours.py) | tours.json |
| [scripts/render_vlnce_tour.py](scripts/render_vlnce_tour.py) | habitat-sim outbound render |
| [scripts/import_vlnce_frames.py](fact3r-map/scripts/import_vlnce_frames.py) | render → Fact3R frames |
| [fact3r/experiments/vlnce_visibility.py](fact3r-map/fact3r/experiments/vlnce_visibility.py) | target-visibility verdicts |
| [fact3r/experiments/habitat_odometry.py](fact3r-map/fact3r/experiments/habitat_odometry.py) | habitat poses -> planar odometry |
| [scripts/check_vlnce_target_visibility.py](fact3r-map/scripts/check_vlnce_target_visibility.py) | per-tour usable/drop report |
| [scripts/vlnce_poses_to_odom.py](fact3r-map/scripts/vlnce_poses_to_odom.py) | odom_*.csv writer |
| [tests/test_vlnce.py](fact3r-map/tests/test_vlnce.py) | 36 tests |
| [tests/test_vlnce_render.py](fact3r-map/tests/test_vlnce_render.py) | 16 tests |
| [tests/test_vlnce_frames.py](fact3r-map/tests/test_vlnce_frames.py) | 13 tests |
| [tests/test_vlnce_visibility.py](fact3r-map/tests/test_vlnce_visibility.py) | 23 tests |
| [fact3r/experiments/vlnce_return.py](fact3r-map/fact3r/experiments/vlnce_return.py) | VLN-CE action-space controller and scoring |
| [scripts/execute_vlnce_return.py](scripts/execute_vlnce_return.py) | plan over the agent's map, drive it in habitat |
| [tests/test_habitat_odometry.py](fact3r-map/tests/test_habitat_odometry.py) | 17 tests |
| [tests/test_vlnce_return.py](fact3r-map/tests/test_vlnce_return.py) | 39 tests |
