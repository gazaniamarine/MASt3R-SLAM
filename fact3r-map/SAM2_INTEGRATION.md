# SAM2 integration with MASt3R-SLAM

## Why the point-prompt example is not the mapping path

`Sam2Model` with `input_points` returns masks only for objects selected by those clicks. Fact3R needs class-agnostic proposals before it knows which objects a later text query may mention.

The integration therefore uses Hugging Face's automatic mask-generation pipeline:

```python
from transformers import pipeline

generator = pipeline(
    "mask-generation",
    model="facebook/sam2.1-hiera-large",
    device=0,
)
outputs = generator(image, points_per_batch=64)
```

SAM2 places a grid of prompts over the image and returns overlapping candidate masks and predicted-IoU scores. These masks are proposals, not persistent object IDs and not a non-overlapping semantic partition.

## Two-stage integration

Run SLAM and SAM2 as separate processes. The large models then do not occupy GPU memory simultaneously, and SAM2 thresholds can be changed without rerunning SLAM.

### 1. Export finalized MASt3R-SLAM keyframes

From the parent `MASt3R-SLAM` directory:

```bash
python main.py \
  --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
  --config config/calib.yaml \
  --save-as fact3r_trial \
  --no-viz \
  --export-fact3r
```

The new flag waits for the MASt3R-SLAM backend to finish its final pose update, then writes:

```text
logs/fact3r_trial/fact3r_keyframes/rgbd_dataset_freiburg1_room/
├── manifest.json
├── keyframe_000000_frame_000000.npz
└── ...
```

Each keyframe contains:

- pointmap-resolution RGB;
- local/camera-frame MASt3R pointmap, constrained to camera rays when calibrated;
- geometry confidence;
- final world-from-camera Sim(3) matrix;
- intrinsics when calibration is active.

The current parent `Frame` does not retain dense downstream MASt3R `D/Q` maps, so this first export uses geometry confidence for mask cleanup. Pair-conditioned `D/Q` handling belongs in the association stage.

### 2. Generate all SAM2 proposals and lift them into 3D

Install the optional dependency set in the same environment:

```bash
cd fact3r-map
pip install -e '.[sam2]'
```

Start with one keyframe and a smaller point batch:

```bash
python scripts/build_sam2_proposals.py \
  --keyframes ../logs/fact3r_trial/fact3r_keyframes/rgbd_dataset_freiburg1_room \
  --device 0 \
  --points-per-batch 16 \
  --max-keyframes 1
```

After inspecting the result, process the sequence:

```bash
python scripts/build_sam2_proposals.py \
  --keyframes ../logs/fact3r_trial/fact3r_keyframes/rgbd_dataset_freiburg1_room \
  --device 0 \
  --points-per-batch 64
```

The default output is:

```text
logs/fact3r_trial/fact3r_sam2/rgbd_dataset_freiburg1_room/
├── manifest.json
├── frame_000000/
│   ├── manifest.json
│   ├── alignment.ply
│   ├── proposal_0000.npz
│   └── ...
└── ...
```

Each proposal NPZ contains its cleaned 2D mask, selected source pixels, world-coordinate 3D points, RGB, and MASt3R geometry confidence. `alignment.ply` shows the filtered masks over the reconstructed keyframe pointmap.

### 3. Build short-term SAM2 video tracklets (optional)

Automatic masks are still generated independently on complete frames. The video
predictor is used only as a continuity measurement: proposals from one keyframe
are supplied as mask prompts, propagated one step, and jointly linked by mask IoU
to the next keyframe's automatic proposals. The next frame's accepted automatic
masks then re-anchor the tracks.

```bash
python scripts/build_sam2_tracklets.py \
  --keyframes ../logs/hm3d/calib_fact3r/fact3r_keyframes/SCENE_NAME \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --device 0 \
  --min-link-iou 0.30 \
  --max-seeds-per-batch 8
```

The default output is `fact3r_sam2_tracklets/SCENE_NAME/manifest.json`. Reduce
`--max-seeds-per-batch` if multi-object video propagation runs out of GPU memory.
The manifest retains every proposal's track ID, incoming source proposal and link
IoU, so the cue can be audited independently of 3D assignment.

### 3a. Segment every captured frame in one second

HM3D is rendered at 30 FPS. To inspect the segmentation itself under continuous
robot motion, run automatic SAM2 independently on all 30 frames in a one-second
window. This example covers the staircase keyframe near frame 248:

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/run_hm3d_one_second_segmentation.py \
  --sequence datasets/hm3d_seqs/00800-TEEsavR23oF \
  --start-frame 240 \
  --duration-seconds 1 \
  --device 0 \
  --points-per-batch 32
```

The default output is
`logs/hm3d/one_second_sam2/00800-TEEsavR23oF/frames_000240_000269`.
It contains every cleaned mask, per-frame overlays, a contact sheet, an animated
GIF with stable short-term track colours, and a manifest with adjacent-frame IoU
link statistics.

This is intentionally a 2D diagnostic: adjacent automatic masks are linked
directly by IoU, without SAM2 video propagation, MASt3R geometry or UOT. It tests
whether mask fragmentation already occurs before persistent-map association.

### 4. Run the complete-frame Hungarian baseline

The proposal builder defaults to Meta's official SAM2 backend. Once it has produced
the scene proposal manifest, run:

```bash
python scripts/run_hungarian_baseline.py \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --tracklets ../logs/hm3d/calib_fact3r/fact3r_sam2_tracklets/SCENE_NAME
```

Every invocation of the mapper processes all masks from one keyframe jointly. SAM2
and MASt3R are not rerun for individual mask/entity pairs. The baseline creates
provisional entities from the first frame, applies one Hungarian solve on each later
proposal-by-entity matrix, updates matched geometry, creates entities for unmatched
masks and retains entities that are not observed.

The default output is a sibling `fact3r_hungarian/SCENE_NAME` directory containing:

```text
manifest.json
frames/frame_XXXXXX_costs.npz
entities/entity-XXXXXX.npz
```

This is intentionally a hard-assignment reference. Its entity fragmentation and
identity switches are measurements for the later balanced/unbalanced transport
models, not behavior that should be hidden with later memory rules.

The Hungarian manifest version 3 also explains every created entity as an empty-map
initialization, missing spatial candidate, above-threshold candidate, or one-to-one
assignment competition. It additionally records how many adjacent-frame tracklet
hints were available and how many assignments honored them. The temporal component
is weighted by propagation IoU and cannot bypass 3D spatial gating. Omitting
`--tracklets` runs the original geometry-first baseline. Re-running association over
existing proposals and tracklets does not require MASt3R-SLAM or SAM2 inference.

## Filtering performed before lifting

The pipeline currently applies:

1. SAM2 predicted-IoU and stability thresholds;
2. minimum and maximum mask area;
3. MASt3R geometry-confidence intersection;
4. boundary erosion;
5. small connected-component removal;
6. mask-IoU duplicate suppression;
7. mask lifting through the finalized world-from-camera transform.

Useful controls include:

```text
--pred-iou-threshold 0.88
--stability-score-threshold 0.95
--min-area-pixels 100
--max-area-fraction 0.8
--erosion-pixels 1
--min-component-pixels 50
--duplicate-iou-threshold 0.9
--min-geometry-confidence 0.0
```

Reduce `--points-per-batch` if automatic mask generation runs out of GPU memory. Increase the confidence and minimum-area thresholds if SAM2 produces too many fragments.

## What this stage does not do

- SAM2 masks are not semantic labels.
- Overlapping masks are not persistent entities.
- SAM2's mask order is not an object ID across frames.
- A SAM2 tracklet is only a short-term cue, not a persistent entity ID.

Persistent identity is still decided jointly from lifted geometry, MASt3R evidence
and the optional temporal cue. The assignment stages now include hard Hungarian,
balanced Sinkhorn, and visibility-conditioned unbalanced transport.

### 5. Run the balanced Sinkhorn comparison

Once the proposal and tracklet artifacts exist, no additional GPU inference is
needed for balanced transport:

```bash
python scripts/run_balanced_sinkhorn.py \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --tracklets ../logs/hm3d/calib_fact3r/fact3r_sam2_tracklets/SCENE_NAME \
  --output ../logs/hm3d/calib_fact3r/fact3r_balanced_sinkhorn/SCENE_NAME
```

Defaults use entropy temperature `0.05`, 2,000 scaling iterations and marginal
tolerance `1e-6`. The output saves the unchanged component costs, candidate mask,
full transport plan, fixed marginals, row-wise hard commitments and numerical
diagnostics for every frame. Because this stage has neither dustbins nor relaxed
marginals, `mean_forbidden_mass` is an expected diagnostic rather than hidden
post-processing.

### 6. Run visibility-conditioned residual transport

This stage needs the keyframes again because it projects persistent entity geometry
into the current view and depth-tests it before setting unbalanced entity demand.
It does not rerun MASt3R-SLAM or SAM2:

```bash
python scripts/run_visibility_residual_transport.py \
  --keyframes ../logs/hm3d/calib_fact3r/fact3r_keyframes/SCENE_NAME \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --tracklets ../logs/hm3d/calib_fact3r/fact3r_sam2_tracklets/SCENE_NAME \
  --output ../logs/hm3d/calib_fact3r/fact3r_visibility_residual_transport/SCENE_NAME
```

There is no dustbin row or column. Spatially forbidden pairs remain exactly zero.
The output separates proposal birth/fragment/noise residual from visible-entity
miss/occlusion residual, and records visibility, marginal relaxation, convergence,
hard-decision confidence, and rejection reasons per frame.

To enable the next lifecycle stage without overwriting that immediate-birth
ablation, add `--delayed-commitment` and omit `--output`:

```bash
python scripts/run_visibility_residual_transport.py \
  --keyframes ../logs/hm3d/calib_fact3r/fact3r_keyframes/SCENE_NAME \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --tracklets ../logs/hm3d/calib_fact3r/fact3r_sam2_tracklets/SCENE_NAME \
  --delayed-commitment
```

The default output becomes
`../logs/hm3d/calib_fact3r/fact3r_delayed_commitment_uot/SCENE_NAME`.
Unmatched residuals are accumulated by track ID. The default confirmation rule is
three observations, mean normalized birth residual at least `0.55`, median link
IoU at least `0.60`, and maximum adjacent 3D-centroid displacement of `0.30 m`.
Single-frame fragments expire instead of creating entities, and an already
committed track cannot create a duplicate after a temporary UOT rejection.

### 7. Render association images

Compare mapping methods over the actual scene RGB and SAM2 masks:

```bash
python scripts/visualize_association.py \
  --keyframes ../logs/hm3d/calib_fact3r/fact3r_keyframes/SCENE_NAME \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --mapping "Hungarian+tracklets=../logs/hm3d/calib_fact3r/fact3r_hungarian_tracklets/SCENE_NAME" \
  --mapping "Balanced Sinkhorn=../logs/hm3d/calib_fact3r/fact3r_balanced_sinkhorn/SCENE_NAME" \
  --mapping "Visibility residual UOT=../logs/hm3d/calib_fact3r/fact3r_visibility_residual_transport/SCENE_NAME" \
  --mapping "Delayed UOT=../logs/hm3d/calib_fact3r/fact3r_delayed_commitment_uot/SCENE_NAME" \
  --output ../logs/hm3d/calib_fact3r/fact3r_association_visualization/SCENE_NAME
```

This writes per-frame side-by-side PNGs, a contact sheet and an animated GIF.
Stable entity colours expose identity switches; green boundaries denote reused
entities, red boundaries denote newly created IDs, yellow denotes pending tracks,
and cyan denotes known tracks held without a memory update. Use `--stride 2` for
a smaller temporal sample or `--no-gif` when only full-resolution PNGs are needed.

### 8. Encode every proposal for semantic recollection

Reuse the completed delayed-UOT artifacts to attach each SigLIP2 mask embedding to
its frame, track and persistent entity:

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/build_siglip_observation_index.py \
  --keyframes logs/hm3d/calib_fact3r/fact3r_keyframes/SCENE_NAME \
  --proposals logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME \
  --mapping logs/hm3d/calib_fact3r/fact3r_delayed_commitment_uot/SCENE_NAME \
  --device 0
```

This stage batches masked crops and records its actual masks-per-second timing.
It also retrospectively resolves early pending observations using the final
track-to-entity commitments.

### 9. Query an object and render all of its frames

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/query_siglip_observations.py \
  --index logs/hm3d/calib_fact3r/fact3r_siglip_observations/SCENE_NAME \
  --query "a clock" \
  --device 0
```

The output query directory contains `index.html`, `matches.gif`,
`contact_sheet.jpg`, per-observation highlighted frames, and machine-readable
scores in `results.json`. These are currently exported keyframe observations;
full 30-FPS histories require the later intermediate-frame SAM2 propagation step.
