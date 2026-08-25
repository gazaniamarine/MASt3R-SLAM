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

### 3. Run the complete-frame Hungarian baseline

The proposal builder defaults to Meta's official SAM2 backend. Once it has produced
the scene proposal manifest, run:

```bash
python scripts/run_hungarian_baseline.py \
  --proposals ../logs/hm3d/calib_fact3r/fact3r_sam2/SCENE_NAME
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
- Similar masks from different keyframes are not associated yet.

The next association step must use the lifted geometry and MASt3R correspondence evidence to decide which observations belong to the same physical entity.
