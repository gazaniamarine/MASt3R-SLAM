# Fact3R: Current Working Pipeline

Last updated: 2026-09-04

This is the practical continuation guide for the implementation that has worked
best on the real-world video so far. It deliberately separates the reliable
path from experimental alternatives.

## 1. Recommended configuration

Use this as the default pipeline:

```text
RGB video
  -> sampled frames
  -> SAM2-Large periodic complete-frame discovery
  -> optical-flow mask propagation between discovery frames
  -> SigLIP2 embedding for each mask observation
  -> sparse MASt3R reciprocal feature matches
  -> image-space UOT persistent identity association
  -> optional depth + odometry semantic BEV
  -> ungated top-K evidence retrieval
```

The important choices are:

- Use **SigLIP2** for semantic retrieval. It has been more reliable than the
  tested Qwen embedding path for objects such as chairs and 3D printers.
- Use **SAM2-Large for discovery**. Tiny and balanced segmentation presets
  missed too many objects in the real scene.
- Use the **subsecond mapping preset** for the best measured speed/recall
  compromise.
- Keep **MASt3R reciprocal feature matching**, but do not require dense MASt3R
  reconstruction for the real-world pipeline.
- Keep **UOT** for persistent identity grouping and duplicate control. Do not
  treat UOT as semantic recognition.
- Use the **top-K retrieved evidence as the result**. Do not let Qwen/VLM
  rejection suppress good evidence in the default path.

## 2. Run a real video from scratch

Run from the repository root while the `mast3r-slam` environment is active:

```bash
bash scripts/run_fact3r_real_uot.sh \
  --video /absolute/path/to/video.mp4 \
  --output logs/fact3r_real_uot/my_run \
  --sam2-env SAM2 \
  --device 0 \
  --sample-fps 1 \
  --subsecond-mapping-preset
```

Use `--sample-fps 1` when the goal is to keep up with a real-time stream. The
measured 60-frame experiment processed about `1.09 FPS`; at a requested `2 FPS`
it achieved only a `0.545x` real-time factor. This is one-machine evidence, not
a general benchmark.

For denser offline evidence, change to `--sample-fps 2`. It will process more
frames but may not keep up in real time.

The runner is resumable. Its principal outputs are:

```text
logs/fact3r_real_uot/my_run/
├── frames/
├── sam2_proposals/
├── sam2_tracklets/
├── siglip_pre_uot/
├── mast3r_pair_matches/
├── image_uot/
└── siglip_observations/
```

Important: use a new `--output` directory when changing a preset or major
parameter. Completed stage manifests are reused automatically.

## 3. Query the evidence directly

For the searchable observation memory, use SigLIP without a VLM:

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/query_siglip_observations.py \
  --index logs/fact3r_real_uot/my_run/siglip_observations \
  --query "a chair" \
  --device 0 \
  --max-entities 5 \
  --entity-top-views 3 \
  --min-entity-margin 0.01 \
  --min-view-margin 0.005 \
  --min-supporting-views 1 \
  --no-map-hard-negatives
```

This writes an HTML gallery, contact sheet, GIF, JSON results, masks and source
frames under:

```text
logs/fact3r_real_uot/my_run/siglip_observations/queries/
```

The permissive margins above are intentional for evidence discovery. Inspect
the returned top candidates instead of interpreting the score as a calibrated
probability.

## 4. Build the semantic BEV

When aligned depth and odometry are available:

```bash
bash scripts/run_depth_semantic_bev.sh \
  --index logs/fact3r_real_uot/my_run/siglip_observations \
  --root /absolute/path/to/aligned_depth_and_odometry \
  --out logs/rover/my_run/map \
  --fx 631 \
  --pitch 2.75 \
  --cam-height 0.5 \
  --scale 0.969 \
  --device 0
```

If the video and odometry clocks do not already agree, also pass the calibrated
`--time-offset` value.

## 5. Recommended final query: top-K BEV evidence

This is currently the preferred final retrieval command:

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/query_semantic_bev.py \
  --map logs/rover/my_run/map \
  --query "computer monitor" \
  --device 0 \
  --top-k 5 \
  --top-views 3
```

It does not apply a VLM rejection gate. It returns the strongest persistent
entities, highlights them on the BEV, and saves the best observed source frame
and mask for every result.

Outputs are written beside the map:

```text
computer-monitor_semantic_bev.png
computer-monitor_semantic_bev.json
computer-monitor_semantic_bev_observed_frames/
```

Multi-word queries automatically include phrase and head-noun variants. For
example, `computer monitor` also tests `monitor`, which fixed the earlier
multi-word retrieval failure.

## 6. Check whether UOT is helping

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/check_uot_mapping.py \
  --mapping logs/fact3r_real_uot/my_run/image_uot \
  --output logs/fact3r_real_uot/my_run/uot_diagnostic.json
```

Healthy signs are:

- numerical convergence close to `100%`;
- track fragmentation and identity-switch rate close to `0%`;
- nonzero identity reuse;
- duplicate observations resolved without creating extra entities.

These statistics cannot detect incorrect over-merging. Always inspect the
retrieved evidence before concluding that fewer entities are better.

UOT is not necessary for independent mask retrieval. It is useful when the map
must remember that observations across frames, SAM2 track breaks, or revisits
belong to the same physical object.

## 7. Run the actual live path

For a webcam or rover camera:

```bash
bash scripts/run_fact3r_live.sh \
  --source 0 \
  --output logs/fact3r_live/rover_run \
  --sample-fps 1 \
  --display
```

Replace `0` with an RTSP URL for a network camera. This path is causal and drops
stale frames instead of building a queue. It uses optical-flow continuity and
does **not** currently run live MASt3R matching. The offline real-video command
in Section 2 remains the better evaluation path.

## 8. Experimental paths not recommended as defaults

- `--qwen-semantic`: slower and less reliable than SigLIP in the current tests.
- `query_semantic_bev_vlm_live.py`: useful as an ablation, but VLM rejection
  removed good candidates even when its input evidence was correct.
- Tiny/ultra-fast segmentation: faster, but missed important small objects.
- Dense MASt3R reconstruction on the real video: unreliable when there are too
  few stable reconstruction keypoints. Sparse reciprocal matches remain useful.

Do not delete these paths; retain them for ablation experiments.

## 9. What to improve next

Work in this order:

1. Measure top-K retrieval recall on a small annotated real-video query set.
2. Compare UOT against track-only grouping to quantify duplicate reduction and
   harmful over-merging.
3. Improve SAM2 discovery recall for small objects without changing the working
   SigLIP retrieval stage.
4. Validate depth/odometry alignment and semantic BEV placement.
5. Connect the returned evidence and BEV target to the VLA/navigation policy.

The core evaluation should be **top-K object recall and correct physical-entity
grouping**, not whether a language model emits a confident label.
