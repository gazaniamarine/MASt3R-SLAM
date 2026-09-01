# Metric depth + persistent semantic BEV

This path fuses the metric Depth Anything V2 model from `depth_to_bev.py` with
the persistent Fact3R observation index. It uses the exact saved Fact3R RGB
frames and masks, then places their depth pixels using timestamped wheel
odometry. This avoids mask/depth misalignment from decoding and resizing the
video a second time.

Run it after the normal Fact3R video pipeline has produced
`siglip_observations`:

```bash
bash scripts/run_depth_semantic_bev.sh \
  --index logs/fact3r_real_uot/full_video_subsecond/siglip_observations \
  --odom /home/nahar4/Gazania/MPL/odom_RUN.csv \
  --out logs/rover/depth_semantic/map \
  --fx 631 \
  --pitch 2.75 \
  --cam-height 0.5 \
  --scale 0.969
```

`--fx`, `--fy`, `--cx`, and `--cy` describe the original video calibration by
default. If the source video is still at the path recorded by Fact3R, the
builder automatically applies Fact3R's resize and centered crop to those
intrinsics. Use `--intrinsics-are-keyframe` only when the values already
describe the saved 512-pixel Fact3R images. If the source video moved to a
different machine, pass its original dimensions with `--source-width` and
`--source-height` so the calibration can still be transformed correctly.

The output stem produces:

- `map.pgm`, `map.yaml`, and `map.npy`: ROS-compatible occupancy;
- `map_semantic.png`: final colored persistent-entity BEV;
- `map_semantic_bev.npz`: occupancy, entity ID, confidence, and support grids;
- `map_semantic.json`: entity legend, calibration, provenance, and statistics;
- `map_entity_embeddings.npy`: one open-vocabulary prototype per entity;
- `map.ply` and `map.txt`: metric depth cloud and rover trajectory;
- `map_semantic.ply`: semantic surface points with persistent entity IDs.

The visual map uses persistent entity IDs instead of permanently assigning a
closed-set class name. The saved entity embeddings retain open-vocabulary
meaning, so the same BEV can later be queried as “chair,” “work setup,” or “3D
printer” without rerunning depth.

For example, render every BEV cell belonging to the strongest “working desk”
entities:

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/query_semantic_bev.py \
  --map logs/rover/depth_semantic/map \
  --query 'working desk' \
  --device 0 \
  --top-k 3
```

This query only encodes text and ranks the already stored observations; depth
and semantic image crops are not recomputed. It writes the highlighted BEV and
also an `*_observed_frames/` directory containing the best original camera
frame and mask for every ranked match.

Multi-word object queries automatically combine the exact phrase with generic
article and head-noun forms. For example, `computer monitor` also evaluates
`a computer monitor`, `monitor`, and `a monitor`, then requires agreement from
the two strongest variants. The terminal and JSON report the winning wording.
Use `--exact-query` only for an ablation of the original single-prompt behavior.

## Resident VLM-verified queries

For repeated queries, keep both retrieval and verification models loaded:

```bash
bash scripts/run_semantic_bev_vlm_live.sh \
  --map logs/rover/depth_semantic/map \
  --siglip-device 0 \
  --vlm-model Qwen/Qwen3-VL-2B-Instruct \
  --vlm-device-map auto \
  --top-k 5
```

Wait for `Ready`, then type queries at the prompt:

```text
query> computer monitor
query> 3D printer
query> chair
query> quit
```

For every query, the fast semantic-BEV retriever first produces its normal
ranked top-K using the robust phrase/head-noun ensemble. That exact shortlist
and order are frozen before Qwen runs. Qwen only accepts or rejects those
candidates; it cannot introduce a different entity or reorder the shortlist.
It sees the two strongest highlighted views of each candidate in small listwise
batches. Accepted results save the best observed camera frame, a verification
gallery, and `verified_semantic_bev.png`. Models, embeddings, BEV arrays, loaded
camera frames, and verification cache remain resident between prompts.

The default Qwen model is 2B to prioritize latency. `--history-frames 1` renders
only the strongest accepted observation; raise it when a longer visual history
is needed. Warm-query latency is printed after every prompt. The first query
still has normal CUDA/kernel warm-up overhead even though model loading was
moved before the prompt.
