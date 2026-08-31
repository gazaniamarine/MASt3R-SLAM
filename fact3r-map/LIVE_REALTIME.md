# Live real-time video mapping

`scripts/run_fact3r_live.sh` causally processes a webcam, rover stream, or video
file. It loads the models once, periodically discovers objects with SAM2, tracks
them between discovery frames with optical flow, encodes mask observations with
SigLIP, and incrementally associates persistent entities with residual UOT.

The default target is one processed frame per second. When a file or stream is
faster than the mapper, stale input frames are skipped instead of building an
ever-growing queue.

## Webcam

```bash
bash scripts/run_fact3r_live.sh \
  --source 0 \
  --output logs/fact3r_live/webcam_01 \
  --sample-fps 1 \
  --display
```

## Rover or network camera

```bash
bash scripts/run_fact3r_live.sh \
  --source 'rtsp://ROVER_ADDRESS/STREAM' \
  --output logs/fact3r_live/rover_01 \
  --sample-fps 1 \
  --display
```

## Replay a recorded video as a real-time stream

```bash
bash scripts/run_fact3r_live.sh \
  --source /path/to/video.mp4 \
  --output logs/fact3r_live/video_01 \
  --sample-fps 1 \
  --display
```

Press `q`, Escape, or Ctrl-C to stop. The runner finalizes the queryable memory
before exiting. During operation, inspect `latest_preview.jpg` and
`live_status.json` inside the output directory. Use a new output directory for
each run.

## Query the completed or checkpointed memory

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/query_siglip_observations.py \
  --index logs/fact3r_live/video_01/siglip_observations \
  --query 'a 3D printer' \
  --device 0 \
  --min-entity-margin 0.01 \
  --min-view-margin 0.005 \
  --min-supporting-views 1 \
  --no-map-hard-negatives
```

The live runner deliberately does not claim to run MASt3R matching. In the
current single-environment live path, temporal evidence comes from optical flow;
the saved status records `mast3r_live_matching: false`. The offline full-video
pipeline remains the path for MASt3R pair matching.
