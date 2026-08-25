#!/bin/bash
# Run MASt3R-SLAM over the HM3D sequences rendered by scripts/render_hm3d_traj.py.
#
#   bash scripts/render_hm3d.sh            # render first (see below), then:
#   bash scripts/eval_hm3d.sh              # calibrated run over every sequence
#   bash scripts/eval_hm3d.sh --no-calib   # uncalibrated (intrinsics estimated)
#   bash scripts/eval_hm3d.sh --print      # skip SLAM, just re-print the metrics
#   bash scripts/eval_hm3d.sh --export-fact3r [--scene 00800-TEEsavR23oF]
#                                          # also dump finalized keyframes for
#                                          # fact3r-map/scripts/build_sam2_proposals.py
#
# Sequences are rendered with an exact pinhole camera, so the calibrated path
# uses config/hm3d_intrinsics.yaml rather than guessing intrinsics.
set -u

dataset_path="datasets/hm3d_seqs/"
no_calib=false
print_only=false
export_fact3r=false
scene=""
# Save permissively: the .ply now carries per-point confidence, so the useful
# threshold is chosen downstream against the occupancy metrics rather than
# baked in here. 1.5 (the viewer slider's default, which --no-viz never moves)
# keeps 94% of points anyway, so it was never really filtering.
conf_thresh=1.0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib) no_calib=true ;;
        --print) print_only=true ;;
        --export-fact3r) export_fact3r=true ;;
        --scene) scene="$2"; shift ;;
        --conf-thresh) conf_thresh="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ "$no_calib" = true ]; then
    run_name="hm3d/no_calib"
else
    run_name="hm3d/calib"
fi

# Keyframe export writes a few MB per keyframe, so it goes to its own run
# directory rather than growing the metrics runs everyone else re-reads.
extra_args=()
if [ "$export_fact3r" = true ]; then
    run_name="${run_name}_fact3r"
    extra_args+=(--export-fact3r)
fi

if [ -n "$scene" ]; then
    sequences=("$dataset_path$scene/")
else
    sequences=("$dataset_path"*/)
fi

if [ "$print_only" = false ]; then
    for dataset_name in "${sequences[@]}"; do
        dataset=$(basename "$dataset_name")
        echo "=== $dataset ==="
        if [ "$no_calib" = true ]; then
            python main.py --dataset "$dataset_name" --no-viz \
                --save-as "$run_name" --config config/eval_no_calib.yaml \
                --conf-thresh "$conf_thresh" "${extra_args[@]}"
        else
            python main.py --dataset "$dataset_name" --no-viz \
                --save-as "$run_name" --config config/eval_calib.yaml \
                --calib config/hm3d_intrinsics.yaml \
                --conf-thresh "$conf_thresh" "${extra_args[@]}"
        fi
    done
fi

python3 scripts/eval_hm3d.py --run "$run_name" --json-out "logs/$run_name/metrics.json"
