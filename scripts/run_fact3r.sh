#!/usr/bin/env bash
# End-to-end Fact3R baseline for one rendered HM3D scene.
#
# MASt3R-SLAM runs in the currently active environment. Official SAM2 and the
# Hungarian mapper run through the environment selected by --sam2-env.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)

scene=""
sam2_environment="SAM2"
device="0"
points_per_batch="32"
points_per_side="32"
skip_slam=false
skip_sam2=false
skip_hungarian=false
max_keyframes=""
max_frames=""

usage() {
    echo "Usage: $0 --scene SCENE_NAME [options]"
    echo
    echo "Required:"
    echo "  --scene NAME              HM3D scene folder, e.g. 00800-TEEsavR23oF"
    echo
    echo "Options:"
    echo "  --sam2-env NAME           Conda environment for SAM2 (default: SAM2)"
    echo "  --device DEVICE           SAM2 device (default: 0)"
    echo "  --points-per-batch N      SAM2 prompt batch size (default: 32)"
    echo "  --points-per-side N       SAM2 grid density (default: 32)"
    echo "  --max-keyframes N         Segment only the first N keyframes"
    echo "  --max-frames N            Associate only the first N proposal frames"
    echo "  --skip-slam               Reuse an existing keyframe export"
    echo "  --skip-sam2               Reuse an existing proposal run"
    echo "  --skip-hungarian          Stop after proposal generation"
    echo "  -h, --help                Show this message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)
            scene=${2:?"--scene requires a value"}
            shift 2
            ;;
        --sam2-env)
            sam2_environment=${2:?"--sam2-env requires a value"}
            shift 2
            ;;
        --device)
            device=${2:?"--device requires a value"}
            shift 2
            ;;
        --points-per-batch)
            points_per_batch=${2:?"--points-per-batch requires a value"}
            shift 2
            ;;
        --points-per-side)
            points_per_side=${2:?"--points-per-side requires a value"}
            shift 2
            ;;
        --max-keyframes)
            max_keyframes=${2:?"--max-keyframes requires a value"}
            shift 2
            ;;
        --max-frames)
            max_frames=${2:?"--max-frames requires a value"}
            shift 2
            ;;
        --skip-slam)
            skip_slam=true
            shift
            ;;
        --skip-sam2)
            skip_sam2=true
            shift
            ;;
        --skip-hungarian)
            skip_hungarian=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$scene" ]]; then
    echo "--scene is required" >&2
    usage >&2
    exit 2
fi

dataset_directory="$repository_root/datasets/hm3d_seqs/$scene"
run_directory="$repository_root/logs/hm3d/calib_fact3r"
keyframe_directory="$run_directory/fact3r_keyframes/$scene"
proposal_directory="$run_directory/fact3r_sam2/$scene"
hungarian_directory="$run_directory/fact3r_hungarian/$scene"

cd "$repository_root"

if [[ "$skip_slam" == false ]]; then
    if [[ ! -d "$dataset_directory" ]]; then
        echo "HM3D sequence not found: $dataset_directory" >&2
        echo "Render or copy the scene before running this script." >&2
        exit 1
    fi
    echo "[1/3] Running MASt3R-SLAM and exporting Fact3R keyframes"
    bash scripts/eval_hm3d.sh --export-fact3r --scene "$scene"
else
    echo "[1/3] Reusing exported keyframes"
fi

if [[ ! -f "$keyframe_directory/manifest.json" ]]; then
    echo "Keyframe manifest not found: $keyframe_directory/manifest.json" >&2
    exit 1
fi

if [[ "$skip_sam2" == false ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required to run the SAM2 environment" >&2
        exit 1
    fi
    echo "[2/3] Generating complete-frame official SAM2 proposals"
    sam2_command=(
        conda run -n "$sam2_environment" python3
        "$repository_root/fact3r-map/scripts/build_sam2_proposals.py"
        --keyframes "$keyframe_directory"
        --output "$proposal_directory"
        --backend official
        --device "$device"
        --points-per-side "$points_per_side"
        --points-per-batch "$points_per_batch"
    )
    if [[ -n "$max_keyframes" ]]; then
        sam2_command+=(--max-keyframes "$max_keyframes")
    fi
    "${sam2_command[@]}"
else
    echo "[2/3] Reusing official SAM2 proposals"
fi

if [[ ! -f "$proposal_directory/manifest.json" ]]; then
    echo "Proposal manifest not found: $proposal_directory/manifest.json" >&2
    exit 1
fi

if [[ "$skip_hungarian" == false ]]; then
    echo "[3/3] Running complete-frame Hungarian persistent mapping"
    hungarian_command=(
        conda run -n "$sam2_environment" python3
        "$repository_root/fact3r-map/scripts/run_hungarian_baseline.py"
        --proposals "$proposal_directory"
        --output "$hungarian_directory"
    )
    if [[ -n "$max_frames" ]]; then
        hungarian_command+=(--max-frames "$max_frames")
    fi
    "${hungarian_command[@]}"

    if [[ ! -f "$hungarian_directory/manifest.json" ]]; then
        echo "Hungarian manifest not found: $hungarian_directory/manifest.json" >&2
        exit 1
    fi
else
    echo "[3/3] Hungarian mapping skipped"
fi

echo
echo "Fact3R HM3D pipeline complete"
echo "  keyframes:  $keyframe_directory"
echo "  proposals:  $proposal_directory"
if [[ "$skip_hungarian" == false ]]; then
    echo "  Hungarian:  $hungarian_directory"
fi
