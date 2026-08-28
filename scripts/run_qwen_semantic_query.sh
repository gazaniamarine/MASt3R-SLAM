#!/usr/bin/env bash
# Build a no-SigLIP semantic memory from an existing real-world Qwen-UOT run.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)

run_root=""
query=""
environment="SAM2"
model="Qwen/Qwen3-VL-Embedding-2B"
device_map="auto"
dtype="bfloat16"
batch_size="8"

usage() {
    echo "Usage: $0 --run-root PATH --query TEXT [options]"
    echo "  --run-root PATH    e.g. logs/fact3r_real_uot/full_video_dense"
    echo "  --query TEXT       e.g. 'a chair'"
    echo "  --model ID         default: Qwen/Qwen3-VL-Embedding-2B"
    echo "  --environment ENV  default: SAM2"
    echo "  --batch-size N     default: 8"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) run_root=${2:?}; shift 2 ;;
        --query) query=${2:?}; shift 2 ;;
        --model) model=${2:?}; shift 2 ;;
        --environment) environment=${2:?}; shift 2 ;;
        --device-map) device_map=${2:?}; shift 2 ;;
        --dtype) dtype=${2:?}; shift 2 ;;
        --batch-size) batch_size=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$run_root" || -z "$query" ]]; then
    usage >&2
    exit 2
fi
if [[ "$run_root" != /* ]]; then
    run_root="$repository_root/$run_root"
fi

frames="$run_root/frames"
proposals="$run_root/sam2_proposals"
tracklets="$run_root/sam2_tracklets"
mapping="$run_root/image_uot_qwen"
index="$run_root/qwen_semantic_observations"

for required in "$frames/manifest.json" "$proposals/manifest.json" \
    "$tracklets/manifest.json" "$mapping/manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required artifact: $required" >&2
        exit 2
    fi
done

cd "$repository_root"
python_command=(conda run -n "$environment" python3)

if [[ ! -f "$index/manifest.json" ]]; then
    echo "[1/2] Building mask-level Qwen semantic memory"
    "${python_command[@]}" \
        fact3r-map/scripts/build_qwen_embedding_observation_index.py \
        --keyframes "$frames" \
        --proposals "$proposals" \
        --tracklets "$tracklets" \
        --mapping "$mapping" \
        --output "$index" \
        --model "$model" \
        --device-map "$device_map" \
        --dtype "$dtype" \
        --batch-size "$batch_size"
else
    echo "[1/2] Reusing Qwen semantic memory"
fi

echo "[2/2] Querying persistent entities"
"${python_command[@]}" fact3r-map/scripts/query_siglip_observations.py \
    --index "$index" \
    --query "$query" \
    --device-map "$device_map" \
    --dtype "$dtype" \
    --fast \
    --no-map-hard-negatives \
    --min-view-margin -1.0 \
    --min-entity-margin -1.0 \
    --min-supporting-views 1
