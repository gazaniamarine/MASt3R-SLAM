#!/usr/bin/env bash
# Real-world mapping without dense reconstruction.
# Retains MASt3R 2D matching, SAM2 temporal masks, UOT, and SigLIP memory.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)

video=""
output=""
name=""
mast3r_environment=""
sam2_environment="SAM2"
device="0"
sample_fps="2"
max_frames=""
query=""
points_per_side="32"
points_per_batch="32"
max_seeds_per_batch="16"
siglip_batch_size="32"

usage() {
    echo "Usage: $0 --video VIDEO [options]"
    echo
    echo "  --query 'a chair'          run one fast query after mapping"
    echo "  --sample-fps FPS           processed video rate (default: 2)"
    echo "  --max-frames N             optional diagnostic limit"
    echo "  --output PATH              default: logs/fact3r_real_uot/VIDEO_NAME"
    echo "  --mast3r-env NAME          optional conda env for MASt3R matching"
    echo "  --sam2-env NAME            segmentation/index env (default: SAM2)"
    echo "  --device DEVICE            CUDA device/index (default: 0)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) video=${2:?}; shift 2 ;;
        --query) query=${2:?}; shift 2 ;;
        --sample-fps) sample_fps=${2:?}; shift 2 ;;
        --max-frames) max_frames=${2:?}; shift 2 ;;
        --output) output=${2:?}; shift 2 ;;
        --name) name=${2:?}; shift 2 ;;
        --mast3r-env) mast3r_environment=${2:?}; shift 2 ;;
        --sam2-env) sam2_environment=${2:?}; shift 2 ;;
        --device) device=${2:?}; shift 2 ;;
        --points-per-side) points_per_side=${2:?}; shift 2 ;;
        --points-per-batch) points_per_batch=${2:?}; shift 2 ;;
        --max-seeds-per-batch) max_seeds_per_batch=${2:?}; shift 2 ;;
        --siglip-batch-size) siglip_batch_size=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$video" || ! -f "$video" ]]; then
    echo "--video must point to an existing video file" >&2
    exit 2
fi
video_parent=$(cd -- "$(dirname -- "$video")" && pwd)
video="$video_parent/$(basename -- "$video")"
if [[ -z "$name" ]]; then
    name=$(basename -- "$video")
    name=${name%.*}
    name=${name//[^[:alnum:]._-]/-}
fi
if [[ -z "$output" ]]; then
    output="$repository_root/logs/fact3r_real_uot/$name"
elif [[ "$output" != /* ]]; then
    output="$repository_root/$output"
fi

mast3r_python=(python3)
if [[ -n "$mast3r_environment" ]]; then
    mast3r_python=(conda run -n "$mast3r_environment" python3)
fi
sam2_python=(conda run -n "$sam2_environment" python3)

frames="$output/frames"
proposals="$output/sam2_proposals"
tracklets="$output/sam2_tracklets"
appearance="$output/siglip_pre_uot"
matches="$output/mast3r_pair_matches"
mapping="$output/image_uot"
observations="$output/siglip_observations"
mkdir -p "$output"
cd "$repository_root"

if [[ ! -f "$frames/manifest.json" ]]; then
    echo "[1/7] Sampling MASt3R-aligned RGB frames; dense reconstruction is disabled"
    command=("${mast3r_python[@]}" fact3r-map/scripts/export_real_video_frames.py
        --video "$video" --output "$frames" --sample-fps "$sample_fps")
    if [[ -n "$max_frames" ]]; then command+=(--max-frames "$max_frames"); fi
    "${command[@]}"
else
    echo "[1/7] Reusing sampled RGB frames"
fi

if [[ ! -f "$proposals/manifest.json" ]]; then
    echo "[2/7] Generating SAM2 masks with duplicate and border-fragment filtering"
    "${sam2_python[@]}" fact3r-map/scripts/build_sam2_proposals.py \
        --keyframes "$frames" --output "$proposals" --backend official \
        --device "$device" --points-per-side "$points_per_side" \
        --points-per-batch "$points_per_batch"
else
    echo "[2/7] Reusing SAM2 proposals"
fi

if [[ ! -f "$tracklets/manifest.json" ]]; then
    echo "[3/7] Building SAM2 temporal mask links"
    "${sam2_python[@]}" fact3r-map/scripts/build_sam2_tracklets.py \
        --keyframes "$frames" --proposals "$proposals" --output "$tracklets" \
        --device "$device" --max-seeds-per-batch "$max_seeds_per_batch"
else
    echo "[3/7] Reusing SAM2 temporal links"
fi

if [[ ! -f "$appearance/manifest.json" ]]; then
    echo "[4/7] Encoding mask observations with SigLIP"
    "${sam2_python[@]}" fact3r-map/scripts/build_siglip_observation_index.py \
        --keyframes "$frames" --proposals "$proposals" --tracklets "$tracklets" \
        --output "$appearance" --device "$device" --batch-size "$siglip_batch_size"
else
    echo "[4/7] Reusing SigLIP observations"
fi

if [[ ! -f "$matches/manifest.json" ]]; then
    echo "[5/7] Computing adjacent-frame MASt3R reciprocal feature matches"
    "${mast3r_python[@]}" fact3r-map/scripts/build_mast3r_pair_matches.py \
        --keyframes "$frames" --output "$matches" --device "cuda:$device"
else
    echo "[5/7] Reusing MASt3R pair matches"
fi

if [[ ! -f "$mapping/manifest.json" ]]; then
    echo "[6/7] Associating persistent identities with image-space UOT"
    "${sam2_python[@]}" fact3r-map/scripts/run_image_uot_mapping.py \
        --proposals "$proposals" --tracklets "$tracklets" \
        --appearance-index "$appearance" --mast3r-matches "$matches" \
        --output "$mapping"
else
    echo "[6/7] Reusing image-space UOT map"
fi

if [[ ! -f "$observations/manifest.json" ]]; then
    echo "[7/7] Attaching UOT identities to semantic observations"
    "${sam2_python[@]}" fact3r-map/scripts/attach_siglip_mapping.py \
        --index "$appearance" --mapping "$mapping" --output "$observations"
else
    echo "[7/7] Reusing searchable observations"
fi

echo
echo "Real-world reconstruction-free UOT map complete"
echo "  observations: $observations"
echo "  UOT map:      $mapping"
if [[ -n "$query" ]]; then
    "${sam2_python[@]}" fact3r-map/scripts/query_siglip_observations.py \
        --index "$observations" --query "$query" --device "$device"
else
    echo "Query command:"
    echo "  conda run -n $sam2_environment python3 fact3r-map/scripts/query_siglip_observations.py --index '$observations' --query 'a chair' --device '$device'"
fi
