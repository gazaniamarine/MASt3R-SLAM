#!/usr/bin/env bash
# Build a complete persistent Fact3R map from one finite RGB video.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)

video=""
map_name=""
output_root=""
slam_config="config/base.yaml"
calibration=""
slam_environment=""
sam2_environment="SAM2"
device="0"
points_per_side="32"
points_per_batch="32"
max_seeds_per_batch="16"
siglip_batch_size="32"
offload_tracklet_state=false

usage() {
    echo "Usage: $0 --video VIDEO [options]"
    echo
    echo "Required:"
    echo "  --video PATH                 finite MP4/AVI/MOV/MKV/M4V or PNG directory"
    echo
    echo "Map location:"
    echo "  --map-name NAME              safe map name; defaults to the video stem"
    echo "  --output PATH                defaults to logs/fact3r_video/MAP_NAME"
    echo
    echo "Environments and camera:"
    echo "  --slam-env NAME              run MASt3R-SLAM through this conda env"
    echo "  --sam2-env NAME              mapping environment (default: SAM2)"
    echo "  --config PATH                SLAM config (default: config/base.yaml)"
    echo "  --calib PATH                 optional camera intrinsics YAML"
    echo "  --device DEVICE              CUDA device for SAM2/SigLIP (default: 0)"
    echo
    echo "Performance:"
    echo "  --points-per-side N          SAM2 prompt grid (default: 32)"
    echo "  --points-per-batch N         SAM2 prompt batch (default: 32)"
    echo "  --max-seeds-per-batch N      SAM2 tracklet seed batch (default: 16)"
    echo "  --siglip-batch-size N        embedding batch (default: 32)"
    echo "  --offload-tracklet-state     reduce VRAM at the cost of speed"
    echo
    echo "Existing completed stages are reused automatically."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) video=${2:?"--video requires a value"}; shift 2 ;;
        --map-name) map_name=${2:?"--map-name requires a value"}; shift 2 ;;
        --output) output_root=${2:?"--output requires a value"}; shift 2 ;;
        --slam-env) slam_environment=${2:?"--slam-env requires a value"}; shift 2 ;;
        --sam2-env) sam2_environment=${2:?"--sam2-env requires a value"}; shift 2 ;;
        --config) slam_config=${2:?"--config requires a value"}; shift 2 ;;
        --calib) calibration=${2:?"--calib requires a value"}; shift 2 ;;
        --device) device=${2:?"--device requires a value"}; shift 2 ;;
        --points-per-side) points_per_side=${2:?"--points-per-side requires a value"}; shift 2 ;;
        --points-per-batch) points_per_batch=${2:?"--points-per-batch requires a value"}; shift 2 ;;
        --max-seeds-per-batch) max_seeds_per_batch=${2:?"--max-seeds-per-batch requires a value"}; shift 2 ;;
        --siglip-batch-size) siglip_batch_size=${2:?"--siglip-batch-size requires a value"}; shift 2 ;;
        --offload-tracklet-state) offload_tracklet_state=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$video" ]]; then
    echo "--video is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -e "$video" ]]; then
    echo "Video or frame directory does not exist: $video" >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required for the mapping environment" >&2
    exit 1
fi

video_parent=$(cd -- "$(dirname -- "$video")" && pwd)
video="$video_parent/$(basename -- "$video")"
sequence_name=$(basename -- "$video")
sequence_name=${sequence_name%.*}
if [[ -z "$map_name" ]]; then
    map_name=${sequence_name//[^[:alnum:]._-]/-}
fi
if [[ ! "$map_name" =~ ^[[:alnum:]._-]+$ ]]; then
    echo "--map-name may contain only letters, numbers, dot, underscore and dash" >&2
    exit 2
fi

if [[ "$slam_config" != /* ]]; then
    slam_config="$repository_root/$slam_config"
fi
if [[ ! -f "$slam_config" ]]; then
    echo "SLAM config not found: $slam_config" >&2
    exit 1
fi
if [[ -n "$calibration" ]]; then
    calibration_parent=$(cd -- "$(dirname -- "$calibration")" && pwd)
    calibration="$calibration_parent/$(basename -- "$calibration")"
    if [[ ! -f "$calibration" ]]; then
        echo "Calibration file not found: $calibration" >&2
        exit 1
    fi
fi
if [[ -z "$output_root" ]]; then
    output_root="$repository_root/logs/fact3r_video/$map_name"
elif [[ "$output_root" != /* ]]; then
    output_root="$repository_root/$output_root"
fi
mkdir -p "$output_root"
output_root=$(cd -- "$output_root" && pwd)

keyframes="$output_root/fact3r_keyframes/$sequence_name"
proposals="$output_root/fact3r_sam2/$sequence_name"
tracklets="$output_root/fact3r_sam2_tracklets/$sequence_name"
mapping="$output_root/fact3r_delayed_commitment_uot/$sequence_name"
appearance_index="$output_root/fact3r_siglip_pre_uot/$sequence_name"
observations="$output_root/fact3r_siglip_observations/$sequence_name"

slam_python=(python3)
if [[ -n "$slam_environment" ]]; then
    slam_python=(conda run -n "$slam_environment" python3)
fi
mapping_python=(conda run -n "$sam2_environment" python3)

cd "$repository_root"

if [[ -f "$keyframes/manifest.json" ]]; then
    echo "[1/6] Reusing MASt3R-SLAM keyframes: $keyframes"
else
    echo "[1/6] Running MASt3R-SLAM on $video"
    slam_command=(
        "${slam_python[@]}" main.py
        --dataset "$video"
        --config "$slam_config"
        --save-as "$output_root"
        --no-viz
        --export-fact3r
    )
    if [[ -n "$calibration" ]]; then
        slam_command+=(--calib "$calibration")
    fi
    "${slam_command[@]}"
fi
if [[ ! -f "$keyframes/manifest.json" ]]; then
    echo "MASt3R-SLAM did not produce $keyframes/manifest.json" >&2
    exit 1
fi

proposal_has_dual_memory=false
if [[ -f "$proposals/manifest.json" ]] && \
    grep -Fq '"version": 2' "$proposals/manifest.json"; then
    proposal_has_dual_memory=true
fi
if [[ "$proposal_has_dual_memory" == true ]]; then
    echo "[2/6] Reusing official SAM2 proposals: $proposals"
else
    if [[ -f "$proposals/manifest.json" ]]; then
        echo "[2/6] Rebuilding proposals to retain 2D-only observations"
    else
        echo "[2/6] Generating complete-frame SAM2 proposals"
    fi
    "${mapping_python[@]}" fact3r-map/scripts/build_sam2_proposals.py \
        --keyframes "$keyframes" \
        --output "$proposals" \
        --backend official \
        --device "$device" \
        --points-per-side "$points_per_side" \
        --points-per-batch "$points_per_batch"
fi

if [[ -f "$tracklets/manifest.json" && \
    ! "$proposals/manifest.json" -nt "$tracklets/manifest.json" ]]; then
    echo "[3/6] Reusing SAM2 short-term tracklets: $tracklets"
else
    echo "[3/6] Building re-anchored SAM2 tracklets"
    tracklet_command=(
        "${mapping_python[@]}" fact3r-map/scripts/build_sam2_tracklets.py
        --keyframes "$keyframes"
        --proposals "$proposals"
        --output "$tracklets"
        --device "$device"
        --max-seeds-per-batch "$max_seeds_per_batch"
    )
    if [[ "$offload_tracklet_state" == true ]]; then
        tracklet_command+=(--offload-state-to-cpu)
    fi
    "${tracklet_command[@]}"
fi

if [[ -f "$appearance_index/manifest.json" && \
    ! "$proposals/manifest.json" -nt "$appearance_index/manifest.json" && \
    ! "$tracklets/manifest.json" -nt "$appearance_index/manifest.json" ]]; then
    echo "[4/6] Reusing pre-UOT SigLIP appearance index: $appearance_index"
else
    echo "[4/6] Encoding pre-UOT SigLIP appearance memory"
    "${mapping_python[@]}" \
        fact3r-map/scripts/build_siglip_observation_index.py \
        --keyframes "$keyframes" \
        --proposals "$proposals" \
        --tracklets "$tracklets" \
        --output "$appearance_index" \
        --device "$device" \
        --batch-size "$siglip_batch_size"
fi

mapping_has_appearance=false
if [[ -f "$mapping/manifest.json" ]] && \
    grep -Fq "\"source_appearance_index\": \"$appearance_index/manifest.json\"" \
        "$mapping/manifest.json" && \
    ! "$appearance_index/manifest.json" -nt "$mapping/manifest.json"; then
    mapping_has_appearance=true
fi
if [[ "$mapping_has_appearance" == true ]]; then
    echo "[5/6] Reusing delayed-commitment UOT entity map: $mapping"
else
    if [[ -f "$mapping/manifest.json" ]]; then
        echo "[5/6] Existing map predates appearance memory; rebuilding it"
    else
        echo "[5/6] Building appearance-aware visibility-conditioned entity map"
    fi
    "${mapping_python[@]}" \
        fact3r-map/scripts/run_visibility_residual_transport.py \
        --keyframes "$keyframes" \
        --proposals "$proposals" \
        --tracklets "$tracklets" \
        --appearance-index "$appearance_index" \
        --output "$mapping" \
        --delayed-commitment
fi

if [[ -f "$observations/manifest.json" && \
    ! "$mapping/manifest.json" -nt "$observations/manifest.json" ]]; then
    echo "[6/6] Reusing mapped SigLIP observation memory: $observations"
else
    echo "[6/6] Attaching persistent identities without re-encoding images"
    "${mapping_python[@]}" \
        fact3r-map/scripts/attach_siglip_mapping.py \
        --index "$appearance_index" \
        --mapping "$mapping" \
        --output "$observations"
fi

finalize_command=(
    "${mapping_python[@]}" fact3r-map/scripts/finalize_video_map.py
    --output "$output_root"
    --video "$video"
    --map-name "$map_name"
    --sequence-name "$sequence_name"
    --keyframes "$keyframes"
    --proposals "$proposals"
    --tracklets "$tracklets"
    --mapping "$mapping"
    --observations "$observations"
)
if [[ -n "$calibration" ]]; then
    finalize_command+=(--calibration "$calibration")
fi
"${finalize_command[@]}"

echo
echo "Fact3R video map complete"
echo "  map:          $output_root/map.json"
echo "  entities:     $mapping"
echo "  observations: $observations"
echo
echo "Fast query:"
echo "  conda run -n $sam2_environment python3 fact3r-map/scripts/query_video_map.py --map '$output_root' --query 'a clock' --mode fast --device '$device'"
echo "VLM query:"
echo "  conda run -n $sam2_environment python3 fact3r-map/scripts/query_video_map.py --map '$output_root' --query 'a clock' --mode vlm --device '$device'"
