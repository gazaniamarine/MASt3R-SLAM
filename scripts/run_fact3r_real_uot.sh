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
points_per_side="64"
points_per_batch="32"
max_seeds_per_batch="16"
siglip_batch_size="32"
qwen_batch_size="8"
qwen_model="Qwen/Qwen3-VL-Embedding-2B"
semantic_backend="siglip"
sam_refresh_seconds="5"
sam_discovery_model="facebook/sam2-hiera-large"
sam_tracking_model="facebook/sam2-hiera-small"
realtime_preset="false"
ultra_fast_preset="false"
high_recall_preset="false"
low_latency_mapping_preset="false"
subsecond_mapping_preset="false"
reuse_propagated_track_embeddings="false"
mast3r_pair_stride="1"
propagation_backend="sam2"
recall_biased_query="false"
pred_iou_threshold="0.75"
stability_score_threshold="0.85"
min_area_pixels="40"
min_area_fraction="0.0002"
min_component_pixels="20"

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
    echo "  --points-per-side N        dense SAM2 prompt grid (default: 64)"
    echo "  --sam-refresh-seconds S    dense discovery period (default: 5)"
    echo "  --sam-tracking-model ID    video model (default: Hiera-Small)"
    echo "  --realtime-preset          Small discovery + Tiny tracking, 48x48 grid"
    echo "  --ultra-fast-preset        Tiny discovery/tracking, 24x24 grid (low recall)"
    echo "  --high-recall-preset       Large discovery + Tiny tracking, 64x64 grid"
    echo "  --low-latency-mapping-preset  Large sparse discovery with cached semantics"
    echo "  --subsecond-mapping-preset  Large 32x32 discovery targeting >1 mapping FPS"
    echo "  --qwen-semantic           replace SigLIP with Qwen3-VL-Embedding throughout"
    echo "  --qwen-batch-size N       Qwen mask-embedding batch (default: 8)"
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
        --sam-refresh-seconds) sam_refresh_seconds=${2:?}; shift 2 ;;
        --sam-discovery-model) sam_discovery_model=${2:?}; shift 2 ;;
        --sam-tracking-model) sam_tracking_model=${2:?}; shift 2 ;;
        --realtime-preset) realtime_preset="true"; shift ;;
        --ultra-fast-preset) ultra_fast_preset="true"; shift ;;
        --high-recall-preset) high_recall_preset="true"; shift ;;
        --low-latency-mapping-preset) low_latency_mapping_preset="true"; shift ;;
        --subsecond-mapping-preset) subsecond_mapping_preset="true"; shift ;;
        --qwen-semantic) semantic_backend="qwen"; shift ;;
        --qwen-model) qwen_model=${2:?}; shift 2 ;;
        --qwen-batch-size) qwen_batch_size=${2:?}; shift 2 ;;
        --max-seeds-per-batch) max_seeds_per_batch=${2:?}; shift 2 ;;
        --siglip-batch-size) siglip_batch_size=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

selected_preset_count=0
if [[ "$realtime_preset" == "true" ]]; then selected_preset_count=$((selected_preset_count + 1)); fi
if [[ "$ultra_fast_preset" == "true" ]]; then selected_preset_count=$((selected_preset_count + 1)); fi
if [[ "$high_recall_preset" == "true" ]]; then selected_preset_count=$((selected_preset_count + 1)); fi
if [[ "$low_latency_mapping_preset" == "true" ]]; then selected_preset_count=$((selected_preset_count + 1)); fi
if [[ "$subsecond_mapping_preset" == "true" ]]; then selected_preset_count=$((selected_preset_count + 1)); fi
if [[ "$selected_preset_count" -gt 1 ]]; then
    echo "Choose only one segmentation preset" >&2
    exit 2
fi

if [[ "$low_latency_mapping_preset" == "true" ]]; then
    # Retain the high-recall discovery model, but amortise it and avoid
    # re-running expensive visual encoders for unchanged track observations.
    sam_discovery_model="facebook/sam2-hiera-large"
    sam_tracking_model="facebook/sam2.1-hiera-tiny"
    points_per_side="64"
    points_per_batch="64"
    max_seeds_per_batch="64"
    sam_refresh_seconds="10"
    pred_iou_threshold="0.70"
    stability_score_threshold="0.80"
    min_area_pixels="20"
    min_area_fraction="0.0001"
    min_component_pixels="10"
    siglip_batch_size="128"
    reuse_propagated_track_embeddings="true"
    mast3r_pair_stride="5"
    propagation_backend="optical-flow"
    recall_biased_query="true"
fi

if [[ "$subsecond_mapping_preset" == "true" ]]; then
    # Keep the accurate Large backbone while reducing the quadratic automatic
    # prompt grid. Optical flow and cached semantics carry evidence between
    # dense discoveries.
    sam_discovery_model="facebook/sam2-hiera-large"
    sam_tracking_model="facebook/sam2.1-hiera-tiny"
    points_per_side="32"
    points_per_batch="64"
    max_seeds_per_batch="64"
    sam_refresh_seconds="10"
    pred_iou_threshold="0.70"
    stability_score_threshold="0.80"
    min_area_pixels="20"
    min_area_fraction="0.0001"
    min_component_pixels="10"
    siglip_batch_size="128"
    reuse_propagated_track_embeddings="true"
    mast3r_pair_stride="10"
    propagation_backend="optical-flow"
    recall_biased_query="true"
fi

if [[ "$high_recall_preset" == "true" ]]; then
    sam_discovery_model="facebook/sam2-hiera-large"
    sam_tracking_model="facebook/sam2.1-hiera-tiny"
    points_per_side="64"
    points_per_batch="64"
    max_seeds_per_batch="16"
    sam_refresh_seconds="5"
    pred_iou_threshold="0.70"
    stability_score_threshold="0.80"
    min_area_pixels="20"
    min_area_fraction="0.0001"
    min_component_pixels="10"
fi

if [[ "$realtime_preset" == "true" ]]; then
    sam_discovery_model="facebook/sam2-hiera-small"
    sam_tracking_model="facebook/sam2.1-hiera-tiny"
    points_per_side="48"
    points_per_batch="64"
    max_seeds_per_batch="12"
    sam_refresh_seconds="5"
    pred_iou_threshold="0.70"
    stability_score_threshold="0.80"
    min_area_pixels="20"
    min_area_fraction="0.0001"
    min_component_pixels="10"
fi

if [[ "$ultra_fast_preset" == "true" ]]; then
    sam_discovery_model="facebook/sam2.1-hiera-tiny"
    sam_tracking_model="facebook/sam2.1-hiera-tiny"
    points_per_side="24"
    points_per_batch="64"
    max_seeds_per_batch="8"
    sam_refresh_seconds="10"
    pred_iou_threshold="0.75"
    stability_score_threshold="0.85"
    min_area_pixels="40"
    min_area_fraction="0.0002"
    min_component_pixels="20"
fi

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
if [[ "$semantic_backend" == "qwen" ]]; then
    appearance="$output/qwen_pre_uot"
    mapping="$output/image_uot_qwen"
    observations="$output/qwen_semantic_observations"
else
    appearance="$output/siglip_pre_uot"
    mapping="$output/image_uot"
    observations="$output/siglip_observations"
fi
matches="$output/mast3r_pair_matches"
mkdir -p "$output"
cd "$repository_root"
refresh_frames=$(awk -v fps="$sample_fps" -v seconds="$sam_refresh_seconds" 'BEGIN {value=int(fps*seconds+0.5); if (value < 1) value=1; print value}')

overall_start=$SECONDS
frames_seconds=0
proposals_seconds=0
tracklets_seconds=0
appearance_seconds=0
matches_seconds=0
mapping_seconds=0
attachment_seconds=0

if [[ ! -f "$frames/manifest.json" ]]; then
    echo "[1/7] Sampling MASt3R-aligned RGB frames; dense reconstruction is disabled"
    command=("${mast3r_python[@]}" fact3r-map/scripts/export_real_video_frames.py
        --video "$video" --output "$frames" --sample-fps "$sample_fps")
    if [[ -n "$max_frames" ]]; then command+=(--max-frames "$max_frames"); fi
    stage_start=$SECONDS
    "${command[@]}"
    frames_seconds=$((SECONDS - stage_start))
else
    echo "[1/7] Reusing sampled RGB frames"
fi

if [[ ! -f "$proposals/manifest.json" || ! -f "$tracklets/manifest.json" ]]; then
    echo "[2-3/7] Causal SAM2: dense discovery every $refresh_frames frames, video memory between"
    stage_start=$SECONDS
    "${sam2_python[@]}" fact3r-map/scripts/build_streaming_sam2_memory.py \
        --keyframes "$frames" \
        --proposals-output "$proposals" \
        --tracklets-output "$tracklets" \
        --discovery-model "$sam_discovery_model" \
        --tracking-model "$sam_tracking_model" \
        --device "$device" \
        --refresh-frames "$refresh_frames" \
        --points-per-side "$points_per_side" \
        --points-per-batch "$points_per_batch" \
        --max-seeds-per-batch "$max_seeds_per_batch" \
        --propagation-backend "$propagation_backend" \
        --pred-iou-threshold "$pred_iou_threshold" \
        --stability-score-threshold "$stability_score_threshold" \
        --min-area-pixels "$min_area_pixels" \
        --min-area-fraction "$min_area_fraction" \
        --min-component-pixels "$min_component_pixels"
    proposals_seconds=$((SECONDS - stage_start))
else
    echo "[2-3/7] Reusing causal SAM2 proposals and temporal links"
fi

if [[ ! -f "$appearance/manifest.json" ]]; then
    echo "[4/7] Encoding mask observations with $semantic_backend"
    stage_start=$SECONDS
    if [[ "$semantic_backend" == "qwen" ]]; then
        semantic_command=("${sam2_python[@]}"
            fact3r-map/scripts/build_qwen_embedding_observation_index.py
            --keyframes "$frames" --proposals "$proposals" --tracklets "$tracklets"
            --output "$appearance" --model "$qwen_model"
            --device-map auto --dtype bfloat16 --batch-size "$qwen_batch_size"
            --context-fraction 0.02 --outside-mask-alpha 0.0)
    else
        semantic_command=("${sam2_python[@]}"
            fact3r-map/scripts/build_siglip_observation_index.py
            --keyframes "$frames" --proposals "$proposals" --tracklets "$tracklets"
            --output "$appearance" --device "$device" --batch-size "$siglip_batch_size"
            --context-fraction 0.02 --outside-mask-alpha 0.0)
    fi
    if [[ "$reuse_propagated_track_embeddings" == "true" ]]; then
        semantic_command+=(--reuse-propagated-track-embeddings)
    fi
    "${semantic_command[@]}"
    appearance_seconds=$((SECONDS - stage_start))
else
    echo "[4/7] Reusing $semantic_backend observations"
fi

if [[ ! -f "$matches/manifest.json" ]]; then
    echo "[5/7] Computing adjacent-frame MASt3R reciprocal feature matches"
    stage_start=$SECONDS
    "${mast3r_python[@]}" fact3r-map/scripts/build_mast3r_pair_matches.py \
        --keyframes "$frames" --output "$matches" --device "cuda:$device" \
        --pair-stride "$mast3r_pair_stride"
    matches_seconds=$((SECONDS - stage_start))
else
    echo "[5/7] Reusing MASt3R pair matches"
fi

if [[ ! -f "$mapping/manifest.json" ]]; then
    echo "[6/7] Associating persistent identities with image-space UOT"
    stage_start=$SECONDS
    "${sam2_python[@]}" fact3r-map/scripts/run_image_uot_mapping.py \
        --proposals "$proposals" --tracklets "$tracklets" \
        --appearance-index "$appearance" --mast3r-matches "$matches" \
        --output "$mapping"
    mapping_seconds=$((SECONDS - stage_start))
else
    echo "[6/7] Reusing image-space UOT map"
fi

if [[ ! -f "$observations/manifest.json" ]]; then
    echo "[7/7] Attaching UOT identities to semantic observations"
    stage_start=$SECONDS
    "${sam2_python[@]}" fact3r-map/scripts/attach_siglip_mapping.py \
        --index "$appearance" --mapping "$mapping" --output "$observations"
    attachment_seconds=$((SECONDS - stage_start))
else
    echo "[7/7] Reusing searchable observations"
fi

echo
echo "Real-world reconstruction-free UOT map complete"
echo "  observations: $observations"
echo "  UOT map:      $mapping"
frame_count=$(awk -F: '/"frame_count"/ {gsub(/[^0-9]/, "", $2); print $2; exit}' "$proposals/manifest.json")
overall_seconds=$((SECONDS - overall_start))
if [[ -z "$frame_count" ]]; then frame_count=0; fi
effective_fps=$(awk -v frames="$frame_count" -v seconds="$overall_seconds" 'BEGIN {if (seconds > 0) printf "%.3f", frames / seconds; else print "inf"}')
realtime_factor=$(awk -v actual="$effective_fps" -v target="$sample_fps" 'BEGIN {if (target > 0) printf "%.3f", actual / target; else print "0"}')
echo
echo "Causal replay performance"
echo "  sampled input rate:   $sample_fps video FPS"
echo "  sampled frames:       $frame_count"
echo "  wall-clock time:      ${overall_seconds}s"
echo "  effective throughput: $effective_fps processed FPS"
echo "  real-time factor:     ${realtime_factor}x (>=1.0 keeps up)"
echo "  semantic backend:     $semantic_backend"
echo "  stage seconds: frames=$frames_seconds streaming_SAM2=$proposals_seconds Semantics=$appearance_seconds MASt3R=$matches_seconds UOT=$mapping_seconds attach=$attachment_seconds"
query_gate_arguments=()
if [[ "$recall_biased_query" == "true" ]]; then
    query_gate_arguments+=(
        --min-entity-margin 0.01
        --min-view-margin 0.005
        --min-supporting-views 1
    )
fi
if [[ -n "$query" ]]; then
    if [[ "$semantic_backend" == "qwen" ]]; then
        "${sam2_python[@]}" fact3r-map/scripts/query_qwen_memory_live.py \
            --index "$observations" --query "$query" --device-map auto \
            --dtype bfloat16 --top-k 20
    else
        "${sam2_python[@]}" fact3r-map/scripts/query_siglip_observations.py \
            --index "$observations" --query "$query" --device "$device" \
            --no-map-hard-negatives "${query_gate_arguments[@]}"
    fi
else
    echo "Query command:"
    if [[ "$semantic_backend" == "qwen" ]]; then
        echo "  conda run -n $sam2_environment python3 fact3r-map/scripts/query_qwen_memory_live.py --index '$observations' --device-map auto --dtype bfloat16 --top-k 50"
    else
        query_gate_text=""
        if [[ "$recall_biased_query" == "true" ]]; then
            query_gate_text=" --min-entity-margin 0.01 --min-view-margin 0.005 --min-supporting-views 1"
        fi
        echo "  conda run -n $sam2_environment python3 fact3r-map/scripts/query_siglip_observations.py --index '$observations' --query 'a chair' --device '$device' --no-map-hard-negatives$query_gate_text"
    fi
fi
