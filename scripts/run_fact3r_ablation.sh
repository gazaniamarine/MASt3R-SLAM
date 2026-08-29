#!/usr/bin/env bash
# Reuse one completed real-video run to ablate image-UOT association cues.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)

run_root=""
output=""
sam2_environment="SAM2"

usage() {
    echo "Usage: $0 --run-root PATH [--output PATH] [--sam2-env NAME]"
    echo
    echo "The run root must contain sam2_proposals, sam2_tracklets,"
    echo "mast3r_pair_matches, and siglip_pre_uot. qwen_pre_uot is optional."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) run_root=${2:?}; shift 2 ;;
        --output) output=${2:?}; shift 2 ;;
        --sam2-env) sam2_environment=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$run_root" ]]; then
    echo "--run-root is required" >&2
    exit 2
fi
if [[ "$run_root" != /* ]]; then run_root="$repository_root/$run_root"; fi
if [[ -z "$output" ]]; then
    output="$run_root/ablation"
elif [[ "$output" != /* ]]; then
    output="$repository_root/$output"
fi

proposals="$run_root/sam2_proposals"
tracklets="$run_root/sam2_tracklets"
matches="$run_root/mast3r_pair_matches"
siglip="$run_root/siglip_pre_uot"
qwen="$run_root/qwen_pre_uot"
for required in "$proposals" "$tracklets" "$matches" "$siglip"; do
    if [[ ! -f "$required/manifest.json" ]]; then
        echo "Missing prerequisite: $required/manifest.json" >&2
        exit 2
    fi
done

python_command=(conda run -n "$sam2_environment" python3)
mkdir -p "$output"
cd "$repository_root"

run_variant() {
    local label=$1
    local appearance=$2
    local cues=$3
    local destination="$output/$label"
    if [[ -f "$destination/manifest.json" ]]; then
        echo "Reusing $label"
        return
    fi
    echo "Running $label with cues=$cues"
    "${python_command[@]}" fact3r-map/scripts/run_image_uot_mapping.py \
        --proposals "$proposals" \
        --tracklets "$tracklets" \
        --appearance-index "$appearance" \
        --mast3r-matches "$matches" \
        --output "$destination" \
        --cues "$cues"
}

run_variant siglip_appearance "$siglip" appearance
run_variant siglip_sam2 "$siglip" appearance,sam2
run_variant siglip_mast3r "$siglip" appearance,mast3r
run_variant siglip_full "$siglip" appearance,sam2,mast3r

variants=(
    --variant "SigLIP-A=$output/siglip_appearance"
    --variant "SigLIP-A+S=$output/siglip_sam2"
    --variant "SigLIP-A+M=$output/siglip_mast3r"
    --variant "SigLIP-A+S+M=$output/siglip_full"
)
if [[ -f "$qwen/manifest.json" ]]; then
    run_variant qwen_full "$qwen" appearance,sam2,mast3r
    variants+=(--variant "Qwen-A+S+M=$output/qwen_full")
else
    echo "Qwen pre-UOT index absent; skipping the semantic-backend ablation"
fi

"${python_command[@]}" fact3r-map/scripts/evaluate_image_uot_ablation.py \
    "${variants[@]}" \
    --output "$output/report"

echo
echo "Ablation complete"
echo "Report: $output/report/report.md"
echo "CSV:    $output/report/metrics.csv"
echo "JSON:   $output/report/metrics.json"
