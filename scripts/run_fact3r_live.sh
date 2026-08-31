#!/usr/bin/env bash
# Start causal Fact3R mapping from a webcam, stream URL, or paced video file.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)
sam2_environment="SAM2"
arguments=()

usage() {
    echo "Usage: $0 --source SOURCE --output PATH [options]"
    echo
    echo "  --source 0                  local webcam"
    echo "  --source rtsp://...         rover/network camera"
    echo "  --source /path/video.mp4    video replayed at real-time speed"
    echo "  --output PATH               live map output directory"
    echo "  --sample-fps FPS            processing target (default: 1)"
    echo "  --display                   show the live mask/entity overlay"
    echo "  --max-frames N              optional finite smoke test"
    echo "  --sam2-env NAME             environment containing SAM2/SigLIP"
    echo "  --no-realtime-pacing        process a video file as fast as possible"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sam2-env) sam2_environment=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) arguments+=("$1"); shift ;;
    esac
done

has_source="false"
has_output="false"
for argument in "${arguments[@]}"; do
    if [[ "$argument" == "--source" ]]; then has_source="true"; fi
    if [[ "$argument" == "--output" ]]; then has_output="true"; fi
done
if [[ "$has_source" != "true" || "$has_output" != "true" ]]; then
    usage >&2
    exit 2
fi

cd "$repository_root"
conda run --no-capture-output -n "$sam2_environment" python3 \
    fact3r-map/scripts/run_fact3r_live.py "${arguments[@]}"
