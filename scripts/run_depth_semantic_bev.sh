#!/usr/bin/env bash
# Build one metric occupancy + persistent semantic BEV from a Fact3R memory.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)
runtime_environment="SAM2"
arguments=()

usage() {
    echo "Usage: $0 --index PATH (--odom CSV | --root DIR) --out STEM [options]"
    echo
    echo "  --index PATH        completed SigLIP/Qwen observation index"
    echo "  --odom CSV          wheel odometry with t,x,y,theta,v columns"
    echo "  --root DIR          alternatively discover odom_*.csv in this directory"
    echo "  --out STEM          output stem, for example logs/rover/run1/semantic_map"
    echo "  --fx PIXELS         focal length on the original video (default: 631)"
    echo "  --pitch DEGREES     camera pitch below horizontal (default: 2.75)"
    echo "  --cam-height METRES camera mount height (default: 0.5)"
    echo "  --scale VALUE       global metric-depth scale (default: 0.969)"
    echo "  --runtime-env NAME  environment with torch, transformers, scipy, and plyfile"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime-env) runtime_environment=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) arguments+=("$1"); shift ;;
    esac
done

cd "$repository_root"
conda run --no-capture-output -n "$runtime_environment" python3 \
    fact3r-map/scripts/build_depth_semantic_bev.py "${arguments[@]}"
