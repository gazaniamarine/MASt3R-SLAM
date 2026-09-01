#!/usr/bin/env bash
# Keep SigLIP + Qwen3-VL resident for repeated verified semantic-BEV queries.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)
runtime_environment="SAM2"
arguments=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime-env) runtime_environment=${2:?}; shift 2 ;;
        *) arguments+=("$1"); shift ;;
    esac
done

cd "$repository_root"
conda run --no-capture-output -n "$runtime_environment" python3 \
    fact3r-map/scripts/query_semantic_bev_vlm_live.py "${arguments[@]}"
