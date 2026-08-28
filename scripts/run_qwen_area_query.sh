#!/usr/bin/env bash
# Build/reuse full-frame semantic areas, then query them without SAM2.

set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_directory/.." && pwd)
run_root=""
query=""
environment="SAM2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) run_root=${2:?}; shift 2 ;;
        --query) query=${2:?}; shift 2 ;;
        --environment) environment=${2:?}; shift 2 ;;
        *) echo "Usage: $0 --run-root PATH [--query TEXT]" >&2; exit 2 ;;
    esac
done
if [[ -z "$run_root" ]]; then
    echo "Usage: $0 --run-root PATH [--query TEXT]" >&2
    exit 2
fi
if [[ "$run_root" != /* ]]; then run_root="$repository_root/$run_root"; fi
frames="$run_root/frames"
memory="$run_root/qwen_scene_memory"
python_command=(conda run --no-capture-output -n "$environment" python3)
cd "$repository_root"

if [[ ! -f "$memory/manifest.json" ]]; then
    "${python_command[@]}" fact3r-map/scripts/build_qwen_scene_memory.py \
        --keyframes "$frames" --output "$memory" --batch-size 4
fi

command=("${python_command[@]}" fact3r-map/scripts/query_qwen_areas.py \
    --memory "$memory")
if [[ -n "$query" ]]; then command+=(--query "$query"); fi
"${command[@]}"
