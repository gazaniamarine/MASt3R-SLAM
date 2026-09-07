#!/usr/bin/env bash
set -euo pipefail
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_directory/.."
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/build_visual_subgoals.py "$@"
