#!/usr/bin/env bash
# End-to-end per-category test: SigLIP+Qwen identification -> A* projection
# -> live NavDP execution in the real HM3D scene -> scored against GOAT's
# own ground truth. One command per category.
#
# Needs a NavDP policy server already running (see
# qwen3vl-2b-navdp/memory_nav/run_policy_server.py) -- this script does not
# start one, since it stays up across multiple category runs.
#
# Usage:
#   ./sim_bridge/run_goat_navdp_category.sh <category> [outbound|return] [query_phrase] [start_from_query]
#
# Examples:
#   ./sim_bridge/run_goat_navdp_category.sh statue outbound
#       # starts from the episode's own spawn point
#   ./sim_bridge/run_goat_navdp_category.sh chair outbound "the chair" "the statue"
#       # starts from wherever the statue leg ended up -- chains start -> statue -> chair
#   ./sim_bridge/run_goat_navdp_category.sh statue return "go back to the statue" "the chair"
#       # starts from wherever the chair leg ended up, recalls the statue anchor (no A*)
#       # -- start -> statue -> chair -> "go back to statue" all in one real session
#
# start_from_query, when given, must exactly match (after normalize_query's
# prefix-stripping) a query whose leg has ALREADY completed and has an entry
# in navdp_return_anchors.jsonl -- see run_return_sim.py's --start-from-anchor.
#
# IMPORTANT: the outbound leg's --query is also the anchor's lookup key
# (see memory_nav/anchor_store.py's normalize_query -- it only strips a
# fixed list of navigation prefixes like "go back to", it does NOT reconcile
# "a X" vs "the X"). The default here is "the <category>" on purpose, so a
# later "go back to the <category>" normalizes to the exact same string and
# actually finds the anchor. If you pass a custom outbound query phrase,
# make sure your return query's noun phrase matches it after prefix-stripping.
#
# Categories with real GOAT ground truth in this scene (y9hTuugGdiq,
# val_unseen): plant, nightstand, picture, vase, mirror, hanger, stair,
# statue, freezer, refrigerator, dishwasher, pillow, rug.

set -euo pipefail

CATEGORY="${1:?usage: $0 <category> [outbound|return] [query_phrase] [start_from_query]}"
MODE="${2:-outbound}"
QUERY="${3:-the $CATEGORY}"
START_FROM="${4:-}"
# Default to the 2B Qwen model: the NavDP policy server (Terminal 1) is
# holding VRAM on the same GPU the whole time this runs, and the 8B model
# needs ~18-20GB on its own -- contention there triggers accelerate's
# CPU-offload fallback ("parameters are on the meta device... offloaded to
# the cpu"), which makes every Qwen forward pass swap layers CPU<->GPU and
# can look like it's hung for minutes. Override with VLM_MODEL=... if you
# have GPU headroom (e.g. the policy server isn't running yet).
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"

REPO=/home/nahar4/Gazania/MASt3R-SLAM
NAVDP_REPO="${NAVDP_REPO:-$HOME/Gazania/qwen3vl-2b-navdp}"
RUN_DIR="$REPO/logs/goat/y9hTuugGdiq"
STEM="$RUN_DIR/map"
EPISODE="$REPO/datasets/goat/data/datasets/goat_bench/hm3d/v1/val_unseen/content/y9hTuugGdiq.json.gz"
SOCKET=/tmp/navdp_policy.sock
ANCHOR_FILE="$RUN_DIR/navdp_return_anchors.jsonl"
OUT_DIR="$RUN_DIR/navdp_sim_$(date +%Y%m%d_%H%M%S)_${CATEGORY}_${MODE}"
CAPTURE_DIR="$OUT_DIR/captures"
mkdir -p "$OUT_DIR"

echo "=== [1/4] ground truth lookup: $CATEGORY ==="
python3 "$REPO/sim_bridge/goat_ground_truth.py" --episode "$EPISODE" --category "$CATEGORY" \
    | tee "$OUT_DIR/ground_truth.txt"
GT_ARGS=()
while read -r x y z; do
    GT_ARGS+=(--gt-position "$x" "$y" "$z")
done < <(tail -n +2 "$OUT_DIR/ground_truth.txt")

# The episode's own start_position AND start_rotation -- a real intended
# facing direction, not an arbitrary yaw=0. Ignoring this previously meant
# the agent could spawn facing straight into a wall, which reads as an
# instant, permanent obstacle-avoidance stall and looks nothing like a
# planning or identification bug. Reuses the already-tested
# habitat_odometry.yaw_from_quaternion (same one
# scripts/execute_vlnce_return.py itself uses) rather than hand-rolling a
# quaternion-to-yaw conversion.
read -r SX SY SZ SYAW <<< "$(python3 -c "
import gzip, importlib.util, json, sys

spec = importlib.util.spec_from_file_location(
    'habitat_odometry', '$REPO/fact3r-map/fact3r/experiments/habitat_odometry.py')
habitat_odometry = importlib.util.module_from_spec(spec)
sys.modules['habitat_odometry'] = habitat_odometry  # @dataclass needs this registered first
spec.loader.exec_module(habitat_odometry)

d = json.load(gzip.open('$EPISODE'))
ep = d['episodes'][0]
p = ep['start_position']
yaw = habitat_odometry.yaw_from_quaternion(ep['start_rotation'])
print(p[0], p[1], p[2], yaw)
")"

START_ARGS=(--start-position "$SX" "$SY" "$SZ" --start-yaw "$SYAW")
if [ -n "$START_FROM" ]; then
    echo "=== starting from a prior leg's anchor: '$START_FROM' (episode spawn point ignored) ==="
    START_ARGS=(--start-from-anchor "$START_FROM")
fi

if [ "$MODE" = "outbound" ]; then
    echo "=== [2/4] locate: SigLIP shortlist -> Qwen verify -> memory check ==="
    REQUEST="$OUT_DIR/goal_request.json"
    conda run --no-capture-output -n SAM2 python3 "$REPO/fact3r-map/scripts/resolve_semantic_goal_verified.py" \
        --map "$STEM" --query "$QUERY" --output "$REQUEST" \
        --memory-file "$RUN_DIR/map_goal_memory.jsonl" \
        --vlm-model "$VLM_MODEL"

    echo "=== [3/4] project to standable goal (A* start) ==="
    GOAL="$OUT_DIR/goal.json"
    conda run --no-capture-output -n mast3r-slam python3 "$REPO/fact3r-map/scripts/project_semantic_goal.py" \
        --request "$REQUEST" --grid "$STEM.npy" --output "$GOAL"

    echo "=== [4/4] outbound: A* centerline, NavDP executes it live in habitat ==="
    conda run --no-capture-output -n habitat-vla python3 "$REPO/sim_bridge/run_return_sim.py" \
        --mode outbound --scene y9hTuugGdiq \
        --goal "$GOAL" --grid "$STEM.npy" \
        --socket "$SOCKET" --anchor-file "$ANCHOR_FILE" --query "$QUERY" \
        "${START_ARGS[@]}" \
        --max-steps 600 --predict-hz 5 --goal-radius 0.4 \
        --capture-dir "$CAPTURE_DIR" --capture-interval-m 1.0 \
        "${GT_ARGS[@]}" \
        --output "$OUT_DIR/result.json"

    echo "=== [captioning] BLIP/SAM2 landmark captions for the outbound frames ==="
    (cd "$NAVDP_REPO" && conda run --no-capture-output -n SAM2 python3 -m memory_nav.caption_landmarks \
        --capture-dir "$CAPTURE_DIR" --anchor-file "$ANCHOR_FILE" --query "$QUERY")
else
    echo "=== [2/4]-[3/4] skipped: return leg uses memory only, no SigLIP/Qwen/A* ==="
    echo "=== [4/4] return: recalled anchor, NavDP executes it live in habitat ==="
    conda run --no-capture-output -n habitat-vla python3 "$REPO/sim_bridge/run_return_sim.py" \
        --mode return --scene y9hTuugGdiq \
        --socket "$SOCKET" --anchor-file "$ANCHOR_FILE" --query "$QUERY" \
        "${START_ARGS[@]}" \
        --max-steps 600 --predict-hz 5 --goal-radius 0.4 \
        "${GT_ARGS[@]}" \
        --output "$OUT_DIR/result.json"
fi

echo "=== plotting ==="
python3 "$REPO/sim_bridge/plot_rollout.py" --result "$OUT_DIR/result.json" \
    --map-manifest "$STEM""_semantic.json" --output "$OUT_DIR/result.png"

echo ""
echo "done -> $OUT_DIR/result.json, $OUT_DIR/result.png"
