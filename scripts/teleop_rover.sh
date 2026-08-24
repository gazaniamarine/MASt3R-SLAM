#!/usr/bin/env bash
# ============================================================
#  Keyboard teleop for the 6WD rover — runs ON THE GPU MACHINE.
#    ./scripts/teleop_rover.sh
#
#  Publishes geometry_msgs/Twist on /cmd_vel over plain DDS. The Pi runs
#  nothing for this: its micro-ROS agent already subscribes /cmd_vel and
#  relays to the ESP32. Verified reaching the Pi from this workstation.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}  → $*${NC}"; }
warn() { echo -e "${YELLOW}  ! $*${NC}"; }

source /opt/ros/humble/setup.bash 2>/dev/null || {
  echo -e "${RED}ROS 2 Humble not found at /opt/ros/humble${NC}" >&2; exit 1; }
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# rclpy's C extensions are built for python3.10. This shell's `python3` is
# conda's 3.12 (poseE), which fails with a _rclpy_pybind11 import error --
# so call the system interpreter explicitly rather than whatever is on PATH.
PY=/usr/bin/python3.10
[ -x "$PY" ] || { echo -e "${RED}$PY missing${NC}" >&2; exit 1; }
"$PY" -c "import rclpy" 2>/dev/null || {
  echo -e "${RED}$PY cannot import rclpy -- is /opt/ros/humble sourced?${NC}" >&2; exit 1; }

info "publishing /cmd_vel from this machine (domain $ROS_DOMAIN_ID)"
warn "teleop starts at 0.4 m/s -- tap 'z' TWICE for the ~0.2 m/s SLAM wants"
warn "pure rotation below 0.174 rad/s does NOTHING: the firmware zeroes any"
warn "per-wheel target under VEL_DEADBAND_MS=0.03 m/s (0.03*2/TRACK_WIDTH 0.345)"
echo

exec "$PY" scripts/teleop/ros2_keyboard_teleop.py "$@"
