#!/usr/bin/env bash
# ============================================================
#  MASt3R-SLAM on the real 6WD rover — launcher.
#  Run on the GPU machine:
#    ./scripts/launch_rover_slam.sh [PI_HOST] [main.py args...]
#
#  Division of labour, deliberately: the Pi does the MINIMUM -- serve the
#  camera as MJPEG, relay /cmd_vel to the ESP32. Everything that thinks runs
#  here: MASt3R-SLAM on the GPU, keyboard teleop on this machine's ROS 2.
#
#  Why MJPEG-over-HTTP and not the ROS /image_raw topic: v4l2_camera
#  publishes RAW sensor_msgs/Image. At 1280x720x3 @30Hz that is ~83 MB/s,
#  which no WiFi link will carry, and this camera node advertises no
#  /compressed transport (checked: only /image_raw and /camera_info). The
#  camera already emits MJPEG, so streaming it costs the Pi zero re-encoding.
#
#  Examples:
#    ./scripts/launch_rover_slam.sh
#    ./scripts/launch_rover_slam.sh diana.local
#    PI_PASS=... ./scripts/launch_rover_slam.sh          # for the sudo step
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}  → $*${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ! $*${NC}"; }
die()  { echo -e "${RED}  ✗ $*${NC}" >&2; exit 1; }

PI_HOST=diana.local
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then PI_HOST="$1"; shift; fi
PI_USER=${PI_USER:-pi}
PORT=${PORT:-8090}
SIZE=${SIZE:-1280x720}
FPS=${FPS:-30}
EXPOSURE=${EXPOSURE:-100}
CONFIG=${CONFIG:-config/base.yaml}

SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes $PI_USER@$PI_HOST"

_CLEANED=0
cleanup() {
  [ "$_CLEANED" = "1" ] && return 0   # TERM fires this, then EXIT fires it again
  _CLEANED=1
  echo
  info "stopping the stream on $PI_HOST"
  $SSH '[ -f /tmp/rover_stream.pid ] && kill $(cat /tmp/rover_stream.pid) 2>/dev/null; pkill -x ffmpeg 2>/dev/null; rm -f /tmp/rover_stream.pid' >/dev/null 2>&1 || true
  warn "rover-camera.service left stopped. Restore when done:"
  echo "      ssh -t $PI_USER@$PI_HOST 'sudo systemctl start rover-camera.service'"
}
trap cleanup EXIT INT TERM

echo -e "${CYAN}=== MASt3R-SLAM rover launcher ===${NC}"

info "1/6  reaching $PI_HOST"
$SSH 'echo ok' >/dev/null 2>&1 || die "SSH failed. mDNS drops when the Pi reboots; \
try 'avahi-resolve -n $PI_HOST', or pass the IP as the first argument."
ok "$($SSH 'echo "$(hostname) up $(uptime -p)"')"

# The v4l2_camera node holds /dev/video0 exclusively and its unit is
# Restart=always, so pkill is useless -- it returns in 3s. Stop the UNIT.
info "2/6  stopping rover-camera.service (frees /dev/video0 for the stream)"
if [ -n "${PI_PASS:-}" ]; then
  # -p "" per FLASHING.md: the default prompt text leaks into stderr and
  # reads as a failure even when the command succeeded.
  $SSH "echo '$PI_PASS' | sudo -S -p '' systemctl stop rover-camera.service" >/dev/null 2>&1
else
  ssh -t "$PI_USER@$PI_HOST" 'sudo systemctl stop rover-camera.service' || true
fi
[ "$($SSH 'systemctl is-active rover-camera.service' 2>/dev/null)" = "inactive" ] \
  && ok "camera service inactive" \
  || die "camera service still active; ffmpeg will fail as 'Device or resource busy'"

info "3/6  locking camera (exposure=$EXPOSURE, MAINS=${MAINS:-50})"
$SSH "MAINS=${MAINS:-50} ~/lock_camera.sh /dev/video0 $EXPOSURE" 2>&1 | sed 's/^/      /' \
  || die "lock_camera.sh missing -- scp scripts/lock_camera.sh $PI_USER@$PI_HOST:~/"

info "4/6  starting MJPEG stream on $PI_HOST:$PORT ($SIZE @ ${FPS}fps, -c copy)"
# ffmpeg's http server EXITS when its client disconnects -- one hiccup from
# MASt3R and the stream would be gone permanently. Wrap it in a restart loop,
# and record the loop's pid so cleanup can kill the loop and not just the
# ffmpeg it will immediately respawn.
$SSH "pkill -x ffmpeg 2>/dev/null; [ -f /tmp/rover_stream.pid ] && kill \$(cat /tmp/rover_stream.pid) 2>/dev/null; sleep 1; \
  setsid bash -c 'echo \$\$ > /tmp/rover_stream.pid; while true; do \
    ffmpeg -f v4l2 -input_format mjpeg -video_size $SIZE -framerate $FPS \
      -i /dev/video0 -c copy -f mpjpeg -listen 1 http://0.0.0.0:$PORT; \
    sleep 1; done' > /tmp/rover_stream.log 2>&1 < /dev/null &" \
  >/dev/null 2>&1 || true

# Readiness is checked ON THE PI, never by connecting from here: ffmpeg's
# http server hands out a limited number of client slots, and a probe that
# dials in consumes one -- the probe would succeed and MASt3R would then get
# "Connection refused" against a server that had already served its client.
STREAM_UP=0
for i in $(seq 1 20); do
  if $SSH "pgrep -x ffmpeg >/dev/null && ss -ltn 2>/dev/null | grep -q ':$PORT '" 2>/dev/null; then
    ok "stream listening on the Pi after ${i}s (no client slot consumed)"; STREAM_UP=1; break
  fi
  sleep 1
done
[ "$STREAM_UP" = "1" ] || die "stream never came up. Check: ssh $PI_USER@$PI_HOST 'cat /tmp/rover_stream.log'"

info "5/6  ESP32 motor link"
if $SSH 'ls /dev/ttyUSB0' >/dev/null 2>&1; then
  if $SSH 'python3 -c "import serial;serial.Serial(\"/dev/ttyUSB0\",115200,timeout=1).close()"' >/dev/null 2>&1; then
    ok "/dev/ttyUSB0 opens -- start the agent on the Pi if it is not running"
  else
    warn "/dev/ttyUSB0 exists but will not open (CP210x stalls control transfers)."
    warn "Driving is unavailable. SLAM still works -- push the rover by hand."
  fi
else
  warn "no /dev/ttyUSB0 -- ESP32 not enumerated. Driving unavailable."
fi

# Activate the SLAM env ourselves rather than assuming the caller did -- the
# bare `python` on this box is conda poseE, which has no lietorch. conda's
# hook references unbound vars (ZSH_VERSION), so relax nounset while sourcing.
if [ "${CONDA_DEFAULT_ENV:-}" != "mast3r-slam" ]; then
  info "activating conda env mast3r-slam"
  set +u
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null \
    || die "conda hook not found; set CONDA_BASE=/path/to/miniconda3"
  conda activate mast3r-slam || die "conda env 'mast3r-slam' not found"
  set -u
fi
python -c "import lietorch" 2>/dev/null \
  || die "lietorch missing -- wrong env active ($(python -c 'import sys;print(sys.prefix)'))"

info "6/6  MASt3R-SLAM on the GPU  <-  http://$PI_HOST:$PORT"
echo -e "      ${GREEN}drive from another terminal:${NC}  ./scripts/teleop_rover.sh"
echo -e "      quit the viewer to save logs/rover_live_<timestamp>.{ply,txt}"
echo
# NOT exec: exec replaces this shell, which would discard the EXIT trap and
# leave ffmpeg serving on the Pi after the viewer closes.
OPENCV_FFMPEG_LOGLEVEL=${OPENCV_FFMPEG_LOGLEVEL:-8} \
  python main.py --dataset "http://$PI_HOST:$PORT" --config "$CONFIG" "$@"
SLAM_RC=$?
info "main.py exited rc=$SLAM_RC"
exit $SLAM_RC
