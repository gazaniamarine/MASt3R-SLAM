#!/usr/bin/env bash
# ============================================================
#  Drive the rover with the keyboard while the Pi records footage
#  for MASt3R-SLAM. Runs ON THE GPU MACHINE.
#
#    ./scripts/record_rover_run.sh [PI_HOST] [options]
#
#  This is the RECORD path (runbook Steps 4-5), automated end to end:
#  stop the camera service, lock exposure, start ffmpeg on the Pi, hand
#  you the keyboard, and on quit stop the recording, pull it over and
#  convert it to the PNG folder main.py wants.
#
#  Use this rather than launch_rover_slam.sh when you want RE-RUNNABLE
#  footage. Live mode streams only -- it archives nothing, so a bad
#  reconstruction there means driving the whole route again.
#
#  There is deliberately NO live preview while recording. ffmpeg's tee
#  muxer cannot both serve MJPEG-over-HTTP and write a file (the `listen`
#  option breaks stream mapping -- verified), and only one process can
#  hold /dev/video0. Frame the shot first with the single-still trick in
#  Step 2 of the runbook.
#
#  Options
#    --name NAME     dataset folder name       (default: next free runN)
#    --max SECONDS   hard cap on the take      (default: 900)
#    --slam          run main.py on the result when the transfer finishes
#    --keep-mkv      keep the .mkv here as well as on the Pi
#    --no-convert    pull the video but skip the PNG conversion
#
#  Env: PI_USER PI_PASS SIZE FPS EXPOSURE MAINS CONFIG CONDA_BASE
#
#  Examples
#    ./scripts/record_rover_run.sh
#    ./scripts/record_rover_run.sh --name lab_loop --slam
#    PI_PASS=... MAINS=60 EXPOSURE=167 ./scripts/record_rover_run.sh
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}  → $*${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ! $*${NC}"; }
die()  { echo -e "${RED}  ✗ $*${NC}" >&2; exit 1; }

PI_HOST=diana.local
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then PI_HOST="$1"; shift; fi
PI_USER=${PI_USER:-pi}
SIZE=${SIZE:-1280x720}
FPS=${FPS:-30}
EXPOSURE=${EXPOSURE:-100}
CONFIG=${CONFIG:-config/base.yaml}
MAXSEC=900
NAME=""
RUN_SLAM=0
KEEP_MKV=0
CONVERT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --name)       NAME="${2:-}"; shift 2 ;;
    --max)        MAXSEC="${2:-}"; shift 2 ;;
    --slam)       RUN_SLAM=1; shift ;;
    --keep-mkv)   KEEP_MKV=1; shift ;;
    --no-convert) CONVERT=0; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *)            die "unknown option: $1  (try --help)" ;;
  esac
done

# The folder loader dispatches on path COMPONENTS, so a run named e.g. "tum"
# is silently routed to the TUM reader and misparsed. Refuse those up front.
[[ "$MAXSEC" =~ ^[0-9]+$ ]] && [ "$MAXSEC" -gt 0 ] \
  || die "--max must be a positive whole number of seconds, got '$MAXSEC'"

RESERVED="tum euroc eth3d 7-scenes realsense webcam"
SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes $PI_USER@$PI_HOST"
REMOTE_MKV=""
RECORDING=0
CAM_STOPPED=0

# ---------------------------------------------------------------- cleanup
_CLEANED=0
stop_recording() {
  [ "$RECORDING" = "1" ] || return 0
  RECORDING=0
  # SIGINT, not SIGKILL: ffmpeg needs to write the matroska trailer or the
  # file has no duration index and some readers see a truncated stream.
  $SSH '[ -f /tmp/rover_rec.pid ] && kill -INT $(cat /tmp/rover_rec.pid) 2>/dev/null' >/dev/null 2>&1
  for _ in $(seq 1 15); do
    $SSH '[ -f /tmp/rover_rec.pid ] && kill -0 $(cat /tmp/rover_rec.pid) 2>/dev/null' >/dev/null 2>&1 || break
    sleep 1
  done
  $SSH 'pkill -x ffmpeg 2>/dev/null; rm -f /tmp/rover_rec.pid' >/dev/null 2>&1
}

stop_rover() {
  # Belt and braces. The teleop node zeroes /cmd_vel on its own exit; this
  # covers it dying some other way and leaving the wheels turning.
  ( source /opt/ros/humble/setup.bash 2>/dev/null || exit 0
    ROS_LOCALHOST_ONLY=0 timeout 5 ros2 topic pub --once /cmd_vel \
      geometry_msgs/msg/Twist '{}' >/dev/null 2>&1 ) || true
}

cleanup() {
  [ "$_CLEANED" = "1" ] && return 0
  _CLEANED=1
  stop_rover
  stop_recording
  if [ "$CAM_STOPPED" = "1" ]; then
    if [ -n "${PI_PASS:-}" ]; then
      $SSH "echo '$PI_PASS' | sudo -S -p '' systemctl start rover-camera.service" >/dev/null 2>&1 \
        && ok "rover-camera.service restarted"
    else
      warn "rover-camera.service left stopped. Restore when done:"
      echo "      ssh -t $PI_USER@$PI_HOST 'sudo systemctl start rover-camera.service'"
    fi
  fi
}
# INT gets its own handler: a trap body does not exit on its own, so without
# the explicit exit a Ctrl+C here would clean up and then carry on regardless.
# The teleop phase swaps this out and puts it back.
trap cleanup EXIT TERM
trap 'cleanup; exit 130' INT

echo -e "${CYAN}=== rover record: drive + capture for MASt3R-SLAM ===${NC}"

# ---------------------------------------------------------------- 1 reach
info "1/7  reaching $PI_HOST"
$SSH 'echo ok' >/dev/null 2>&1 || die "SSH failed. mDNS drops when the Pi reboots; \
try 'avahi-resolve -n $PI_HOST', or pass the IP as the first argument."
ok "$($SSH 'echo "$(hostname) up $(uptime -p)"')"

# Name the run only after we know the Pi is there, so a failed connect does
# not burn a run number.
if [ -z "$NAME" ]; then
  n=1; while [ -e "datasets/rover/run$n" ]; do n=$((n+1)); done; NAME="run$n"
fi
for r in $RESERVED; do
  [ "$NAME" = "$r" ] && die "'$NAME' is a reserved dataset name -- the loader would \
route it to the $r reader. Pick another --name."
done
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--name must be a plain filename, got '$NAME'"
OUTDIR="datasets/rover/$NAME"
[ -e "$OUTDIR" ] && die "$OUTDIR already exists -- pick another --name or remove it"
REMOTE_MKV="/home/$PI_USER/rover_$NAME.mkv"
LOCAL_MKV="/tmp/rover_$NAME.mkv"

# ---------------------------------------------------------------- 2 free the camera
info "2/7  stopping rover-camera.service (frees /dev/video0 for ffmpeg)"
if [ "$($SSH 'systemctl is-active rover-camera.service' 2>/dev/null)" = "inactive" ]; then
  ok "already inactive"
else
  if [ -n "${PI_PASS:-}" ]; then
    # -p "" : the default sudo prompt leaks into stderr and reads as failure.
    $SSH "echo '$PI_PASS' | sudo -S -p '' systemctl stop rover-camera.service" >/dev/null 2>&1
  else
    ssh -t "$PI_USER@$PI_HOST" 'sudo systemctl stop rover-camera.service' || true
  fi
  [ "$($SSH 'systemctl is-active rover-camera.service' 2>/dev/null)" = "inactive" ] \
    || die "camera service still active; ffmpeg will fail as 'Device or resource busy'"
  CAM_STOPPED=1
  ok "camera service inactive"
fi

# ---------------------------------------------------------------- 3 lock
info "3/7  locking camera (exposure=$EXPOSURE, MAINS=${MAINS:-50})"
$SSH "MAINS=${MAINS:-50} ~/lock_camera.sh /dev/video0 $EXPOSURE" 2>&1 | sed 's/^/      /' \
  || die "lock_camera.sh missing -- scp scripts/lock_camera.sh $PI_USER@$PI_HOST:~/"

# ---------------------------------------------------------------- 4 motors
info "4/7  ESP32 motor link"
DRIVE_OK=0
if $SSH 'ls /dev/ttyUSB0' >/dev/null 2>&1; then
  if $SSH 'systemctl is-active rover-agent.service' >/dev/null 2>&1; then
    ok "/dev/ttyUSB0 present, rover-agent.service running"; DRIVE_OK=1
  else
    warn "/dev/ttyUSB0 present but rover-agent.service is not running:"
    echo "      ssh -t $PI_USER@$PI_HOST 'sudo systemctl start rover-agent.service'"
  fi
else
  warn "no /dev/ttyUSB0 -- the ESP32 is not enumerated (unplugged, or no power)."
  warn "The wheels will NOT move. Recording still works: push the rover by hand,"
  warn "which is fine for SLAM as long as the camera translates smoothly."
fi

# ---------------------------------------------------------------- 5 record
# ~2 MB/s for 720p30 MJPEG; refuse to start a take we cannot store.
FREE_MB=$($SSH "df -Pm /home/$PI_USER | awk 'NR==2{print \$4}'" 2>/dev/null || echo 0)
NEED_MB=$(( MAXSEC * 2 ))
[ "$FREE_MB" -ge "$NEED_MB" ] 2>/dev/null \
  || warn "only ${FREE_MB} MB free on the Pi; a ${MAXSEC}s take needs about ${NEED_MB} MB"

info "5/7  recording to $PI_HOST:$REMOTE_MKV  ($SIZE @ ${FPS}fps, -c copy, cap ${MAXSEC}s)"
# -c copy: the Pi stores the camera's own MJPEG untouched, so it spends no CPU
# encoding and nothing is re-compressed before MASt3R sees it.
# setsid + exec: $$ inside the subshell BECOMES the ffmpeg pid, so the pidfile
# names ffmpeg itself and not a wrapper that would swallow our SIGINT.
# -nostdin: ffmpeg otherwise fights the ssh channel for stdin.
$SSH "pkill -x ffmpeg 2>/dev/null; rm -f /tmp/rover_rec.pid; sleep 1; \
  setsid bash -c 'echo \$\$ > /tmp/rover_rec.pid; \
    exec ffmpeg -nostdin -f v4l2 -input_format mjpeg -video_size $SIZE -framerate $FPS \
      -i /dev/video0 -t $MAXSEC -c copy -y $REMOTE_MKV' \
  > /tmp/rover_rec.log 2>&1 < /dev/null &" >/dev/null 2>&1 || true

REC_UP=0
for i in $(seq 1 20); do
  if $SSH "[ -f /tmp/rover_rec.pid ] && kill -0 \$(cat /tmp/rover_rec.pid) 2>/dev/null \
           && [ -s $REMOTE_MKV ]" 2>/dev/null; then
    RECORDING=1; REC_UP=1; ok "recording started (${i}s), file is growing"; break
  fi
  sleep 1
done
[ "$REC_UP" = "1" ] || { $SSH 'tail -5 /tmp/rover_rec.log' 2>&1 | sed 's/^/      /'
  die "recording never started. Full log: ssh $PI_USER@$PI_HOST 'cat /tmp/rover_rec.log'"; }

# ---------------------------------------------------------------- 6 drive
echo
echo -e "${GREEN}  === RECORDING. Drive now. Ctrl+C in the teleop screen ends the take. ===${NC}"
echo -e "${YELLOW}  Drive for SLAM, not for speed:${NC}"
echo "      • tap 'z' TWICE first -- teleop opens at 0.4 m/s, this wants ~0.2"
echo "      • never spin in place: zero parallax loses tracking. Turn on an arc."
echo "      • rotation under 0.174 rad/s is silently discarded by the firmware"
echo "      • aim at textured, lit surfaces; a blank wall gives the matcher nothing"
echo "      • return to where you started so the loop closes and drift cancels"
echo "      • 30-60 s is plenty for a first run"
echo
sleep 3

# The teleop node reads Ctrl+C as a raw \x03 keystroke and exits on its own,
# so normally no signal is involved. Trap INT anyway to cover the brief window
# each poll where the terminal is out of raw mode and would raise a real
# SIGINT into this whole process group -- without this, that race kills the
# script and the take is left running on the Pi.
TELEOP_RC=0
trap 'echo; info "interrupted -- ending the take"' INT
if [ "$DRIVE_OK" = "1" ]; then
  info "6/7  keyboard control (this machine publishes /cmd_vel)"
else
  info "6/7  keyboard control -- NO motor link; press Ctrl+C when the run is done"
fi
./scripts/teleop_rover.sh || TELEOP_RC=$?
# rc 0 means you quit teleop deliberately; anything else means it fell over,
# and you should not assume the whole route was driven before it did.
[ "$TELEOP_RC" = "0" ] || warn "teleop exited rc=$TELEOP_RC (crash, not a clean quit)"
trap 'cleanup; exit 130' INT
stop_rover

# ---------------------------------------------------------------- 7 collect
echo
info "7/7  ending the take and collecting it"
stop_recording
$SSH "[ -s $REMOTE_MKV ]" 2>/dev/null \
  || die "no video on the Pi at $REMOTE_MKV -- check /tmp/rover_rec.log there"
ok "on the Pi: $($SSH "du -h $REMOTE_MKV | cut -f1") $REMOTE_MKV"

info "pulling to $LOCAL_MKV"
scp -q "$PI_USER@$PI_HOST:$REMOTE_MKV" "$LOCAL_MKV" || die "scp failed"
ok "$(du -h "$LOCAL_MKV" | cut -f1) here"

if [ "$CONVERT" = "0" ]; then
  ok "video at $LOCAL_MKV; convert it with the Step 5 command when ready"
  exit 0
fi

command -v ffmpeg >/dev/null || die "ffmpeg not on this machine -- \
video is at $LOCAL_MKV, convert it once ffmpeg is installed"

# A PNG folder, not the .mkv directly: without torchcodec the MP4/video loader
# seeks frame by frame through OpenCV, which is slow and unreliable on
# inter-frame-compressed video. And the folder loader globs *.png ONLY -- JPEGs
# there give an empty dataset rather than an error.
info "converting to PNG frames in $OUTDIR"
mkdir -p "$OUTDIR"
ffmpeg -loglevel error -i "$LOCAL_MKV" -vsync 0 "$OUTDIR/%06d.png" \
  || die "conversion failed"
NFRAMES=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
[ "$NFRAMES" -gt 0 ] || die "conversion produced no frames"
ok "$NFRAMES frames in $OUTDIR ($(du -sh "$OUTDIR" | cut -f1))"

[ "$KEEP_MKV" = "1" ] || { rm -f "$LOCAL_MKV"; }

echo
echo -e "${GREEN}  Watch it before you trust it:${NC}  vlc $LOCAL_MKV"
echo "      sharp under motion? exposure steady? if not, redo the drive --"
echo "      no amount of tuning downstream recovers motion blur."
echo
echo -e "${GREEN}  Reconstruct:${NC}"
echo "      python main.py --dataset $OUTDIR --config $CONFIG"
echo "      (slow, or the rover crawled? raise dataset.subsample in $CONFIG)"
echo

if [ "$RUN_SLAM" = "1" ]; then
  if [ "${CONDA_DEFAULT_ENV:-}" != "mast3r-slam" ]; then
    info "activating conda env mast3r-slam"
    # conda's hook references unbound vars (ZSH_VERSION); relax nounset for it.
    set +u
    source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null \
      || die "conda hook not found; set CONDA_BASE=/path/to/miniconda3"
    conda activate mast3r-slam || die "conda env 'mast3r-slam' not found"
    set -u
  fi
  python -c "import lietorch" 2>/dev/null \
    || die "lietorch missing -- wrong env active ($(python -c 'import sys;print(sys.prefix)'))"
  info "running MASt3R-SLAM on $OUTDIR"
  python main.py --dataset "$OUTDIR" --config "$CONFIG"
fi
