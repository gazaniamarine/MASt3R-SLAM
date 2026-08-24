#!/usr/bin/env bash
# Lock the rover webcam's exposure, white balance, gain and flicker handling.
#
# Runs ON THE PI, not the workstation. Copy it over with:
#   scp scripts/lock_camera.sh pi@diana.local:~/
#
# Usage:  ./lock_camera.sh [device] [exposure]
#   device    v4l2 node, default /dev/video0
#   exposure  in 100 us units -- 100 = 10 ms. Default 100.
# Env:
#   MAINS=50|60  local mains frequency, for the anti-flicker control (default 50)
#   GAIN=<n>     0-255, raise if 10 ms is too dark. Prefer this over long exposure.
#   WB=<kelvin>  white balance, default 4500
#   FOCUS=<n>    only applied if the camera actually has a focus motor
#
# UVC control names changed around kernel 5.10, so this probes for whichever
# spelling your kernel exposes instead of assuming one. Auto modes are always
# switched off before the corresponding manual value is written: the reverse
# order is silently discarded while the auto mode still owns the control.
#
# Cameras with no focus controls (e.g. the fixed-focus Brio 100) simply skip
# that group -- absence is not an error, it is one less thing that can drift.

set -u

DEV="${1:-/dev/video0}"
EXPOSURE="${2:-100}"
FOCUS="${FOCUS:-0}"
WB="${WB:-4500}"
MAINS="${MAINS:-50}"

command -v v4l2-ctl >/dev/null || {
  echo "v4l2-ctl not found -- sudo apt install -y v4l-utils" >&2; exit 1; }
[ -e "$DEV" ] || { echo "no such device: $DEV" >&2; exit 1; }

CTRLS="$(v4l2-ctl -d "$DEV" --list-ctrls 2>/dev/null)"
[ -n "$CTRLS" ] || {
  echo "$DEV exposes no controls. It is probably the metadata node rather than" >&2
  echo "the capture node -- run 'v4l2-ctl --list-devices' and try the other one." >&2
  exit 1; }

has()  { grep -qE "^[[:space:]]*$1[[:space:]]+0x" <<<"$CTRLS"; }
pick() { for n in "$@"; do if has "$n"; then echo "$n"; return; fi; done; }

put() {  # put <control> <value>
  [ -n "$1" ] || return 0
  if v4l2-ctl -d "$DEV" -c "$1=$2" 2>/dev/null; then
    printf '  ok      %s = %s\n' "$1" "$2"
  else
    printf '  FAILED  %s = %s\n' "$1" "$2"
  fi
}

AF=$(pick focus_automatic_continuous focus_auto)
FA=$(pick focus_absolute)
AE=$(pick auto_exposure exposure_auto)
EA=$(pick exposure_time_absolute exposure_absolute)
EDF=$(pick exposure_dynamic_framerate)
AWB=$(pick white_balance_automatic white_balance_temperature_auto)
WBT=$(pick white_balance_temperature)
GA=$(pick gain)
BLC=$(pick backlight_compensation)
PLF=$(pick power_line_frequency)

echo "locking $DEV"

# Order matters throughout: auto off first, manual value second.
if [ -n "$AF" ] || [ -n "$FA" ]; then
  echo "focus:"
  put "$AF" 0
  put "$FA" "$FOCUS"
else
  echo "focus: no controls -- fixed-focus camera, nothing can drift. good."
fi

echo "exposure:"          # 1 = Manual Mode on both old and new spellings.
put "$AE" 1               # 3 = Aperture Priority, i.e. auto. Do not use 0.
put "$EA" "$EXPOSURE"
put "$EDF" 0              # never let the camera drop fps to buy light

echo "white balance:"
put "$AWB" 0
put "$WBT" "$WB"

echo "other adaptive processing:"
put "$BLC" 0              # backlight compensation is another feedback loop
[ "$MAINS" = "60" ] && put "$PLF" 2 || put "$PLF" 1
[ -n "${GAIN:-}" ] && put "$GA" "$GAIN"

echo
echo "readback -- these are the values that will actually be recorded:"
for c in "$FA" "$AE" "$EA" "$EDF" "$AWB" "$WBT" "$GA" "$BLC" "$PLF"; do
  [ -n "$c" ] || continue
  printf '  %-30s %s\n' "$c" "$(v4l2-ctl -d "$DEV" -C "$c" 2>/dev/null | cut -d' ' -f2-)"
done

echo
echo "Any manual control still showing flags=inactive in --list-ctrls is still"
echo "owned by an auto mode. Re-run this AFTER ffmpeg starts if values drift."
