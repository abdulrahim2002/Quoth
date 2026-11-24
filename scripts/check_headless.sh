#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Check installed dictation end to end through a virtual microphone, a headless sway
# compositor and real text targets (a GTK window and nano in foot).
# Usage: check_headless.sh MODEL AUDIO EVIDENCE_DIR
set -euo pipefail
model=$(realpath "$1")
audio=$(realpath "$2")
evidence=$(realpath "$3")
session_root=$(mktemp -d)
export XDG_RUNTIME_DIR="$session_root/runtime" XDG_CONFIG_HOME="$session_root/config"
export XDG_DATA_HOME="$session_root/data" XDG_STATE_HOME="$session_root/state"
export WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1 WLR_RENDERER=pixman
export WLR_HEADLESS_OUTPUTS=1 GSK_RENDERER=cairo GDK_BACKEND=wayland GTK_A11Y=none
export QUOTH_LOG_LEVEL=DEBUG
unset DISPLAY WAYLAND_DISPLAY SWAYSOCK
mkdir -p "$XDG_RUNTIME_DIR" "$XDG_CONFIG_HOME/quoth" "$XDG_DATA_HOME/quoth/models"
chmod 700 "$XDG_RUNTIME_DIR"
# Services the session bus starts, dconf among them, must share these directories,
# or the settings written below land in a database Quoth never reads.
dbus-update-activation-environment \
  XDG_RUNTIME_DIR XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME
pids=()
cleanup() {
  trap - EXIT
  grim "$evidence/final.png" 2>/dev/null || true
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  if [[ -f "$XDG_STATE_HOME/quoth/quoth.log" ]]; then
    cp "$XDG_STATE_HOME/quoth/quoth.log" "$evidence/quoth-state.log"
  fi
  rm -rf -- "$session_root"
}
trap cleanup EXIT
wait_for() {
  local attempt
  for ((attempt=0; attempt<600; attempt++)); do
    if "$@"; then return; fi
    sleep 0.1
  done
  cat "$evidence"/*.log >&2
  return 1
}
pipewire > "$evidence/pipewire.log" 2>&1 & pids+=("$!")
wait_for test -S "$XDG_RUNTIME_DIR/pipewire-0"
wireplumber > "$evidence/wireplumber.log" 2>&1 & pids+=("$!")
pipewire-pulse > "$evidence/pipewire-pulse.log" 2>&1 & pids+=("$!")
# Wait for the socket, not a client: libpulse creates pulse/ itself, and one made
# between pipewire-pulse's check and its mkdir makes the server exit.
wait_for test -S "$XDG_RUNTIME_DIR/pulse/native"
pactl load-module module-null-sink sink_name=quoth_fixture rate=16000 channels=1 channel_map=mono
pactl load-module module-remap-source master=quoth_fixture.monitor source_name=quoth_mic channels=1 channel_map=mono
cat > "$evidence/sway.conf" <<'CONF'
xwayland disable
output HEADLESS-1 mode 1280x720
focus_follows_mouse no
CONF
sway -c "$evidence/sway.conf" > "$evidence/sway.log" 2>&1 & pids+=("$!")
wait_for sh -c 'find "$XDG_RUNTIME_DIR" -maxdepth 1 -name "wayland-*" -type s | grep -q .'
export WAYLAND_DISPLAY
WAYLAND_DISPLAY=$(find "$XDG_RUNTIME_DIR" -maxdepth 1 -name 'wayland-*' -type s -printf '%f\n' -quit)
export SWAYSOCK
SWAYSOCK=$(find "$XDG_RUNTIME_DIR" -maxdepth 1 -name 'sway-ipc.*.sock' -print -quit)
ln -s "$model" "$XDG_DATA_HOME/quoth/models/ggml-base-q8_0.bin"
while read -r key value; do
  gsettings set dev.quoth.Quoth "$key" "$value"
done <<'SETTINGS'
onboarding-complete true
calibration-offered true
device 'CPU'
model 'base'
language 'en'
mode 'tap-to-toggle'
mic-source 'quoth_mic'
launch-at-login false
sound-feedback 'off'
notification-feedback 'off'
SETTINGS
quoth > "$evidence/quoth.log" 2>&1 & pids+=("$!")
wait_for grep -q 'Quoth ready' "$evidence/quoth.log"
window_ready() {
  swaymsg -t get_tree | jq -e --arg id "$1" '.. | objects | select(.app_id? == $id)' >/dev/null
}
count_matches() {
  local count
  count=$(grep -c "$1" "$evidence/quoth.log" || true)
  test "$count" -ge "$2"
}
status=0
cycle=0
for target in gtk foot; do
  cycle=$((cycle + 1))
  if [[ "$target" == gtk ]]; then
    python -I "$(dirname "$0")/gtk_dictation_target.py" "$evidence/gtk.txt" > "$evidence/gtk.log" 2>&1 &
    target_id=dev.quoth.QuothTestTarget
  else
    touch "$evidence/foot.txt"
    foot --app-id=quoth-foot nano --saveonexit "$evidence/foot.txt" > "$evidence/foot.log" 2>&1 &
    target_id=quoth-foot
  fi
  target_pid=$!
  pids+=("$target_pid")
  wait_for window_ready "$target_id"
  swaymsg "[app_id=\"$target_id\"] focus" >/dev/null
  quoth --toggle
  wait_for count_matches 'Recording started' "$cycle"
  paplay --device=quoth_fixture "$audio"
  quoth --toggle
  wait_for count_matches 'Chord-sender .* -> exit 0' "$cycle"
  sleep 1
  grim "$evidence/$target.png"
  if [[ "$target" == gtk ]]; then
    swaymsg "[app_id=\"$target_id\"] kill" >/dev/null
  else
    wtype -M ctrl -k x -m ctrl
  fi
  wait "$target_pid"
  if ! python - "$evidence/$target.txt" <<'PY'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
normalized = " ".join(re.findall(r"[a-z]+", text.lower()))
expected = "ask not what your country can do for you ask what you can do for your country"
print(f"{Path(sys.argv[1]).name}: {text!r}")
assert expected in normalized, "dictation did not reach the target"
PY
  then status=1; fi
done
quoth --quit
exit "$status"
