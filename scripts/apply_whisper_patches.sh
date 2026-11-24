#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Apply or verify the complete source patch set required by Quoth's whisper build.
# Usage: apply_whisper_patches.sh --apply|--check WHISPER_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-}"
WHISPER_DIR="${2:-}"
PATCHES=(
  "$ROOT/packaging/patches/whisper-openvino-encode-failure.patch"
  "$ROOT/packaging/patches/whisper-invalid-ftype.patch"
  "$ROOT/packaging/patches/whisper-cpu-os-state.patch"
  "$ROOT/packaging/patches/whisper-vulkan-unsupported-device.patch"
  "$ROOT/packaging/patches/whisper-no-speech-prob.patch"
)

case "$MODE" in
  --apply|--check) ;;
  *)
    echo "usage: $0 --apply|--check WHISPER_DIR" >&2
    exit 2
    ;;
esac
[ -d "$WHISPER_DIR" ] || { echo "ERROR: whisper.cpp not found at $WHISPER_DIR" >&2; exit 1; }
command -v patch >/dev/null || { echo "ERROR: patch missing" >&2; exit 1; }

# Exact context only, in both modes: at patch's default fuzz a pure deletion also
# reverses onto unpatched source, and a patch applies to source that has drifted.
for patch_file in "${PATCHES[@]}"; do
  [ -f "$patch_file" ] || { echo "ERROR: whisper patch missing at $patch_file" >&2; exit 1; }
  if [ "$MODE" = "--apply" ]; then
    patch --fuzz=0 -d "$WHISPER_DIR" -p1 < "$patch_file"
  elif ! patch --force --silent --dry-run --reverse --fuzz=0 -d "$WHISPER_DIR" -p1 \
      < "$patch_file"; then
    echo "ERROR: whisper.cpp is missing ${patch_file#"$ROOT/"}." >&2
    printf 'Apply the full patch set: %q --apply %q\n' "$0" "$WHISPER_DIR" >&2
    exit 1
  fi
done
