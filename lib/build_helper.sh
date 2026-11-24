#!/usr/bin/env bash
# Build libwhisper_helper.so, the C shim that keeps whisper's struct-by-value calls
# out of ctypes, linked against an existing libwhisper.so build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHISPER_DIR="${WHISPER_DIR:-$HOME/whisper.cpp}"
# The build tree to link libwhisper.so from; override per channel (e.g. the
# Vulkan build at build-quoth/). Headers come from the source tree (WHISPER_DIR).
WHISPER_BUILD_DIR="${WHISPER_BUILD_DIR:-$WHISPER_DIR/build}"

# The distro's build flags (makepkg, dpkg-buildflags) carry its hardening, such as
# RELRO and BIND_NOW; each variable holds several words.
# shellcheck disable=SC2086
gcc ${CPPFLAGS:-} ${CFLAGS:-} -shared -fPIC \
    -o "$SCRIPT_DIR/libwhisper_helper.so" \
    "$SCRIPT_DIR/whisper_helper.c" \
    -I "$WHISPER_DIR/include" \
    -I "$WHISPER_DIR/ggml/include" \
    ${LDFLAGS:-} \
    -L "$WHISPER_BUILD_DIR/bin" \
    -lwhisper \
    -Wl,-rpath,'$ORIGIN'

echo "Built: $SCRIPT_DIR/libwhisper_helper.so"

"$PYTHON" -I "$REPO_ROOT/scripts/check_install.py" --elf-root "$SCRIPT_DIR"
