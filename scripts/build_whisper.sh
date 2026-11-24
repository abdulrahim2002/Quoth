#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Build a portable, relocatable libwhisper.so for Quoth.
#
# Builds whisper with runtime-selected CPU modules and the Vulkan GPU module.
# The OpenVINO encoder path is linked in only with WITH_OPENVINO=ON.
# Baseline libraries avoid host-native tuning; GGML selects eligible CPU kernels.
# Sibling-relative RPATH keeps the native bundle relocatable.
#
# Vulkan shaders are compiled to SPIR-V at build time (needs glslc) and baked into
# the binary; the running app needs only the Vulkan loader and an ICD driver.
#
# This is the shared build engine: the Makefile's `build` target runs it, and every
# packaging recipe builds through that target.
#
# Usage:
#   scripts/build_whisper.sh                   # configure, build and verify
#   scripts/build_whisper.sh --configure-only  # stop after configure (toolchain check)
# Env overrides: WHISPER_DIR, BUILD_DIR, WITH_OPENVINO, OpenVINO_DIR, JOBS, PYTHON;
# WHISPER_TEST_MODEL also runs the model boundary check.
set -euo pipefail

CONFIGURE_ONLY=0
[ "${1:-}" = "--configure-only" ] && CONFIGURE_ONLY=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHISPER_DIR="${WHISPER_DIR:-$HOME/whisper.cpp}"
BUILD_DIR="${BUILD_DIR:-$WHISPER_DIR/build-quoth}"
WITH_OPENVINO="${WITH_OPENVINO:-OFF}"   # ON links the Intel NPU encoder
OpenVINO_DIR="${OpenVINO_DIR:-$HOME/openvino-env/lib/python3.12/site-packages/openvino/cmake}"
JOBS="${JOBS:-$(nproc)}"

# --- toolchain sanity ---
command -v cmake >/dev/null || { echo "ERROR: cmake missing (apt: cmake)" >&2; exit 1; }
command -v glslc >/dev/null \
  || { echo "ERROR: glslc not found. whisper.cpp's Vulkan backend (GGML_VULKAN=ON) needs glslc (pkg: shaderc) to compile shaders to SPIR-V at build; glslangValidator alone won't satisfy its CMake." >&2; exit 1; }
[ -d "$WHISPER_DIR" ] || { echo "ERROR: whisper.cpp not found at $WHISPER_DIR" >&2; exit 1; }
"$REPO_ROOT/scripts/apply_whisper_patches.sh" --check "$WHISPER_DIR"
[ ! -e "$BUILD_DIR/bin/libggml-cpu.so" ] && [ ! -L "$BUILD_DIR/bin/libggml-cpu.so" ] \
  || { echo "ERROR: stale monolithic CPU library in $BUILD_DIR/bin. Use a clean BUILD_DIR." >&2; exit 1; }

args=(
  -S "$WHISPER_DIR" -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DGGML_VULKAN=ON
  -DGGML_NATIVE=OFF
  -DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON
  -DWHISPER_OPENVINO="$WITH_OPENVINO"
  -DWHISPER_BUILD_EXAMPLES=OFF
  -DWHISPER_BUILD_TESTS=OFF
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  # Resolve dependencies next to the .so and never bake build or link dirs into RPATH,
  # so the bundle stays relocatable.
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  -DCMAKE_INSTALL_RPATH='$ORIGIN'
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
)
if [ "$WITH_OPENVINO" = "ON" ]; then
  args+=( -DOpenVINO_DIR="$OpenVINO_DIR" )
fi

echo "==> configure ($WHISPER_DIR -> $BUILD_DIR; Vulkan=ON, OpenVINO=$WITH_OPENVINO, NATIVE=OFF)"
cmake "${args[@]}"

echo "==> CPU compile policy check"
"$PYTHON" -I "$REPO_ROOT/scripts/check_install.py" \
  --compile-commands "$BUILD_DIR/compile_commands.json"

if [ "$CONFIGURE_ONLY" = "1" ]; then
  echo "OK: configure succeeded (toolchain + OpenVINO discovery fine). Stopping (--configure-only)."
  exit 0
fi

echo "==> build (-j$JOBS)"
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

echo "==> ELF relocatability check"
"$PYTHON" -I "$REPO_ROOT/scripts/check_install.py" --elf-root "$BUILD_DIR/bin"
if [ -n "${WHISPER_TEST_MODEL:-}" ]; then
  "$PYTHON" -I "$REPO_ROOT/scripts/check_whisper_model_boundary.py" \
    "$BUILD_DIR/bin/libwhisper.so" "$WHISPER_TEST_MODEL"
fi
echo "Built: $BUILD_DIR/bin/libwhisper.so (+ bin/libggml*.so)"
