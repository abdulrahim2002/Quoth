#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Stage the relocatable libwhisper.so and its GGML backend libs into one directory.
# The build emits every shared library into bin/; staging them together beside the C
# shim lets their $ORIGIN RUNPATH resolve in the installed package.
#
# Usage: stage_libs.sh [BUILD_DIR] [DEST_DIR]
#   BUILD_DIR defaults to ~/whisper.cpp/build-quoth (scripts/build_whisper.sh output)
#   DEST_DIR  defaults to the repo lib/ (where whisper.py:_find_lib looks)
set -euo pipefail

BUILD_DIR="${1:-${BUILD_DIR:-$HOME/whisper.cpp/build-quoth}}"
DEST="${2:-${DEST:-$(cd "$(dirname "$0")/.." && pwd)/lib}}"
# whisper.cpp source tree (holds the LICENSE); the build dir's parent by convention.
WHISPER_DIR="${WHISPER_DIR:-$(dirname "$BUILD_DIR")}"
[ -d "$BUILD_DIR" ] || { echo "ERROR: build dir not found: $BUILD_DIR" >&2; exit 1; }
mkdir -p "$DEST"

# Clear stale engine libs (leave the shim, its source, and .gitkeep in place).
rm -f "$DEST"/libwhisper.so* "$DEST"/libggml*.so*

# Copy each lib's full symlink chain (-aP keeps the soname symlinks as links).
cp -aP "$BUILD_DIR"/bin/libwhisper.so* "$BUILD_DIR"/bin/libggml*.so* "$DEST/"

# whisper.cpp and ggml share one MIT LICENSE; ship it alongside the libs so the binary
# distribution carries its attribution (install-data bundles lib/). Mirrors how
# vendor_layer_shell.sh and vendor_openvino.sh vendor the other components' licenses.
lic="$(find "$WHISPER_DIR" -maxdepth 1 -iname 'LICENSE*' -print -quit)"
[ -n "$lic" ] || { echo "ERROR: whisper.cpp LICENSE not found in $WHISPER_DIR" >&2; exit 1; }
# The terms of third-party code the build compiles in follow it: YaRN, credited only in
# a ggml source comment, and Vulkan-Hpp (Apache-2.0 OR MIT), whose MIT terms apply.
yarn="Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng"
grep -qF "$yarn" "$WHISPER_DIR/ggml/src/ggml-cpu/ops.cpp" ||
  { echo "ERROR: YaRN credit not found in ggml/src/ggml-cpu/ops.cpp" >&2; exit 1; }
vulkan_include="$(sed -n 's/^Vulkan_INCLUDE_DIR:PATH=//p' "$BUILD_DIR/CMakeCache.txt")"
hpp="$vulkan_include/vulkan/vulkan.hpp"
[ -f "$hpp" ] || { echo "ERROR: vulkan.hpp not found under '$vulkan_include'" >&2; exit 1; }
grep -qF 'SPDX-License-Identifier: Apache-2.0 OR MIT' "$hpp" ||
  { echo "ERROR: $hpp is no longer offered under MIT" >&2; exit 1; }
khronos="$(grep -m1 -oE 'Copyright [0-9-]+ The Khronos Group Inc\.' "$hpp")" ||
  { echo "ERROR: no Khronos copyright in $hpp" >&2; exit 1; }
mit() {
  cat <<'EOF'
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
}
{
  cat "$lic"
  printf '\n\n# YaRN\nggml'"'"'s RoPE scaling, in libggml-base, the CPU backends and the Vulkan\n'
  printf 'shaders, is based on YaRN (https://github.com/jquesnelle/yarn):\n\n'
  printf 'MIT License\n\n%s\n\n' "$yarn"
  mit
  printf '\n\n# Vulkan-Hpp\nlibggml-vulkan compiles in Vulkan-Hpp (vulkan/vulkan.hpp), offered under\n'
  printf 'Apache-2.0 OR MIT and used here under MIT:\n\nMIT License\n\n%s\n\n' "$khronos"
  mit
} >"$DEST/LICENSE.whisper-cpp"

echo "Staged into $DEST:"
ls -1 "$DEST"/libwhisper.so* "$DEST"/libggml*.so* 2>/dev/null | sed 's#.*/#  #'
