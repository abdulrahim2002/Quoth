#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Vendor the trimmed OpenVINO runtime for the Intel NPU encoder.
# Extracts only the files Quoth's NPU path actually loads (~40 MiB) from an OpenVINO
# wheel, dropping the CPU/GPU plugins and every import frontend but IR, since Quoth
# ships pre-converted IR models. The "GPU" device is Vulkan (in libwhisper), never
# the OpenVINO GPU plugin.
#
# Keep-set rationale (from the NEEDED graph):
#   libopenvino.so*           core: libwhisper.so NEEDs it
#   libopenvino_ir_frontend.so*  reads the IR encoder model (.xml/.bin)
#   libopenvino_intel_npu_plugin.so  the NPU device (loaded by name at runtime)
#   libtbb* + libhwloc        threading stack the core/plugin depend on
#   cache.json                NPU compiled-blob cache (warm starts)
# The core's $ORIGIN rpath resolves all siblings in this one dir; no plugins.xml
# is needed (OpenVINO 2026 finds the plugin by convention).
#
# The result lands beside the package as lib/openvino/, which
# _native.openvino_lib_dirs() searches first, so __init__.py preloads the core
# (libwhisper.so NEEDs it) without an LD_LIBRARY_PATH change.
#
# Usage: scripts/vendor_openvino.sh [DEST_DIR]
#   DEST_DIR defaults to the repo lib/openvino.
# Source of the libs (first match wins):
#   OPENVINO_WHEEL_DIR  a dir containing openvino/libs (an extracted wheel or a
#                       pip site-packages). The packaging recipes pass their
#                       pinned, pre-extracted wheel here so build() needs no network.
#   (unset)             dev convenience: download the wheel packaging/pins.sh pins
#                       and check its SHA-256.
# Env: PYTHON.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/lib/openvino}"
# shellcheck source=../packaging/pins.sh
. "$(cd "$(dirname "$0")/.." && pwd)/packaging/pins.sh"
PYTHON="${PYTHON:-python3}"
OPENVINO_WHEEL_DIR="${OPENVINO_WHEEL_DIR:-}"

KEEP=(
  'libopenvino.so*'
  'libopenvino_ir_frontend.so*'
  'libopenvino_intel_npu_plugin.so*'
  'libtbb.so*' 'libtbbmalloc.so*' 'libtbbmalloc_proxy.so*' 'libtbbbind*.so*'
  'libhwloc.so*'
  'cache.json'
)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -n "$OPENVINO_WHEEL_DIR" ]; then
  libs="$OPENVINO_WHEEL_DIR/openvino/libs"
else
  echo "==> downloading $QUOTH_OPENVINO_WHEEL (dev convenience; recipes pass OPENVINO_WHEEL_DIR)"
  wheel="$tmp/$QUOTH_OPENVINO_WHEEL"
  curl -fsSL -o "$wheel" "$QUOTH_OPENVINO_URL"
  echo "$QUOTH_OPENVINO_SHA256  $wheel" | sha256sum -c -
  echo "==> extracting $(basename "$wheel")"
  "$PYTHON" -m zipfile -e "$wheel" "$tmp/unpacked"
  libs="$tmp/unpacked/openvino/libs"
fi

[ -d "$libs" ] || { echo "ERROR: openvino/libs not found ($libs)" >&2; exit 1; }

mkdir -p "$DEST"
rm -f "$DEST"/*.so* "$DEST"/cache.json
copied=0
for pat in "${KEEP[@]}"; do
  for f in "$libs"/$pat; do
    [ -e "$f" ] || continue
    cp -aP "$f" "$DEST/"
    copied=$((copied + 1))
  done
done
[ "$copied" -gt 0 ] || { echo "ERROR: keep-set matched nothing in $libs" >&2; exit 1; }

# Apache-2.0 compliance: keep OpenVINO's LICENSE and third-party notices, which the
# wheel holds in <root>/openvino-*.dist-info/licenses/, with the libs (install-data
# bundles the whole dir).
root="$(dirname "$(dirname "$libs")")"
licsrc="$(echo "$root"/openvino-*.dist-info/licenses)"
[ -f "$licsrc/LICENSE" ] || { echo "ERROR: OpenVINO LICENSE not found in $licsrc" >&2; exit 1; }
rm -rf "$DEST/licenses"
cp -a "$licsrc" "$DEST/licenses"

"$PYTHON" -I "$(cd "$(dirname "$0")" && pwd)/check_install.py" \
  --elf-root "$DEST"

echo "Vendored $copied file(s) into $DEST ($(du -sh "$DEST" | cut -f1))"
