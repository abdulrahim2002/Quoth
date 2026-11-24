#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Vendor Intel's NPU driver stack for the .deb: the Level Zero loader and Intel's NPU
# user-mode driver with its compiler, which Debian and Ubuntu do not package. Arch
# packages them and the Flatpak bundles its own, so only the .deb calls this. The
# kernel driver and firmware stay the system's.
#
# The libraries land in lib/npu/, their notices in lib/npu/licenses/. The driver loads
# its compiler from its own directory, the compiler uses OpenVINO's TBB, and
# quoth/__init__.py points OpenVINO and the loader at this copy.
#
# Usage: scripts/vendor_npu_driver.sh [DEST_DIR]
#        scripts/vendor_npu_driver.sh --fetch DIR
#   DEST_DIR defaults to the repo lib/npu. --fetch only downloads the pinned files
#   into DIR, for a recipe to pass as NPU_DRIVER_SRC later.
# Source of the files (first match wins):
#   NPU_DRIVER_SRC  a dir filled by --fetch; packaging recipes pass it so the build
#                   needs no network.
#   (unset)         dev convenience: fetch them.
# Every file is checked against its packaging/pins.sh SHA-256 either way.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../packaging/pins.sh
. "$REPO/packaging/pins.sh"
PYTHON="${PYTHON:-python3}"

FILES=(
  "npu-driver.tar.gz $QUOTH_NPU_DRIVER_URL $QUOTH_NPU_DRIVER_SHA256"
  "level-zero.deb $QUOTH_LEVEL_ZERO_URL $QUOTH_LEVEL_ZERO_SHA256"
  "npu-driver-LICENSE.md $QUOTH_NPU_LICENSE_URL $QUOTH_NPU_LICENSE_SHA256"
  "npu-driver-third-party-programs.txt $QUOTH_NPU_NOTICES_URL $QUOTH_NPU_NOTICES_SHA256"
  "npu-compiler-LICENSE $QUOTH_NPU_COMPILER_LICENSE_URL $QUOTH_NPU_COMPILER_LICENSE_SHA256"
  "level-zero-LICENSE $QUOTH_LEVEL_ZERO_LICENSE_URL $QUOTH_LEVEL_ZERO_LICENSE_SHA256"
)

fetch() {
  mkdir -p "$1"
  for entry in "${FILES[@]}"; do
    read -r name url _ <<<"$entry"
    curl -fsSL -o "$1/$name" "$url"
  done
}

verify() {
  for entry in "${FILES[@]}"; do
    read -r name _ sha <<<"$entry"
    echo "$sha  $1/$name" | sha256sum -c -
  done
}

if [ "${1:-}" = "--fetch" ]; then
  [ -n "${2:-}" ] || { echo "usage: $0 --fetch DIR" >&2; exit 2; }
  fetch "$2"
  verify "$2"
  exit 0
fi

DEST="${1:-$REPO/lib/npu}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
src="${NPU_DRIVER_SRC:-}"
if [ -z "$src" ]; then
  echo "==> downloading the pinned NPU driver stack (dev convenience; recipes pass NPU_DRIVER_SRC)"
  src="$tmp/src"
  fetch "$src"
fi
verify "$src"

tar -xzf "$src/npu-driver.tar.gz" -C "$tmp"
mkdir "$tmp/root"
for deb in "$tmp/$QUOTH_NPU_UMD_DEB" "$tmp/$QUOTH_NPU_COMPILER_DEB" "$src/level-zero.deb"; do
  ar p "$deb" data.tar.gz | tar -xzf - -C "$tmp/root"
done

lib="$tmp/root/usr/lib/x86_64-linux-gnu"
mkdir -p "$DEST"
rm -f "$DEST"/*.so*
cp -P "$lib"/libze_loader.so.1* "$lib"/libze_intel_npu.so.1* \
  "$lib"/libopenvino_intel_npu_compiler*.so "$DEST/"
rm -rf "$DEST/licenses"
install -Dm644 "$src/npu-driver-LICENSE.md" "$DEST/licenses/intel-npu-driver/LICENSE.md"
install -Dm644 "$src/npu-driver-third-party-programs.txt" \
  "$DEST/licenses/intel-npu-driver/third-party-programs.txt"
install -Dm644 "$src/npu-compiler-LICENSE" "$DEST/licenses/intel-npu-compiler/LICENSE"
# The compiler compiles in LLVM and MLIR, whose notices its package does not carry.
install -Dm644 "$REPO/data/licenses/LICENSE.npu-compiler-llvm" \
  "$DEST/licenses/intel-npu-compiler/LICENSE.llvm"
install -Dm644 "$src/level-zero-LICENSE" "$DEST/licenses/level-zero/LICENSE"

"$PYTHON" -I "$REPO/scripts/check_install.py" --elf-root "$DEST"

echo "Vendored the NPU driver stack into $DEST ($(du -sh "$DEST" | cut -f1))"
