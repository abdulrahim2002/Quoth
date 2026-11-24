#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Vendor ThorVG (libthorvg.so) for every channel: no distro ships a usable one.
# Arch has it in the AUR only, Debian and Ubuntu not at all, so unlike
# gtk4-layer-shell there is no system-package path and every recipe calls this.
#
# Built with the lottie loader alone. That drops the png/jpg/webp decoders and with
# them every non-trivial dependency, leaving libstdc++/libm/libc: ~550 KB stripped
# that adds no system dependency on any channel. Lottie expressions are compiled out
# (-Dextra=""), which removes the vendored jerryscript interpreter; Quoth's own
# animations use none, and an animation file is untrusted input.
#
# The result lands in lib/ beside libwhisper, where _native.thorvg_lib_paths() looks
# first. It is loaded by ctypes at an absolute path, so it needs no preload in
# __init__.py and carries no $ORIGIN.
#
# Usage: scripts/vendor_thorvg.sh [DEST_DIR]
#   DEST_DIR defaults to the repo lib/.
# Source of the tree (first match wins):
#   THORVG_SRC  a pre-fetched thorvg source tree; packaging recipes pass their pinned
#               checkout so the build needs no network.
#   (unset)     dev convenience: git-clone the pinned tag and check its commit.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/lib}"
# The tag and commit come from the single source of truth (packaging/pins.sh).
# shellcheck source=../packaging/pins.sh
. "$(cd "$(dirname "$0")/.." && pwd)/packaging/pins.sh"
THORVG_VERSION="$QUOTH_THORVG_TAG"
THORVG_SRC="${THORVG_SRC:-}"
PYTHON="${PYTHON:-python3}"

for tool in meson ninja; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found (build dep)" >&2; exit 1; }
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -n "$THORVG_SRC" ]; then
  src="$THORVG_SRC"
else
  echo "==> cloning thorvg $THORVG_VERSION (dev convenience; recipes pass THORVG_SRC)"
  git clone --depth 1 --branch "$THORVG_VERSION" \
    https://github.com/thorvg/thorvg.git "$tmp/src"
  actual="$(git -C "$tmp/src" rev-parse HEAD)"
  [ "$actual" = "$QUOTH_THORVG_COMMIT" ] || {
    echo "ERROR: thorvg $THORVG_VERSION resolved to $actual, expected $QUOTH_THORVG_COMMIT" >&2
    exit 1
  }
  src="$tmp/src"
fi
[ -f "$src/meson.build" ] || { echo "ERROR: no meson.build in $src" >&2; exit 1; }

prefix="$tmp/prefix"
# capi: Quoth binds through ctypes, not C++. extra="": no expressions, no OpenMP.
meson setup "$tmp/build" "$src" --prefix="$prefix" --buildtype=release \
  -Dloaders="lottie" -Dbindings="capi" -Dengines="cpu" \
  -Dextra="" -Dsavers="" -Dtools="" -Dtests=false
ninja -C "$tmp/build"
ninja -C "$tmp/build" install

# Locate the built artifact (libdir varies by distro: lib/ or lib/<multiarch>/).
# The soname carries ThorVG's API generation: libthorvg-1.so.1.x.y. A 1 -> 2 bump
# would change the C API too, so pin the name rather than globbing it away.
so="$(find "$prefix" -name 'libthorvg-1.so*' -print -quit)"
[ -n "$so" ] || { echo "ERROR: libthorvg.so not built" >&2; exit 1; }

# The expression interpreter must not be in the shipped binary: it evaluates script
# carried by the animation file. The stripped build keeps no jerryscript names, but
# the expression bindings' After Effects names survive as string literals.
if grep -qa -e thisComp -e valueAtTime "$so"; then
  echo "FAIL: jerryscript is linked in, expressions were not compiled out" >&2; exit 1
fi

mkdir -p "$DEST"
rm -f "$DEST"/libthorvg-1.so*
# -aP keeps the soname symlink chain (.so -> .so.1 -> .so.1.x.y) as links.
cp -aP "$(dirname "$so")"/libthorvg-1.so* "$DEST/"

# Ship the MIT license beside the lib so its attribution travels with it
# (install-data bundles lib/), followed by the terms of the third-party code this
# build compiles in. Two of them live only in source comments: the FreeType License,
# whose binary-distribution clause asks for the credit below, and one MPL-2.0 file.
lic="$(find "$src" -maxdepth 1 -iname 'LICENSE*' -print -quit)"
[ -n "$lic" ] || { echo "ERROR: ThorVG LICENSE not found in $src" >&2; exit 1; }
rapidjson="$src/src/loaders/lottie/rapidjson/LICENSE"
rle="src/renderer/cpu_engine/tvgSwRle.cpp"
interpolator="src/loaders/lottie/tvgLottieInterpolator.cpp"
ftl="$(sed -n '/The FreeType Project LICENSE/,/\*\//{/^\*\/$/d;s/^ \{0,1\}\*\{0,1\} \{0,1\}//;p}' "$src/$rle")"
mpl="$(sed -n '/This Source Code Form/,/MPL\/2\.0/{s/^\/\* //;s/^ \* //;s/ \*\/$//;p}' "$src/$interpolator")"
[ -f "$rapidjson" ] || { echo "ERROR: RapidJSON LICENSE not found in $src" >&2; exit 1; }
grep -q 'based in part of the work of the' <<<"$ftl" ||
  { echo "ERROR: FreeType License not found in $rle" >&2; exit 1; }
[ -n "$mpl" ] || { echo "ERROR: MPL-2.0 notice not found in $interpolator" >&2; exit 1; }
{
  cat "$lic"
  printf '\n\n'
  cat "$rapidjson"
  printf '\n\n# FreeType\nThis software is based in part on the work of the FreeType Team\n'
  printf '(https://freetype.org). ThorVG'"'"'s %s carries\nthe FreeType License:\n\n%s\n' \
    "$rle" "$ftl"
  printf '\n\n# Mozilla Public License 2.0\nThorVG'"'"'s %s carries this notice:\n\n%s\n\n' \
    "$interpolator" "$mpl"
  printf 'Its source: https://github.com/thorvg/thorvg/blob/%s/%s\n' \
    "$QUOTH_THORVG_COMMIT" "$interpolator"
} >"$DEST/LICENSE.thorvg"

"$PYTHON" -I "$(cd "$(dirname "$0")" && pwd)/check_install.py" \
  --elf-root "$DEST"

echo "Vendored thorvg $THORVG_VERSION into $DEST:"
find "$DEST" -name 'libthorvg-1.so*' -printf '  %f\n'
