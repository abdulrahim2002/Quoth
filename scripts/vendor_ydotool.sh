#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Vendor ydotool (ydotool, ydotoold) for the .deb. It sends the paste chord through
# /dev/uinput, which reaches every compositor, and Debian 13 packages it only in
# backports. Arch ships the pinned release and the Flatpak builds its own, so only the
# .deb calls this.
#
# The tools land in lib/bin/, with ydotool's AGPL-3.0 LICENSE as lib/LICENSE.ydotool.
# install-data stages both beside the native libs, where typer.py runs the bundled
# copy in place of the distro's. They link only libc.
#
# Usage: scripts/vendor_ydotool.sh [DEST_DIR]
#   DEST_DIR defaults to the repo lib/.
# Source of the tree (first match wins):
#   YDOTOOL_SRC  a pre-fetched ydotool git checkout; packaging recipes pass their
#                pinned clone so the build needs no network.
#   (unset)      dev convenience: git-clone the pinned tag and check its commit.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/lib}"
# shellcheck source=../packaging/pins.sh
. "$(cd "$(dirname "$0")/.." && pwd)/packaging/pins.sh"
YDOTOOL_VERSION="$QUOTH_YDOTOOL_TAG"
YDOTOOL_SRC="${YDOTOOL_SRC:-}"

command -v cmake >/dev/null || { echo "ERROR: cmake not found (build dep)" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -n "$YDOTOOL_SRC" ]; then
  src="$YDOTOOL_SRC"
else
  echo "==> cloning ydotool $YDOTOOL_VERSION (dev convenience; recipes pass YDOTOOL_SRC)"
  git clone --depth 1 --branch "$YDOTOOL_VERSION" \
    https://github.com/ReimuNotMoe/ydotool.git "$tmp/src"
  actual="$(git -C "$tmp/src" rev-parse HEAD)"
  [ "$actual" = "$QUOTH_YDOTOOL_COMMIT" ] || {
    echo "ERROR: ydotool $YDOTOOL_VERSION resolved to $actual, expected $QUOTH_YDOTOOL_COMMIT" >&2
    exit 1
  }
  src="$tmp/src"
fi
[ -f "$src/CMakeLists.txt" ] || { echo "ERROR: no CMakeLists.txt in $src" >&2; exit 1; }

# The version ydotoold reports comes from `git describe` in the working directory, so
# configure from the source tree. CMake ignores CPPFLAGS, where dpkg-buildflags puts
# _FORTIFY_SOURCE. Building just the two tools skips the man pages, which need scdoc.
# v1.0.4 declares a CMake minimum that CMake 4 rejects without the policy floor.
(cd "$src" && CFLAGS="${CPPFLAGS:-} ${CFLAGS:-}" cmake -S . -B "$tmp/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
cmake --build "$tmp/build" --target ydotool ydotoold

install -d "$DEST/bin"
install -m755 "$tmp/build/ydotool" "$tmp/build/ydotoold" "$DEST/bin/"
install -m644 "$src/LICENSE" "$DEST/LICENSE.ydotool"

echo "Vendored ydotool $("$DEST/bin/ydotoold" --version) into $DEST/bin"
