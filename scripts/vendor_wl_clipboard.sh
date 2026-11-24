#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Vendor wl-clipboard (wl-copy, wl-paste) for the .deb. Every Debian and Ubuntu target
# ships 2.2.1, which lacks `wl-copy --sensitive` (the password-manager hint that keeps
# a transcript out of clipboard history) and ext-data-control-v1 (the only
# data-control protocol on KWin 6.5 and later). Arch ships a release with both and the
# Flatpak builds its own, so only the .deb calls this.
#
# The tools land in lib/bin/, with wl-clipboard's GPL-3.0 COPYING and the notices of
# the protocols they compile in as lib/LICENSE.wl-clipboard. install-data stages both
# beside the native libs, where typer.py runs the bundled copy in place of the distro's.
# They link only libwayland-client and libc.
#
# Usage: scripts/vendor_wl_clipboard.sh [DEST_DIR]
#   DEST_DIR defaults to the repo lib/.
# Source of the tree (first match wins):
#   WL_CLIPBOARD_SRC  a pre-fetched wl-clipboard source tree; packaging recipes pass
#                     their pinned checkout so the build needs no network.
#   (unset)           dev convenience: git-clone the pinned tag and check its commit.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/lib}"
# shellcheck source=../packaging/pins.sh
. "$(cd "$(dirname "$0")/.." && pwd)/packaging/pins.sh"
WL_CLIPBOARD_VERSION="$QUOTH_WL_CLIPBOARD_TAG"
WL_CLIPBOARD_SRC="${WL_CLIPBOARD_SRC:-}"

for tool in meson ninja strings; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found (build dep)" >&2; exit 1; }
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -n "$WL_CLIPBOARD_SRC" ]; then
  src="$WL_CLIPBOARD_SRC"
else
  echo "==> cloning wl-clipboard $WL_CLIPBOARD_VERSION (dev convenience; recipes pass WL_CLIPBOARD_SRC)"
  git clone --depth 1 --branch "$WL_CLIPBOARD_VERSION" \
    https://github.com/bugaevc/wl-clipboard.git "$tmp/src"
  actual="$(git -C "$tmp/src" rev-parse HEAD)"
  [ "$actual" = "$QUOTH_WL_CLIPBOARD_COMMIT" ] || {
    echo "ERROR: wl-clipboard $WL_CLIPBOARD_VERSION resolved to $actual, expected $QUOTH_WL_CLIPBOARD_COMMIT" >&2
    exit 1
  }
  src="$tmp/src"
fi
[ -f "$src/meson.build" ] || { echo "ERROR: no meson.build in $src" >&2; exit 1; }

meson setup "$tmp/build" "$src" --buildtype=release
ninja -C "$tmp/build"

# Below wayland-protocols 1.39 meson drops ext-data-control without an error, and the
# build would lose the KDE clipboard path it is vendored for. No `grep -q`: its early
# exit kills `strings` with SIGPIPE, which pipefail reports as a failed match.
if ! strings "$tmp/build/src/wl-copy" | grep -x ext_data_control_manager_v1 >/dev/null; then
  echo "FAIL: wl-copy built without ext-data-control-v1 (needs wayland-protocols >= 1.39)" >&2
  exit 1
fi

install -d "$DEST/bin"
install -m755 "$tmp/build/src/wl-copy" "$tmp/build/src/wl-paste" "$DEST/bin/"
notices="$("$(cd "$(dirname "$0")" && pwd)/protocol_notices.sh" "$tmp/build")"
{
  cat "$src/COPYING"
  printf '\n\n# Wayland protocols\nwl-copy and wl-paste compile in code generated from these protocols:\n\n'
  printf '%s\n' "$notices"
} >"$DEST/LICENSE.wl-clipboard"

echo "Vendored $("$DEST/bin/wl-copy" --version | head -1) into $DEST/bin"
