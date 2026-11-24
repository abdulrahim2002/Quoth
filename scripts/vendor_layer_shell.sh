#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Vendor gtk4-layer-shell (the .so and GI typelib) for channels whose distro lacks
# the system package: the .deb and the Flatpak. Without it the pill is suppressed.
#
# Built from source (Meson) with introspection on (for the typelib) and everything
# else off. The result lands in lib/ beside libwhisper, where
# _native.layer_shell_{lib_paths,typelib_dirs}() search first and __init__.py preloads
# the .so (RTLD_GLOBAL, before libwayland-client, as the symbol shim requires; see
# gtk4-layer-shell's linking.md). The built lib links only system GTK4, GLib and
# Wayland (declared .deb deps), so it needs no $ORIGIN.
#
# Not for the AUR: Arch ships gtk4-layer-shell as a system dependency.
#
# Usage: scripts/vendor_layer_shell.sh [DEST_DIR]
#   DEST_DIR defaults to the repo lib/.
# Source of the tree (first match wins):
#   LAYER_SHELL_SRC  a pre-fetched gtk4-layer-shell source tree; packaging recipes
#                    pass their pinned checkout so the build needs no network.
#   (unset)          dev convenience: git-clone the pinned tag and check its commit.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/lib}"
# The tag and commit come from the single source of truth (packaging/pins.sh).
# shellcheck source=../packaging/pins.sh
. "$(cd "$(dirname "$0")/.." && pwd)/packaging/pins.sh"
LAYER_SHELL_VERSION="$QUOTH_LAYER_SHELL_TAG"
LAYER_SHELL_SRC="${LAYER_SHELL_SRC:-}"
PYTHON="${PYTHON:-python3}"

for tool in meson ninja; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found (build dep)" >&2; exit 1; }
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -n "$LAYER_SHELL_SRC" ]; then
  src="$LAYER_SHELL_SRC"
else
  echo "==> cloning gtk4-layer-shell $LAYER_SHELL_VERSION (dev convenience; recipes pass LAYER_SHELL_SRC)"
  git clone --depth 1 --branch "$LAYER_SHELL_VERSION" \
    https://github.com/wmww/gtk4-layer-shell.git "$tmp/src"
  actual="$(git -C "$tmp/src" rev-parse HEAD)"
  [ "$actual" = "$QUOTH_LAYER_SHELL_COMMIT" ] || {
    echo "ERROR: gtk4-layer-shell $LAYER_SHELL_VERSION resolved to $actual, expected $QUOTH_LAYER_SHELL_COMMIT" >&2
    exit 1
  }
  src="$tmp/src"
fi
[ -f "$src/meson.build" ] || { echo "ERROR: no meson.build in $src" >&2; exit 1; }

prefix="$tmp/prefix"
meson setup "$tmp/build" "$src" --prefix="$prefix" --buildtype=release \
  -Dintrospection=true -Dvapi=false -Dexamples=false -Ddocs=false \
  -Dtests=false -Dsmoke-tests=false
ninja -C "$tmp/build"
ninja -C "$tmp/build" install

# Locate the built artifacts (libdir varies by distro: lib/ or lib/<multiarch>/).
so="$(find "$prefix" -name 'libgtk4-layer-shell.so*' -print -quit)"
typelib="$(find "$prefix" -name 'Gtk4LayerShell-1.0.typelib' -print -quit)"
[ -n "$so" ] || { echo "ERROR: libgtk4-layer-shell.so not built" >&2; exit 1; }
[ -n "$typelib" ] || { echo "ERROR: Gtk4LayerShell-1.0.typelib not built (introspection?)" >&2; exit 1; }

mkdir -p "$DEST"
rm -f "$DEST"/libgtk4-layer-shell.so* "$DEST"/Gtk4LayerShell-1.0.typelib
# -aP keeps the soname symlink chain (.so -> .so.0 -> .so.0.x.y) as links.
cp -aP "$(dirname "$so")"/libgtk4-layer-shell.so* "$DEST/"
cp -a "$typelib" "$DEST/"

# Ship the MIT license beside the lib so its attribution travels with it
# (install-data bundles lib/), with the notices of the Wayland protocols it compiles in.
lic="$(find "$src" -maxdepth 1 -iname 'LICENSE*' -print -quit)"
[ -n "$lic" ] || { echo "ERROR: gtk4-layer-shell LICENSE not found in $src" >&2; exit 1; }
notices="$("$(cd "$(dirname "$0")" && pwd)/protocol_notices.sh" "$tmp/build")"
{
  cat "$lic"
  printf '\n\n# Wayland protocols\nThe library compiles in code generated from these protocols:\n\n'
  printf '%s\n' "$notices"
} >"$DEST/LICENSE.gtk4-layer-shell"

"$PYTHON" -I "$(cd "$(dirname "$0")" && pwd)/check_install.py" \
  --elf-root "$DEST"

echo "Vendored gtk4-layer-shell $LAYER_SHELL_VERSION into $DEST:"
ls -1 "$DEST"/libgtk4-layer-shell.so* "$DEST"/Gtk4LayerShell-1.0.typelib | sed 's#.*/#  #'
