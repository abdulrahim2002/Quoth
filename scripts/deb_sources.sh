#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Pack the source of the .deb's copyleft components into one release asset beside it:
# wl-clipboard (GPL-3.0-or-later), ydotool (AGPL-3.0-or-later) and ThorVG, which
# compiles in an MPL-2.0 file. Each is built unmodified from its packaging/pins.sh commit.
# wl-clipboard's build also generates code from the build distribution's
# wayland-protocols, so the files it reads go in too, with that package's version.
#
# Usage: scripts/deb_sources.sh SRC_DIR OUT_FILE
#   SRC_DIR holds the recipe's pinned checkouts: wl-clipboard, ydotool and thorvg.
# Run it where the package builds, after its build dependencies are installed.
set -euo pipefail

src="${1:?usage: $0 SRC_DIR OUT_FILE}"
out="${2:?usage: $0 SRC_DIR OUT_FILE}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../packaging/pins.sh
. "$REPO/packaging/pins.sh"
version="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$REPO/pyproject.toml" | head -1)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
root="$tmp/quoth-deb-sources"
mkdir "$root"
newest=0
while read -r name tag commit url terms; do
  actual="$(git -C "$src/$name" rev-parse HEAD)"
  [ "$actual" = "$commit" ] || { echo "ERROR: $name is at $actual, expected $commit" >&2; exit 1; }
  # export-ignore or export-subst would make the archive differ from the tree that is built.
  if git -C "$src/$name" grep -qE 'export-(ignore|subst)' "$commit" -- ':(glob)**/.gitattributes'; then
    echo "ERROR: $name's .gitattributes changes what git archive exports" >&2
    exit 1
  fi
  git -C "$src/$name" archive --format=tar --prefix="$name-$tag/" "$commit" | tar -x -C "$root"
  time="$(git -C "$src/$name" show -s --format=%ct "$commit")"
  [ "$time" -gt "$newest" ] && newest="$time"
  printf '  %s-%s (%s)\n    %s at %s\n' "$name" "$tag" "$terms" "$url" "$commit" >>"$tmp/list"
done <<EOF
wl-clipboard $QUOTH_WL_CLIPBOARD_TAG $QUOTH_WL_CLIPBOARD_COMMIT https://github.com/bugaevc/wl-clipboard GPL-3.0-or-later
ydotool $QUOTH_YDOTOOL_TAG $QUOTH_YDOTOOL_COMMIT https://github.com/ReimuNotMoe/ydotool AGPL-3.0-or-later
thorvg $QUOTH_THORVG_TAG $QUOTH_THORVG_COMMIT https://github.com/thorvg/thorvg MIT, with code under MPL-2.0 and the FreeType License
EOF

# The protocol files wl-clipboard's meson.build feeds wayland-scanner from wayland-protocols
# (a build dependency); its other protocols are in its own tree.
protocols="$(realpath "$(pkg-config --variable=pkgdatadir wayland-protocols)")"
protocols_version="$(dpkg-query -W -f='${Version}' wayland-protocols)"
protocols_dir="wayland-protocols-$protocols_version"
for xml in stable/xdg-shell/xdg-shell.xml \
  unstable/primary-selection/primary-selection-unstable-v1.xml \
  staging/ext-data-control/ext-data-control-v1.xml \
  staging/xdg-activation/xdg-activation-v1.xml; do
  install -Dm644 "$protocols/$xml" "$root/$protocols_dir/$xml"
done

{
  printf 'Source of the components bundled in quoth_%s_amd64.deb\n\n' "$version"
  printf 'These directories hold the unmodified upstream source the package builds:\n\n'
  cat "$tmp/list"
  printf '\nwl-clipboard'"'"'s build also generates code from files of the build\n'
  printf 'distribution'"'"'s wayland-protocols package (MIT and HPND-sell-variant),\n'
  printf 'version %s.\n' "$protocols_version"
  printf 'They are in %s/,\nat their paths under %s.\n' "$protocols_dir" "$protocols"
  printf '\nThe scripts that build them (scripts/vendor_wl_clipboard.sh,\n'
  printf 'scripts/vendor_ydotool.sh and scripts/vendor_thorvg.sh, run by\n'
  printf 'packaging/debian/rules) are in quoth-%s.tar.gz, Quoth'"'"'s own source, on\n' "$version"
  printf 'the same release.\n'
} >"$root/README"

tar --sort=name --mtime="@$newest" --owner=0 --group=0 --numeric-owner --mode=u+rw,go=rX \
  -C "$tmp" -cf - quoth-deb-sources | gzip -9n >"$out"
echo "Packed the .deb's component sources into $out"
