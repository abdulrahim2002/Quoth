#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Validate the Quoth AUR recipe end to end in a clean Arch container.
# Builds the stable PKGBUILD against the local working tree, staged in place of the
# release tarball, proving build(), package(), the install and the removal.
# quoth-git shares the same build and install body, so this covers it too.
set -euo pipefail

SRC="${SRC:-/work}"                 # repo, bind-mounted read-only
OUTPUT_DIR="${OUTPUT_DIR:-}"        # optional directory for the validated package
VERSION="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$SRC/pyproject.toml" | head -1)"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SRC" rev-parse HEAD)}"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid SOURCE_COMMIT" >&2; exit 2; }
QUOTH_RELEASE="${QUOTH_RELEASE:-quoth@${VERSION}+g${SOURCE_COMMIT}}"
QUOTH_ENVIRONMENT="${QUOTH_ENVIRONMENT:-candidate}"
export QUOTH_RELEASE QUOTH_ENVIRONMENT
BUILD="$HOME/build"
rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"

STAGE="quoth-$VERSION"
echo "==> staging local working tree as $STAGE.tar.gz (stands in for the release tarball)"
cp -a "$SRC" "$STAGE"
rm -rf "$STAGE"/.git "$STAGE"/.venv "$STAGE"/dist \
       "$STAGE"/lib/*.so "$STAGE"/lib/*.so.* "$STAGE"/lib/openvino "$STAGE"/lib/npu \
       "$STAGE"/lib/bin \
       "$STAGE"/.pytest_cache "$STAGE"/.ruff_cache
tar czf "$STAGE.tar.gz" "$STAGE"
rm -rf "$STAGE"

cp "$SRC/packaging/aur/PKGBUILD" .
cp "$SRC/packaging/aur/quoth.install" .

echo "==> pacman -Syu (a reused image keeps the databases it was built with)"
# Mirrors drop superseded packages, so a stale database fails makepkg's install.
sudo pacman -Syu --noconfirm

echo "==> makepkg -si (pulls deps, builds libwhisper.so, installs)"
# makepkg uses the staged Quoth tarball as is (its checksum is SKIP); the pinned
# sources download and are sha256-verified.
makepkg -si --noconfirm
package_list=$(makepkg --packagelist)
mapfile -t packages <<< "$package_list"

echo "==> namcap (the recipe and the built package)"
# namcap exits 0 whatever it reports and has no override file, so each accepted
# warning is matched here with its reason; any other error or warning fails.
site='usr/lib/python3\.[0-9]+/site-packages/quoth'
namcap_accepted=(
  # GStreamer loads these plugins by element name and the paste path runs these
  # tools as commands, so no file in the package links them.
  '^quoth W: dependency-not-needed (gst-plugins-base-libs|gst-plugins-good|gst-plugin-pipewire|wl-clipboard|wtype|ydotool)$'
  # OpenVINO loads the Intel NPU user-mode driver and compiler dynamically by soname.
  '^quoth W: dependency-not-needed (intel-npu-driver|intel-npu-compiler)$'
  # Only the Flatpak asks the shortcuts and Background portals; native installs use evdev
  # and an autostart file. namcap lists the importing files in set order.
  "^quoth W: python-module-no-package-associated gi\.repository\.Xdp \[('$site/(autostart|portal_shortcuts)\.py'(, )?)+\]$"
  # quoth preloads the bundled OpenVINO core, which satisfies libwhisper's NEEDED.
  "^quoth W: library-no-package-associated libopenvino\.so\.[0-9]+ \['$site/lib/libwhisper\.so\.[0-9.]+'\]$"
  # Intel's prebuilt OpenVINO runtime, shipped as the pinned wheel's bytes.
  "^quoth W: elffile-without-relro $site/lib/openvino/libhwloc\.so\.[0-9]+$"
  "^quoth W: unused-sodepend /usr/lib/lib(dl\.so\.2|pthread\.so\.0|m\.so\.6) $site/lib/openvino/[^ /]+$"
)
namcap_report=$(namcap -m PKGBUILD "${packages[@]}")
namcap_unaccepted=$(grep -v -E -f <(printf '%s\n' "${namcap_accepted[@]}") <<< "$namcap_report" || true)
if [ -n "$namcap_unaccepted" ]; then
  printf '%s\n' "$namcap_unaccepted"
  echo "ERROR: namcap is not clean" >&2
  exit 1
fi

echo "==> smoke test (headless: install + lib load; not GPU/NPU acceleration)"
command -v quoth                                                              # binary on PATH
test -f /usr/share/applications/dev.quoth.Quoth.desktop            # desktop entry
test -f /usr/share/metainfo/dev.quoth.Quoth.metainfo.xml           # metainfo
test -f /usr/share/icons/hicolor/256x256/apps/dev.quoth.Quoth.png  # icon
env GDK_BACKEND=x11 GTK_A11Y=none timeout 30s \
  xvfb-run -a -s '-screen 0 1024x768x24' \
  python -I "$SRC/scripts/check_install.py" \
  --package-channel arch \
  --expected-release "$QUOTH_RELEASE" \
  --expected-environment "$QUOTH_ENVIRONMENT"
quoth --help >/dev/null 2>&1 && echo "quoth --help: OK" || echo "quoth --help: skipped (headless)"

echo "==> pacman -Rns (no package file may stay)"
package_path_list=$(pacman -Qlq quoth)
mapfile -t package_paths <<< "$package_path_list"
sudo pacman -Rns --noconfirm quoth
leftovers=()
for path in "${package_paths[@]}"; do
  # Shared directories stay. Files may not, nor anything in Quoth's own directories.
  case "$path" in
    */quoth/* | */quoth-*) ;;
    *) [ -d "$path" ] && [ ! -L "$path" ] && continue ;;
  esac
  if [ -e "$path" ] || [ -L "$path" ]; then leftovers+=("$path"); fi
done
if [ "${#leftovers[@]}" -gt 0 ]; then
  printf 'left behind: %s\n' "${leftovers[@]}" >&2
  echo "ERROR: removing the package left its files behind" >&2
  exit 1
fi

if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  cp "${packages[@]}" "$OUTPUT_DIR/"
fi
echo "OK: AUR recipe validated (Vulkan + NPU build; OpenVINO bundled & loads)"
