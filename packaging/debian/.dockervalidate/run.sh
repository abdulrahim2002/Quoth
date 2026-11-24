#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Validate the Quoth .deb recipe end to end in a clean Ubuntu 24.04 (noble) box.
# Noble is the hardest target (the Pop!_OS base): no system gtk4-layer-shell, and
# numpy 1.26. Proves the bundled gtk4-layer-shell, OpenVINO and the Vulkan whisper
# build, then apt-installs the .deb, smoke-tests that it loads, and removes and
# purges it. Runs as root (dpkg-buildpackage permits it; the recipe is
# Rules-Requires-Root: no). Mirrors packaging/aur/.dockervalidate/.
set -euo pipefail

SRC="${SRC:-/work}"                 # repo, bind-mounted read-only
OUTPUT_DIR="${OUTPUT_DIR:-}"        # optional writable mount for the validated .deb

VERSION="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$SRC/pyproject.toml" | head -1)"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SRC" rev-parse HEAD)}"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid SOURCE_COMMIT" >&2; exit 2; }
QUOTH_RELEASE="${QUOTH_RELEASE:-quoth@${VERSION}+g${SOURCE_COMMIT}}"
QUOTH_ENVIRONMENT="${QUOTH_ENVIRONMENT:-candidate}"
export QUOTH_RELEASE QUOTH_ENVIRONMENT

# Pinned native sources come from their single source of truth, packaging/pins.sh.
. "$SRC/packaging/pins.sh"
WHISPER_TAG="$QUOTH_WHISPER_TAG"
WHISPER_SHA="$QUOTH_WHISPER_SHA256"
OV_WHEEL="$QUOTH_OPENVINO_WHEEL"
OV_URL="$QUOTH_OPENVINO_URL"
OV_SHA="$QUOTH_OPENVINO_SHA256"
LS_TAG="$QUOTH_LAYER_SHELL_TAG"
LS_COMMIT="$QUOTH_LAYER_SHELL_COMMIT"
TVG_TAG="$QUOTH_THORVG_TAG"
TVG_COMMIT="$QUOTH_THORVG_COMMIT"
WLC_TAG="$QUOTH_WL_CLIPBOARD_TAG"
WLC_COMMIT="$QUOTH_WL_CLIPBOARD_COMMIT"
YDT_TAG="$QUOTH_YDOTOOL_TAG"
YDT_COMMIT="$QUOTH_YDOTOOL_COMMIT"
SENTRY_WHEEL="$QUOTH_SENTRY_SDK_WHEEL"
SENTRY_URL="$QUOTH_SENTRY_SDK_URL"
SENTRY_SHA="$QUOTH_SENTRY_SDK_SHA256"

BUILD="$HOME/build"; rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"

echo "==> staging working tree (strips built artifacts)"
cp -a "$SRC" quoth
cd quoth
rm -rf .git .venv dist lib/*.so lib/*.so.* lib/*.typelib lib/LICENSE.* lib/openvino lib/npu \
       lib/bin \
       .pytest_cache .ruff_cache packaging/debian/.build-src debian

echo "==> placing debian/ recipe at the source root"
cp -a packaging/debian debian

echo "==> installing build deps (apt build-dep reads debian/control)"
apt-get update
apt-get build-dep -y ./

echo "==> pre-fetching pinned native sources into .build-src/ (keeps debian/rules offline)"
mkdir -p packaging/debian/.build-src
( cd packaging/debian/.build-src
  curl -fsSL -o whisper.tar.gz "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/${WHISPER_TAG}.tar.gz"
  echo "${WHISPER_SHA}  whisper.tar.gz" | sha256sum -c -
  tar xzf whisper.tar.gz && mv "whisper.cpp-${WHISPER_TAG#v}" whisper.cpp && rm whisper.tar.gz
  "$SRC/scripts/apply_whisper_patches.sh" --apply whisper.cpp
  curl -fsSL -o "$OV_WHEEL" "$OV_URL"
  echo "${OV_SHA}  ${OV_WHEEL}" | sha256sum -c -
  python3 -m zipfile -e "$OV_WHEEL" openvino-wheel && rm "$OV_WHEEL"
  "$SRC/scripts/vendor_npu_driver.sh" --fetch npu-driver
  curl -fsSL -o "$SENTRY_WHEEL" "$SENTRY_URL"
  echo "${SENTRY_SHA}  ${SENTRY_WHEEL}" | sha256sum -c -
  python3 -m zipfile -e "$SENTRY_WHEEL" sentry-sdk-wheel && rm "$SENTRY_WHEEL"
  git clone --depth 1 --branch "$LS_TAG" https://github.com/wmww/gtk4-layer-shell.git gtk4-layer-shell
  git clone --depth 1 --branch "$TVG_TAG" https://github.com/thorvg/thorvg.git thorvg
  git clone --depth 1 --branch "$WLC_TAG" https://github.com/bugaevc/wl-clipboard.git wl-clipboard
  git clone --depth 1 --branch "$YDT_TAG" https://github.com/ReimuNotMoe/ydotool.git ydotool
  LS_ACTUAL="$(git -C gtk4-layer-shell rev-parse HEAD)"
  [ "$LS_ACTUAL" = "$LS_COMMIT" ] || {
    echo "ERROR: gtk4-layer-shell $LS_TAG resolved to $LS_ACTUAL, expected $LS_COMMIT" >&2
    exit 1
  }
  TVG_ACTUAL="$(git -C thorvg rev-parse HEAD)"
  [ "$TVG_ACTUAL" = "$TVG_COMMIT" ] || {
    echo "ERROR: ThorVG $TVG_TAG resolved to $TVG_ACTUAL, expected $TVG_COMMIT" >&2
    exit 1
  }
  WLC_ACTUAL="$(git -C wl-clipboard rev-parse HEAD)"
  [ "$WLC_ACTUAL" = "$WLC_COMMIT" ] || {
    echo "ERROR: wl-clipboard $WLC_TAG resolved to $WLC_ACTUAL, expected $WLC_COMMIT" >&2
    exit 1
  }
  YDT_ACTUAL="$(git -C ydotool rev-parse HEAD)"
  [ "$YDT_ACTUAL" = "$YDT_COMMIT" ] || {
    echo "ERROR: ydotool $YDT_TAG resolved to $YDT_ACTUAL, expected $YDT_COMMIT" >&2
    exit 1
  }
  # The copyleft components' source, from these checkouts, goes out beside the .deb.
  "$SRC/scripts/deb_sources.sh" . "$BUILD/quoth-deb-sources.tar.gz"
)

echo "==> dpkg-buildpackage -b (binary only)"
dpkg-buildpackage -b -us -uc
DEB="$(ls "$BUILD"/quoth_*_amd64.deb | head -1)"

echo "==> lintian (errors and warnings fail; quoth.lintian-overrides names each exception)"
lintian --fail-on error,warning --allow-root "$DEB"

echo "==> apt install the built .deb (pulls runtime deps)"
if dpkg-deb -f "$DEB" Depends | grep -F 'python3-sentry-sdk' >/dev/null; then
  echo "ERROR: bundled Sentry SDK leaked into Debian Depends" >&2
  exit 1
fi
echo "-- .deb layout (quoth package + bundled libs):"
# No `head` here: closing the pipe early SIGPIPEs dpkg-deb under `set -o pipefail`.
dpkg-deb -c "$DEB" | grep -E 'dist-packages/quoth/(whisper\.py|__init__\.py|lib/)' || true
apt-get install -y --no-install-recommends "$DEB"
echo "-- installed quoth location + lib dir:"
python3 -c "import quoth, os; d=os.path.dirname(quoth.__file__); print('quoth.__file__ dir:', d); print('contents:', sorted(os.listdir(d))); ld=os.path.join(d,'lib'); print('lib/ exists:', os.path.isdir(ld)); print('lib/ contents:', sorted(os.listdir(ld)) if os.path.isdir(ld) else 'N/A')"

echo "==> smoke test"
command -v quoth                                                              # binary on PATH
test -f /usr/share/applications/dev.quoth.Quoth.desktop            # desktop entry
test -f /usr/share/metainfo/dev.quoth.Quoth.metainfo.xml           # metainfo
test -f /usr/share/icons/hicolor/256x256/apps/dev.quoth.Quoth.png  # icon
# The shared smoke test covers native libs, OpenVINO, the layer-shell load and
# relocatable ELF paths. Noble has no system gtk4-layer-shell, so a successful load
# proves the bundled copy works; the dpkg-deb layout above shows it in quoth/lib/.
env GDK_BACKEND=x11 GTK_A11Y=none timeout 30s \
  xvfb-run -a -s '-screen 0 1024x768x24' \
  python3 -I "$SRC/scripts/check_install.py" \
  --package-channel debian \
  --expected-release "$QUOTH_RELEASE" \
  --expected-environment "$QUOTH_ENVIRONMENT"
quoth --help >/dev/null 2>&1 && echo "quoth --help: OK"

echo "==> apt remove, then purge (maintainer scripts must succeed; no package file may stay)"
package_path_list="$(dpkg-query -L quoth)"
mapfile -t package_paths <<< "$package_path_list"
# Both fail when prerm or postrm does. With no conffiles, remove already erases dpkg's
# record of the package, so apt-get could no longer find it to purge.
apt-get remove -y quoth
dpkg --purge quoth
status="$(dpkg-query -W -f='${db:Status-Status}' quoth 2>/dev/null || true)"
case "$status" in
  "" | not-installed) ;;
  *) echo "ERROR: quoth is $status after purge" >&2; exit 1 ;;
esac
leftovers=()
for path in "${package_paths[@]}"; do
  # Shared directories stay. Files may not, nor anything in Quoth's own directories.
  case "$path" in
    */quoth | */quoth/* | */quoth-*) ;;
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
  cp "$DEB" "$BUILD/quoth-deb-sources.tar.gz" "$OUTPUT_DIR/"
fi
echo "OK: .deb recipe validated on Ubuntu 24.04 (native payload)"
