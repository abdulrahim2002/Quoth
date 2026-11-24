#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Propagate packaging/pins.sh into the files that can't source it at build time: the
# AUR PKGBUILDs and .SRCINFO, the Flatpak manifest and its NPU and Python-deps modules,
# and check_install.py's Sentry SDK pin. Repo-built consumers (the .deb harness, the
# vendor scripts, and the Flatpak build) source pins.sh directly.
#
#   scripts/sync_pins.sh           write the pinned values into those files
#   scripts/sync_pins.sh --check   verify those files and the app version sites;
#                                  exit 1 on drift (CI guard)
#
# Bump an engine in packaging/pins.sh, then run this. The .SRCINFO source/sha lines
# are rewritten textually here; the canonical `makepkg --printsrcinfo` regen still
# happens at AUR-publish time (it produces an identical result).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../packaging/pins.sh
. "$ROOT/packaging/pins.sh"

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

PKGBUILDS=("$ROOT/packaging/aur/PKGBUILD" "$ROOT/packaging/aur/quoth-git/PKGBUILD")
SRCINFOS=("$ROOT/packaging/aur/.SRCINFO" "$ROOT/packaging/aur/quoth-git/.SRCINFO")
MANIFEST="$ROOT/packaging/flatpak/dev.quoth.Quoth.yaml"
NPU_MODULE="$ROOT/packaging/flatpak/openvino-npu.yaml"
PYTHON_DEPS="$ROOT/packaging/flatpak/python3-deps.yaml"
CHECK_INSTALL="$ROOT/scripts/check_install.py"
APP_INIT="$ROOT/src/quoth/__init__.py"
METAINFO="$ROOT/data/dev.quoth.Quoth.metainfo.xml"

# Rewrite the named pin vars, replacing only the quoted value so trailing comments
# survive.
_pkgbuild() {
  sed -E \
    -e "s|^(_sentryver=)\"[^\"]*\"|\1\"$QUOTH_SENTRY_SDK_VERSION\"|" \
    -e "s|^(_whispertag=)\"[^\"]*\"|\1\"$QUOTH_WHISPER_TAG\"|" \
    -e "s|^(_whispersha=)\"[^\"]*\"|\1\"$QUOTH_WHISPER_SHA256\"|" \
    -e "s|^(_ovwheel=)\"[^\"]*\"|\1\"$QUOTH_OPENVINO_WHEEL\"|" \
    -e "s|^(_ovurl=)\"[^\"]*\"|\1\"$QUOTH_OPENVINO_URL\"|" \
    -e "s|^(_ovsha=)\"[^\"]*\"|\1\"$QUOTH_OPENVINO_SHA256\"|" \
    -e "s|^(_thorvgtag=)\"[^\"]*\"|\1\"$QUOTH_THORVG_TAG\"|" \
    -e "s|^(_thorvgsha=)\"[^\"]*\"|\1\"$QUOTH_THORVG_SHA256\"|" \
    "$1"
}

# Rewrite the Sentry depends line and the whisper, OpenVINO and ThorVG source and sha
# lines. Source order is fixed (quoth, whisper, openvino, thorvg), so sha256sums #2 is
# whisper, #3 openvino and #4 thorvg (#1 is SKIP).
_srcinfo() {
  awk -v sentry="$QUOTH_SENTRY_SDK_VERSION" \
      -v wtag="$QUOTH_WHISPER_TAG" -v wsha="$QUOTH_WHISPER_SHA256" \
      -v ovurl="$QUOTH_OPENVINO_URL" -v ovsha="$QUOTH_OPENVINO_SHA256" \
      -v tvtag="$QUOTH_THORVG_TAG" -v tvsha="$QUOTH_THORVG_SHA256" '
    /^[[:space:]]*depends = python-sentry_sdk/ {
      print "\tdepends = python-sentry_sdk>=" sentry; next }
    /^[[:space:]]*source = whisper\.cpp-/ {
      print "\tsource = whisper.cpp-" wtag ".tar.gz::https://github.com/ggml-org/whisper.cpp/archive/refs/tags/" wtag ".tar.gz"; next }
    /^[[:space:]]*source = https:\/\/files\.pythonhosted\.org\/.*openvino-.*\.whl/ {
      print "\tsource = " ovurl; next }
    /^[[:space:]]*source = thorvg-/ {
      print "\tsource = thorvg-" tvtag ".tar.gz::https://github.com/thorvg/thorvg/archive/refs/tags/" tvtag ".tar.gz"; next }
    /^[[:space:]]*sha256sums = / { n++
      if (n == 2) { print "\tsha256sums = " wsha; next }
      if (n == 3) { print "\tsha256sums = " ovsha; next }
      if (n == 4) { print "\tsha256sums = " tvsha; next } }
    { print }
  ' "$1"
}

_python_deps() {
  awk -v url="$QUOTH_SENTRY_SDK_URL" -v sha="$QUOTH_SENTRY_SDK_SHA256" '
    /^[[:space:]]*- name: python3-sentry-sdk$/ { sentry=1; print; next }
    sentry && /^[[:space:]]*url:/ { sub(/url:.*/, "url: " url); print; next }
    sentry && /^[[:space:]]*sha256:/ { sub(/sha256:.*/, "sha256: " sha); sentry=0; print; next }
    { print }
  ' "$1"
}

_check_install() {
  sed -E \
    "s|^(_SENTRY_VERSION = )\"[^\"]*\"|\1\"$QUOTH_SENTRY_SDK_VERSION\"|" \
    "$1"
}

# Rewrite the pinned source lines in the Flatpak manifest. A `url:` match arms the
# rewrite of the lines that follow it within the same source: the whisper.cpp archive
# arms the next `sha256:`; the git sources (gtk4-layer-shell, SPIRV-Headers, ThorVG,
# wl-clipboard, ydotool) arm the next `tag:` then `commit:`. The quoth source's
# placeholder sha comes before any armed url, so it is left untouched, and Flatpak-only
# sources pins.sh does not carry (wtype and the like) pass through verbatim. Flatpak's
# OpenVINO and NPU sources live in openvino-npu.yaml and are rewritten separately below.
_manifest() {
  awk -v wtag="$QUOTH_WHISPER_TAG" -v wsha="$QUOTH_WHISPER_SHA256" \
      -v lstag="$QUOTH_LAYER_SHELL_TAG" -v lscommit="$QUOTH_LAYER_SHELL_COMMIT" \
      -v shtag="$QUOTH_SPIRV_HEADERS_TAG" -v shcommit="$QUOTH_SPIRV_HEADERS_COMMIT" \
      -v tvtag="$QUOTH_THORVG_TAG" -v tvcommit="$QUOTH_THORVG_COMMIT" \
      -v wctag="$QUOTH_WL_CLIPBOARD_TAG" -v wccommit="$QUOTH_WL_CLIPBOARD_COMMIT" \
      -v ydtag="$QUOTH_YDOTOOL_TAG" -v ydcommit="$QUOTH_YDOTOOL_COMMIT" '
    /^[[:space:]]*sha256:/ && want=="whisper" { sub(/sha256:.*/, "sha256: " wsha); want=""; print; next }
    /^[[:space:]]*tag:/    && want=="ls" { sub(/tag:.*/,    "tag: " lstag);       print; next }
    /^[[:space:]]*commit:/ && want=="ls" { sub(/commit:.*/, "commit: " lscommit); want=""; print; next }
    /^[[:space:]]*tag:/    && want=="sh" { sub(/tag:.*/,    "tag: " shtag);       print; next }
    /^[[:space:]]*commit:/ && want=="sh" { sub(/commit:.*/, "commit: " shcommit); want=""; print; next }
    /^[[:space:]]*tag:/    && want=="tv" { sub(/tag:.*/,    "tag: " tvtag);       print; next }
    /^[[:space:]]*commit:/ && want=="tv" { sub(/commit:.*/, "commit: " tvcommit); want=""; print; next }
    /^[[:space:]]*tag:/    && want=="wc" { sub(/tag:.*/,    "tag: " wctag);       print; next }
    /^[[:space:]]*commit:/ && want=="wc" { sub(/commit:.*/, "commit: " wccommit); want=""; print; next }
    /^[[:space:]]*tag:/    && want=="yd" { sub(/tag:.*/,    "tag: " ydtag);       print; next }
    /^[[:space:]]*commit:/ && want=="yd" { sub(/commit:.*/, "commit: " ydcommit); want=""; print; next }
    /url:.*whisper\.cpp\/archive\/refs\/tags\/.*\.tar\.gz/ {
      sub(/url:.*/, "url: https://github.com/ggml-org/whisper.cpp/archive/refs/tags/" wtag ".tar.gz"); want="whisper"; print; next }
    /url:.*gtk4-layer-shell\.git/ { want="ls"; print; next }
    /url:.*SPIRV-Headers\.git/    { want="sh"; print; next }
    /url:.*thorvg\.git/          { want="tv"; print; next }
    /url:.*wl-clipboard\.git/    { want="wc"; print; next }
    /url:.*ydotool\.git/         { want="yd"; print; next }
    { print }
  ' "$1"
}

_npu_module() {
  awk -v ovurl="$QUOTH_OPENVINO_URL" -v ovsha="$QUOTH_OPENVINO_SHA256" \
      -v npuurl="$QUOTH_NPU_DRIVER_URL" -v npusha="$QUOTH_NPU_DRIVER_SHA256" \
      -v npudeb="$QUOTH_NPU_UMD_DEB" -v npudebsha="$QUOTH_NPU_UMD_SHA256" \
      -v npucmp="$QUOTH_NPU_COMPILER_DEB" -v npucmpsha="$QUOTH_NPU_COMPILER_SHA256" \
      -v npulicenseurl="$QUOTH_NPU_LICENSE_URL" -v npulicensesha="$QUOTH_NPU_LICENSE_SHA256" \
      -v npunoticesurl="$QUOTH_NPU_NOTICES_URL" -v npunoticessha="$QUOTH_NPU_NOTICES_SHA256" \
      -v npucmplicenseurl="$QUOTH_NPU_COMPILER_LICENSE_URL" -v npucmplicensesha="$QUOTH_NPU_COMPILER_LICENSE_SHA256" \
      -v zeurl="$QUOTH_LEVEL_ZERO_URL" -v zesha="$QUOTH_LEVEL_ZERO_SHA256" \
      -v zelicenseurl="$QUOTH_LEVEL_ZERO_LICENSE_URL" -v zelicensesha="$QUOTH_LEVEL_ZERO_LICENSE_SHA256" '
    /^[[:space:]]*- type:/ { source++; print; next }
    /sha256sum -c -/ && /npu-driver\/intel-level-zero-npu_/ {
      sub(/[0-9a-f]{64}  npu-driver\/[^"]+/, npudebsha "  npu-driver/" npudeb); inner_sha++; print; next }
    /ar p npu-driver\/intel-level-zero-npu_/ {
      sub(/npu-driver\/[^ ]+/, "npu-driver/" npudeb); inner_deb++; print; next }
    /sha256sum -c -/ && /npu-driver\/intel-driver-compiler-npu_/ {
      sub(/[0-9a-f]{64}  npu-driver\/[^"]+/, npucmpsha "  npu-driver/" npucmp); cmp_sha++; print; next }
    /ar p npu-driver\/intel-driver-compiler-npu_/ {
      sub(/npu-driver\/[^ ]+/, "npu-driver/" npucmp); cmp_deb++; print; next }
    /^[[:space:]]*url:/ && source >= 1 && source <= 7 {
      url = source == 1 ? ovurl : source == 2 ? npuurl : source == 3 ? zeurl : source == 4 ? npulicenseurl : source == 5 ? npunoticesurl : source == 6 ? zelicenseurl : npucmplicenseurl
      sub(/url:.*/, "url: " url); urls++; print; next }
    /^[[:space:]]*sha256:/ && source >= 1 && source <= 7 {
      sha = source == 1 ? ovsha : source == 2 ? npusha : source == 3 ? zesha : source == 4 ? npulicensesha : source == 5 ? npunoticessha : source == 6 ? zelicensesha : npucmplicensesha
      sub(/sha256:.*/, "sha256: " sha); hashes++; print; next }
    { print }
    END { if (source != 7 || urls != 7 || hashes != 7 || inner_sha != 1 || inner_deb != 1 || cmp_sha != 1 || cmp_deb != 1) exit 2 }
  ' "$1"
}

drift=0
_process() {  # label, file, transform-fn
  [ -f "$2" ] || { [ "$CHECK" = 1 ] && echo "WARN: ${2#"$ROOT"/} missing (skipped)"; return 0; }
  local new; new="$("$3" "$2")"
  if [ "$CHECK" = 1 ]; then
    diff -q <(printf '%s\n' "$new") "$2" >/dev/null 2>&1 || { echo "DRIFT: ${2#"$ROOT"/}"; drift=1; }
  else
    printf '%s\n' "$new" >"$2"; echo "synced: ${2#"$ROOT"/}"
  fi
}

# App-version parity (check-only). pyproject.toml is the source of truth; the recipes,
# quoth.__version__ and the AppStream release hardcode the version and must agree.
# Versions are never rewritten (debian/changelog and .SRCINFO are structured,
# release-time formats); drift is only flagged so a release can't ship mismatched
# versions. The quoth-git recipes carry a VCS pkgver and debian a revision suffix, so
# the version is matched as a prefix there; the stable pins must match exactly.
_version_check() {
  local ver
  ver="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$ROOT/pyproject.toml" | head -1)"
  [ -n "$ver" ] || { echo "VERSION: cannot read pyproject.toml version"; drift=1; return; }
  _vchk() {  # label, value, mode(exact|starts)
    if [ "$3" = starts ]; then case "$2" in "$ver"*) return 0 ;; esac
    else [ "$2" = "$ver" ] && return 0; fi
    echo "VERSION DRIFT: $1 = '$2' (pyproject.toml = $ver)"; drift=1
  }
  _vchk "aur/PKGBUILD pkgver"           "$(sed -nE 's/^pkgver=(.+)/\1/p'               "$ROOT/packaging/aur/PKGBUILD")"           exact
  _vchk "aur/.SRCINFO pkgver"           "$(sed -nE 's/^[[:space:]]*pkgver = (.+)/\1/p'  "$ROOT/packaging/aur/.SRCINFO")"           exact
  _vchk "debian/changelog"              "$(sed -nE '1s/^quoth \(([^)]+)\).*/\1/p'       "$ROOT/packaging/debian/changelog")"       starts
  _vchk "flatpak manifest quoth url"    "$(sed -nE 's@.*/Quoth/releases/download/v([^/]+)/quoth-\1\.tar\.gz.*@\1@p' "$MANIFEST" | head -1)" exact
  _vchk "flatpak manifest release"      "$(sed -nE 's/^[[:space:]]*QUOTH_RELEASE:[[:space:]]*quoth@(.+)/\1/p' "$MANIFEST" | head -1)" exact
  _vchk "quoth.__version__"              "$(sed -nE 's/^__version__ = "([^"]+)"/\1/p' "$APP_INIT" | head -1)" exact
  _vchk "latest AppStream release"       "$(sed -nE 's/.*<release version="([^"]+)".*/\1/p' "$METAINFO" | head -1)" exact
  _vchk "aur/quoth-git/PKGBUILD pkgver" "$(sed -nE 's/^pkgver=(.+)/\1/p'               "$ROOT/packaging/aur/quoth-git/PKGBUILD")" starts
  _vchk "aur/quoth-git/.SRCINFO pkgver" "$(sed -nE 's/^[[:space:]]*pkgver = (.+)/\1/p'  "$ROOT/packaging/aur/quoth-git/.SRCINFO")" starts
  grep -Fx "    \"sentry-sdk>=$QUOTH_SENTRY_SDK_VERSION\"," "$ROOT/pyproject.toml" >/dev/null || {
    echo "SENTRY SDK DRIFT: pyproject.toml must require sentry-sdk>=$QUOTH_SENTRY_SDK_VERSION"
    drift=1
  }
}

for f in "${PKGBUILDS[@]}"; do _process pkgbuild "$f" _pkgbuild; done
for f in "${SRCINFOS[@]}"; do _process srcinfo "$f" _srcinfo; done
_process manifest "$MANIFEST" _manifest
_process flatpak-npu "$NPU_MODULE" _npu_module
_process python-deps "$PYTHON_DEPS" _python_deps
_process check-install "$CHECK_INSTALL" _check_install

if [ "$CHECK" = 1 ]; then
  _version_check
  [ "$drift" = 0 ] && echo "pins OK: all recipes match packaging/pins.sh (versions + packaging pins)" \
    || { echo "FAIL: drift; run scripts/sync_pins.sh (and bump version sites by hand if flagged)" >&2; exit 1; }
fi
