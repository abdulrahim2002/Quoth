#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Validate the Quoth Flatpak locally.
# Builds via org.flatpak.Builder, installs, smoke-tests with the shared
# scripts/check_install.py inside the sandbox, and exports a self-hostable .flatpak
# with its corresponding-source bundle. Mirrors the AUR and .deb .dockervalidate
# harnesses.
#
# The committed manifest pins the quoth source to the release tarball; this harness
# swaps in a `git archive` of HEAD, the same files that tarball holds, so any commit
# builds without a published release. Commit a change before validating it.
# Needs flatpak, org.flatpak.Builder and the GNOME 50 runtime and SDK (pulled from
# Flathub automatically).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
APPID="dev.quoth.Quoth"
MANIFEST_SRC="$ROOT/packaging/flatpak/$APPID.yaml"
if [ -z "${QUOTH_RELEASE:-}" ]; then
  VERSION="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$ROOT/pyproject.toml" | head -1)"
  SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
  QUOTH_RELEASE="quoth@${VERSION}+g${SOURCE_COMMIT}"
fi
QUOTH_ENVIRONMENT="${QUOTH_ENVIRONMENT:-candidate}"
# Under $HOME, not /tmp: org.flatpak.Builder's `--filesystem=host` excludes /tmp
# (the sandbox has its own), so the manifest/sources must live in a host-visible path.
WORK="${WORK:-$HOME/.cache/quoth-flatpak-build}"
rm -rf "$WORK"; mkdir -p "$WORK"

echo "==> staging HEAD as the release tarball would hold it"
# --bundle-sources publishes this tree, so it must hold only what the release does.
SRC="$WORK/src"
mkdir "$SRC"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$SRC"

echo "==> generating local manifest (quoth source -> the staged tree)"
MANIFEST="$WORK/$APPID.yaml"
python3 - "$MANIFEST_SRC" "$MANIFEST" "$SRC" "$QUOTH_RELEASE" "$QUOTH_ENVIRONMENT" <<'PY'
import sys, yaml
src, dst, localdir, release, environment = sys.argv[1:6]
m = yaml.safe_load(open(src))
for mod in m["modules"]:
    if isinstance(mod, dict) and mod.get("name") == "quoth":
        mod["sources"][0] = {"type": "dir", "path": localdir}  # swap the release archive for the staged tree
        mod.setdefault("build-options", {}).setdefault("env", {}).update(
            {"QUOTH_RELEASE": release, "QUOTH_ENVIRONMENT": environment}
        )
yaml.safe_dump(m, open(dst, "w"), sort_keys=False)
print("  wrote", dst)
PY
cp "$ROOT/packaging/flatpak/"python3-*.yaml "$WORK/"   # runtime + build-only dep includes
cp "$ROOT/packaging/flatpak/openvino-npu.yaml" "$WORK/"
cp "$ROOT/packaging/flatpak/pygobject-transfer-full.patch" "$WORK/"

cd "$WORK"
echo "==> flatpak-builder build + install (org.flatpak.Builder)"
# --bundle-sources exports every module's sources as the APPID.Sources runtime: the
# GPL corresponding source for the bundled copyleft components and Quoth itself.
flatpak run --die-with-parent org.flatpak.Builder --force-clean --assumeyes --user --install-deps-from=flathub \
  --bundle-sources --repo=repo --install builddir "$MANIFEST"

echo "==> smoke test in the sandbox"
flatpak run --command=quoth "$APPID" --help >/dev/null
echo "  quoth --help: OK"
# The shared smoke test, fed in via -c (it is a validation script, not shipped in /app).
# Quoth grants no X11 socket, so this check grants one for xvfb's display.
timeout 30s xvfb-run -a -s '-screen 0 1024x768x24' \
  flatpak run --socket=x11 --env=GDK_BACKEND=x11 --env=GTK_A11Y=none --command=python3 \
  "$APPID" -I -c "$(cat "$ROOT/scripts/check_install.py")" \
    --package-channel flatpak \
    --expected-release "$QUOTH_RELEASE" \
    --expected-environment "$QUOTH_ENVIRONMENT"

echo "==> export self-hostable .flatpak bundle and its sources"
flatpak build-bundle repo "$WORK/quoth.flatpak" "$APPID"
flatpak build-bundle --runtime repo "$WORK/quoth-sources.flatpak" "$APPID.Sources"
ls -lh "$WORK/quoth.flatpak" "$WORK/quoth-sources.flatpak" | awk '{print "  bundle:", $5, $NF}'
echo "OK: Flatpak validated (build + load + bundle); bundles in $WORK"
