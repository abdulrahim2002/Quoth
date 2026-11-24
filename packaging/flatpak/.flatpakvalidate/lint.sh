#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Lint the Flatpak manifest with Flathub's linter. Its build and repository checks
# report the screenshots only Flathub's pipeline mirrors; the install test checks the
# installed desktop entry, metainfo and icons instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
flatpak run --command=flatpak-builder-lint org.flatpak.Builder \
  --exceptions --user-exceptions "$ROOT/packaging/flatpak/.flatpakvalidate/exceptions.json" \
  manifest "$ROOT/packaging/flatpak/dev.quoth.Quoth.yaml"
