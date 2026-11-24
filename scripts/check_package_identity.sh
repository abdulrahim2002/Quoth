#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Check a downloaded package artifact DIR: its files match its SHA256SUMS, and its
# sidecars name this checkout's commit and the release identity in RELEASE_METADATA.
#
#   RELEASE_METADATA='{...}' scripts/check_package_identity.sh DIR
#
# Run it from a checkout at the package's commit, with RELEASE_METADATA set to the
# identity the run resolved (its identity job's `metadata` output). Copying it from the
# artifact's own release-metadata.json would make that check pass trivially.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$1"
sha256sum -c SHA256SUMS
built="$(cat source-commit.txt)"
head="$(git -C "$root" rev-parse HEAD)"
[ "$built" = "$head" ] || {
  echo "ERROR: $1 was built from $built, but this checkout is at $head" >&2
  exit 1
}
metadata="$(cat release-metadata.json)"
[ "$metadata" = "$RELEASE_METADATA" ] || {
  printf 'ERROR: %s/release-metadata.json is\n  %s\nbut RELEASE_METADATA is\n  %s\n' \
    "$1" "$metadata" "$RELEASE_METADATA" >&2
  exit 1
}
