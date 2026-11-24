#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Check a prepared site's signatures as a client would: the APT InRelease against the
# keyring in GNUPGHOME, and the Flatpak remote through a fresh user installation in
# CLIENT_DIR that imports the site's quoth.gpg.
#
#   GNUPGHOME=... scripts/check_repository_signatures.sh SITE CLIENT_DIR
#
# GNUPGHOME needs only the signing key's public key. CLIENT_DIR must be a new
# directory: a remote left there by an earlier run makes remote-add fail.
set -euo pipefail

# The remote's file:// URL needs an absolute path.
site="$(realpath "$1")"
gpg --batch --verify "$site/apt/dists/stable/InRelease"
FLATPAK_USER_DIR="$(realpath -m "$2")"
export FLATPAK_USER_DIR
flatpak remote-add --user --gpg-import="$site/quoth.gpg" \
  quoth-test "file://$site/flatpak/repo"
flatpak remote-info --user quoth-test dev.quoth.Quoth
