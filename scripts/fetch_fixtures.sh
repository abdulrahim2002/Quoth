#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Fetch the public transcription fixtures the package install tests use into DIR and
# check both against their pinned hashes. A model already in DIR, as the workflow's
# cache restores it, is kept but still checked.
#
#   MODEL_URL=... MODEL_SHA256=... AUDIO_URL=... AUDIO_SHA256=... \
#     scripts/fetch_fixtures.sh DIR
#
# The four values live in package.yml's env, which also keys the model's cache.
set -euo pipefail

mkdir -p "$1"
cd "$1"
if [ ! -f ggml-base-q8_0.bin ]; then
  curl --fail --location --show-error --output ggml-base-q8_0.bin "$MODEL_URL"
fi
curl --fail --location --show-error --output jfk.wav "$AUDIO_URL"
printf '%s  ggml-base-q8_0.bin\n%s  jfk.wav\n' "$MODEL_SHA256" "$AUDIO_SHA256" | sha256sum -c -
