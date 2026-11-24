#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Cross-distro reach matrix for a built Quoth .deb.
# Proves one noble-built artifact installs and loads across Python versions wherever
# libadwaita >= 1.5 (Debian 13 py3.13, Ubuntu 26.04 py3.14; the unversioned
# dist-packages install is Python-version-agnostic), and is refused cleanly where the
# declared compatibility floors are unavailable (Ubuntu 22.04, Debian 12). It installs
# the prebuilt .deb only.
#
# Usage: matrix.sh /path/to/quoth_0.1.0_amd64.deb <release> <environment>
set -uo pipefail

DEB="${1:?usage: matrix.sh <path-to-quoth_*.deb> <release> <environment>}"
EXPECTED_RELEASE="${2:?missing expected release}"
EXPECTED_ENVIRONMENT="${3:?missing expected environment}"
[ -f "$DEB" ] || { echo "no such .deb: $DEB"; exit 1; }
DEBDIR="$(cd "$(dirname "$DEB")" && pwd)"; DEBNAME="$(basename "$DEB")"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="${TMPDIR:-/tmp}"
echo "Matrix target: $DEBDIR/$DEBNAME"

expect_success() {  # $1 image  $2 label
  docker run --rm -v "$DEBDIR":/deb:ro -v "$REPO/scripts":/scripts:ro "$1" bash -c "
    set -e; export DEBIAN_FRONTEND=noninteractive LC_ALL=C
    apt-get update -o APT::Update::Error-Mode=any -qq
    apt-get install -y -qq --no-install-recommends /deb/$DEBNAME
    python3 --version
    apt-get install -y -qq --no-install-recommends binutils xvfb xauth
    env GDK_BACKEND=x11 GTK_A11Y=none \\
      timeout 30s xvfb-run -a -s '-screen 0 1024x768x24' \\
      python3 -I /scripts/check_install.py \\
      --package-channel debian \\
      --expected-release '$EXPECTED_RELEASE' \\
      --expected-environment '$EXPECTED_ENVIRONMENT'
  " > "$LOGDIR/matrix-$2.log" 2>&1
}

expect_refusal() {  # $1 image  $2 label
  docker run --rm -v "$DEBDIR":/deb:ro "$1" bash -c "
    set -e; export DEBIAN_FRONTEND=noninteractive LC_ALL=C
    apt-get update -o APT::Update::Error-Mode=any -qq
    if apt-get install -y --no-install-recommends /deb/$DEBNAME >/tmp/inst.log 2>&1; then
      echo UNEXPECTED_INSTALL; exit 1
    fi
    # Expected only when every relation apt rejects is a version floor the package
    # declares and this distro's candidate is below.
    DECLARED=\$(dpkg-deb -f /deb/$DEBNAME Depends)
    KNOWN=0; UNEXPECTED=0
    while IFS= read -r relation; do
      PACKAGE=''; FLOOR=''; CANDIDATE=''
      read -r PACKAGE FLOOR < <(printf '%s\n' \"\$relation\" |
        sed -n 's/.*Depends: \([^ ]*\) (>= \([^)]*\)).*/\1 \2/p') || true
      if [ -n \"\$FLOOR\" ] && [[ \"\$DECLARED\" == *\"\$PACKAGE (>= \$FLOOR)\"* ]]; then
        CANDIDATE=\$(apt-cache policy \"\$PACKAGE\" |
          sed -n 's/^[[:space:]]*Candidate:[[:space:]]*//p')
      fi
      if [ -n \"\$CANDIDATE\" ] && [ \"\$CANDIDATE\" != '(none)' ] &&
        dpkg --compare-versions \"\$CANDIDATE\" lt \"\$FLOOR\"; then
        KNOWN=1
      else
        UNEXPECTED=1
      fi
    done < <(grep -E '(Depends|Pre-?Depends|Conflicts|Breaks):' /tmp/inst.log || true)
    if [ \"\$KNOWN\" != 1 ] || [ \"\$UNEXPECTED\" = 1 ]; then
      echo 'Install failed outside the declared compatibility floors'
      cat /tmp/inst.log
      exit 1
    fi
    echo 'REFUSED (expected):'
    grep -E 'Depends:' /tmp/inst.log | head -4
  " > "$LOGDIR/matrix-$2.log" 2>&1
}

echo "== 4 distro tests in parallel =="
expect_success "debian:trixie" trixie   & P1=$!
expect_success "ubuntu:26.04"  resolute & P2=$!
expect_refusal "ubuntu:22.04"  jammy    & P3=$!
expect_refusal "debian:12"     bookworm & P4=$!
wait $P1; R1=$?; wait $P2; R2=$?; wait $P3; R3=$?; wait $P4; R4=$?

v() { [ "$1" = 0 ] && echo PASS || echo FAIL; }
echo ""
echo "================ MATRIX RESULTS ================"
echo "Debian 13 trixie (py3.13, Adw1.7)  install+load : $(v $R1)"
echo "Ubuntu 26.04     (py3.14, Adw1.9)  install+load : $(v $R2)"
echo "Ubuntu 22.04     (Adw1.1/urllib3)   refusal      : $(v $R3)"
echo "Debian 12        (Adw1.2)           refusal      : $(v $R4)"
echo "==============================================="
grep -hE "RESULT:|REFUSED|Depends|FAIL|Error" "$LOGDIR"/matrix-{trixie,resolute,jammy,bookworm}.log | sed 's/^/  /'
[ "$R1" = 0 ] && [ "$R2" = 0 ] && [ "$R3" = 0 ] && [ "$R4" = 0 ] && echo "MATRIX_RESULT=PASS" || { echo "MATRIX_RESULT=FAIL"; exit 1; }
