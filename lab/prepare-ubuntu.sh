#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

# Enable key-only SSH for the lab user in a disposable Ubuntu copy.
# Run it as root inside that copy, with the public key (id_ed25519.pub) beside it.
# Usage: prepare-ubuntu.sh [USER]   (USER defaults to quoth)
set -eu

user=${1:-quoth}
home_dir=$(getent passwd "$user" | cut -d: -f6)
test -n "$home_dir"
test "$(id -u)" -eq 0
apt-get update
apt-get install -y --no-install-recommends openssh-server
install -d -m 700 -o "$user" -g "$user" "$home_dir/.ssh"
install -m 600 -o "$user" -g "$user" "$(dirname "$0")/id_ed25519.pub" \
    "$home_dir/.ssh/authorized_keys"
printf '%s\n' 'PasswordAuthentication no' 'KbdInteractiveAuthentication no' \
    'PermitRootLogin no' > /etc/ssh/sshd_config.d/00-quoth-lab.conf
systemctl enable --now ssh
systemctl restart ssh
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
