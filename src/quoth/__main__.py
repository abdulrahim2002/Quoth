# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The `python -m quoth` entry point."""

import sys

from quoth.cli import main

sys.exit(main())
