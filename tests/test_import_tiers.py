# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for the `pyproject.toml` import tiers on what import-linter cannot check."""

import subprocess
import sys
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Gtk, Gdk and Adw are subpackages of the external gi package, which import-linter
# cannot forbid separately from GLib, so this imports the modules and names the first
# one that loads any of them.
_PROBE = """
import importlib, sys
for name in sys.argv[1:]:
    importlib.import_module("quoth." + name)
    if {"gi.repository.Gtk", "gi.repository.Gdk", "gi.repository.Adw"} & sys.modules.keys():
        print(name)
        break
"""


_CONTRACTS = tomllib.loads(_PYPROJECT.read_text())["tool"]["importlinter"]["contracts"]


def _below_ui() -> list[list[str]]:
    (layers,) = (c["layers"] for c in _CONTRACTS if c["type"] == "layers")
    tiers = [[name.strip() for name in tier.split(":")] for tier in layers]
    ui = next(i for i, tier in enumerate(tiers) if "settings_window" in tier)
    return tiers[ui + 1 :]


def test_the_gi_rule_covers_every_service_and_domain_module():
    services, _adapters, domain = _below_ui()
    (rule,) = (c for c in _CONTRACTS if c.get("forbidden_modules") == ["gi"])
    assert sorted(rule["source_modules"]) == sorted(
        "quoth." + name for name in services + domain
    )


def test_nothing_below_the_ui_tier_loads_gtk():
    below_ui = [name for tier in _below_ui() for name in tier]
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE, *below_ui],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout == ""
