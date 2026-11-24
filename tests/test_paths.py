# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.paths`: each location under its XDG base, by the spec's rules."""

import pytest

from quoth import paths

_LOCATIONS = [
    (paths.config_dir, "XDG_CONFIG_HOME", ".config", "quoth"),
    (paths.autostart_dir, "XDG_CONFIG_HOME", ".config", "autostart"),
    (paths.state_dir, "XDG_STATE_HOME", ".local/state", "quoth"),
    (paths.models_dir, "XDG_DATA_HOME", ".local/share", "quoth/models"),
    (paths.openvino_cache_dir, "XDG_CACHE_HOME", ".cache", "quoth/openvino"),
]


@pytest.mark.parametrize(("location", "variable", "default", "tail"), _LOCATIONS)
def test_an_absolute_base_is_used(
    location, variable, default, tail, tmp_path, monkeypatch
):
    monkeypatch.setenv(variable, str(tmp_path))
    assert location() == tmp_path / tail


@pytest.mark.parametrize("value", [None, "", "relative/dir"])
@pytest.mark.parametrize(("location", "variable", "default", "tail"), _LOCATIONS)
def test_an_unset_empty_or_relative_base_falls_back_to_home(
    location, variable, default, tail, value, tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    assert location() == tmp_path / default / tail
    assert location().is_absolute()


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ('File "/home/alice/x.py"', 'File "~/x.py"'),
        ("file:///home/alice/x", "file://~/x"),
        ("PATH=/usr/bin:/home/alice/bin", "PATH=/usr/bin:~/bin"),
        # A letter before it may end another directory's name, as in /var/home.
        ("-I/home/alice/x", "-I/home/alice/x"),
        ("/var/home/alice", "/var/home/alice"),
        ("/home/alice2", "/home/alice2"),
    ],
)
def test_only_a_whole_home_path_is_shown_as_a_tilde(text, shown, monkeypatch):
    monkeypatch.setenv("HOME", "/home/alice")
    monkeypatch.setattr(paths, "_HOME", paths._home_pattern())
    assert paths.tilde(text) == shown
