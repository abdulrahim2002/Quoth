# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.presentation`: the overlay, its native hint and feedback routing."""

from unittest.mock import patch

import pytest

from quoth import presentation


def test_overlay_unavailable_without_layer_shell():
    with patch("quoth.presentation.Gtk4LayerShell", None):
        assert presentation.overlay_available() is False


@pytest.mark.parametrize("supported", [True, False])
def test_overlay_follows_the_layer_shell_probe(supported):
    with patch("quoth.presentation.Gtk4LayerShell") as layer_shell:
        layer_shell.is_supported.return_value = supported
        assert presentation.overlay_available() is supported


@pytest.mark.parametrize(
    ("in_flatpak", "overlay_avail", "desktop_supports", "expected"),
    [
        (True, False, True, True),  # sandboxed COSMIC, no pill → suggest native
        (
            False,
            False,
            True,
            False,
        ),  # native build → never suggest (it's already native)
        (True, True, True, False),  # sandboxed but pill works (e.g. wlroots) → no need
        (True, False, False, False),  # sandboxed GNOME → native wouldn't help either
        (False, True, True, False),  # native with a working pill → no need
    ],
)
def test_suggests_native_overlay(in_flatpak, overlay_avail, desktop_supports, expected):
    with (
        patch("quoth.presentation.environment.in_flatpak", return_value=in_flatpak),
        patch("quoth.presentation.overlay_available", return_value=overlay_avail),
        patch(
            "quoth.presentation.environment.desktop_supports_layer_shell",
            return_value=desktop_supports,
        ),
    ):
        assert presentation.suggests_native_overlay() is expected


@pytest.mark.parametrize(
    ("setting", "overlay_avail", "expected"),
    [
        ("on", True, True),  # forced on, regardless of overlay
        ("on", False, True),
        ("off", True, False),  # forced off, regardless of overlay
        ("off", False, False),
        ("auto", True, False),  # auto + pill present → quiet
        ("auto", False, True),  # auto + no pill → feedback on
    ],
)
def test_feedback_enabled(setting, overlay_avail, expected):
    with patch("quoth.presentation.overlay_available", return_value=overlay_avail):
        assert presentation.feedback_enabled(setting) is expected


@pytest.mark.parametrize(
    ("essential", "overlay", "expected"),
    [
        (False, True, False),  # transient + pill → pill only, no notification
        (False, False, True),  # transient + no pill → notification is the only channel
        (True, True, True),  # essential + pill → also notify (persists past auto-hide)
        (True, False, True),  # essential + no pill → notification
    ],
)
def test_error_uses_notification(essential, overlay, expected):
    assert presentation.error_uses_notification(essential, overlay) is expected
