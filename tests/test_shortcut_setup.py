# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `shortcut_setup`: desktop-managed shortcuts and manual toggle setup."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quoth import APP_ID, environment, shortcut_setup


class TestToggleCommand:
    def test_flatpak_form(self):
        with patch.object(environment, "in_flatpak", return_value=True):
            assert shortcut_setup.toggle_command() == [
                "flatpak",
                "run",
                APP_ID,
                "--toggle",
            ]
            assert shortcut_setup.toggle_command_str() == (
                f"flatpak run {APP_ID} --toggle"
            )

    def test_native_form(self):
        with patch.object(environment, "in_flatpak", return_value=False):
            assert shortcut_setup.toggle_command() == ["quoth", "--toggle"]


class TestKeyboardSettingsArgv:
    @pytest.fixture(autouse=True)
    def _native(self, monkeypatch):
        monkeypatch.setattr(environment, "in_flatpak", lambda: False)

    def test_cosmic(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "pop:COSMIC")
        assert shortcut_setup.keyboard_settings_argv() == [
            "cosmic-settings",
            "keyboard",
        ]

    def test_gnome(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
        assert shortcut_setup.keyboard_settings_argv()[0] == "gnome-control-center"

    def test_kde_and_plasma(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
        assert shortcut_setup.keyboard_settings_argv() == [
            "systemsettings",
            "kcm_keys",
        ]
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "plasma")
        assert shortcut_setup.keyboard_settings_argv() == [
            "systemsettings",
            "kcm_keys",
        ]

    def test_tiling_wms_have_none(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
        assert shortcut_setup.keyboard_settings_argv() is None
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "Hyprland")
        assert shortcut_setup.keyboard_settings_argv() is None

    def test_unknown_has_none(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "")
        assert shortcut_setup.keyboard_settings_argv() is None

    def test_matches_whole_entries_only(self, monkeypatch):
        # A desktop whose name merely contains another's is not that desktop.
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "mygnome")
        assert shortcut_setup.keyboard_settings_argv() is None


class TestSetupSteps:
    # The snippets avoid Pause, the native hotkey, so a desktop binding and a native
    # listener that starts later never both toggle on one press.
    @pytest.fixture(autouse=True)
    def _native(self, monkeypatch):
        monkeypatch.setattr(environment, "in_flatpak", lambda: False)

    def test_hyprland_gives_the_lua_and_the_conf_form(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "Hyprland")
        steps = shortcut_setup.setup_steps()
        lua = 'hl.bind("SUPER + SHIFT + D", hl.dsp.exec_cmd("quoth --toggle"))'
        assert "hyprland.lua" in steps and lua in steps
        assert "hyprland.conf" in steps
        assert "bind = SUPER SHIFT, D, exec, quoth --toggle" in steps
        assert "Pause" not in steps

    def test_sway_binds_a_free_key_once_per_press(self, monkeypatch):
        # Super+D is the stock launcher; a held key must not toggle repeatedly.
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
        steps = shortcut_setup.setup_steps()
        assert "bindsym --no-repeat Mod4+Shift+d exec quoth --toggle" in steps
        assert "Pause" not in steps

    def test_gui_desktop_walkthrough(self, monkeypatch):
        # Both forms say "custom shortcut"; only this one leads to the launcher.
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "COSMIC")
        steps = shortcut_setup.setup_steps()
        assert "Open keyboard settings" in steps

    def test_unknown_generic(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "")
        steps = shortcut_setup.setup_steps()
        assert "custom shortcut" in steps.lower()
        assert "Open keyboard settings" not in steps  # no launcher to open

    @pytest.mark.parametrize(
        ("desktop", "snippet"), [("swayfx", "bindsym"), ("hyprland-git", "hyprland")]
    )
    def test_a_name_containing_another_is_not_that_desktop(
        self, monkeypatch, desktop, snippet
    ):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
        assert snippet not in shortcut_setup.setup_steps()


class TestOpenKeyboardSettings:
    @pytest.fixture(autouse=True)
    def _native_cosmic(self, monkeypatch):
        monkeypatch.setattr(environment, "in_flatpak", lambda: False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "COSMIC")

    def test_opens_native_launcher(self):
        with patch("quoth.shortcut_setup.subprocess.Popen") as popen:
            assert shortcut_setup.open_keyboard_settings() is True
            assert popen.call_args.args[0] == [
                "cosmic-settings",
                "keyboard",
            ]

    def test_returns_false_without_launcher(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
        assert shortcut_setup.open_keyboard_settings() is False

    def test_returns_false_when_the_launcher_cannot_start(self):
        # The caller then says so and keeps the command to bind by hand.
        with patch(
            "quoth.shortcut_setup.subprocess.Popen",
            side_effect=FileNotFoundError("cosmic-settings"),
        ):
            assert shortcut_setup.open_keyboard_settings() is False


def test_flatpak_never_launches_host_settings(monkeypatch):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    with (
        patch.object(environment, "in_flatpak", return_value=True),
        patch("quoth.shortcut_setup.subprocess.Popen") as popen,
    ):
        assert shortcut_setup.keyboard_settings_argv() is None
        assert not shortcut_setup.open_keyboard_settings()
        popen.assert_not_called()
        steps = shortcut_setup.setup_steps()
        assert "custom shortcut" in steps and "Open keyboard settings" not in steps


@pytest.mark.parametrize("version", [1, 2])
def test_portal_binding_is_desktop_authority(version):
    view = shortcut_setup.portal_presentation(
        SimpleNamespace(status="ready", binding="Super+D", version=version)
    )
    assert view.title == "Super+D"
    assert view.ready and not view.manual
    unnamed = shortcut_setup.portal_presentation(
        SimpleNamespace(status="ready", binding="", version=version)
    )
    assert unnamed.title and unnamed.ready
    assert view.can_configure == (version >= 2)
    assert bool(view.action) == view.can_configure


@pytest.mark.parametrize("version", [1, 2])
def test_portal_shortcut_without_a_key_points_to_the_desktop(monkeypatch, version):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    view = shortcut_setup.portal_presentation(
        SimpleNamespace(status="needs-key", binding="", version=version)
    )
    assert view.title
    assert "your desktop's keyboard settings" in view.description
    assert not view.ready and not view.manual
    assert view.can_configure == (version >= 2)
    assert bool(view.action) == view.can_configure


@pytest.mark.parametrize(
    ("desktop", "place"),
    [
        ("ubuntu:GNOME", "Settings → Apps → Quoth → Global Shortcuts."),
        ("Hyprland", "bind = SUPER SHIFT, D, global, dev.quoth.Quoth:toggle-dictation"),
        ("KDE", "your desktop's keyboard settings."),
        # A name that only contains another desktop's is not that desktop.
        ("GNOME-Classic", "your desktop's keyboard settings."),
        ("hyprland-git", "your desktop's keyboard settings."),
    ],
)
def test_portal_key_steps_name_where_the_desktop_keeps_them(
    monkeypatch, desktop, place
):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
    for status in ("needs-key", "ready"):
        view = shortcut_setup.portal_presentation(
            SimpleNamespace(status=status, binding="", version=1)
        )
        assert place in view.description


@pytest.mark.parametrize(
    "status",
    ["checking", "unbound", "requesting", "denied", "closed", "error", "unavailable"],
)
def test_portal_setup_actions_follow_status(status):
    view = shortcut_setup.portal_presentation(SimpleNamespace(status=status))
    assert view.title and view.description
    assert not view.ready
    assert view.manual == (status == "unavailable")
    assert view.can_configure == (status in {"unbound", "denied", "closed", "error"})
