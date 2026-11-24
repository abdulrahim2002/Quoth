# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.environment`: sandbox, host-exec and desktop facts."""

import pytest

from quoth import environment


class TestFlatpakDetection:
    def test_in_flatpak_and_app_path(self, tmp_path, monkeypatch):
        info = tmp_path / "flatpak-info"
        info.write_text("[Instance]\napp-path=/host/app\n")
        monkeypatch.setattr(environment, "_FLATPAK_INFO", str(info))
        assert environment.in_flatpak() is True
        assert environment.flatpak_app_path() == "/host/app"

    def test_not_in_flatpak(self, monkeypatch):
        monkeypatch.setattr(environment, "_FLATPAK_INFO", "/nonexistent/flatpak-info")
        assert environment.in_flatpak() is False
        assert environment.flatpak_app_path() is None

    def test_app_path_missing_key(self, tmp_path, monkeypatch):
        info = tmp_path / "flatpak-info"
        info.write_text("[Instance]\nother=x\n")
        monkeypatch.setattr(environment, "_FLATPAK_INFO", str(info))
        assert environment.flatpak_app_path() is None


class TestSnapDetection:
    def test_in_snap_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SNAP", "/snap/quoth/42")
        assert environment.in_snap() is True

    def test_not_in_snap_when_unset(self, monkeypatch):
        monkeypatch.delenv("SNAP", raising=False)
        assert environment.in_snap() is False


class TestAppImageDetection:
    def test_in_appimage_when_env_set(self, monkeypatch):
        monkeypatch.setenv("APPIMAGE", "/tmp/Quoth.AppImage")
        assert environment.in_appimage() is True

    def test_not_in_appimage_when_unset(self, monkeypatch):
        monkeypatch.delenv("APPIMAGE", raising=False)
        assert environment.in_appimage() is False


class TestHostArgv:
    def test_wraps_inside_sandbox(self, tmp_path, monkeypatch):
        info = tmp_path / "flatpak-info"
        info.write_text("[Instance]\napp-path=/a\n")
        monkeypatch.setattr(environment, "_FLATPAK_INFO", str(info))
        assert environment.host_argv(["foo", "bar"]) == [
            "flatpak-spawn",
            "--host",
            "foo",
            "bar",
        ]

    def test_passthrough_outside_sandbox(self, monkeypatch):
        monkeypatch.setattr(environment, "_FLATPAK_INFO", "/nonexistent")
        assert environment.host_argv(["foo", "bar"]) == ["foo", "bar"]


class TestDesktopNames:
    def test_splits_and_lowercases(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "pop:GNOME")
        assert environment.desktop_names() == ["pop", "gnome"]

    def test_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("XDG_CURRENT_DESKTOP", raising=False)
        assert environment.desktop_names() == []

    def test_strips_blanks(self, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "COSMIC:")
        assert environment.desktop_names() == ["cosmic"]


class TestWaylandSession:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        monkeypatch.setenv("DISPLAY", ":0")

    def test_the_named_socket(self, monkeypatch, tmp_path):
        (tmp_path / "wayland-1").touch()
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
        assert environment.wayland_session() is True

    def test_a_name_without_its_socket(self, monkeypatch):
        # Flatpak sets the name even when the host has no Wayland socket to share.
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert environment.wayland_session() is False

    def test_the_default_socket_without_a_name(self, tmp_path):
        # A service manager can start the app without the compositor's environment.
        (tmp_path / "wayland-0").touch()
        assert environment.wayland_session() is True

    def test_no_socket_without_a_name(self):
        assert environment.wayland_session() is False

    def test_an_absolute_name_needs_no_runtime_dir(self, monkeypatch, tmp_path):
        (tmp_path / "compositor").touch()
        monkeypatch.setenv("WAYLAND_DISPLAY", str(tmp_path / "compositor"))
        monkeypatch.delenv("XDG_RUNTIME_DIR")
        assert environment.wayland_session() is True

    def test_a_relative_name_needs_the_runtime_dir(self, monkeypatch, tmp_path):
        (tmp_path / "wayland-0").touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("XDG_RUNTIME_DIR")
        assert environment.wayland_session() is False


class TestDesktopSupportsLayerShell:
    @pytest.mark.parametrize(
        ("desktop", "supported"),
        [
            ("COSMIC", True),
            ("sway", True),
            ("Hyprland", True),
            ("KDE", False),
            ("ubuntu:GNOME", False),
        ],
    )
    def test_desktop(self, monkeypatch, desktop, supported):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
        assert environment.desktop_supports_layer_shell() is supported

    def test_unset_does_not(self, monkeypatch):
        monkeypatch.delenv("XDG_CURRENT_DESKTOP", raising=False)
        assert environment.desktop_supports_layer_shell() is False


class TestUsableCpus:
    def test_returns_at_least_one(self):
        assert environment.usable_cpus() >= 1

    def test_uses_affinity_mask(self, monkeypatch):
        # The affinity mask respects cgroup/taskset/Flatpak CPU limits.
        monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1, 2, 3})
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        assert environment.usable_cpus() == 4

    def test_falls_back_to_cpu_count(self, monkeypatch):
        def _no_affinity(pid):
            raise OSError("unavailable")

        monkeypatch.setattr("os.sched_getaffinity", _no_affinity)
        monkeypatch.setattr("os.cpu_count", lambda: 6)
        assert environment.usable_cpus() == 6

    def test_floors_at_one(self, monkeypatch):
        def _no_affinity(pid):
            raise OSError("unavailable")

        monkeypatch.setattr("os.sched_getaffinity", _no_affinity)
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert environment.usable_cpus() == 1
