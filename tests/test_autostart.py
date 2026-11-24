# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `autostart`."""

import os
import resource
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from gi.repository import Gio, GLib

from quoth import autostart


def _desktop_file():
    return Path(os.environ["XDG_CONFIG_HOME"], "autostart", autostart.DESKTOP_FILENAME)


def test_enable_writes_desktop_file(monkeypatch, caplog):
    monkeypatch.setattr(autostart.shutil, "which", lambda name: f"/opt/bin/{name}")
    autostart.set_enabled(True)
    desktop = _desktop_file()
    assert desktop.exists()
    assert "Exec=quoth --background\n" in desktop.read_text()
    assert autostart.is_enabled() is True
    # Whether the entry's command is found is the first question when it does nothing.
    assert caplog.messages[-1] == "Autostart command resolves to /opt/bin/quoth"


def test_disable_removes_desktop_file():
    autostart.set_enabled(True)
    autostart.set_enabled(False)
    assert not _desktop_file().exists()
    assert autostart.is_enabled() is False


@pytest.mark.parametrize("switch", ["Hidden=true", "X-GNOME-Autostart-enabled=false"])
def test_an_entry_the_desktop_switched_off_reads_as_off(switch):
    autostart.set_enabled(True)
    desktop = _desktop_file()
    desktop.write_text(
        desktop.read_text().replace("X-GNOME-Autostart-enabled=true", switch)
    )
    assert autostart.is_enabled() is False


def test_disable_leaves_an_entry_the_desktop_switched_off():
    autostart.set_enabled(True)
    desktop = _desktop_file()
    desktop.write_text(desktop.read_text() + "Hidden=true\n")
    autostart.set_enabled(False)
    assert "Hidden=true" in desktop.read_text()


def test_disable_when_absent_is_noop():
    autostart.set_enabled(False)  # should not raise
    assert autostart.is_enabled() is False


def test_set_enabled_swallows_and_logs_oserror(caplog):
    # A filesystem failure (here a plain file where the autostart dir must be) is
    # logged and swallowed, so a launch-at-login toggle can't crash the app.
    autostart_dir = _desktop_file().parent
    autostart_dir.parent.mkdir()
    autostart_dir.write_text("not a directory")

    caplog.set_level("ERROR")
    autostart.set_enabled(True)  # must not raise

    [record] = caplog.records
    assert "Autostart enable failed" in record.getMessage()
    assert record.exc_info is None


def test_a_write_cut_short_leaves_no_entry():
    # The disk fills partway through the entry; half an entry would read as enabled.
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (autostart._DESKTOP_ENTRY.index("Exec="), hard)
    )
    try:
        autostart.set_enabled(True)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous)

    assert autostart.is_enabled() is False
    assert list(_desktop_file().parent.iterdir()) == []


class _FakePortal:
    """Records the background request and answers it as libportal would."""

    def __init__(self):
        self.request = None
        self.answer = None

    def request_background(self, *args):
        self.request = args

    def respond(self, answer):
        self.answer = answer
        on_done = self.request[5]
        on_done(self, object(), self.request[6])

    def request_background_finish(self, _result):
        if isinstance(self.answer, GLib.Error):
            raise self.answer
        return self.answer


def _portal_error(code):
    return GLib.Error.new_literal(Gio.io_error_quark(), "Background request", code)


# GDBus's reply when the portal exports no Background interface, as on COSMIC.
_NO_BACKGROUND_PORTAL = GLib.Error.new_literal(
    Gio.DBusError.quark(),
    "No such interface “org.freedesktop.portal.Background”",
    Gio.DBusError.UNKNOWN_METHOD,
)


class TestFlatpakPortal:
    """The libportal background request and how its answer is read."""

    @pytest.fixture
    def portal(self, monkeypatch):
        portal = _FakePortal()
        xdp = SimpleNamespace(
            Portal=SimpleNamespace(initable_new=lambda: portal),
            BackgroundFlags=SimpleNamespace(NONE=0, AUTOSTART=1),
        )
        monkeypatch.setattr(autostart.environment, "in_flatpak", lambda: True)
        monkeypatch.setattr(autostart, "_Xdp", xdp)
        return portal

    @pytest.mark.parametrize(("enabled", "flags"), [(True, 1), (False, 0)])
    def test_request_carries_the_autostart_wish(self, portal, enabled, flags):
        autostart.set_enabled(enabled)
        parent, _reason, commandline, sent_flags, cancellable = portal.request[:5]
        assert parent is None
        assert commandline == ["quoth", "--background"]
        assert sent_flags == flags
        assert cancellable is None

    @pytest.mark.parametrize(
        ("enabled", "answer", "declined"),
        [
            (True, True, []),
            (True, False, [False]),
            (False, True, []),
            (False, False, []),
            (True, _portal_error(Gio.IOErrorEnum.CANCELLED), [False]),
            (False, _portal_error(Gio.IOErrorEnum.CANCELLED), [False]),
            (True, _portal_error(Gio.IOErrorEnum.FAILED), []),
            (True, _NO_BACKGROUND_PORTAL, [True]),
            (False, _NO_BACKGROUND_PORTAL, [True]),
        ],
    )
    def test_only_an_actual_decline_is_reported(
        self, portal, enabled, answer, declined
    ):
        reported = []
        autostart.set_enabled(enabled, reported.append)
        portal.respond(answer)
        assert reported == declined

    def test_host_path_never_reports_a_decline(self):
        autostart_dir = _desktop_file().parent
        autostart_dir.parent.mkdir()
        autostart_dir.write_text("not a directory")
        declined = []
        autostart.set_enabled(True, declined.append)
        assert declined == []

    def test_no_bus_is_logged_not_raised(self, portal, monkeypatch, caplog):
        def no_bus():
            raise GLib.Error("no bus")

        monkeypatch.setattr(autostart._Xdp.Portal, "initable_new", no_bus)
        caplog.set_level("WARNING")
        autostart.set_enabled(True)  # must not raise
        assert portal.request is None
        assert [r.levelname for r in caplog.records] == ["WARNING"]
