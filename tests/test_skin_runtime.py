# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `skin_runtime`, the live app skin.

A real config directory drives each runtime; the scheduler fake holds monitor work
until a test runs it, as the main loop would.
"""

from unittest.mock import MagicMock, patch

import pytest
from gi.repository import Gio, GLib

from quoth import atomic_write
from quoth.skin import DEFAULT_SKIN, load_skin
from quoth.skin_runtime import SkinRuntime
from tests.support import DeferredScheduler, wait_until

_LOAD_ERROR = "Couldn't load the skin. Keeping the current appearance."
_RENDER_ERROR = "Couldn't update the app appearance."
_QUIET = "kind = 'skin'\n[dark]\naccent = '#AA0000'\n[meta]\nname = 'Quiet'\n"
_EDITED = "kind = 'skin'\n[dark]\naccent = '#00AA00'\n"
_BROKEN = "kind = 'skin'\n[dark]\naccent = '#12345'\n"


@pytest.fixture
def config_dir(tmp_path):
    shelf = tmp_path / "skins"
    shelf.mkdir()
    (shelf / "quiet.toml").write_text(_QUIET)
    return tmp_path


@pytest.fixture
def runtime(config_dir):
    runtime = SkinRuntime(
        config_dir, DeferredScheduler(), render=MagicMock(), on_changed=MagicMock()
    )
    yield runtime
    runtime.shutdown()


def _drawn(runtime):
    return [call.args[0] for call in runtime._render.call_args_list]


def _changed(path, event=Gio.FileMonitorEvent.CHANGES_DONE_HINT):
    return (None, Gio.File.new_for_path(str(path)), None, event)


def _edit(runtime, content):
    """Write the active file as an editor would and deliver the monitor event."""
    if content is None:
        runtime._path.unlink()
    else:
        runtime._path.write_text(content)
    runtime._on_file_changed(*_changed(runtime._path))
    runtime._scheduler.fire_dispatches()


class TestStartup:
    def test_no_active_file_keeps_the_installed_default(self, runtime):
        runtime.start()

        runtime._render.assert_not_called()
        runtime._on_changed.assert_called_once_with(None)
        assert runtime.current() == "default"
        assert runtime.load_error() is None

    def test_an_active_skin_is_drawn_once(self, runtime, config_dir):
        (config_dir / "skin.toml").write_text(_QUIET)

        runtime.start()

        assert _drawn(runtime) == [runtime.skin_of("user:quiet")]
        assert runtime.current() == "user:quiet"

    def test_a_broken_file_keeps_default_and_still_lists_the_shelf(
        self, runtime, config_dir, caplog
    ):
        (config_dir / "skin.toml").write_text(_BROKEN)

        runtime.start()

        # The log names the file, which the toast cannot.
        assert any(
            m.startswith(f"Could not load app skin {config_dir / 'skin.toml'}: ")
            for m in caplog.messages
        )

        runtime._render.assert_not_called()
        runtime._on_changed.assert_called_once_with(_LOAD_ERROR)
        assert runtime.load_error() == _LOAD_ERROR
        assert runtime.current() is None  # Settings asks before replacing it
        assert runtime.user_skins() == ["user:quiet"]

    def test_monitor_failure_still_loads_and_applies(self, runtime, config_dir, caplog):
        (config_dir / "skin.toml").write_text(_QUIET)
        caplog.set_level("WARNING")
        with patch("quoth.skin_runtime.Gio.File.new_for_path") as file:
            monitor_directory = file.return_value.monitor_directory
            monitor_directory.side_effect = GLib.Error("unavailable")
            runtime.start()

        [warning] = caplog.messages
        assert "hot-reload unavailable" in warning
        runtime.apply("default")
        assert _drawn(runtime) == [runtime.skin_of("user:quiet"), DEFAULT_SKIN]


class TestHotReload:
    def test_the_directory_monitor_follows_saves_and_removal(self, runtime):
        runtime.start()

        # An atomic save reaches the monitor as a rename onto skin.toml.
        atomic_write.write_text(runtime._path, _QUIET)
        wait_until(lambda: runtime._scheduler.dispatches)
        runtime._scheduler.fire_dispatches()
        assert _drawn(runtime) == [runtime.skin_of("user:quiet")]

        runtime._path.unlink()
        wait_until(lambda: runtime._scheduler.dispatches)
        runtime._scheduler.fire_dispatches()
        assert _drawn(runtime)[-1] == DEFAULT_SKIN

    def test_an_invalid_edit_keeps_the_last_appearance_until_fixed(self, runtime):
        runtime._path.write_text(_QUIET)
        runtime.start()

        _edit(runtime, _BROKEN)
        assert runtime.load_error() == _LOAD_ERROR
        assert _drawn(runtime) == [runtime.skin_of("user:quiet")]
        runtime._on_changed.assert_called_with(_LOAD_ERROR)

        _edit(runtime, _EDITED)
        assert runtime.load_error() is None
        assert _drawn(runtime) == [
            runtime.skin_of("user:quiet"),
            load_skin(runtime._path),
        ]
        assert runtime.current() is None  # the edit matches no installed skin

    def test_removing_the_active_file_returns_to_default(self, runtime):
        runtime._path.write_text(_QUIET)
        runtime.start()

        _edit(runtime, None)

        assert _drawn(runtime)[-1] == DEFAULT_SKIN
        assert runtime.current() == "default"

    def test_a_metadata_only_edit_redraws_nothing(self, runtime):
        runtime._path.write_text(_QUIET)
        runtime.start()

        _edit(runtime, _QUIET.replace("Quiet", "Renamed"))

        assert len(_drawn(runtime)) == 1
        runtime._on_changed.assert_called_with(None)

    def test_a_render_failure_is_reported_and_retried_on_the_next_load(
        self, runtime, caplog
    ):
        runtime.start()
        runtime._render.side_effect = RuntimeError("provider gone")
        caplog.set_level("ERROR")
        _edit(runtime, _QUIET)
        [record] = caplog.records
        assert record.exc_info[0] is RuntimeError
        assert runtime.load_error() == _RENDER_ERROR
        runtime._on_changed.assert_called_with(_RENDER_ERROR)

        runtime._render.side_effect = None
        _edit(runtime, _QUIET)

        assert runtime.load_error() is None
        assert _drawn(runtime)[-1] == runtime.skin_of("user:quiet")

    def test_filters_other_files_and_coalesces_atomic_moves(self, runtime):
        active = Gio.File.new_for_path(str(runtime._path))
        temporary = Gio.File.new_for_path(str(runtime._path.parent / ".skin.tmp"))
        runtime._on_file_changed(None, temporary, None, Gio.FileMonitorEvent.DELETED)
        runtime._on_file_changed(*_changed(runtime._path, Gio.FileMonitorEvent.CHANGED))
        assert runtime._scheduler.dispatches == []

        runtime._on_file_changed(None, temporary, active, Gio.FileMonitorEvent.RENAMED)
        runtime._on_file_changed(*_changed(runtime._path))
        assert len(runtime._scheduler.dispatches) == 1  # one reload for both events
        runtime._scheduler.fire_dispatches()
        runtime._on_changed.assert_called_once_with(None)

        runtime._on_file_changed(*_changed(runtime._path))
        assert len(runtime._scheduler.dispatches) == 1  # a later edit queues again


class TestApply:
    def test_applying_copies_the_skin_and_draws_it_once(self, runtime):
        runtime.start()

        runtime.apply("user:quiet")
        # The monitor then reports Quoth's own write.
        runtime._on_file_changed(*_changed(runtime._path))
        runtime._scheduler.fire_dispatches()

        assert runtime._path.read_text() == _QUIET
        assert _drawn(runtime) == [runtime.skin_of("user:quiet")]
        assert runtime.current() == "user:quiet"

    def test_default_removes_the_active_file(self, runtime):
        runtime._path.write_text(_QUIET)
        runtime.start()

        runtime.apply("default")

        assert not runtime._path.exists()
        assert _drawn(runtime)[-1] == DEFAULT_SKIN

    def test_a_rejected_skin_leaves_the_file_and_still_reloads(
        self, runtime, config_dir
    ):
        runtime._path.write_text(_EDITED)
        (config_dir / "skins" / "broken.toml").write_text(_BROKEN)
        runtime.start()
        runtime._on_changed.reset_mock()

        with pytest.raises(ValueError, match="opaque"):
            runtime.apply("user:broken")

        assert runtime._path.read_text() == _EDITED
        runtime._on_changed.assert_called_once_with(None)

    def test_a_failed_write_still_renders_what_the_file_holds(
        self, runtime, monkeypatch
    ):
        runtime.start()

        def replace_then_fail(path, data, **_kwargs):
            path.write_bytes(data)
            raise OSError("sync failed")

        monkeypatch.setattr(atomic_write, "replace_bytes", replace_then_fail)
        with pytest.raises(OSError, match="sync failed"):
            runtime.apply("user:quiet")

        assert _drawn(runtime) == [runtime.skin_of("user:quiet")]


class TestImport:
    def test_import_adds_to_the_shelf_without_applying(self, runtime, tmp_path):
        runtime.start()
        source = tmp_path / "Petrol.toml"
        source.write_text(_EDITED)

        assert runtime.import_file(source) == "user:Petrol"

        assert runtime.user_skins() == ["user:Petrol", "user:quiet"]
        assert not runtime._path.exists()
        runtime._render.assert_not_called()


class TestShutdown:
    def test_stops_watching(self, runtime):
        with patch("quoth.skin_runtime.Gio.File.new_for_path") as file:
            runtime.start()
        monitor = file.return_value.monitor_directory.return_value

        runtime.shutdown()
        runtime.shutdown()

        monitor.cancel.assert_called_once_with()
