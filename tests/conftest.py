# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Shared test fixtures."""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

# Redirect the XDG base dirs to a throwaway tree before any quoth module is imported:
# quoth.log names its state directory, where crash.log goes, from XDG_STATE_HOME at
# import, and test modules import it at collection, before any fixture runs. Override
# unconditionally so a developer's own XDG_* never receives the suite's writes; the
# per-test `_isolate_xdg_dirs` fixture then gives each test a fresh tree.
_XDG_ROOT = tempfile.mkdtemp(prefix="quoth-tests-")
atexit.register(shutil.rmtree, _XDG_ROOT, ignore_errors=True)
for _var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
    os.environ[_var] = os.path.join(_XDG_ROOT, _var.lower())

# Settings live in GSettings: compile the app's schema into the throwaway tree and use
# GIO's in-memory backend, so no test reads or writes the developer's real settings.
# Both are read when GIO first opens settings, so they too are set at conftest load.
_SCHEMA_DIR = os.path.join(_XDG_ROOT, "schemas")
os.makedirs(_SCHEMA_DIR)
shutil.copy(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "dev.quoth.Quoth.gschema.xml"
    ),
    _SCHEMA_DIR,
)
subprocess.run(["glib-compile-schemas", "--strict", _SCHEMA_DIR], check=True)
os.environ["GSETTINGS_SCHEMA_DIR"] = _SCHEMA_DIR
os.environ["GSETTINGS_BACKEND"] = "memory"

# GTK's test accessibility backend, as CI uses: it needs no session accessibility bus,
# whose absence GTK warns about, and lets tests read widgets' accessible state.
os.environ.setdefault("GTK_A11Y", "test")

# Import quoth before anything below touches gi: quoth/__init__ puts the bundled
# gtk4-layer-shell typelib on GI_TYPELIB_PATH, which gi reads only on its first
# namespace lookup, so the suite loads the namespaces the app does. This must stay
# after the XDG block above, for the reason given there.
import quoth  # noqa: E402
from quoth import config as _config  # noqa: E402

# The suite's freshly compiled schema, not a checkout's `make schemas` copy that may be
# older than the XML.
_config._SOURCE_SCHEMAS = Path(_SCHEMA_DIR)

import gi  # noqa: E402

# The GI versions the app declares, set before any test module imports a namespace:
# importing one unversioned warns, and the suite fails on warnings.
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import GLib  # noqa: E402

# GLib reports a failed precondition or a misused widget as a critical or warning on
# stderr and carries on, so the test that caused it would still pass. This writer notes
# those for `_fail_on_glib_warnings`; GLib aborts on a second one, so nothing else may
# install a writer.
_glib_warnings: list[str] = []


def _note_glib_warnings(level, fields, n_fields, user_data):
    if level & (GLib.LogLevelFlags.LEVEL_CRITICAL | GLib.LogLevelFlags.LEVEL_WARNING):
        _glib_warnings.append(GLib.log_writer_format_fields(level, fields, False))
    return GLib.log_writer_default(level, fields, user_data)


GLib.log_set_writer_func(_note_glib_warnings, None)

# Format every log call, TRACE included, so a broken placeholder fails its test: the
# app's own sinks skip anything below DEBUG. quoth.log replaces every sink when first
# imported, so it comes first.
import quoth.log  # noqa: E402

logger.add(lambda _message: None, level="TRACE", diagnose=False, backtrace=False)

# `scheduler.guarded` logs a callback's raise and carries on, as PyGObject does through
# `sys.excepthook`: this notes those for `_fail_on_callback_errors`.
_guarded_raises: list[str] = []
logger.add(
    lambda message: _guarded_raises.append(message.record["message"]),
    level="ERROR",
    filter="quoth.scheduler",
)


def _display_available() -> bool:
    """Whether there is a display to build real widgets on.

    Without one, constructing a GTK widget crashes the whole test process instead of
    raising, and `Gtk.init_check()` still returns True, so this asks Gdk directly.
    Importing Gtk triggers PyGObject's implicit init, which the Gdk check requires.
    """
    from gi.repository import Gdk, Gtk  # noqa: F401 (import Gtk to trigger GTK init)

    return Gdk.Display.get_default() is not None


@pytest.fixture
def display():
    """Skip a test that builds real widgets when there is no display."""
    if not _display_available():
        pytest.skip("no display")


@pytest.fixture(autouse=True)
def _isolate_layer_shell(monkeypatch):
    """Headless and X11 tests have no layer-shell unless a test fakes it."""
    from gi.repository import Gdk

    from quoth import presentation

    display = Gdk.Display.get_default()
    if display is None or display.__gtype__.name == "GdkX11Display":
        monkeypatch.setattr(presentation, "Gtk4LayerShell", None)


@pytest.fixture
def find_widget():
    from tests.support import descendants

    def find(root, widget_type, title):
        return next(
            (
                widget
                for widget in (root, *descendants(root))
                if isinstance(widget, widget_type) and widget.get_title() == title
            ),
            None,
        )

    return find


@pytest.fixture
def layer_shell(display):
    """Grant every `WaveformPill` built during the test a fake wlr-layer-shell.

    A pill stays inert without one (its show_* methods no-op), and a virtual display
    grants none. A hidden pill stays mapped, so each is destroyed before the fake goes:
    a later resize would reach the real shell. Yields the fake `Gtk4LayerShell`.
    """
    from unittest.mock import patch

    from gi.repository import Gtk

    from quoth import overlay

    with (
        patch.object(overlay.presentation, "overlay_available", return_value=True),
        patch.object(overlay, "Gtk4LayerShell") as shell,
    ):
        yield shell
        # A pill's queued idle would otherwise reach the real shell after the fake. The
        # bound turns an idle that repeats forever into a failure, not a hang.
        context = GLib.MainContext.default()
        for _ in range(1000):
            if not context.pending():
                break
            context.iteration(False)
        else:
            pytest.fail("a pill left an idle that keeps repeating")
        for window in Gtk.Window.list_toplevels():
            if isinstance(window, overlay.WaveformPill):
                window.destroy()


@pytest.fixture
def private_bus(tmp_path):
    """Start a session bus of the test's own and yield its address.

    Quoth's name can be owned there without reaching an instance on the real bus. With
    no service directories the bus starts no services, so none of the host's portals
    runs outside the desktop session and stalls the app's startup.
    """
    if shutil.which("dbus-daemon") is None:
        pytest.skip("needs dbus-daemon")
    config = tmp_path / "bus.conf"
    config.write_text(
        "<busconfig><type>session</type><listen>unix:tmpdir=/tmp</listen>"
        '<policy context="default"><allow send_destination="*"/>'
        '<allow receive_sender="*"/><allow own="*"/></policy></busconfig>'
    )
    daemon = subprocess.Popen(
        ["dbus-daemon", f"--config-file={config}", "--nofork", "--print-address"],
        stdout=subprocess.PIPE,
        text=True,
    )
    yield daemon.stdout.readline().strip()
    daemon.terminate()
    daemon.wait()
    daemon.stdout.close()


@pytest.fixture
def chooser(monkeypatch):
    """Stand in for `Gtk.FileDialog`: `answer(value)` finishes the last one opened."""
    from gi.repository import Gtk

    from tests.support import FinishedDialog

    opened = []

    class FileDialog:
        def __init__(self, **properties):
            self.properties = properties

        def set_default_filter(self, file_filter):
            self.default_filter = file_filter

        def open(self, parent, cancellable, callback):
            self.parent, self.cancellable, self.callback = parent, cancellable, callback
            opened.append(self)

    def answer(value):
        dialog = FinishedDialog(value)
        opened[-1].callback(dialog, None)
        return dialog

    monkeypatch.setattr(Gtk, "FileDialog", FileDialog)
    return SimpleNamespace(opened=opened, answer=answer)


@pytest.fixture(autouse=True)
def _isolate_xdg_dirs(tmp_path, monkeypatch):
    """Point every XDG base dir at a per-test temp tree.

    Tests that touch logging, the update check or model storage would otherwise share
    the session-wide tree and see each other's files. `paths.py` reads these vars at
    call time, so setting them here reaches its callers; `log.py` names its directory
    once at import, so it keeps the session-wide tree.
    """
    for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))


@pytest.fixture(autouse=True)
def detected_devices(monkeypatch):
    """The inference devices detection finds: the CPU alone, unless a test adds more.

    Detection probes the host's Vulkan GPU, Intel NPU and OpenVINO runtime, so no result
    may depend on the machine running the suite. The probes are pinned because modules
    import `available_devices` by name, out of reach of a patch on `hardware`.
    `test_hardware` tests the probes.
    """
    from quoth import hardware

    found = ["CPU"]
    monkeypatch.setattr(hardware, "_npu_present", lambda: "NPU" in found)
    monkeypatch.setattr(hardware, "_vulkan_gpu_present", lambda: "GPU" in found)
    monkeypatch.setattr(hardware, "openvino_core_lib_paths", lambda: [])
    return found


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset every key after each test: the memory backend is shared by the run."""
    yield
    from gi.repository import Gio

    settings = Gio.Settings.new(quoth.APP_ID)
    for key in settings.props.settings_schema.list_keys():
        settings.reset(key)


@pytest.fixture(autouse=True)
def _fail_on_callback_errors(monkeypatch):
    """Fail a test when a GLib or GTK callback it ran raised.

    PyGObject hands an exception from a callback (a draw function, a signal handler,
    an idle) to `sys.excepthook` and carries on, and `scheduler.guarded` logs one and
    carries on, so the test would still pass. A test that makes a guarded callback
    raise on purpose takes `callback_raises`.
    """
    errors = []
    monkeypatch.setattr(
        sys, "excepthook", lambda *exc: errors.append(traceback.format_exception(*exc))
    )
    _guarded_raises.clear()
    yield
    errors += [[f"{message}\n"] for message in _guarded_raises]
    if errors:
        pytest.fail(
            "A callback raised:\n" + "".join(map("".join, errors)), pytrace=False
        )


@pytest.fixture
def callback_raises():
    """What `scheduler.guarded` logged for each callback that raised, in order.

    For a test that makes one raise on purpose, which therefore does not fail it.
    """
    yield _guarded_raises
    # Autouse fixtures tear down last, so this empties the list before
    # `_fail_on_callback_errors` reads it.
    _guarded_raises.clear()


@pytest.fixture(autouse=True)
def _fail_on_glib_warnings():
    """Fail a test during which GLib, GTK or GStreamer logged a warning or critical."""
    _glib_warnings.clear()
    yield
    if _glib_warnings:
        pytest.fail("GLib logged:" + "".join(_glib_warnings), pytrace=False)


@pytest.fixture
def long_home(monkeypatch):
    """Make the home `/home/maximiliananderson`, long enough to be cut by a cap."""
    from quoth import paths

    monkeypatch.setenv("HOME", "/home/maximiliananderson")
    monkeypatch.setattr(paths, "_HOME", paths._home_pattern())
    return Path("/home/maximiliananderson")


@pytest.fixture
def malformed_proxy(monkeypatch):
    """Set `http://bob:s3cret` as the proxy: urllib takes the password for a port.

    Its error then names the password, and it fails before any connection is made.
    """
    import urllib.request

    for name in ("https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://bob:s3cret")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    # urlopen keeps the opener, and the proxies it read, from its first call.
    monkeypatch.setattr(urllib.request, "_opener", None)


@pytest.fixture
def caplog(caplog):
    """Loguru's records in pytest's `caplog`: assert on what was logged, formatted.

    A patched `logger` never formats its message, so a broken placeholder that raises
    in the app passes the test. `caplog.set_level("ERROR")` narrows the capture. A
    record's message is the text alone; a logged exception is its `exc_info`.
    """
    sink = logger.add(
        caplog.handler,
        format=lambda _record: "{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield caplog
    logger.remove(sink)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for model storage testing."""
    data_dir = tmp_path / "share" / "quoth" / "models"
    data_dir.mkdir(parents=True)
    return data_dir
