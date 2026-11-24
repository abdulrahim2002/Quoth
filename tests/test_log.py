# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.log`: the process-level safety nets and the persisted log.

Uncaught exceptions from GTK callbacks, threads and `__del__` land in the log, and a
native crash leaves a Python stack behind.
"""

import faulthandler
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time

import pytest
from loguru import logger

import quoth.log as log


@pytest.fixture
def hooks(monkeypatch):
    """Install the crash hooks and yield the ERROR+ records logged meanwhile.

    Restores the hooks it replaces afterwards. faulthandler stays on pytest's stream,
    which no test can name to restore; a child process checks the crash file.
    """
    records = []
    sink = logger.add(lambda msg: records.append(msg.record), level="ERROR")
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    monkeypatch.setattr(faulthandler, "enable", lambda **_: None)
    log.install_crash_hooks()
    yield records
    logger.remove(sink)
    log._crash_file.close()


def test_excepthook_logs_uncaught_and_passes_keyboardinterrupt(hooks, monkeypatch):
    passed = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: passed.append(a[0]))
    try:
        raise RuntimeError("probe")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert [r["level"].name for r in hooks] == ["CRITICAL"]
    assert hooks[0]["exception"].type is RuntimeError
    assert passed == [KeyboardInterrupt]


def test_threading_hook_logs_with_thread_name(hooks):
    thread = threading.Thread(name="worker-7")
    args = threading.ExceptHookArgs([RuntimeError, RuntimeError("probe"), None, thread])

    threading.excepthook(args)

    assert len(hooks) == 1
    assert "worker-7" in hooks[0]["message"]
    assert hooks[0]["exception"].type is RuntimeError


def test_unraisable_hook_never_reprs_object(hooks):
    reprs = []

    class Buffer:
        def __repr__(self):
            reprs.append(self)
            return "private transcript"

    class Unraisable:
        exc_type = RuntimeError
        exc_value = None
        exc_traceback = None
        err_msg = None
        object = Buffer()

    sys.unraisablehook(Unraisable())

    assert reprs == []
    assert len(hooks) == 1
    assert hooks[0]["message"] == "Exception ignored (Buffer)"


def test_unraisable_hook_logs_the_exception(hooks):
    class Finalizer:
        def __del__(self):
            raise RuntimeError("probe")

    Finalizer()

    assert len(hooks) == 1
    assert hooks[0]["exception"].type is RuntimeError


def test_faulthandler_writes_to_crash_log(tmp_path):
    script = """
import os
import resource
import signal
import quoth.log as log

def crash():
    os.kill(os.getpid(), signal.SIGSEGV)

# No core file for the deliberate crash.
_soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
log.install_crash_hooks()
crash()
"""
    result = _run(script, tmp_path)

    assert result.returncode == -signal.SIGSEGV, result.stderr
    stack = (tmp_path / "quoth" / "crash.log").read_text()
    assert stack.startswith("Fatal Python error: Segmentation fault")
    assert "in crash" in stack


# The stack is from the run that just crashed, or has waited in crash.prev.log since an
# earlier launch for consent to send it.
@pytest.mark.parametrize("stack_in", ["crash.log", "crash.prev.log"])
def test_previous_crash_is_kept_for_the_next_launch(monkeypatch, tmp_path, stack_in):
    """A native crash's stack survives the restart that truncates crash.log."""
    crash_log, previous = tmp_path / "crash.log", tmp_path / "crash.prev.log"
    crash_log.touch()
    (tmp_path / stack_in).write_text("Fatal Python error: Segmentation fault\n")
    monkeypatch.setattr(log, "crash_log", crash_log)
    monkeypatch.setattr(log, "previous_crash_log", previous)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    monkeypatch.setattr(faulthandler, "enable", lambda **_: None)
    log.install_crash_hooks()
    log._crash_file.close()

    assert previous.read_text().startswith("Fatal Python error")
    assert crash_log.read_text() == ""


def _run(script: str, state_home) -> subprocess.CompletedProcess:
    """Run `script` in a fresh process, as the app runs, against `state_home`."""
    env = os.environ.copy()
    env["XDG_STATE_HOME"] = str(state_home)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )


def test_persisted_tracebacks_keep_types_and_frames_but_not_values(tmp_path):
    script = """
import sys
from loguru import logger
import quoth.log as log
log.install_file_log()

# Doubled because a traceback can quote its source lines: only a leaked value can then
# read "private-2000" and the like.
def private(n):
    return "private-" + str(n * 2)

def failing_step():
    try:
        raise KeyError(private(1000))
    except KeyError as e:
        raise ValueError(private(2000)) from e

try:
    failing_step()
except ValueError:
    logger.exception("Step failed for {}", "the model")
logger.opt(exception=True).error("Logged outside a handler")
log.install_crash_hooks()
try:
    try:
        raise OSError(private(4000))
    except OSError:
        raise RuntimeError(private(3000))
except RuntimeError:
    sys.excepthook(*sys.exc_info())
logger.complete()
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    text = (tmp_path / "quoth" / "quoth.log").read_text()
    assert "Step failed for the model" in text
    assert "failing_step" in text
    assert "KeyError\n" in text and "ValueError\n" in text and "RuntimeError\n" in text
    assert "direct cause of the following exception" in text
    assert "OSError\n" in text
    assert "During handling of the above exception" in text
    assert "Logged outside a handler" in text
    # Captured stderr is not a terminal: it may be persisted, like the journal.
    assert "Step failed for the model" in result.stderr
    for value in ("private-2000", "private-4000", "private-6000", "private-8000"):
        assert value not in text and value not in result.stderr


def test_a_failing_file_sink_reports_no_exception_values(tmp_path):
    """A sink's failed write (full disk) prints the whole record to stderr."""
    script = """
import resource
import signal
from loguru import logger
import quoth.log as log
log.install_file_log()

# Writes past 200 bytes fail with EFBIG, as on a full disk; stderr is a pipe, unaffected.
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (200, resource.RLIM_INFINITY))
try:
    try:
        raise KeyError("private-" + str(1000 * 2))
    except KeyError as e:
        raise ValueError("private-" + str(2000 * 2)) from e
except ValueError:
    logger.exception("Step failed")
logger.complete()
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Logging error in Loguru Handler" in result.stderr
    assert "ValueError" in result.stderr
    for value in ("private-2000", "private-4000"):
        assert value not in result.stderr


@pytest.mark.parametrize("primary", [False, True])
def test_only_the_instance_that_installs_it_writes_the_log_file(tmp_path, primary):
    # Every launch imports the log configuration; a second one, such as a desktop
    # action while Quoth runs, must not write into the running instance's log.
    install = "log.install_file_log()" if primary else ""
    script = f"""
from loguru import logger
import quoth.log as log
{install}
logger.warning("launched")
logger.complete()
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "launched" in result.stderr
    file = tmp_path / "quoth" / "quoth.log"
    assert file.exists() is primary
    if primary:
        assert "launched" in file.read_text()


def test_lines_logged_before_the_file_sink_reach_it_first(tmp_path):
    # Settings and the overlay check log before the primary instance's startup, and
    # a report needs those lines in the file, in order and in the file's form. A path
    # that is not UTF-8, such as a library's, must not stop the file from opening.
    script = """
import os
from loguru import logger
import quoth.log as log
logger.debug("early detail")
try:
    raise KeyError("private-" + str(1000 * 2))
except KeyError:
    logger.exception("Early failure")
logger.warning("Library {} failed to load", os.fsdecode(b"/opt/lib\\xff.so"))
log.install_file_log()
logger.info("later")
logger.warning("Library {} failed to load", os.fsdecode(b"/opt/lib\\xfe.so"))
logger.complete()
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    text = (tmp_path / "quoth" / "quoth.log").read_text()
    assert (
        text.index("early detail") < text.index("Early failure") < text.index("later")
    )
    first = text.splitlines()[0]
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \| DEBUG   \| MainThread \| "
        r"__main__:<module>:5 \| early detail",
        first,
    ), first
    assert "KeyError\n" in text
    assert "private-2000" not in text
    assert "Library /opt/lib\\udcff.so failed to load" in text
    assert "Library /opt/lib\\udcfe.so failed to load" in text


def test_rotation_keeps_a_few_recent_archives(tmp_path):
    """A logging loop cannot fill the disk with archives inside the age window."""
    state = tmp_path / "quoth"
    state.mkdir()
    now = time.time()
    for hour in range(8):
        archive = state / f"quoth.2026-01-01_0{hour}-00-00_000000.log.gz"
        archive.write_bytes(b"")
        os.utime(archive, (now - hour * 3600, now - hour * 3600))
    stale = state / "quoth.2025-12-01_00-00-00_000000.log.gz"
    stale.write_bytes(b"")
    os.utime(stale, (now - 4 * 86400, now - 4 * 86400))

    # Past the 5 MB rotation threshold.
    script = """
from loguru import logger
import quoth.log as log
log.install_file_log()
for _ in range(600):
    logger.debug("x" * 10000)
logger.complete()
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    archives = sorted(p.name for p in state.glob("quoth.*.log.gz"))
    assert len(archives) == 5
    assert stale.name not in archives
    # The archive that rotation just wrote is among the five kept.
    assert any(not name.startswith("quoth.2026-01-01") for name in archives)


def test_diagnostics_are_private_to_the_user(tmp_path):
    existing = tmp_path / "quoth"
    existing.mkdir(mode=0o755)

    result = _run("import quoth.log as log\nlog.install_file_log()", tmp_path)

    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(existing.stat().st_mode) == 0o700


def test_the_console_level_is_read_in_any_case(monkeypatch, tmp_path):
    monkeypatch.setenv("QUOTH_LOG_LEVEL", "debug")

    result = _run(
        "import quoth.log\nfrom loguru import logger\nlogger.debug('detail')", tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert "detail" in result.stderr


def test_an_unknown_console_level_falls_back_to_info(monkeypatch, tmp_path):
    monkeypatch.setenv("QUOTH_LOG_LEVEL", "verbose")

    result = _run(
        "import quoth.log as log\nfrom loguru import logger\nlog.install_file_log()\n"
        "logger.debug('detail')\nlogger.info('running')\nlogger.complete()",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    warning = "QUOTH_LOG_LEVEL=VERBOSE is not usable here; logging at INFO"
    assert warning in result.stderr
    # The log users attach to a report says why their console was quieter.
    assert warning in (tmp_path / "quoth" / "quoth.log").read_text()
    assert "running" in result.stderr
    assert "detail" not in result.stderr


def test_trace_reaches_only_a_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("QUOTH_LOG_LEVEL", "TRACE")

    result = _run(
        "import quoth.log as log\nfrom loguru import logger\nlog.install_file_log()\n"
        "logger.trace('dictated words')\nlogger.debug('detail')\nlogger.complete()",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "dictated words" not in result.stderr
    assert "detail" in result.stderr
    persisted = (tmp_path / "quoth" / "quoth.log").read_text()
    assert "dictated words" not in persisted
    assert "detail" in persisted


def test_lines_show_the_home_directory_as_a_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "alice"))
    # A source checkout under the home directory puts it in every traceback frame.
    (tmp_path / "alice").mkdir()
    (tmp_path / "alice" / "checkout.py").write_text("def fail():\n    raise OSError\n")
    script = """
import sys
from pathlib import Path
from loguru import logger
import quoth.log as log
log.install_file_log()
logger.info("Saved {}", Path.home() / "notes.toml")
sys.path.insert(0, str(Path.home()))
import checkout
try:
    checkout.fail()
except OSError:
    logger.exception("Failed")
logger.complete()
print((log.log_dir / "quoth.log").read_text())
"""
    result = _run(script, tmp_path / "state")

    assert result.returncode == 0, result.stderr
    for output in (result.stdout, result.stderr):
        assert "Saved ~/notes.toml" in output
        assert 'File "~/checkout.py", line 2, in fail' in output
        assert "alice" not in output


def test_lines_mask_a_bluetooth_address_but_not_a_longer_hex_string(tmp_path):
    fingerprint = ":".join(f"{n:02X}" for n in range(16))
    script = f"""
from loguru import logger
import quoth.log as log
log.install_file_log()
logger.warning("Mic: bluez_input.AA_BB_CC_DD_EE_FF.0 and bluez_input.a1:b2:c3:d4:e5:f6")
logger.warning("Certificate {fingerprint}")
logger.complete()
print((log.log_dir / "quoth.log").read_text())
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    for output in (result.stdout, result.stderr):
        assert (
            "Mic: bluez_input.XX_XX_XX_XX_XX_XX.0 and bluez_input.XX_XX_XX_XX_XX_XX"
            in output
        )
        assert f"Certificate {fingerprint}" in output


def test_only_a_whole_home_path_becomes_a_tilde(monkeypatch, tmp_path):
    # A home reached through a symlink shows its real path wherever code resolves one.
    real = tmp_path / "disk" / "alice"
    real.mkdir(parents=True)
    (tmp_path / "alice").symlink_to(real)
    monkeypatch.setenv("HOME", str(tmp_path / "alice"))
    script = """
from pathlib import Path
from loguru import logger
import quoth.log as log
log.install_file_log()
home = str(Path.home())
logger.info("Paths: {} {} {}", Path.home().resolve() / "a", home + "2/b", "/var" + home)
logger.complete()
print((log.log_dir / "quoth.log").read_text())
"""
    result = _run(script, tmp_path / "state")

    assert result.returncode == 0, result.stderr
    home = tmp_path / "alice"
    assert f"Paths: ~/a {home}2/b /var{home}\n" in result.stdout


def test_an_exception_of_a_local_class_is_logged_without_its_value(tmp_path):
    # Such a class cannot be pickled by reference for the file's writer thread.
    script = """
from loguru import logger
import quoth.log as log
log.install_file_log()

def local_class():
    class LocalError(Exception):
        pass
    return LocalError

LocalError = local_class()
try:
    raise LocalError("private-canary")
except LocalError:
    logger.exception("Local failure")
logger.complete()
print((log.log_dir / "quoth.log").read_text())
"""
    result = _run(script, tmp_path / "state")

    assert result.returncode == 0, result.stderr
    assert "Local failure" in result.stdout
    assert "LocalError" in result.stdout
    assert "private-canary" not in result.stdout + result.stderr


def test_an_argument_cannot_forge_a_line(tmp_path):
    # A USB device, a microphone or a file chooses its own name.
    script = r"""
from loguru import logger
import quoth.log as log
log.install_file_log()
logger.info("Device {}", "Mic\n2026-01-01 | ERROR | forged\r\x1b[2K\u2028end")
try:
    raise RuntimeError
except RuntimeError:
    logger.exception("Failed")
logger.complete()
print((log.log_dir / "quoth.log").read_text())
"""
    result = _run(script, tmp_path)

    assert result.returncode == 0, result.stderr
    for output in (result.stdout, result.stderr):
        assert "Device Mic\\n2026-01-01 | ERROR | forged\\r\\x1b[2K\\u2028end" in output
        assert not any(line.startswith("2026-01-01") for line in output.splitlines())
        assert "Traceback (most recent call last):\n" in output


def test_unusable_state_storage_keeps_console_logging(tmp_path):
    blocked = tmp_path / "state"
    blocked.write_text("not a directory")
    script = """
from loguru import logger
import quoth.log as log
log.install_file_log()
log.install_crash_hooks()
logger.info("still running")
"""
    result = _run(script, blocked)

    assert result.returncode == 0, result.stderr
    assert "File logging unavailable" in result.stderr
    assert "Native crash stacks unavailable" in result.stderr
    assert "still running" in result.stderr
