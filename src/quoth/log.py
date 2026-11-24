# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Quoth's loguru configuration: a human-readable console sink and a rotated file sink.

`quoth.app` imports this first, before anything that logs. The console shows INFO and
up by default (QUOTH_LOG_LEVEL=DEBUG for more). The file, which only the primary
instance writes, keeps every level from DEBUG up for post-mortem debugging, rotated by
size and pruned by count and age.

Privacy: dictated text is logged only at TRACE, which neither sink captures by default,
so transcribed content never touches disk; QUOTH_LOG_LEVEL=TRACE shows it live, but
only in a terminal. Both sinks run with diagnose=False, so a traceback never renders
local variable values (which may hold transcript text). The persisted log also omits
exception values, keeping each exception's type and frames, and so does the console
unless it is a terminal. Every message, and every traceback but a terminal's, shows the
home directory, which names the user, as `~`. Every message also masks a Bluetooth
address, as in an audio node's name, since it identifies the hardware.
"""

import faulthandler
import os
import pickle
import re
import sys
import threading
import time
import traceback
from collections import deque, namedtuple

from loguru import logger

from quoth import paths

logger.remove()


class _Exception(namedtuple("_Exception", ("type", "value", "traceback"))):
    """A record's exception whose repr and queued copy leave out the value.

    A failing sink prints the whole record to stderr, which off a terminal is the
    persistent journal; sinks in this process still read the value.
    """

    def __repr__(self) -> str:
        return f"(type={self.type!r})"

    def __reduce__(self):
        # A type pickle cannot name (a class made in a function) would fail the queue
        # put, and loguru then prints the exception being logged, value and all.
        try:
            pickle.dumps(self.type)
        except Exception:
            return (_Exception, (None, None, None))
        return (_Exception, (self.type, None, None))


# Control characters and line breaks in a message, as `repr` writes them, so a logged
# name cannot forge a line or drive a terminal. Tabs stay.
_ESCAPES = {
    c: repr(chr(c))[1:-1]
    for c in (*range(0x20), *range(0x7F, 0xA0), 0x2028, 0x2029)
    if c != 0x09
}


# Six hex pairs, as `AA_BB_CC_DD_EE_FF` or with colons, but no run of a longer
# colon-separated hex string such as a certificate fingerprint.
_BLUETOOTH_ADDRESS = re.compile(
    r"(?<![0-9A-Fa-f])(?<![0-9A-Fa-f]:)"
    r"(?:[0-9A-Fa-f]{2}[_:]){5}[0-9A-Fa-f]{2}"
    r"(?!:?[0-9A-Fa-f])"
)


def _patch(record) -> None:
    message = _BLUETOOTH_ADDRESS.sub("XX_XX_XX_XX_XX_XX", record["message"])
    record["message"] = paths.tilde(message).translate(_ESCAPES)
    if record["exception"]:
        record["exception"] = _Exception(*record["exception"])


logger.configure(patcher=_patch)

_CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"
_CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n\n"


def _traceback(kind, value, tb, seen=None) -> str:
    """Format an exception chain in Python's traceback layout, without the values.

    Each exception keeps its type and frames but not its value: an exception's message
    can carry whatever the failing code was holding.
    """
    seen = seen or set()
    seen.add(id(value))
    name = kind.__qualname__
    if kind.__module__ not in ("builtins", "__main__"):
        name = f"{kind.__module__}.{name}"
    text = name + "\n"
    if tb:
        frames = "".join(traceback.format_tb(tb))
        text = "Traceback (most recent call last):\n" + frames + text
    chained, header = None, ""
    if value is not None and value.__cause__ is not None:
        chained, header = value.__cause__, _CAUSE
    elif value is not None and not value.__suppress_context__:
        chained, header = value.__context__, _CONTEXT
    if chained is None or id(chained) in seen:
        return text
    return (
        _traceback(type(chained), chained, chained.__traceback__, seen) + header + text
    )


def _without_values(template: str):
    """Wrap `template` as a loguru format that renders exceptions via `_traceback`."""

    def render(record) -> str:
        error = record["exception"]
        # loguru records an empty exception for exception=True outside an except.
        record["extra"]["traceback"] = (
            paths.tilde(_traceback(*error)) if error and error.type else ""
        )
        return template + "\n{extra[traceback]}"

    return render


# Console handler: human-readable, INFO+ by default. A terminal keeps exception values
# for live debugging; anywhere else (the desktop's journal) gets the file's form.
requested = os.environ.get("QUOTH_LOG_LEVEL", "INFO").strip().upper()
try:
    logger.level(requested)
    level = requested
except ValueError:
    level = "INFO"
# TRACE carries dictated text, which must never reach the journal.
if level == "TRACE" and not sys.stderr.isatty():
    level = "DEBUG"
_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>"
)
logger.add(
    sys.stderr,
    level=level,
    format=_CONSOLE_FORMAT if sys.stderr.isatty() else _without_values(_CONSOLE_FORMAT),
    colorize=None,
    diagnose=False,
    backtrace=False,
)

# The file sink's directory, also where crash.log lives.
log_dir = paths.state_dir()


def _retain(archives: list[str]) -> None:
    """Keep the newest five rotated archives, none older than three days."""
    cutoff = time.time() - 3 * 86400
    newest = sorted(archives, key=os.path.getmtime, reverse=True)
    for rank, path in enumerate(newest):
        if rank >= 5 or os.path.getmtime(path) < cutoff:
            os.remove(path)


_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {thread.name} | "
    "{name}:{function}:{line} | {message}"
)
# Lines logged before the file sink exists, held as the file would write them, for
# `install_file_log` to write first; a process that never installs it drops them. The
# bound keeps a long-lived importer, such as the test suite, from growing it forever.
_held: deque[str] = deque(maxlen=1000)
_hold = logger.add(
    lambda message: _held.append(str(message)),
    level="DEBUG",
    format=_without_values(_FILE_FORMAT),
    diagnose=False,
    backtrace=False,
)

if level != requested:
    logger.warning(
        "QUOTH_LOG_LEVEL={} is not usable here; logging at {}", requested, level
    )


def install_file_log() -> None:
    """Add the rotated DEBUG file sink, for post-mortem analysis.

    Call once from the primary instance's startup, never at import: a second instance
    (`quoth --settings` while running) would write into the live log and share its
    rotation. enqueue=True hands writes to a background thread, so the whisper, evdev
    and GStreamer threads never wait on a rotation's gz compression, and lines cannot
    interleave. The directory is the user's alone. What was logged since import is
    written first.
    """
    logger.remove(_hold)
    try:
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_dir.chmod(0o700)
        with open(
            log_dir / "quoth.log", "a", encoding="utf-8", errors="backslashreplace"
        ) as file:
            file.writelines(_held)
        logger.add(
            log_dir / "quoth.log",
            level="DEBUG",
            format=_without_values(_FILE_FORMAT),
            rotation="5 MB",
            retention=_retain,
            compression="gz",
            errors="backslashreplace",
            enqueue=True,
            diagnose=False,
            backtrace=False,
        )
    except OSError as e:
        logger.warning("File logging unavailable ({}); logging to the console only", e)
    _held.clear()


# A native crash leaves its Python stack in crash.log. Each primary instance moves a
# non-empty one aside before starting a fresh file, so the stack survives the
# restart for crash_reports (or the user) to pick up.
crash_log = log_dir / "crash.log"
previous_crash_log = log_dir / "crash.prev.log"
# faulthandler keeps the file descriptor, not the object; the handle lives here so
# the descriptor stays open for the life of the process.
_crash_file = None


def _log_uncaught(exc_type, exc_value, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, tb)
        return
    logger.opt(exception=(exc_type, exc_value, tb)).critical("Uncaught exception")


def _log_thread_uncaught(args):
    if issubclass(args.exc_type, SystemExit):
        return
    logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).critical(
        "Uncaught exception in thread {}", args.thread.name if args.thread else "?"
    )


def _log_unraisable(unraisable):
    # Never repr the object: it may be a widget or buffer holding transcript text.
    exc = (
        (unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)
        if unraisable.exc_value is not None
        else None
    )
    logger.opt(exception=exc).error(
        "{} ({})",
        unraisable.err_msg or "Exception ignored",
        type(unraisable.object).__name__,
    )


def install_crash_hooks() -> None:
    """Log what would otherwise hit stderr, and keep Python stacks for native crashes.

    Uncaught errors in GTK callbacks, threads and `__del__` go to the log; a native
    crash leaves every thread's stack in crash.log. Call once from the primary
    instance's startup, never at import: a second instance (`quoth --settings` while
    running) would truncate the live crash file.
    """
    global _crash_file
    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_uncaught
    sys.unraisablehook = _log_unraisable
    try:
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if crash_log.exists() and crash_log.stat().st_size:
            crash_log.replace(previous_crash_log)
        _crash_file = open(crash_log, "w")
    except OSError as e:
        logger.warning("Native crash stacks unavailable ({})", e)
        return
    faulthandler.enable(file=_crash_file, all_threads=True)
