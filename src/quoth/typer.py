# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Text output by clipboard paste: set the selections, then send one Shift+Insert.

Wayland has no text-injection IPC that works across compositors, and one paste chord
drops no characters where a key event per character does under load. Setting both the
clipboard (read by GTK/Qt widgets) and the primary selection (read by most terminals)
lands it everywhere.

`_ChordSender` is how the chord is sent: `wtype` (wlroots virtual keyboard, no setup) or
`ydotool` (kernel uinput, needed on GNOME and KDE, which gate the virtual-keyboard
protocol). `_ExecutionContext` is where commands run: directly, or from a Flatpak as
the bundled tool on the host through `flatpak-spawn --host`.
"""

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from quoth import environment

# On Shift+Insert GTK/Qt widgets paste the CLIPBOARD, most terminals the PRIMARY.
_SELECTIONS = ("clipboard", "primary")

# The first wl-clipboard with `wl-copy --sensitive`, which asks clipboard-history
# managers to skip a write. Older wl-copy fails on the unknown flag, so it is omitted.
_MIN_SENSITIVE_VERSION = (2, 3, 0)

# Grace period for the focused app to finish its (async) read of the paste before
# the selections are restored. Blocks only the injection worker thread.
RESTORE_DELAY_MS = 250

# Per-call ceiling for the clipboard and chord spawns, so a stalled `wl-copy`,
# `wl-paste` or `flatpak-spawn` becomes a logged failure instead of a hang.
_SUBPROCESS_TIMEOUT_S = 5


# --- Execution context: where a command runs (direct vs flatpak host) ---

_SANDBOX_BIN = "/app/bin"  # where our bundled tools live inside the sandbox
# A native package's bundled tools, beside its native libs. The .deb ships wl-clipboard
# here (Debian and Ubuntu ship one without --sensitive and ext-data-control) and
# ydotool (Debian 13 ships it only in backports).
_PACKAGE_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "bin")


def _package_tool(tool: str) -> str | None:
    """Return the package's bundled copy of `tool`, or None to use the one on PATH."""
    path = os.path.join(_PACKAGE_BIN, tool)
    return path if os.path.isfile(path) else None


@dataclass(frozen=True)
class _ExecutionContext:
    """Where injection subprocesses run: directly, or on the Flatpak host.

    Inside a Flatpak, privileged Wayland protocols (virtual keyboard, wl-clipboard) are
    denied to the sandboxed client, so the bundled tools run on the host through
    `flatpak-spawn --host`.
    """

    sandboxed: bool
    host_bin: str | None

    @classmethod
    def detect(cls) -> "_ExecutionContext":
        if not environment.in_flatpak():
            logger.debug("Text injection: direct execution (no sandbox)")
            return cls(sandboxed=False, host_bin=None)
        app_path = environment.flatpak_app_path()
        if not app_path:
            logger.warning(
                "Sandboxed but /.flatpak-info has no app-path; using direct execution"
            )
            return cls(sandboxed=False, host_bin=None)
        host_bin = f"{app_path}/bin"
        logger.debug("Text injection: flatpak sandbox, host tool bin {}", host_bin)
        return cls(sandboxed=True, host_bin=host_bin)

    def wrap(
        self, tool: str, args: list[str], env: dict[str, str] | None = None
    ) -> list[str]:
        """Return the argv that runs `tool` with `args` in this context.

        Inside a sandbox that is the bundled tool on the host, otherwise the package's
        bundled copy where it ships one, else the tool on PATH. `env` sets variables for
        the command; in a sandbox they reach the host process through `flatpak-spawn
        --env`, since the sandbox's own environment does not, so pair this with
        `subprocess_env`.
        """
        if self.sandboxed and self.host_bin:
            env_flags = [f"--env={k}={v}" for k, v in (env or {}).items()]
            return environment.host_argv([*env_flags, f"{self.host_bin}/{tool}", *args])
        return [_package_tool(tool) or tool, *args]

    def subprocess_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Return the `env=` to pass `subprocess.run` for a `wrap(..., env=env)` argv.

        Direct execution needs `env` merged into the inherited environment. A sandbox
        carries it through `--env` flags instead, so this returns None there, as it
        does when there is nothing to add (None inherits).
        """
        if not env or self.sandboxed:
            return None
        return {**os.environ, **env}

    def available(self, tool: str) -> bool:
        """Whether `tool` can be run in this context."""
        if self.sandboxed:
            return os.path.exists(f"{_SANDBOX_BIN}/{tool}")
        return _package_tool(tool) is not None or shutil.which(tool) is not None


# --- Chord senders: how Shift+Insert is delivered ---


@dataclass(frozen=True)
class _ChordSender:
    """A tool and argv that emit one Shift+Insert paste chord.

    `requires_daemon` names a background service the tool needs running, which Quoth
    starts at launch or on first use unless one already runs.
    """

    tool: str
    args: tuple[str, ...]
    requires_daemon: str | None = None


# wtype: -M/-m press/release a modifier, -P/-p press/release a key (upstream form).
_WTYPE_CHORD = _ChordSender(
    "wtype", ("-M", "shift", "-P", "Insert", "-p", "Insert", "-m", "shift")
)
# ydotool 1.x: layout-independent uinput keycodes (Shift 42, Insert 110). It needs its
# daemon, ydotoold.
_YDOTOOL_CHORD = _ChordSender(
    "ydotool", ("key", "42:1", "110:1", "110:0", "42:0"), requires_daemon="ydotoold"
)
# ydotool 0.1.8 accepts named chords and can inject directly without ydotoold.
_YDOTOOL_LEGACY_CHORD = _ChordSender("ydotool", ("key", "Shift+Insert"))

_YDOTOOL_HELP_HEADER = ("Usage: ydotool <cmd> <args>", "Available commands:")
_YDOTOOL_LEGACY_COMMANDS = {"click", "key", "mousemove", "recorder", "type"}
_YDOTOOL_MODERN_COMMANDS = {"click", "debug", "key", "mousemove", "type"}

# The daemon Quoth starts uses ydotool's own per-user default socket, so a user's
# ydotoold service is shared rather than doubled. $YDOTOOL_SOCKET overrides it.
_YDOTOOL_SOCKET_NAME = ".ydotool_socket"
# ydotoold answers on its socket before it creates its virtual device, then gives the
# compositor a second to add the device before it reads a chord.
_YDOTOOLD_SETUP_S = 1.1  # that second, plus a margin
_YDOTOOLD_READY_TIMEOUT_S = 2.0
_YDOTOOLD_POLL_INTERVAL_S = 0.1

# Every chord-sender; `init_typer` orders the available ones for the execution context.
_CHORD_SENDERS = (_WTYPE_CHORD, _YDOTOOL_CHORD)


# --- Clipboard injector ---


class InjectionResult(Enum):
    """How a dictation reached the focused app, or why it did not."""

    SUCCESS = "success"
    # Nothing was pasted; the transcript is on the clipboard for a manual paste.
    CLIPBOARD_ONLY = "clipboard-only"  # every chord-sender failed; the log says why
    NO_CHORD_SENDER = "no-chord-sender"
    UINPUT_SETUP_REQUIRED = "uinput-setup-required"
    # Not on the clipboard either.
    FAILED = "failed"
    NO_WAYLAND = "no-wayland"
    NO_WL_CLIPBOARD = "no-wl-clipboard"


class _ClipboardBackend:
    """The clipboard-paste injector: set both selections, then send one Shift+Insert.

    Chord-senders are tried in order on every injection: exit zero only confirms the
    tool accepted the chord, so one that exited 0 once must not displace a preferred
    one. After a paste the user's selections are given back (see `_restore`); a failed
    paste leaves the transcript on the clipboard for a manual paste.

    Beside `inject`, `copy` only sets the clipboard and returns success; `prewarm`
    starts any helper process the first paste would wait for, and `shutdown` stops one
    it started.
    """

    def __init__(
        self,
        ctx: _ExecutionContext,
        chords: list[_ChordSender],
        *,
        supports_sensitive: bool = False,
        unusable: InjectionResult | None = None,
    ) -> None:
        self._ctx = ctx
        self._chords = chords
        # Why the clipboard cannot be reached at all, or None when it can.
        self._unusable = unusable
        # Whether wl-copy understands --sensitive.
        self._supports_sensitive = supports_sensitive
        # Set once the ydotoold socket answers; until then, and after a chord through it
        # exits with an error, each chord that needs the daemon looks again.
        self._daemon_socket: str | None = None
        # The daemon this session launched: a live one is not duplicated, and shutdown
        # stops it. Launching and stopping share the lock, so a launch that a worker
        # began cannot outlive shutdown.
        self._daemon: subprocess.Popen | None = None
        self._daemon_lock = threading.Lock()
        self._closed = False
        # Host /dev/uinput access once a check proves it either way (Flatpak only).
        self._host_uinput: bool | None = None
        # In a Flatpak, the chords are ordered by that access once it is proven.
        self._order_pending = ctx.sandboxed
        self._primary_warned = False
        self._no_chord_warned = False
        # Tools whose failure has warned once; a tool that keeps failing logs DEBUG.
        self._failed_tools: set[str] = set()

    def copy(self, text: str) -> bool:
        """Put `text` on the clipboard alone, tagged sensitive; return success."""
        return self._write("clipboard", text.encode(), sensitive=True)

    def inject(self, text: str) -> InjectionResult:
        """Paste `text` into the focused app and report how it went."""
        logger.debug(
            "Injecting {} chars via clipboard paste (ctx={})",
            len(text),
            "host" if self._ctx.sandboxed else "direct",
        )
        logger.trace("Injection text: {!r}", text)  # TRACE-only: never hits disk
        if self._unusable:
            return self._unusable
        ours = text.encode()
        # The user's selections, saved for the restore, may hold passwords or keys: like
        # the transcript, their contents must never be logged.
        previous = {sel: self._read(sel) for sel in _SELECTIONS}
        # Tag the transcript sensitive (where wl-copy supports it) so clipboard-history
        # managers do not keep it during the brief window before the restore.
        if not self._write("clipboard", ours, sensitive=True):
            return InjectionResult.FAILED
        written = ["clipboard"]
        if self._write("primary", ours, sensitive=True):
            written.append("primary")
        elif not self._primary_warned:
            # No primary selection here, or a failed write. Paste anyway: only a
            # terminal that pastes PRIMARY misses out.
            logger.warning("Primary selection unavailable; pasting from the clipboard")
            self._primary_warned = True

        tool = self._send_chord()

        if tool:
            logger.debug("Paste sent via {} (selections={})", tool, written)
            # Give the focused app time to finish reading the paste before the restore.
            time.sleep(RESTORE_DELAY_MS / 1000)
            self._restore(written, previous, ours)
            return InjectionResult.SUCCESS
        else:
            # Nothing was pasted: leave the transcript on the clipboard for a manual
            # paste, but give primary back.
            self._restore([sel for sel in written if sel == "primary"], previous, ours)
            if self._read("clipboard") == ours:
                if not self._chords:
                    return InjectionResult.NO_CHORD_SENDER
                if self._host_uinput_writable() is False:
                    return InjectionResult.UINPUT_SETUP_REQUIRED
                return InjectionResult.CLIPBOARD_ONLY
            return InjectionResult.FAILED

    def prewarm(self) -> None:
        """Start the first chord-sender's daemon ahead of the first paste."""
        self._order_chords()
        first = self._chords[0] if self._chords else None
        if first and first.requires_daemon:
            self._ensure_daemon(first.requires_daemon)

    def shutdown(self) -> None:
        """Stop the ydotoold this session started; an adopted one is not ours to stop.

        Left running in a Flatpak, its flatpak-spawn client would keep the sandbox, and
        the lock on the app's deployment, alive.
        """
        with self._daemon_lock:
            self._closed = True
            if self._daemon is not None:
                logger.debug("Stopping ydotoold")
                self._daemon.terminate()  # a no-op once it has exited

    def _host_uinput_writable(self) -> bool | None:
        """Whether the Flatpak host proves /dev/uinput writable; None when unknown.

        None also outside a Flatpak. The fixed host script prints its answer apart from
        flatpak-spawn's exit status, so a failed host command stays unknown instead of
        triggering the setup. A proven answer holds for the session: the setup asks
        the user to log in again.
        """
        if self._host_uinput is not None or not self._ctx.sandboxed:
            return self._host_uinput
        script = (
            "if test -w /dev/uinput; then printf writable; else printf unwritable; fi"
        )
        try:
            result = subprocess.run(
                environment.host_argv(["sh", "-c", script]),
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Could not check host /dev/uinput access: {}", e)
            return None
        if result.returncode != 0:
            logger.warning(
                "Host /dev/uinput check exited {}: {}",
                result.returncode,
                result.stderr.strip()[:200],
            )
            return None
        status = result.stdout.strip()
        if status in ("writable", "unwritable"):
            self._host_uinput = status == "writable"
            return self._host_uinput
        logger.warning("Host /dev/uinput check returned an unknown result")
        return None

    def _restore(
        self, written: list[str], previous: dict[str, bytes | None], ours: bytes
    ) -> None:
        """Give back the selections we changed, but only those still holding our text.

        One that no longer does was replaced meanwhile, and writing over it would lose a
        new copy or revive a password meant to expire. The restore is tagged sensitive
        so a history manager does not record the user's copy twice.
        """
        for selection in written:
            saved = previous[selection]
            if saved is None or self._read(selection) != ours:
                continue
            self._write(selection, saved, sensitive=True)
            logger.debug("Restored the {} selection", selection)

    def _daemon_ready(self, path: str) -> bool:
        """Whether the ydotool client can connect to a daemon on `path`.

        `ydotool debug` connects through the normal client path but emits no input
        events, so it rejects a stale socket file without disturbing a live daemon.
        """
        env = {"YDOTOOL_SOCKET": path}
        try:
            return (
                subprocess.run(
                    self._ctx.wrap("ydotool", ["debug"], env=env),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=_SUBPROCESS_TIMEOUT_S,
                    env=self._ctx.subprocess_env(env),
                ).returncode
                == 0
            )
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _ensure_daemon(self, tool: str) -> None:
        """Find or start the chord tool's background service.

        It adopts a live daemon on ydotool's socket, such as a ydotoold user service, or
        launches one: natively the package's ydotoold, else the one on PATH; from a
        Flatpak, the bundled one on the host. One that exits at once, as it does without
        /dev/uinput access, is started again by the next chord.
        """
        if self._daemon_socket:
            return

        # ydotool's own default, /tmp only without a runtime dir. Flatpak always sets
        # XDG_RUNTIME_DIR to /run/user/$UID, the host's per-user runtime dir.
        path = os.environ.get("YDOTOOL_SOCKET") or os.path.join(
            os.environ.get("XDG_RUNTIME_DIR", "/tmp"), _YDOTOOL_SOCKET_NAME
        )
        if self._daemon_ready(path):
            self._daemon_socket = path
            return
        if self._daemon is not None and self._daemon.poll() is None:
            return
        with self._daemon_lock:
            if self._closed:
                return
            logger.debug("Starting {} (socket {})", tool, path)
            try:
                # Never waited on: it serves every later chord until shutdown stops it.
                self._daemon = daemon = subprocess.Popen(
                    self._ctx.wrap(tool, ["-p", path]),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as e:
                self._failure_log(tool)("Could not start {}: {}", tool, e)
                return
        deadline = time.monotonic() + _YDOTOOLD_READY_TIMEOUT_S
        while time.monotonic() < deadline and daemon.poll() is None:
            if self._daemon_ready(path):
                time.sleep(_YDOTOOLD_SETUP_S)
                self._daemon_socket = path
                logger.info("{} ready (socket {})", tool, path)
                return
            time.sleep(_YDOTOOLD_POLL_INTERVAL_S)
        if daemon.returncode is not None:
            self._failure_log(tool)("{} exited with code {}", tool, daemon.returncode)
        else:
            self._failure_log(tool)(
                "{} did not become ready within {}s", tool, _YDOTOOLD_READY_TIMEOUT_S
            )

    def _order_chords(self) -> None:
        """In a Flatpak, put ydotool first once the host proves /dev/uinput writable.

        A virtual keyboard can exit 0 and deliver nothing (COSMIC), while uinput reaches
        every compositor, so ydotool goes first once it can run.
        """
        if not self._order_pending:
            return
        writable = self._host_uinput_writable()
        self._order_pending = writable is None
        if writable:
            self._chords.sort(key=lambda chord: chord.tool != "ydotool")
            logger.debug("Host /dev/uinput writable; ydotool goes first")

    def _send_chord(self) -> str | None:
        """Send Shift+Insert, falling back across the senders in order.

        Return the tool that sent it, or None when none did.
        """
        if not self._chords:
            # Every dictation lands here; `init_typer` has already said why at startup.
            if not self._no_chord_warned:
                logger.error("No chord-sender available; cannot paste")
                self._no_chord_warned = True
            return None
        self._order_chords()
        for chord in self._chords:
            chord_env: dict[str, str] | None = None
            if chord.requires_daemon:
                self._ensure_daemon(chord.requires_daemon)
                if self._daemon_socket:
                    chord_env = {"YDOTOOL_SOCKET": self._daemon_socket}
            argv = self._ctx.wrap(chord.tool, list(chord.args), env=chord_env)
            try:
                result = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=_SUBPROCESS_TIMEOUT_S,
                    env=self._ctx.subprocess_env(chord_env),
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                self._failure_log(chord.tool)(
                    "Chord-sender {} could not run: {}", chord.tool, e
                )
                continue
            logger.debug(
                "Chord-sender {} argv={} -> exit {}",
                chord.tool,
                argv,
                result.returncode,
            )
            if result.returncode == 0:
                return chord.tool
            if chord.requires_daemon:
                # A daemon that died after answering is looked for again next time.
                self._daemon_socket = None
            # ydotool 1.x reports a missing daemon on stdout. The output is capped
            # because it is untrusted host output.
            self._failure_log(chord.tool)(
                "Chord-sender {} failed (code {}): {}",
                chord.tool,
                result.returncode,
                (result.stderr or result.stdout).strip()[:200],
            )
        logger.error(
            "Text injection failed: every chord-sender failed ({})",
            ", ".join(c.tool for c in self._chords),
        )
        return None

    def _failure_log(self, tool: str):
        """Return how to log `tool` failing: WARNING the first time, then DEBUG."""
        if tool in self._failed_tools:
            return logger.debug
        self._failed_tools.add(tool)
        return logger.warning

    def _read(self, selection: str) -> bytes | None:
        """Return a selection's text as raw bytes, or None when it cannot be read.

        `b""` means it is empty or holds no text; None means it could not be read at all
        (timeout, wl-paste missing). Bytes, so any encoding round-trips and compares
        exactly.
        """
        # `text` prefers the UTF-8 type: GTK serves bare text/plain as escaped ASCII.
        args = ["--no-newline", "--type", "text"]
        if selection == "primary":
            args.append("--primary")
        try:
            result = subprocess.run(
                self._ctx.wrap("wl-paste", args),
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("wl-paste ({}) failed: {}", selection, e)
            return None
        if result.returncode != 0:
            # wl-paste exits non-zero for an empty selection and for non-text content.
            logger.debug(
                "wl-paste ({}) returned {} (treated as empty)",
                selection,
                result.returncode,
            )
            return b""
        return result.stdout

    def _write(self, selection: str, data: bytes, *, sensitive: bool = False) -> bool:
        """Set a selection, or clear it when `data` is empty; return success.

        `sensitive` adds --sensitive where wl-copy supports it.
        """
        args = ["--primary"] if selection == "primary" else []
        # wl-copy forks a daemon that owns the selection and inherits our stdio, so
        # capturing output would block until the selection is next replaced.
        try:
            if data:
                # Untyped, wl-copy sniffs the content and offers text starting
                # "From " as a mailbox only, which no text field pastes.
                args += ["--type", "text/plain"]
                if sensitive and self._supports_sensitive:
                    args.append("--sensitive")
                # wl-copy keeps the text in a file under TMPDIR while it owns the
                # selection; the runtime dir is private tmpfs, where /tmp may be disk.
                runtime = os.environ.get("XDG_RUNTIME_DIR")
                env = {"TMPDIR": runtime} if runtime else None
                run = subprocess.run(
                    self._ctx.wrap("wl-copy", args, env=env),
                    input=data,
                    env=self._ctx.subprocess_env(env),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=_SUBPROCESS_TIMEOUT_S,
                )
            else:
                run = subprocess.run(
                    self._ctx.wrap("wl-copy", [*args, "--clear"]),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=_SUBPROCESS_TIMEOUT_S,
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error("wl-copy ({}) failed: {}", selection, e)
            return False
        if run.returncode != 0:
            # A compositor without primary selection fails every primary write; the
            # caller reports that once.
            log = logger.debug if selection == "primary" else logger.error
            log("wl-copy ({}) failed (code {})", selection, run.returncode)
            return False
        return True


# --- Module-level backend ---

_backend: _ClipboardBackend | None = None
# One dictation at a time: the clipboard transaction (save, write, paste, restore)
# must not interleave with another, or with the prewarm's daemon setup. A Lock is not
# FIFO, so injections queued behind a stalled spawn may paste out of order.
_inject_lock = threading.Lock()


def _parse_wl_clipboard_version(output: str) -> tuple[int, int, int]:
    """Parse `wl-copy --version` output into (major, minor, patch).

    An unrecognisable version reads as (0, 0, 0), so an unknown build is unsupported.
    """
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", output)
    if not match:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _wl_copy_supports_sensitive(ctx: _ExecutionContext) -> bool:
    """Whether the available wl-copy understands --sensitive (wl-clipboard >= 2.3.0).

    Detected from `wl-copy --version`; any failure reads as unsupported.
    """
    try:
        result = subprocess.run(
            ctx.wrap("wl-copy", ["--version"]),
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    return _parse_wl_clipboard_version(result.stdout) >= _MIN_SENSITIVE_VERSION


def _native_ydotool_chord(ctx: _ExecutionContext) -> _ChordSender | None:
    """Return the chord for the installed ydotool's CLI, or None when unrecognised.

    Identified from `ydotool --help`, which touches no input backend: 0.1.8 lists its
    commands on stderr and exits 1, 1.x lists them on stdout and exits 0.
    """
    try:
        result = subprocess.run(
            ctx.wrap("ydotool", ["--help"]),
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Could not identify native ydotool command interface: {}", e)
        return None

    def commands(output: str) -> set[str]:
        lines = output.strip().splitlines()
        if tuple(lines[:2]) != _YDOTOOL_HELP_HEADER:
            return set()
        return {line[2:] for line in lines[2:] if line.startswith("  ")}

    if (
        result.returncode == 1
        and not result.stdout
        and _YDOTOOL_LEGACY_COMMANDS <= commands(result.stderr)
    ):
        return _YDOTOOL_LEGACY_CHORD
    if (
        result.returncode == 0
        and not result.stderr
        and _YDOTOOL_MODERN_COMMANDS <= commands(result.stdout)
    ):
        return _YDOTOOL_CHORD
    logger.warning("Unsupported native ydotool command interface; skipping ydotool")
    return None


def init_typer() -> InjectionResult | None:
    """Install the clipboard-paste backend; return why paste cannot work, or None.

    Paste needs a Wayland session, `wl-clipboard` and a chord-sender, and a Flatpak
    needs host /dev/uinput access to paste everywhere. Where one is missing the
    backend is still installed, so dictation still transcribes, and the caller
    surfaces the reason.
    """
    global _backend
    ctx = _ExecutionContext.detect()
    available = {s.tool: ctx.available(s.tool) for s in _CHORD_SENDERS}
    logger.debug("Chord-sender availability: {}", available)
    chords = [_WTYPE_CHORD] if available["wtype"] else []
    if available["ydotool"]:
        ydotool_chord = _YDOTOOL_CHORD if ctx.sandboxed else _native_ydotool_chord(ctx)
        if ydotool_chord:
            # Native uinput avoids silent virtual-keyboard modifier loss. In a Flatpak
            # it goes first once the host proves /dev/uinput access (`_order_chords`).
            if ctx.sandboxed:
                chords.append(ydotool_chord)
            else:
                chords.insert(0, ydotool_chord)
    has_wl_copy = ctx.available("wl-copy")
    supports_sensitive = has_wl_copy and _wl_copy_supports_sensitive(ctx)
    logger.debug("wl-copy --sensitive supported: {}", supports_sensitive)
    if not environment.wayland_session():
        unusable = InjectionResult.NO_WAYLAND
    elif not has_wl_copy:
        unusable = InjectionResult.NO_WL_CLIPBOARD
    else:
        unusable = None
    _backend = backend = _ClipboardBackend(
        ctx, chords, supports_sensitive=supports_sensitive, unusable=unusable
    )

    reason = unusable or (None if chords else InjectionResult.NO_CHORD_SENDER)
    # Without host uinput the Flatpak has only wtype, which can exit 0 without pasting
    # (on COSMIC, and into GTK 3 apps under Sway), so the need shows at launch.
    if reason is None and ctx.sandboxed and backend._host_uinput_writable() is False:
        reason = InjectionResult.UINPUT_SETUP_REQUIRED
    if reason is InjectionResult.UINPUT_SETUP_REQUIRED:
        logger.warning(
            "Host /dev/uinput not writable; paste uses wtype until the uinput setup runs"
        )
    elif reason:
        logger.error(
            "Text injection unavailable ({}): dictation will transcribe but cannot paste",
            reason.value,
        )
    else:
        logger.info(
            "Text backend: clipboard paste · chords={} · ctx={}",
            [c.tool for c in chords],
            "host-spawn" if ctx.sandboxed else "direct",
        )
    return reason


def type_text(text: str) -> InjectionResult:
    """Paste `text` into the focused window and return how it went.

    Blank text counts as success. Blocks on subprocesses and the restore delay, so
    callers run it on a worker thread.
    """
    if not text or not text.strip():
        return InjectionResult.SUCCESS

    with _inject_lock:
        return _backend.inject(text)


def copy_text(text: str) -> bool:
    """Put `text` on the clipboard for a manual paste, with no chord; return success."""
    with _inject_lock:
        return _backend.copy(text)


def prewarm_typer() -> None:
    """Start, on a worker thread, what the first paste would wait for.

    Call it after `init_typer`. ydotoold takes about a second to set up its device; a
    paste meanwhile waits for it.
    """

    def run() -> None:
        with _inject_lock:
            _backend.prewarm()

    threading.Thread(target=run, name="typer-prewarm", daemon=True).start()


def shutdown_typer() -> None:
    """Stop what the backend started, if `init_typer` has run."""
    if _backend is not None:
        _backend.shutdown()


# --- One-time uinput setup for ydotool under Flatpak ---

# Grant the logged-in user access to /dev/uinput so bundled ydotool, which reaches
# every compositor, can send Shift+Insert. The native packages install the same rule
# under the same name, so this copy masks theirs.
_UINPUT_RULE = (
    'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"'
)
# A generous ceiling for the pkexec prompt, so one left unanswered cannot park the
# worker thread forever.
_UINPUT_SETUP_TIMEOUT_S = 120


def uinput_setup_command() -> list[str]:
    """Return the argv that installs /dev/uinput access for ydotool on the host.

    It runs through `flatpak-spawn --host` with admin rights from `pkexec`. The rule
    text arrives on stdin (`cat > file`) and is never interpolated into the fixed
    script, so no data reaches the privileged shell as code. A symlink there is an
    administrator's mask, which the script refuses to write through.
    """
    script = (
        "f=/etc/udev/rules.d/60-quoth-uinput.rules; "
        '[ -L "$f" ] && { echo "$f is masked" >&2; exit 1; }; '
        'cat > "$f" && udevadm control --reload-rules && modprobe uinput && '
        "udevadm trigger --sysname-match=uinput"
    )
    return ["flatpak-spawn", "--host", "pkexec", "sh", "-c", script]


def run_uinput_setup() -> bool:
    """Run the one-time uinput setup on the Flatpak host; return whether it succeeded.

    Pipes the udev rule to `uinput_setup_command()`; the user must log out and back in
    to apply it.
    """
    try:
        result = subprocess.run(
            uinput_setup_command(),
            input=_UINPUT_RULE + "\n",
            capture_output=True,
            text=True,
            timeout=_UINPUT_SETUP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("uinput setup timed out; polkit prompt left unanswered")
        return False
    except OSError as e:
        logger.error("uinput setup failed to launch: {}", e)
        return False
    if result.returncode != 0:
        logger.warning(
            "uinput setup exited {}: {}", result.returncode, result.stderr.strip()[:200]
        )
        return False
    logger.info("uinput setup installed; log out and back in to enable typing")
    return True
