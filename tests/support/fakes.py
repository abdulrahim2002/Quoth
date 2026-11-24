# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Fakes for the injected collaborators."""

from __future__ import annotations

import functools
import inspect
import subprocess
import threading
from pathlib import Path

from quoth.models import AVAILABLE_MODELS


class FakeWhisper:
    """The TranscriptionBackend surface, configured per test.

    The controller builds it through `whisper_factory(model_path, *, device, ...)`, so
    options go in via `FakeWhisper.factory(gate=gate)`. `transcribe_buffer` records its
    language and translate flag into `captured`, sets `entered`, waits on `gate`, then
    raises `raise_on_transcribe` or returns `result`. `warm_up` has its own counter so
    readiness work does not look like a dictation. `init_kwargs` keeps the factory's
    keyword arguments.
    """

    def __init__(
        self,
        *args,
        result: str = "hello world",
        gate: threading.Event | None = None,
        entered: threading.Event | None = None,
        raise_on_init: Exception | None = None,
        raise_on_transcribe: Exception | None = None,
        raise_on_warm_up: Exception | None = None,
        captured: dict | None = None,
        **kwargs,
    ):
        if raise_on_init is not None:
            raise raise_on_init
        self.is_loaded = True
        self.openvino_active = False
        self.closed = False
        self.cancelled = 0
        self.calls = 0
        self.warmups = 0
        self.init_kwargs = kwargs
        self._result = result
        self._gate = gate
        self._entered = entered
        self._raise_on_transcribe = raise_on_transcribe
        self._raise_on_warm_up = raise_on_warm_up
        self._captured = captured

    @classmethod
    def factory(cls, **options):
        return functools.partial(cls, **options)

    def transcribe_buffer(self, audio, language="auto", translate=False):
        self.calls += 1
        if self._captured is not None:
            self._captured.update(language=language, translate=translate)
        if self._entered is not None:
            self._entered.set()
        if self._gate is not None:
            self._gate.wait()
        if self._raise_on_transcribe is not None:
            raise self._raise_on_transcribe
        return self._result

    def warm_up(self):
        self.warmups += 1
        if self._raise_on_warm_up is not None:
            raise self._raise_on_warm_up

    def cancel(self):
        self.cancelled += 1

    def close(self):
        self.closed = True


class FakeScheduler:
    """The whole `Scheduler` protocol, synchronously, under GLib's source rules.

    `dispatch` runs inline. Timers are recorded as `[id, fn, args, delay_ms]` rather
    than run, so a test can assert what was armed and fire it explicitly; `fire_all`
    drains them in order. As in GLib, a callback's True keeps its source: a timer
    stays armed under its id, and a dispatch fails, since GLib would run it on every
    idle. Cancelling an id that is not armed fails, as `GLib.source_remove` warns.
    """

    def __init__(self):
        self.timers: list[list] = []
        self._next_id = 1

    def dispatch(self, fn, *args):
        assert not fn(*args), f"{fn} returned True, so GLib would rerun it forever"

    def after_ms(self, ms, fn, *args):
        return self._add(fn, args, ms)

    def after_s(self, seconds, fn, *args):
        return self._add(fn, args, seconds * 1000)

    def _add(self, fn, args, delay_ms):
        tid = self._next_id
        self._next_id += 1
        self.timers.append([tid, fn, args, delay_ms])
        return tid

    def cancel(self, source_id):
        assert source_id in [t[0] for t in self.timers], f"no armed timer {source_id!r}"
        self.timers = [t for t in self.timers if t[0] != source_id]

    def fire_all(self):
        # A timer stays armed while it runs, as in GLib: it may cancel itself or a
        # later one, which then does not run.
        for timer in list(self.timers):
            if timer not in self.timers:
                continue
            _, fn, args, _delay_ms = timer
            if not fn(*args) and timer in self.timers:
                self.timers.remove(timer)


class DeferredScheduler(FakeScheduler):
    """`FakeScheduler` whose dispatches wait in `dispatches` until `fire_dispatches`.

    As on the main loop, a dispatched callback has not run when the dispatching
    thread moves on, so a test can tell it from one called on the worker thread.
    """

    def __init__(self):
        super().__init__()
        self.dispatches: list[tuple] = []

    def dispatch(self, fn, *args):
        self.dispatches.append((fn, args))

    def fire_dispatches(self):
        pending, self.dispatches = self.dispatches, []
        for fn, args in pending:
            super().dispatch(fn, *args)


class FakeAudio:
    """The AudioBackend surface, counting calls instead of touching GStreamer."""

    def __init__(self):
        self.mic_source = "auto"
        self.microphones = []
        self.on_audio_buffer = None
        self.on_level = None
        self.monitoring = False
        self.started = 0
        self.stopped = 0
        self.warmed = 0
        self.destroyed = 0

    def set_mic_source(self, source):
        source = source or "auto"
        if source == self.mic_source:
            return False
        self.mic_source = source
        return True

    def list_microphones(self):
        return self.microphones

    def set_monitoring(self, enabled):
        self.monitoring = enabled

    def warm_up(self):
        self.warmed += 1

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def destroy(self):
        self.destroyed += 1


_CATALOG = frozenset(m.name for m in AVAILABLE_MODELS)


def _present(models, model) -> bool:
    """Whether `models` (True for the whole catalog, False for none) holds `model`."""
    return model in (_CATALOG if models is True else models or ())


class FakeModelManager:
    """The ModelManager surface the controllers and open_context actually touch.

    Paths come back as `Path`, matching the real manager: a fake handing back plain
    strings lets a test pass while the code under test does path arithmetic that would
    fail against the real one. `encoder` drives both `openvino_path` and
    `has_openvino`, since on disk they are the same fact. `ggml` and `encoder` are the
    models present, a set of names or True for the whole catalog. As on disk, a name
    outside the catalog is never present, so a lookup by the wrong value fails.
    """

    models_dir = Path("/models")

    def __init__(self, *, ggml=True, encoder=True, vad=True):
        self._ggml = ggml
        self._encoder = encoder
        self._vad = vad

    def is_downloaded(self, model) -> bool:
        return _present(self._ggml, model)

    def model_path(self, model):
        return (
            Path(f"/models/ggml-{model}.bin") if _present(self._ggml, model) else None
        )

    def openvino_path(self, model):
        if not _present(self._encoder, model):
            return None
        return Path(f"/models/ggml-{model}-encoder-openvino.xml")

    def has_openvino(self, model) -> bool:
        return _present(self._encoder, model)

    def vad_model_path(self):
        return Path("/models/ggml-silero-v6.2.0.bin") if self._vad else None

    @property
    def cache_dir(self):
        return Path("/models/cache")


class FakePortalShortcuts:
    """The `PortalShortcuts` surface onboarding and Settings read and drive.

    `change(status, binding)` plays the desktop reporting a new state to the subscriber;
    `configure_calls` and `unsubscribe_calls` count what the view asked of the session.
    """

    def __init__(self, status="checking", binding=""):
        self.status = status
        self.binding = binding
        self.version = 2
        self.configure_calls = 0
        self.unsubscribe_calls = 0
        self._callback = None

    def subscribe(self, callback):
        self._callback = callback

        def unsubscribe():
            self.unsubscribe_calls += 1

        return unsubscribe

    def configure(self):
        self.configure_calls += 1
        return True

    def change(self, status, binding=""):
        self.status = status
        self.binding = binding
        self._callback()


# wl-copy offers text under its given type plus these aliases.
_TEXT_TYPES = (
    "text/plain;charset=utf-8",
    "text/plain",
    "TEXT",
    "STRING",
    "UTF8_STRING",
)


def wl_copy_offers(data: bytes, mime: str = "text/plain") -> dict[str, bytes]:
    """The MIME types wl-copy offers `data` under, given `--type mime`."""
    if not data:
        return {}
    if mime.startswith("text/"):
        return dict.fromkeys(_TEXT_TYPES, data) | {mime: data}
    return {mime: data}


# Read at import, so a test that also patches Popen still checks against the real one.
_POPEN_SIGNATURE = inspect.signature(subprocess.Popen)


class FakeClipboard:
    """A `subprocess.run` stand-in for the typer's clipboard backend.

    `wl-paste` and `wl-copy` read and write in-memory selections; any other tool is a
    chord sender that "pastes" the clipboard. Like the real tools they speak MIME:
    `offers[selection]` maps each offered type to its bytes, and `selections` is what a
    text field would read from each. wl-copy offers its `--type` (with text aliases
    when it is text); untyped, it sniffs the content, modelled by the sniff that bites:
    a leading "From " is offered only as application/mbox. `wl-paste --type text`
    prefers UTF-8 and reads only text, an explicit type matches exactly, and untyped it
    falls back to any type.

    `sensitive` records whether the last write was tagged, `chord_codes` scripts the
    chord tools' exit codes in order (then 0), and `calls` keeps every argv.
    `on_paste(board)` runs after a successful chord: the moment the user, a password
    manager or another app could change a selection. `failed_reads` counts reads per
    selection that time out before reads succeed again. A tool run through
    `flatpak-spawn --host` is the same tool, its `--env` flags its environment. With a
    `daemon`, ydotool 1.x works only through a socket that `FakeYdotoold` serves, named
    by its YDOTOOL_SOCKET; there `ydotool debug` answers without typing.
    """

    def __init__(
        self,
        clipboard: bytes = b"",
        primary: bytes = b"",
        *,
        chord_codes=(),
        primary_supported: bool = True,
        fail_clipboard_write: bool = False,
        daemon: FakeYdotoold | None = None,
    ):
        self.offers = {
            "clipboard": wl_copy_offers(clipboard),
            "primary": wl_copy_offers(primary),
        }
        self.sensitive = {"clipboard": False, "primary": False}
        self.chord_codes = list(chord_codes)
        self.primary_supported = primary_supported
        self.fail_clipboard_write = fail_clipboard_write
        self.daemon = daemon
        self.failed_reads = {"clipboard": 0, "primary": 0}
        self.on_paste = None
        self.pasted: list[bytes] = []
        self.calls: list[list[str]] = []

    @property
    def selections(self) -> dict[str, bytes]:
        return {name: self._serve(name, "text") or b"" for name in self.offers}

    def __call__(
        self, argv, *, input=None, timeout=None, capture_output=False, **popen_kwargs
    ):
        # subprocess.run hands other keywords to Popen, so a misspelt one fails here.
        _POPEN_SIGNATURE.bind(argv, **popen_kwargs)
        self.calls.append(list(argv))
        tool, args, env = _unwrap(argv, popen_kwargs.get("env"))
        if tool == "ydotool" and self.daemon is not None:
            if env.get("YDOTOOL_SOCKET") not in self.daemon.serving:
                return subprocess.CompletedProcess(argv, 2, stdout=b"", stderr=b"")
            if args == ["debug"]:
                return subprocess.CompletedProcess(argv, 0)
        selection = "primary" if "--primary" in argv else "clipboard"
        typed = argv[argv.index("--type") + 1] if "--type" in argv else None
        unsupported = selection == "primary" and not self.primary_supported
        if tool == "wl-paste":
            if self.failed_reads[selection]:
                self.failed_reads[selection] -= 1
                raise subprocess.TimeoutExpired(argv, timeout)
            data = None if unsupported else self._serve(selection, typed)
            return subprocess.CompletedProcess(
                argv, 0 if data else 1, stdout=data or b""
            )
        if tool == "wl-copy":
            if unsupported or (selection == "clipboard" and self.fail_clipboard_write):
                return subprocess.CompletedProcess(argv, 1)
            cleared = "--clear" in argv
            if cleared:
                self.offers[selection] = {}
            elif typed is not None:
                self.offers[selection] = wl_copy_offers(input, typed)
            elif input.startswith(b"From "):
                self.offers[selection] = {"application/mbox": input}
            else:
                self.offers[selection] = wl_copy_offers(input)
            self.sensitive[selection] = not cleared and "--sensitive" in argv
            return subprocess.CompletedProcess(argv, 0)
        code = self.chord_codes.pop(0) if self.chord_codes else 0
        if code == 0:
            self.pasted.append(self._serve("clipboard", "text") or b"")
            if self.on_paste is not None:
                self.on_paste(self)
        return subprocess.CompletedProcess(argv, code, stdout=b"", stderr=b"")

    def _serve(self, selection, request):
        """What `wl-paste [--type <request>]` prints, or None when no offer matches."""
        offers = self.offers[selection]
        if request in ("text", None):
            for mime in ("text/plain;charset=utf-8", "text/plain", *offers):
                if mime.startswith("text/") and mime in offers:
                    return offers[mime]
            return None if request else next(iter(offers.values()), None)
        return offers.get(request)


def _unwrap(argv, env):
    """The tool `argv` runs, its arguments and its environment."""
    env = dict(env or {})
    if argv[0] == "flatpak-spawn":
        argv = argv[2:]
        while argv[0].startswith("--env="):
            key, value = argv.pop(0).removeprefix("--env=").split("=", 1)
            env[key] = value
    return Path(argv[0]).name, argv[1:], env


class FakeYdotoold:
    """A `subprocess.Popen` stand-in for ydotoold, and the sockets a daemon serves.

    `serving` starts with the sockets an already running daemon answers on, such as a
    user service's. A launched daemon serves its `-p` socket until it is terminated,
    unless it is not `ready` or exits at once with `exit_code`, as it does without
    /dev/uinput. `launched` keeps every argv.
    """

    def __init__(self, *serving: str, exit_code: int | None = None, ready: bool = True):
        self.serving = set(serving)
        self.exit_code = exit_code
        self.ready = ready
        self.launched: list[list[str]] = []

    def __call__(self, argv, **popen_kwargs):
        _POPEN_SIGNATURE.bind(argv, **popen_kwargs)
        self.launched.append(list(argv))
        socket = argv[argv.index("-p") + 1]
        if self.exit_code is None and self.ready:
            self.serving.add(socket)
        return _Daemon(self.serving, socket, self.exit_code)


class _Daemon:
    def __init__(self, serving: set[str], socket: str, returncode: int | None):
        self._serving, self._socket, self.returncode = serving, socket, returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self._serving.discard(self._socket)


class FinishedDialog:
    """A `Gtk.FileDialog` the user has answered: `open_finish` returns `value`.

    An exception `value` (a cancel or a failure as `GLib.Error`) is raised rather than
    returned. `finished` records that the callback collected the answer.
    """

    def __init__(self, value):
        self.value = value
        self.finished = False

    def open_finish(self, _result):
        self.finished = True
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def fake_engine(first_level=None):
    """A specced `RecordingController` as onboarding and its Try it step drive it.

    Monitoring adds the watcher to `watchers` and hands it `first_level` when set, or
    raises as a missing microphone does while `unavailable` is set. A take begins
    unless `begin_take.return_value` is False; `end_take` keeps `on_text` for the test
    to answer.
    """
    from unittest.mock import create_autospec

    from quoth.recording import RecordingController

    engine = create_autospec(RecordingController, instance=True)
    engine.watchers = set()
    engine.unavailable = False
    engine.on_text = None

    def set_input_monitoring(watcher, enabled):
        if engine.unavailable:
            raise RuntimeError("no microphone")
        (engine.watchers.add if enabled else engine.watchers.discard)(watcher)
        if enabled and first_level is not None:
            watcher.push_level(first_level)

    def end_take(device, on_text):
        engine.on_text = on_text

    engine.set_input_monitoring.side_effect = set_input_monitoring
    engine.begin_take.return_value = True
    engine.end_take.side_effect = end_take
    return engine


class FakeBenchmark:
    """The app's benchmark start, as onboarding and its optimize step call it.

    The run holds the engine until `finish`; `accept=False` plays a busy engine.
    """

    def __init__(self, *, accept=True):
        self.accept = accept
        self.on_progress = None
        self.on_done = None

    def __call__(self, _models, _devices, on_progress, on_done):
        self.on_progress = on_progress
        self.on_done = on_done
        return self.accept

    def finish(self, report):
        self.on_done(report)
