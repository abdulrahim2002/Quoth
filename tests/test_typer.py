# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `typer`: clipboard-paste text injection and its chord senders.

Many tests run `subprocess.run` through `FakeClipboard`, which models wl-copy, wl-paste
and the chord senders.
"""

import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quoth.typer import InjectionResult
from tests.support import FakeClipboard, FakeYdotoold, wl_copy_offers


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # No test waits out the paste's restore delay or the ydotoold poll.
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _no_backend(monkeypatch):
    # The backend init_typer installs is module state, so no test may leave one behind.
    monkeypatch.setattr("quoth.typer._backend", None)


class TestClipboardBackend:
    """Save, paste, restore; a failed chord leaves the transcript on the clipboard."""

    def _backend(self, *, sensitive=False):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        return mod._ClipboardBackend(
            ctx, [mod._WTYPE_CHORD], supports_sensitive=sensitive
        )

    def _inject(self, board, text="Hello world", **kwargs):
        with patch("subprocess.run", new=board):
            return self._backend(**kwargs).inject(text)

    def test_pastes_then_gives_both_selections_back(self):
        board = FakeClipboard(b"old clip", b"old prim")
        # Terminals paste PRIMARY on Shift+Insert, so it holds the transcript too.
        at_paste = []
        board.on_paste = lambda b: at_paste.append(b.selections)
        assert self._inject(board) is InjectionResult.SUCCESS
        assert board.pasted == [b"Hello world"]
        assert at_paste == [{"clipboard": b"Hello world", "primary": b"Hello world"}]
        assert board.selections == {"clipboard": b"old clip", "primary": b"old prim"}
        chord = next(c for c in board.calls if c[0] == "wtype")
        assert chord == [
            "wtype",
            "-M",
            "shift",
            "-P",
            "Insert",
            "-p",
            "Insert",
            "-m",
            "shift",
        ]

    @pytest.mark.parametrize("chord_codes", [[0], [1]])
    def test_the_transcript_is_logged_only_at_trace(self, caplog, chord_codes):
        # Transcript text is logged only at TRACE, which no sink writes to disk.
        board = FakeClipboard(chord_codes=chord_codes)
        self._inject(board, "sentinel dictation")
        heard = {r.levelname for r in caplog.records if "sentinel" in r.getMessage()}
        assert heard == {"TRACE"}

    def test_clears_selections_that_were_empty(self):
        board = FakeClipboard()
        assert self._inject(board) is InjectionResult.SUCCESS
        assert board.selections == {"clipboard": b"", "primary": b""}

    def test_failed_chord_keeps_text_on_clipboard_and_gives_primary_back(self):
        board = FakeClipboard(b"old clip", b"old prim", chord_codes=[1])
        assert self._inject(board, "Hello") is InjectionResult.CLIPBOARD_ONLY
        # Left for a manual paste; primary was never consumed, so it goes back.
        assert board.selections == {"clipboard": b"Hello", "primary": b"old prim"}

    def test_no_chord_sender_keeps_text_on_clipboard(self):
        import quoth.typer as mod

        board = FakeClipboard(b"old clip", b"old prim")
        backend = mod._ClipboardBackend(
            mod._ExecutionContext(sandboxed=False, host_bin=None), []
        )
        with patch("subprocess.run", new=board):
            assert backend.inject("Hello") is InjectionResult.NO_CHORD_SENDER
        assert board.selections == {"clipboard": b"Hello", "primary": b"old prim"}

    def test_no_chord_sender_is_logged_once(self, caplog):
        # init_typer reports it at startup; each dictation repeating it says nothing.
        import quoth.typer as mod

        caplog.set_level("ERROR")
        board = FakeClipboard(b"old clip")
        backend = mod._ClipboardBackend(
            mod._ExecutionContext(sandboxed=False, host_bin=None), []
        )
        with patch("subprocess.run", new=board):
            for text in ("one", "two"):
                assert backend.inject(text) is InjectionResult.NO_CHORD_SENDER
        [record] = caplog.records
        assert "No chord-sender" in record.getMessage()

    @pytest.mark.parametrize(
        "reason", [InjectionResult.NO_WAYLAND, InjectionResult.NO_WL_CLIPBOARD]
    )
    def test_unusable_clipboard_reports_its_reason_without_running_tools(self, reason):
        import quoth.typer as mod

        board = FakeClipboard(b"old clip")
        backend = mod._ClipboardBackend(
            mod._ExecutionContext(sandboxed=False, host_bin=None),
            [mod._WTYPE_CHORD],
            unusable=reason,
        )
        with patch("subprocess.run", new=board):
            assert backend.inject("Hello") is reason
        assert board.calls == []

    def test_reports_failure_when_clipboard_write_fails(self):
        board = FakeClipboard(b"old", b"old", fail_clipboard_write=True)
        assert self._inject(board) is InjectionResult.FAILED
        assert board.pasted == []

    def test_failed_chord_reports_failure_if_clipboard_changed(self):
        board = FakeClipboard(b"old clip", b"old prim", chord_codes=[1])
        original_run = board

        def copy_during_failed_chord(argv, **kwargs):
            result = original_run(argv, **kwargs)
            if argv[0] == "wtype":
                board.offers["clipboard"] = wl_copy_offers(b"copied meanwhile")
            return result

        with patch("subprocess.run", new=copy_during_failed_chord):
            result = self._backend().inject("Hello")
        assert result is InjectionResult.FAILED
        assert board.selections == {
            "clipboard": b"copied meanwhile",
            "primary": b"old prim",
        }

    def test_copy_puts_text_on_the_clipboard_alone(self):
        board = FakeClipboard(b"old clip", b"old prim")
        with patch("subprocess.run", new=board):
            assert self._backend(sensitive=True).copy("Hello world")
        assert board.selections == {"clipboard": b"Hello world", "primary": b"old prim"}
        assert board.sensitive["clipboard"] is True
        assert board.pasted == []

    @pytest.mark.parametrize("sandboxed", [False, True])
    def test_wl_copy_keeps_the_text_in_the_runtime_dir(self, monkeypatch, sandboxed):
        # wl-copy holds its input in a file under TMPDIR, which defaults to /tmp.
        import quoth.typer as mod

        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        host_bin = "/host/app/files/bin" if sandboxed else None
        ctx = mod._ExecutionContext(sandboxed=sandboxed, host_bin=host_bin)
        backend = mod._ClipboardBackend(ctx, [mod._WTYPE_CHORD])
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=0)) as run,
            patch.object(mod.environment, "in_flatpak", return_value=sandboxed),
        ):
            assert backend.copy("Hello")
        argv, env = run.call_args.args[0], run.call_args.kwargs["env"]
        if sandboxed:
            assert "--env=TMPDIR=/run/user/1000" in argv
        else:
            assert env["TMPDIR"] == "/run/user/1000"

    def test_tags_transcript_and_restore_sensitive_when_supported(self):
        # The transcript is dictation, and the restore is not a new user copy: history
        # managers should record neither.
        board = FakeClipboard(b"old clip", b"old prim")
        assert (
            self._inject(board, "secret words", sensitive=True)
            is InjectionResult.SUCCESS
        )
        writes = [c for c in board.calls if c[0] == "wl-copy"]
        assert writes and all("--sensitive" in c for c in writes)
        assert board.sensitive == {"clipboard": True, "primary": True}

    def test_omits_sensitive_flag_when_unsupported(self):
        # Below wl-clipboard 2.3.0 the flag would make wl-copy exit non-zero.
        board = FakeClipboard(b"old", b"old")
        assert self._inject(board) is InjectionResult.SUCCESS
        writes = [c for c in board.calls if c[0] == "wl-copy"]
        assert writes and not any("--sensitive" in c for c in writes)

    def test_leaves_a_selection_that_changed_during_the_paste(self):
        # The user copied something, or a password manager cleared its secret (KeePassXC
        # clears only while its value is still there): either way it is no longer ours.
        board = FakeClipboard(b"old clip", b"old prim")
        board.on_paste = lambda b: b.offers.update(
            clipboard=wl_copy_offers(b"copied meanwhile")
        )
        assert self._inject(board) is InjectionResult.SUCCESS
        assert board.selections == {
            "clipboard": b"copied meanwhile",
            "primary": b"old prim",
        }

    def test_never_restores_a_selection_it_could_not_read(self):
        # A failed read is not an empty selection: clearing it would lose its contents.
        board = FakeClipboard(b"unread", b"old prim")
        board.failed_reads["clipboard"] = 1
        assert self._inject(board, "Hello") is InjectionResult.SUCCESS
        assert board.selections == {"clipboard": b"Hello", "primary": b"old prim"}

    def test_non_utf8_selection_round_trips(self):
        board = FakeClipboard(b"caf\xe9 \xff\xfe", b"\r\nline")
        assert self._inject(board) is InjectionResult.SUCCESS
        assert board.selections == {
            "clipboard": b"caf\xe9 \xff\xfe",
            "primary": b"\r\nline",
        }

    def test_gives_back_a_gtk_clipboard_as_utf8(self):
        # GTK serves bare text/plain as escaped ASCII; only its UTF-8 type is the text.
        board = FakeClipboard("Grüße".encode(), b"old prim")
        board.offers["clipboard"] = {
            "text/plain;charset=utf-8": "Grüße".encode(),
            "text/plain": b"Gr\\C3\\BC\\C3\\9Fe",
        }
        assert self._inject(board) is InjectionResult.SUCCESS
        assert board.selections["clipboard"] == "Grüße".encode()

    def test_clears_a_clipboard_that_held_no_text(self):
        # An image is not text: it cannot be given back as text, so it is not kept.
        board = FakeClipboard(b"old clip", b"old prim")
        board.offers["clipboard"] = {"image/png": b"\x89PNG"}
        assert self._inject(board) is InjectionResult.SUCCESS
        assert board.offers["clipboard"] == {}

    def test_offers_a_transcript_as_text_whatever_it_starts_with(self):
        # Left to sniff, wl-copy offers text starting "From " as a mailbox only.
        board = FakeClipboard(b"old clip", b"old prim")
        assert self._inject(board, "From now on. ") is InjectionResult.SUCCESS
        assert board.pasted == [b"From now on. "]
        assert board.selections == {"clipboard": b"old clip", "primary": b"old prim"}

    def test_gives_the_app_time_to_read_the_paste_before_restoring(self):
        import quoth.typer as mod

        board = FakeClipboard(b"old clip", b"old prim")
        with (
            patch("subprocess.run", new=board),
            patch("time.sleep", side_effect=lambda s: board.calls.append(["sleep", s])),
        ):
            self._backend().inject("Hello")
        chord = next(i for i, c in enumerate(board.calls) if c[0] == "wtype")
        assert board.calls[chord + 1] == ["sleep", mod.RESTORE_DELAY_MS / 1000]

    def test_pastes_without_primary_and_warns_once(self, caplog):
        # A compositor without primary selection fails every primary write; failing the
        # paste there would break every dictation.
        caplog.set_level("WARNING")
        board = FakeClipboard(b"old clip", primary_supported=False)
        backend = self._backend()
        with patch("subprocess.run", new=board):
            assert backend.inject("one") is InjectionResult.SUCCESS
            assert backend.inject("two") is InjectionResult.SUCCESS
        assert board.pasted == [b"one", b"two"]
        assert board.selections["clipboard"] == b"old clip"
        [record] = caplog.records
        assert "Primary selection unavailable" in record.getMessage()

    def test_every_spawn_is_bounded(self):
        # A stalled wl-copy, wl-paste or chord-sender becomes a failure, not a hang.
        # The empty clipboard makes the restore clear it, the other kind of write.
        import quoth.typer as mod

        board = FakeClipboard(b"", b"old prim")
        timeouts = []

        def run(argv, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            return board(argv, **kwargs)

        with patch("subprocess.run", new=run):
            assert self._backend().inject("Hello") is InjectionResult.SUCCESS
        assert timeouts and set(timeouts) == {mod._SUBPROCESS_TIMEOUT_S}


class TestOneDictationAtATime:
    """Overlapping dictations paste their own text in turn; the clipboard survives."""

    def test_overlapping_injections_paste_their_own_text_in_turn(self):
        import quoth.typer as mod

        board = FakeClipboard(b"original")
        in_paste, release = threading.Event(), threading.Event()

        def hold_first_paste(_board):
            if not in_paste.is_set():
                in_paste.set()
                release.wait(2)

        board.on_paste = hold_first_paste
        mod._backend = mod._ClipboardBackend(
            mod._ExecutionContext(sandboxed=False, host_bin=None), [mod._WTYPE_CHORD]
        )
        with patch("subprocess.run", new=board):
            first = threading.Thread(target=mod.type_text, args=("first",))
            first.start()
            assert in_paste.wait(2)
            calls_before = len(board.calls)
            second = threading.Thread(target=mod.type_text, args=("second",))
            second.start()
            second.join(0.05)
            # The second transaction has not begun while the first is mid-paste.
            assert second.is_alive() and len(board.calls) == calls_before
            release.set()
            first.join(2)
            assert not first.is_alive()
            second.join(2)
            assert not second.is_alive()
        assert board.pasted == [b"first", b"second"]
        assert board.selections["clipboard"] == b"original"

    def test_a_paste_waits_for_the_prewarm(self):
        # A chord through a ydotoold still setting up its device would paste nothing.
        import quoth.typer as mod

        in_prewarm, release = threading.Event(), threading.Event()
        backend = MagicMock()
        backend.prewarm.side_effect = lambda: in_prewarm.set() or release.wait(2)
        mod._backend = backend
        mod.prewarm_typer()
        assert in_prewarm.wait(2)
        paste = threading.Thread(target=mod.type_text, args=("hi",))
        paste.start()
        paste.join(0.05)
        assert paste.is_alive() and not backend.inject.called
        release.set()
        paste.join(2)
        assert not paste.is_alive()
        backend.inject.assert_called_once_with("hi")


class TestExecutionContext:
    """`_ExecutionContext`, which decides where injection commands run."""

    def test_detect_direct_when_no_flatpak_info(self):
        import quoth.typer as mod

        with patch("os.path.exists", return_value=False):
            ctx = mod._ExecutionContext.detect()
        assert ctx.sandboxed is False
        assert ctx.host_bin is None

    def test_detect_sandbox_reads_app_path(self):
        import quoth.typer as mod

        def fake_read(self, _path):
            self.read_string("[Instance]\napp-path=/host/app/files\n")

        with (
            patch("os.path.exists", return_value=True),
            patch("configparser.ConfigParser.read", fake_read),
        ):
            ctx = mod._ExecutionContext.detect()
        assert ctx.sandboxed is True
        assert ctx.host_bin == "/host/app/files/bin"

    def test_detect_falls_back_to_direct_when_app_path_missing(self):
        import quoth.typer as mod

        def fake_read(self, _path):
            self.read_string("[Instance]\nfoo=bar\n")

        with (
            patch("os.path.exists", return_value=True),
            patch("configparser.ConfigParser.read", fake_read),
        ):
            ctx = mod._ExecutionContext.detect()
        assert ctx.sandboxed is False

    def test_wrap_direct_is_identity(self):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        assert ctx.wrap("wtype", ["-M", "shift"]) == ["wtype", "-M", "shift"]

    def test_wrap_sandbox_prefixes_flatpak_spawn_host(self):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=True, host_bin="/host/app/files/bin")
        # A sandboxed context only exists inside a Flatpak, where /.flatpak-info is
        # present: that's what host_argv() re-checks to decide on the host escape.
        with patch("os.path.exists", return_value=True):
            assert ctx.wrap("wtype", ["-M", "shift"]) == [
                "flatpak-spawn",
                "--host",
                "/host/app/files/bin/wtype",
                "-M",
                "shift",
            ]

    def test_available_direct_uses_which(self):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        with patch("shutil.which", return_value="/usr/bin/wtype"):
            assert ctx.available("wtype") is True
        with patch("shutil.which", return_value=None):
            assert ctx.available("wtype") is False

    def test_direct_execution_prefers_the_package_bundled_tool(self, tmp_path):
        import quoth.typer as mod

        bundled = tmp_path / "wl-copy"
        bundled.write_text("#!/bin/sh\n")
        bundled.chmod(0o755)
        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        with patch.object(mod, "_PACKAGE_BIN", str(tmp_path)):
            assert ctx.wrap("wl-copy", ["--version"]) == [str(bundled), "--version"]
            assert ctx.wrap("wtype", []) == ["wtype"]
            with patch.dict(os.environ, {"PATH": ""}):
                assert ctx.available("wl-copy") is True
                assert ctx.available("wtype") is False

    def test_available_sandbox_checks_bundled_path(self):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=True, host_bin="/host/app/files/bin")
        with patch("os.path.exists", lambda p: p == "/app/bin/wtype"):
            assert ctx.available("wtype") is True
            assert ctx.available("ydotool") is False


class TestChordSenders:
    """The chord-sender argv data, without running anything."""

    def test_ydotool_chord_argv(self):
        import quoth.typer as mod

        assert mod._YDOTOOL_CHORD.tool == "ydotool"
        # Shift(42) down, Insert(110) down, Insert up, Shift up: raw uinput codes.
        assert list(mod._YDOTOOL_CHORD.args) == [
            "key",
            "42:1",
            "110:1",
            "110:0",
            "42:0",
        ]

    def test_legacy_ydotool_chord_argv(self):
        import quoth.typer as mod

        assert list(mod._YDOTOOL_LEGACY_CHORD.args) == ["key", "Shift+Insert"]
        assert mod._YDOTOOL_LEGACY_CHORD.requires_daemon is None


class TestChordFallback:
    """Chord-sender fallback in `_ClipboardBackend`."""

    def _backend(self, chords=None):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        if chords is None:
            chords = [mod._WTYPE_CHORD, mod._YDOTOOL_LEGACY_CHORD]
        # Native, with a ydotool that needs no daemon, so only the chords run.
        return mod._ClipboardBackend(ctx, chords)

    def test_falls_back_to_the_next_chord_and_keeps_the_order(self, caplog):
        board = FakeClipboard(b"old", b"old", chord_codes=[1, 0, 1, 0])
        backend = self._backend()
        with patch("subprocess.run", new=board):
            assert backend.inject("one") is InjectionResult.SUCCESS
            assert backend.inject("two") is InjectionResult.SUCCESS
        chords = [c[0] for c in board.calls if c[0] in ("wtype", "ydotool")]
        # A fallback's exit 0 proves no paste, so the first sender stays first.
        assert chords == ["wtype", "ydotool", "wtype", "ydotool"]
        # A sender that keeps failing warns once.
        [record] = [r for r in caplog.records if r.levelname == "WARNING"]
        assert "Chord-sender wtype failed" in record.getMessage()
        # Each paste names the sender that landed it, never the text.
        summary = "Paste sent via ydotool (selections=['clipboard', 'primary'])"
        assert caplog.messages.count(summary) == 2

    def test_reports_clipboard_only_when_all_chords_fail(self, caplog):
        caplog.set_level("ERROR")
        board = FakeClipboard(b"old clip", b"old prim", chord_codes=[1, 1])
        with patch("subprocess.run", new=board):
            assert self._backend().inject("hi") is InjectionResult.CLIPBOARD_ONLY
        [record] = caplog.records
        assert "every chord-sender failed" in record.getMessage()
        # The dictated text stays on the clipboard; only primary is given back.
        assert board.selections == {"clipboard": b"hi", "primary": b"old prim"}

    @pytest.mark.parametrize(
        "error",
        [subprocess.TimeoutExpired("wtype", 5), FileNotFoundError("wtype")],
    )
    def test_falls_back_past_a_sender_that_cannot_run(self, error):
        # A stalled sender, or one removed since startup, is skipped like a failed one.
        board = FakeClipboard(b"old clip", b"old prim")

        def run(argv, **kwargs):
            if argv[0] == "wtype":
                board.calls.append(list(argv))
                raise error
            return board(argv, **kwargs)

        with patch("subprocess.run", new=run):
            assert self._backend().inject("hi") is InjectionResult.SUCCESS
        assert board.pasted == [b"hi"]
        assert board.selections == {"clipboard": b"old clip", "primary": b"old prim"}

    def test_logs_a_sender_error_printed_on_stdout(self, caplog):
        # ydotool 1.x prints a missing daemon on stdout, not stderr.
        caplog.set_level("WARNING")
        board = FakeClipboard(b"old", b"old")

        def run(argv, **kwargs):
            if argv[0] in ("wtype", "ydotool"):
                # Like subprocess.run, output comes back only when it is captured.
                captured = kwargs.get("capture_output")
                return subprocess.CompletedProcess(
                    argv,
                    2,
                    stdout="failed to connect socket" if captured else None,
                    stderr="" if captured else None,
                )
            return board(argv, **kwargs)

        with patch("subprocess.run", new=run):
            self._backend().inject("hi")
        assert any(m.endswith(": failed to connect socket") for m in caplog.messages)

    def test_writes_selections_once_despite_fallback(self):
        board = FakeClipboard(b"old", b"old", chord_codes=[1, 0])
        with patch("subprocess.run", new=board):
            self._backend().inject("hi")
        # The second chord pasted, after the first failed.
        assert board.pasted == [b"hi"]
        transcript_writes = [
            c for c in board.calls if c[0] == "wl-copy" and "--clear" not in c
        ]
        # 2 writes (clipboard + primary) + 2 restores: the failed chord does
        # not trigger an extra selection write.
        assert len(transcript_writes) == 4


class TestYdotoolDaemon:
    """ydotoold (1.0+): Quoth adopts a running daemon or starts its own."""

    _SOCKET = "/run/user/1000/.ydotool_socket"
    _HOST_BIN = "/host/app/files/bin"

    @pytest.fixture(autouse=True)
    def _runtime_dir(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        monkeypatch.delenv("YDOTOOL_SOCKET", raising=False)

    @pytest.fixture
    def flatpak(self, monkeypatch):
        monkeypatch.setattr("quoth.typer.environment.in_flatpak", lambda: True)

    def _backend(self, chords=None, *, sandboxed=True):
        import quoth.typer as mod

        host_bin = self._HOST_BIN if sandboxed else None
        ctx = mod._ExecutionContext(sandboxed=sandboxed, host_bin=host_bin)
        if chords is None:
            chords = [mod._YDOTOOL_CHORD]
        backend = mod._ClipboardBackend(ctx, chords)
        # ydotoold runs only where the host grants /dev/uinput.
        if sandboxed:
            backend._host_uinput = True
        return backend

    def _inject(self, backend, daemon, *texts, board=None):
        board = board or FakeClipboard(b"old", b"old", daemon=daemon)
        with patch("subprocess.run", new=board), patch("subprocess.Popen", new=daemon):
            return board, [backend.inject(text) for text in texts]

    @staticmethod
    def _chords(board):
        # ydotool's key command sends the chord; the probe is `ydotool debug`.
        return [c for c in board.calls if "key" in c]

    @staticmethod
    def _probes(board):
        return [c for c in board.calls if c[-1] == "debug"]

    def test_starts_daemon_when_socket_absent_and_pins_env(self, flatpak):
        daemon = FakeYdotoold()
        board, results = self._inject(self._backend(), daemon, "hi")

        assert results == [InjectionResult.SUCCESS]
        assert board.pasted == [b"hi"]
        [argv] = daemon.launched
        assert argv[:3] == ["flatpak-spawn", "--host", f"{self._HOST_BIN}/ydotoold"]
        # ydotool's own per-user path, never the shared /tmp another account could own.
        assert argv[-2:] == ["-p", self._SOCKET]
        # The ydotool chord receives the pinned host socket via flatpak-spawn.
        [chord] = self._chords(board)
        assert chord[:4] == [
            "flatpak-spawn",
            "--host",
            f"--env=YDOTOOL_SOCKET={self._SOCKET}",
            f"{self._HOST_BIN}/ydotool",
        ]

    def test_adopts_running_daemon_without_launching(self, flatpak):
        daemon = FakeYdotoold(self._SOCKET)  # a service's, or an earlier session's
        _board, results = self._inject(self._backend(), daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert daemon.launched == []

    def test_a_named_socket_overrides_the_default(self, flatpak, monkeypatch):
        named = "/run/user/1000/custom.sock"
        monkeypatch.setenv("YDOTOOL_SOCKET", named)
        daemon = FakeYdotoold(named)
        _board, results = self._inject(self._backend(), daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert daemon.launched == []

    def test_ready_daemon_is_reused(self, flatpak):
        daemon = FakeYdotoold(self._SOCKET)
        board, results = self._inject(self._backend(), daemon, "one", "two")
        assert results == [InjectionResult.SUCCESS] * 2
        assert len(self._probes(board)) == 1  # the second reuses the resolved daemon

    def test_daemon_that_exited_is_started_again(self, flatpak):
        # Before the typing setup grants /dev/uinput, ydotoold exits at once; once it
        # has, the next dictation must start it rather than stay clipboard-only.
        daemon = FakeYdotoold(exit_code=2)
        backend = self._backend()
        board, results = self._inject(backend, daemon, "one")
        assert results == [InjectionResult.CLIPBOARD_ONLY]
        # The dead daemon is not polled to the deadline.
        assert len(self._probes(board)) == 1

        daemon.exit_code = None
        _board, results = self._inject(backend, daemon, "two")

        assert results == [InjectionResult.SUCCESS]
        assert len(daemon.launched) == 2

    def test_a_daemon_still_running_unready_is_not_launched_again(self, flatpak):
        # One slower than the wait, or alive but not answering, must not gain a twin
        # on every dictation; it is adopted once it answers.
        import quoth.typer as mod

        daemon = FakeYdotoold(ready=False)
        backend = self._backend()
        with patch.object(mod, "_YDOTOOLD_READY_TIMEOUT_S", 0):
            _board, results = self._inject(backend, daemon, "one", "two")
            daemon.serving.add(self._SOCKET)
            _board, more = self._inject(backend, daemon, "three")

        assert results + more == [InjectionResult.CLIPBOARD_ONLY] * 2 + [
            InjectionResult.SUCCESS
        ]
        assert len(daemon.launched) == 1

    def test_native_adopts_a_running_service(self):
        daemon = FakeYdotoold(self._SOCKET)
        board, results = self._inject(self._backend(sandboxed=False), daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert daemon.launched == []
        assert self._chords(board)[0][0] == "ydotool"

    def test_native_starts_the_installed_daemon(self):
        daemon = FakeYdotoold()
        _board, results = self._inject(self._backend(sandboxed=False), daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert daemon.launched == [["ydotoold", "-p", self._SOCKET]]

    def test_a_started_daemon_sets_up_its_device_before_the_chord(self, monkeypatch):
        # It answers before its device exists; an early chord lands after the restore.
        daemon = FakeYdotoold()
        board = FakeClipboard(b"old", b"old", daemon=daemon)
        monkeypatch.setattr("time.sleep", lambda s: board.calls.append(["sleep", s]))
        self._inject(self._backend(sandboxed=False), daemon, "hi", board=board)

        [chord] = self._chords(board)
        wait = board.calls[board.calls.index(chord) - 1]
        # The wait right before the chord covers the device's one-second setup.
        assert wait[0] == "sleep" and wait[1] >= 1

    def test_native_honours_a_named_socket(self, monkeypatch):
        named = "/run/user/1000/custom-ydotool"
        monkeypatch.setenv("YDOTOOL_SOCKET", named)
        daemon = FakeYdotoold(named)
        _board, results = self._inject(self._backend(sandboxed=False), daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert daemon.launched == []

    def test_a_daemon_without_uinput_keeps_the_text_and_warns_once(self, caplog):
        # Where /dev/uinput stays closed, ydotoold exits at once on every dictation.
        daemon = FakeYdotoold(exit_code=2)
        board, results = self._inject(
            self._backend(sandboxed=False), daemon, "one", "two"
        )

        assert results == [InjectionResult.CLIPBOARD_ONLY] * 2
        assert len(daemon.launched) == 2
        assert board.selections["clipboard"] == b"two"
        # One warning for the daemon and one for the chord, not one per dictation.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 2

    def test_a_daemon_that_died_is_looked_for_again(self):
        daemon = FakeYdotoold(self._SOCKET)
        backend = self._backend(sandboxed=False)
        _board, results = self._inject(backend, daemon, "one")
        daemon.serving.clear()  # the service stopped after answering
        _board, more = self._inject(backend, daemon, "two", "three")

        assert results + more == [
            InjectionResult.SUCCESS,
            InjectionResult.CLIPBOARD_ONLY,
            InjectionResult.SUCCESS,
        ]
        assert len(daemon.launched) == 1

    def test_wtype_chord_never_touches_the_daemon(self):
        import quoth.typer as mod

        daemon = FakeYdotoold()
        backend = self._backend(chords=[mod._WTYPE_CHORD], sandboxed=False)
        board, results = self._inject(backend, daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert daemon.launched == [] and self._probes(board) == []

    def test_the_probe_is_bounded_and_prints_nothing(self, flatpak):
        # A stalled flatpak-spawn must not hang the paste, and a stale socket's error
        # must not reach Quoth's output on every dictation.
        import quoth.typer as mod

        daemon = FakeYdotoold(self._SOCKET)
        board, probes = FakeClipboard(daemon=daemon), []

        def run(argv, **kwargs):
            if argv[-1] == "debug":
                probes.append(kwargs)
            return board(argv, **kwargs)

        with patch("subprocess.run", new=run), patch("subprocess.Popen", new=daemon):
            self._backend().prewarm()
        [kwargs] = probes
        assert kwargs["timeout"] == mod._SUBPROCESS_TIMEOUT_S
        assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL

    def test_prewarm_starts_the_daemon_before_the_first_paste(self):
        import quoth.typer as mod

        daemon = FakeYdotoold()
        board = FakeClipboard(b"old", b"old", daemon=daemon)
        backend = self._backend([mod._YDOTOOL_CHORD, mod._WTYPE_CHORD], sandboxed=False)
        with patch("subprocess.run", new=board), patch("subprocess.Popen", new=daemon):
            backend.prewarm()
            assert len(daemon.launched) == 1
            assert backend.inject("hi") is InjectionResult.SUCCESS

        assert len(daemon.launched) == 1
        assert len(self._probes(board)) == 2  # the paste found the socket resolved
        assert self._chords(board)[0][0] == "ydotool"

    def test_prewarm_puts_ydotool_first_where_the_flatpak_host_allows(self, flatpak):
        import quoth.typer as mod

        daemon = FakeYdotoold()
        backend = self._backend([mod._WTYPE_CHORD, mod._YDOTOOL_CHORD])
        with (
            patch("subprocess.run", new=FakeClipboard(daemon=daemon)),
            patch("subprocess.Popen", new=daemon),
        ):
            backend.prewarm()
        assert daemon.launched[0][:3] == [
            "flatpak-spawn",
            "--host",
            f"{self._HOST_BIN}/ydotoold",
        ]

        # With wtype first, its chord would paste and ydotool would send none.
        board, results = self._inject(backend, daemon, "hi")
        assert results == [InjectionResult.SUCCESS]
        assert len(self._chords(board)) == 1

    @pytest.mark.parametrize(
        ("chords", "sandboxed", "host_uinput"),
        [
            (["_WTYPE_CHORD"], False, None),
            (["_YDOTOOL_LEGACY_CHORD", "_WTYPE_CHORD"], False, None),
            # Until host uinput is proven, wtype stays first and ydotoold cannot start.
            (["_WTYPE_CHORD", "_YDOTOOL_CHORD"], True, False),
            (["_WTYPE_CHORD", "_YDOTOOL_CHORD"], True, None),
            ([], False, None),
        ],
    )
    def test_prewarm_starts_nothing_unless_the_first_sender_needs_a_daemon(
        self, chords, sandboxed, host_uinput
    ):
        import quoth.typer as mod

        daemon = FakeYdotoold()
        board = FakeClipboard(daemon=daemon)
        backend = self._backend(
            [getattr(mod, name) for name in chords], sandboxed=sandboxed
        )
        with (
            patch.object(backend, "_host_uinput_writable", return_value=host_uinput),
            patch("subprocess.run", new=board),
            patch("subprocess.Popen", new=daemon),
        ):
            backend.prewarm()

        assert board.calls == []
        assert daemon.launched == []

    def test_shutdown_stops_the_daemon_it_started(self, flatpak):
        # A live Flatpak helper would hold the app's deployment and block an uninstall.
        daemon = FakeYdotoold()
        backend = self._backend()
        self._inject(backend, daemon, "hi")
        backend.shutdown()
        assert daemon.serving == set()

    def test_a_quit_during_the_prewarm_starts_no_daemon(self):
        # Nothing would stop one started after shutdown; in a Flatpak it would keep the
        # sandbox alive.
        probing, release = threading.Event(), threading.Event()
        daemon = FakeYdotoold()
        board = FakeClipboard(daemon=daemon)

        def run(argv, **kwargs):
            if argv[-1] == "debug":
                probing.set()
                release.wait(2)
            return board(argv, **kwargs)

        backend = self._backend(sandboxed=False)
        with patch("subprocess.run", new=run), patch("subprocess.Popen", new=daemon):
            warm = threading.Thread(target=backend.prewarm)
            warm.start()
            assert probing.wait(2)
            backend.shutdown()
            release.set()
            warm.join(2)
            assert not warm.is_alive()

        assert daemon.launched == []

    def test_shutdown_leaves_an_adopted_daemon_running(self, flatpak):
        daemon = FakeYdotoold(self._SOCKET)
        backend = self._backend()
        self._inject(backend, daemon, "hi")
        backend.shutdown()
        assert daemon.serving == {self._SOCKET}


class TestTypeText:
    """`type_text` with its empty-text guard, `copy_text` and `shutdown_typer`."""

    @patch("subprocess.run")
    def test_skips_empty_text(self, mock_run):
        from quoth.typer import type_text

        assert type_text("") is InjectionResult.SUCCESS
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_skips_whitespace_only(self, mock_run):
        from quoth.typer import type_text

        assert type_text("   ") is InjectionResult.SUCCESS
        mock_run.assert_not_called()

    def test_pastes_through_the_installed_backend(self):
        import quoth.typer as mod

        with (
            patch(
                "shutil.which",
                side_effect=lambda tool: (
                    "/usr/bin/x" if tool in ("wtype", "wl-copy") else None
                ),
            ),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch(
                "quoth.typer._ClipboardBackend.inject",
                return_value=InjectionResult.CLIPBOARD_ONLY,
            ) as inject,
            patch("subprocess.run") as run,
        ):
            mod.init_typer()
            # The caller gets the backend's own result, here a paste that failed.
            assert mod.type_text("hello") is InjectionResult.CLIPBOARD_ONLY
            inject.assert_called_once_with("hello")
            run.assert_not_called()

    @pytest.mark.parametrize(
        ("write_fails", "copied", "clipboard"),
        [(False, True, b"Hello"), (True, False, b"old clip")],
    )
    def test_copy_text_reports_whether_the_text_reached_the_clipboard(
        self, write_fails, copied, clipboard
    ):
        # Copy last dictation sends no chord, and says when the copy failed.
        import quoth.typer as mod

        board = FakeClipboard(b"old clip", fail_clipboard_write=write_fails)
        mod._backend = mod._ClipboardBackend(
            mod._ExecutionContext(sandboxed=False, host_bin=None), [mod._WTYPE_CHORD]
        )
        with patch("subprocess.run", new=board):
            assert mod.copy_text("Hello") is copied
        assert board.selections["clipboard"] == clipboard
        assert board.pasted == []

    def test_shutdown_stops_what_the_installed_backend_started(self):
        import quoth.typer as mod

        mod._backend = None
        mod.shutdown_typer()  # a quit before init_typer has nothing to stop
        mod._backend = backend = MagicMock(spec=mod._ClipboardBackend)
        mod.shutdown_typer()
        backend.shutdown.assert_called_once_with()


class TestBackendDetection:
    """Backend selection in `init_typer`."""

    @pytest.fixture(autouse=True)
    def _wayland(self, monkeypatch):
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    @pytest.mark.parametrize(
        ("returncode", "stdout", "stderr", "expected"),
        [
            (
                1,
                "",
                "Usage: ydotool <cmd> <args>\nAvailable commands:\n"
                "  click\n  key\n  mousemove\n  recorder\n  type",
                "_YDOTOOL_LEGACY_CHORD",
            ),
            (
                0,
                "Usage: ydotool <cmd> <args>\nAvailable commands:\n"
                "  click\n  debug\n  key\n  mousemove\n  type",
                "",
                "_YDOTOOL_CHORD",
            ),
        ],
    )
    def test_native_prefers_recognized_ydotool(
        self, returncode, stdout, stderr, expected
    ):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        with (
            patch.object(mod._ExecutionContext, "detect", return_value=ctx),
            patch.object(mod._ExecutionContext, "available", return_value=True),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch(
                "subprocess.run",
                return_value=MagicMock(
                    returncode=returncode, stdout=stdout, stderr=stderr
                ),
            ),
        ):
            mod.init_typer()

        assert mod._backend._chords == [getattr(mod, expected), mod._WTYPE_CHORD]

    def test_native_wtype_does_not_probe_ydotool(self):
        import quoth.typer as mod

        with (
            patch(
                "shutil.which",
                side_effect=lambda tool: (
                    "/usr/bin/x" if tool in ("wtype", "wl-copy") else None
                ),
            ),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch("subprocess.run") as run,
        ):
            mod.init_typer()
            assert isinstance(mod._backend, mod._ClipboardBackend)
            assert mod._backend._chords == [mod._WTYPE_CHORD]
            run.assert_not_called()

    def test_flatpak_keeps_pinned_modern_ydotool_without_probe(self):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=True, host_bin="/host/app/files/bin")
        with (
            patch.object(mod._ExecutionContext, "detect", return_value=ctx),
            patch.object(mod._ExecutionContext, "available", return_value=True),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch.object(
                mod._ClipboardBackend, "_host_uinput_writable", return_value=True
            ),
            patch("subprocess.run") as run,
        ):
            mod.init_typer()
            assert mod._backend._chords == [mod._WTYPE_CHORD, mod._YDOTOOL_CHORD]
            run.assert_not_called()

    @pytest.mark.parametrize(
        ("desktop", "host_writable", "reason"),
        [
            ("COSMIC", False, InjectionResult.UINPUT_SETUP_REQUIRED),
            ("sway", False, InjectionResult.UINPUT_SETUP_REQUIRED),
            ("GNOME", False, InjectionResult.UINPUT_SETUP_REQUIRED),
            ("sway", True, None),
            ("sway", None, None),
        ],
    )
    def test_a_flatpak_needs_host_uinput_before_the_first_dictation(
        self, monkeypatch, desktop, host_writable, reason
    ):
        # wtype can exit 0 without pasting, so a failed paste cannot prove the need.
        import quoth.typer as mod

        monkeypatch.setenv("XDG_CURRENT_DESKTOP", desktop)
        ctx = mod._ExecutionContext(sandboxed=True, host_bin="/host/app/files/bin")
        with (
            patch.object(mod._ExecutionContext, "detect", return_value=ctx),
            patch.object(mod._ExecutionContext, "available", return_value=True),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch.object(mod.environment, "wayland_session", return_value=True),
            patch.object(
                mod._ClipboardBackend,
                "_host_uinput_writable",
                return_value=host_writable,
            ),
        ):
            assert mod.init_typer() is reason

    def test_unrecognized_native_ydotool_is_omitted(self, caplog):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        with (
            patch.object(mod._ExecutionContext, "detect", return_value=ctx),
            patch.object(mod._ExecutionContext, "available", return_value=True),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch(
                "subprocess.run",
                return_value=MagicMock(
                    returncode=0, stdout="Usage: ydotool", stderr=""
                ),
            ),
        ):
            mod.init_typer()
        assert mod._backend._chords == [mod._WTYPE_CHORD]
        [warning] = [r for r in caplog.records if r.levelname == "WARNING"]
        assert "Unsupported native ydotool" in warning.getMessage()

    @pytest.mark.parametrize(
        ("wayland", "installed", "reason"),
        [
            (True, {"wtype", "wl-copy"}, None),
            (True, {"wtype"}, InjectionResult.NO_WL_CLIPBOARD),
            (True, {"wl-copy"}, InjectionResult.NO_CHORD_SENDER),
            (False, {"wtype", "wl-copy"}, InjectionResult.NO_WAYLAND),
        ],
    )
    def test_returns_why_paste_cannot_work(self, wayland, installed, reason, caplog):
        import quoth.typer as mod

        caplog.set_level("ERROR")

        # A missing piece still installs a backend (the app stays alive), and every
        # dictation then reports the same reason the caller surfaces at startup.
        with (
            patch(
                "shutil.which",
                side_effect=lambda tool: "/usr/bin/x" if tool in installed else None,
            ),
            patch.object(mod.environment, "wayland_session", return_value=wayland),
            patch.object(mod, "_wl_copy_supports_sensitive", return_value=False),
            patch("subprocess.run") as run,
        ):
            assert mod.init_typer() is reason
            assert isinstance(mod._backend, mod._ClipboardBackend)
            assert bool(caplog.records) is (reason is not None)
            run.assert_not_called()
        if reason in (InjectionResult.NO_WAYLAND, InjectionResult.NO_WL_CLIPBOARD):
            assert mod._backend.inject("hi") is reason

    @pytest.mark.parametrize(
        ("version", "sensitive"), [("2.3.0", True), ("2.2.1", False)]
    )
    def test_dictation_is_tagged_sensitive_only_where_wl_copy_supports_it(
        self, version, sensitive
    ):
        # Below wl-clipboard 2.3.0 the flag would make wl-copy exit non-zero.
        import quoth.typer as mod

        board = FakeClipboard(b"old clip", b"old prim")

        def run(argv, **kwargs):
            if argv == ["wl-copy", "--version"]:
                return subprocess.CompletedProcess(
                    argv, 0, stdout=f"wl-clipboard {version}\n"
                )
            return board(argv, **kwargs)

        with (
            patch(
                "shutil.which",
                side_effect=lambda tool: (
                    "/usr/bin/x" if tool in ("wtype", "wl-copy") else None
                ),
            ),
            patch.object(mod.environment, "wayland_session", return_value=True),
            patch("subprocess.run", new=run),
        ):
            assert mod.init_typer() is None
            assert mod.type_text("hi") is InjectionResult.SUCCESS
        assert board.sensitive == {"clipboard": sensitive, "primary": sensitive}


class TestNativeYdotoolDetection:
    """Native ydotool 0.1.8 and 1.x expose incompatible key command forms."""

    LEGACY_HELP = """Usage: ydotool <cmd> <args>
Available commands:
  type
  recorder
  mousemove
  key
  click"""
    MODERN_HELP = """Usage: ydotool <cmd> <args>
Available commands:
  click
  mousemove
  type
  key
  debug
  bakers
Use environment variable YDOTOOL_SOCKET to specify daemon socket."""

    def _detect(self, *, returncode, stdout="", stderr=""):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        result = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
        with patch("subprocess.run", return_value=result) as run:
            chord = mod._native_ydotool_chord(ctx)
        [(args, kwargs)] = run.call_args_list
        assert args == (["ydotool", "--help"],)
        # 0.1.8 lists its commands on stderr and 1.x on stdout, parsed as text.
        pipes = (kwargs.get("stdout"), kwargs.get("stderr"))
        assert kwargs.get("capture_output") or pipes == (subprocess.PIPE,) * 2
        assert (kwargs["timeout"], kwargs["text"]) == (mod._SUBPROCESS_TIMEOUT_S, True)
        return chord

    def test_selects_legacy_named_chord(self):
        import quoth.typer as mod

        assert (
            self._detect(returncode=1, stderr=self.LEGACY_HELP)
            is mod._YDOTOOL_LEGACY_CHORD
        )

    def test_selects_modern_numeric_chord(self):
        import quoth.typer as mod

        assert self._detect(returncode=0, stdout=self.MODERN_HELP) is mod._YDOTOOL_CHORD

    def test_tolerates_extra_commands_and_trailing_help(self):
        import quoth.typer as mod

        help_text = self.MODERN_HELP.replace("  click\n", "  future-command\n  click\n")
        assert (
            self._detect(returncode=0, stdout=f"{help_text}\nMore help follows.")
            is mod._YDOTOOL_CHORD
        )

    def test_probe_errors_are_bounded_and_logged_once(self, caplog):
        import quoth.typer as mod

        caplog.set_level("WARNING")
        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        for error in (
            FileNotFoundError("ydotool"),
            subprocess.TimeoutExpired("ydotool", mod._SUBPROCESS_TIMEOUT_S),
        ):
            caplog.clear()
            with patch("subprocess.run", side_effect=error):
                assert mod._native_ydotool_chord(ctx) is None
            [record] = caplog.records
            assert "Could not identify native ydotool" in record.getMessage()


class TestSensitiveDetection:
    """`--sensitive` capability is gated on the wl-clipboard version, fail-open."""

    def test_parse_version(self):
        import quoth.typer as mod

        assert mod._parse_wl_clipboard_version("wl-clipboard 2.3.0\n") == (2, 3, 0)
        assert mod._parse_wl_clipboard_version("wl-clipboard 2.2.1") == (2, 2, 1)
        assert mod._parse_wl_clipboard_version("wl-clipboard 2.4") == (2, 4, 0)
        assert mod._parse_wl_clipboard_version("garbage") == (0, 0, 0)

    def _supported(self, version_output, returncode=0):
        import quoth.typer as mod

        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=returncode, stdout=version_output),
        ) as run:
            supported = mod._wl_copy_supports_sensitive(ctx)
        [(args, kwargs)] = run.call_args_list
        assert args == (["wl-copy", "--version"],)
        assert kwargs.get("capture_output") or kwargs.get("stdout") == subprocess.PIPE
        assert (kwargs["timeout"], kwargs["text"]) == (mod._SUBPROCESS_TIMEOUT_S, True)
        return supported

    def test_supported_at_and_above_floor(self):
        assert self._supported("wl-clipboard 2.3.0") is True
        assert self._supported("wl-clipboard 2.4.1") is True

    def test_unsupported_below_floor(self):
        # The real 2.2.1 case: predates --sensitive, so the flag must be omitted.
        assert self._supported("wl-clipboard 2.2.1") is False

    def test_fail_open_on_anomalies(self):
        import quoth.typer as mod

        # Non-zero exit and unparseable output read as unsupported.
        assert self._supported("wl-clipboard 2.3.0", returncode=1) is False
        assert self._supported("???") is False

        # A missing binary or a timeout must never propagate: paste stays unchanged.
        ctx = mod._ExecutionContext(sandboxed=False, host_bin=None)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert mod._wl_copy_supports_sensitive(ctx) is False
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("wl-copy", 5)
        ):
            assert mod._wl_copy_supports_sensitive(ctx) is False


class TestUinputSetup:
    """Host access detection and the one-time uinput setup."""

    def _sandboxed_backend(self):
        import quoth.typer as mod

        # Keep clipboard subprocesses direct in these unit tests; sandboxed is the
        # behaviour under test, while host-path wrapping is covered above.
        ctx = mod._ExecutionContext(sandboxed=True, host_bin=None)
        return mod._ClipboardBackend(ctx, [mod._WTYPE_CHORD])

    def test_command_runs_pkexec_on_host_without_interpolating_the_rule(self):
        import quoth.typer as mod

        cmd = mod.uinput_setup_command()
        assert cmd[:5] == ["flatpak-spawn", "--host", "pkexec", "sh", "-c"]
        script = cmd[-1]
        # The rule is fed on stdin (cat >), never put into the privileged script.
        assert "f=/etc/udev/rules.d/60-quoth-uinput.rules;" in script
        assert 'cat > "$f"' in script
        assert "udevadm control --reload-rules" in script
        assert "uaccess" not in script and "KERNEL==" not in script
        # A masked rule is refused, and only the uinput device is replayed.
        assert script.index('[ -L "$f" ] && {') < script.index("cat >")
        assert script.endswith("udevadm trigger --sysname-match=uinput")

    def test_the_flatpak_setup_writes_the_rule_the_packages_ship(self):
        import quoth.typer as mod

        shipped = Path(__file__).parents[1] / "data/60-quoth-uinput.rules"
        rules = [
            line
            for line in shipped.read_text().splitlines()
            if not line.startswith("#")
        ]
        assert rules == [mod._UINPUT_RULE]

    @pytest.mark.parametrize(
        ("host_writable", "expected"),
        [
            (False, InjectionResult.UINPUT_SETUP_REQUIRED),
            (True, InjectionResult.CLIPBOARD_ONLY),
            (None, InjectionResult.CLIPBOARD_ONLY),
        ],
    )
    def test_failed_flatpak_paste_requires_setup_only_with_host_evidence(
        self, host_writable, expected
    ):
        board = FakeClipboard(b"old", b"old")
        backend = self._sandboxed_backend()
        with (
            patch("os.access", return_value=True) as sandbox_access,
            patch("subprocess.run", new=board),
            patch.object(backend, "_send_chord", return_value=False),
            patch.object(backend, "_host_uinput_writable", return_value=host_writable),
        ):
            assert backend.inject("hi") is expected
        sandbox_access.assert_not_called()
        assert board.selections["clipboard"] == b"hi"

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (MagicMock(returncode=0, stdout="writable", stderr=""), True),
            (MagicMock(returncode=0, stdout="unwritable", stderr=""), False),
            (MagicMock(returncode=1, stdout="unwritable", stderr="spawn failed"), None),
            (MagicMock(returncode=0, stdout="unexpected", stderr=""), None),
        ],
    )
    def test_host_access_probe_distinguishes_evidence_from_failure(
        self, result, expected
    ):
        import quoth.typer as mod

        backend = self._sandboxed_backend()
        with (
            patch.object(mod.environment, "in_flatpak", return_value=True),
            patch("subprocess.run", return_value=result) as run,
        ):
            assert backend._host_uinput_writable() is expected
        argv = run.call_args.args[0]
        assert argv[:3] == ["flatpak-spawn", "--host", "sh"]
        assert "test -w /dev/uinput" in argv[-1]
        assert run.call_args.kwargs["timeout"] == mod._SUBPROCESS_TIMEOUT_S

    def test_host_access_script_runs_and_answers(self):
        # Outside a Flatpak the fixed script runs in the local shell, for real.
        import quoth.typer as mod

        backend = self._sandboxed_backend()
        with patch.object(mod.environment, "in_flatpak", return_value=False):
            assert isinstance(backend._host_uinput_writable(), bool)

    @pytest.mark.parametrize(("status", "checks"), [("unwritable", 1), ("other", 2)])
    def test_host_access_is_checked_until_proven(self, status, checks):
        backend = self._sandboxed_backend()
        result = MagicMock(returncode=0, stdout=status, stderr="")
        with patch("subprocess.run", return_value=result) as run:
            backend._host_uinput_writable()
            backend._host_uinput_writable()
        assert run.call_count == checks

    @pytest.mark.parametrize(
        ("answers", "order"),
        [
            ([True, True], ["ydotool", "ydotool"]),
            ([False, False], ["wtype", "wtype"]),
            # An unknown answer leaves wtype first until a later check proves access.
            ([None, True], ["wtype", "ydotool"]),
        ],
    )
    def test_flatpak_tries_ydotool_first_where_host_uinput_is_writable(
        self, answers, order
    ):
        import quoth.typer as mod

        board = FakeClipboard(b"old", b"old")
        ctx = mod._ExecutionContext(sandboxed=True, host_bin=None)
        backend = mod._ClipboardBackend(ctx, [mod._WTYPE_CHORD, mod._YDOTOOL_CHORD])
        with (
            patch("subprocess.run", new=board),
            patch.object(backend, "_ensure_daemon"),
            patch.object(backend, "_host_uinput_writable", side_effect=answers),
        ):
            backend.inject("one")
            backend.inject("two")
        chords = [c[0] for c in board.calls if c[0] in ("wtype", "ydotool")]
        assert chords == order

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("flatpak-spawn"),
            subprocess.TimeoutExpired("flatpak-spawn", 5),
        ],
    )
    def test_host_access_probe_errors_are_unknown(self, error):
        backend = self._sandboxed_backend()
        with patch("subprocess.run", side_effect=error):
            assert backend._host_uinput_writable() is None

    def test_native_failure_keeps_clipboard_fallback_without_probe(self):
        import quoth.typer as mod

        board = FakeClipboard(b"old", b"old", chord_codes=[1])
        backend = mod._ClipboardBackend(
            mod._ExecutionContext(sandboxed=False, host_bin=None), [mod._WTYPE_CHORD]
        )
        with patch("subprocess.run", new=board):
            assert backend.inject("hi") is InjectionResult.CLIPBOARD_ONLY
        assert not any("/dev/uinput" in " ".join(call) for call in board.calls)

    @patch("subprocess.run")
    def test_run_setup_returns_true_and_pipes_rule_on_stdin(self, mock_run):
        import quoth.typer as mod

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        assert mod.run_uinput_setup() is True
        # The udev rule reaches the privileged process on stdin, not the command line,
        # and an unanswered polkit prompt gives up after two minutes.
        mock_run.assert_called_once_with(
            mod.uinput_setup_command(),
            input=mod._UINPUT_RULE + "\n",
            capture_output=True,
            text=True,
            timeout=120,
        )

    @patch("subprocess.run")
    def test_run_setup_returns_false_on_nonzero_exit(self, mock_run):
        import quoth.typer as mod

        mock_run.return_value = MagicMock(returncode=1, stderr="cancelled")
        assert mod.run_uinput_setup() is False

    @patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pkexec", timeout=120),
    )
    def test_run_setup_returns_false_on_timeout(self, mock_run):
        import quoth.typer as mod

        assert mod.run_uinput_setup() is False
