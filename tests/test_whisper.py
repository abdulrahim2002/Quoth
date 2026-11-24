# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `whisper`, the ctypes binding to whisper.cpp speech-to-text."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest

from quoth.whisper import (
    ModelLoadFailed,
    WarmupAborted,
    WhisperContext,
    clean_text,
    open_context,
)
from tests.support import FakeModelManager


class _FakeCFunction:
    def __init__(self, callback):
        self._callback = callback

    def __call__(self, *args):
        return self._callback(*args)


class _FakeBackendLibrary:
    def __init__(self, registered=(), register_on_load=(), events=None):
        self.registered = set(registered)
        self.register_on_load = set(register_on_load)
        self.events = events
        self.load_calls = []
        self.queries = []

        def reg_by_name(name):
            self.queries.append(name)
            return 1 if name.decode() in self.registered else None

        def load_all(path):
            self.load_calls.append(path)
            if self.events is not None:
                self.events.append("backends")
            self.registered.update(self.register_on_load)

        self.ggml_backend_reg_by_name = _FakeCFunction(reg_by_name)
        self.ggml_backend_load_all_from_path = _FakeCFunction(load_all)


class TestOpenContext:
    """The factory holds `open_context` to the real `WhisperContext` constructor."""

    def test_none_when_model_not_downloaded(self):
        factory = create_autospec(WhisperContext)
        assert (
            open_context("small", "CPU", FakeModelManager(ggml=False), factory=factory)
            is None
        )

    def test_cpu_passes_no_encoder(self):
        factory = create_autospec(WhisperContext)
        ctx = open_context("small", "CPU", FakeModelManager(), factory=factory)
        assert ctx is not None
        assert factory.call_args.kwargs["device"] == "CPU"
        assert (
            factory.call_args.kwargs["openvino_model_path"] is None
        )  # CPU never wants the OpenVINO encoder

    def test_npu_with_encoder_passes_the_xml(self):
        factory = create_autospec(WhisperContext)
        open_context("small", "NPU", FakeModelManager(encoder=True), factory=factory)
        assert factory.call_args.kwargs["openvino_model_path"].endswith(
            "ggml-small-encoder-openvino.xml"
        )

    def test_npu_without_encoder_falls_through_to_cpu_encoder(self):
        factory = create_autospec(WhisperContext)
        open_context("small", "NPU", FakeModelManager(encoder=False), factory=factory)
        assert (
            factory.call_args.kwargs["openvino_model_path"] is None
        )  # missing, so None (warned, CPU encoder)

    def test_passes_cache_dir_from_model_mgr(self):
        # The blob-cache path comes from ModelManager.cache_dir, never derived here.
        factory = create_autospec(WhisperContext)
        open_context("small", "NPU", FakeModelManager(), factory=factory)
        assert factory.call_args.kwargs["cache_dir"] == "/models/cache"

    def test_none_when_factory_raises(self):
        def boom(*a, **k):
            raise RuntimeError("load failed")

        assert open_context("small", "CPU", FakeModelManager(), factory=boom) is None

    @pytest.mark.parametrize(
        ("error", "traceback"),
        [
            (ModelLoadFailed("no context"), False),
            (RuntimeError("libwhisper.so not found"), True),
            (FileNotFoundError("Model file not found"), True),
            (RuntimeError("defect"), True),
            (TypeError("defect"), True),
        ],
    )
    def test_a_refused_device_is_logged_plainly_and_a_broken_load_with_its_traceback(
        self, error, traceback, caplog
    ):
        # The traceback is also the crash report, which a broken install must reach.
        caplog.set_level("ERROR")

        def boom(*a, **k):
            raise error

        assert open_context("small", "GPU", FakeModelManager(), factory=boom) is None
        [record] = caplog.records
        assert (record.exc_info is not None) is traceback
        if traceback and isinstance(error, (OSError, RuntimeError)):
            # The file sink drops exception values, so the reason is in the message.
            assert record.getMessage().endswith(f": {error}")

    def test_vad_off_passes_no_vad_path(self):
        factory = create_autospec(WhisperContext)
        open_context("small", "CPU", FakeModelManager(), factory=factory)
        assert factory.call_args.kwargs["vad_model_path"] is None

    def test_vad_on_passes_the_vad_model(self):
        factory = create_autospec(WhisperContext)
        open_context(
            "small", "CPU", FakeModelManager(vad=True), vad=True, factory=factory
        )
        assert factory.call_args.kwargs["vad_model_path"].endswith(
            "ggml-silero-v6.2.0.bin"
        )

    def test_vad_on_but_missing_model_leaves_it_off(self):
        factory = create_autospec(WhisperContext)
        open_context(
            "small", "CPU", FakeModelManager(vad=False), vad=True, factory=factory
        )
        assert (
            factory.call_args.kwargs["vad_model_path"] is None
        )  # missing, so off rather than an error


class TestCleanText:
    """`clean_text` normalises whitespace only; decoding suppresses non-speech cues."""

    @pytest.mark.parametrize(
        ("raw", "cleaned"),
        [
            ("Hello   world", "Hello world"),
            ("  Hello world  ", "Hello world"),
            ("Hello\tworld\n", "Hello world"),
            ("   ", ""),
            ("This is a normal sentence.", "This is a normal sentence."),
            # Brackets are kept, so genuinely dictated punctuation survives.
            ("the array [0] is empty", "the array [0] is empty"),
            ("Das ist ein Test über die Straße.", "Das ist ein Test über die Straße."),
        ],
    )
    def test_normalizes_only_whitespace(self, raw, cleaned):
        assert clean_text(raw) == cleaned


class TestWhisperContext:
    def test_raises_on_missing_model(self):
        with pytest.raises(FileNotFoundError):
            WhisperContext("/nonexistent/model.bin")

    @pytest.mark.parametrize(
        ("missing", "library"),
        [
            ("find_libwhisper", "libwhisper.so"),
            ("find_libwhisper_helper", "libwhisper_helper.so"),
        ],
    )
    def test_a_missing_library_asks_for_a_reinstall(
        self, tmp_path, monkeypatch, missing, library
    ):
        # A packaged install has no source tree to build in.
        from quoth import whisper

        model_path = tmp_path / "ggml-base.bin"
        model_path.write_bytes(b"")
        monkeypatch.setattr(whisper, "find_libwhisper", lambda: "/l/libwhisper.so")
        monkeypatch.setattr(
            whisper, "find_libwhisper_helper", lambda: "/l/libwhisper_helper.so"
        )
        monkeypatch.setattr(whisper, missing, lambda: None)
        with pytest.raises(RuntimeError) as raised:
            WhisperContext(str(model_path))
        assert str(raised.value) == (
            f"{library} not found. Reinstall Quoth, then restart it."
        )


class TestCpuBackendInit:
    @pytest.fixture(autouse=True)
    def _reset_attempts(self):
        from quoth import whisper

        with whisper._BACKEND_INIT_LOCK:
            whisper._BACKEND_INIT_ATTEMPTED_DIRS.clear()
        yield
        with whisper._BACKEND_INIT_LOCK:
            whisper._BACKEND_INIT_ATTEMPTED_DIRS.clear()

    def test_linked_cpu_backend_needs_no_directory_scan(self):
        from quoth import whisper

        lib = _FakeBackendLibrary(registered={"CPU"})

        whisper._ensure_cpu_backend(lib, "libwhisper.so")

        assert lib.load_calls == []

    def test_repeated_initialization_loads_a_directory_once(self, tmp_path):
        from quoth import whisper

        lib = _FakeBackendLibrary(register_on_load={"CPU"})
        engine = tmp_path / "lib/libwhisper.so"

        whisper._ensure_cpu_backend(lib, str(engine))
        whisper._ensure_cpu_backend(lib, str(engine))

        assert lib.load_calls == [os.fsencode(engine.resolve().parent)]

    def test_concurrent_initialization_loads_a_directory_once(self, tmp_path):
        from quoth import whisper

        lib = _FakeBackendLibrary(register_on_load={"CPU"})
        engine = str(tmp_path / "lib/libwhisper.so")
        load = lib.ggml_backend_load_all_from_path

        def slow_load(path):
            # Held open so the other threads arrive while the load is under way.
            time.sleep(0.05)
            load(path)

        lib.ggml_backend_load_all_from_path = _FakeCFunction(slow_load)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(whisper._ensure_cpu_backend, lib, engine)
                for _ in range(8)
            ]
            for future in futures:
                future.result()

        assert len(lib.load_calls) == 1

    def test_partial_gpu_registration_is_not_retried(self, tmp_path):
        from quoth import whisper

        lib = _FakeBackendLibrary(register_on_load={"Vulkan"})
        engine = str(tmp_path / "lib/libwhisper.so")

        for _ in range(2):
            with pytest.raises(RuntimeError, match=r"Reinstall Quoth, then restart it"):
                whisper._ensure_cpu_backend(lib, engine)

        lib.registered.add("CPU")
        whisper._ensure_cpu_backend(lib, engine)

        assert len(lib.load_calls) == 1
        assert set(lib.queries) == {b"CPU"}

    def test_bare_soname_does_not_scan_the_working_directory(self):
        from quoth import whisper

        lib = _FakeBackendLibrary()

        with pytest.raises(RuntimeError, match="dynamic backend directory"):
            whisper._ensure_cpu_backend(lib, "libwhisper.so.1")

        assert lib.load_calls == []

    def test_non_ascii_backend_path_uses_filesystem_encoding(self, tmp_path):
        from quoth import whisper

        lib = _FakeBackendLibrary(register_on_load={"CPU"})
        engine = tmp_path / "Quøth libs/libwhisper.so"

        whisper._ensure_cpu_backend(lib, str(engine))

        assert lib.load_calls == [os.fsencode(engine.resolve().parent)]

    def test_context_registers_cpu_before_loading_helper_or_model(
        self, tmp_path, monkeypatch
    ):
        from quoth import whisper

        events = []
        engine_path = tmp_path / "lib/libwhisper.so"
        helper_path = tmp_path / "lib/libwhisper_helper.so"
        model_path = tmp_path / "model.bin"
        model_path.touch()
        engine = _FakeBackendLibrary(register_on_load={"CPU"}, events=events)
        for name in (
            "whisper_full_n_segments",
            "whisper_full_get_segment_text",
            "whisper_free",
            "whisper_ctx_init_openvino_encoder",
        ):
            setattr(engine, name, _FakeCFunction(lambda *_args: None))
        helper = type("Helper", (), {})()
        helper.whisper_helper_init = _FakeCFunction(
            lambda *_args: events.append("model") or 1
        )
        helper.whisper_helper_transcribe = _FakeCFunction(lambda *_args: 0)

        def load(path):
            if path == str(engine_path):
                events.append("engine")
                return engine
            assert path == str(helper_path)
            events.append("helper")
            return helper

        monkeypatch.setattr(whisper, "find_libwhisper", lambda: str(engine_path))
        monkeypatch.setattr(whisper, "find_libwhisper_helper", lambda: str(helper_path))
        monkeypatch.setattr(whisper.ctypes, "CDLL", load)
        monkeypatch.setattr(whisper, "usable_cpus", lambda: 1)

        context = WhisperContext(str(model_path))
        context.close()

        assert events[:4] == ["engine", "backends", "helper", "model"]

    def test_a_null_context_is_a_model_load_failure(self, tmp_path, monkeypatch):
        # What ggml-vulkan returns for a GPU that cannot run the model.
        from quoth import whisper

        model_path = tmp_path / "model.bin"
        model_path.touch()
        engine = _FakeBackendLibrary(registered={"CPU"})
        for name in (
            "whisper_full_n_segments",
            "whisper_full_get_segment_text",
            "whisper_free",
            "whisper_ctx_init_openvino_encoder",
        ):
            setattr(engine, name, _FakeCFunction(lambda *_args: None))
        helper = type("Helper", (), {})()
        helper.whisper_helper_init = _FakeCFunction(lambda *_args: None)
        helper.whisper_helper_transcribe = _FakeCFunction(lambda *_args: 0)
        libs = {"libwhisper.so": engine, "libwhisper_helper.so": helper}

        monkeypatch.setattr(whisper, "find_libwhisper", lambda: "/l/libwhisper.so")
        monkeypatch.setattr(
            whisper, "find_libwhisper_helper", lambda: "/l/libwhisper_helper.so"
        )
        monkeypatch.setattr(whisper.ctypes, "CDLL", lambda path: libs[Path(path).name])

        with pytest.raises(ModelLoadFailed):
            WhisperContext(str(model_path), device="GPU")

    def test_failed_registration_never_loads_helper_or_model(
        self, tmp_path, monkeypatch
    ):
        from quoth import whisper

        events = []
        engine_path = tmp_path / "lib/libwhisper.so"
        helper_path = tmp_path / "lib/libwhisper_helper.so"
        model_path = tmp_path / "model.bin"
        model_path.touch()
        engine = _FakeBackendLibrary(events=events)

        def load(path):
            events.append(Path(path).name)
            if path == str(engine_path):
                return engine
            raise AssertionError("helper must not load after backend failure")

        monkeypatch.setattr(whisper, "find_libwhisper", lambda: str(engine_path))
        monkeypatch.setattr(whisper, "find_libwhisper_helper", lambda: str(helper_path))
        monkeypatch.setattr(whisper.ctypes, "CDLL", load)

        with pytest.raises(RuntimeError, match="CPU backend did not register"):
            WhisperContext(str(model_path))

        assert events == ["libwhisper.so", "backends"]


_CANCELLED = object()  # a decode pass the user cancels while it runs
# Contexts `_open` made, closed as each test ends: one the collector frees later logs
# its closing into whichever test is running then.
_opened: list = []


@pytest.fixture(autouse=True)
def _close_opened_contexts():
    yield
    while _opened:
        _opened.pop().close()


def _open(
    tmp_path, monkeypatch, device="CPU", *, openvino_ret=0, passes=((),), vad=True
):
    """Open a context over fake libraries; return it and the native calls made.

    Each decode pass replies with the next of `passes`: the segments it found, a
    failure's return code, or `_CANCELLED`.
    """
    from quoth import whisper

    calls = {"free": [], "transcribe": []}
    replies = list(passes)
    found = []
    model_path = tmp_path / "model.bin"
    model_path.touch()

    def init(*args):
        calls["init"] = args
        return 7  # the context pointer

    def init_openvino(*args):
        calls["openvino"] = args
        return openvino_ret

    def transcribe(*args):
        calls["transcribe"].append(args)
        reply = replies.pop(0)
        if reply is _CANCELLED:
            context.cancel()
        # The shim's abort callback stops whisper.cpp once the flag is raised.
        if args[-1]._obj.value:
            return -6
        if isinstance(reply, int):
            return reply
        found[:] = reply
        return 0

    engine = _FakeBackendLibrary(registered={"CPU"})
    engine.whisper_full_n_segments = _FakeCFunction(lambda _ctx: len(found))
    engine.whisper_full_get_segment_text = _FakeCFunction(lambda _ctx, i: found[i])
    engine.whisper_free = _FakeCFunction(calls["free"].append)
    engine.whisper_ctx_init_openvino_encoder = _FakeCFunction(init_openvino)
    helper = type("Helper", (), {})()
    helper.whisper_helper_init = _FakeCFunction(init)
    helper.whisper_helper_transcribe = _FakeCFunction(transcribe)
    libs = {"libwhisper.so": engine, "libwhisper_helper.so": helper}

    monkeypatch.setattr(whisper, "find_libwhisper", lambda: "/l/libwhisper.so")
    monkeypatch.setattr(
        whisper, "find_libwhisper_helper", lambda: "/l/libwhisper_helper.so"
    )
    monkeypatch.setattr(whisper.ctypes, "CDLL", lambda path: libs[Path(path).name])

    context = WhisperContext(
        str(model_path),
        device=device,
        openvino_model_path="/models/encoder.xml",
        cache_dir="/models/cache",
        vad_model_path="/models/vad.bin" if vad else None,
    )
    _opened.append(context)
    return context, calls


class TestNativeCalls:
    """What a context hands whisper.cpp for its device, and what it makes of replies."""

    @pytest.mark.parametrize(
        ("device", "use_gpu", "offloads_encoder"),
        [("CPU", False, False), ("GPU", True, False), ("NPU", False, True)],
    )
    def test_the_device_sets_the_gpu_flag_and_the_npu_encoder_offload(
        self, tmp_path, monkeypatch, device, use_gpu, offloads_encoder
    ):
        _, calls = _open(tmp_path, monkeypatch, device)
        assert calls["init"][1].value is use_gpu
        assert ("openvino" in calls) is offloads_encoder

    @pytest.mark.parametrize(("ret", "active"), [(0, True), (-1, False)])
    def test_openvino_active_says_whether_the_npu_offload_engaged(
        self, tmp_path, monkeypatch, ret, active
    ):
        # A failed init leaves whisper.cpp running the encoder on the CPU, silently.
        context, calls = _open(tmp_path, monkeypatch, "NPU", openvino_ret=ret)
        assert calls["openvino"] == (
            7,
            b"/models/encoder.xml",
            b"NPU",
            b"/models/cache",
        )
        assert context.openvino_active is active

    def test_the_transcript_is_the_segments_with_whitespace_normalised(
        self, tmp_path, monkeypatch
    ):
        # whisper.cpp starts each segment with a space.
        context, _ = _open(
            tmp_path, monkeypatch, passes=[[b" Hello there.", b" How are you? "]]
        )
        audio = np.zeros(16000, dtype=np.float32)
        assert context.transcribe_buffer(audio) == "Hello there. How are you?"

    def test_the_transcript_is_logged_only_at_trace(
        self, tmp_path, monkeypatch, caplog
    ):
        # Transcript text is logged only at TRACE, which no sink writes to disk. The
        # segment's leading space makes the raw and cleaned text differ.
        context, _ = _open(tmp_path, monkeypatch, passes=[[b" sentinel dictation "]])
        context.transcribe_buffer(np.zeros(16000, dtype=np.float32))
        heard = {r.levelname for r in caplog.records if "sentinel" in r.getMessage()}
        assert heard == {"TRACE"}

    def test_language_task_and_vad_settings_reach_the_decoder(
        self, tmp_path, monkeypatch
    ):
        context, calls = _open(tmp_path, monkeypatch, "CPU")
        context.transcribe_buffer(
            np.zeros(16000, dtype=np.float32), language="de", translate=True
        )
        lang, translate, _threads, vad, min_speech, pad = calls["transcribe"][0][3:9]
        # Not whisper.cpp's defaults: 250 ms minimum speech drops a short "yes", and a
        # 30 ms pad loses short words.
        assert (lang, translate.value, vad, min_speech.value, pad.value) == (
            b"de",
            True,
            b"/models/vad.bin",
            32,
            200,
        )

    def test_the_decoder_gets_a_thread_per_usable_cpu(self, tmp_path, monkeypatch):
        from quoth import whisper

        monkeypatch.setattr(whisper, "usable_cpus", lambda: 6)
        context, calls = _open(tmp_path, monkeypatch, "CPU")
        context.transcribe_buffer(np.zeros(16000, dtype=np.float32))
        assert calls["transcribe"][0][5].value == 6

    def test_close_frees_the_native_context_once(self, tmp_path, monkeypatch):
        context, calls = _open(tmp_path, monkeypatch, "CPU")
        context.close()
        context.close()  # as __del__ does after an explicit close
        assert calls["free"] == [7]
        assert not context.is_loaded


class TestFindLib:
    @pytest.mark.parametrize("from_source", [True, False])
    def test_local_build_is_only_used_from_source(
        self, tmp_path, monkeypatch, from_source
    ):
        from quoth import whisper

        library = tmp_path / "whisper.cpp/build/bin/libwhisper.so"
        library.parent.mkdir(parents=True)
        library.touch()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(whisper._native, "bundled_lib_dirs", lambda: [])
        monkeypatch.setattr(whisper._native, "running_from_source", lambda: from_source)

        assert whisper._find_lib("libwhisper.so") == (
            str(library) if from_source else None
        )

    def test_returns_first_existing_candidate(self, monkeypatch):

        from quoth import whisper

        # The first bundled dir lacks the library; the second and third both have it.
        monkeypatch.setattr(
            whisper._native,
            "bundled_lib_dirs",
            lambda: ["/opt/quoth", "/opt/quoth/lib", "/src/lib"],
        )
        monkeypatch.setattr(
            Path, "exists", lambda self: str(self).endswith("/lib/libwhisper.so")
        )
        assert whisper._find_lib("libwhisper.so") == "/opt/quoth/lib/libwhisper.so"

    def test_returns_none_when_no_candidate_exists(self, monkeypatch):

        from quoth import whisper

        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert whisper._find_lib("libwhisper.so") is None


class TestVadFallback:
    """VAD never costs a word: a loud buffer it calls silent is retried untrimmed."""

    def _audio(self, peak):
        buf = np.zeros(16000, dtype=np.float32)
        buf[0] = peak
        return buf

    def test_loud_buffer_vad_called_silent_is_retried_untrimmed(
        self, tmp_path, monkeypatch
    ):
        context, calls = _open(tmp_path, monkeypatch, passes=[[], [b" and"]])
        assert context.transcribe_buffer(self._audio(0.6)) == "and"
        vad_per_pass = [args[6] for args in calls["transcribe"]]
        assert vad_per_pass == [b"/models/vad.bin", None]

    def test_quiet_buffer_keeps_the_empty_result(self, tmp_path, monkeypatch):
        # Nothing was said. Returning empty is what kills the silence hallucination
        # and what makes the silent case fast; retrying here would undo both.
        context, calls = _open(tmp_path, monkeypatch, passes=[[]])
        assert context.transcribe_buffer(self._audio(0.01)) == ""
        assert len(calls["transcribe"]) == 1

    def test_speech_is_never_transcribed_twice(self, tmp_path, monkeypatch):
        context, calls = _open(tmp_path, monkeypatch, passes=[[b" hello there"]])
        assert context.transcribe_buffer(self._audio(0.9)) == "hello there"
        assert len(calls["transcribe"]) == 1

    def test_no_retry_when_vad_is_off(self, tmp_path, monkeypatch):
        # Empty is then whisper's own answer, so there is nothing to second-guess.
        context, calls = _open(tmp_path, monkeypatch, passes=[[]], vad=False)
        assert context.transcribe_buffer(self._audio(0.9)) == ""
        assert len(calls["transcribe"]) == 1

    def test_no_retry_after_a_cancel(self, tmp_path, monkeypatch):
        # The empty answer is the cancel's, and a retry would delay it by a decode.
        context, calls = _open(tmp_path, monkeypatch, passes=[_CANCELLED, [b" and"]])
        assert context.transcribe_buffer(self._audio(0.9)) == ""
        assert len(calls["transcribe"]) == 1


class TestAbort:
    """Cancelling stops the native decode: the inference lock holds until it returns."""

    def test_cancel_raises_the_flag_the_shim_polls(self, tmp_path, monkeypatch):
        context, calls = _open(tmp_path, monkeypatch)
        context.transcribe_buffer(np.zeros(16, dtype=np.float32))
        flag = calls["transcribe"][0][-1]._obj
        context.cancel()
        assert flag.value == 1

    def test_aborted_run_returns_nothing_instead_of_raising(
        self, tmp_path, monkeypatch
    ):
        context, _ = _open(tmp_path, monkeypatch, passes=[_CANCELLED])
        assert context.transcribe_buffer(np.zeros(16, dtype=np.float32)) == ""

    def test_failure_without_abort_still_raises(self, tmp_path, monkeypatch):
        context, _ = _open(tmp_path, monkeypatch, passes=[-6])
        with pytest.raises(RuntimeError):
            context.transcribe_buffer(np.zeros(16, dtype=np.float32))

    def test_each_transcription_starts_unaborted(self, tmp_path, monkeypatch):
        context, _ = _open(tmp_path, monkeypatch, passes=[[b" hello"]])
        context.cancel()
        assert context.transcribe_buffer(np.zeros(16, dtype=np.float32)) == "hello"

    def test_warmup_runs_one_second_of_float32_silence_without_vad(
        self, tmp_path, monkeypatch
    ):
        context, calls = _open(tmp_path, monkeypatch)
        context.cancel()  # left over from an earlier decode
        context.warm_up()

        [args] = calls["transcribe"]
        audio = np.ctypeslib.as_array(args[1], shape=(args[2].value,))
        assert audio.dtype == np.float32
        assert audio.shape == (16000,)
        assert not np.any(audio)
        assert (args[3], args[4].value, args[6]) == (b"auto", False, None)

    def test_aborted_warmup_raises_after_run(self, tmp_path, monkeypatch):
        context, _ = _open(tmp_path, monkeypatch, passes=[_CANCELLED])
        with pytest.raises(WarmupAborted, match="warmup aborted"):
            context.warm_up()
