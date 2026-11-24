#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Verify an installed Quoth package can transcribe the pinned JFK fixture.

The decode takes the default path, VAD on with the package's own Silero weights, and a
first decode must stop when the abort flag is set from another thread.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

EXPECTED = (
    "ask not what your country can do for you ask what you can do for your country"
)


def _mapped_cpu_modules(maps_path: Path = Path("/proc/self/maps")) -> list[Path]:
    modules = set()
    for line in maps_path.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = Path(fields[5])
            if path.name.startswith("libggml-cpu-") and path.name.endswith(".so"):
                modules.add(path)
    return sorted(modules)


def _selected_cpu_module(package_lib: Path) -> Path:
    modules = _mapped_cpu_modules()
    if len(modules) != 1:
        names = ", ".join(os.fspath(path) for path in modules) or "none"
        raise RuntimeError(f"expected one mapped GGML CPU module, found {names}")
    module = modules[0].resolve(strict=True)
    if not module.is_relative_to(package_lib.resolve(strict=True)):
        raise RuntimeError(f"mapped GGML CPU module is outside the package: {module}")
    return module


def _normalized(text: str) -> str:
    return " ".join(
        "".join(
            character if character.isalnum() or character.isspace() else " "
            for character in text.casefold()
        ).split()
    )


def _read_audio(path: str) -> np.ndarray:
    with wave.open(path, "rb") as audio:
        if (
            audio.getcomptype() != "NONE"
            or audio.getnchannels() != 1
            or audio.getframerate() != 16_000
            or audio.getsampwidth() != 2
        ):
            raise ValueError(
                "audio must be uncompressed mono 16 kHz signed 16-bit PCM WAV"
            )
        frames = audio.readframes(audio.getnframes())

    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _aborted_decode(context, samples: np.ndarray) -> str:
    """Decode while another thread keeps setting the abort flag; return the text.

    The decode clears the flag on entry, so the flag is set until the decode returns.
    """
    done = threading.Event()

    def cancel_until_done() -> None:
        while not done.wait(0.01):
            context.cancel()

    canceller = threading.Thread(target=cancel_until_done)
    canceller.start()
    try:
        return context.transcribe_buffer(samples, language="en")
    finally:
        done.set()
        canceller.join()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a GGML Whisper model")
    parser.add_argument("--audio", required=True, help="Path to the JFK WAV fixture")
    args = parser.parse_args()

    try:
        import quoth
        from quoth import whisper
        from quoth.models import ModelManager

        samples = _read_audio(args.audio)
        # With no user copy to prefer, the lookup can only find the bundled weights.
        with tempfile.TemporaryDirectory() as empty:
            vad = ModelManager(Path(empty)).vad_model_path()
        if vad is None:
            raise RuntimeError("the package ships no VAD weights")
        # No untrimmed retry: the phrase must survive VAD itself.
        whisper.VAD_FALLBACK_PEAK = math.inf
        context = whisper.WhisperContext(
            args.model, device="CPU", vad_model_path=os.fspath(vad)
        )
        try:
            aborted = _aborted_decode(context, samples)
            transcript = context.transcribe_buffer(samples, language="en")
            cpu_module = _selected_cpu_module(Path(quoth.__file__).parent / "lib")
        finally:
            context.close()
    except Exception as error:
        print(f"transcription check failed: {error}", file=sys.stderr)
        return 1

    if aborted:
        print(
            "transcription check failed: the abort did not stop a decode",
            file=sys.stderr,
        )
        return 1

    if _normalized(EXPECTED) not in _normalized(transcript):
        print(
            "transcription check failed: expected JFK phrase was not recognized",
            file=sys.stderr,
        )
        return 1

    print(f"transcription check passed with VAD and abort; CPU module: {cpu_module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
