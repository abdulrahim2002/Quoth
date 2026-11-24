# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Quoth recording-feedback earcons: synthesised, licence-free, reproducible.

The two cues in `src/quoth/sounds/` are generated here from pure math and written as
16-bit PCM WAV with the stdlib `wave`, so they carry no third-party licensing, stay
tiny, and reproduce exactly on any machine. They are a matched pair: warm low-mid
plucks (F4 349 Hz, A4 440 Hz, plus a quiet sub-octave for body) with a fast attack and
exponential decay rather than a flat beep; start rises F4 -> A4 and stop settles
A4 -> F4. Each lasts about 170 ms, long enough to register and short enough to feel
instant, and a few ms of true silence pad both ends so playback starts and ends at
zero amplitude and never clicks. GStreamer's `wavparse` (always present) decodes
them, and they ship as package data like `themes/`.

Usage:
    python scripts/generate_sounds.py          # (re)write start.wav and stop.wav
    python scripts/generate_sounds.py --info   # just report the current files
"""

from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "src" / "quoth" / "sounds"
_RATE = 48000

# Loudness of the written peak. ~-8.9 dBFS: audible over dictation without being harsh
# (the warm frequencies keep it pleasant at this level); user volume scales it down.
_PEAK = 0.36
_PAD_MS = 4  # leading/trailing digital silence so sink startup lands in silence
_ATTACK_MS = 4  # short rise so the onset is soft, not a click, without dulling the tick
_WARMTH = 0.28  # amount of sub-octave mixed in for body
_DECAY = 3.2  # exponential decay rate over the note body (higher = shorter tail)

# (frequency Hz, duration ms) per note. F4 and A4: warm, no highs.
_START_NOTES = ((349.0, 70), (440.0, 90))  # rising: opening
_STOP_NOTES = ((440.0, 70), (349.0, 95))  # falling: closing


def _pluck(freq: float, ms: int) -> list[float]:
    """Return one soft percussive note: a fast attack, then an exponential decay.

    A quiet sub-octave adds warmth, and the decay brings the note near zero by its end.
    """
    n = int(_RATE * ms / 1000)
    a = int(_RATE * _ATTACK_MS / 1000)
    out = []
    for i in range(n):
        env = (i / a) if i < a else math.exp(-_DECAY * (i - a) / (n - a))
        tone = math.sin(2 * math.pi * freq * i / _RATE)
        sub = _WARMTH * math.sin(2 * math.pi * (freq / 2) * i / _RATE)
        out.append(env * (tone + sub))
    return out


def _silence(ms: int) -> list[float]:
    return [0.0] * int(_RATE * ms / 1000)


def _cue(notes: tuple[tuple[float, int], ...]) -> list[float]:
    body: list[float] = []
    for freq, ms in notes:
        body += _pluck(freq, ms)
    peak = max((abs(x) for x in body), default=1.0) or 1.0
    body = [x / peak * _PEAK for x in body]
    return _silence(_PAD_MS) + body + _silence(_PAD_MS)


def _write_wav(name: str, samples: list[float]) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    frames = b"".join(
        int(max(-32768, min(32767, round(x * 32767)))).to_bytes(
            2, "little", signed=True
        )
        for x in samples
    )
    with wave.open(str(_OUT / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_RATE)
        w.writeframes(frames)
    dur_ms = len(samples) / _RATE * 1000
    print(f"  {name}  {dur_ms:.0f}ms  ends {samples[0]:.4f} / {samples[-1]:.4f}")


def _synthesize() -> None:
    print("Synthesizing earcons:")
    _write_wav("start.wav", _cue(_START_NOTES))
    _write_wav("stop.wav", _cue(_STOP_NOTES))
    print("Done.")


def _info() -> None:
    print("Earcon files:")
    for name in ("start.wav", "stop.wav"):
        p = _OUT / name
        if not p.exists():
            print(f"  {name}  MISSING")
            continue
        with wave.open(str(p)) as w:
            dur_ms = w.getnframes() / w.getframerate() * 1000
            ch = w.getnchannels()
            rate = w.getframerate()
            bits = w.getsampwidth() * 8
        size_kb = p.stat().st_size / 1024
        print(f"  {name}  {dur_ms:.0f}ms  {ch}ch  {rate}Hz  {bits}bit  {size_kb:.0f}KB")


def main() -> None:
    if "--info" in sys.argv[1:]:
        _info()
    else:
        _synthesize()


if __name__ == "__main__":
    main()
