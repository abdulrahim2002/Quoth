# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Audio capture diagnostic: records through the app's real pipeline.

Captures audio via the same `AudioPipeline` the app uses (warm-up, then the
GStreamer/PipeWire path), saves a WAV for listening, and prints quantitative metrics
so capture quality can be judged objectively alongside what you hear.

Usage:
    QUOTH_LOG_LEVEL=WARNING uv run python scripts/debug_audio.py [seconds]

Diagnostic tool (not shipped with the app). Reproduces the exact capture path,
so it's the first thing to reach for on an audio-quality regression.
"""

import sys
import wave

import numpy as np
from gi.repository import GLib

import quoth
import quoth.log  # noqa: F401 (configures loguru's sinks)
from quoth.audio import AudioPipeline

OUT_WAV = "/tmp/quoth_debug_capture.wav"
RECORD_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
TICK_MS = 100


def analyze(audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Print quantitative metrics for a mono float32 buffer."""
    n = len(audio)
    dur = n / sample_rate
    peak = float(np.abs(audio).max()) if n else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if n else 0.0
    dbfs_peak = 20 * np.log10(peak) if peak > 0 else -np.inf
    dbfs_rms = 20 * np.log10(rms) if rms > 0 else -np.inf
    clipped = int((np.abs(audio) > 0.99).sum())
    dc_offset = float(audio.mean()) if n else 0.0

    # Discontinuity check: large sample-to-sample jumps hint at dropped or glitched
    # chunks (clicks, choppiness). The threshold is generous.
    if n > 1:
        diffs = np.abs(np.diff(audio))
        max_jump = float(diffs.max())
        big_jumps = int((diffs > 0.5).sum())
    else:
        max_jump = 0.0
        big_jumps = 0

    # Silence ratio: fraction of 20ms frames below a quiet threshold.
    frame = int(0.02 * sample_rate)
    if n >= frame:
        frames = audio[: n // frame * frame].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        silent_ratio = float((frame_rms < 0.005).mean())
        active_rms = (
            float(frame_rms[frame_rms >= 0.005].mean())
            if (frame_rms >= 0.005).any()
            else 0.0
        )
    else:
        silent_ratio = 1.0
        active_rms = 0.0

    print("--- QUANTITATIVE METRICS ---")
    print(f"  duration      : {dur:.2f} s  ({n} samples @ {sample_rate} Hz)")
    print(f"  peak          : {peak:.4f}  ({dbfs_peak:+.1f} dBFS)")
    print(f"  rms (overall) : {rms:.4f}  ({dbfs_rms:+.1f} dBFS)")
    print(f"  rms (active)  : {active_rms:.4f}   ← speech-frame loudness")
    print(f"  silence ratio : {silent_ratio * 100:.0f}%   ← share of quiet frames")
    print(f"  clipped       : {clipped} samples (|x| > 0.99)")
    print(f"  dc offset     : {dc_offset:+.5f}   (should be ~0)")
    print(
        f"  max jump      : {max_jump:.3f}   big jumps(>0.5): {big_jumps}  ← clicks/dropouts"
    )
    print()
    print("--- INTERPRETATION HINTS ---")
    if peak < 0.05:
        print("  ! Very low level: mic gain likely too low for reliable ASR.")
    if clipped > 0:
        print("  ! Clipping present: input too hot / distorted.")
    if abs(dc_offset) > 0.01:
        print("  ! Significant DC offset: capture path may be wrong.")
    if big_jumps > 5:
        print("  ! Discontinuities: possible dropped chunks / choppy audio.")
    if silent_ratio > 0.7:
        print("  ! Mostly silent: gate timing or wrong source?")
    if peak >= 0.05 and clipped == 0 and big_jumps <= 5 and silent_ratio <= 0.7:
        print("  Looks clean on paper. If it still mis-transcribes, suspect")
        print("  content (accent/model) rather than capture.")


def main() -> None:
    chunks: list[np.ndarray] = []
    pipe = AudioPipeline(
        mic_source="auto",
        on_audio_buffer=chunks.append,
        on_level=lambda _a: None,
    )
    loop = GLib.MainLoop()
    state = {"tick": 0}
    warm_ticks = int(1500 / TICK_MS)  # ~1.5s to warm before recording
    record_ticks = int(RECORD_SECONDS * 1000 / TICK_MS)

    def drive() -> bool:
        t = state["tick"]
        if t == 0:
            pipe.warm_up()
        elif t == warm_ticks:
            chunks.clear()
            pipe.start()
            print(
                f"\n>>> SPEAK NOW (recording {RECORD_SECONDS:.0f}s) <<<\n", flush=True
            )
        elif t == warm_ticks + record_ticks:
            pipe.stop()
            if not chunks:
                print("NO AUDIO CAPTURED")
                pipe.destroy()
                loop.quit()
                return False
            audio = np.concatenate(chunks).astype(np.float32)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(OUT_WAV, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            print(f"saved: {OUT_WAV}\n")
            analyze(audio)
            print(f"\nListen back with:  pw-play {OUT_WAV}")
            pipe.destroy()
            loop.quit()
            return False
        state["tick"] += 1
        return True

    GLib.timeout_add(TICK_MS, drive)
    loop.run()


if __name__ == "__main__":
    main()
