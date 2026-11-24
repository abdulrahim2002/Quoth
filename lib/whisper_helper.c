/*
 * A thin C shim that lets Quoth's ctypes bindings drive whisper.cpp.
 *
 * whisper_full_default_params() and whisper_full() pass large structs by value.
 * Calling them from ctypes needs a Python copy of each struct's layout, which
 * whisper.h changes between releases, and a stale copy corrupts memory without
 * an error. These wrappers build the structs in C against the header, so Python
 * only ever uses pointer-based functions via ctypes.
 *
 * Build: see build_helper.sh
 */

#include "whisper.h"
#include <string.h>
#include <stdbool.h>

/* Load a model with whisper's default context params and the given use_gpu.
 * The params struct is returned and passed by value, so it never leaves C. */
struct whisper_context * whisper_helper_init(
    const char *model_path,
    bool use_gpu
) {
    struct whisper_context_params ctx_params = whisper_context_default_params();
    ctx_params.use_gpu = use_gpu;
    return whisper_init_from_file_with_params(model_path, ctx_params);
}

/* whisper checks this after each encoder pass and after each decoded token; a
 * set flag makes whisper_full stop there and return non-zero. */
static bool abort_when_flagged(void *user_data) {
    return *(const volatile int *)user_data != 0;
}

/* Transcribe samples in one shot; Python never touches the params struct.
 * Returns whisper_full()'s result, 0 on success.
 *
 * translate: run Whisper's translate task, which only ever outputs English
 * (a Whisper limitation) regardless of the source language.
 *
 * n_threads: when > 0, the CPU thread count, overriding whisper.cpp's default
 * (which caps at 4). It sizes what GGML runs on the CPU: the decoder on every
 * backend but Vulkan, and the whole model on the CPU device.
 *
 * vad_model_path: when non-NULL and non-empty, whisper.cpp runs its built-in
 * Silero VAD from that model file and transcribes only the detected speech,
 * trimming silence that would otherwise cost time and provoke hallucination.
 * NULL or empty disables it. whisper_vad_default_params() returns a struct by
 * value, so it is called here in C, never from ctypes.
 *
 * vad_min_speech_ms: when > 0, the shortest speech segment VAD keeps. The
 * upstream default suits live-stream endpointing; a key-delimited recording
 * needs a smaller value or short words are discarded.
 *
 * vad_speech_pad_ms: when > 0, the audio in ms kept on each side of a speech
 * segment. The upstream default of 30 ms loses short words.
 *
 * abort_flag: when non-NULL, an int the caller may set from any thread to stop
 * the run early on cancellation or shutdown. */
int whisper_helper_transcribe(
    struct whisper_context *ctx,
    const float *samples,
    int n_samples,
    const char *language,
    bool translate,
    int n_threads,
    const char *vad_model_path,
    int vad_min_speech_ms,
    int vad_speech_pad_ms,
    const volatile int *abort_flag
) {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    if (abort_flag) {
        params.abort_callback           = abort_when_flagged;
        params.abort_callback_user_data = (void *)abort_flag;
    }
    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_timestamps = false;
    params.print_special    = false;
    params.single_segment   = false;
    params.translate        = translate;
    /* Each dictation is an independent utterance, so no cross-utterance token
     * carry-over. It is already the default; pinning it guards an upstream flip. */
    params.no_context       = true;
    /* Suppress non-speech tokens (music/laughter/sound markers) at the logit level,
     * so they never consume decode budget or anchor a hallucination. */
    params.suppress_nst     = true;

    if (n_threads > 0) {
        params.n_threads = n_threads;
    }

    if (vad_model_path && strlen(vad_model_path) > 0) {
        params.vad            = true;
        params.vad_model_path = vad_model_path;
        params.vad_params     = whisper_vad_default_params();
        if (vad_min_speech_ms > 0) {
            params.vad_params.min_speech_duration_ms = vad_min_speech_ms;
        }
        if (vad_speech_pad_ms > 0) {
            params.vad_params.speech_pad_ms = vad_speech_pad_ms;
        }
    }

    /* Forward the language verbatim, including "auto". whisper.cpp treats
     * "auto" (or an empty string) as "detect the language, then transcribe", so
     * passing it through is what makes auto-detect work. Dropping it would leave
     * params.language at whisper's default ("en"), silently forcing English. */
    if (language && strlen(language) > 0) {
        params.language = language;
    }

    return whisper_full(ctx, params, samples, n_samples);
}
