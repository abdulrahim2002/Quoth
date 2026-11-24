# Quoth

Offline, private, instant voice dictation for Linux. Press a key, speak, text appears in the focused window. Runs on the CPU everywhere, and on a Vulkan GPU or an Intel NPU where one is present.

## Project Identity

- **Repository**: https://github.com/abullard1/Quoth
- **License**: GPL-3.0-or-later
- **Python**: 3.12+ (use `uv` for all tooling)
- **API floors**: GTK 4.14, libadwaita 1.5 and GLib 2.80, as on Ubuntu 24.04; use no newer GTK, libadwaita or GLib API without a version or feature check
- **Contributing**: source setup, checks, test conventions, commit style and the AI-assistance rule are in `CONTRIBUTING.md`

## Behaviour Rules

- **Read before writing.** Open the file, understand existing code, then edit. Don't guess function signatures, class names, or import paths.
- **Surgical changes only.** Touch only what the task requires. Don't refactor adjacent code, don't "fix" formatting elsewhere, don't add type hints to unchanged functions. If you see unrelated dead code, mention it; don't delete it.
- **Goal-driven execution.** Transform every task into a verifiable goal. State success criteria before implementing; prove they're met after. Don't claim "it should work"; test it.
- **Simplicity first.** Minimum code to solve the problem. No abstractions for single-use code, no future-proofing, no error handling for impossible states. Clean up unused imports you create, but don't purge pre-existing dead code.
- **Verify, don't assume.** After changes run `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run lint-imports`, plus `uv run zizmor .` when a workflow changed; run `uv run quoth --help` when CLI args change. Without a display `uv run pytest` skips the widget tests; the headless suite command under Development Commands runs them.
- **No guessing.** Don't invent API versions, package names, file paths, or config keys. Read the source to confirm.
- **One file per scope.** A nested `AGENTS.md` closer to the files being edited wins over this root file; keep this root file to what is true everywhere in the repo.

## Security

- **Never commit secrets**: no tokens, API keys, credentials, or signing keys in any file, commit message, or log. They are provisioned outside the repository (CI secrets, local keyrings).
- **Keep this file public-safe.** It ships in the public repository: no internal hostnames, private paths, personal details, or unpublished release material here — internal working docs live in private files that are not exported.

## Architecture

The import contract in `pyproject.toml` sorts every module into six tiers by meaning (entry, app, ui, services, adapters, domain), and `uv run lint-imports` enforces them; a new module goes into the tier its meaning names.

- **Composition**: `app.py` (`QuothApp`) builds every component and owns their lifecycle. `recording.py` (`RecordingController`) runs the `IDLE` → `RECORDING` → `TRANSCRIBING` → `IDLE` state machine and reaches the main thread through the injected scheduler (`scheduler.py`).
- **Triggers**: `listener.py` (native evdev hotkey), `portal_shortcuts.py` (the Flatpak's GlobalShortcuts session) and `ipc.py` (`quoth --toggle` and the `dev.quoth.Dictation1` D-Bus interface, which never carries text) report presses; `trigger.py` is the one rule for what each does. `hotkey_capture.py` previews a native key in GTK.
- **Dictation**: `audio.py` (`AudioPipeline`, GStreamer capture), `whisper.py` (whisper.cpp through ctypes and the C shim), `typer.py` (clipboard paste).
- **Settings**: `data/dev.quoth.Quoth.gschema.xml` is the one source of keys, defaults and valid values, and `config.py` (`Config`) reads and writes them. `settings.py` holds `ModelSwitcher`, which owns the app's one pending model or device change (it may need a download first), and `SettingsController`, the window's logic over it; `settings_window.py`, with its `engine_section.py`, `overlay_section.py` and `skin_section.py` groups, is the GTK view.
- **Devices and models**: `hardware.py` detects devices, `accel.py` (`backend_for`) is the one source of what `CPU`/`GPU`/`NPU` mean, and everything else derives (never re-test `device in (...)`). `resolution.py` composes the effective backend from detection, the accel rules, the model catalog (`models.py`) and `calibration.py`'s measured per-model ranking; `app.py`, `settings.py`, `engine_section.py`, `onboarding.py`, `try_it.py` and `prepare.py` use it. `benchmark_dialog.py` is the staged view of a calibration run. `downloads.py` is the one owner of model downloads.
- **Overlay**: `overlay.py` shows the themed pill or, with the `overlay-style` key set to `orb`, the audio-reactive orb; `pill_view.py` is the theme-aware view layer (`draw_pill`, the preview widgets, `OrbDirector`) over the `pill_render.py` and `orb.py` leaves. `presentation.py` decides whether the overlay can show; `tray.py` serves the tray icon over D-Bus.
- **Onboarding**: `onboarding.py`, with the guided dictation in `try_it.py`, the calibration step in `optimize.py`, its background in `backdrop.py` and its motion in `animation.py` over the `lottie.py`/ThorVG seam. It has no model step: the default model starts downloading as it opens.
- **Appearance**: overlay themes live in `theme.py`. App skins are separate and colour-only: `skin.py` owns their format, presets and shelf, `brand.py` renders the one CSS provider, and `skin_runtime.py` reloads the active skin when its file changes.
- **Privacy edges**: `crash_reports.py` (opt-in) is the only module that touches `sentry_sdk`; `log.py` keeps transcript text off disk.

**Flow**: hotkey press or release (evdev, the GlobalShortcuts portal, `quoth --toggle`, or a script's `--start`/`--stop`) → `trigger.hotkey_action`, applied by `app._on_hotkey` → `RecordingController` → `AudioPipeline` (`pulsesrc`, else `pipewiresrc`, else `autoaudiosrc`) → float32 16 kHz mono buffer → whisper.cpp on a background thread, cooperatively cancellable → cleaned text → `typer.py`: `wl-copy`, then a Shift+Insert chord from ydotool or wtype.

## Critical Invariants (DO NOT BREAK)

These guard against crashes, memory corruption and lost dictations.

- **Never hide the overlay with `set_visible(False)` while layer-shell is active**: it crashes COSMIC with a "Broken pipe". Use `hide_pill()`, which renders transparent instead.
- **Never create/destroy the GStreamer pipeline between recordings**: create once, destroy on shutdown. Gate delivery with the software `_recording` flag. While neither a recording nor the level meter needs it, a Pulse or auto source is stopped (NULL), closing the microphone; pipewiresrc stays PLAYING, since its PAUSED→PLAYING resume is unreliable. Two exceptions, both while idle: a microphone change rebuilds it through `RecordingController.apply_mic_source()`, deferred until the engine is idle; and `warm_up()`, at startup and after such a change, may rebuild the cold pipeline to beat pipewiresrc's first-buffer race (it runs only when the selected source is pipewiresrc, which the audio source seam decides) and self-aborts once `_recording` is true.
- **Never call whisper struct-by-value functions from ctypes**: that needs a Python copy of the struct's layout, which whisper.h changes between releases, and a stale copy corrupts memory without an error. Always go through `libwhisper_helper.so` (the C shim in `lib/`), which builds those structs against the header.
- **Touch GTK only on the main thread**: GTK4 is single-threaded. Background threads (evdev, GStreamer, whisper) hand UI work to it: GLib-free code such as `RecordingController` through the injected scheduler (`scheduler.py`, whose `GLibScheduler` wraps `GLib.idle_add()`), GTK code through `GLib.idle_add()`.
- **Guard the audio buffer with `RecordingController._buffer_lock`**: the GStreamer thread appends to it and the main thread drains it.
- **Drive recording only through `RecordingController`** (`start`/`stop`/`cancel`): its `_transition_to()` validates state moves. Don't reassign state outside it.
- **Keep the small delay before text injection** (`recording.TYPE_DELAY_MS`, 50 ms): without it the injected keystrokes can be dropped.

## Conventions

- **Type hints**: Python 3.12+ style (`str | None`, `list[str]`); never `Optional`/`List`.
- **Comments**: explain what the code does and why it must be this way, never how it came to be. **No workflow leakage**: no narration of the change, the review, the alternatives considered, or the conversation that produced it; no issue, PR or decision-record references; state the reason inline instead. Write for the next reader, not the reviewer. Keep them short: the median block here is **2 lines**. No em dashes.
- **Private members**: underscore prefix (`_config`, `_state`).
- **Logging**: loguru. `INFO` milestones, `DEBUG` lifecycle detail, `WARNING` recoverable degradation, `ERROR`/`exception` failures; transcript text only ever at `TRACE`, which no sink writes to disk; lazy `{}` placeholders, never f-strings; errors in background threads and scheduled callbacks must be logged.
- **XDG base dirs**: name every location through `paths.py`, which applies the spec's rules to `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_DATA_HOME` and `XDG_CACHE_HOME`.
- **GTK/Adw**: no Glade/XML UI files; build UI programmatically. `Adw.PreferencesWindow` for settings; `Adw.StatusPage`, or `brand.status_column` when the visual is a widget, for first-launch pages. `gi.require_version()` before importing any namespace other than GLib, GObject and Gio.
- **Error handling**: graceful fallbacks (layer-shell unavailable → pill suppressed, state via tray and notifications; a device that can't run the model → the next one, down to the CPU). `logger.exception()` on failure; user-facing toasts for recoverable errors.

## Native Dependencies

- **whisper.cpp**: `libwhisper.so` and the C shim `libwhisper_helper.so`, built into `lib/` by `make build` (a source checkout also finds `~/whisper.cpp/build/bin/`). The built `.so` files are gitignored.
- **GStreamer**: the base and good plugins plus PipeWire's plugin, for capture in `audio.py` as float32 16 kHz mono.
- **gtk4-layer-shell**: the overlay, where the compositor grants layer-shell (`presentation.py`). Elsewhere the pill is suppressed and the tray carries state, never a regular window, which would take focus and break the paste.
- **Paste**: `wl-copy` (wl-clipboard) puts the text on the clipboard, then `ydotool` (kernel `uinput`, needing `/dev/uinput` access; 1.x also needs `ydotoold`, which Quoth starts unless one runs) or `wtype` (the compositor's virtual keyboard) sends Shift+Insert. Neither types text. Native installs try ydotool first; the Flatpak does once the host's `/dev/uinput` is writable.
- **OpenVINO**: optional; runs the encoder on an Intel NPU. GPU acceleration is Vulkan through GGML.
- **ThorVG** (`libthorvg`): renders onboarding's Lottie animations, built into `lib/` by `make vendor-thorvg`. Without it the animations draw nothing and onboarding still works.
- **libportal**: the Flatpak's GlobalShortcuts session and its Background portal request for launch at login; the Flatpak bundles it with GIR support.

## Development Commands

```bash
uv run quoth                       # Run the app
uv run quoth -s                    # Open the Settings window
uv run quoth -q                    # Quit running instance
uv run quoth -t                    # Toggle dictation, starting Quoth if needed
uv run quoth --status              # Print the running instance's state (also --start/--stop/--cancel)
uv run pytest                      # Run the test suite
dbus-run-session -- xvfb-run -a env GDK_BACKEND=x11 GSK_RENDERER=cairo GTK_A11Y=test \
  uv run --locked --group icons pytest -ra --cov=quoth  # The headless suite as CI runs it
uv run ruff check .                # Lint (matches CI)
uv run ruff format --check .       # Check formatting (matches CI)
uv run lint-imports                # Enforce the layered import contract
make schemas                       # Compile the settings schema for source runs; rerun after editing it
QUOTH_LOG_LEVEL=DEBUG uv run quoth # Verbose console logging
```
