# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The floating overlay: the themed waveform pill, or the opt-in orb.

One window carries the overlay's states and its entry and exit animations; colours,
geometry, position, font and pacing come from the `Theme`, and drawing goes through
`pill_view`. Recording shows the status dot, waveform and timer; transcribing decays
the bars, expands them to fill the pill and animates a travelling sine wave.

gtk4-layer-shell positions the surface where the compositor grants layer-shell. Once
shown, the surface stays mapped and hides by drawing transparent, because
`set_visible(False)` on a layer-shell surface crashes COSMIC with a "Broken pipe".
Where the compositor does not grant layer-shell (GNOME; a sandboxed Flatpak on
COSMIC, Sway, Hyprland or niri) the pill stays inert and state shows via the tray and
notifications instead (see `presentation`).
"""

import math
import time

import cairo

# GI_TYPELIB_PATH and the library preload are set up in quoth/__init__.py, which
# must run before gi is first imported.
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Pango", "1.0")
from gi.repository import Gdk, GLib, Gtk, Pango
from loguru import logger

from quoth import environment, presentation
from quoth.easing import ease
from quoth.pill_render import (
    LABEL_LEFT_X,
    LABEL_PAD_X,
    NUM_BARS,
    PHASE_RATE,
    TRANSITION_TOTAL_S,
    LevelTrace,
    draw_background,
    draw_label,
    format_timer,
    ink_extents,
    label_layout,
)
from quoth.pill_view import (
    OrbDirector,
    OverlayState,
    PillFrame,
    animations_on,
    draw_pill,
    draw_status_dot,
    frame_dt,
    make_orb,
)
from quoth.scheduler import guarded
from quoth.theme import DEFAULT_THEME, Theme, anchor_to_edges

# gtk4-layer-shell supplies the OVERLAY-layer, no-focus surface the pill sits on.
# Whether it is usable is decided only by `presentation.overlay_available()`; this
# import just provides the symbol for the positioning calls.
try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell
except (ImportError, ValueError):
    Gtk4LayerShell = None

MIN_WAVE_DISPLAY_S = 0.45  # minimum wave play before allowing hide (kept snappy)

# The orb style's square surface, roughly twice the pill's height. The orb carries no
# text, so it needs no extra width.
ORB_SURFACE = 96

# Libadwaita's back curves peak just over 1.01 after the overlay's 10% scale mix.
_BACK_SCALE_MAX = 1.011
_GINT_MIN = -(2**31)
# Each side of the visible input region, larger than any overlay surface.
_REGION_SPAN = 1 << 16

# Error auto-dismiss (Material 3's 4-10s band for action-less transients).
ERROR_HIDE_MS = 5000


class WaveformPill(Gtk.Window):
    """The floating overlay window, drawing the pill or the orb for each state.

    Call it from the GTK main thread only. Without layer-shell every `show_*` method
    is a no-op.
    """

    def __init__(self, theme: Theme = DEFAULT_THEME, style: str = "pill"):
        super().__init__()
        self._theme = theme
        self._style = style
        # The orb style draws the audio-reactive theme orb instead of the pill; its
        # director maps state and voice level onto the orb.
        self._orb, self._director = self._style_objects(style)
        self._motion_enabled = True  # system-level gate (reduced motion)
        self.set_decorated(False)
        self.set_default_size(*self._surface_size())
        self.set_resizable(False)
        self.set_title("Quoth")
        # A window group of its own: a modal window's grab (Settings, a file chooser)
        # covers only its group, so it never swallows the overlay's cancel click.
        Gtk.WindowGroup().add_window(self)

        # State
        self._state: OverlayState = OverlayState.IDLE
        self._start_time = 0.0
        self._trace = LevelTrace()
        self._tick_id = None
        self._dot_phase = 0.0
        self._last_frame_us: int | None = None
        self._on_cancel = None
        self._error_message = ""
        self._hint_message = ""
        self._error_guidance: str | None = None
        self._hint_guidance: str | None = None
        self._hide_timeout_id = None

        # Entry/Exit animation state
        self._is_entering = False
        self._entry_start = 0.0
        self._is_exiting = False
        self._exit_start = 0.0

        # Transcribing transition state
        self._transition_start = 0.0
        self._hide_pending = False
        self._frozen_amplitudes = [0.0] * NUM_BARS

        # Transparent background (scoped to this overlay only)
        self.add_css_class("quoth-pill")
        css = Gtk.CssProvider()
        css.load_from_string("window.quoth-pill { background: transparent; }")
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Layer-shell positioning where the compositor grants it, which only
        # `presentation.overlay_available()` decides.
        self._has_layer_shell = presentation.overlay_available()
        if self._has_layer_shell:
            Gtk4LayerShell.init_for_window(self)
            # Named, so compositor layer rules (blur, no animation) can target the
            # pill and it is not an anonymous "gtk4-layer-shell" surface in listings.
            Gtk4LayerShell.set_namespace(self, "quoth")
            Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
            Gtk4LayerShell.set_exclusive_zone(self, -1)
            Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
            self._apply_position()
            logger.debug(
                "Layer shell active: overlay on OVERLAY layer (anchor={}, style={})",
                self._theme.position.anchor,
                self._style,
            )
        else:
            # A regular Wayland toplevel cannot refuse keyboard focus, so showing one
            # would steal focus and the paste would land on the pill. It stays inert
            # and state shows via the tray and notifications.
            if Gtk4LayerShell is None:
                reason = "gtk4-layer-shell not installed"
            elif environment.in_flatpak():
                reason = "the compositor does not grant layer-shell to the Flatpak"
            else:
                reason = "the compositor does not grant layer-shell"
            logger.warning(
                "Overlay unavailable ({}); state shown via tray and notifications",
                reason,
            )

        self._canvas = Gtk.DrawingArea()
        self._canvas.set_draw_func(self._draw)
        self._canvas.connect("resize", self._on_resize)
        w, h = self._surface_size()
        self._canvas.set_content_width(w)
        self._canvas.set_content_height(h)

        # Click to cancel, and a hand cursor to say so
        click = Gtk.GestureClick()
        click.connect("released", self._on_click)
        self._canvas.add_controller(click)
        self._canvas.set_cursor_from_name("pointer")

        self.set_child(self._canvas)

        # The surface stays mapped while IDLE (see hide_pill()), and a mapped surface
        # takes clicks over its whole rectangle by default. Recompute the input region
        # on every map so IDLE is click-through.
        self.connect("map", lambda *_: self._apply_input_region())

    # --- Theming ---

    def _style_objects(self, style: str) -> tuple:
        """Return the orb and its director for `style`, or (None, None) for the pill."""
        if style == "orb":
            return make_orb(self._theme), OrbDirector(
                self._theme.animation.duration_scale
            )
        return None, None

    def set_style(self, style: str):
        """Switch the visualisation in place (the orb switch's hot-apply).

        The surface stays mapped throughout, as on a theme hot-reload, so the
        layer-shell unmap that crashes COSMIC is never reached. On a switch
        mid-state, the orb morphs in from neutral and the pill draws from its trace.
        """
        if style == self._style:
            return
        self._style = style
        self._orb, self._director = self._style_objects(style)
        self._sync_surface()
        self._canvas.queue_draw()

    def _content_size(self) -> tuple[int, int]:
        """Return the authored drawing size for the active style."""
        if self._style == "orb":
            return ORB_SURFACE, ORB_SURFACE
        g = self._theme.geometry
        return g.width, g.height

    def _surface_padding(self) -> tuple[int, int]:
        """Return the transparent room per side for a back curve's scale overshoot."""
        anim = self._theme.animation
        if not self._animations_on() or not any(
            spec.easing.endswith("-back") for spec in (anim.entry, anim.exit)
        ):
            return 0, 0
        width, height = self._content_size()
        overshoot = _BACK_SCALE_MAX - 1.0
        return math.ceil(width * overshoot / 2), math.ceil(height * overshoot / 2)

    def _surface_size(self) -> tuple[int, int]:
        width, height = self._content_size()
        pad_x, pad_y = self._surface_padding()
        return width + 2 * pad_x, height + 2 * pad_y

    def _on_resize(self, _area, width: int, height: int) -> None:
        self._apply_position((width, height))

    def _sync_surface(self) -> None:
        """Apply active style, motion, and theme geometry to the mapped surface."""
        width, height = self._surface_size()
        self._canvas.set_content_width(width)
        self._canvas.set_content_height(height)
        self.set_default_size(width, height)
        self._apply_position()
        self._apply_input_region()

    def apply_theme(self, theme: Theme):
        """Swap the active theme (hot-reload entry point) and re-render."""
        self._theme = theme
        if self._style == "orb":
            # A new palette is a new orb; the director keeps morphing from where it
            # stands.
            self._orb = make_orb(theme)
            self._director.set_duration_scale(theme.animation.duration_scale)
        self._reconcile_motion()
        self._sync_surface()
        self._canvas.queue_draw()

    def set_motion_enabled(self, enabled: bool):
        """Set the system-level decorative-motion gate (reduced-motion preference)."""
        if enabled != self._motion_enabled:
            self._motion_enabled = enabled
            self._reconcile_motion()
            self._sync_surface()

    def _reconcile_motion(self) -> None:
        """Settle interrupted decoration while keeping causal clocks and data live."""
        if self._animations_on():
            if self._state != OverlayState.IDLE:
                self._start_animation()
            return
        now = time.monotonic()
        self._is_entering = False
        if self._state == OverlayState.TRANSCRIBING:
            self._transition_start = now - self._scaled(TRANSITION_TOTAL_S)
        if self._hide_pending or self._is_exiting:
            self._do_hide()
            return
        if self._director is not None and self._state != OverlayState.IDLE:
            self._orb.set_params(
                self._director.update(0.0, self._state, motion_enabled=False)
            )
        self._canvas.queue_draw()

    def _animations_on(self) -> bool:
        return animations_on(self._theme, self._motion_enabled)

    def _scaled(self, seconds: float) -> float:
        return seconds * self._theme.animation.duration_scale

    def _apply_position(self, dimensions: tuple[int, int] | None = None) -> None:
        """Apply the themed anchor/margins via layer-shell.

        Safe to call on a mapped surface: anchor/margin changes are
        double-buffered protocol state, applied live on the next commit.
        No-op when layer-shell is unavailable.
        """
        if not self._has_layer_shell:
            return
        edges = anchor_to_edges(self._theme.position.anchor)
        width, height = dimensions or (self.get_width(), self.get_height())
        if width <= 0 or height <= 0:
            width, height = self._surface_size()
        content_width, content_height = self._content_size()
        inset_x = max(0, (width - content_width) // 2)
        inset_y = max(0, (height - content_height) // 2)
        top, right, bottom, left = self._theme.position.margin
        for name, edge, margin, inset in (
            ("top", Gtk4LayerShell.Edge.TOP, top, inset_y),
            ("right", Gtk4LayerShell.Edge.RIGHT, right, inset_x),
            ("bottom", Gtk4LayerShell.Edge.BOTTOM, bottom, inset_y),
            ("left", Gtk4LayerShell.Edge.LEFT, left, inset_x),
        ):
            anchored = name in edges
            Gtk4LayerShell.set_anchor(self, edge, anchored)
            # Margins on unanchored edges are reset to 0: layer-shell margin
            # state persists per-edge and would re-apply on a later re-anchor.
            visible_margin = max(_GINT_MIN, margin - inset)
            Gtk4LayerShell.set_margin(self, edge, visible_margin if anchored else 0)

    # --- State ---

    @property
    def shows_text(self) -> bool:
        """Whether this overlay can carry a message.

        The pill renders error and hint text; the orb is a status object and never
        renders text, so its callers must put words on the notification channel.
        """
        return self._style == "pill"

    def prewarm(self):
        """Map the layer-shell surface before the first recording, for an instant show.

        The overlay stays IDLE and draws nothing, so only the compositor's surface
        allocation is paid now rather than on the hotkey press. The COSMIC crash
        comes from `set_visible(False)` alone, not from mapping the surface.
        """
        if not self._has_layer_shell:
            return
        self.set_visible(True)
        self._apply_input_region()

    def _apply_input_region(self) -> None:
        """Match the surface's clickable region to whether the overlay is visible.

        Empty while IDLE, so the mapped but transparent surface passes pointer events
        through to the window beneath; a span covering the whole surface at any size in
        every visible state, so click-to-cancel works. A no-op until the surface exists.
        """
        if not self._has_layer_shell:
            return
        surface = self.get_surface()
        if surface is None:
            return
        if self._state == OverlayState.IDLE:
            surface.set_input_region(cairo.Region())  # empty region: click-through
        else:
            # Larger than any surface: the compositor ignores the part outside it, so
            # the whole surface takes clicks at every size, before and after a relayout.
            region = cairo.Region(cairo.RectangleInt(0, 0, _REGION_SPAN, _REGION_SPAN))
            surface.set_input_region(region)

    def show_recording(self):
        """Show the recording state from a fresh trace and timer."""
        if not self._has_layer_shell:
            return
        was_idle = self._state == OverlayState.IDLE
        self._state = OverlayState.RECORDING
        self._start_time = time.monotonic()
        self._trace.reset()
        if self._director is not None:
            self._director.reset()  # a new recording starts from silence
        self._hide_pending = False
        self._is_exiting = False
        if was_idle and self._animations_on():
            self._is_entering = True
            self._entry_start = time.monotonic()
        self.set_visible(True)
        self._apply_input_region()  # now visible: capture the cancel click
        self._start_animation()

    def show_transcribing(self):
        """Enter transcribing, freezing the wave to decay into the travelling sine."""
        # Idle here means the recording's show already ended (a failed draw, an
        # expired error).
        if not self._has_layer_shell or self._state == OverlayState.IDLE:
            return
        self._state = OverlayState.TRANSCRIBING
        self._transition_start = time.monotonic()
        if not self._animations_on():
            self._transition_start -= self._scaled(TRANSITION_TOTAL_S)
        self._frozen_amplitudes = self._trace.values()
        # The tick callback is already running from show_recording().

    def hide_pill(self):
        """Hide the overlay, through its exit animation when motion is on.

        A transcribing wave first plays for at least MIN_WAVE_DISPLAY_S after its
        transition. The layer-shell surface stays mapped and draws nothing: calling
        `set_visible(False)` on a layer-shell surface crashes COSMIC with a "Broken
        pipe".
        """
        if not self._has_layer_shell:
            return
        if not self._animations_on():
            if self._state != OverlayState.IDLE:
                self._do_hide()
            return

        if self._state == OverlayState.TRANSCRIBING:
            elapsed = time.monotonic() - self._transition_start
            if elapsed < self._scaled(TRANSITION_TOTAL_S + MIN_WAVE_DISPLAY_S):
                logger.debug(
                    "Pill hide deferred: transcribing wave within its min-display "
                    "window ({:.2f}s elapsed)",
                    elapsed,
                )
                self._hide_pending = True
                return

        if not self._is_exiting and self._state != OverlayState.IDLE:
            self._is_exiting = True
            self._exit_start = time.monotonic()
            self._start_animation()

    def _do_hide(self):
        """Hide the overlay now; the deferred and animated hides end here.

        It only redraws as transparent IDLE, never `set_visible(False)`, which
        crashes COSMIC with a "Broken pipe". Only a layer-shell show leads here.
        """
        self._state = OverlayState.IDLE
        self._hide_pending = False
        self._is_exiting = False
        self._is_entering = False
        self._stop_animation()
        self._sync_surface()
        self._canvas.queue_draw()

    def show_error(self, message: str, *, guidance: str | None = None):
        """Show an error heading and optional guidance until `ERROR_HIDE_MS` passes."""
        if not self._has_layer_shell:
            return
        logger.debug("Pill error shown: {}", message)
        was_idle = self._state == OverlayState.IDLE
        self._state = OverlayState.ERROR
        self._error_message = message
        self._error_guidance = guidance
        self._hide_pending = False
        self._is_exiting = False
        if was_idle and self._animations_on():
            self._is_entering = True
            self._entry_start = time.monotonic()
        self.set_visible(True)
        self._apply_input_region()  # now visible: capture clicks
        self._reconcile_motion()

        if self._hide_timeout_id is not None:
            GLib.source_remove(self._hide_timeout_id)
        self._hide_timeout_id = GLib.timeout_add(ERROR_HIDE_MS, self._auto_hide_error)

    def _auto_hide_error(self):
        self._hide_timeout_id = None
        if self._state == OverlayState.ERROR:
            self.hide_pill()
        return False

    def show_hint(self, message: str, *, guidance: str | None = None):
        """Show a persistent hint during recording, such as "No input detected".

        Unlike `show_error` it never auto-hides; real input or the end of the recording
        clears it.
        """
        if not self._has_layer_shell or self._state != OverlayState.RECORDING:
            return
        self._state = OverlayState.HINT
        self._hint_message = message
        self._hint_guidance = guidance
        self._canvas.queue_draw()

    def push_amplitude(self, amplitude: float):
        """Push a new amplitude value (0.0-1.0) for the wave, or the orb's mood."""
        if self._state == OverlayState.HINT:
            # Input recovered; drop back to the live view.
            self._state = OverlayState.RECORDING
        self._trace.push(amplitude)
        if self._director is not None:
            self._director.push_level(amplitude)

    def set_on_cancel(self, callback):
        """Set the callback run when the user clicks the overlay to cancel."""
        self._on_cancel = callback

    def _on_click(self, gesture, n_press, x, y):
        if self._on_cancel:
            self._on_cancel()

    def _start_animation(self):
        if self._tick_id is None:
            self._last_frame_us = None  # a fresh delta baseline, not a resume jump
            self._tick_id = self._canvas.add_tick_callback(
                guarded(self._tick, on_error=self._forget_tick)
            )

    def _forget_tick(self):
        self._tick_id = None

    def _stop_animation(self):
        if self._tick_id is not None:
            self._canvas.remove_tick_callback(self._tick_id)
            self._tick_id = None

    def _tick(self, widget, frame_clock):
        now_us = frame_clock.get_frame_time()
        dt = frame_dt(self._last_frame_us, now_us)
        self._last_frame_us = now_us
        if self._animations_on():
            self._dot_phase += PHASE_RATE * dt / self._theme.animation.duration_scale
        if self._orb is not None and self._state != OverlayState.IDLE:
            # Params always follow (they carry the causal voice/state response);
            # the flow itself advances only while decorative motion is allowed.
            self._orb.set_params(
                self._director.update(
                    dt, self._state, motion_enabled=self._animations_on()
                )
            )
            if self._animations_on():
                self._orb.advance(dt / self._theme.animation.duration_scale)
        now = time.monotonic()

        if self._hide_pending:
            elapsed = now - self._transition_start
            if elapsed >= self._scaled(TRANSITION_TOTAL_S + MIN_WAVE_DISPLAY_S):
                self._hide_pending = False
                if not self._is_exiting:
                    self._is_exiting = True
                    self._exit_start = now

        if self._is_exiting:
            if now - self._exit_start >= self._scaled(
                self._theme.animation.exit.duration
            ):
                self._do_hide()
                return False

        self._canvas.queue_draw()
        return True

    # --- Drawing ---

    def _draw(self, area, cr, width, height):
        try:
            self._draw_frame(cr, width, height)
        except Exception:
            # Every tick queues a draw; going idle stops the repeat until the next show.
            # The full hide runs from the main loop once this failed draw returns.
            logger.exception("Overlay draw failed; hiding the overlay")
            self._stop_animation()
            self._state = OverlayState.IDLE
            GLib.idle_add(self._hide_after_failed_draw)

    def _hide_after_failed_draw(self) -> bool:
        if self._state == OverlayState.IDLE:  # unless a new show has started since
            self._do_hide()
        return False

    def _draw_frame(self, cr, width, height):
        if self._state == OverlayState.IDLE:
            return

        # Clear to transparent
        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        # Handle entry/exit animations (scale + alpha)
        scale = 1.0
        alpha = 1.0
        now = time.monotonic()
        anim = self._theme.animation

        if self._is_entering:
            duration = self._scaled(anim.entry.duration)
            t = (now - self._entry_start) / duration if duration > 0 else 1.0
            if t >= 1.0:
                self._is_entering = False
            else:
                t_ease = ease(anim.entry.easing, t)
                scale = 0.90 + 0.10 * t_ease
                # Back-family curves leave [0, 1], so alpha is clamped to an opacity.
                alpha = min(1.0, max(0.0, t_ease))
        elif self._is_exiting:
            duration = self._scaled(anim.exit.duration)
            t = (now - self._exit_start) / duration if duration > 0 else 1.0
            if t >= 1.0:
                self._do_hide()
                return
            else:
                t_ease = ease(anim.exit.easing, t)
                scale = 1.0 - 0.10 * t_ease
                alpha = min(1.0, max(0.0, 1.0 - t_ease))

        if alpha < 1.0:
            cr.push_group()

        content_width, content_height = self._content_size()
        cr.translate(width / 2, height / 2)
        if scale != 1.0:
            cr.scale(scale, scale)
        cr.translate(-content_width / 2, -content_height / 2)

        if self._style == "orb":
            # Every visible state is the orb, shaped by the params set in _tick. It
            # renders no text; the app sends messages to notifications instead.
            size = min(ORB_SURFACE, content_width, content_height)
            cr.translate((content_width - size) / 2, (content_height - size) / 2)
            self._orb.draw(cr, size=size, motion=self._animations_on())
        elif self._state in (OverlayState.RECORDING, OverlayState.TRANSCRIBING):
            # The pill proper, drawn exactly as every view of it is drawn.
            draw_pill(
                cr,
                theme=self._theme,
                frame=self._frame(),
                width=content_width,
                height=content_height,
            )
        elif self._state in (OverlayState.ERROR, OverlayState.HINT):
            # Overlay-only affordances: nothing else renders them, so they stay here.
            self._draw_message(cr, content_width, content_height)

        if alpha < 1.0:
            cr.pop_group_to_source()
            cr.paint_with_alpha(alpha)

    def _frame(self) -> PillFrame:
        """Build the pill's frame now: the microphone's trace, the recording's clock."""
        return PillFrame(
            state=self._state,
            amplitudes=self._trace.values(),
            phase=self._dot_phase,
            timer=format_timer(time.monotonic() - self._start_time),
            frozen=self._frozen_amplitudes,
            transition_elapsed=time.monotonic() - self._transition_start,
            duration_scale=self._theme.animation.duration_scale,
            motion_enabled=self._animations_on(),
        )

    def _draw_message(self, cr, width, height):
        """Draw the error or hint state: the pill with the error dot and a message.

        Only the overlay shows these states, so they draw here rather than through
        `draw_pill`.
        """
        pal = self._theme.palette
        draw_background(
            cr,
            width=width,
            height=height,
            radius=height / 2,
            bg=pal.background,
            border=pal.border,
            border_width=self._theme.geometry.border_width,
        )
        draw_status_dot(
            cr,
            pal.dot_error,
            self._dot_phase,
            width,
            height,
            1.0,
            ripple=self._animations_on(),
        )
        message = (
            self._error_message
            if self._state == OverlayState.ERROR
            else self._hint_message
        )
        guidance = (
            self._error_guidance
            if self._state == OverlayState.ERROR
            else self._hint_guidance
        )
        font = self._theme.font
        detail_size = font.size * 5 / 6
        max_width = width - LABEL_LEFT_X - LABEL_PAD_X
        title_y = height / 2
        guidance_y = None
        if guidance:
            title = label_layout(
                cr,
                message,
                font_family=font.family,
                font_size=font.size,
                max_width=max_width,
                font_weight=Pango.Weight.BOLD,
            )
            detail = label_layout(
                cr,
                guidance,
                font_family=font.family,
                font_size=detail_size,
                max_width=max_width,
            )
            if title is not None and detail is not None:
                title_height = ink_extents(title)[3]
                guidance_height = ink_extents(detail)[3]
                block_height = title_height + 4 + guidance_height
                # Compact themes keep their chosen font size and show only the heading.
                if block_height <= height - 8:
                    title_y = (height - block_height + title_height) / 2
                    guidance_y = (height + block_height - guidance_height) / 2
        draw_label(
            cr,
            message,
            width=width,
            height=height,
            color=pal.text,
            font_family=font.family,
            font_size=font.size,
            font_weight=Pango.Weight.BOLD,
            center_y=title_y,
            align="left",
        )
        if guidance_y is not None:
            draw_label(
                cr,
                guidance,
                width=width,
                height=height,
                color=pal.text,
                font_family=font.family,
                font_size=detail_size,
                center_y=guidance_y,
                opacity=0.85,
            )
