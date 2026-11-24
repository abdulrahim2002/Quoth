# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""See every overlay state on both the pill and the orb, side by side.

The five states the live overlay runs: idle, recording, hint, error, transcribing.
The pill uses the real primitives (`draw_dot`, `draw_waveform`, `draw_label`). Its
idle <-> recording morph is a prototype the live pill does not run: the width grows
from a circle to the full pill, the dot stays anchored, and the waveform and timer wipe
in behind a growing clip. Transcribing runs the real `draw_pill`; error and hint reuse
the message primitives; the orb runs its real director, which morphs between states as
it does live.

    uv run python scripts/simulate_overlay_states.py
"""

# Select GI namespace versions before importing the view modules.
# ruff: noqa: E402

import math
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GLib, Gtk

from quoth.pill_render import (
    BARS_START_X,
    DOT_RADIUS,
    DOT_RIPPLE,
    DOT_X,
    LABEL_LEFT_X,
    LABEL_PAD_X,
    NUM_BARS,
    PHASE_RATE,
    WAVE_PEAK,
    draw_background,
    draw_dot,
    draw_label,
    draw_waveform,
    format_timer,
    wave_end_x,
)
from quoth.pill_view import OrbView, OverlayState, PillView, _scripted_level, pill_width
from quoth.theme import DEFAULT_THEME

THEME = DEFAULT_THEME
PILL_H = 76
PILL_W = pill_width(THEME, PILL_H)
ORB = 150
STATES = [
    OverlayState.IDLE,
    OverlayState.RECORDING,
    OverlayState.HINT,
    OverlayState.ERROR,
    OverlayState.TRANSCRIBING,
]
MESSAGE = {"error": "Paste failed, text copied", "hint": "No input detected"}
_HALF_PI = math.pi / 2

_msg_state = {"s": "error"}
_phase = {"v": 0.0}  # advanced by a tick so the message dot ripple pulses


def _rounded_path(cr, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -_HALF_PI, 0)
    cr.arc(x + w - r, y + h - r, r, 0, _HALF_PI)
    cr.arc(x + r, y + h - r, r, _HALF_PI, math.pi)
    cr.arc(x + r, y + r, r, math.pi, math.pi + _HALF_PI)
    cr.close_path()


def _pill_bg(cr, w, h):
    pal = THEME.palette
    scale = h / THEME.geometry.height
    draw_background(
        cr,
        width=w,
        height=h,
        radius=h / 2,
        bg=pal.background,
        border=pal.border,
        border_width=THEME.geometry.border_width * scale,
    )


def _scaled_dot(cr, h, color, phase):
    """Draw the status dot, placed and sized exactly as `draw_pill`'s recording dot."""
    scale = h / THEME.geometry.height
    draw_dot(
        cr,
        x=DOT_X * scale,
        y=h / 2,
        color=color,
        phase=phase,
        radius=DOT_RADIUS * scale,
        ripple_reach=DOT_RIPPLE * scale,
    )


class MorphPill(Gtk.DrawingArea):
    """A drawing area that plays idle <-> recording as one width animation.

    The pill stays centred and expands symmetrically from a circle to the full pill.
    The dot sits `DOT_X` from the moving left edge, so it begins at the circle's centre
    and drifts out to its home on the left as the pill opens, while the waveform and
    timer wipe in behind the growing clip.
    """

    def __init__(self):
        super().__init__()
        self.set_content_width(PILL_W)  # full width; the drawn pill grows within it
        self.set_content_height(PILL_H)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self._reveal = 0.0  # 0 = idle circle, 1 = recording pill
        self._phase = 0.0
        self._amps = [0.0] * NUM_BARS
        self._t0 = time.monotonic()
        self.set_draw_func(self._draw)
        self._anim = Adw.TimedAnimation.new(
            self, 0.0, 0.0, 380, Adw.CallbackAnimationTarget.new(self._on_reveal)
        )
        self._anim.set_easing(Adw.Easing.EASE_OUT_CUBIC)
        GLib.timeout_add(33, self._tick)

    def _on_reveal(self, v):
        self._reveal = v
        self.queue_draw()

    def animate_to(self, target: float) -> None:
        self._anim.set_value_from(self._reveal)
        self._anim.set_value_to(target)
        self._anim.play()

    def _tick(self) -> bool:
        self._phase += PHASE_RATE * 0.033
        self._amps = [*self._amps[1:], _scripted_level(self._phase)]
        self.queue_draw()
        return GLib.SOURCE_CONTINUE

    def _draw(self, _area, cr, w, h):
        pal = THEME.palette
        scale = h / THEME.geometry.height
        current = h + (w - h) * self._reveal  # circle (h) -> full pill (w)
        left = (
            w - current
        ) / 2  # centred: the pill expands symmetrically from the middle

        cr.translate(left, 0)
        _pill_bg(cr, current, h)  # the growing pill, with its border
        _rounded_path(cr, 0, 0, current, h, h / 2)
        cr.clip()  # content wipes in as the pill widens

        _scaled_dot(
            cr, h, pal.dot_recording, self._phase
        )  # DOT_X from the moving left edge
        if self._reveal > 0.02:  # idle is just the dot; content appears as it opens
            draw_waveform(
                cr,
                height=h,
                start_x=BARS_START_X * scale,
                end_x=wave_end_x(THEME.geometry.bar_gap) * scale,
                color=pal.accent,
                amplitudes=self._amps,
                line_width=THEME.geometry.line_width * scale,
                peak=WAVE_PEAK * scale,
                gated=True,
            )
            draw_label(
                cr,
                format_timer(time.monotonic() - self._t0),
                width=current,
                height=h,
                color=pal.text,
                font_family=THEME.font.family,
                font_size=THEME.font.size * scale,
                left_x=LABEL_LEFT_X * scale,
                pad_x=LABEL_PAD_X * scale,
                align="right",
                opacity=self._reveal,
            )


def _draw_message(_area, cr, w, h):
    """Draw the error or hint pill, scaled like every other state."""
    pal = THEME.palette
    scale = h / THEME.geometry.height
    _pill_bg(cr, w, h)
    # Red only signals error, so a hint gets the neutral live dot here; the live
    # overlay still draws a hint with the error dot.
    _scaled_dot(
        cr,
        h,
        pal.dot_error if _msg_state["s"] == "error" else pal.dot_recording,
        _phase["v"],
    )
    draw_label(
        cr,
        MESSAGE[_msg_state["s"]],
        width=w,
        height=h,
        color=pal.text,
        font_family=THEME.font.family,
        font_size=THEME.font.size * scale,
        left_x=BARS_START_X * scale,  # start past the dot, where the waveform begins
        pad_x=LABEL_PAD_X * scale,
        align="left",
    )


def _column(title: str, child: Gtk.Widget) -> Gtk.Box:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    box.set_valign(Gtk.Align.CENTER)
    label = Gtk.Label(label=title)
    label.add_css_class("dim-label")
    holder = Gtk.Box(halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
    holder.set_size_request(max(PILL_W, ORB) + 20, ORB)
    holder.append(child)
    box.append(label)
    box.append(holder)
    return box


def run(app):
    morph = MorphPill()
    trans = PillView(THEME, height=PILL_H, animated=True, scripted=True, timer=True)
    trans.set_halign(Gtk.Align.CENTER)
    trans.set_valign(Gtk.Align.CENTER)
    msg = Gtk.DrawingArea()
    msg.set_content_width(PILL_W)
    msg.set_content_height(PILL_H)
    msg.set_draw_func(_draw_message)
    msg.set_halign(Gtk.Align.CENTER)
    msg.set_valign(Gtk.Align.CENTER)

    pill_stack = Gtk.Stack()
    pill_stack.add_named(morph, "morph")  # idle <-> recording (animated)
    pill_stack.add_named(trans, "trans")  # transcribing
    pill_stack.add_named(msg, "message")  # error / hint

    orb = OrbView(THEME, size=ORB, live=True)

    def select(state: str) -> None:
        orb.set_state(state)
        if state in ("idle", "recording"):
            pill_stack.set_visible_child_name("morph")
            morph.animate_to(1.0 if state == "recording" else 0.0)
        elif state in ("error", "hint"):
            _msg_state["s"] = state
            msg.queue_draw()
            pill_stack.set_visible_child_name("message")
        else:
            trans.set_state("transcribing")
            pill_stack.set_visible_child_name("trans")
        print(f"  [sim] state = {state}")

    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    buttons.set_halign(Gtk.Align.CENTER)
    first = None
    for state in STATES:
        b = Gtk.ToggleButton(label=state.capitalize())
        b.add_css_class("pill")
        if first is None:
            first = b
        else:
            b.set_group(first)
        b.connect("toggled", lambda btn, s=state: btn.get_active() and select(s))
        buttons.append(b)

    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=40)
    row.set_halign(Gtk.Align.CENTER)
    row.append(_column("Pill", pill_stack))
    row.append(_column("Orb", orb))

    content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
    content.set_margin_top(24)
    content.set_margin_bottom(24)
    content.set_margin_start(24)
    content.set_margin_end(24)
    content.append(buttons)
    content.append(row)

    win = Adw.ApplicationWindow(application=app, default_width=680, default_height=420)
    win.set_title("Overlay states (pill vs orb)")
    toolbar = Adw.ToolbarView()
    toolbar.add_top_bar(Adw.HeaderBar())
    toolbar.set_content(content)
    win.set_content(toolbar)

    def _tick() -> bool:
        _phase["v"] += PHASE_RATE * 0.033
        msg.queue_draw()  # pulse the message dot (morph/PillView animate themselves)
        return GLib.SOURCE_CONTINUE

    GLib.timeout_add(33, _tick)
    win.present()
    first.set_active(True)  # start on idle


app = Adw.Application(application_id="dev.quoth.SimulateOverlayStates")
app.connect("activate", run)
app.run([])
