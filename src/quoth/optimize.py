# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The onboarding Optimize step: benchmark the model on available devices.

The run is the app's benchmark (`RecordingController.calibrate`, through the same
`start`/`cancel` pair the benchmark dialog uses), so it holds the engine like any
other. The sweep is the configured model across every device, so `auto` learns this
machine's fastest device for it. The page moves through three states: offer, running
and result.

A skip during the run moves on at once, but the measurement in flight cannot be
interrupted, so the engine stays held until it lands; whatever must not share the
engine waits for it there.

Like the Try it step, the page owns nothing about navigation: onboarding wraps
`content` and `actions` in its chrome, `on_advance` moves the flow on, and
`on_lock(True)` prevents returning to earlier engine users until the run ends.
"""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk
from loguru import logger

from quoth import brand, calibration
from quoth.animation import DotField
from quoth.calibration import DoneFn, ProgressFn
from quoth.config import Config
from quoth.models import model_label
from quoth.progress import SmoothProgressBar

# The widest the dot field grows; the text stays at the status column's width.
_FIELD_WIDTH = 720


def _under_field(column: Gtk.Box) -> None:
    """Seat a status column under the dot field, without vertical margins of its own.

    The block above centres the two together.
    """
    column.set_valign(Gtk.Align.START)
    column.set_margin_top(0)
    column.set_margin_bottom(0)


class OptimizePage:
    """View and run of the onboarding calibration."""

    def __init__(
        self,
        config: Config,
        devices: list[str],
        *,
        start: Callable[[list[str], list[str], ProgressFn, DoneFn], bool],
        cancel: Callable[[], None],
        on_advance: Callable[[], None],
        on_lock: Callable[[bool], None],
        motion_enabled: bool = True,
    ):
        """Build the page over the injected run callbacks.

        `start` begins the run and returns False when the engine is busy; `cancel`
        stops it at the next cell.
        """
        self._model = config.model
        self._devices = devices
        self._start_run = start
        self._cancel_run = cancel
        self._on_advance = on_advance
        self._on_lock = on_lock
        self._motion = motion_enabled
        self._cells = calibration.cell_order([self._model], devices)
        self._cancelled = False
        self._closed = False
        self._build()

    def close(self) -> None:
        """Stop at the next cell as the window goes, and touch no widgets after."""
        self._cancelled = True
        self._closed = True
        self._cancel_run()

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion:
            return
        self._motion = enabled
        duration = self._transition_ms if enabled else 0
        self._stack.set_transition_duration(duration)
        self.actions.set_transition_duration(duration)
        self._field.set_motion_enabled(enabled)
        self._bar.set_motion_enabled(enabled)

    # --- building ---

    def _build(self) -> None:
        self._stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self._transition_ms = self._stack.get_transition_duration()
        # The crossfade is motion; hold it still under reduced motion.
        if not self._motion:
            self._stack.set_transition_duration(0)
        # Each state's actions follow its content in a second synchronised stack.
        self.actions = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.actions.set_transition_duration(self._stack.get_transition_duration())
        self._stack.bind_property(
            "visible-child-name", self.actions, "visible-child-name"
        )
        self._stack.add_named(self._build_offer(), "offer")
        self._stack.add_named(self._build_running(), "running")
        self._stack.add_named(self._build_result(), "result")
        # One dot field for the whole step, above the text that changes under it:
        # idle on the offer, flickering while measuring, the winner's wave on the
        # result.
        self._field = DotField(motion_enabled=self._motion)
        self._field.set_margin_start(24)
        self._field.set_margin_end(24)
        # The field and the text are centred together as one block; the field
        # takes the width of its clamp and so grows with the window.
        self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        self.content.set_valign(Gtk.Align.CENTER)
        self.content.set_vexpand(True)
        self.content.set_margin_top(24)
        self.content.set_margin_bottom(24)
        self.content.append(Adw.Clamp(maximum_size=_FIELD_WIDTH, child=self._field))
        self.content.append(self._stack)

    def _build_offer(self) -> Gtk.Widget:
        column = brand.status_column(
            None,
            "Optimize for your hardware",
            f"Quoth can benchmark {model_label(self._model)} on your "
            f"{calibration.device_phrase(self._devices)} so Automatic can use the "
            f"fastest measured device.{calibration.preparation_note(self._devices)} "
            "You can skip and benchmark later in Settings.",
        )
        _under_field(column)
        skip = Gtk.Button(label="Skip")
        # Flat, so Optimize is the only weighted action on the page.
        skip.add_css_class("flat")
        skip.connect("clicked", lambda *_: self._on_advance())
        self._optimize = Gtk.Button(label="Optimize")
        self._optimize.add_css_class("suggested-action")
        self._optimize.connect("clicked", lambda *_: self._start())
        self.actions.add_named(brand.action_row([skip, self._optimize]), "offer")
        return Adw.Clamp(maximum_size=brand.STATUS_WIDTH, child=column)

    def _build_running(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_valign(Gtk.Align.START)
        box.set_margin_bottom(24)
        box.set_margin_start(24)
        box.set_margin_end(24)
        self._current = Gtk.Label(label="Preparing…", halign=Gtk.Align.CENTER)
        self._current.add_css_class("heading")
        self._bar = SmoothProgressBar(motion_enabled=self._motion)
        self._bar.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Benchmark progress"]
        )
        helper = Gtk.Label(
            label="Measuring how fast each device runs.", halign=Gtk.Align.CENTER
        )
        helper.add_css_class("dim-label")
        helper.add_css_class("caption")
        for widget in (self._current, self._bar, helper):
            box.append(widget)
        skip = Gtk.Button(label="Skip")
        skip.connect("clicked", lambda *_: self._skip())
        self.actions.add_named(brand.action_row([skip]), "running")
        return box

    def _build_result(self) -> Gtk.Widget:
        # The title is set on completion.
        column = brand.status_column(
            None, "", "You can re-run this any time in Settings."
        )
        self._result_title = column.get_first_child()
        self._result_description = column.get_last_child()
        _under_field(column)
        cont = Gtk.Button(label="Continue")
        cont.add_css_class("suggested-action")
        cont.connect("clicked", lambda *_: self._on_advance())
        self.actions.add_named(brand.action_row([cont]), "result")
        return Adw.Clamp(maximum_size=brand.STATUS_WIDTH, child=column)

    # --- the run ---

    def _start(self) -> None:
        if not self._start_run(
            [self._model], self._devices, self._progress, self._done
        ):
            # The engine is held elsewhere (a benchmark or NPU preparation from
            # Settings); the step stays passable.
            logger.info("Onboarding benchmark refused: the engine is busy")
            self._show_result(
                "Quoth is busy", "You can continue and benchmark later in Settings."
            )
            return
        self._cancelled = False
        self._on_lock(True)
        self._set_current(calibration.measuring_label(*self._cells[0]))
        self._field.set_busy()
        self._stack.set_visible_child_name("running")
        # Progress only lands between devices, so creep toward the first milestone
        # rather than sitting at zero through the long first (compile) measurement.
        self._bar.trickle_to(1 / len(self._cells))

    def _progress(self, done: int, total: int, *_cell) -> None:
        if self._cancelled:
            return
        # Glide to the device that just finished, then trickle on toward the next.
        self._bar.glide_to(done / total)
        if done < total:
            self._bar.trickle_to((done + 1) / total)
            self._set_current(calibration.measuring_label(*self._cells[done]))

    def _set_current(self, text: str) -> None:
        self._current.set_label(text)
        self._bar.update_property([Gtk.AccessibleProperty.DESCRIPTION], [text])

    def _skip(self) -> None:
        """Move on at once; every finished device is already saved.

        The page goes back to its offer, which cannot start again until the run ends.
        """
        logger.debug("Onboarding benchmark skipped mid-run")
        self._cancelled = True
        self._cancel_run()
        self._field.set_idle()
        self._bar.glide_to(self._bar.get_fraction())
        self._optimize.set_sensitive(False)
        self._stack.set_visible_child_name("offer")
        self._on_advance()

    def _done(self, report: calibration.Report | None) -> None:
        if self._closed:
            return
        self._optimize.set_sensitive(True)
        self._on_lock(False)
        if self._cancelled:
            return
        self._bar.glide_to(self._bar.get_fraction() if report is None else 1.0)
        if report is None:
            self._field.set_idle()
            self._show_result(
                "Benchmark failed", "You can continue and benchmark later in Settings."
            )
            return
        won = calibration.winners(report)
        if self._model in won:
            self._field.set_hardware(won[self._model][0])
        else:
            self._field.set_idle()
        self._show_result(
            calibration.optimized_line(won, self._model),
            "You can re-run this any time in Settings.",
        )

    def _show_result(self, title: str, description: str) -> None:
        # Showing the result moves the focus to Continue, past this text, so it is read.
        self._result_title.set_label(title)
        self._result_description.set_label(description)
        self._stack.set_visible_child_name("result")
        self._result_title.announce(
            f"{title}. {description}", Gtk.AccessibleAnnouncementPriority.MEDIUM
        )
