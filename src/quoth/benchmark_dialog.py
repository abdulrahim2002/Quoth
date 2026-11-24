# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The hardware-benchmark dialog: a staged view over the calibration run.

Clicking "Run" in the Hardware acceleration settings opens this modal, which walks
Prepare -> Measure -> Result. It is a view only: the engine is
`RecordingController.calibrate`, reached through the injected `start` and `cancel`
callbacks; the composition root owns the run and pushes progress and the result back
through `advance()` and `finish()`. The result copy comes from
`calibration.speed_label` and `verdict`.
"""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, GObject, Gtk
from loguru import logger

from quoth import brand, calibration
from quoth.animation import DotField
from quoth.calibration import CellResult, DoneFn, ProgressFn, Report
from quoth.models import model_label

_OUTER_GUTTER = 24


class BenchmarkDialog(Adw.Dialog):
    """The staged benchmark modal, shown with `.present(parent_window)`.

    It is driven by `start(models, devices, on_progress, on_done) -> bool`, which begins
    a run over the cells it displays (False if the engine is busy), and `cancel()`; the
    app pushes results back through `advance` and `finish`. `models` must be non-empty:
    the caller (`app._open_benchmark`) guards the "no model installed" case.
    """

    def __init__(
        self,
        models: list[str],
        devices: list[str],
        *,
        start: Callable[[list[str], list[str], ProgressFn, DoneFn], bool],
        cancel: Callable[[], None],
        motion_enabled: bool = True,
    ):
        super().__init__(title="Hardware benchmark")
        self._motion_enabled = motion_enabled
        self.set_follows_content_size(True)
        self.set_size_request(360, 360)
        self._start = start
        self._cancel = cancel
        self._models = models
        self._devices = devices
        self._cells = calibration.cell_order(models, devices)
        self._cell_status: dict[tuple[str, str], Gtk.Label] = {}
        self._done_count = 0
        self._cancelled = False
        self._closed = False
        self._result_announced = False
        self._action_rows: list[Gtk.Box] = []

        self._stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_hhomogeneous(False)
        self._stack.set_vhomogeneous(False)
        self._transition_ms = self._stack.get_transition_duration()
        if not self._motion_enabled:
            self._stack.set_transition_duration(0)
        self._stack.add_named(self._build_prepare(), "prepare")
        self._stack.add_named(self._build_measure(), "measure")
        self._stack.add_named(self._build_result(), "result")
        # One dot field above the stages: idle before the run, flickering during
        # it, the winner's wave on the result; it stays while the text changes.
        self._field = DotField(motion_enabled=self._motion_enabled)
        self._content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        self._content.set_size_request(brand.STATUS_WIDTH, -1)
        self._content.append(self._field)
        self._content.append(self._stack)

        clamp = Adw.Clamp(maximum_size=brand.STATUS_WIDTH, child=self._content)
        for side in ("top", "bottom", "start", "end"):
            getattr(clamp, f"set_margin_{side}")(_OUTER_GUTTER)
        self._scroller = Gtk.ScrolledWindow(vexpand=True)
        self._scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._scroller.set_propagate_natural_height(True)
        self._scroller.set_child(clamp)

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())
        toolbar.set_content(self._scroller)
        self.set_child(toolbar)
        narrow = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 480sp"))
        narrow.add_setter(self._content, "width-request", GObject.Value(int, -1))
        vertical = GObject.Value(Gtk.Orientation.__gtype__, Gtk.Orientation.VERTICAL)
        for actions in self._action_rows:
            narrow.add_setter(actions, "orientation", vertical)
        self.add_breakpoint(narrow)
        brand.pointer_cursors(toolbar)
        self.connect("closed", self._on_closed)

    def set_motion_enabled(self, enabled: bool) -> None:
        if enabled == self._motion_enabled:
            return
        self._motion_enabled = enabled
        self._stack.set_transition_duration(self._transition_ms if enabled else 0)
        self._field.set_motion_enabled(enabled)

    # --- stage 1: prepare ---

    def _build_prepare(self) -> Gtk.Widget:
        page = brand.status_column(
            None,
            "Find the fastest setup for your hardware",
            "Quoth benchmarks your downloaded models on available hardware. Automatic mode "
            "uses the fastest measured device for each model. Larger models can take longer."
            + calibration.preparation_note(self._devices),
        )
        page.set_valign(Gtk.Align.START)
        for side in ("top", "bottom", "start", "end"):
            getattr(page, f"set_margin_{side}")(0)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        box.set_margin_top(12)

        note = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        info = Gtk.Image.new_from_icon_name("dialog-information-symbolic")
        info.add_css_class("dim-label")
        info.set_valign(Gtk.Align.START)
        self._note = Gtk.Label(
            label=(
                "Close GPU- or NPU-heavy apps for more consistent results. "
                "You can run the benchmark anyway."
            )
        )
        self._note.set_wrap(True)
        self._note.set_xalign(0)
        self._note.set_hexpand(True)
        self._note.set_max_width_chars(44)
        self._note.add_css_class("dim-label")
        self._note.add_css_class("caption")
        note.append(info)
        note.append(self._note)
        box.append(note)

        cancel = Gtk.Button(label="Cancel")
        cancel.add_css_class("pill")
        cancel.connect("clicked", lambda *_: self.close())
        run = Gtk.Button(label="Run benchmark")
        run.add_css_class("pill")
        run.add_css_class("suggested-action")
        run.connect("clicked", lambda *_: self._on_run())
        box.append(self._action_row(cancel, run))

        page.append(box)
        return page

    # --- stage 2: measure ---

    def _build_measure(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)

        self._current = Gtk.Label(xalign=0)
        self._current.set_wrap(True)
        self._current.add_css_class("heading")
        self._bar = Gtk.ProgressBar()
        self._bar.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Benchmark progress"]
        )
        helper = Gtk.Label(label="Measuring how fast each device runs.", xalign=0)
        helper.set_wrap(True)
        helper.add_css_class("dim-label")
        helper.add_css_class("caption")

        self._cell_list = Gtk.ListBox()
        self._cell_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._cell_list.add_css_class("boxed-list")
        for model, device in self._cells:
            row = Adw.ActionRow(title=f"{model_label(model)} on {device}")
            status = Gtk.Label(label="Waiting")
            # Wrap only when the row is too narrow, not to the shortest width.
            status.set_wrap(True)
            status.set_natural_wrap_mode(Gtk.NaturalWrapMode.NONE)
            status.add_css_class("dim-label")
            row.add_suffix(status)
            self._cell_status[(model, device)] = status
            self._cell_list.append(row)
        self._stop = Gtk.Button(label="Cancel", halign=Gtk.Align.CENTER)
        self._stop.add_css_class("pill")
        self._stop.connect("clicked", lambda *_: self._on_cancel())

        for w in (self._current, self._bar, helper, self._cell_list):
            box.append(w)
        box.append(self._action_row(self._stop))
        return box

    # --- stage 3: result ---

    def _build_result(self) -> Gtk.Widget:
        page = brand.status_column(None, "Benchmark results", None)
        self._result_heading = page.get_first_child()
        page.set_valign(Gtk.Align.START)
        for side in ("top", "bottom", "start", "end"):
            getattr(page, f"set_margin_{side}")(0)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)

        self._verdict = Gtk.Label()
        self._verdict.set_wrap(True)
        self._verdict.set_justify(Gtk.Justification.CENTER)
        box.append(self._verdict)

        self._result_list = Gtk.ListBox()
        self._result_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._result_list.add_css_class("boxed-list")
        box.append(self._result_list)

        rerun = Gtk.Button(label="Re-run")
        rerun.add_css_class("pill")
        rerun.connect("clicked", lambda *_: self._on_run())
        self._done = Gtk.Button(label="Done")
        self._done.add_css_class("pill")
        self._done.add_css_class("suggested-action")
        self._done.connect("clicked", lambda *_: self.close())
        box.append(self._action_row(rerun, self._done))

        page.append(box)
        return page

    def _action_row(self, *buttons: Gtk.Button) -> Gtk.Box:
        row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=10,
            halign=Gtk.Align.CENTER,
        )
        for button in buttons:
            row.append(button)
        self._action_rows.append(row)
        return row

    def _set_current(self, text: str) -> None:
        self._current.set_label(text)
        self._bar.update_property([Gtk.AccessibleProperty.DESCRIPTION], [text])

    # --- flow ---

    def _on_run(self) -> None:
        """Begin the run from the prepare stage, or repeat it from the result stage.

        A False from `start` means the engine is busy; the prepare note says why.
        """
        if not self._start(self._models, self._devices, self.advance, self.finish):
            logger.debug("Benchmark run refused")
            self._note.set_label(
                "Quoth is busy. Wait for the current operation to finish, then try again."
            )
            self._field.set_idle()
            self.set_title("Hardware benchmark")
            self._stack.set_visible_child_name("prepare")
            return
        self._cancelled = False
        self._stop.set_sensitive(True)
        self._result_announced = False
        self._done_count = 0
        first = self._cells[0]
        for cell, status in self._cell_status.items():
            status.set_label("Measuring…" if cell == first else "Waiting")
        self._bar.set_fraction(0.0)
        self._set_current(
            f"{calibration.measuring_label(*first)} (1 of {len(self._cells)})"
        )
        self.set_title("Measuring your hardware")
        self._field.set_busy()
        self._stack.set_visible_child_name("measure")

    def _on_cancel(self) -> None:
        # A cell in flight cannot be interrupted; the run stops when it ends.
        logger.debug(
            "Benchmark cancel requested after {}/{} cells",
            self._done_count,
            len(self._cells),
        )
        self._cancelled = True
        self._stop.set_sensitive(False)
        self._set_current("Stopping after this measurement…")
        self._cancel()

    def advance(
        self, done: int, total: int, model: str, device: str, result: CellResult
    ) -> None:
        """Mark a finished cell with its speed, advance the bar and show the next cell.

        The app pushes this once per cell; an attempted cell without a timing reads "No
        result", and one skipped for files not downloaded says so.
        """
        if self._closed:
            return
        self._done_count = done
        self._bar.set_fraction(done / total)
        status = self._cell_status.get((model, device))
        if status is not None:
            if result.skipped:
                status.set_label("Not downloaded")
            elif result.compiled and result.ms is not None:
                status.set_label(calibration.speed_label(result.ms))
            else:
                status.set_label("No result")
        if done < total and not self._cancelled:
            nxt = self._cells[done]
            self._cell_status[nxt].set_label("Measuring…")
            self._set_current(
                f"{calibration.measuring_label(*nxt)} ({done + 1} of {total})"
            )

    def finish(self, report: Report | None) -> None:
        """Render the verdict and per-model winners when the run ends or is cancelled.

        A `report` of None means the whole run raised, shown as an explicit failure.
        """
        if self._closed:
            return
        won = calibration.winners(report) if report is not None else {}
        if won:
            self._field.set_hardware(next(iter(won.values()))[0])
        else:
            self._field.set_idle()
        total = len(self._cells)
        measured = self._done_count if self._cancelled else total
        if self._cancelled:
            for cell in self._cells[self._done_count :]:
                self._cell_status[cell].set_label("Not measured")
        if report is None:
            heading = "Benchmark failed"
            verdict = (
                "The benchmark couldn't finish. You can try again or close this window."
            )
        else:
            heading = "Benchmark results"
            verdict = calibration.verdict(won, measured=measured, total=total)
        self._result_heading.set_label(heading)
        self._verdict.set_label(verdict)
        self._result_list.remove_all()
        for model in self._models:
            if model not in won:
                continue
            device, ms = won[model]
            row = Adw.ActionRow(
                title=model_label(model), subtitle=f"Fastest on your {device}"
            )
            star = Gtk.Image.new_from_icon_name("starred-symbolic")
            star.add_css_class("accent")
            row.add_prefix(star)
            speed = Gtk.Label(label=calibration.speed_label(ms))
            speed.set_wrap(True)
            speed.add_css_class("dim-label")
            speed.add_css_class("numeric")
            row.add_suffix(speed)
            self._result_list.append(row)
        brand.pointer_cursors(self._result_list)
        self._done.set_label("Close" if report is None else "Done")
        self.set_title("Benchmark failed" if report is None else "Benchmark complete")
        self._stack.set_visible_child_name("result")
        if not self._result_announced:
            self._verdict.announce(verdict, Gtk.AccessibleAnnouncementPriority.MEDIUM)
            self._result_announced = True

    def _on_closed(self, *_args) -> None:
        """Stop the sweep when the dialog closes mid-run.

        A live run holds the engine and blocks recording, so it must not outlive the
        dialog.
        """
        if self._closed:
            return
        self._closed = True
        if self._stack.get_visible_child_name() == "measure" and not self._cancelled:
            logger.debug("Benchmark dialog closed mid-run; cancelling")
            self._cancelled = True
            self._cancel()
