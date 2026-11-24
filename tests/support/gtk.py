# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Walking, checking and waiting on real GTK widgets."""

from __future__ import annotations

import ctypes
import time
from contextlib import contextmanager

from tests.support.colour import contrast


def descendants(root):
    """Every widget below `root`, depth first."""
    child = root.get_first_child()
    while child is not None:
        yield child
        yield from descendants(child)
        child = child.get_next_sibling()


def accessible_mismatch(widget, attribute, *expected):
    """GTK's complaint when an accessible relation or property is not `expected`.

    `expected` is widgets for a relation and one string for a property; None means
    `widget` matches.
    """
    from gi.repository import Gtk

    def pointer(target):
        get = ctypes.pythonapi.PyCapsule_GetPointer
        get.restype = ctypes.c_void_p
        get.argtypes = [ctypes.py_object, ctypes.c_char_p]
        return ctypes.c_void_p(get(target.__gpointer__, None))

    gtk = ctypes.CDLL("libgtk-4.so.1")
    if isinstance(attribute, Gtk.AccessibleRelation):
        check = gtk.gtk_test_accessible_check_relation
        values = [*map(pointer, expected), None]
    else:
        check = gtk.gtk_test_accessible_check_property
        values = [ctypes.c_char_p(text.encode()) for text in expected]
    check.restype = ctypes.c_char_p
    return check(pointer(widget), ctypes.c_int(int(attribute)), *values)


def button(root, label: str):
    """The first button below `root` labelled `label`, or None."""
    from gi.repository import Gtk

    buttons = (w for w in descendants(root) if isinstance(w, Gtk.Button))
    return next((w for w in buttons if w.get_label() == label), None)


def shown_texts(widget) -> list[str]:
    """What the labels under `widget` display, after any markup parsing."""
    from gi.repository import Gtk

    return [w.get_text() for w in descendants(widget) if isinstance(w, Gtk.Label)]


def settle() -> None:
    """Run the default main context until nothing is pending."""
    from gi.repository import GLib

    context = GLib.MainContext.default()
    while context.pending():
        context.iteration(False)


def wait_until(predicate, timeout: float = 3.0) -> None:
    """Settle the main loop until `predicate()` holds, and fail after `timeout`.

    Frame clocks and file monitors deliver on their own schedule, so a test waits for
    the outcome rather than for a fixed delay.
    """
    deadline = time.monotonic() + timeout
    while True:
        settle()
        if predicate():
            return
        assert time.monotonic() < deadline, "timed out waiting on the main loop"
        time.sleep(0.005)


@contextmanager
def appearance(stylesheet: str, *, dark: bool):
    """Show widgets in one colour scheme with `stylesheet`, and animations off.

    With animations off a dialog or page opens at once, so a frame measured is the
    settled one. Everything changed is put back on exit.
    """
    from gi.repository import Adw, Gdk, Gtk

    manager = Adw.StyleManager.get_default()
    scheme = manager.get_color_scheme()
    settings = Gtk.Settings.get_default()
    animations = settings.get_property("gtk-enable-animations")
    provider = Gtk.CssProvider()
    provider.load_from_string(stylesheet)
    try:
        manager.set_color_scheme(
            Adw.ColorScheme.FORCE_DARK if dark else Adw.ColorScheme.FORCE_LIGHT
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
        settings.set_property("gtk-enable-animations", False)
        yield
    finally:
        settings.set_property("gtk-enable-animations", animations)
        Gtk.StyleContext.remove_provider_for_display(
            Gdk.Display.get_default(), provider
        )
        manager.set_color_scheme(scheme)


def legibility(window, widget, hide: str | None = None) -> float:
    """The contrast of `widget`'s ink against the worst of the ground under it.

    The ink is the pixel that stands out most from what lies behind it, so a focus
    ring or outline brighter than the text would count in its place; the ground is
    what lies behind each pixel the widget changes. `hide`, a stylesheet that takes
    away only the ink, keeps a backing the widget draws itself in the ground.
    """
    from gi.repository import Gtk

    shown, stride = _next_frame(window)
    if hide is None:
        widget.set_opacity(0)
        behind, _ = _next_frame(window)
        widget.set_opacity(1)
    else:
        provider = Gtk.CssProvider()
        provider.load_from_string(hide)
        display = window.get_display()
        priority = Gtk.STYLE_PROVIDER_PRIORITY_USER + 1
        Gtk.StyleContext.add_provider_for_display(display, provider, priority)
        try:
            behind, _ = _next_frame(window)
        finally:
            Gtk.StyleContext.remove_provider_for_display(display, provider)
    _, frame = window.compute_bounds(window)
    _, bounds = widget.compute_bounds(window)
    x0 = int(bounds.get_x() - frame.get_x())
    y0 = int(bounds.get_y() - frame.get_y())
    drawn = []
    for y in range(y0, y0 + int(bounds.get_height())):
        for x in range(x0, x0 + int(bounds.get_width())):
            i = y * stride + x * 3
            if shown[i : i + 3] != behind[i : i + 3]:
                drawn.append(
                    ("#" + shown[i : i + 3].hex(), "#" + behind[i : i + 3].hex())
                )
    assert drawn, f"{widget} draws nothing"
    ink = max(drawn, key=lambda pair: contrast(*pair))[0]
    return min(contrast(ink, ground) for _, ground in drawn)


def _next_frame(window) -> tuple[bytes, int]:
    """Wait for `window`'s next full styling, layout and drawing cycle, and return it.

    Only a mapped window's frame clock runs, so an unmapped one fails here rather
    than waiting forever.
    """
    from gi.repository import Gtk

    assert window.get_mapped(), f"{window} is not mapped, so it never draws"
    Gtk.test_widget_wait_for_draw(window)
    frame = _drawn(window)
    assert frame is not None, f"{window} drew no frame"
    return frame


def _drawn(widget) -> tuple[bytes, int] | None:
    """Render `widget` as it is drawn now, as RGB bytes and the row stride.

    None until its first frame.
    """
    from gi.repository import Gdk, Graphene, Gsk, Gtk

    paintable = Gtk.WidgetPaintable(widget=widget)
    # The intrinsic size takes in what the widget draws outside its allocation, such
    # as a window's client-side frame, so the render is not scaled.
    width = paintable.get_intrinsic_width()
    height = paintable.get_intrinsic_height()
    snapshot = Gtk.Snapshot()
    paintable.snapshot(snapshot, width, height)
    node = snapshot.to_node()
    if node is None:
        return None
    # Cairo, not the window's own renderer, so the pixels match on any GL or Vulkan.
    renderer = Gsk.CairoRenderer()
    renderer.realize_for_display(widget.get_display())
    texture = renderer.render_texture(node, Graphene.Rect().init(0, 0, width, height))
    renderer.unrealize()
    downloader = Gdk.TextureDownloader.new(texture)
    downloader.set_format(Gdk.MemoryFormat.R8G8B8)
    data, stride = downloader.download_bytes()
    return data.get_data(), stride
