# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The app's one brand CSS provider and shared page helpers.

Renders the selected app skin into GTK's Adwaita colour roles plus the brand's own
classes (keycap, buttons, headlines, artwork pages), and reloads on dark,
high-contrast and reduced-motion changes. Also holds the pointer-cursor binding and
the hand-laid status column and action row that onboarding and the benchmark dialog
share.
"""

from __future__ import annotations

import weakref

import gi
import numpy as np

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GObject, Gtk  # noqa: E402

from quoth.color import oklab_to_srgb, srgb_to_oklab  # noqa: E402
from quoth.skin import DEFAULT_SKIN, Skin  # noqa: E402

# Raised surfaces lift by a wash of the foreground over the page, as the site's
# cards do. The foreground makes the wash follow the selected palette.
SURFACE_WASH = 0.05
SURFACE_LINE = 0.08
# The one brand red, a warm deep carmine: a destructive button is this with bone on
# it, and error text is a lighter cut that still clears 4.5:1 on ink. Desktops
# otherwise supply their own (COSMIC: a pastel pink).
RED = "#9a1f2e"
RED_TEXT = "#dc5a4a"
# Photography has fixed light foregrounds independent of the app palette.
_ART_FOREGROUND = "#f2f0ea"
_ART_FOREGROUND_DIM = "#cfccc3"
_ART_ON_FOREGROUND = "#0e0e11"
_ART_BACKGROUND = "#08080a"
# The onboarding keycap's corner, which the halo around it follows.
KEYCAP_CORNER_PX = 14

_PROVIDER: Gtk.CssProvider | None = None
_SKIN = DEFAULT_SKIN
_MOTION_ENABLED = True


def _standalone_accent(value: str, *, dark: bool) -> str:
    """Return `value` with its OKLab lightness clamped to read as text on the page."""
    rgb = np.array(
        [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)],
        dtype=np.float32,
    )
    lab = srgb_to_oklab(rgb)
    lab[0] = max(lab[0], 0.85) if dark else min(lab[0], 0.5)
    standalone = oklab_to_srgb(lab)
    return "#" + "".join(f"{int(channel * 255 + 0.5):02x}" for channel in standalone)


# The brand's own classes, the same text in every appearance mode.
_BRAND_RULES = (
    "drawingarea.quoth-animation { color: @accent_color; }"
    # Artwork keeps a light foreground over its dark photograph in every mode,
    # as does a header bar drawn over it.
    ".quoth-welcome, .quoth-welcome headerbar, .quoth-artwork,"
    f" headerbar.quoth-over-artwork {{ color: {_ART_FOREGROUND}; }}"
    # Its close button carries its own ink disc, so it still reads once the
    # artwork has scrolled away from under it.
    "headerbar.quoth-over-artwork windowcontrols > button > image"
    f" {{ background-color: alpha({_ART_BACKGROUND}, 0.6); }}"
    "headerbar.quoth-over-artwork windowcontrols > button:hover > image"
    f" {{ background-color: alpha({_ART_BACKGROUND}, 0.8); }}"
    "headerbar.quoth-over-artwork windowcontrols > button:active > image"
    f" {{ background-color: alpha({_ART_BACKGROUND}, 0.95); }}"
    f".quoth-welcome drawingarea.quoth-animation {{ color: {_ART_FOREGROUND}; }}"
    f".quoth-welcome button.suggested-action {{ background: {_ART_FOREGROUND};"
    f" color: {_ART_ON_FOREGROUND}; }}"
    # The site's primary button (a tall pill, bold label) and its hero headline
    # (large, light, tight tracking; the emphasised words carry weight in markup).
    "button.quoth-cta { min-height: 48px; padding: 0 32px;"
    "  font-weight: bold; font-size: 1.1em; }"
    "label.quoth-headline { font-size: 2.5em; font-weight: 300;"
    "  letter-spacing: -0.03em; line-height: 1.05; }"
    "label.quoth-headline.compact { font-size: 2em; }"
    "label.quoth-headline.roomy { font-size: 3.2em; }"
    # Secondary text over the photograph takes its own dim colour: dim-label's
    # opacity would let the smoke through.
    ".quoth-welcome label.quoth-tagline, .quoth-welcome label.quoth-blurb,"
    " .quoth-welcome label.quoth-dim, .quoth-artwork label.quoth-tagline,"
    f" .quoth-artwork label.quoth-dim {{ color: {_ART_FOREGROUND_DIM}; }}"
    # The welcome's smoke darkens towards its foot, as on the site, under a veil
    # that keeps the copy legible over the brightest frames of its video.
    f"clamp.quoth-scrim {{ background: linear-gradient(to top, {_ART_BACKGROUND},"
    f"  alpha({_ART_BACKGROUND}, 0.7) 18%, alpha({_ART_BACKGROUND}, 0.4) 60%); }}"
    # The dictation keycap, drawn in the text colour so it reads in either scheme. A
    # transform sinks the face without touching layout; down at once, up slowly.
    f".quoth-keycap {{ background: alpha(currentColor, 0.08);"
    f"  border-radius: {KEYCAP_CORNER_PX}px;"
    "  border: 1px solid alpha(currentColor, 0.14); padding: 18px 40px;"
    "  box-shadow: 0 6px 0 alpha(currentColor, 0.12); margin-bottom: 6px;"
    "  transition: box-shadow 600ms cubic-bezier(0.3, 0.7, 0.4, 1),"
    "    transform 600ms cubic-bezier(0.3, 0.7, 0.4, 1); }"
    ".quoth-keycap.pressed { box-shadow: 0 2px 0 alpha(currentColor, 0.12);"
    "  transform: translateY(4px); transition: box-shadow 34ms, transform 34ms; }"
    # The Shortcut page's sizes, set on its column by the window breakpoints.
    ".compact .quoth-keycap { padding: 12px 28px; }"
    ".compact .quoth-keycap > label { font-size: 15pt; }"
    ".roomy .quoth-keycap { padding: 26px 60px; }"
    ".roomy .quoth-keycap > label { font-size: 34pt; }"
    "box.roomy > label.title-1 { font-size: 26pt; }"
    # The About hero: smoke fading into the dialog, the mark in full bone. Adwaita
    # paints a dialog's sheet in the window background, not @dialog_bg_color.
    "box.quoth-fade { background: linear-gradient(to bottom,"
    "  alpha(@window_bg_color, 0) 25%, alpha(@window_bg_color, 0.7) 75%,"
    "  @window_bg_color); }"
    f".quoth-welcome, .quoth-artwork {{ background-color: {_ART_BACKGROUND}; }}"
    f".quoth-artwork image.quoth-mark {{ color: {_ART_FOREGROUND}; }}"
    "label.quoth-blurb { font-size: 1.1em; line-height: 1.5; }"
    # The site's ghost button: a hairline pill that fills faintly on hover.
    "button.quoth-ghost { background: none;"
    "  border: 1px solid alpha(currentColor, 0.25); box-shadow: none; }"
    "button.quoth-ghost:hover { background: alpha(currentColor, 0.06);"
    "  border-color: alpha(currentColor, 0.5); }"
    # The support heart beats under the pointer; the brand motion gate removes
    # this decorative animation when the system asks for reduced motion.
    "@keyframes quoth-heartbeat {"
    "  0% { -gtk-icon-transform: scale(1); }"
    "  14% { -gtk-icon-transform: scale(1.3); }"
    "  28% { -gtk-icon-transform: scale(1); }"
    "  42% { -gtk-icon-transform: scale(1.18); }"
    "  60% { -gtk-icon-transform: scale(1); }"  # GTK takes one selector per keyframe
    "  100% { -gtk-icon-transform: scale(1); } }"
    ".quoth-support:hover image.quoth-heart {"
    "  animation: quoth-heartbeat 1.2s ease-in-out infinite; }"
)


def css(
    skin: Skin = DEFAULT_SKIN,
    *,
    dark: bool,
    high_contrast: bool = False,
    motion_enabled: bool = True,
) -> str:
    """Return the stylesheet for one appearance mode."""
    palette = skin.dark if dark else skin.light
    accent = _standalone_accent(palette.accent, dark=dark)
    sheet = ""
    if not high_contrast:
        sheet = (
            f"@define-color accent_bg_color {palette.accent};"
            f"@define-color accent_fg_color {palette.on_accent};"
            f"@define-color accent_color {accent};"
        )
        if Gtk.get_minor_version() >= 16:
            sheet += f":root {{ --accent-color: {accent}; }}"
        if palette.window is not None:
            sheet += (
                f"@define-color window_bg_color {palette.window};"
                f"@define-color headerbar_bg_color {palette.window};"
                f"@define-color headerbar_backdrop_color {palette.window};"
            )
        if palette.text is not None:
            for role in (
                "window_fg_color",
                "view_fg_color",
                "headerbar_fg_color",
                "card_fg_color",
                "dialog_fg_color",
                "popover_fg_color",
            ):
                sheet += f"@define-color {role} {palette.text};"
        if palette.view is not None:
            sheet += (
                f"@define-color dialog_bg_color {palette.view};"
                f"@define-color view_bg_color {palette.view};"
            )
        if palette.card is not None:
            sheet += f"@define-color card_bg_color {palette.card};"
        elif dark and palette.text is not None:
            sheet += (
                f"@define-color card_bg_color alpha({palette.text}, {SURFACE_WASH});"
            )
        if palette.popover is not None:
            sheet += f"@define-color popover_bg_color {palette.popover};"
        if dark:
            sheet += (
                f"@define-color destructive_bg_color {RED};"
                f"@define-color destructive_fg_color {_ART_FOREGROUND};"
                f"@define-color destructive_color {RED_TEXT};"
                f"@define-color error_bg_color {RED};"
                f"@define-color error_fg_color {_ART_FOREGROUND};"
                f"@define-color error_color {RED_TEXT};"
            )
        if dark or palette.card is not None:
            sheet += (
                ".card, list.boxed-list { box-shadow: 0 0 0 1px "
                f"alpha(currentColor, {SURFACE_LINE}); }}"
            )
    sheet += _BRAND_RULES
    if not dark:
        # A light dialog would grey the smoke under the hero's bone text, so there
        # the fade starts below that text.
        sheet += (
            "box.quoth-fade { background: linear-gradient(to bottom,"
            "  alpha(@window_bg_color, 0) 74%, alpha(@window_bg_color, 0.55) 89%,"
            "  @window_bg_color); }"
        )
    if high_contrast:
        sheet += (
            ".quoth-keycap, .quoth-keycap.pressed { background: none;"
            " border-color: currentColor; box-shadow: none; }"
            "button.quoth-ghost, button.quoth-ghost:hover { background: none;"
            " border-color: currentColor; }"
            ".quoth-settings row:not(:last-child) { border-bottom: 1px solid @borders; }"
            ".quoth-settings row.expander list.nested { background-color: "
            "alpha(@card_shade_color, 0.5); }"
        )
    else:
        sheet += (
            ".quoth-settings row:not(:last-child) { border-bottom: 1px solid "
            f"alpha(currentColor, {SURFACE_LINE}); }}"
            ".quoth-settings row.expander list.nested { background-color: "
            "alpha(currentColor, 0.04); }"
        )
    if not motion_enabled:
        sheet += (
            ".quoth-keycap, .quoth-keycap.pressed { transition: none; }"
            ".quoth-support:hover image.quoth-heart { animation: none; }"
        )
    return sheet


def _queue_animation_draws() -> None:
    def queue(root: Gtk.Widget) -> None:
        if isinstance(root, Gtk.DrawingArea) and root.has_css_class("quoth-animation"):
            root.queue_draw()
        child = root.get_first_child()
        while child is not None:
            queue(child)
            child = child.get_next_sibling()

    for window in Gtk.Window.get_toplevels():
        queue(window)


def _reload(*_args) -> None:
    if _PROVIDER is None:
        return
    manager = Adw.StyleManager.get_default()
    _PROVIDER.load_from_string(
        css(
            _SKIN,
            dark=manager.get_dark(),
            high_contrast=manager.get_high_contrast(),
            motion_enabled=_MOTION_ENABLED,
        )
    )
    if _PROVIDER.find_property("prefers-reduced-motion") is not None:
        _PROVIDER.set_property("prefers-reduced-motion", not _MOTION_ENABLED)
    _queue_animation_draws()


def apply_skin(skin: Skin) -> None:
    """Apply `skin` to every open app window."""
    global _SKIN
    _SKIN = skin
    _reload()


def set_motion_enabled(enabled: bool) -> None:
    """Update the brand's decorative CSS when the system preference changes."""
    global _MOTION_ENABLED
    if enabled == _MOTION_ENABLED:
        return
    _MOTION_ENABLED = enabled
    _reload()


def install(*, motion_enabled: bool = True) -> None:
    """Install one app-skin provider and follow system appearance changes.

    A later call re-applies the current skin to the provider already installed.
    """
    global _MOTION_ENABLED, _PROVIDER
    _MOTION_ENABLED = motion_enabled
    if _PROVIDER is not None:
        apply_skin(_SKIN)
        return
    _PROVIDER = Gtk.CssProvider()
    manager = Adw.StyleManager.get_default()
    manager.connect("notify::dark", _reload)
    manager.connect("notify::high-contrast", _reload)
    # USER priority beats desktop accents; Quoth's personal style provider uses
    # USER + 1 so it still has the last word.
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(), _PROVIDER, Gtk.STYLE_PROVIDER_PRIORITY_USER
    )
    apply_skin(_SKIN)


_POINTER = (Gtk.Button, Gtk.Switch, Gtk.CheckButton, Gtk.ListBoxRow)
_POINTER_CURSOR = Gdk.Cursor.new_from_name("pointer")
_DEFAULT_CURSOR = Gdk.Cursor.new_from_name("default")
# The widest a status column grows, so its copy keeps its measure on a wide window.
STATUS_WIDTH = 460


def pointer_cursors(root: Gtk.Widget) -> None:
    """Keep the hand cursor aligned with each widget's live interactivity."""

    def action_target(row: Gtk.ListBoxRow) -> Gtk.Widget | None:
        if isinstance(row, Adw.ActionRow):
            return row.get_activatable_widget()
        return None

    def has_activation_handler(row: Gtk.ListBoxRow) -> bool:
        signal = "activated" if isinstance(row, Adw.ActionRow) else "activate"
        signal_id = GObject.signal_lookup(signal, type(row))
        return bool(
            signal_id and GObject.signal_has_handler_pending(row, signal_id, 0, True)
        )

    def update(widget: Gtk.Widget, *_args) -> None:
        actionable = widget.is_sensitive() and widget.get_can_target()
        if isinstance(widget, Gtk.ListBoxRow):
            target = action_target(widget)
            usable_target = target is not None and (
                target.is_sensitive() and target.get_can_target()
            )
            if isinstance(widget, Adw.ComboRow):
                model = widget.get_model()
                row_action = model is not None and model.get_n_items() > 0
            elif isinstance(widget, Adw.ExpanderRow):
                row_action = (
                    not widget.get_show_enable_switch() or widget.get_enable_expansion()
                )
            else:
                row_action = widget.get_activatable() and (
                    usable_target
                    if target is not None
                    else has_activation_handler(widget)
                )
            actionable = actionable and row_action
        cursor = _POINTER_CURSOR if actionable else _DEFAULT_CURSOR
        if (
            not actionable
            and isinstance(widget, Adw.ActionRow)
            and widget.get_activatable()
            and action_target(widget) is None
            and widget.get_ancestor(Adw.ExpanderRow) is not None
            and widget.has_css_class("header")
        ):
            # The native ExpanderRow header inherits its outer row's live state.
            cursor = None
        widget.set_cursor(cursor)

    def bind_action_target(row: Gtk.ListBoxRow, *_args) -> None:
        target = action_target(row)
        previous = getattr(row, "_quoth_cursor_target", None)
        if previous is target:
            update(row)
            return
        if previous is not None:
            for handler in row._quoth_cursor_target_handlers:
                previous.disconnect(handler)
        handlers = []
        if target is not None:
            row_ref = weakref.ref(row)

            def update_row(*_args) -> None:
                if owner := row_ref():
                    update(owner)

            handlers = [
                target.connect("state-flags-changed", update_row),
                target.connect("notify::can-target", update_row),
            ]
        row._quoth_cursor_target = target
        row._quoth_cursor_target_handlers = handlers
        update(row)

    def visit(widget: Gtk.Widget) -> None:
        if isinstance(widget, _POINTER):
            update(widget)
            if not getattr(widget, "_quoth_cursor_bound", False):
                widget.connect("state-flags-changed", update)
                widget.connect("notify::can-target", update)
                if isinstance(widget, Gtk.ListBoxRow):
                    widget.connect("notify::activatable", update)
                    if isinstance(widget, Adw.ActionRow):
                        widget.connect("notify::activatable-widget", bind_action_target)
                    if isinstance(widget, Adw.ComboRow):
                        widget.connect("notify::model", update)
                    if isinstance(widget, Adw.ExpanderRow):
                        widget.connect("notify::show-enable-switch", update)
                        widget.connect("notify::enable-expansion", update)
                widget._quoth_cursor_bound = True
            if isinstance(widget, Gtk.ListBoxRow):
                bind_action_target(widget)
        child = widget.get_first_child()
        while child is not None:
            visit(child)
            child = child.get_next_sibling()

    visit(root)


def status_column(
    visual: Gtk.Widget | None, title: str, description: str | None
) -> Gtk.Box:
    """Build a status page by hand: an optional `visual`, `title` and `description`.

    For pages whose visual is a widget rather than an icon, or sits outside the page:
    Adw.StatusPage takes icons only. Callers append what a status page's child would
    hold.
    """
    heading = Gtk.Label(label=title, wrap=True, justify=Gtk.Justification.CENTER)
    heading.add_css_class("title-1")
    column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    column.set_valign(Gtk.Align.CENTER)
    column.set_margin_top(36)
    column.set_margin_bottom(36)
    column.set_margin_start(12)
    column.set_margin_end(12)
    if visual is not None:
        visual.set_halign(Gtk.Align.CENTER)
        visual.set_margin_bottom(12)
        column.append(visual)
    column.append(heading)
    if description is not None:
        column.append(
            Gtk.Label(label=description, wrap=True, justify=Gtk.Justification.CENTER)
        )
    return column


def action_row(buttons: list[Gtk.Button]) -> Gtk.Box:
    """Build a page's action row: `buttons` as pills, bottom right."""
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    row.set_halign(Gtk.Align.END)
    row.set_margin_top(6)
    row.set_margin_bottom(12)
    row.set_margin_start(12)
    row.set_margin_end(12)
    for button in buttons:
        button.add_css_class("pill")
        row.append(button)
    return row
