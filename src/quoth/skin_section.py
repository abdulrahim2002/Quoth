# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Settings > General > Appearance: the application skin picker."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gi
from loguru import logger

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from quoth import brand, paths  # noqa: E402
from quoth.atomic_write import diagnostic  # noqa: E402

if TYPE_CHECKING:
    from quoth.skin import Skin


_CUSTOM_SKIN = "custom"
# Unreadable shelf skins already warned about, with their error: rows are rebuilt on
# every render, so the same failure warns once.
_warned_skins: set[tuple[str, str]] = set()
_FALLBACK_LABEL_MAX = 60
_RIBBON_SIZE = 40
_RIBBON_BANDS = (14, 10, 10, 6)
_RIBBON_RADIUS = 6
_AUTOMATIC_BANDS_HELP = "Patterned bands show system colors or tints of the text color"
_CUSTOM_BANDS_HELP = "Palette preview unavailable for this custom skin"


def _display_label(value: str) -> str:
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value).strip()
    return cleaned[:_FALLBACK_LABEL_MAX].rstrip() or "Unnamed skin"


def _skin_label(skin_id: str) -> str:
    """Return the fallback picker label for a skin id, such as 'Catppuccin'.

    A [meta] name, when the file carries one, wins over this in `_skin_texts`.
    """
    if skin_id == "default":
        return "Default"
    if skin_id == _CUSTOM_SKIN:
        return "Custom"
    fallback = skin_id.removeprefix("user:")
    return _display_label(fallback.replace("-", " ").title())


def _skin_texts(skin_id: str, skin: Skin) -> tuple[str, str | None]:
    """Return a skin row's title and subtitle.

    The title is the skin's [meta] name over the filename-derived label, and the
    subtitle credits the author. The loader caps and sanitises these strings and the
    rows render with markup off, so a shared skin file cannot spoof or restyle the
    picker.
    """
    title = skin.meta.name or _skin_label(skin_id)
    subtitle = f"by {skin.meta.author}" if skin.meta.author else None
    return title, subtitle


def _skin_action(picked: str, on_disk: str | None) -> str:
    """Return what clicking the `picked` row does: "apply", "confirm" or "ignore".

    `on_disk` is the skin id skin.toml holds, None when the file is the user's own.
    Only that case prompts, since the copy replaces the whole file.
    """
    if picked == on_disk:
        return "ignore"
    if on_disk is None:
        return "confirm"
    return "apply"


def _ribbon_bands(
    skin: Skin | None, *, dark: bool, high_contrast: bool
) -> tuple[str | None, ...]:
    """Return the window, view, card and accent colours; None draws a patterned band."""
    if skin is None or high_contrast:
        return (None,) * len(_RIBBON_BANDS)
    palette = skin.dark if dark else skin.light
    return palette.window, palette.view, palette.card, palette.accent


def _palette_ribbon(skin: Skin | None, manager: Adw.StyleManager) -> Gtk.DrawingArea:
    """Return a skin's palette as bands, patterned where the system colour shows.

    With no skin to draw, as for the Custom row, every band is patterned.
    """
    bands = _ribbon_bands(
        skin,
        dark=manager.get_dark(),
        high_contrast=manager.get_high_contrast(),
    )
    ribbon = Gtk.DrawingArea(
        accessible_role=Gtk.AccessibleRole.PRESENTATION,
        can_target=False,
        focusable=False,
        valign=Gtk.Align.CENTER,
    )
    ribbon.set_content_width(_RIBBON_SIZE)
    ribbon.set_content_height(_RIBBON_SIZE)
    ribbon._quoth_palette_bands = bands

    def draw(area, context, width, height) -> None:
        radius = min(_RIBBON_RADIUS, width / 2, height / 2)
        context.new_sub_path()
        context.arc(width - radius, radius, radius, -math.pi / 2, 0)
        context.arc(width - radius, height - radius, radius, 0, math.pi / 2)
        context.arc(radius, height - radius, radius, math.pi / 2, math.pi)
        context.arc(radius, radius, radius, math.pi, 3 * math.pi / 2)
        context.close_path()
        outline = context.copy_path()
        context.save()
        context.clip()

        foreground = area.get_color()
        y = 0.0
        for value, units in zip(bands, _RIBBON_BANDS, strict=True):
            band_height = height * units / _RIBBON_SIZE
            if value is not None:
                colour = Gdk.RGBA()
                colour.parse(value)
                context.set_source_rgba(
                    colour.red, colour.green, colour.blue, colour.alpha
                )
                context.rectangle(0, y, width, band_height)
                context.fill()
            else:
                context.set_source_rgba(
                    foreground.red, foreground.green, foreground.blue, 0.08
                )
                context.rectangle(0, y, width, band_height)
                context.fill()
                context.save()
                context.rectangle(0, y, width, band_height)
                context.clip()
                context.set_source_rgba(
                    foreground.red, foreground.green, foreground.blue, 0.28
                )
                context.set_line_width(1)
                for x in range(-height, width + height, 6):
                    context.move_to(x, y + band_height)
                    context.line_to(x + band_height, y)
                context.stroke()
                context.restore()
            y += band_height

        context.restore()
        context.set_source_rgba(foreground.red, foreground.green, foreground.blue, 0.18)
        context.set_line_width(1)
        context.append_path(outline)
        context.stroke()

    ribbon.set_draw_func(draw)
    return ribbon


def _shelf_divider_row() -> Gtk.ListBoxRow:
    """Build the caption row separating shipped presets from the user's shelf.

    A plain row, not an ActionRow, so it reads as a heading rather than a peer row.
    """
    label = Gtk.Label(label="Your skins", xalign=0)
    label.add_css_class("caption-heading")
    label.add_css_class("dim-label")
    label.set_margin_top(14)
    label.set_margin_bottom(4)
    label.set_margin_start(12)
    row = Gtk.ListBoxRow()
    row.set_child(label)
    row.set_activatable(False)
    row.set_selectable(False)
    row.set_focusable(False)
    return row


@dataclass
class SkinBridge:
    """App-side skin operations the settings view consumes.

    `SkinRuntime` owns skin.toml and the monitor watching it, so the view never touches
    the file. `presets` is a snapshot, since package data cannot change at runtime.
    """

    presets: list[str]  # bundled preset ids, in picker order
    # The 'user:<stem>' ids of the user skins directory, re-read on every render.
    user_skins: Callable[[], list[str]]
    # The applied skin id: "default" when there is no file, None for the user's own.
    current: Callable[[], str | None]
    skin_of: Callable[[str], Skin]  # any picker id parsed, for drawing it
    load_error: Callable[[], str | None]  # last active-file load failure, if any
    # Copy a preset or user skin over skin.toml, or remove it for Default.
    apply: Callable[[str], None]
    import_file: Callable[[Path], str]  # copy one local TOML file to the user shelf


class SkinSection:
    """The application skin picker on the General page."""

    def __init__(self, bridge: SkinBridge, *, toast: Callable[[str], None]):
        self._bridge = bridge
        self._toast = toast
        self._window: Adw.PreferencesWindow | None = None
        self._group: Adw.PreferencesGroup | None = None
        self._skin_expander: Adw.ExpanderRow | None = None
        self._import_button: Gtk.Button | None = None
        self._import_cancellable: Gio.Cancellable | None = None
        self._skin_rows: dict[str, Adw.ActionRow] = {}
        self._shelf_divider: Gtk.ListBoxRow | None = None
        self._style_manager: Adw.StyleManager | None = None
        self._style_handlers: list[int] = []

    def build(self, page: Adw.PreferencesPage, window: Adw.PreferencesWindow) -> None:
        """Add the Appearance group and its skin picker to `page`.

        Clicking a row applies that skin through the bridge. `window` parents the
        confirm and import dialogs.
        """
        self._window = window
        group = Adw.PreferencesGroup(title="Appearance")
        self._group = group
        page.add(group)
        self._style_manager = Adw.StyleManager.get_default()
        self._style_handlers = [
            self._style_manager.connect(signal, self._on_appearance_changed)
            for signal in ("notify::dark", "notify::high-contrast")
        ]
        self._update_group_description()

        self._skin_expander = Adw.ExpanderRow(
            title="App skin",
            tooltip_text="Changes the colors of Quoth's own windows, in light and dark",
        )
        # Named by its title: an expander row has no name of its own, and GTK
        # would otherwise take its tooltip.
        self._skin_expander.update_property(
            [Gtk.AccessibleProperty.LABEL], [self._skin_expander.get_title()]
        )
        self._skin_expander.set_use_markup(False)
        self._import_button = Gtk.Button(label="Import…", valign=Gtk.Align.CENTER)
        self._import_button.set_tooltip_text(
            "Adds a .toml skin to this list without applying it"
        )
        # The label child names the button through a relation that outranks the
        # label property, so a screen reader would hear only "Import…".
        self._import_button.reset_relation(Gtk.AccessibleRelation.LABELLED_BY)
        self._import_button.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Import skin"]
        )
        self._import_button.connect("clicked", self._on_import_clicked)
        self._skin_expander.add_suffix(self._import_button)
        group.add(self._skin_expander)
        self._render_skin_rows()

    def close(self) -> None:
        """Release the window: cancel a pending import and stop following the style."""
        if self._import_cancellable is not None:
            self._import_cancellable.cancel()
            self._import_cancellable = None
        if self._style_manager is not None:
            for handler in self._style_handlers:
                self._style_manager.disconnect(handler)
        self._style_handlers.clear()
        self._style_manager = None
        self._window = None

    def _on_appearance_changed(self, *_args) -> None:
        self._update_group_description()
        self.refresh_skin()

    def _update_group_description(self, *_args) -> None:
        self._set_group_description(self._bridge.load_error())

    def _set_group_description(self, error: str | None) -> None:
        if self._group is None:
            return
        messages = []
        if error:
            messages.append(error)
        if self._style_manager is not None and self._style_manager.get_high_contrast():
            messages.append(
                "High contrast is on. System colors are shown while your selected skin is remembered."
            )
        self._group.set_description("\n".join(messages))

    def refresh_skin(self) -> None:
        """Re-render after skin.toml changes, from an apply or a hand edit.

        `_on_appearance_changed` also calls it when dark mode or high contrast changes.
        """
        if self._skin_expander is not None and self._window is not None:
            self._render_skin_rows()

    def _on_import_clicked(self, button: Gtk.Button) -> None:
        if self._import_cancellable is not None or self._window is None:
            return
        button.set_sensitive(False)
        window = self._window
        cancellable = Gio.Cancellable()
        self._import_cancellable = cancellable
        skin_filter = Gtk.FileFilter()
        skin_filter.set_name("App skins (*.toml)")
        skin_filter.add_pattern("*.toml")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(skin_filter)
        dialog = Gtk.FileDialog(title="Import skin", filters=filters)
        dialog.set_default_filter(skin_filter)
        dialog.open(
            window,
            cancellable,
            lambda chooser, result: self._on_import_file_chosen(
                chooser, result, window, cancellable
            ),
        )

    def _on_import_file_chosen(
        self,
        dialog: Gtk.FileDialog,
        result,
        window: Adw.PreferencesWindow,
        cancellable: Gio.Cancellable,
    ) -> None:
        error: GLib.Error | None = None
        try:
            file = dialog.open_finish(result)
        except GLib.Error as exc:
            file = None
            error = exc

        if self._window is not window or self._import_cancellable is not cancellable:
            return
        self._import_cancellable = None
        self._import_button.set_sensitive(True)

        if error is not None:
            if any(
                error.matches(domain, code)
                for domain, code in (
                    (Gtk.DialogError.quark(), Gtk.DialogError.CANCELLED),
                    (Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED),
                    (Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED),
                )
            ):
                return
            logger.error("Could not choose skin file: {}", error)
            self._toast("Couldn't open the skin file")
            return

        path = file.get_path()
        if path is None:
            # The scheme alone: a remote URI names the account and the host.
            logger.warning(
                "Rejected skin import: a {} location has no local path",
                file.get_uri_scheme(),
            )
            self._toast("Choose a local skin file")
            return
        try:
            skin_id = self._bridge.import_file(Path(path))
        except ValueError as exc:
            logger.warning("Rejected skin import {}: {}", path, exc)
            self._toast(str(exc))
            return
        except OSError as exc:
            logger.error("Could not import skin file {}: {}", path, exc)
            self._render_skin_rows()
            self._skin_expander.set_expanded(True)
            self._toast("Couldn't confirm the import. Check the skin list.")
            return

        self._render_skin_rows()
        if row := self._skin_rows.get(skin_id):

            def focus_imported_row() -> bool:
                if (
                    self._window is window
                    and self._skin_rows.get(skin_id) is row
                    and row.get_mapped()
                ):
                    row.grab_focus()
                return False

            map_handler: int | None = None

            def focus_after_map(_row=None) -> None:
                nonlocal map_handler
                if map_handler is not None:
                    row.disconnect(map_handler)
                    map_handler = None
                GLib.idle_add(focus_imported_row)

            if row.get_mapped():
                focus_after_map()
            else:
                map_handler = row.connect("map", focus_after_map)
        self._skin_expander.set_expanded(True)
        self._toast("Skin imported. Select it to apply.")

    def _render_skin_rows(self) -> None:
        """Rebuild every skin row and the expander subtitle from the file on disk.

        `current` is read once, so every row and the subtitle agree.
        """
        current = self._bridge.current()
        self._set_group_description(self._bridge.load_error())
        for row in self._skin_rows.values():
            self._skin_expander.remove(row)
        self._skin_rows.clear()
        if self._shelf_divider is not None:
            self._skin_expander.remove(self._shelf_divider)
            self._shelf_divider = None

        # Default and the bundled presets first, then the user's skins beneath a
        # divider that exists only while a shelf row does.
        for skin_id in ["default", *self._bridge.presets]:
            if row := self._skin_row(skin_id, current):
                self._skin_expander.add_row(row)
                self._skin_rows[skin_id] = row
        for skin_id in self._bridge.user_skins():
            if row := self._skin_row(skin_id, current):
                self._add_shelf_row(skin_id, row)

        # A shelf file can change after identification, so only a surviving row
        # marks the active skin; anything else is the user's own file.
        if current in self._skin_rows:
            self._skin_expander.set_subtitle(self._skin_rows[current].get_title())
        else:
            row = Adw.ActionRow(title="Custom", subtitle="Your current skin.toml")
            row.set_use_markup(False)
            row.add_prefix(_palette_ribbon(None, self._style_manager))
            row.set_tooltip_text(_CUSTOM_BANDS_HELP)
            self._add_current_indicator(row)
            self._add_shelf_row(_CUSTOM_SKIN, row)
            self._skin_expander.set_subtitle("Custom")
        brand.pointer_cursors(self._skin_expander)

    def _add_shelf_row(self, skin_id: str, row: Adw.ActionRow) -> None:
        if self._shelf_divider is None:
            self._shelf_divider = _shelf_divider_row()
            self._skin_expander.add_row(self._shelf_divider)
        self._skin_expander.add_row(row)
        self._skin_rows[skin_id] = row

    def _skin_row(self, skin_id: str, current: str | None) -> Adw.ActionRow | None:
        """Build one skin row, or return None when the skin cannot be read.

        The active row gets the marker and is not activatable, so a stray click cannot
        rewrite skin.toml with what it already holds.
        """
        try:
            skin = self._bridge.skin_of(skin_id)
        except (ValueError, OSError) as exc:
            failure = (skin_id, str(exc))
            warn = failure not in _warned_skins
            _warned_skins.add(failure)
            (logger.warning if warn else logger.debug)(
                "Could not preview skin {} ({}: {})",
                diagnostic(skin_id),
                type(exc).__name__,
                diagnostic(paths.tilde(str(exc))),
            )
            return None
        title, subtitle = _skin_texts(skin_id, skin)
        row = Adw.ActionRow(title=title)
        # Skin names and credits are arbitrary file content; markup stays off so
        # they render as text, never as Pango styling.
        row.set_use_markup(False)
        ribbon = _palette_ribbon(skin, self._style_manager)
        row.add_prefix(ribbon)
        if subtitle:
            row.set_subtitle(subtitle)
        help_text = (
            _AUTOMATIC_BANDS_HELP
            if any(value is None for value in ribbon._quoth_palette_bands)
            else None
        )
        tooltip = [skin.meta.description, help_text]
        if text := "\n".join(part for part in tooltip if part):
            row.set_tooltip_text(text)
        selected = skin_id == current
        if selected:
            self._add_current_indicator(row)
        else:
            row.set_activatable(True)
            row.connect("activated", lambda _row, name=skin_id: self._on_selected(name))
        return row

    @staticmethod
    def _add_current_indicator(row: Adw.ActionRow) -> None:
        check = Gtk.Image(
            icon_name="object-select-symbolic",
            accessible_role=Gtk.AccessibleRole.PRESENTATION,
        )
        check.add_css_class("dim-label")
        row.add_suffix(check)
        # A list box overwrites its rows' selected state, so the name carries it.
        row.reset_relation(Gtk.AccessibleRelation.LABELLED_BY)
        row.update_property(
            [Gtk.AccessibleProperty.LABEL], [f"{row.get_title()} Active"]
        )

    def _on_selected(self, skin_id: str) -> None:
        current = self._bridge.current()
        if current not in self._skin_rows:
            current = None
        action = _skin_action(skin_id, current)
        if action == "confirm":
            self._confirm_overwrite(skin_id)
        elif action == "apply":
            self._apply_skin(skin_id)

    def _confirm_overwrite(self, skin_id: str) -> None:
        """Ask before copying `skin_id` over a skin.toml that matches no known skin.

        The copy replaces the whole file, so changes not saved as a skin are lost.
        """
        window = self._window
        if window is None:
            return
        dialog = Adw.AlertDialog(
            heading="Replace your app skin?",
            body=(
                "Your skin.toml has changes that aren't in a saved skin. "
                f"Applying {_skin_label(skin_id)} replaces the whole file."
            ),
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("replace", "Replace")
        dialog.set_response_appearance("replace", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_overwrite_response, skin_id, window)
        dialog.connect("map", brand.pointer_cursors)
        dialog.present(window)

    def _on_overwrite_response(
        self,
        _dialog,
        response: str,
        skin_id: str,
        window: Adw.PreferencesWindow,
    ) -> None:
        if response == "replace" and self._window is window:
            self._apply_skin(skin_id)

    def _apply_skin(self, skin_id: str) -> None:
        try:
            self._bridge.apply(skin_id)
        except ValueError as exc:
            logger.warning("Could not apply skin {}: {}", diagnostic(skin_id), exc)
            self._toast(str(exc))
        except OSError as exc:
            logger.error("Could not apply skin {}: {}", diagnostic(skin_id), exc)
            self._toast("Couldn't confirm the change. Check the selected skin.")
        self._render_skin_rows()  # to the truth on disk, applied or not
