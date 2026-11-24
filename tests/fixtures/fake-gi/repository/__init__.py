# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""The `gi.repository` namespaces the installed check uses."""


class Adw:
    Application = object


class _MediaFile:
    def __init__(self):
        self.prepared = False
        self.callbacks = []

    def connect(self, _signal, callback):
        self.callbacks.append(callback)

    def set_filename(self, _path):
        self.prepared = True
        for callback in self.callbacks:
            callback(self, None)

    def get_error(self):
        return None

    def is_prepared(self):
        return self.prepared

    def has_video(self):
        return True


class _MediaFactory:
    @staticmethod
    def new():
        return _MediaFile()


class Gtk:
    Window = object
    MediaFile = _MediaFactory


class _Loop:
    def is_running(self):
        return False

    def run(self):
        raise AssertionError("prepared media entered main loop")

    def quit(self):
        pass


class GLib:
    SOURCE_REMOVE = False
    MainLoop = _Loop

    @staticmethod
    def timeout_add(*args):
        return 1

    @staticmethod
    def source_remove(*args):
        pass


class Gtk4LayerShell:
    @staticmethod
    def init_for_window(*args):
        pass


class _Schemas:
    def lookup(self, schema_id, _recursive):
        return object() if schema_id == "dev.quoth.Quoth" else None


class _SchemaSource:
    @staticmethod
    def get_default():
        return _Schemas()


class Gio:
    SettingsSchemaSource = _SchemaSource


class _Portal:
    initable_new = create_global_shortcuts_session = object()
    create_global_shortcuts_session_finish = object()
    request_background = request_background_finish = object()


class _GlobalShortcut:
    new = object()


class _GlobalShortcutAssigned:
    get_shortcut_id = get_trigger_description = object()


class _GlobalShortcutsSession:
    get_session = bind_shortcuts = bind_shortcuts_finish = object()
    configure_shortcuts = configure_shortcuts_finish = close = object()


class _Session:
    close = object()


class Xdp:
    Portal = _Portal
    GlobalShortcut = _GlobalShortcut
    GlobalShortcutAssigned = _GlobalShortcutAssigned
    GlobalShortcutsSession = _GlobalShortcutsSession
    Session = _Session
