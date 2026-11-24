# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Transcription language catalog.

The languages Quoth offers, as `(display label, whisper code)` pairs in menu order,
shared by the Settings combo and the tray submenu. The code lands in
`config.language`; `"auto"` lets whisper detect the spoken language.
"""

import os

# (display label, whisper code), in menu order.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Swedish", "sv"),
    ("Norwegian", "no"),
    ("Danish", "da"),
    ("Finnish", "fi"),
)

_SUPPORTED_CODES = frozenset(code for _, code in LANGUAGES)

# glibc names Norwegian locales by written standard (`nb_NO`, `nn_NO`); the catalog
# offers whisper's one Norwegian, `no`.
_LOCALE_ALIASES = {"nb": "no", "nn": "no"}


def default_from_locale() -> str:
    """Return the starting transcription language, derived from the system locale.

    That is the language subtag of the highest-precedence set locale variable
    (`de_DE.UTF-8` -> `de`, `nb_NO` -> `no`) when Quoth offers that language, else
    `en`. It seeds `config.language` on first run so a non-English speaker's first
    dictation is not silently forced to English.
    """
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var)
        if value:
            subtag = value.split("_", 1)[0].split(".", 1)[0].lower()
            subtag = _LOCALE_ALIASES.get(subtag, subtag)
            return subtag if subtag in _SUPPORTED_CODES else "en"
    return "en"
