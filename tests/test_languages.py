# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.languages`, the language catalog."""

import pytest

from quoth.languages import LANGUAGES, default_from_locale

_LOCALE_VARS = ("LC_ALL", "LC_MESSAGES", "LANG")


class TestCatalog:
    def test_first_entry_is_auto_detect(self):
        assert LANGUAGES[0] == ("Auto-detect", "auto")

    def test_english_and_german_present(self):
        codes = {code for _, code in LANGUAGES}
        assert {"en", "de"} <= codes

    def test_codes_are_unique(self):
        codes = [code for _, code in LANGUAGES]
        assert len(codes) == len(set(codes))

    def test_labels_are_unique(self):
        labels = [label for label, _ in LANGUAGES]
        assert len(labels) == len(set(labels))

    def test_entries_are_label_code_pairs(self):
        for entry in LANGUAGES:
            assert len(entry) == 2
            label, code = entry
            assert label and code


class TestDefaultFromLocale:
    def _clear(self, monkeypatch):
        for var in _LOCALE_VARS:
            monkeypatch.delenv(var, raising=False)

    @pytest.mark.parametrize(
        ("locale", "language"),
        [
            ("de_DE.UTF-8", "de"),
            # Every Norwegian locale is Norwegian.
            ("nb_NO.UTF-8", "no"),
            ("nn_NO.UTF-8", "no"),
            ("no_NO.UTF-8", "no"),
            ("eu_ES.UTF-8", "en"),  # Basque, not in the catalog
            ("C.UTF-8", "en"),
            ("ja", "ja"),  # a bare subtag without a region
        ],
    )
    def test_lang_picks_a_catalog_language(self, monkeypatch, locale, language):
        self._clear(monkeypatch)
        monkeypatch.setenv("LANG", locale)
        assert default_from_locale() == language

    def test_unset_locale_falls_back_to_english(self, monkeypatch):
        self._clear(monkeypatch)
        assert default_from_locale() == "en"

    def test_lc_all_takes_precedence(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
        assert default_from_locale() == "fr"

    def test_lc_messages_takes_precedence_over_lang(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("LC_MESSAGES", "de_DE.UTF-8")
        assert default_from_locale() == "de"

    def test_lc_all_takes_precedence_over_lc_messages(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("LC_MESSAGES", "de_DE.UTF-8")
        monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
        assert default_from_locale() == "fr"

    def test_an_empty_variable_counts_as_unset(self, monkeypatch):
        # Sessions often export an empty LC_ALL, which POSIX reads as unset.
        self._clear(monkeypatch)
        monkeypatch.setenv("LC_ALL", "")
        monkeypatch.setenv("LANG", "es_ES.UTF-8")
        assert default_from_locale() == "es"
