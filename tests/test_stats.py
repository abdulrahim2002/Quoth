# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `stats`, the local usage stats."""

import json

from quoth import stats


class TestLoad:
    def test_missing_file_is_defaults(self, tmp_path):
        state = stats.load(tmp_path / "absent.json")
        assert state["dictations_total"] == 0
        assert state["first_seen"] is None
        assert state["nudge_shown"] is False

    def test_corrupt_file_is_defaults(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("{ not json")
        assert stats.load(path)["dictations_total"] == 0

    def test_non_object_json_is_defaults(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("[1, 2, 3]")
        assert stats.load(path)["dictations_total"] == 0

    def test_partial_file_is_filled_from_defaults(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"dictations_total": 5}))
        state = stats.load(path)
        assert state["dictations_total"] == 5
        assert state["nudge_shown"] is False  # filled in

    def test_wrong_types_revert_to_defaults(self, tmp_path):
        path = tmp_path / "stats.json"
        # A bool must not pass as the integer count, although bool subclasses int.
        path.write_text(
            json.dumps(
                {"dictations_total": True, "nudge_shown": "yes", "first_seen": 5}
            )
        )
        state = stats.load(path)
        assert state["dictations_total"] == 0
        assert state["nudge_shown"] is False
        assert state["first_seen"] is None

    def test_a_text_count_reverts_to_zero(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"dictations_total": "5"}))
        assert stats.load(path)["dictations_total"] == 0


class TestSave:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "stats.json"
        stats.save(path, {"dictations_total": 7, "first_seen": "2026-07-17"})
        assert stats.load(path)["dictations_total"] == 7

    def test_writes_owner_only_permissions(self, tmp_path):
        path = tmp_path / "stats.json"
        stats.save(path, dict(stats._DEFAULTS))
        assert (path.stat().st_mode & 0o777) == 0o600

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "stats.json"
        stats.save(path, dict(stats._DEFAULTS))
        assert path.exists()

    def test_a_failed_write_warns_and_is_swallowed(self, tmp_path, caplog):
        (tmp_path / "state").write_text("not a directory")
        caplog.set_level("WARNING")
        stats.save(tmp_path / "state" / "stats.json", dict(stats._DEFAULTS))
        assert [r.levelname for r in caplog.records] == ["WARNING"]


class TestRecordDictation:
    def test_increments_total(self):
        state = stats.record_dictation(dict(stats._DEFAULTS), today="2026-07-17")
        assert state["dictations_total"] == 1

    def test_stamps_first_seen_once(self):
        first = stats.record_dictation(dict(stats._DEFAULTS), today="2026-07-17")
        second = stats.record_dictation(first, today="2026-08-01")
        assert first["first_seen"] == "2026-07-17"
        assert second["first_seen"] == "2026-07-17"  # not overwritten

    def test_does_not_mutate_input(self):
        original = dict(stats._DEFAULTS)
        stats.record_dictation(original, today="2026-07-17")
        assert original["dictations_total"] == 0


class TestNudge:
    def test_not_due_below_threshold(self):
        state = {**stats._DEFAULTS, "dictations_total": stats.NUDGE_THRESHOLD - 1}
        assert stats.due_for_nudge(state) is False

    def test_due_at_threshold(self):
        state = {**stats._DEFAULTS, "dictations_total": stats.NUDGE_THRESHOLD}
        assert stats.due_for_nudge(state) is True

    def test_not_due_once_shown(self):
        state = {
            **stats._DEFAULTS,
            "dictations_total": stats.NUDGE_THRESHOLD,
            "nudge_shown": True,
        }
        assert stats.due_for_nudge(state) is False

    def test_mark_nudged_sets_flag(self):
        assert stats.mark_nudged(dict(stats._DEFAULTS))["nudge_shown"] is True


def test_default_path_lives_under_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert stats.default_path() == tmp_path / "state" / "quoth" / "stats.json"
