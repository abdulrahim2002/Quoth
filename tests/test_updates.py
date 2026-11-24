# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `updates`, the pure core of the opt-in update check."""

import errno

import pytest

from quoth import updates


class TestParseVersion:
    @pytest.mark.parametrize(
        ("tag", "version"),
        [
            ("0.1.1", (0, 1, 1)),
            ("v0.1.1", (0, 1, 1)),
            # GitHub's "latest" excludes prereleases; a prerelease tag still parses.
            ("v1.2.0-rc1", (1, 2, 0)),
            ("v1.2.0+build.5", (1, 2, 0)),
            ("nightly", None),
            ("", None),
            # Digits int() cannot read, such as a superscript, fail safe too.
            ("v1.\u00b2.0", None),
        ],
    )
    def test_parse(self, tag, version):
        assert updates.parse_version(tag) == version


class TestCheckForUpdate:
    # Only a newer release names a version, normalised without its "v".
    @pytest.mark.parametrize(
        ("latest", "version"),
        [("v0.2.0", "0.2.0"), ("v0.1.0", None), ("v0.0.9", None)],
    )
    def test_a_reached_check_names_only_a_newer_version(self, latest, version):
        result = updates.check_for_update("0.1.0", fetch=lambda: latest)
        assert result == updates.UpdateResult(version=version, reached=True)

    def test_a_reached_check_reads_the_latest_release_tag(self, monkeypatch):
        import contextlib
        import io

        requests = []

        def urlopen(request, timeout=None):
            requests.append((request.full_url, timeout))
            return contextlib.nullcontext(io.BytesIO(b'{"tag_name": "v0.2.0"}'))

        monkeypatch.setattr(updates.urllib.request, "urlopen", urlopen)
        assert updates.check_for_update("0.1.0") == updates.UpdateResult(
            version="0.2.0", reached=True
        )
        # The release API, with a timeout so an unanswered check gives up.
        ((url, timeout),) = requests
        assert url == updates.API_URL
        assert timeout

    def test_a_release_without_a_tag_is_not_reached(self, monkeypatch):
        import contextlib
        import io

        monkeypatch.setattr(
            updates.urllib.request,
            "urlopen",
            lambda *_a, **_k: contextlib.nullcontext(io.BytesIO(b'{"tag_name": null}')),
        )
        assert updates.check_for_update("0.1.0") == updates.UpdateResult(
            version=None, reached=False
        )

    def test_a_defect_is_not_mistaken_for_offline(self):
        def defect():
            raise AttributeError("bug")

        with pytest.raises(AttributeError):
            updates.check_for_update("0.1.0", fetch=defect)

    def test_fetch_error_is_not_reached(self, caplog):
        def boom():
            raise OSError(errno.ENETUNREACH, "Network is unreachable")

        # A miss must be distinguishable from "up to date": reached is False, so the
        # caller shows "couldn't reach" and the throttle doesn't count it.
        caplog.set_level("WARNING")
        assert updates.check_for_update("0.1.0", fetch=boom) == updates.UpdateResult(
            version=None, reached=False
        )
        assert caplog.messages == [
            "Update check failed: OSError: [Errno 101] Network is unreachable"
        ]

    def test_a_malformed_proxy_setting_never_reaches_the_log(
        self, malformed_proxy, caplog
    ):
        assert not updates.check_for_update("0.1.0").reached
        assert caplog.messages[-1] == "Update check failed: InvalidURL"
        assert not any("s3cret" in message for message in caplog.messages)

    def test_malformed_tag_is_not_reached(self):
        # The server answered but no version could be read: a miss, not "up to date".
        assert updates.check_for_update(
            "0.1.0", fetch=lambda: "garbage"
        ) == updates.UpdateResult(version=None, reached=False)


class TestDueForCheck:
    def test_never_checked_is_due(self):
        assert updates.due_for_check({}, now=1000.0) is True

    def test_recent_check_is_not_due(self):
        state = {"last_check": 1_700_000_000.0}
        assert (
            updates.due_for_check(state, now=1_700_000_100.0, interval=86400) is False
        )

    def test_old_check_is_due(self):
        state = {"last_check": 1000.0}
        assert updates.due_for_check(state, now=1000.0 + 86400, interval=86400) is True


class TestRegisterResult:
    @staticmethod
    def _reached(version):
        return updates.UpdateResult(version=version, reached=True)

    def test_records_last_check_even_when_up_to_date(self):
        state = {}
        assert updates.register_result(state, self._reached(None), now=123.0) is False
        assert state["last_check"] == 123.0
        assert "last_notified" not in state

    def test_new_version_notifies_and_records(self):
        state = {}
        assert updates.register_result(state, self._reached("0.2.0"), now=5.0) is True
        assert state == {"last_check": 5.0, "last_notified": "0.2.0"}

    def test_same_version_does_not_renotify(self):
        state = {"last_notified": "0.2.0"}
        assert updates.register_result(state, self._reached("0.2.0"), now=9.0) is False
        assert state["last_check"] == 9.0  # but the check is still recorded

    def test_a_further_new_version_notifies_again(self):
        state = {"last_notified": "0.2.0"}
        assert updates.register_result(state, self._reached("0.3.0"), now=9.0) is True
        assert state["last_notified"] == "0.3.0"

    def test_unreached_check_records_nothing(self):
        # A failed check must not stamp last_check, or an offline machine stays
        # update-blind for a day after it reconnects.
        state = {}
        miss = updates.UpdateResult(version=None, reached=False)
        assert updates.register_result(state, miss, now=123.0) is False
        assert state == {}

    def test_unreached_check_preserves_existing_state(self):
        state = {"last_check": 100.0, "last_notified": "0.2.0"}
        miss = updates.UpdateResult(version=None, reached=False)
        assert updates.register_result(state, miss, now=999.0) is False
        assert state == {"last_check": 100.0, "last_notified": "0.2.0"}


class TestState:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "update_check.json"
        updates.save_state(path, {"last_check": 42.0, "last_notified": "0.2.0"})
        assert updates.load_state(path) == {
            "last_check": 42.0,
            "last_notified": "0.2.0",
        }

    def test_missing_file_is_empty(self, tmp_path):
        assert updates.load_state(tmp_path / "absent.json") == {}

    def test_corrupt_file_is_empty(self, tmp_path):
        path = tmp_path / "update_check.json"
        path.write_text("{not json")
        assert updates.load_state(path) == {}

    def test_a_failed_save_is_logged_not_raised(self, tmp_path, caplog):
        blocker = tmp_path / "state"
        blocker.write_text("a file where the directory belongs")
        caplog.set_level("WARNING")
        updates.save_state(blocker / "update_check.json", {"last_check": 1.0})
        assert [r.levelname for r in caplog.records] == ["WARNING"]


class TestCurrentChannel:
    @staticmethod
    def _clear(monkeypatch):
        for var in ("QUOTH_CHANNEL", "SNAP", "APPIMAGE"):
            monkeypatch.delenv(var, raising=False)

    def test_env_override_beats_everything(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("QUOTH_CHANNEL", "deb")
        monkeypatch.setattr(updates, "in_flatpak", lambda: True)
        assert updates.current_channel() == "deb"

    def test_flatpak_runtime_marker(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setattr(updates, "in_flatpak", lambda: True)
        assert updates.current_channel() == "flatpak"

    def test_snap_runtime_marker(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setattr(updates, "in_flatpak", lambda: False)
        monkeypatch.setenv("SNAP", "/snap/quoth/42")
        assert updates.current_channel() == "snap"

    def test_appimage_runtime_marker(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setattr(updates, "in_flatpak", lambda: False)
        monkeypatch.setenv("APPIMAGE", "/tmp/Quoth.AppImage")
        assert updates.current_channel() == "appimage"

    def test_build_stamp_when_no_runtime_marker(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setattr(updates, "in_flatpak", lambda: False)
        monkeypatch.setattr(updates, "_read_stamp", lambda: "aur")
        assert updates.current_channel() == "aur"

    def test_unknown_when_nothing_detected(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setattr(updates, "in_flatpak", lambda: False)
        monkeypatch.setattr(updates, "_read_stamp", lambda: None)
        assert updates.current_channel() == "unknown"


class TestUpdateHint:
    def test_deb_distinguishes_repository_and_direct_installs(self):
        hint = updates.update_hint("deb").lower()
        assert "release" in hint
        assert "if you added quoth's apt repository" in hint
        assert "otherwise, download" in hint

    def test_aur_covers_helper_and_repository_builds(self):
        hint = updates.update_hint("aur")
        assert "AUR helper" in hint
        assert "rebuild the package from the new release tag" in hint

    def test_flatpak_covers_downloads_and_repository_updates(self):
        hint = updates.update_hint("flatpak")
        assert "Download the new .flatpak" in hint
        assert "repository installs" in hint
        assert "flatpak update" in hint

    @pytest.mark.parametrize("channel", list(updates._HINTS))
    def test_each_hint_is_spaced_as_a_sentence(self, channel):
        hint = updates.update_hint(channel)
        assert "  " not in hint and hint == hint.strip()

    def test_unknown_is_generic_and_nonempty(self):
        assert updates.update_hint("unknown")

    def test_unrecognized_channel_falls_back_to_unknown(self):
        assert updates.update_hint("weird") == updates.update_hint("unknown")

    def test_defaults_to_current_channel(self, monkeypatch):
        monkeypatch.setattr(updates, "current_channel", lambda: "deb")
        assert updates.update_hint() == updates.update_hint("deb")


class TestStatusLine:
    @pytest.mark.parametrize(
        ("version", "checking", "checked", "failed", "line"),
        [
            # Even with everything else set, an in-flight check wins.
            ("0.2.0", True, True, True, "Checking…"),
            (None, False, True, True, "Couldn't check for updates"),
            (None, False, True, False, "Up to date"),
            (None, False, False, False, "Not checked yet"),
        ],
    )
    def test_line(self, version, checking, checked, failed, line):
        assert (
            updates.status_line(
                version, checking=checking, checked=checked, failed=failed
            )
            == line
        )

    def test_available_shows_version_and_hint(self, monkeypatch):
        monkeypatch.setattr(updates, "update_hint", lambda: "Run: foo")
        line = updates.status_line("0.2.0", checking=False, checked=True, failed=False)
        assert line == "0.2.0 available. Run: foo"

    def test_available_outranks_failed(self):
        # A known update from a prior success beats a transient failed re-check.
        line = updates.status_line("0.2.0", checking=False, checked=True, failed=True)
        assert line.startswith("0.2.0 available")


def test_default_path_lives_under_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert updates.default_path() == tmp_path / "state" / "quoth" / "update_check.json"
