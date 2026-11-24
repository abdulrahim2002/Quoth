# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `atomic_write`."""

import os
import stat
import tempfile
from pathlib import Path

import pytest

from quoth import atomic_write


class TestCreateBytes:
    def test_creates_private_exact_bytes_without_debris(self, tmp_path: Path):
        target = tmp_path / "theme.toml"

        atomic_write.create_bytes(target, b"schema = 1\r\n")

        assert target.read_bytes() == b"schema = 1\r\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert [path.name for path in tmp_path.iterdir()] == ["theme.toml"]

    def test_publishes_only_after_file_sync_then_cleans_before_directory_sync(
        self, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "theme.toml"
        events = []
        real_fsync = os.fsync
        real_link = os.link
        real_unlink = Path.unlink

        def fsync(fd):
            events.append(
                "dir-sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-sync"
            )
            real_fsync(fd)

        def link(src, dst):
            source = Path(src)
            assert source.parent == tmp_path  # one filesystem, so the link can publish
            assert source.name.startswith(".theme.toml.")
            assert source.name.endswith(".tmp")
            assert not source.name.endswith(".toml")
            assert source.read_bytes() == b"complete"
            assert not Path(dst).exists()
            events.append("link")
            real_link(src, dst)

        def unlink(path, *args, **kwargs):
            events.append("unlink")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "link", link)
        monkeypatch.setattr(Path, "unlink", unlink)

        atomic_write.create_bytes(target, b"complete")

        assert events[-4:] == ["file-sync", "link", "unlink", "dir-sync"]

    def test_existing_file_is_untouched(self, tmp_path: Path):
        target = tmp_path / "theme.toml"
        target.write_bytes(b"first")

        with pytest.raises(FileExistsError):
            atomic_write.create_bytes(target, b"second")

        assert target.read_bytes() == b"first"
        assert [path.name for path in tmp_path.iterdir()] == ["theme.toml"]

    def test_existing_symlink_and_its_target_are_untouched(self, tmp_path: Path):
        victim = tmp_path / "victim"
        victim.write_bytes(b"victim")
        target = tmp_path / "theme.toml"
        target.symlink_to(victim)

        with pytest.raises(FileExistsError):
            atomic_write.create_bytes(target, b"replacement")

        assert target.is_symlink()
        assert victim.read_bytes() == b"victim"
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "theme.toml",
            "victim",
        ]

    @pytest.mark.parametrize("failure", ["write", "flush", "close"])
    def test_stream_failure_leaves_no_destination_or_debris(
        self, tmp_path: Path, monkeypatch, failure: str
    ):
        real_named_temporary_file = tempfile.NamedTemporaryFile

        class BrokenTemporaryFile:
            def __init__(self, *args, **kwargs):
                self._file = real_named_temporary_file(*args, **kwargs)
                self.name = self._file.name

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                self._file.close()
                if failure == "close":
                    raise OSError("close failed")

            def write(self, data):
                if failure == "write":
                    self._file.write(data[:1])
                    raise OSError("write failed")
                return self._file.write(data)

            def flush(self):
                if failure == "flush":
                    raise OSError("flush failed")
                self._file.flush()

            def fileno(self):
                return self._file.fileno()

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", BrokenTemporaryFile)
        target = tmp_path / "theme.toml"

        with pytest.raises(OSError, match=failure):
            atomic_write.create_bytes(target, b"complete")

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_file_sync_failure_leaves_no_destination_or_debris(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(atomic_write, "sync_directory", lambda _path: None)
        monkeypatch.setattr(
            os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(OSError("file sync failed")),
        )
        target = tmp_path / "theme.toml"

        with pytest.raises(OSError, match="file sync failed"):
            atomic_write.create_bytes(target, b"complete")

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_link_failure_has_no_unsafe_fallback_or_debris(
        self, tmp_path: Path, monkeypatch
    ):
        def unsupported(_src, _dst):
            raise OSError("hard links unsupported")

        monkeypatch.setattr(os, "link", unsupported)
        target = tmp_path / "theme.toml"

        with pytest.raises(OSError, match="hard links unsupported"):
            atomic_write.create_bytes(target, b"complete")

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_destination_sync_failure_leaves_published_file(
        self, tmp_path: Path, monkeypatch
    ):
        real_sync_directory = atomic_write.sync_directory

        def sync_directory(path):
            if path == tmp_path:
                raise OSError("directory sync failed")
            real_sync_directory(path)

        monkeypatch.setattr(atomic_write, "sync_directory", sync_directory)
        target = tmp_path / "theme.toml"

        with pytest.raises(OSError, match="directory sync failed"):
            atomic_write.create_bytes(target, b"complete")

        assert target.read_bytes() == b"complete"
        assert [path.name for path in tmp_path.iterdir()] == ["theme.toml"]

    def test_new_parent_sync_failure_aborts_publication_and_retry_repairs_it(
        self, tmp_path: Path, monkeypatch
    ):
        parent = tmp_path / "config" / "themes"
        target = parent / "shared.toml"
        real_sync_directory = atomic_write.sync_directory
        failed = False
        parent_syncs = 0

        def sync_directory(path):
            nonlocal failed, parent_syncs
            if path == parent.parent:
                parent_syncs += 1
                if not failed:
                    failed = True
                    raise OSError("parent sync failed")
            real_sync_directory(path)

        monkeypatch.setattr(atomic_write, "sync_directory", sync_directory)

        with pytest.raises(OSError, match="parent sync failed"):
            atomic_write.create_bytes(target, b"complete")
        assert parent.is_dir()
        assert not target.exists()

        atomic_write.create_bytes(target, b"complete")

        assert parent_syncs == 2
        assert target.read_bytes() == b"complete"
        assert [path.name for path in parent.iterdir()] == ["shared.toml"]


class TestReplaceBytes:
    def test_replaces_with_exact_private_bytes_and_no_debris(self, tmp_path: Path):
        target = tmp_path / "skin.toml"
        target.write_bytes(b"old")

        atomic_write.replace_bytes(target, b"kind = 'skin'\r\n")

        assert target.read_bytes() == b"kind = 'skin'\r\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert [path.name for path in tmp_path.iterdir()] == ["skin.toml"]

    def test_syncs_file_before_replace_and_directory_after(
        self, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "skin.toml"
        target.write_bytes(b"old")
        events = []
        real_fsync = os.fsync
        real_replace = os.replace

        def fsync(fd):
            events.append(
                "dir-sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-sync"
            )
            real_fsync(fd)

        def replace(source, destination):
            assert Path(source).parent == tmp_path  # one filesystem, so it is atomic
            events.append("replace")
            real_replace(source, destination)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)

        atomic_write.replace_bytes(target, b"new")

        assert events[-3:] == ["file-sync", "replace", "dir-sync"]

    def test_optional_mode_is_set_before_file_sync(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "theme.toml"
        target.write_bytes(b"old")
        events = []
        real_fchmod = os.fchmod
        real_fsync = os.fsync

        def fchmod(fd, mode):
            events.append(("chmod", mode))
            real_fchmod(fd, mode)

        def fsync(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                events.append(("sync", stat.S_IMODE(os.fstat(fd).st_mode)))
            real_fsync(fd)

        monkeypatch.setattr(os, "fchmod", fchmod)
        monkeypatch.setattr(os, "fsync", fsync)

        atomic_write.replace_bytes(target, b"new", mode=0o640)

        assert events[:2] == [("chmod", 0o640), ("sync", 0o640)]
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    def test_failed_file_sync_keeps_original_and_cleans_temp(
        self, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "skin.toml"
        target.write_bytes(b"old")
        real_fsync = os.fsync

        def fsync(fd):
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("disk failed")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)

        with pytest.raises(OSError, match="disk failed"):
            atomic_write.replace_bytes(target, b"new")

        assert target.read_bytes() == b"old"
        assert [path.name for path in tmp_path.iterdir()] == ["skin.toml"]

    def test_failed_replace_keeps_original_and_cleans_temp(
        self, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "skin.toml"
        target.write_bytes(b"old")
        monkeypatch.setattr(
            os,
            "replace",
            lambda _source, _destination: (_ for _ in ()).throw(
                OSError("replace failed")
            ),
        )

        with pytest.raises(OSError, match="replace failed"):
            atomic_write.replace_bytes(target, b"new")

        assert target.read_bytes() == b"old"
        assert [path.name for path in tmp_path.iterdir()] == ["skin.toml"]

    def test_replaces_symlink_without_touching_its_target(self, tmp_path: Path):
        victim = tmp_path / "victim"
        victim.write_bytes(b"victim")
        target = tmp_path / "skin.toml"
        target.symlink_to(victim)

        atomic_write.replace_bytes(target, b"skin")

        assert not target.is_symlink()
        assert target.read_bytes() == b"skin"
        assert victim.read_bytes() == b"victim"


class TestRemove:
    def test_removes_entry_then_syncs_directory(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "skin.toml"
        target.write_bytes(b"skin")
        events = []
        real_unlink = Path.unlink
        real_sync_directory = atomic_write.sync_directory

        def unlink(path, *args, **kwargs):
            events.append("unlink")
            return real_unlink(path, *args, **kwargs)

        def sync_directory(path):
            events.append("dir-sync")
            real_sync_directory(path)

        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(atomic_write, "sync_directory", sync_directory)

        atomic_write.remove(target)

        assert events == ["unlink", "dir-sync"]
        assert not target.exists()

    def test_missing_path_is_already_removed(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            atomic_write,
            "sync_directory",
            lambda _path: (_ for _ in ()).throw(AssertionError("unexpected sync")),
        )

        atomic_write.remove(tmp_path / "missing.toml")

    def test_removes_symlink_without_touching_target(self, tmp_path: Path):
        victim = tmp_path / "victim"
        victim.write_bytes(b"victim")
        target = tmp_path / "skin.toml"
        target.symlink_to(victim)

        atomic_write.remove(target)

        assert not target.exists()
        assert victim.read_bytes() == b"victim"


class TestWriteText:
    def test_creates_the_file(self, tmp_path: Path):
        target = tmp_path / "thing.json"
        atomic_write.write_text(target, "hello\n")
        assert target.read_text() == "hello\n"

    def test_temp_name_keeps_the_full_filename(self, tmp_path: Path):
        """`with_suffix` would map config.json and config.toml to one config.tmp."""
        seen = []
        real_replace = os.replace

        def spy(src, dst):
            seen.append(Path(src).name)
            real_replace(src, dst)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", spy)
            atomic_write.write_text(tmp_path / "config.json", "{}")
        [name] = seen
        assert name.startswith(".config.json.") and name.endswith(".tmp")

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        target = tmp_path / "state" / "quoth" / "thing.json"
        atomic_write.write_text(target, "x")
        assert target.read_text() == "x"

    def test_applies_mode_before_the_swap(self, tmp_path: Path):
        target = tmp_path / "secret.json"
        swapped = []
        real_replace = os.replace

        def spy(src, dst):
            swapped.append(stat.S_IMODE(os.stat(src).st_mode))
            real_replace(src, dst)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", spy)
            # Not 0o600, which a new temp file already has.
            atomic_write.write_text(target, "x", mode=0o640)
        assert swapped == [0o640]
        assert target.stat().st_mode & 0o777 == 0o640


class TestReadState:
    @pytest.mark.parametrize("content", [None, b"\xff\xfe{", b"{not json", b"[1, 2]"])
    def test_missing_unreadable_or_non_object_state_is_none(self, tmp_path, content):
        path = tmp_path / "state.json"
        if content is not None:
            path.write_bytes(content)
        assert atomic_write.read_state(path, "State") is None

    def test_an_object_is_returned(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text('{"last_check": 1}')
        assert atomic_write.read_state(path, "State") == {"last_check": 1}
