# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Samuel Ruairí Bullard

"""Tests for `quoth.models`: the model catalog and on-disk resolution."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from quoth import paths
from quoth.models import (
    AVAILABLE_MODELS,
    MODELS_BY_NAME,
    ModelManager,
    encoder_bin_name,
    encoder_xml_name,
    ggml_name,
    model_label,
    needs_openvino_encoder,
)


class TestAvailableModels:
    def test_catalog_set(self):
        assert {m.name for m in AVAILABLE_MODELS} == {
            "tiny",
            "base",
            "small",
            "large-v3-turbo",
        }

    def test_ggml_sizes_positive(self):
        for m in AVAILABLE_MODELS:
            assert m.ggml_size_mb > 0

    def test_all_models_are_npu_capable(self):
        # Every catalog model compiles and runs on the NPU, so each ships an fp16
        # OpenVINO encoder.
        assert all(m.has_openvino_encoder for m in AVAILABLE_MODELS)

    def test_turbo_present_and_largest(self):
        turbo = MODELS_BY_NAME["large-v3-turbo"]
        assert turbo.has_openvino_encoder is True
        assert turbo.ggml_size_mb == max(m.ggml_size_mb for m in AVAILABLE_MODELS)

    def test_labels_name_each_model_and_pass_unknown_names_through(self):
        assert [model_label(m.name) for m in AVAILABLE_MODELS] == [
            "Tiny",
            "Base",
            "Small",
            "Turbo",
        ]
        assert model_label("mystery") == "mystery"

    def test_only_turbo_lacks_translation(self):
        assert MODELS_BY_NAME["large-v3-turbo"].supports_translation is False
        assert all(
            model.supports_translation
            for model in AVAILABLE_MODELS
            if model.name != "large-v3-turbo"
        )


class TestFilenames:
    def test_ggml_name(self):
        assert ggml_name("small") == "ggml-small-q8_0.bin"

    def test_encoder_names(self):
        assert encoder_xml_name("base") == "ggml-base-encoder-openvino.xml"
        assert encoder_bin_name("base") == "ggml-base-encoder-openvino.bin"


class TestNeedsOpenvinoEncoder:
    @pytest.mark.parametrize(
        ("device", "model", "needed"),
        [
            ("NPU", "small", True),
            ("NPU", "base", True),
            ("NPU", "large-v3-turbo", True),
            # GPU is Vulkan: full encode and decode, no OpenVINO encoder asset.
            ("GPU", "small", False),
            ("GPU", "base", False),
            ("CPU", "small", False),
        ],
    )
    def test_only_the_npu_needs_the_encoder(self, device, model, needed):
        assert needs_openvino_encoder(device, model) is needed


class TestModelManager:
    def test_list_downloaded_empty(self, tmp_data_dir):
        assert ModelManager(models_dir=tmp_data_dir).list_downloaded() == []

    def test_an_unusable_directory_is_logged_with_its_path(self, tmp_path, caplog):
        blocked = tmp_path / "file" / "models"
        (tmp_path / "file").touch()
        with pytest.raises(NotADirectoryError):
            ModelManager(models_dir=blocked)
        assert caplog.messages[0].startswith(f"Models directory {blocked} unusable: ")

    def test_list_downloaded_finds_ggml(self, tmp_data_dir):
        (tmp_data_dir / "ggml-small-q8_0.bin").write_bytes(b"\x00" * 100)
        assert "small" in ModelManager(models_dir=tmp_data_dir).list_downloaded()

    def test_is_downloaded(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        assert mgr.is_downloaded("base") is False
        (tmp_data_dir / "ggml-base-q8_0.bin").write_bytes(b"\x00" * 100)
        assert mgr.is_downloaded("base") is True

    def test_has_openvino_requires_both_files(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        assert mgr.has_openvino("small") is False
        (tmp_data_dir / "ggml-small-encoder-openvino.xml").touch()
        assert mgr.has_openvino("small") is False  # partial
        (tmp_data_dir / "ggml-small-encoder-openvino.bin").touch()
        assert mgr.has_openvino("small") is True

    def test_model_path(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        assert mgr.model_path("tiny") is None
        (tmp_data_dir / "ggml-tiny-q8_0.bin").touch()
        assert mgr.model_path("tiny") == tmp_data_dir / "ggml-tiny-q8_0.bin"

    def test_openvino_path(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        assert mgr.openvino_path("base") is None
        (tmp_data_dir / "ggml-base-encoder-openvino.xml").touch()
        assert (
            mgr.openvino_path("base") == tmp_data_dir / "ggml-base-encoder-openvino.xml"
        )

    def test_vad_model_path_falls_back_to_the_bundled_weights(self, tmp_data_dir):
        # Shipped in the package, so VAD works on a fresh install and survives the user
        # deleting the copy in their data dir.
        from quoth.models import _BUNDLED_VAD, VAD_MODEL_FILE

        mgr = ModelManager(models_dir=tmp_data_dir)
        assert _BUNDLED_VAD.exists(), "the VAD weights are not bundled in the package"
        assert mgr.vad_model_path() == _BUNDLED_VAD

        # A copy in the data dir wins, so a user-provisioned file still takes effect.
        (tmp_data_dir / VAD_MODEL_FILE).touch()
        assert mgr.vad_model_path() == tmp_data_dir / VAD_MODEL_FILE

    def test_vad_model_path_is_none_without_any_weights(
        self, tmp_data_dir, monkeypatch
    ):
        # A source tree without the asset leaves VAD off rather than failing.
        from quoth import models

        monkeypatch.setattr(models, "_BUNDLED_VAD", tmp_data_dir / "absent.bin")
        assert ModelManager(models_dir=tmp_data_dir).vad_model_path() is None

    def test_delete_model_removes_all_files(self, tmp_data_dir):
        for n in (
            "ggml-small-q8_0.bin",
            "ggml-small-encoder-openvino.xml",
            "ggml-small-encoder-openvino.bin",
        ):
            (tmp_data_dir / n).touch()
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.delete_model("small")
        assert not any(
            (tmp_data_dir / n).exists()
            for n in (
                "ggml-small-q8_0.bin",
                "ggml-small-encoder-openvino.xml",
                "ggml-small-encoder-openvino.bin",
            )
        )

    def test_delete_model_removes_download_partials(self, tmp_data_dir):
        # A partial left by a failed download is space only deleting can give back.
        for n in ("ggml-small-q8_0.bin.tmp", "ggml-small-encoder-openvino.bin.tmp"):
            (tmp_data_dir / n).write_bytes(b"partial")
        ModelManager(models_dir=tmp_data_dir).delete_model("small")
        assert list(tmp_data_dir.iterdir()) == []

    def test_delete_model_noop_if_absent(self, tmp_data_dir):
        ModelManager(models_dir=tmp_data_dir).delete_model("tiny")  # no raise

    def test_models_dir_property(self, tmp_data_dir):
        assert ModelManager(models_dir=tmp_data_dir).models_dir == tmp_data_dir

    def test_a_missing_models_dir_is_created(self, tmp_path):
        # On a first launch, so a download has somewhere to land.
        models_dir = tmp_path / "share" / "quoth" / "models"
        ModelManager(models_dir=models_dir)
        assert models_dir.is_dir()


class TestCacheManagement:
    """The OpenVINO compiled-blob cache, surfaced and cleared from Settings."""

    def test_cache_dir_is_the_xdg_cache(self, tmp_data_dir):
        # Single-sourced: what whisper.open_context hands OpenVINO, outside the models.
        cache = ModelManager(models_dir=tmp_data_dir).cache_dir
        assert cache == paths.openvino_cache_dir()

    def test_size_zero_when_absent(self, tmp_data_dir):
        assert ModelManager(models_dir=tmp_data_dir).cache_size_bytes() == 0

    def test_size_sums_files_recursively(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        (mgr.cache_dir / "sub").mkdir(parents=True, exist_ok=True)
        (mgr.cache_dir / "blob").write_bytes(b"x" * 100)
        (mgr.cache_dir / "sub" / "blob").write_bytes(b"y" * 50)
        assert mgr.cache_size_bytes() == 150

    def test_clear_removes_the_dir(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True, exist_ok=True)
        (mgr.cache_dir / "blob").write_bytes(b"z" * 10)
        mgr.clear_cache()
        assert not mgr.cache_dir.exists()
        assert mgr.cache_size_bytes() == 0

    def test_clear_noop_when_absent(self, tmp_data_dir):
        ModelManager(models_dir=tmp_data_dir).clear_cache()  # no raise

    def test_size_ignores_a_file_that_disappears(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)
        (mgr.cache_dir / "blob").write_bytes(b"x" * 10)
        with patch(
            "quoth.models.os.walk",
            return_value=[(mgr.cache_dir, [], ["disappeared.blob", "blob"])],
        ):
            assert mgr.cache_size_bytes() == 10

    def test_size_ignores_a_directory_that_disappears(self, tmp_data_dir, monkeypatch):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)
        (mgr.cache_dir / "blob").write_bytes(b"x" * 10)
        real_walk = os.walk

        def walk(path, *, onerror):
            onerror(FileNotFoundError("vanished"))
            return real_walk(path, onerror=onerror)

        monkeypatch.setattr("quoth.models.os.walk", walk)
        assert mgr.cache_size_bytes() == 10

    def test_size_propagates_file_errors(self, tmp_data_dir, monkeypatch):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)
        blob = mgr.cache_dir / "blob"
        blob.touch()
        real_stat = Path.stat

        def stat(path, *args, **kwargs):
            if path == blob:
                raise PermissionError("blocked")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)
        with pytest.raises(PermissionError, match="blocked"):
            mgr.cache_size_bytes()

    def test_size_propagates_traversal_errors(self, tmp_data_dir, monkeypatch):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)

        def walk(_path, *, onerror):
            onerror(PermissionError("blocked"))
            return ()

        monkeypatch.setattr("quoth.models.os.walk", walk)
        with pytest.raises(PermissionError, match="blocked"):
            mgr.cache_size_bytes()

    def test_file_at_cache_path_is_not_reported_as_empty(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.parent.mkdir(parents=True)
        mgr.cache_dir.write_bytes(b"not a directory")

        with pytest.raises(NotADirectoryError):
            mgr.cache_size_bytes()

    def test_size_propagates_cache_stat_errors(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        with (
            patch.object(Path, "stat", side_effect=PermissionError("blocked")),
            pytest.raises(PermissionError, match="blocked"),
        ):
            mgr.cache_size_bytes()

    def test_clear_propagates_removal_errors(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)
        with (
            patch("quoth.models.shutil.rmtree", side_effect=PermissionError("blocked")),
            pytest.raises(PermissionError, match="blocked"),
        ):
            mgr.clear_cache()

    def test_clear_propagates_cache_stat_errors(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        with (
            patch.object(Path, "stat", side_effect=PermissionError("blocked")),
            pytest.raises(PermissionError, match="blocked"),
        ):
            mgr.clear_cache()

    def test_clear_accepts_a_cache_removed_meanwhile(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)

        def removed_meanwhile(path):
            path.rmdir()
            raise FileNotFoundError("gone")

        with patch("quoth.models.shutil.rmtree", side_effect=removed_meanwhile):
            mgr.clear_cache()
        assert not mgr.cache_dir.exists()

    def test_clear_does_not_hide_child_disappearance(self, tmp_data_dir):
        mgr = ModelManager(models_dir=tmp_data_dir)
        mgr.cache_dir.mkdir(parents=True)
        with (
            patch("quoth.models.shutil.rmtree", side_effect=FileNotFoundError("child")),
            pytest.raises(FileNotFoundError, match="child"),
        ):
            mgr.clear_cache()
