"""LABELLED tiny-cache storage vectors; these are not real model-load evidence."""

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from huggingface_hub import constants as hf_constants

from iris.application.scientific_optimization import _same_runtime
from iris.utils.seeds import capture_runtime_state
from tests.integration._offline_composition_fixture import tiny_offline_cache


class LabelledConfig:
    def save_pretrained(self, path: Path) -> None:
        (path / "config.json").write_bytes(b'{"labelled":true}\n')


class LabelledModel:
    config = LabelledConfig()

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(b"LABELLED_NOT_WEIGHTS")


class LabelledTokenizer:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "tokenizer.json").write_bytes(b'{"labelled":true}\n')


def specimen() -> tuple[Any, Any, Any]:
    random.random()
    torch.rand(1)
    return object(), LabelledTokenizer(), LabelledModel()


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "fixture/tiny-hf-cache"
    monkeypatch.setenv("IRIS_ABLATION_COMPOSITION_TEST_CACHE", str(path))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(path))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    return path


def test_explicit_offline_configuration_is_required_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_ABLATION_COMPOSITION_TEST_CACHE", raising=False)
    build = Mock(side_effect=AssertionError("No fixture without explicit cache"))
    with pytest.raises(RuntimeError, match="Configure a fresh"):
        tiny_offline_cache(build)
    build.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
def test_nonoffline_configuration_is_refused(
    configured: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setenv(field, "0")
    build = Mock()
    with pytest.raises(RuntimeError):
        tiny_offline_cache(build)
    build.assert_not_called()
    assert not configured.parent.exists()


@pytest.mark.parametrize("fault", ("relative", "late_cache", "late_offline"))
def test_late_or_relative_cache_settings_do_not_reconfigure_globals(
    configured: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "relative":
        monkeypatch.setenv("IRIS_ABLATION_COMPOSITION_TEST_CACHE", "relative-cache")
    elif fault == "late_cache":
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(configured / "different"))
    else:
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    build = Mock()
    with pytest.raises(RuntimeError):
        tiny_offline_cache(build)
    build.assert_not_called()
    assert not configured.parent.exists()


def test_fixture_is_created_once_reused_exactly_and_caller_RNG_restored(configured: Path) -> None:
    before = capture_runtime_state(torch.device("cpu"))
    build = Mock(side_effect=specimen)
    assert tiny_offline_cache(build) == configured
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    files = {str(p): p.read_bytes() for p in configured.rglob("*") if p.is_file()}
    manifest = configured.parent / "tiny-hf-cache-fixture-manifest.json"
    manifest_bytes = manifest.read_bytes()
    assert tiny_offline_cache(build) == configured and build.call_count == 1
    assert files == {str(p): p.read_bytes() for p in configured.rglob("*") if p.is_file()}
    assert manifest.read_bytes() == manifest_bytes
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("fault", ("changed", "extra", "missing", "digest", "kind", "duplicate"))
def test_existing_fixture_corruption_never_triggers_rebuild(configured: Path, fault: str) -> None:
    tiny_offline_cache(specimen)
    build = Mock(side_effect=AssertionError("No repair or replacement of unknown fixture"))
    file = next(configured.rglob("model.safetensors"))
    manifest_path = configured.parent / "tiny-hf-cache-fixture-manifest.json"
    if fault == "changed":
        file.write_bytes(b"changed")
    elif fault == "extra":
        (configured / "foreign.json").write_bytes(b"foreign")
    elif fault == "missing":
        file.unlink()
    elif fault == "duplicate":
        manifest_path.write_bytes(
            b'{"kind":"random-tiny-fixture-not-pretrained","files":{},"files":{}}'
        )
    else:
        manifest = json.loads(manifest_path.read_bytes())
        if fault == "kind":
            manifest["kind"] = "pretrained"
        else:
            manifest["files"][file.relative_to(configured).as_posix()] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        tiny_offline_cache(build)
    build.assert_not_called()


def test_orphan_manifest_is_preserved_without_building(configured: Path) -> None:
    configured.parent.mkdir()
    manifest = configured.parent / "tiny-hf-cache-fixture-manifest.json"
    manifest.write_bytes(b"preserve orphan")
    build = Mock()
    with pytest.raises(ValueError, match="Orphan"):
        tiny_offline_cache(build)
    build.assert_not_called()
    assert manifest.read_bytes() == b"preserve orphan" and not configured.exists()


def test_primary_builder_signal_restores_RNG_without_replacing_it(configured: Path) -> None:
    before = capture_runtime_state(torch.device("cpu"))
    signal = KeyboardInterrupt()

    def interrupted() -> tuple[Any, Any, Any]:
        random.random()
        torch.rand(1)
        raise signal

    with pytest.raises(KeyboardInterrupt) as caught:
        tiny_offline_cache(interrupted)
    assert caught.value is signal
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
