"""Actual source-layout delivery, precise resume and canonical-file rejection."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_tensors
from tokenizers import Tokenizer  # type: ignore[import-untyped]

import iris.application.scientific_delivery as delivery
from iris.application.scientific_checkpoint import build_scientific_checkpoint_v1
from iris.application.scientific_delivery import (
    IrisScientificDeliveryBundleV1,
    IrisScientificDeliveryError,
    build_scientific_delivery_v1,
    read_scientific_delivery_v1,
    restore_scientific_delivery_v1,
    write_scientific_delivery_v1,
)
from iris.application.scientific_inference import build_scientific_inference_v1
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_validation import evaluate_scientific_validation_v1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_checkpoint import _run, _stochastic
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


def _changed(
    bundle: IrisScientificDeliveryBundleV1, name: str, data: bytes
) -> IrisScientificDeliveryBundleV1:
    files = dict(bundle.files)
    files[name] = data
    return IrisScientificDeliveryBundleV1(delivery._identity(files), tuple(sorted(files.items())))


@pytest.mark.parametrize("cut", [0, 1, 3])
def test_source_nine_file_layout_resumes_identically_and_exports_for_inference(
    tmp_path: Path, cut: int
) -> None:
    source = _run(minimum=70)
    with _stochastic(source):
        for _ in range(cut):
            source.step()
    source_checkpoint = build_scientific_checkpoint_v1(source)
    runtime = capture_runtime_state(torch.device("cpu"))
    bundle = build_scientific_delivery_v1(source)
    assert bundle == build_scientific_delivery_v1(source)
    assert build_scientific_checkpoint_v1(source) == source_checkpoint
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), runtime)
    path = write_scientific_delivery_v1(bundle, root=tmp_path)
    assert write_scientific_delivery_v1(bundle, root=tmp_path, maximum_storage_bytes=1) == path
    assert {file.name for file in path.iterdir()} == set(delivery._FILE_LIMITS)
    restored = _run(minimum=70)
    with patch("torch.load", side_effect=AssertionError("untrusted pickle must not be executed")):
        read = read_scientific_delivery_v1(path)
        assert read == bundle
        restore_scientific_delivery_v1(read, run=restored)
    assert build_scientific_checkpoint_v1(restored) == source_checkpoint
    assert build_scientific_inference_v1(restored) == build_scientific_inference_v1(source)
    with _stochastic(source):
        expected = [source.step() for _ in range(2)]
    with _stochastic(restored):
        assert [restored.step() for _ in range(2)] == expected
    assert build_scientific_checkpoint_v1(restored) == build_scientific_checkpoint_v1(source)
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), runtime)


def test_exact_weight_separation_complete_policy_and_readable_tokenizer() -> None:
    run = _run()
    run.step()
    metric = evaluate_scientific_validation_v1(run, _validation())
    files = dict(build_scientific_delivery_v1(run, validation=metric).files)
    params = load_tensors(files["iris_model.safetensors"])
    assert set(params) == set(dict(run._optimizer._model.named_parameters()))
    assert any(name.startswith("probes.") for name in params)
    assert all(not name.startswith("backbone") for name in params)
    # This external interoperability check loads only bytes just built in this test.
    training = torch.load(
        BytesIO(files["training_state.pt"]), weights_only=True, map_location="cpu"
    )
    assert set(training) == {"format", "control_json", "tensors"}
    assert not {f"model/{name}" for name in params}.intersection(training["tensors"])
    assert any(name.startswith("adamw/") for name in training["tensors"])
    assert "runtime/cpu" in training["tensors"]
    control = json.loads(training["control_json"])
    assert control["global_step"] == 1 and control["grad_scaler"] is None
    assert control["scheduler"]["current_learning_rate"] == run._optimizer._current_lr
    config = json.loads(files["config.yaml"])
    assert json.loads(config["probe_configuration_json"])["init_seed"] == 42
    assert config["probe_vocabulary"] == run._optimizer._model.probes.vocabulary.model_dump(
        mode="json"
    )
    assert (
        json.loads(files["metrics.json"])["validation"]["target_nll_per_token"]
        == metric.target_nll_per_token
    )
    tokenizer = Tokenizer.from_str(files["tokenizer.json"].decode())
    row = _rows()[0]
    assert tokenizer.encode(row.prompt).ids == run._backbone._tokenizer(row.prompt)["input_ids"]
    backbone = json.loads(files["backbone.json"])
    assert set(backbone) == {"repository", "revision", "tokenizer_revision", "frozen"}
    assert backbone["frozen"] is True


def test_missing_metric_is_null_and_wrong_step_metric_rejected() -> None:
    run = _run()
    metric = evaluate_scientific_validation_v1(run, _validation())
    assert (
        json.loads(dict(build_scientific_delivery_v1(run).files)["metrics.json"])["validation"]
        is None
    )
    run.step()
    before = build_scientific_checkpoint_v1(run)
    with pytest.raises(IrisScientificDeliveryError):
        build_scientific_delivery_v1(run, validation=metric)
    assert build_scientific_checkpoint_v1(run) == before


@pytest.mark.parametrize("name", list(delivery._FILE_LIMITS))
def test_rehashed_corrupted_member_is_rejected_before_restore_or_publication(
    tmp_path: Path, name: str
) -> None:
    run = _run()
    bundle = build_scientific_delivery_v1(run)
    before = build_scientific_checkpoint_v1(run)
    content = dict(bundle.files)[name]
    bad = _changed(bundle, name, content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(IrisScientificDeliveryError):
        restore_scientific_delivery_v1(bad, run=run)
    with pytest.raises(IrisScientificDeliveryError):
        write_scientific_delivery_v1(bad, root=tmp_path / "must-not-create")
    assert not (tmp_path / "must-not-create").exists()
    assert not run.poisoned and build_scientific_checkpoint_v1(run) == before


@pytest.mark.parametrize(
    "kind", ["probe_labels", "teacher_policy", "backbone", "environment", "metric_step"]
)
def test_rehashed_but_contradictory_metadata_is_rejected(kind: str) -> None:
    run = _run()
    run.step()
    metric = evaluate_scientific_validation_v1(run, _validation())
    bundle = build_scientific_delivery_v1(run, validation=metric)
    files = dict(bundle.files)
    name = "config.yaml"
    if kind == "backbone":
        name = "backbone.json"
    elif kind == "environment":
        name = "environment.json"
    elif kind == "metric_step":
        name = "metrics.json"
    values: dict[str, Any] = json.loads(files[name])
    if kind == "probe_labels":
        values["probe_vocabulary"]["operation"].reverse()
    elif kind == "teacher_policy":
        teacher = json.loads(values["teacher_configuration_json"])
        teacher["policy"]["diversity_margin"] = 0.6
        values["teacher_configuration_json"] = canonical_json_file_bytes(teacher)[:-1].decode()
    elif kind == "backbone":
        values["frozen"] = False
    elif kind == "environment":
        values["torch"] = "different"
    else:
        values["validation"]["at_optimizer_step"] = 0
    bad = _changed(bundle, name, canonical_json_file_bytes(values))
    with pytest.raises(IrisScientificDeliveryError):
        restore_scientific_delivery_v1(bad, run=_run())


def test_different_corpus_rejects_before_any_restore_mutation() -> None:
    source = _run()
    source.step()
    bundle = build_scientific_delivery_v1(source)
    target = _run(reverse=True)
    before = build_scientific_checkpoint_v1(target)
    with pytest.raises(IrisScientificDeliveryError):
        restore_scientific_delivery_v1(bundle, run=target)
    assert not target.poisoned and build_scientific_checkpoint_v1(target) == before


def test_budget_and_failed_or_uncertain_publication_are_retryable(tmp_path: Path) -> None:
    bundle = build_scientific_delivery_v1(_run())
    with pytest.raises(IrisScientificDeliveryError):
        write_scientific_delivery_v1(bundle, root=tmp_path, maximum_storage_bytes=1)
    with (
        patch.object(
            delivery,
            "rename_directory_with_windows_retry",
            side_effect=OSError("failed before publish"),
        ),
        pytest.raises(IrisScientificDeliveryError),
    ):
        write_scientific_delivery_v1(bundle, root=tmp_path)
    assert list((tmp_path / "scientific-delivery-v1").iterdir()) == []

    def uncertain(source: Path, destination: Path) -> None:
        rename_directory_with_windows_retry(source, destination)
        raise OSError("lost acknowledgement")

    with patch.object(delivery, "rename_directory_with_windows_retry", side_effect=uncertain):
        path = write_scientific_delivery_v1(bundle, root=tmp_path)
    assert read_scientific_delivery_v1(path) == bundle
    assert list(path.parent.iterdir()) == [path]
    (path / "README.md").write_bytes(b"conflicting content")
    with pytest.raises(IrisScientificDeliveryError):
        write_scientific_delivery_v1(bundle, root=tmp_path)
    assert (path / "README.md").read_bytes() == b"conflicting content"


def test_failure_after_restore_copy_poisoning_is_preserved() -> None:
    source = _run()
    source.step()
    bundle = build_scientific_delivery_v1(source)
    target = _run()
    original = target._validate_boundary
    calls = 0

    def fail_after_copy() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("post-copy failure")
        original()

    with (
        patch.object(target, "_validate_boundary", side_effect=fail_after_copy),
        pytest.raises(IrisScientificDeliveryError),
    ):
        restore_scientific_delivery_v1(bundle, run=target)
    assert target.poisoned and target._optimizer.poisoned and calls == 3


def test_public_delivery_api_resolves() -> None:
    import iris.application as application

    for name in delivery.__all__:
        assert getattr(application, name) is getattr(delivery, name)
