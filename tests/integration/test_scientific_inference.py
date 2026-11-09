"""Real tiny-Qwen scientific export, independent request parity and rejection checks."""

import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

import iris.application.request_sidecar_state as sidecar_state
import iris.application.scientific_inference as inference
from iris.application.composition import build_request_orchestrator
from iris.application.request import IrisRequestOrchestrator
from iris.application.scientific_checkpoint import (
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_inference import (
    IrisScientificInferenceBundleV1,
    IrisScientificInferenceError,
    build_scientific_inference_v1,
    read_scientific_inference_v1,
    transfer_scientific_inference_to_request_v1,
    write_scientific_inference_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_inference_v1 import (
    IrisScientificInferenceManifestV1,
    ScientificInferencePayloadV1,
)
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    domain_sha256,
)
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_checkpoint import _run
from tests.integration.test_scientific_lifecycle import _lifecycle
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


def _request(run: IrisScientificRunV1, *, changed_config: bool = False) -> IrisRequestOrchestrator:
    provenance = run._optimizer._model.teacher.checkpoint_provenance
    cognitive = CognitiveConfig.model_validate_json(provenance.cognitive_configuration_json)
    if changed_config:
        cognitive = cognitive.model_copy(update={"name": "different-same-shape-config"})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        return build_request_orchestrator(
            backbone=run._backbone,
            cognitive_config=cognitive,
            workspace_config=WorkspaceConfig.model_validate_json(
                provenance.workspace_configuration_json
            ),
            # One real greedy token, avoiding an irrelevant 256-token random-model loop.
            stop_token_ids=tuple(range(run._backbone.vocabulary_size)),
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )


def _request_state(request: IrisRequestOrchestrator) -> dict[str, torch.Tensor]:
    return {
        f"{component}.{name}": value.detach().clone()
        for component, module in request._sidecar_modules
        for name, value in module.state_dict().items()
    }


def _rehashed(
    bundle: IrisScientificInferenceBundleV1, **updates: Any
) -> IrisScientificInferenceBundleV1:
    values = bundle.manifest.payload.model_dump(mode="python")
    values.update(updates)
    payload = ScientificInferencePayloadV1.model_validate(values)
    return replace(
        bundle,
        manifest=IrisScientificInferenceManifestV1(
            sha256=domain_sha256(
                inference._DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        ),
    )


@pytest.mark.parametrize("steps", [0, 2])
def test_export_is_exact_parameter_only_projection_and_does_not_advance_run(steps: int) -> None:
    run = _run()
    for _ in range(steps):
        run.step()
    before = build_scientific_checkpoint_v1(run)
    ambient = capture_runtime_state(torch.device("cpu"))
    bundle = build_scientific_inference_v1(run)
    assert bundle == build_scientific_inference_v1(run)
    payload = bundle.manifest.payload
    assert payload.source_checkpoint_sha256 == before.manifest.sha256
    assert payload.completed_steps == steps and payload.purpose == "smoke"
    state = load_tensors(bundle.state_bytes)
    teacher = run._optimizer._model.teacher
    assert set(state) == set(dict(teacher.named_parameters()))
    assert all(torch.equal(state[n], p.detach()) for n, p in teacher.named_parameters())
    assert all(
        not any(s in n for s in ("probe", "adamw", "runtime", "configuration_identity"))
        for n in state
    )
    assert len(bundle.state_bytes) < len(before.state_bytes)
    assert build_scientific_checkpoint_v1(run) == before
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), ambient)
    run.step()
    assert bundle.manifest.payload.completed_steps == steps
    assert build_scientific_inference_v1(run) != bundle


def test_disk_roundtrip_transfers_exact_weights_and_real_qwen_teacher_behavior(
    tmp_path: Path,
) -> None:
    run = _run()
    run.step()
    bundle = build_scientific_inference_v1(run)
    location = write_scientific_inference_v1(bundle, root=tmp_path)
    assert write_scientific_inference_v1(bundle, root=tmp_path, maximum_storage_bytes=1) == location
    assert {p.name for p in location.iterdir()} == {"manifest.json", "iris_model.safetensors"}
    loaded = read_scientific_inference_v1(location)
    assert loaded == bundle
    independent = _run()
    request = _request(independent)
    assert request.backbone is not run._backbone
    assert request.backbone._model is not run._backbone._model
    previous = request.sidecar_state_sha256
    caller = capture_runtime_state(torch.device("cpu"))
    result = transfer_scientific_inference_to_request_v1(loaded, request=request)
    assert result.source_checkpoint_sha256 == bundle.manifest.payload.source_checkpoint_sha256
    assert result.installation.previous_state_sha256 == previous
    assert result.installation.installed_state_sha256 == request.sidecar_state_sha256
    assert previous != request.sidecar_state_sha256
    assert all(
        torch.equal(value, load_tensors(bundle.state_bytes)[name])
        for name, value in _request_state(request).items()
    )
    expected_logits: list[torch.Tensor] = []
    forward = run._backbone.forward_embeddings

    def capture(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        output = forward(*args, **kwargs)
        labels = kwargs["labels"]
        first_target = int(torch.nonzero(labels[0] != -100)[0, 0])
        expected_logits.append(output.logits[:, first_target - 1, :].detach().clone())
        return output

    model = run._optimizer._model
    model.eval()
    try:
        with patch.object(run._backbone, "forward_embeddings", side_effect=capture):
            teacher = model.evaluation_forward(_rows()[0], terminal_token_id=1)
    finally:
        model.train(True)
    prefill_logits: list[torch.Tensor] = []
    request_forward = request.backbone.forward_embeddings

    def capture_request(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        output = request_forward(*args, **kwargs)
        prefill_logits.append(output.logits[:, -1, :].detach().clone())
        return output

    with patch.object(request.backbone, "forward_embeddings", side_effect=capture_request):
        generated = request.generate(_rows()[0].prompt, strategy="greedy")
    assert len(prefill_logits) == len(expected_logits) == 1
    # Different labelled/full-sequence vs cached-prefill GEMM shapes need numerical tolerance.
    torch.testing.assert_close(prefill_logits[0], expected_logits[0], rtol=1e-5, atol=1e-6)
    assert generated.token_ids.tolist() == [expected_logits[0].argmax(dim=-1).tolist()]
    assert len(generated.workspace_trace) == len(teacher.trace.teacher.workspace_trace) == 3
    for actual, expected in zip(
        generated.workspace_trace, teacher.trace.teacher.workspace_trace, strict=True
    ):
        for field in fields(actual):
            assert torch.equal(getattr(actual, field.name), getattr(expected, field.name))
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), caller)
    assert request.sidecar_state_sha256 == loaded.manifest.payload.model.sha256
    assert all(p.grad is None and not p.requires_grad for p in request.backbone._model.parameters())


def test_best_checkpoint_exports_selected_step_not_latest_training_state(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    while lifecycle.state.phase != "stopped":
        lifecycle.advance(checkpoint_root=tmp_path / "training")
    best = lifecycle.best_checkpoint
    assert best is not None and lifecycle.state.step == 3 and lifecycle.state.best_step == 1
    restored = _run()
    restore_scientific_checkpoint_v1(best, run=restored)
    selected = build_scientific_inference_v1(restored)
    latest = build_scientific_inference_v1(lifecycle._run)
    assert selected.manifest.payload.completed_steps == 1
    assert latest.manifest.payload.completed_steps == 3
    assert selected.manifest.payload.source_checkpoint_sha256 == best.manifest.sha256
    assert selected.state_bytes != latest.state_bytes
    assert (
        read_scientific_inference_v1(write_scientific_inference_v1(selected, root=tmp_path))
        == selected
    )


@pytest.mark.parametrize(
    "kind", ["weights", "tokenizer", "attention", "config", "runtime", "topology"]
)
def test_incompatible_or_rehashed_projection_rejects_before_request_copy(kind: str) -> None:
    run = _run()
    bundle = build_scientific_inference_v1(run)
    target = _run()
    if kind == "weights":
        with torch.no_grad():
            next(target._backbone._model.parameters()).add_(0.01)
    elif kind == "tokenizer":
        target._backbone._tokenizer.split_special_tokens = True
    elif kind == "attention":
        target._backbone._model.config._attn_implementation = "eager"
    request = _request(target, changed_config=kind == "config")
    if kind in ("runtime", "topology"):
        values = bundle.manifest.payload.model.model_dump(mode="python")
        if kind == "runtime":
            values["runtime"]["hidden_size"] += 1
        else:
            values["state_names"] = tuple(reversed(values["state_names"]))
        bundle = _rehashed(bundle, model=values)
    before = _request_state(request)
    digest = request.sidecar_state_sha256
    with pytest.raises(IrisScientificInferenceError):
        transfer_scientific_inference_to_request_v1(bundle, request=request)
    assert request.sidecar_state_sha256 == digest
    assert all(torch.equal(value, _request_state(request)[name]) for name, value in before.items())


@pytest.mark.parametrize("kind", ["hash", "extra_tensor", "nan", "shape"])
def test_corrupt_weight_bundles_reject_without_filesystem_or_target_mutation(
    tmp_path: Path, kind: str
) -> None:
    bundle = build_scientific_inference_v1(_run())
    state = load_tensors(bundle.state_bytes)
    name = next(iter(state))
    if kind == "hash":
        bundle = replace(
            bundle, state_bytes=bundle.state_bytes[:-1] + bytes([bundle.state_bytes[-1] ^ 1])
        )
    else:
        if kind == "extra_tensor":
            state["probes.extra"] = torch.zeros(1)
        elif kind == "nan":
            state[name].flatten()[0] = float("nan")
        else:
            state[name] = state[name].unsqueeze(0)
        data = save_tensors(dict(sorted(state.items())), metadata=inference._META)
        bundle = _rehashed(
            replace(bundle, state_bytes=data),
            state_sha256=bytes_sha256(data),
            state_size_bytes=len(data),
        )
    root = tmp_path / "must-not-create"
    with pytest.raises(IrisScientificInferenceError):
        write_scientific_inference_v1(bundle, root=root)
    assert not root.exists()


def test_publisher_retries_uncertain_rename_and_enforces_storage_budget(tmp_path: Path) -> None:
    run = _run()
    bundle = build_scientific_inference_v1(run)
    with pytest.raises(IrisScientificInferenceError):
        write_scientific_inference_v1(bundle, root=tmp_path, maximum_storage_bytes=1)
    assert list((tmp_path / "scientific-inference-v1").iterdir()) == []
    rename = rename_directory_with_windows_retry

    def uncertain(source: Path, destination: Path) -> None:
        rename(source, destination)
        raise OSError("lost acknowledgement after successful rename")

    with patch.object(inference, "rename_directory_with_windows_retry", side_effect=uncertain):
        location = write_scientific_inference_v1(bundle, root=tmp_path)
    assert read_scientific_inference_v1(location) == bundle
    assert list(location.parent.iterdir()) == [location]
    run.step()
    another = build_scientific_inference_v1(run)
    with pytest.raises(IrisScientificInferenceError):
        write_scientific_inference_v1(another, root=tmp_path, maximum_storage_bytes=1)
    assert read_scientific_inference_v1(location) == bundle


def test_public_exports_resolve_to_canonical_modules() -> None:
    import iris.application as application
    import iris.artifacts as artifacts

    for name in inference.__all__:
        assert getattr(application, name) is getattr(inference, name)
    assert artifacts.IrisScientificInferenceManifestV1 is IrisScientificInferenceManifestV1


def test_mid_transfer_failure_restores_request_and_is_retryable() -> None:
    run = _run()
    run.step()
    bundle = build_scientific_inference_v1(run)
    request = _request(_run())
    before, digest = _request_state(request), request.sidecar_state_sha256
    copy = sidecar_state._copy_tensor
    calls = 0

    def fail_second(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("failed second transfer copy")
        copy(target, source)

    with (
        patch.object(sidecar_state, "_copy_tensor", side_effect=fail_second),
        pytest.raises(IrisScientificInferenceError),
    ):
        transfer_scientific_inference_to_request_v1(bundle, request=request)
    assert calls == 2 + len(before)
    assert request.sidecar_state_sha256 == digest
    assert all(torch.equal(value, _request_state(request)[name]) for name, value in before.items())
    transfer_scientific_inference_to_request_v1(bundle, request=request)
    assert request.sidecar_state_sha256 == bundle.manifest.payload.model.sha256


@pytest.mark.parametrize("kind", ["partial_write", "rename", "reserve"])
def test_failed_publication_leaves_no_checkpoint_and_allows_identical_retry(
    tmp_path: Path, kind: str
) -> None:
    bundle = build_scientific_inference_v1(_run())
    original_write = _write_exact_file

    def partial_write(path: Path, content: bytes) -> None:
        original_write(path, content[:16])
        raise OSError("simulated disk failure")

    if kind == "partial_write":
        operation = patch.object(inference, "_write_exact_file", side_effect=partial_write)
    elif kind == "rename":
        operation = patch.object(
            inference, "rename_directory_with_windows_retry", side_effect=OSError("rename failed")
        )
    else:
        usage = shutil.disk_usage(tmp_path)
        operation = patch.object(shutil, "disk_usage", return_value=usage._replace(free=1))
    with operation, pytest.raises(IrisScientificInferenceError):
        write_scientific_inference_v1(bundle, root=tmp_path)
    assert list((tmp_path / "scientific-inference-v1").iterdir()) == []
    location = write_scientific_inference_v1(bundle, root=tmp_path)
    assert read_scientific_inference_v1(location) == bundle


@pytest.mark.parametrize(
    "kind", ["extra_file", "duplicate_json", "missing_newline", "changed_weights"]
)
def test_reader_and_republisher_reject_conflicting_directory_without_overwrite(
    tmp_path: Path, kind: str
) -> None:
    bundle = build_scientific_inference_v1(_run())
    location = write_scientific_inference_v1(bundle, root=tmp_path)
    manifest = location / "manifest.json"
    if kind == "extra_file":
        (location / "unexpected.txt").write_bytes(b"not an export member")
    elif kind == "duplicate_json":
        manifest.write_bytes(
            b'{"sha256":"' + bundle.manifest.sha256.encode() + b'",' + manifest.read_bytes()[1:]
        )
    elif kind == "missing_newline":
        manifest.write_bytes(manifest.read_bytes()[:-1])
    else:
        (location / "iris_model.safetensors").write_bytes(bundle.state_bytes[:-1] + b"x")
    before = {p.name: p.read_bytes() for p in location.iterdir()}
    with pytest.raises(IrisScientificInferenceError):
        read_scientific_inference_v1(location)
    with pytest.raises(IrisScientificInferenceError):
        write_scientific_inference_v1(bundle, root=tmp_path)
    assert {p.name: p.read_bytes() for p in location.iterdir()} == before


def test_artifact_byte_cap_is_checked_before_decoding_or_creating_directories(
    tmp_path: Path,
) -> None:
    bundle = build_scientific_inference_v1(_run())
    with (
        patch.object(inference, "_MAX_ARTIFACT", 1),
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificInferenceError),
    ):
        write_scientific_inference_v1(bundle, root=tmp_path / "not-created")
    decode.assert_not_called()
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("kind", ["busy", "poisoned", "gradients", "parameters", "backbone"])
def test_unclean_export_source_is_rejected_and_never_cleared(kind: str) -> None:
    run = _run()
    parameter = next(run._optimizer._model.parameters())
    if kind == "busy":
        run._busy = True
    elif kind == "poisoned":
        run._poisoned = True
    elif kind == "gradients":
        parameter.grad = torch.zeros_like(parameter)
    elif kind == "parameters":
        with torch.no_grad():
            parameter.add_(0.01)
    else:
        with torch.no_grad():
            next(run._backbone._model.parameters()).add_(0.01)
    with pytest.raises(IrisScientificInferenceError):
        build_scientific_inference_v1(run)
    assert run._busy is (kind == "busy")
    assert run.poisoned is (kind == "poisoned")
    assert (parameter.grad is not None) is (kind == "gradients")
