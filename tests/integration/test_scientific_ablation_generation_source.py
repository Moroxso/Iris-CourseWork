"""Real persisted terminal-best binding; not full held-out generation acceptance."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application import scientific_ablation_checkpoint as native
from iris.application import scientific_ablation_generation_source as source
from iris.application import scientific_ablation_inference as inference
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
    IrisScientificAblationGenerationSourceError,
    read_scientific_ablation_generation_source_v1,
    verify_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    build_scientific_ablation_inference_v1,
    verify_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_inference_io import write_scientific_ablation_inference_v1
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    read_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_optimization import IrisScientificAblationOptimizerV1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_generation_source_v1 import (
    IrisScientificAblationGenerationSourceManifestV1,
    ScientificAblationGenerationSourcePayloadV1,
)
from iris.artifacts.scientific_inference_v1 import REQUEST_COMPONENTS_V1
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    domain_sha256,
    named_module_state_sha256,
)
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt
from tests.integration.test_scientific_ablation_inference_projection import _rehashed
from tests.integration.test_scientific_ablation_run import _run
from tests.integration.test_scientific_optimization import _policy

pytestmark = pytest.mark.integration
Routing = Literal["learned", "random", "uniform"]


@dataclass(frozen=True)
class Prepared:
    native_path: Path
    inference_path: Path
    lifecycle: IrisScientificAblationLifecycleBundleV1
    inference: IrisScientificAblationInferenceBundleV1


def _prepared(
    root: Path,
    routing: Routing = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    maximum: int = 3,
    validate: int = 1,
    checkpoint: int = 3,
    patience: int = 8,
    selection: Literal["current", "selected_best"] = "selected_best",
) -> Prepared:
    controller = _durable(
        root / "native",
        routing,
        goal=goal,
        broadcast=broadcast,
        maximum=maximum,
        validate=validate,
        checkpoint=checkpoint,
        patience=patience,
    )
    receipt = _receipt(controller)
    lifecycle = read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    projected = build_scientific_ablation_inference_v1(
        lifecycle,
        expected_lifecycle_sha256=receipt.manifest.sha256,
        make_fresh_run=lambda: _run(
            routing,
            goal=goal,
            broadcast=broadcast,
            cycles=3,
            policy=_policy(minimum_target_tokens=1, max_optimizer_steps=maximum),
        ),
        selection=selection,
    )
    projected_path = write_scientific_ablation_inference_v1(
        projected,
        root=root / "inference",
        expected_sha256=projected.manifest.sha256,
        expected_source_checkpoint_sha256=projected.manifest.payload.source_checkpoint_sha256,
        expected_lifecycle_sha256=receipt.manifest.sha256,
    )
    return Prepared(receipt.path, projected_path, lifecycle, projected)


def _read(case: Prepared) -> IrisScientificAblationGenerationSourceBundleV1:
    return read_scientific_ablation_generation_source_v1(
        lifecycle_path=case.native_path,
        inference_path=case.inference_path,
        expected_lifecycle_sha256=case.lifecycle.manifest.sha256,
        expected_inference_sha256=case.inference.manifest.sha256,
        expected_source_checkpoint_sha256=case.inference.manifest.payload.source_checkpoint_sha256,
    )


def _verify(bundle: IrisScientificAblationGenerationSourceBundleV1) -> None:
    # Self-derived hashes are ONLY for semantic/fully-rebound adversarial probes.
    # Independent-anchor denial has separately captured good expected bindings.
    verify_scientific_ablation_generation_source_v1(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_lifecycle_sha256=bundle.lifecycle.manifest.sha256,
        expected_inference_sha256=bundle.inference.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.best_checkpoint_sha256,
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    return _prepared(tmp_path_factory.mktemp("terminal-source"))


@pytest.fixture(scope="module")
def bound(prepared: Prepared) -> IrisScientificAblationGenerationSourceBundleV1:
    return _read(prepared)


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_terminal_earlier_best_all12_c3_persisted_weights_and_redacted_binding(
    tmp_path: Path,
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    case = _prepared(tmp_path, routing, goal=goal, broadcast=broadcast)
    caller = capture_runtime_state(torch.device("cpu"))
    original = (
        case.lifecycle.current_state_bytes,
        case.lifecycle.best_state_bytes,
        case.inference.state_bytes,
    )
    result = _read(case)
    _verify(result)
    p = result.manifest.payload
    assert (p.selected_step, p.terminal_step, p.stop_reason) == (1, 3, "step_budget")
    assert p.current_checkpoint_sha256 != p.best_checkpoint_sha256
    assert p.ablation == case.inference.manifest.payload.ablation
    assert p.inference_sha256 == case.inference.manifest.sha256
    assert p.lifecycle_sha256 == case.lifecycle.manifest.sha256
    assert p.scope == "terminal_source_binding_not_test_or_launch_approval"
    assert case.lifecycle.manifest.payload.best is not None
    native_best = load_tensors(case.lifecycle.best_state_bytes or b"")
    native_current = load_tensors(case.lifecycle.current_state_bytes)
    projected = load_tensors(result.inference.state_bytes)
    assert all(torch.equal(native_best["model/base.teacher." + k], v) for k, v in projected.items())
    assert any(
        not torch.equal(native_current["model/base.teacher." + k], v) for k, v in projected.items()
    )
    encoded = canonical_json_bytes(result.manifest.model_dump(mode="json"))
    assert all(
        word not in encoded for word in (b"python_state", b"runtime/cpu", b"adamw/", b"cursor")
    )
    assert (
        p.initial_scientific_state_sha256
        in case.lifecycle.manifest.payload.best.payload.configuration_json
    )
    assert p.teacher_sidecar_sha256 != p.request_sidecar_sha256
    assert (
        case.lifecycle.current_state_bytes,
        case.lifecycle.best_state_bytes,
        case.inference.state_bytes,
    ) == original
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("kind", ("terminal_shared_best", "early_stopping"))
def test_actual_shared_best_and_off_cadence_early_terminal_sources(
    tmp_path: Path,
    kind: str,
) -> None:
    case = (
        _prepared(tmp_path, maximum=2, validate=5, checkpoint=3)
        if kind == "terminal_shared_best"
        else _prepared(tmp_path, maximum=5, checkpoint=5, patience=1)
    )
    result = _read(case)
    _verify(result)
    p = result.manifest.payload
    assert p.terminal_step == 2
    if kind == "terminal_shared_best":
        assert p.selected_step == 2 and p.current_checkpoint_sha256 == p.best_checkpoint_sha256
        assert result.lifecycle.best_state_bytes is None
    else:
        assert p.selected_step == 1 and p.stop_reason == "early_stopping"


@pytest.mark.parametrize("kind", ("nonterminal", "current_under_best"))
def test_ineligible_real_sources_reject_before_either_state_reader(
    tmp_path: Path,
    kind: str,
) -> None:
    case = (
        _prepared(tmp_path, maximum=5)
        if kind == "nonterminal"
        else _prepared(tmp_path, selection="current")
    )
    with (
        patch.object(source, "read_scientific_ablation_lifecycle_checkpoint_v1") as native_read,
        patch.object(source, "read_scientific_ablation_inference_v1") as inference_read,
        pytest.raises(IrisScientificAblationGenerationSourceError),
    ):
        _read(case)
    native_read.assert_not_called()
    inference_read.assert_not_called()


def test_equal_weight_current_tag_is_not_selected_best_provenance(tmp_path: Path) -> None:
    case = _prepared(tmp_path, maximum=2, validate=5, checkpoint=3, selection="current")
    assert case.lifecycle.manifest.payload.best == case.lifecycle.manifest.payload.current
    assert case.inference.manifest.payload.completed_steps == 2
    with (
        patch.object(source, "read_scientific_ablation_lifecycle_checkpoint_v1") as native_read,
        patch.object(source, "read_scientific_ablation_inference_v1") as inference_read,
        pytest.raises(IrisScientificAblationGenerationSourceError),
    ):
        _read(case)
    native_read.assert_not_called()
    inference_read.assert_not_called()


def test_verified_source_metadata_is_detached_from_private_bundle(
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    metadata = verify_scientific_ablation_generation_source_v1(
        bound,
        expected_sha256=bound.manifest.sha256,
        expected_lifecycle_sha256=bound.lifecycle.manifest.sha256,
        expected_inference_sha256=bound.inference.manifest.sha256,
        expected_source_checkpoint_sha256=bound.manifest.payload.best_checkpoint_sha256,
    )
    assert metadata == bound.manifest and metadata is not bound.manifest
    assert metadata.payload is not bound.manifest.payload
    vars(metadata.payload)["training_corpus_sha256"] = "0" * 64
    vars(metadata.payload.ablation)["hidden"] = True
    _verify(bound)


@pytest.mark.parametrize("anchor", ("lifecycle", "inference", "checkpoint"))
def test_wrong_trusted_read_anchor_rejects_before_state_read(
    prepared: Prepared, anchor: str
) -> None:
    expected = {
        "lifecycle": prepared.lifecycle.manifest.sha256,
        "inference": prepared.inference.manifest.sha256,
        "checkpoint": prepared.inference.manifest.payload.source_checkpoint_sha256,
    }
    expected[anchor] = "0" * 64
    with (
        patch.object(source, "read_scientific_ablation_lifecycle_checkpoint_v1") as native_read,
        patch.object(source, "read_scientific_ablation_inference_v1") as inference_read,
        pytest.raises(IrisScientificAblationGenerationSourceError),
    ):
        read_scientific_ablation_generation_source_v1(
            lifecycle_path=prepared.native_path,
            inference_path=prepared.inference_path,
            expected_lifecycle_sha256=expected["lifecycle"],
            expected_inference_sha256=expected["inference"],
            expected_source_checkpoint_sha256=expected["checkpoint"],
        )
    native_read.assert_not_called()
    inference_read.assert_not_called()


@pytest.mark.parametrize("anchor", ("descriptor", "lifecycle", "inference", "checkpoint"))
def test_wrong_trusted_verify_anchor_rejects_before_decode(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    anchor: str,
) -> None:
    values = {
        "descriptor": bound.manifest.sha256,
        "lifecycle": bound.lifecycle.manifest.sha256,
        "inference": bound.inference.manifest.sha256,
        "checkpoint": bound.manifest.payload.best_checkpoint_sha256,
    }
    values[anchor] = "0" * 64
    with (
        patch.object(native, "load_tensors") as native_decode,
        patch.object(inference, "load_tensors") as inference_decode,
        pytest.raises(IrisScientificAblationGenerationSourceError),
    ):
        verify_scientific_ablation_generation_source_v1(
            bound,
            expected_sha256=values["descriptor"],
            expected_lifecycle_sha256=values["lifecycle"],
            expected_inference_sha256=values["inference"],
            expected_source_checkpoint_sha256=values["checkpoint"],
        )
    native_decode.assert_not_called()
    inference_decode.assert_not_called()


@pytest.mark.parametrize(
    "field",
    (
        "training_corpus_sha256",
        "validation_corpus_sha256",
        "initial_scientific_state_sha256",
        "backbone_runtime_sha256",
        "teacher_sidecar_sha256",
        "request_sidecar_sha256",
        "run_configuration_sha256",
        "model_configuration_sha256",
    ),
)
def test_fully_rehashed_false_descriptor_rejects_before_decode(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    field: str,
) -> None:
    p = ScientificAblationGenerationSourcePayloadV1.model_validate(
        {**bound.manifest.payload.model_dump(mode="python"), field: "0" * 64}
    )
    manifest = IrisScientificAblationGenerationSourceManifestV1(
        sha256=domain_sha256(source._DOMAIN, canonical_json_bytes(p.model_dump(mode="json"))),
        payload=p,
    )
    with (
        patch.object(native, "load_tensors") as native_decode,
        patch.object(inference, "load_tensors") as inference_decode,
        pytest.raises(IrisScientificAblationGenerationSourceError),
    ):
        _verify(replace(bound, manifest=manifest))
    native_decode.assert_not_called()
    inference_decode.assert_not_called()


@pytest.mark.parametrize("fault", ("extra", "policy_extra", "bool_step", "nan_nll", "legacy_tag"))
def test_unchecked_nested_descriptors_reject_before_decode(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    candidate = replace(bound, manifest=bound.manifest.model_copy(deep=True))
    p = candidate.manifest.payload
    if fault == "extra":
        vars(p)["hidden"] = "forbidden"
    elif fault == "policy_extra":
        vars(p.ablation)["hidden"] = "forbidden"
    else:
        values: dict[str, Any] = {
            "bool_step": True,
            "nan_nll": float("nan"),
            "legacy_tag": "iris-scientific-generation-v1",
        }
        fields = {
            "bool_step": "selected_step",
            "nan_nll": "best_validation_nll_per_token",
            "legacy_tag": "kind",
        }
        vars(p)[fields[fault]] = values[fault]
    with (
        patch.object(native, "load_tensors") as native_decode,
        patch.object(inference, "load_tensors") as inference_decode,
        pytest.raises(IrisScientificAblationGenerationSourceError),
    ):
        _verify(candidate)
    native_decode.assert_not_called()
    inference_decode.assert_not_called()


def test_fully_rebound_valid_inference_weights_cannot_lie_about_actual_native_best(
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    tensors = load_tensors(bound.inference.state_bytes)
    tensors[next(iter(tensors))].add_(1.0)
    raw = save_tensors(dict(sorted(tensors.items())), metadata=inference._META)
    exported = _rehashed(
        bound.inference,
        state_sha256=bytes_sha256(raw),
        state_size_bytes=len(raw),
        model=bound.inference.manifest.payload.model.model_copy(
            update={
                "sha256": named_module_state_sha256(
                    component_names=REQUEST_COMPONENTS_V1, state=tensors
                )
            }
        ),
        source_teacher_sidecar_sha256=named_module_state_sha256(
            component_names=inference._TEACHER_COMPONENTS, state=tensors
        ),
    )
    exported = replace(exported, state_bytes=raw)
    # The standalone export is internally valid; source anchoring still MUST compare
    # actual native tensors, not accept this newly rebound sidecar/hash metadata.
    verify_scientific_ablation_inference_v1(
        exported,
        expected_sha256=exported.manifest.sha256,
        expected_source_checkpoint_sha256=bound.manifest.payload.best_checkpoint_sha256,
        expected_lifecycle_sha256=bound.lifecycle.manifest.sha256,
    )
    metadata = source._metadata(
        bound.lifecycle.manifest,
        exported.manifest,
        expected_lifecycle_sha256=bound.lifecycle.manifest.sha256,
        expected_inference_sha256=exported.manifest.sha256,
        expected_source_checkpoint_sha256=bound.manifest.payload.best_checkpoint_sha256,
    )
    candidate = replace(bound, manifest=source._descriptor(*metadata), inference=exported)
    with pytest.raises(IrisScientificAblationGenerationSourceError) as caught:
        _verify(candidate)
    assert "Actual inference weights" in str(caught.value.__cause__)


@pytest.mark.parametrize("member", ("current", "best", "inference"))
def test_changed_private_state_bytes_are_never_a_valid_source(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    member: str,
) -> None:
    if member == "inference":
        candidate = replace(bound, inference=replace(bound.inference, state_bytes=b"bad"))
    elif member == "current":
        candidate = replace(bound, lifecycle=replace(bound.lifecycle, current_state_bytes=b"bad"))
    else:
        candidate = replace(bound, lifecycle=replace(bound.lifecycle, best_state_bytes=b"bad"))
    with pytest.raises(IrisScientificAblationGenerationSourceError):
        _verify(candidate)


def test_reader_constructs_no_training_owner_or_optimizer(prepared: Prepared) -> None:
    with (
        patch.object(IrisScientificAblationRunV1, "__init__") as run,
        patch.object(IrisScientificAblationTeacherV1, "__init__") as teacher,
        patch.object(IrisScientificAblationOptimizerV1, "__init__") as optimizer,
        patch.object(torch.optim, "AdamW") as adamw,
    ):
        result = _read(prepared)
        _verify(result)
    for constructor in (run, teacher, optimizer, adamw):
        constructor.assert_not_called()


@pytest.mark.parametrize(
    "primary",
    (
        KeyboardInterrupt("read interrupt"),
        SystemExit(41),
        BaseExceptionGroup("source signal", [ValueError("other"), KeyboardInterrupt("signal")]),
    ),
)
def test_source_read_preserves_exact_primary_signal_and_has_no_outputs(
    prepared: Prepared,
    primary: BaseException,
) -> None:
    before = tuple(
        sorted(
            (p.relative_to(prepared.native_path), bytes_sha256(p.read_bytes()))
            for p in prepared.native_path.iterdir()
        )
    )
    with (
        patch.object(
            source, "read_scientific_ablation_lifecycle_checkpoint_v1", side_effect=primary
        ),
        patch.object(source, "read_scientific_ablation_inference_v1") as inference_read,
        pytest.raises(type(primary)) as caught,
    ):
        _read(prepared)
    assert caught.value is primary
    inference_read.assert_not_called()
    assert (
        tuple(
            sorted(
                (p.relative_to(prepared.native_path), bytes_sha256(p.read_bytes()))
                for p in prepared.native_path.iterdir()
            )
        )
        == before
    )
