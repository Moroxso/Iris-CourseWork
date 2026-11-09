"""Real trained tiny-Qwen selection/projection, exact metadata and caller transactions.

These tests do not claim request/logit parity, publication or R01.4 completion.
"""

import json
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal
from unittest.mock import Mock, patch

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.application import scientific_ablation_checkpoint as native
from iris.application import scientific_ablation_inference as inference
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    IrisScientificAblationInferenceError,
    build_scientific_ablation_inference_v1,
    verify_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_lifecycle import (
    IrisScientificAblationLifecycleBoundaryV1,
    IrisScientificAblationLifecycleV1,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    build_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_validation import _snapshot
from iris.application.scientific_inference import _checked as legacy_checked
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_inference_v1 import (
    IrisScientificAblationInferenceManifestV1,
    ScientificAblationInferencePayloadV1,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import IrisScientificAblationLifecyclePolicyV1
from iris.artifacts.scientific_inference_v1 import IrisScientificInferenceManifestV1
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_run import _run
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy

pytestmark = pytest.mark.integration

Routing = Literal["learned", "random", "uniform"]


def _fresh(
    routing: Routing = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    model_seed: int = 1907,
) -> IrisScientificAblationRunV1:
    return _run(
        routing,
        goal=goal,
        broadcast=broadcast,
        cycles=3,
        model_seed=model_seed,
        policy=_policy(minimum_target_tokens=1, max_optimizer_steps=5),
    )


def _source(
    routing: Routing = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    validate: int = 1,
    checkpoint: int = 3,
) -> tuple[IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1]:
    controller = IrisScientificAblationLifecycleV1(
        run=_fresh(routing, goal=goal, broadcast=broadcast),
        validation=_validation(),
        policy=IrisScientificAblationLifecyclePolicyV1(
            purpose="smoke",
            validate_every_steps=validate,
            checkpoint_every_steps=checkpoint,
            minimum_improvement=1e6,
        ),
    )
    for _ in range(30):
        result = controller.advance()
        if type(result) is IrisScientificAblationLifecycleBoundaryV1:
            return build_scientific_ablation_lifecycle_checkpoint_v1(result), controller
    raise AssertionError("No actual coherent boundary")


@pytest.fixture(scope="module")
def learned_source() -> tuple[
    IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
]:
    return _source()


@pytest.fixture(scope="module")
def learned_export(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
) -> IrisScientificAblationInferenceBundleV1:
    bundle, _ = learned_source
    return build_scientific_ablation_inference_v1(
        bundle,
        expected_lifecycle_sha256=bundle.manifest.sha256,
        make_fresh_run=_fresh,
    )


def _verify(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> IrisScientificAblationInferenceManifestV1:
    # Used only for semantic/rehash probes. Actual trust-anchor denial uses
    # separately captured good bindings and never accepts candidate self hashes.
    return verify_scientific_ablation_inference_v1(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.source_checkpoint_sha256,
        expected_lifecycle_sha256=bundle.manifest.payload.selection.lifecycle_sha256,
    )


def _rehashed(
    bundle: IrisScientificAblationInferenceBundleV1, **updates: Any
) -> IrisScientificAblationInferenceBundleV1:
    payload = ScientificAblationInferencePayloadV1.model_validate(
        {**bundle.manifest.payload.model_dump(mode="python"), **updates}
    )
    return replace(
        bundle,
        manifest=IrisScientificAblationInferenceManifestV1(
            sha256=domain_sha256(
                inference._DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        ),
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_earlier_best_c3_all_modes_exact_projection_preserves_current(
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    source, controller = _source(routing, goal=goal, broadcast=broadcast)
    best = source.manifest.payload.best
    assert best is not None and best.payload.completed_steps == 1
    assert source.manifest.payload.current.payload.completed_steps == 3
    assert best.sha256 != source.manifest.payload.current.sha256
    before = build_scientific_ablation_checkpoint_v1(controller._run)
    snapshot = _snapshot(controller._run)
    modes = tuple(module.training for module in controller._run._optimizer._model.modules())
    gradients = tuple(
        None if p.grad is None else p.grad.detach().clone()
        for p in controller._run._optimizer._model.parameters()
    )
    private_runtime = controller._run._clone_runtime(controller._run._runtime)
    ambient = capture_runtime_state(torch.device("cpu"))
    created: list[IrisScientificAblationRunV1] = []

    def factory() -> IrisScientificAblationRunV1:
        # Simulate constructor RNG use BEFORE native restore's own snapshot.
        random.random()
        torch.rand(3, device="cpu")
        run = _fresh(routing, goal=goal, broadcast=broadcast)
        created.append(run)
        return run

    bundle = build_scientific_ablation_inference_v1(
        source,
        expected_lifecycle_sha256=source.manifest.sha256,
        make_fresh_run=factory,
    )
    manifest = _verify(bundle)
    payload = manifest.payload
    assert payload.source_checkpoint_sha256 == best.sha256 and payload.completed_steps == 1
    assert payload.selection.current_checkpoint_sha256 == before.manifest.sha256
    assert (
        payload.selection.state == controller.state
        and payload.ablation == created[0]._optimizer._model.ablation
    )
    parameters = dict(created[0]._optimizer._model.teacher.named_parameters())
    tensors = load_tensors(bundle.state_bytes)
    assert set(tensors) == set(parameters)
    assert all(torch.equal(tensors[n], p.detach().cpu()) for n, p in parameters.items())
    assert source.best_state_bytes is not None
    native_best = load_tensors(source.best_state_bytes)
    assert all(torch.equal(tensors[n], native_best[f"model/base.teacher.{n}"]) for n in parameters)
    native_current = load_tensors(source.current_state_bytes)
    assert any(
        not torch.equal(tensors[n], native_current[f"model/base.teacher.{n}"]) for n in parameters
    )
    assert all(
        not any(part in n for part in ("probes", "adamw", "runtime", "configuration_identity"))
        for n in tensors
    )
    public = canonical_json_bytes(manifest.model_dump(mode="json"))
    assert all(
        field not in public
        for field in (
            b'"python_state"',
            b'"python_gaussian"',
            b'"parameter_steps"',
            b'"history_sha256"',
            b'"runtime/cpu"',
        )
    )
    assert _snapshot(controller._run) == snapshot
    assert build_scientific_ablation_checkpoint_v1(controller._run) == before
    assert _same_runtime(controller._run._runtime, private_runtime)
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    assert modes == tuple(module.training for module in controller._run._optimizer._model.modules())
    assert all(
        (old is None and p.grad is None)
        or (old is not None and p.grad is not None and torch.equal(old, p.grad))
        for old, p in zip(gradients, controller._run._optimizer._model.parameters(), strict=True)
    )
    saved_bytes = bundle.state_bytes
    with torch.no_grad():
        next(created[0]._optimizer._model.teacher.parameters()).add_(1)
    assert bundle.state_bytes == saved_bytes and _verify(bundle) == manifest


@pytest.mark.parametrize("which", ("current", "selected_best"))
def test_current_and_shared_best_coherent_boundaries(
    which: Literal["current", "selected_best"],
) -> None:
    source, _ = _source(checkpoint=1)
    assert (
        source.best_state_bytes is None
        and source.manifest.payload.best == source.manifest.payload.current
    )
    exported = build_scientific_ablation_inference_v1(
        source,
        expected_lifecycle_sha256=source.manifest.sha256,
        make_fresh_run=_fresh,
        selection=which,
    )
    assert _verify(exported).payload.completed_steps == 1
    assert exported.manifest.payload.selection.kind == which


def test_current_without_best_and_selected_best_absent_rejects_before_factory() -> None:
    source, _ = _source(validate=4, checkpoint=1)
    assert source.manifest.payload.best is None
    current = build_scientific_ablation_inference_v1(
        source,
        expected_lifecycle_sha256=source.manifest.sha256,
        make_fresh_run=_fresh,
        selection="current",
    )
    assert _verify(current).payload.selection.best_checkpoint_sha256 is None
    factory = Mock(side_effect=_fresh)
    with pytest.raises(IrisScientificAblationInferenceError):
        build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=factory
        )
    factory.assert_not_called()


@pytest.mark.parametrize("which", ("current", "selected_best"))
def test_terminal_coherent_boundary_remains_redacted_engineering_selection(
    which: Literal["current", "selected_best"],
) -> None:
    _, controller = _source()
    for _ in range(30):
        result = controller.advance()
        if (
            type(result) is IrisScientificAblationLifecycleBoundaryV1
            and result.state.phase == "stopped"
        ):
            break
    else:
        raise AssertionError("No actual terminal coherent boundary")
    source = build_scientific_ablation_lifecycle_checkpoint_v1(result)
    assert source.manifest.payload.state.stop_reason == "step_budget"
    assert source.manifest.payload.current.payload.completed_steps == 5
    exported = build_scientific_ablation_inference_v1(
        source,
        expected_lifecycle_sha256=source.manifest.sha256,
        make_fresh_run=_fresh,
        selection=which,
    )
    payload = _verify(exported).payload
    assert payload.completed_steps == (5 if which == "current" else 1)
    assert payload.selection.state == source.manifest.payload.state


@pytest.mark.parametrize(
    "fault", ("external_digest", "parent_hash", "hidden_runtime", "current_bytes", "best_bytes")
)
def test_invalid_source_rejects_before_native_decode_or_factory(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
    fault: str,
) -> None:
    source, _ = learned_source
    trusted = source.manifest.sha256
    source = replace(source, manifest=source.manifest.model_copy(deep=True))
    if fault == "external_digest":
        trusted = "0" * 64
    elif fault == "parent_hash":
        source = replace(source, manifest=source.manifest.model_copy(update={"sha256": "0" * 64}))
    elif fault == "hidden_runtime":
        vars(source.manifest.payload.current.payload.runtime)["ghost"] = True
    elif fault == "current_bytes":
        source = replace(source, current_state_bytes=source.current_state_bytes + b"x")
    else:
        assert source.best_state_bytes is not None
        source = replace(source, best_state_bytes=source.best_state_bytes + b"x")
    factory = Mock(side_effect=_fresh)
    with (
        patch.object(native, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=trusted, make_fresh_run=factory
        )
    decode.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize(
    "binding", ("expected_sha256", "expected_source_checkpoint_sha256", "expected_lifecycle_sha256")
)
def test_external_inference_binding_denial_before_decode(
    learned_export: IrisScientificAblationInferenceBundleV1, binding: str
) -> None:
    values = {
        "expected_sha256": learned_export.manifest.sha256,
        "expected_source_checkpoint_sha256": (
            learned_export.manifest.payload.source_checkpoint_sha256
        ),
        "expected_lifecycle_sha256": learned_export.manifest.payload.selection.lifecycle_sha256,
    }
    values[binding] = "0" * 64
    with (
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        verify_scientific_ablation_inference_v1(learned_export, **values)
    decode.assert_not_called()


@pytest.mark.parametrize(
    "path",
    (
        "",
        "payload",
        "payload.selection",
        "payload.selection.state",
        "payload.ablation",
        "payload.generation",
        "payload.model",
        "payload.model.runtime",
        "payload.model.backbone",
        "payload.model.backbone.tokenizer",
        "payload.model.cognitive",
        "payload.model.cognitive.layer_sampler",
        "payload.model.workspace",
        "tensor",
    ),
)
def test_hidden_unchecked_nested_extra_rejects_before_dump_decode(
    learned_export: IrisScientificAblationInferenceBundleV1, path: str
) -> None:
    candidate = replace(learned_export, manifest=learned_export.manifest.model_copy(deep=True))
    value: Any = candidate.manifest
    if path == "tensor":
        value = value.payload.tensors[0]
    else:
        for name in path.split(".") if path else ():
            value = getattr(value, name)
    vars(value)["ghost"] = {"private_rng": 3}
    with (
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        _verify(candidate)
    decode.assert_not_called()


def test_unchecked_metadata_budget_rejects_before_revalidation_dump_decode(
    learned_export: IrisScientificAblationInferenceBundleV1,
) -> None:
    candidate = replace(learned_export, manifest=learned_export.manifest.model_copy(deep=True))
    vars(candidate.manifest.payload)["source_run_configuration_json"] = (
        "x" * inference._MAX_MANIFEST
    )
    with (
        patch.object(IrisScientificAblationInferenceManifestV1, "model_validate") as validate,
        patch.object(IrisScientificAblationInferenceManifestV1, "model_dump") as dump,
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError) as caught,
    ):
        _verify(candidate)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "byte budget" in str(caught.value.__cause__)
    validate.assert_not_called()
    dump.assert_not_called()
    decode.assert_not_called()


@pytest.mark.parametrize("fault", ("bool_step", "wrong_optional_digest", "wrong_optional_literal"))
def test_unchecked_union_and_scalar_variants_reject_before_decode(
    learned_export: IrisScientificAblationInferenceBundleV1, fault: str
) -> None:
    candidate = replace(learned_export, manifest=learned_export.manifest.model_copy(deep=True))
    selection = candidate.manifest.payload.selection
    if fault == "bool_step":
        vars(selection.state)["step"] = True
    elif fault == "wrong_optional_digest":
        vars(selection)["best_checkpoint_sha256"] = 5
    else:
        vars(selection.state)["stop_reason"] = "different"
    with (
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        _verify(candidate)
    decode.assert_not_called()


@pytest.mark.parametrize(
    "fault", ("purpose", "backbone", "model_config", "lifecycle_config", "run_config", "generation")
)
def test_rehashed_incoherent_metadata_rejects_before_decode(
    learned_export: IrisScientificAblationInferenceBundleV1, fault: str
) -> None:
    p = learned_export.manifest.payload
    updates: dict[str, Any]
    if fault == "purpose":
        updates = {"purpose": "scientific"}
    elif fault == "backbone":
        updates = {
            "model": p.model.model_copy(
                update={"backbone": p.model.backbone.model_copy(update={"revision": "different"})}
            )
        }
    elif fault == "model_config":
        updates = {"source_model_configuration_json": p.source_model_configuration_json + " "}
    elif fault == "lifecycle_config":
        updates = {
            "selection": p.selection.model_copy(update={"lifecycle_configuration_sha256": "0" * 64})
        }
    elif fault == "run_config":
        updates = {"source_run_configuration_sha256": "0" * 64}
    else:
        updates = {"generation": p.generation.model_copy(update={"terminal_token_id": 2})}
    candidate = _rehashed(learned_export, **updates)
    with (
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        _verify(candidate)
    decode.assert_not_called()


@pytest.mark.parametrize("component", ("backbone", "cognitive", "workspace"))
def test_fully_rebound_noncanonical_nested_configuration_rejects_before_decode(
    learned_export: IrisScientificAblationInferenceBundleV1, component: str
) -> None:
    p = learned_export.manifest.payload
    model = json.loads(p.source_model_configuration_json)
    model["base_configuration"][f"{component}_configuration_json"] += " "
    model["base_configuration_sha256"] = domain_sha256(
        "iris/scientific-teacher-configuration/v1",
        canonical_json_bytes(model["base_configuration"]),
    )
    model_json = canonical_json_bytes(model).decode()
    model_sha = domain_sha256(
        "iris/scientific-ablation-teacher-configuration/v1", model_json.encode()
    )
    run = json.loads(p.source_run_configuration_json)
    run["model_configuration_sha256"] = model_sha
    run_json = canonical_json_bytes(run).decode()
    run_sha = domain_sha256("iris/scientific-ablation-run-configuration/v1", run_json.encode())
    lifecycle = json.loads(p.selection.lifecycle_configuration_json)
    lifecycle["run_sha256"] = run_sha
    lifecycle_json = canonical_json_bytes(lifecycle).decode()
    lifecycle_sha = domain_sha256(
        "iris/scientific-ablation-lifecycle-configuration/v1", lifecycle_json.encode()
    )
    candidate = _rehashed(
        learned_export,
        source_model_configuration_json=model_json,
        source_model_configuration_sha256=model_sha,
        source_run_configuration_json=run_json,
        source_run_configuration_sha256=run_sha,
        selection=p.selection.model_copy(
            update={
                "lifecycle_configuration_json": lifecycle_json,
                "lifecycle_configuration_sha256": lifecycle_sha,
            }
        ),
    )
    with (
        patch.object(inference, "load_tensors") as decode,
        pytest.raises(IrisScientificAblationInferenceError) as caught,
    ):
        _verify(candidate)
    assert "Exact nested" in str(caught.value.__cause__)
    decode.assert_not_called()


@pytest.mark.parametrize(
    "fault", ("dtype", "nan", "inventory", "encoding", "request_hash", "teacher_hash")
)
def test_rehashed_tensor_and_topology_faults_rejected_after_decode(
    learned_export: IrisScientificAblationInferenceBundleV1, fault: str
) -> None:
    state = load_tensors(learned_export.state_bytes)
    name = next(iter(state))
    metadata = inference._META
    updates: dict[str, Any] = {}
    if fault == "dtype":
        state[name] = state[name].to(torch.int32)
    elif fault == "nan":
        state[name].view(-1)[0] = float("nan")
    elif fault == "inventory":
        state[name + "_unexpected"] = state.pop(name)
    elif fault == "encoding":
        metadata = {"format": "foreign"}
    elif fault == "request_hash":
        updates["model"] = learned_export.manifest.payload.model.model_copy(
            update={"sha256": "0" * 64}
        )
    else:
        updates["source_teacher_sidecar_sha256"] = "0" * 64
    raw = save_tensors(dict(sorted(state.items())), metadata=metadata)
    candidate = _rehashed(
        replace(learned_export, state_bytes=raw),
        state_sha256=bytes_sha256(raw),
        state_size_bytes=len(raw),
        **updates,
    )
    with (
        patch.object(inference, "load_tensors", wraps=load_tensors) as decode,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        _verify(candidate)
    decode.assert_called_once_with(raw)


def test_wrong_initial_model_seed_rejected_without_relaxing_configuration(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
) -> None:
    source, controller = learned_source
    before = _snapshot(controller._run)
    ambient = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(IrisScientificAblationInferenceError):
        build_scientific_ablation_inference_v1(
            source,
            expected_lifecycle_sha256=source.manifest.sha256,
            make_fresh_run=lambda: _fresh(model_seed=901),
        )
    assert _snapshot(controller._run) == before and not controller.poisoned
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


def test_factory_cannot_return_attached_current_source(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
) -> None:
    source, controller = learned_source
    before = _snapshot(controller._run)
    with (
        patch.object(inference, "restore_scientific_ablation_checkpoint_v1") as restore,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        build_scientific_ablation_inference_v1(
            source,
            expected_lifecycle_sha256=source.manifest.sha256,
            make_fresh_run=lambda: controller._run,
        )
    restore.assert_not_called()
    assert _snapshot(controller._run) == before and not controller.poisoned


@pytest.mark.parametrize("fault", ("foreign", "busy", "poisoned", "progressed", "buffer"))
def test_factory_result_guards_reject_before_native_restore(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
    fault: str,
) -> None:
    source, controller = learned_source
    before = _snapshot(controller._run)
    ambient = capture_runtime_state(torch.device("cpu"))

    def factory() -> Any:
        if fault == "foreign":
            return object()
        run = _fresh()
        if fault == "busy":
            run._busy = True
        elif fault == "poisoned":
            run._poisoned = True
        elif fault == "progressed":
            run.step()
        else:
            run._optimizer._model.teacher.register_buffer("ghost", torch.ones(1, device="cpu"))
        return run

    with (
        patch.object(inference, "restore_scientific_ablation_checkpoint_v1") as restore,
        pytest.raises(IrisScientificAblationInferenceError),
    ):
        build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=factory
        )
    restore.assert_not_called()
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    assert _snapshot(controller._run) == before and not controller.poisoned


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("factory"),
        KeyboardInterrupt("factory"),
        BaseExceptionGroup("factory", [ValueError("other"), SystemExit(17)]),
    ),
)
def test_constructor_failure_restores_caller_full_runtime_preserves_signals(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
    failure: BaseException,
) -> None:
    source, _ = learned_source
    ambient = capture_runtime_state(torch.device("cpu"))

    def fail() -> IrisScientificAblationRunV1:
        random.gauss(0, 1)
        torch.rand(11, device="cpu")
        torch.use_deterministic_algorithms(not ambient.deterministic_algorithms)
        raise failure

    expected = (
        IrisScientificAblationInferenceError if isinstance(failure, Exception) else type(failure)
    )
    with pytest.raises(expected) as caught:
        build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=fail
        )
    if not isinstance(failure, Exception):
        assert caught.value is failure
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


def test_primary_signal_survives_secondary_caller_restoration_failure(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
) -> None:
    source, _ = learned_source
    signal = KeyboardInterrupt("primary")
    factory: Callable[[], IrisScientificAblationRunV1] = Mock(side_effect=signal)
    with (
        patch.object(inference, "restore_runtime_state", side_effect=RuntimeError("cleanup")),
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=factory
        )
    assert caught.value is signal and isinstance(caught.value.__cause__, RuntimeError)


def test_success_with_failed_caller_restoration_withholds_export_and_poisons_fresh(
    learned_source: tuple[
        IrisScientificAblationLifecycleBundleV1, IrisScientificAblationLifecycleV1
    ],
) -> None:
    source, _ = learned_source
    ambient = capture_runtime_state(torch.device("cpu"))
    created: list[IrisScientificAblationRunV1] = []

    def factory() -> IrisScientificAblationRunV1:
        run = _fresh()
        created.append(run)
        return run

    try:
        with (
            patch.object(inference, "restore_runtime_state", side_effect=RuntimeError("cleanup")),
            pytest.raises(IrisScientificAblationInferenceError, match="withheld"),
        ):
            build_scientific_ablation_inference_v1(
                source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=factory
            )
        assert len(created) == 1 and created[0]._poisoned and created[0]._optimizer._poisoned
    finally:
        restore_runtime_state(ambient)


def test_legacy_schema_and_bundle_boundaries_do_not_accept_new_artifact(
    learned_export: IrisScientificAblationInferenceBundleV1,
) -> None:
    with pytest.raises(ValidationError):
        IrisScientificInferenceManifestV1.model_validate(
            learned_export.manifest.model_dump(mode="python")
        )
    with pytest.raises(ValueError, match="Exact inference bundle"):
        legacy_checked(learned_export)  # type: ignore[arg-type]
    assert (
        json.loads(learned_export.manifest.model_dump_json())["payload"]["kind"]
        == "iris-scientific-ablation-inference-v1"
    )
