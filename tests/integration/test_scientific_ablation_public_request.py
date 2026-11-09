"""Real in-memory owned API/receipt/caller transactions, NOT persisted acceptance."""

import random
from copy import copy
from dataclasses import fields, replace
from functools import partial
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_request as runtime
from iris.application.composition import _compose_request_orchestrator
from iris.application.errors import IrisRequestError
from iris.application.request import IrisRequestExecutionOutput, IrisRequestOrchestrator
from iris.application.request_sidecar_state import install_request_sidecar_state_v1
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    build_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_receipt import (
    IrisScientificAblationRequestOutputV1,
    IrisScientificAblationRequestReceiptV1,
)
from iris.application.scientific_ablation_request import (
    IrisScientificAblationRequestV1,
    build_scientific_ablation_request_v1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_optimization import _same_runtime
from iris.backbone.adapter import BackboneAdapter
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import TemperatureTopPSamplingConfig
from iris.utils.hashing import domain_sha256, tensor_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_scientific_ablation_inference_projection import _fresh, _source
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trusted() -> tuple[IrisScientificAblationInferenceBundleV1, tuple[str, str, str]]:
    source, _ = _source()
    bundle = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=_fresh
    )
    # Trusted fresh producer; retain anchors BEFORE malformed candidate probes.
    anchors = (
        bundle.manifest.sha256,
        bundle.manifest.payload.source_checkpoint_sha256,
        source.manifest.sha256,
    )
    return bundle, anchors


def _load(
    trusted: tuple[IrisScientificAblationInferenceBundleV1, tuple[str, str, str]],
    *,
    backbone: BackboneAdapter | None = None,
) -> IrisScientificAblationRequestV1:
    bundle, anchors = trusted
    if backbone is None:
        backbone = _fresh()._backbone  # External fixture creation, NOT loader behavior.
    return build_scientific_ablation_request_v1(
        bundle,
        expected_sha256=anchors[0],
        expected_source_checkpoint_sha256=anchors[1],
        expected_lifecycle_sha256=anchors[2],
        backbone=backbone,
        stop_token_ids=tuple(range(backbone.vocabulary_size)),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )


@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_loader_and_real_calls_preserve_caller_context_and_distinct_facts(
    trusted: Any, default: str, inference: bool
) -> None:
    backbone = _fresh()._backbone
    caller = capture_runtime_state(torch.device("cpu"))
    with torch.device(default), torch.inference_mode(inference):
        request = _load(trusted, backbone=backbone)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        first = request.execute(_rows()[0].prompt, strategy="greedy")
        second = request.execute(_rows()[0].prompt, strategy="greedy")
        assert torch.get_default_device() == torch.device(default)
        assert torch.is_inference_mode_enabled() is inference
    assert type(first) is IrisScientificAblationRequestOutputV1
    assert not isinstance(first, IrisRequestExecutionOutput)
    assert type(first.receipt) is IrisScientificAblationRequestReceiptV1
    assert first.receipt.selection == "selected_best" and first.receipt.completed_steps == 1
    assert first.receipt.artifact_sha256 == trusted[1][0]
    assert first.receipt.source_checkpoint_sha256 == trusted[1][1]
    assert first.receipt.lifecycle_sha256 == trusted[1][2]
    assert first.receipt.cognitive_prompt_sha256 == domain_sha256(
        "iris/cognitive-ablation-prompt/v1", _rows()[0].prompt.encode()
    )
    assert first.receipt.prompt_sha256 != first.receipt.cognitive_prompt_sha256
    assert torch.equal(first.generation.token_ids, second.generation.token_ids)
    assert first.generation.workspace_trace[0] is not second.generation.workspace_trace[0]
    assert request._owner._state == request._state == "ready"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert not any(hasattr(request, name) for name in ("backbone", "restore", "session", "policy"))


def test_defensive_metadata_and_source_mutation_do_not_change_owned_policy(trusted: Any) -> None:
    request = _load(trusted)
    metadata = request.manifest
    object.__setattr__(metadata.payload.ablation, "goal", "disabled")
    vars(metadata.payload.ablation)["ghost"] = 1
    assert request.manifest.payload.ablation.goal == "enabled"
    assert "ghost" not in vars(request.manifest.payload.ablation)
    request.execute(_rows()[0].prompt, strategy="greedy")


@pytest.mark.parametrize("anchor", (0, 1, 2))
def test_external_trust_mismatch_before_decode_adapter_or_construction(
    trusted: Any, anchor: int
) -> None:
    bundle, good = trusted
    bad = list(good)
    bad[anchor] = "0" * 64
    bad_backbone: Any = object()
    with (
        patch("iris.application.scientific_ablation_inference.load_tensors") as decode,
        patch.object(runtime, "_compatible_backbone") as metadata,
        patch.object(runtime, "_compose_request_orchestrator") as construct,
        pytest.raises(IrisRequestError),
    ):
        build_scientific_ablation_request_v1(
            bundle,
            expected_sha256=bad[0],
            expected_source_checkpoint_sha256=bad[1],
            expected_lifecycle_sha256=bad[2],
            backbone=bad_backbone,
            stop_token_ids=(),
            sampling=TemperatureTopPSamplingConfig(),
        )
    decode.assert_not_called()
    metadata.assert_not_called()
    construct.assert_not_called()


@pytest.mark.parametrize(
    "slot", ("_guard", "_request", "_owner", "_binding", "_metadata", "_state")
)
@pytest.mark.parametrize("missing", (False, True))
def test_missing_or_replaced_wrapper_fields_deny_before_runtime_and_forward(
    trusted: Any, slot: str, missing: bool
) -> None:
    request = _load(trusted)
    owner, backbone = request._owner, request._request._backbone
    if missing:
        delattr(request, slot)
    else:
        setattr(request, slot, object())
    with (
        patch.object(runtime, "capture_runtime_state") as capture,
        patch.object(backbone, "forward_prompt") as forward,
        pytest.raises(IrisRequestError),
    ):
        IrisScientificAblationRequestV1.execute(request, _rows()[0].prompt, strategy="greedy")
    capture.assert_not_called()
    forward.assert_not_called()
    assert owner._state == "poisoned"


def test_equal_copied_guard_is_not_original_guard(trusted: Any) -> None:
    request = _load(trusted)
    request._guard = copy(request._guard)
    with pytest.raises(IrisRequestError):
        request.execute(_rows()[0].prompt, strategy="greedy")
    assert request._owner._state == "poisoned"


def test_callback_reentry_unbound_public_and_baseline_are_denied(trusted: Any) -> None:
    request = _load(trusted)
    backbone = request._request._backbone
    native = backbone.forward_prompt

    def forward(prompt: str) -> Any:
        assert request._state == "busy" and request._owner._state == "executing"
        with patch.object(runtime, "capture_runtime_state") as capture:
            for method in (
                IrisScientificAblationRequestV1.execute,
                IrisScientificAblationRequestV1.generate,
            ):
                with pytest.raises(IrisRequestError, match="busy"):
                    method(request, prompt, strategy="greedy")
            capture.assert_not_called()
        with pytest.raises(IrisRequestError, match="Owned ablation"):
            IrisRequestOrchestrator.execute(request._request, prompt, strategy="greedy")
        return native(prompt)

    with patch.object(backbone, "forward_prompt", side_effect=forward):
        result = request.execute(_rows()[0].prompt, strategy="greedy")
    assert result.receipt.artifact_sha256 == trusted[1][0]
    assert request._state == request._owner._state == "ready"


def test_sampling_uses_advanced_full_state_and_commits_only_after_restoration(trusted: Any) -> None:
    request = _load(trusted)
    generator = torch.Generator(device="cpu").manual_seed(8421)
    torch.rand((47,), generator=generator, device="cpu")
    initial = generator.get_state().clone()
    caller = capture_runtime_state(torch.device("cpu"))
    native = runtime._restore_context
    observed: list[str] = []

    def restore(snapshot: Any, device: torch.device) -> None:
        assert request._owner._state == "executing" and request._state == "busy"
        assert torch.equal(generator.get_state(), initial)
        observed.append("restore")
        native(snapshot, device)

    with patch.object(runtime, "_restore_context", side_effect=restore):
        output = request.execute(_rows()[0].prompt, strategy="sampling", generator=generator)
    assert observed == ["restore"]
    final = generator.get_state().clone()
    assert not torch.equal(initial, final)
    assert output.receipt.sampling_seed == 8421
    domain = runtime._RECEIPT_DOMAIN + "/sampling-state"
    assert output.receipt.sampling_initial_state_sha256 == tensor_sha256(initial, domain=domain)
    assert output.receipt.sampling_final_state_sha256 == tensor_sha256(final, domain=domain)
    assert output.receipt.sampling_initial_state_sha256 != tensor_sha256(
        torch.Generator(device="cpu").manual_seed(8421).get_state(), domain=domain
    )
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "failure", ("receipt", "output", "runtime", "caller_generator", "session_length")
)
def test_post_core_failure_withholds_and_restores_full_caller_state(
    trusted: Any, failure: str
) -> None:
    request = _load(trusted)
    generator = torch.Generator(device="cpu").manual_seed(912)
    torch.rand((37,), generator=generator, device="cpu")
    initial = generator.get_state().clone()
    caller = capture_runtime_state(torch.device("cpu"))
    native = runtime._receipt

    def faulty(*args: Any, **kwargs: Any) -> Any:
        assert request._owner._state == "executing"
        assert not torch.equal(args[-1], initial)
        if failure == "receipt":
            raise ValueError("actual post-core receipt failure")
        if failure == "runtime":
            random.random()
            torch.rand((4,), device="cpu")
        elif failure == "caller_generator":
            generator.manual_seed(27)
        elif failure == "session_length":
            args[3]._length += 1
        return native(*args, **kwargs)

    with patch.object(runtime, "_receipt", side_effect=faulty):
        if failure == "output":
            with (
                patch.object(
                    runtime,
                    "IrisScientificAblationRequestOutputV1",
                    side_effect=ValueError("output"),
                ),
                pytest.raises(IrisRequestError),
            ):
                request.execute(_rows()[0].prompt, strategy="sampling", generator=generator)
        else:
            with pytest.raises(IrisRequestError):
                request.execute(_rows()[0].prompt, strategy="sampling", generator=generator)
    assert torch.equal(generator.get_state(), initial) and generator.initial_seed() == 912
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert request._state == request._owner._state == "poisoned"


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, BaseExceptionGroup))
@pytest.mark.parametrize("rollback_failure", (False, True))
def test_actual_partial_commit_preserves_first_signal_and_secondary_evidence(
    trusted: Any, signal: Any, rollback_failure: bool
) -> None:
    request = _load(trusted)
    generator = torch.Generator(device="cpu").manual_seed(917)
    initial = generator.get_state().clone()
    caller = capture_runtime_state(torch.device("cpu"))
    primary = (
        BaseExceptionGroup("primary", [KeyboardInterrupt("first"), ValueError("also")])
        if signal is BaseExceptionGroup
        else signal("first")
    )
    native = runtime._restore_sampling_state

    def commit(target: torch.Generator, state: torch.Tensor) -> None:
        assert request._owner._state == "ready" and request._state == "busy"
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        target.set_state(state)
        raise primary

    def rollback(target: torch.Generator, state: torch.Tensor) -> None:
        native(target, state)
        if rollback_failure:
            raise RuntimeError("secondary rollback uncertainty")

    with (
        patch.object(runtime, "_commit_sampling_state", side_effect=commit),
        patch.object(runtime, "_restore_sampling_state", side_effect=rollback),
        pytest.raises(BaseException) as caught,
    ):
        request.execute(_rows()[0].prompt, strategy="sampling", generator=generator)
    assert caught.value is primary
    assert torch.equal(generator.get_state(), initial)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert request._state == request._owner._state == "poisoned"
    if rollback_failure:
        assert isinstance(primary.__cause__, BaseExceptionGroup)


def test_loader_constructor_signal_and_restore_failure_preserve_first_object(trusted: Any) -> None:
    backbone = _fresh()._backbone
    caller = capture_runtime_state(torch.device("cpu"))
    primary = KeyboardInterrupt("constructor")
    native = runtime._restore_context

    def construct(**kwargs: Any) -> Any:
        torch.rand((8,), device="cpu")
        random.random()
        raise primary

    def restore(snapshot: Any, device: torch.device) -> None:
        native(snapshot, device)
        raise RuntimeError("restoration uncertain")

    with (
        patch.object(runtime, "_compose_request_orchestrator", side_effect=construct),
        patch.object(runtime, "_restore_context", side_effect=restore),
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        _load(trusted, backbone=backbone)
    assert caught.value is primary
    assert isinstance(primary.__cause__, BaseExceptionGroup)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_alias_equal_value_reinstall_is_detected_before_forward(trusted: Any) -> None:
    from safetensors.torch import load as load_tensors

    request = _load(trusted)
    raw = request._request
    decoded = load_tensors(trusted[0].state_bytes)
    state = {name: decoded[name] for name in request.manifest.payload.model.state_names}
    install_request_sidecar_state_v1(
        modules=raw._sidecar_modules,
        state=state,
        expected_current_sha256=raw.sidecar_state_sha256,
        expected_installed_sha256=raw.sidecar_state_sha256,
    )
    with patch.object(raw._backbone, "forward_prompt") as forward, pytest.raises(IrisRequestError):
        request.execute(_rows()[0].prompt, strategy="greedy")
    forward.assert_not_called()
    assert request._state == request._owner._state == "poisoned"


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_genuine_earlier_best_public_all12_c3_workspace_logits_and_distinct_receipt(
    routing: Any, goal: str, broadcast: str
) -> None:
    source, controller = _source(routing, goal=goal, broadcast=broadcast)
    current = build_scientific_ablation_checkpoint_v1(controller._run)
    restored: list[IrisScientificAblationRunV1] = []
    factory = partial(_fresh, routing, goal=goal, broadcast=broadcast)

    def make() -> IrisScientificAblationRunV1:
        run = factory()
        restored.append(run)
        return run

    bundle = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=make
    )
    anchors = (
        bundle.manifest.sha256,
        bundle.manifest.payload.source_checkpoint_sha256,
        source.manifest.sha256,
    )
    model = restored[0]._optimizer._model
    model.eval()
    row = _rows()[0]
    teacher_logits: list[torch.Tensor] = []
    native_teacher = model.teacher.backbone.forward_embeddings

    def teacher_forward(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        result = native_teacher(*args, **kwargs)
        target = int(torch.nonzero(kwargs["labels"][0] != -100)[0, 0])
        teacher_logits.append(result.logits[:, target - 1, :].detach().clone())
        return result

    with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=teacher_forward):
        reference = model.evaluation_forward(row, terminal_token_id=1)
    request = _load((bundle, anchors), backbone=factory()._backbone)
    assert request._request._backbone._model is not model.teacher.backbone._model
    native_request = request._request._backbone.forward_embeddings
    request_logits: list[torch.Tensor] = []

    def request_forward(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        result = native_request(*args, **kwargs)
        request_logits.append(result.logits[:, -1, :].detach().clone())
        return result

    caller = capture_runtime_state(torch.device("cpu"))
    for _ in range(2):
        with patch.object(
            request._request._backbone, "forward_embeddings", side_effect=request_forward
        ):
            output = request.execute(
                row.prompt,
                strategy="greedy",
                diagnostics=WorkspaceTraceOptionsV1(
                    mode="metadata", example_id=row.id, checkpoint_id=anchors[1], seed=901
                ),
            )
        for actual, expected in zip(
            output.generation.workspace_trace, reference.trace.teacher.workspace_trace, strict=True
        ):
            for field in fields(actual):
                assert torch.equal(getattr(actual, field.name), getattr(expected, field.name))
        torch.testing.assert_close(request_logits[-1], teacher_logits[0], rtol=1e-5, atol=1e-6)
        assert output.generation.token_ids.tolist() == [teacher_logits[0].argmax(-1).tolist()]
        assert output.receipt.artifact_sha256 == anchors[0]
        assert output.receipt.source_checkpoint_sha256 == anchors[1]
        assert output.receipt.completed_steps == 1 and output.receipt.cycles == 3
        assert output.diagnostic_trace is not None
        assert (
            output.diagnostic_trace.intervention_plan_sha256
            == output.receipt.ablation_policy_sha256
        )
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert len(request_logits) == 2 and len(teacher_logits) == 1
    assert build_scientific_ablation_checkpoint_v1(controller._run) == current


@pytest.mark.parametrize(
    "field,value",
    (
        ("kind", "iris-request-execution-v1"),
        ("semantics", "baseline"),
        ("selection", "best"),
        ("cycles", True),
        ("artifact_sha256", "0"),
        ("sampling_seed", 1),
        ("ablation_policy_json", "{}"),
    ),
)
def test_receipt_rejects_baseline_labels_and_malformed_exact_fields(
    trusted: Any, field: str, value: Any
) -> None:
    receipt = _load(trusted).execute(_rows()[0].prompt, strategy="greedy").receipt
    with pytest.raises((ValueError, TypeError)):
        replace(receipt, **{field: value})


@pytest.mark.parametrize(
    "field", ("configuration_json", "checkpoint_runtime_sha256", "hidden_size", "device", "dtype")
)
def test_actual_incompatible_backbone_fails_before_composition_and_restores_getter_rng(
    trusted: Any, field: str
) -> None:
    backbone = _fresh()._backbone
    caller = capture_runtime_state(torch.device("cpu"))
    wrong: dict[str, Any] = {
        "configuration_json": "{}",
        "checkpoint_runtime_sha256": "0" * 64,
        "hidden_size": backbone.hidden_size + 1,
        "device": torch.device("meta"),
        "dtype": torch.float64,
    }

    def getter(self: BackboneAdapter) -> Any:
        random.random()
        torch.rand((3,), device="cpu")
        return wrong[field]

    with (
        patch.object(type(backbone), field, new=property(getter)),
        patch.object(runtime, "_compose_request_orchestrator") as compose,
        pytest.raises(IrisRequestError),
    ):
        _load(trusted, backbone=backbone)
    compose.assert_not_called()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, BaseExceptionGroup))
@pytest.mark.parametrize("rollback_failure", (False, True))
def test_actual_loader_partial_copy_first_signal_outer_runtime_and_poison(
    trusted: Any, signal: Any, rollback_failure: bool
) -> None:
    from iris.application import request_sidecar_state as installer

    backbone = _fresh()._backbone
    caller = capture_runtime_state(torch.device("cpu"))
    primary = (
        BaseExceptionGroup("copy-primary", [KeyboardInterrupt("copy"), ValueError("also")])
        if signal is BaseExceptionGroup
        else signal("copy")
    )
    observed: list[Any] = []
    native_compose, native_copy = _compose_request_orchestrator, installer._copy_tensor

    def compose(**kwargs: Any) -> Any:
        request = native_compose(**kwargs)
        observed.append(request)
        return request

    count = 0

    def copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise primary
        if count == 3 and rollback_failure:
            raise RuntimeError("actual secondary copy rollback failure")
        native_copy(target, source)
        if count == 1:
            random.random()
            torch.rand((3,), device="cpu")

    with (
        patch.object(runtime, "_compose_request_orchestrator", side_effect=compose),
        patch.object(installer, "_copy_tensor", side_effect=copy_tensor),
        pytest.raises(BaseException) as caught,
    ):
        _load(trusted, backbone=backbone)
    assert caught.value is primary and count >= 3
    assert observed[0]._ablation_anchor._state == "poisoned"
    assert observed[0]._ablation_anchor._wrapper._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    if rollback_failure:
        assert primary.__cause__ is not None


def test_loader_rejects_silent_runtime_restoration_without_releasing_wrapper(trusted: Any) -> None:
    backbone = _fresh()._backbone
    caller = capture_runtime_state(torch.device("cpu"))
    observed: list[Any] = []
    native = _compose_request_orchestrator

    def compose(**kwargs: Any) -> Any:
        request = native(**kwargs)
        observed.append(request)
        return request

    try:
        with (
            patch.object(runtime, "_compose_request_orchestrator", side_effect=compose),
            patch.object(runtime, "restore_runtime_state"),
            pytest.raises(IrisRequestError) as caught,
        ):
            _load(trusted, backbone=backbone)
        assert isinstance(caught.value.__cause__, BaseExceptionGroup)
        assert observed[0]._ablation_anchor._state == "poisoned"
        assert observed[0]._ablation_anchor._wrapper._state == "poisoned"
    finally:
        # Fault injection intentionally prevented production restoration.
        restore_runtime_state(caller)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
