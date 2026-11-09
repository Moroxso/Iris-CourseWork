"""Physical original reconstruction plus NEW controlled ownership/generation.

Tiny engineering source/rows only. Never ordinary capture, held-out quality or
candidate execution. Whole native/current/inference files are retained by pytest.
"""

import random
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from iris.application import scientific_ablation_controlled_generation_context as module
from iris.application.request_sidecar_state import _target_state
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_controlled_generation_context import (
    IrisScientificAblationControlledGenerationContextError,
    IrisScientificAblationControlledGenerationContextV1,
    build_scientific_ablation_controlled_generation_context_v1,
)
from iris.application.scientific_ablation_controlled_request import (
    IrisScientificAblationControlledRequestV1,
)
from iris.application.scientific_ablation_controlled_request import (
    build_scientific_ablation_controlled_request_v1 as build_request,
)
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import _bundle_parts
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_generation_reconstruction_v1 import (
    IrisScientificAblationGenerationReconstructionV1,
)
from iris.data.schemas import IrisExample
from iris.generation import TemperatureTopPSamplingConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_controlled_request import control
from tests.integration.test_scientific_ablation_generation_source import (
    Routing,
    _prepared,
    _read,
)
from tests.integration.test_scientific_ablation_lifecycle import _controller
from tests.integration.test_scientific_ablation_run import _run
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration
Error = IrisScientificAblationControlledGenerationContextError


@pytest.fixture(scope="module", autouse=True)
def no_loaders_candidates_or_ordinary_owner() -> Iterator[None]:
    with (
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No model load"),
        ) as models,
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer load"),
        ) as tokenizers,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate execution"),
        ) as candidates,
        patch(
            "iris.application.scientific_ablation_request.build_scientific_ablation_request_v1",
            side_effect=AssertionError("No ordinary owner creation/adoption"),
        ) as ordinary,
    ):
        yield
        for forbidden in (models, tokenizers, candidates, ordinary):
            forbidden.assert_not_called()


def _fresh(
    routing: Routing = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    maximum: int = 3,
) -> IrisScientificAblationRunV1:
    return _run(
        routing,
        goal=goal,
        broadcast=broadcast,
        cycles=3,
        policy=_policy(minimum_target_tokens=1, max_optimizer_steps=maximum),
    )


def _build(
    bundle: IrisScientificAblationGenerationSourceBundleV1,
    *,
    factory: Callable[[], IrisScientificAblationRunV1] = _fresh,
    validation: tuple[IrisExample, ...] | None = None,
) -> IrisScientificAblationControlledGenerationContextV1:
    return build_scientific_ablation_controlled_generation_context_v1(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_lifecycle_sha256=bundle.lifecycle.manifest.sha256,
        expected_inference_sha256=bundle.inference.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.best_checkpoint_sha256,
        make_fresh_run=factory,
        validation=_validation() if validation is None else validation,
        sampling=TemperatureTopPSamplingConfig(0.7, 0.9),
    )


@pytest.fixture(scope="module")
def bound(
    tmp_path_factory: pytest.TempPathFactory,
) -> IrisScientificAblationGenerationSourceBundleV1:
    return _read(_prepared(tmp_path_factory.mktemp("context-source")))


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_original_reconstruction_restores_earlier_best_and_owns_request(
    tmp_path: Path,
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    prepared = _prepared(tmp_path, routing, goal=goal, broadcast=broadcast)
    bundle = _read(prepared)
    source_bytes = (
        bundle.lifecycle.current_state_bytes,
        bundle.lifecycle.best_state_bytes,
        bundle.inference.state_bytes,
    )
    caller = capture_runtime_state(torch.device("cpu"))
    initial = _fresh(routing, goal=goal, broadcast=broadcast)
    observed_initial = build_scientific_ablation_checkpoint_v1(initial)
    result = _build(bundle, factory=lambda: initial)
    receipt = result.reconstruction
    assert receipt.sha256 == domain_sha256(
        module._DOMAIN, canonical_json_bytes(receipt.payload.model_dump(mode="json"))
    )
    p, source = receipt.payload, bundle.manifest.payload
    assert p.kind == "iris-scientific-ablation-controlled-generation-reconstruction-v1"
    assert p.request_semantics == "fresh_owned_controlled_generation_v1"
    with pytest.raises(ValidationError):
        IrisScientificAblationGenerationReconstructionV1.model_validate(
            receipt.model_dump(mode="python")
        )
    assert p.scope == "reconstructed_source_not_heldout_capture_or_launch_approval"
    assert p.source_sha256 == bundle.manifest.sha256
    assert p.fresh_initial_checkpoint_sha256 == observed_initial.manifest.sha256
    assert p.initial_scientific_state_sha256 == source.initial_scientific_state_sha256
    assert p.training_corpus_sha256 == source.training_corpus_sha256
    assert p.validation_corpus_sha256 == source.validation_corpus_sha256
    assert p.validation_target_tokens == source.validation_target_tokens
    assert (p.training_examples, p.validation_examples, p.selected_step) == (
        5,
        len(_validation()),
        1,
    )
    assert (
        source.terminal_step == 3
        and source.current_checkpoint_sha256 != p.selected_checkpoint_sha256
    )
    _, _, best = _bundle_parts(bundle.lifecycle)
    assert best is not None and build_scientific_ablation_checkpoint_v1(initial) == best
    assert initial._controller_attachment is None and not initial.poisoned
    assert result._request.manifest.payload.ablation == source.ablation
    assert result._request._request._backbone is initial._backbone
    assert all(parameter.grad is None for parameter in initial._optimizer._model.parameters())
    request_tensors = _target_state(result._request._request._sidecar_modules)
    native_tensors = initial._optimizer._model.base.teacher.state_dict()
    assert request_tensors.keys() == native_tensors.keys()
    for name, tensor in request_tensors.items():
        assert torch.equal(tensor, native_tensors[name])
        assert (
            tensor.untyped_storage().data_ptr() != native_tensors[name].untyped_storage().data_ptr()
        )
    assert source_bytes == (
        bundle.lifecycle.current_state_bytes,
        bundle.lifecycle.best_state_bytes,
        bundle.inference.state_bytes,
    )
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert type(result._request) is IrisScientificAblationControlledRequestV1
    output = result._request.execute(_rows()[0].prompt, control=control("zero"), strategy="greedy")
    assert output.receipt.artifact_sha256 == source.inference_sha256
    assert output.receipt.source_checkpoint_sha256 == source.best_checkpoint_sha256
    assert output.receipt.lifecycle_sha256 == source.lifecycle_sha256
    assert output.receipt.max_new_tokens == 256 and output.receipt.stop_token_ids == (1,)
    assert output.generation.stop_reason in {"max_new_tokens", "stop_token"}
    assert result.reconstruction == receipt
    assert build_scientific_ablation_checkpoint_v1(initial) == best
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    witness = {
        "reconstruction": receipt.model_dump(mode="json"),
        "source": bundle.manifest.model_dump(mode="json"),
        "source_lifecycle_path": str(prepared.native_path),
        "source_inference_path": str(prepared.inference_path),
        "receipt": asdict(output.receipt),
        "token_ids": output.generation.token_ids.tolist(),
        "stop_reason": output.generation.stop_reason,
        "scientific_acceptance": "not_assessed",
        "durable_controlled_capture": False,
    }
    target = tmp_path / "actual-controlled-source-reconstruction.json"
    with target.open("xb") as stream:
        stream.write(canonical_json_bytes(witness) + b"\n")


def test_shared_best_restores_actual_shared_native_bytes(tmp_path: Path) -> None:
    bundle = _read(_prepared(tmp_path, maximum=2, validate=5, checkpoint=3))
    assert bundle.lifecycle.best_state_bytes is None
    result = _build(bundle, factory=lambda: _fresh(maximum=2))
    assert result.reconstruction.payload.selected_step == 2
    _, _, best = _bundle_parts(bundle.lifecycle)
    assert build_scientific_ablation_checkpoint_v1(result._run) == best


@pytest.mark.parametrize("anchor", ("descriptor", "lifecycle", "inference", "checkpoint"))
def test_wrong_independent_anchor_precedes_factory(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    anchor: str,
) -> None:
    anchors = {
        "descriptor": bound.manifest.sha256,
        "lifecycle": bound.lifecycle.manifest.sha256,
        "inference": bound.inference.manifest.sha256,
        "checkpoint": bound.manifest.payload.best_checkpoint_sha256,
    }
    anchors[anchor] = "0" * 64
    with (
        patch.object(module, "restore_scientific_ablation_checkpoint_v1") as restore,
        patch(
            "tests.integration.test_scientific_ablation_controlled_generation_context._fresh"
        ) as factory,
        pytest.raises(Error),
    ):
        build_scientific_ablation_controlled_generation_context_v1(
            bound,
            expected_sha256=anchors["descriptor"],
            expected_lifecycle_sha256=anchors["lifecycle"],
            expected_inference_sha256=anchors["inference"],
            expected_source_checkpoint_sha256=anchors["checkpoint"],
            make_fresh_run=factory,
            validation=_validation(),
            sampling=TemperatureTopPSamplingConfig(),
        )
    restore.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize(
    "fault",
    (
        "model_seed",
        "run_seed",
        "train_order",
        "train_target",
        "mode",
        "optimizer",
        "progressed",
        "attached",
    ),
)
def test_wrong_original_factory_denied_before_native_copy_without_poisoning_foreign_run(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    rows = _rows()[:5]
    if fault == "train_order":
        rows = tuple(reversed(rows))
    elif fault == "train_target":
        rows = (
            IrisExample.model_validate(
                {
                    **rows[0].model_dump(mode="python"),
                    "expected_response": rows[0].expected_response + " ",
                }
            ),
            *rows[1:],
        )
    controller = _controller(maximum=3) if fault == "attached" else None
    candidate = (
        controller._run
        if controller is not None
        else _run(
            "uniform" if fault == "mode" else "learned",
            cycles=3,
            model_seed=1908 if fault == "model_seed" else 1907,
            run_seed=902 if fault == "run_seed" else 901,
            examples=rows,
            policy=_policy(
                minimum_target_tokens=1, max_optimizer_steps=4 if fault == "optimizer" else 3
            ),
        )
    )
    if fault == "progressed":
        candidate.step()
    before = build_scientific_ablation_checkpoint_v1(candidate)
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(module, "restore_scientific_ablation_checkpoint_v1") as restore,
        pytest.raises(Error),
    ):
        _build(bound, factory=lambda: candidate)
    restore.assert_not_called()
    assert not candidate.poisoned
    assert build_scientific_ablation_checkpoint_v1(candidate) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "fault", ("empty", "row_order", "target", "seed_overlap", "duplicate", "split", "unchecked")
)
def test_full_original_validation_is_required_before_selected_best_copy(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    rows = _validation()
    if fault == "empty":
        rows = ()
    elif fault == "row_order":
        rows = tuple(reversed(rows))
    elif fault == "duplicate":
        rows = (rows[0], rows[0], *rows[2:])
    else:
        row = rows[0].model_copy(deep=True)
        if fault == "target":
            vars(row)["expected_response"] += " "
        elif fault == "seed_overlap":
            vars(row)["seed"] = 0
        elif fault == "split":
            vars(row)["split"] = "test"
        else:
            vars(row)["untracked"] = True
        rows = (row, *rows[1:])
    with (
        patch.object(module, "restore_scientific_ablation_checkpoint_v1") as restore,
        pytest.raises(Error),
    ):
        _build(bound, validation=rows)
    restore.assert_not_called()


@pytest.mark.parametrize("inference", (False, True))
def test_ambient_meta_double_and_grad_context_do_not_leak_into_private_reconstruction(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    inference: bool,
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.inference_mode(inference), torch.no_grad():
            result = _build(bound)
            assert torch.empty(()).device.type == "meta"
            assert torch.get_default_dtype() == torch.float64
            assert torch.is_inference_mode_enabled() is inference and not torch.is_grad_enabled()
            assert result.reconstruction.payload.selected_step == 1
            assert all(
                p.dtype == torch.float32 and p.device.type == "cpu" and not p.is_inference()
                for p in result._run._optimizer._model.parameters()
            )
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(original_dtype)
        restore_runtime_state(caller)


def test_metadata_and_original_input_aliases_cannot_mutate_owned_context(
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    candidate = replace(
        bound,
        manifest=bound.manifest.model_copy(deep=True),
        lifecycle=replace(bound.lifecycle, manifest=bound.lifecycle.manifest.model_copy(deep=True)),
        inference=replace(bound.inference, manifest=bound.inference.manifest.model_copy(deep=True)),
    )
    rows = _validation()
    result = _build(candidate, validation=rows)
    before = result.reconstruction
    vars(result.reconstruction.payload)["selected_step"] = 100
    vars(result.source_manifest.payload.ablation)["untracked"] = True
    vars(candidate.manifest.payload)["training_corpus_sha256"] = "0" * 64
    vars(candidate.lifecycle.manifest.payload)["untracked"] = True
    vars(candidate.inference.manifest.payload)["untracked"] = True
    vars(rows[0])["expected_response"] = "CHANGED"
    assert result.reconstruction == before
    assert result.source_manifest == bound.manifest


@pytest.mark.parametrize(
    "fault",
    ("tensor", "equal_tensor_write", "gradient", "mode", "runtime", "request", "foreign_run"),
)
def test_retained_factory_alias_mutation_denies_context_and_poison_sealed_owners(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    result = _build(bound)
    native, request = result._run, result._request
    parameter = native._optimizer._topology.parameters[0]
    foreign = None
    if fault in ("tensor", "equal_tensor_write"):
        with torch.no_grad():
            parameter.add_(1.0 if fault == "tensor" else 0.0)
    elif fault == "gradient":
        parameter.grad = torch.ones_like(parameter)
    elif fault == "mode":
        native._optimizer._model.eval()
    elif fault == "runtime":
        native._runtime.cpu_rng_state[0] ^= 1
    elif fault == "request":
        with torch.no_grad():
            next(iter(_target_state(request._request._sidecar_modules).values())).add_(0.0)
    else:
        foreign = _fresh()
        result._run = foreign
    with pytest.raises(Error):
        _ = result.reconstruction
    assert result._state == "poisoned" and native.poisoned
    assert request._state == "poisoned" and request._owner._state == "poisoned"
    assert foreign is None or not foreign.poisoned


@pytest.mark.parametrize(
    "primary",
    (
        ValueError("factory"),
        KeyboardInterrupt("factory"),
        SystemExit(43),
        BaseExceptionGroup("factory", [ValueError("other"), KeyboardInterrupt("signal")]),
    ),
)
def test_factory_failure_restores_actual_caller_globals_and_preserves_primary_signal(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    primary: BaseException,
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()

    def factory() -> IrisScientificAblationRunV1:
        random.random()
        torch.rand(())
        torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(not caller.deterministic_algorithms)
        torch.backends.cudnn.benchmark = not caller.cudnn_benchmark
        raise primary

    expected = Error if isinstance(primary, Exception) else type(primary)
    with pytest.raises(expected) as caught:
        _build(bound, factory=factory)
    if isinstance(primary, Exception):
        assert caught.value.__cause__ is primary
    else:
        assert caught.value is primary
    assert torch.get_default_dtype() == dtype
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("phase", ("native_copy", "request_construction"))
@pytest.mark.parametrize(
    "primary",
    (
        ValueError("partial"),
        KeyboardInterrupt("partial"),
        SystemExit(47),
        BaseExceptionGroup("partial", [ValueError("other"), SystemExit(49)]),
    ),
)
def test_uncertain_restore_or_request_failure_withholds_and_poison_owner_preserving_signals(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    phase: str,
    primary: BaseException,
) -> None:
    native = _fresh()
    caller = capture_runtime_state(torch.device("cpu"))

    def uncertain(*args: object, **kwargs: object) -> None:
        with torch.no_grad():
            native._optimizer._topology.parameters[0].add_(1.0)
        raise primary

    target = (
        "restore_scientific_ablation_checkpoint_v1"
        if phase == "native_copy"
        else "build_scientific_ablation_controlled_request_v1"
    )
    expected = Error if isinstance(primary, Exception) else type(primary)
    with patch.object(module, target, side_effect=uncertain), pytest.raises(expected) as caught:
        _build(bound, factory=lambda: native)
    assert native.poisoned and native._optimizer.poisoned
    if isinstance(primary, Exception):
        assert caught.value.__cause__ is primary
    else:
        assert caught.value is primary
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "primary",
    (
        ValueError("primary"),
        KeyboardInterrupt("primary"),
        BaseExceptionGroup("first", [ValueError("other"), SystemExit(51)]),
    ),
)
def test_secondary_restoration_signal_cannot_replace_first_signal(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    primary: BaseException,
) -> None:
    secondary = SystemExit(53)
    caller = capture_runtime_state(torch.device("cpu"))

    def factory() -> IrisScientificAblationRunV1:
        raise primary

    actual_restore = module._restore_caller

    def restore_then_signal(
        snapshot: RuntimeStateSnapshot, dtype: torch.dtype, device: torch.device
    ) -> None:
        actual_restore(snapshot, dtype, device)
        raise secondary

    expected = type(primary) if _has_signal(primary) else type(secondary)
    with (
        patch.object(module, "_restore_caller", side_effect=restore_then_signal),
        pytest.raises(expected) as caught,
    ):
        _build(bound, factory=factory)
    assert caught.value is (primary if _has_signal(primary) else secondary)
    assert caught.value.__cause__ is (secondary if _has_signal(primary) else primary)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_successful_build_with_failed_caller_readback_is_withheld_and_poisoned(
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    native = _fresh()
    caller = capture_runtime_state(torch.device("cpu"))
    actual_restore = module._restore_caller
    requests: list[IrisScientificAblationControlledRequestV1] = []
    actual_build = build_request

    def restore_then_fail(
        snapshot: RuntimeStateSnapshot, dtype: torch.dtype, device: torch.device
    ) -> None:
        actual_restore(snapshot, dtype, device)
        raise ValueError("readback failure")

    def observe(*args: object, **kwargs: object) -> IrisScientificAblationControlledRequestV1:
        # Explicit forwarding avoids generic kwargs relaxing the production signature.
        result = actual_build(
            bound.inference,
            expected_sha256=bound.inference.manifest.sha256,
            expected_source_checkpoint_sha256=bound.manifest.payload.best_checkpoint_sha256,
            expected_lifecycle_sha256=bound.lifecycle.manifest.sha256,
            backbone=native._backbone,
            stop_token_ids=native._backbone.eos_token_ids,
            sampling=TemperatureTopPSamplingConfig(0.7, 0.9),
        )
        requests.append(result)
        return result

    with (
        patch.object(module, "_restore_caller", side_effect=restore_then_fail),
        patch.object(
            module, "build_scientific_ablation_controlled_request_v1", side_effect=observe
        ),
        pytest.raises(Error),
    ):
        _build(bound, factory=lambda: native)
    assert native.poisoned and requests[0]._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_caller_restore_attempts_rng_even_if_dtype_restore_fails() -> None:
    snapshot = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    with (
        patch.object(torch, "set_default_dtype", side_effect=ValueError("dtype")),
        patch.object(module, "restore_runtime_state", wraps=restore_runtime_state) as restore,
        pytest.raises(ExceptionGroup) as caught,
    ):
        module._restore_caller(snapshot, dtype, torch.device("cpu"))
    restore.assert_called_once_with(snapshot)
    assert str(caught.value.exceptions[0]) == "dtype"
    assert _same_runtime(snapshot, capture_runtime_state(torch.device("cpu")))


def test_silent_caller_rng_restoration_drift_is_rejected() -> None:
    snapshot = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    try:
        random.random()
        with patch.object(module, "restore_runtime_state"), pytest.raises(ExceptionGroup) as caught:
            module._restore_caller(snapshot, dtype, torch.device("cpu"))
        assert "without exact restoration" in str(caught.value.exceptions[0])
    finally:
        restore_runtime_state(snapshot)
