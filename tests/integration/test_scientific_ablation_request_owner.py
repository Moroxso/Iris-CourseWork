"""Real private ownership/copy/core boundaries; NOT a public loader/RNG proof."""

from copy import copy
from dataclasses import fields
from functools import partial
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application._scientific_ablation_request_owner import (
    _AblationRequestBinding,
    _AblationRequestOwner,
)
from iris.application.composition import _compose_request_orchestrator, build_request_orchestrator
from iris.application.errors import IrisRequestError
from iris.application.request import IrisRequestOrchestrator, _ScientificAblationOwnedRequest
from iris.application.request_sidecar_state import install_request_sidecar_state_v1
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    _checked,
    build_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_optimization import _same_runtime
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import TemperatureTopPSamplingConfig
from iris.utils.hashing import module_state_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_cognitive_ablation_request_core import _session
from tests.integration.test_scientific_ablation_inference_projection import _fresh, _source
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bundle() -> IrisScientificAblationInferenceBundleV1:
    source, _ = _source()
    return build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=_fresh
    )


def _new(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> tuple[_ScientificAblationOwnedRequest, _AblationRequestOwner, object, dict[str, torch.Tensor]]:
    # Trusted freshly constructed bundle; independent public loader will require
    # EXTERNAL expected digests, not these candidate-derived fixture values.
    manifest, decoded = _checked(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.source_checkpoint_sha256,
        expected_lifecycle_sha256=bundle.manifest.payload.selection.lifecycle_sha256,
    )
    fresh = _fresh()
    request = _compose_request_orchestrator(
        backbone=fresh._backbone,
        cognitive_config=manifest.payload.model.cognitive,
        workspace_config=manifest.payload.model.workspace,
        stop_token_ids=tuple(range(fresh._backbone.vocabulary_size)),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        constructor=_ScientificAblationOwnedRequest,
    )
    wrapper = object()
    payload = manifest.payload
    owner = _AblationRequestOwner(
        request,
        wrapper=wrapper,
        binding=_AblationRequestBinding(
            artifact_sha256=manifest.sha256,
            source_checkpoint_sha256=payload.source_checkpoint_sha256,
            lifecycle_sha256=payload.selection.lifecycle_sha256,
            policy_sha256=payload.ablation_policy_sha256,
            sidecar_sha256=payload.model.sha256,
        ),
    )
    state = {name: decoded[name] for name in payload.model.state_names}
    return request, owner, wrapper, state


def _ready(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> tuple[_ScientificAblationOwnedRequest, _AblationRequestOwner, object]:
    request, owner, wrapper, state = _new(bundle)
    cap = owner._reserve_bootstrap(wrapper)
    request._install_owned_sidecar(state, _authorization=cap)
    assert cap.consumed and owner._state == "ready"
    return request, owner, wrapper


@pytest.mark.parametrize(
    "method",
    (
        "generate",
        "execute",
        "execute_intervened",
        "restore_teacher_sidecar_state_v1",
    ),
)
@pytest.mark.parametrize("unbound", (False, True))
@pytest.mark.parametrize("descendant", (False, True))
def test_intrinsic_public_denial_without_anchor_before_context_or_effects(
    method: str, unbound: bool, descendant: bool
) -> None:
    class Descendant(_ScientificAblationOwnedRequest):
        pass

    request = object.__new__(Descendant if descendant else _ScientificAblationOwnedRequest)
    request.__dict__["is_owned"] = lambda: False
    call = getattr(IrisRequestOrchestrator, method) if unbound else getattr(request, method)
    args: tuple[Any, ...] = (request,) if unbound else ()
    kwargs: dict[str, Any] = {}
    if method == "restore_teacher_sidecar_state_v1":
        args += ({},)
    else:
        args += ("prompt",)
        kwargs["strategy"] = "greedy"
        if method == "execute_intervened":
            kwargs["plan"] = object()
    with (
        patch.object(torch.inference_mode, "__enter__") as context,
        patch("iris.application.request.install_request_sidecar_state_v1") as install,
        patch("iris.application.request._capture_request_execution_receipt_v1") as receipt,
        pytest.raises(IrisRequestError, match="Owned ablation"),
    ):
        call(*args, **kwargs)
    context.assert_not_called()
    install.assert_not_called()
    receipt.assert_not_called()


def test_private_constructor_reuses_real_public_initialization_order(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> None:
    fresh = _fresh()
    kwargs: dict[str, Any] = dict(
        backbone=fresh._backbone,
        cognitive_config=bundle.manifest.payload.model.cognitive,
        workspace_config=bundle.manifest.payload.model.workspace,
        stop_token_ids=(1,),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(937)
        baseline = build_request_orchestrator(**kwargs)
        consumed = torch.get_rng_state().clone()
        torch.manual_seed(937)
        private = _compose_request_orchestrator(
            **kwargs, constructor=_ScientificAblationOwnedRequest
        )
        assert torch.equal(torch.get_rng_state(), consumed)
    assert type(baseline) is IrisRequestOrchestrator
    assert type(private) is _ScientificAblationOwnedRequest
    assert baseline.sidecar_state_sha256 == private.sidecar_state_sha256
    for (_, original), (_, owned) in zip(
        baseline._sidecar_modules, private._sidecar_modules, strict=True
    ):
        assert original is not owned


def test_bootstrap_capability_is_consumed_before_copy_and_cannot_retransfer(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> None:
    request, owner, wrapper, state = _new(bundle)
    cap = owner._reserve_bootstrap(wrapper)
    from iris.application import request_sidecar_state

    native = request_sidecar_state._copy_tensor

    def guarded_copy(target: torch.Tensor, source: torch.Tensor) -> None:
        assert cap.consumed and owner._entered and owner._state == "bootstrapping"
        with pytest.raises(IrisRequestError):
            IrisRequestOrchestrator.restore_teacher_sidecar_state_v1(request, state)
        native(target, source)

    with patch.object(request_sidecar_state, "_copy_tensor", side_effect=guarded_copy):
        request._install_owned_sidecar(state, _authorization=cap)
    assert owner._state == "ready" and request.sidecar_state_sha256 == owner._binding.sidecar_sha256
    with patch("iris.application.request.install_request_sidecar_state_v1") as install:
        for stale in (cap, copy(cap), object()):
            with pytest.raises(IrisRequestError):
                request._install_owned_sidecar(state, _authorization=stale)
        install.assert_not_called()
    with pytest.raises(IrisRequestError):
        owner._reserve_bootstrap(wrapper)


@pytest.mark.parametrize("fault", ("missing", "foreign", "binding", "binding_values"))
def test_corrupt_anchor_or_binding_fails_closed_before_copy_and_forward(
    bundle: IrisScientificAblationInferenceBundleV1,
    fault: str,
) -> None:
    request, owner, wrapper, state = _new(bundle)
    cap = owner._reserve_bootstrap(wrapper)
    if fault == "missing":
        del request._ablation_anchor
    elif fault == "foreign":
        request._ablation_anchor = object()
    elif fault == "binding":
        request._ablation_binding = object()
    else:
        object.__setattr__(owner._binding, "policy_sha256", "0" * 64)
    with patch("iris.application.request.install_request_sidecar_state_v1") as install:
        with pytest.raises(IrisRequestError):
            request._install_owned_sidecar(state, _authorization=cap)
        install.assert_not_called()
    with patch.object(request.backbone, "forward_prompt") as forward:
        with pytest.raises(IrisRequestError):
            request._execute_core(_rows()[0].prompt, strategy="greedy")
        forward.assert_not_called()
    with pytest.raises(IrisRequestError):
        owner._live(wrapper)
    assert owner._state == "poisoned"


def test_valid_owned_capability_cannot_authorize_an_ordinary_request(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> None:
    request, owner, wrapper = _ready(bundle)
    prompt = _rows()[0].prompt
    session = _session(request, prompt, bundle.manifest.payload.ablation)
    cap = owner._reserve_execution(wrapper, prompt=prompt, session=session, strategy="greedy")
    ordinary = object.__new__(IrisRequestOrchestrator)
    with (
        patch.object(torch.inference_mode, "__enter__") as context,
        patch("iris.application.request.WorkspaceTraceRecorderV1") as diagnostics,
        pytest.raises(IrisRequestError, match="ordinary request"),
    ):
        IrisRequestOrchestrator._execute_core(
            ordinary, prompt, strategy="greedy", ablation=session, _authorization=cap
        )
    context.assert_not_called()
    diagnostics.assert_not_called()
    assert not cap.consumed


@pytest.mark.parametrize(
    "fault",
    (
        "policy_value",
        "policy_extra",
        "prompt_digest",
        "geometry",
        "hidden_session",
    ),
)
def test_completion_rejects_post_core_session_binding_corruption_and_poisons(
    bundle: IrisScientificAblationInferenceBundleV1,
    fault: str,
) -> None:
    request, owner, wrapper = _ready(bundle)
    prompt = _rows()[0].prompt
    session = _session(request, prompt, bundle.manifest.payload.ablation)
    cap = owner._reserve_execution(wrapper, prompt=prompt, session=session, strategy="greedy")
    request._execute_core(prompt, strategy="greedy", ablation=session, _authorization=cap)
    if fault == "policy_value":
        object.__setattr__(session._policy, "goal", "disabled")
    elif fault == "policy_extra":
        vars(session._policy)["ghost"] = True
    elif fault == "prompt_digest":
        session.prompt_sha256 = "0" * 64
    elif fault == "geometry":
        session.modules += 1
    else:
        vars(session)["ghost"] = True
    with pytest.raises(IrisRequestError):
        owner._complete_execution(wrapper, cap)
    assert owner._state == "poisoned"


@pytest.mark.parametrize("fault", ("foreign_owner", "wrong_operation", "wrong_wrapper"))
def test_genuine_foreign_or_wrong_operation_capabilities_reject_before_copy_or_forward(
    bundle: IrisScientificAblationInferenceBundleV1,
    fault: str,
) -> None:
    request, owner, wrapper = _ready(bundle)
    prompt = _rows()[0].prompt
    session = _session(request, prompt, bundle.manifest.payload.ablation)
    own_cap = owner._reserve_execution(wrapper, prompt=prompt, session=session, strategy="greedy")
    other_request, other_owner, other_wrapper, state = _new(bundle)
    bootstrap = other_owner._reserve_bootstrap(other_wrapper)
    if fault == "foreign_owner":
        other_request._install_owned_sidecar(state, _authorization=bootstrap)
        wrong: object = other_owner._reserve_execution(
            other_wrapper,
            prompt=prompt,
            session=_session(other_request, prompt, bundle.manifest.payload.ablation),
            strategy="greedy",
        )
    elif fault == "wrong_operation":
        wrong = bootstrap
    else:
        wrong = copy(own_cap)
        object.__setattr__(wrong, "wrapper", object())
    with (
        patch.object(torch.inference_mode, "__enter__") as context,
        patch.object(request.backbone, "forward_prompt") as forward,
        pytest.raises(IrisRequestError),
    ):
        request._execute_core(prompt, strategy="greedy", ablation=session, _authorization=wrong)
    context.assert_not_called()
    forward.assert_not_called()
    assert not own_cap.consumed


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_genuine_earlier_best_all12_owned_c3_workspace_logits_and_greedy(
    routing: Any,
    goal: str,
    broadcast: str,
) -> None:
    source, controller = _source(routing, goal=goal, broadcast=broadcast)
    current_before = build_scientific_ablation_checkpoint_v1(controller._run)
    restored: list[IrisScientificAblationRunV1] = []
    factory = partial(_fresh, routing, goal=goal, broadcast=broadcast)

    def capture_fresh() -> IrisScientificAblationRunV1:
        run = factory()
        restored.append(run)
        return run

    projected = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=capture_fresh
    )
    assert projected.manifest.payload.completed_steps == 1 and controller.state.step == 3
    model = restored[0]._optimizer._model
    model.eval()
    row = _rows()[0]
    teacher_logits: list[torch.Tensor] = []
    teacher_native = model.teacher.backbone.forward_embeddings

    def teacher_forward(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        output = teacher_native(*args, **kwargs)
        target = int(torch.nonzero(kwargs["labels"][0] != -100)[0, 0])
        teacher_logits.append(output.logits[:, target - 1, :].detach().clone())
        return output

    with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=teacher_forward):
        reference = model.evaluation_forward(row, terminal_token_id=1)
    request, owner, wrapper = _ready(projected)
    assert request.backbone._model is not model.teacher.backbone._model
    request_native = request.backbone.forward_embeddings
    request_logits: list[torch.Tensor] = []

    def request_forward(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        output = request_native(*args, **kwargs)
        request_logits.append(output.logits[:, -1, :].detach().clone())
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    for _ in range(2):
        session = _session(request, row.prompt, projected.manifest.payload.ablation)
        cap = owner._reserve_execution(
            wrapper, prompt=row.prompt, session=session, strategy="greedy"
        )
        with patch.object(request.backbone, "forward_embeddings", side_effect=request_forward):
            # Also exercises genuine unbound keyword-self compatibility.
            actual = IrisRequestOrchestrator._execute_core(
                self=request,
                prompt=row.prompt,
                strategy="greedy",
                ablation=session,
                _authorization=cap,
            )
        assert not hasattr(actual, "receipt")
        assert len(actual.generation.workspace_trace) == 3
        for observed, expected in zip(
            actual.generation.workspace_trace, reference.trace.teacher.workspace_trace, strict=True
        ):
            for field in fields(observed):
                assert torch.equal(getattr(observed, field.name), getattr(expected, field.name))
        torch.testing.assert_close(request_logits[-1], teacher_logits[0], rtol=1e-5, atol=1e-6)
        assert actual.generation.token_ids.tolist() == [teacher_logits[0].argmax(-1).tolist()]
        owner._complete_execution(wrapper, cap)
        assert owner._state == "ready"
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert len(request_logits) == 2 and len(teacher_logits) == 1
    assert build_scientific_ablation_checkpoint_v1(controller._run) == current_before


def test_authorized_genuine_core_and_unbound_callback_reentry_are_separate(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> None:
    request, owner, wrapper = _ready(bundle)
    prompt = _rows()[0].prompt
    session = _session(request, prompt, bundle.manifest.payload.ablation)
    cap = owner._reserve_execution(wrapper, prompt=prompt, session=session, strategy="greedy")
    native = request.backbone.forward_prompt

    def forward(text: str) -> Any:
        assert cap.consumed and owner._entered and owner._state == "executing"
        with pytest.raises(IrisRequestError):
            IrisRequestOrchestrator.execute(request, text, strategy="greedy")
        with pytest.raises(IrisRequestError):
            IrisRequestOrchestrator._execute_core(
                request, text, strategy="greedy", ablation=session
            )
        with pytest.raises(IrisRequestError):
            request._execute_core(text, strategy="greedy", ablation=session, _authorization=cap)
        with pytest.raises(IrisRequestError):
            owner._reserve_execution(wrapper, prompt=prompt, session=session, strategy="greedy")
        return native(text)

    with patch.object(request.backbone, "forward_prompt", side_effect=forward):
        result = request._execute_core(
            prompt, strategy="greedy", ablation=session, _authorization=cap
        )
    assert not hasattr(result, "receipt") and session._finished
    assert owner._state == "executing"  # full outer facade transaction NOT silently completed
    owner._complete_execution(wrapper, cap)
    assert owner._state == "ready" and owner._pending is None
    with patch.object(request.backbone, "forward_prompt") as no_forward:
        with pytest.raises(IrisRequestError):
            request._execute_core(prompt, strategy="greedy", ablation=session, _authorization=cap)
        no_forward.assert_not_called()


@pytest.mark.parametrize("fault", ("missing", "cloned", "foreign", "wrong_prompt", "wrong_session"))
def test_wrong_execution_capability_rejects_before_inference_or_forward(
    bundle: IrisScientificAblationInferenceBundleV1,
    fault: str,
) -> None:
    request, owner, wrapper = _ready(bundle)
    prompt = _rows()[0].prompt
    session = _session(request, prompt, bundle.manifest.payload.ablation)
    cap: object = owner._reserve_execution(
        wrapper, prompt=prompt, session=session, strategy="greedy"
    )
    if fault == "missing":
        cap = None
    elif fault == "cloned":
        cap = copy(cap)
    elif fault == "foreign":
        cap = object()
    elif fault == "wrong_prompt":
        prompt += "x"
    else:
        session = _session(request, prompt, bundle.manifest.payload.ablation)
    with (
        patch.object(torch.inference_mode, "__enter__") as context,
        patch.object(request.backbone, "forward_prompt") as forward,
        patch("iris.application.request.WorkspaceTraceRecorderV1") as diagnostics,
        pytest.raises(IrisRequestError),
    ):
        request._execute_core(prompt, strategy="greedy", ablation=session, _authorization=cap)
    context.assert_not_called()
    forward.assert_not_called()
    diagnostics.assert_not_called()


@pytest.mark.parametrize("signal", ("interrupt", "exit", "group"))
@pytest.mark.parametrize("rollback_fails", (False, True))
def test_partial_bootstrap_preserves_exact_primary_signal_and_poisons(
    bundle: IrisScientificAblationInferenceBundleV1,
    signal: str,
    rollback_fails: bool,
) -> None:
    request, owner, wrapper, state = _new(bundle)
    original = module_state_sha256(request._sidecar_modules)
    failure: BaseException = (
        KeyboardInterrupt("owned-copy")
        if signal == "interrupt"
        else SystemExit(19)
        if signal == "exit"
        else BaseExceptionGroup("owned-copy", [KeyboardInterrupt("first")])
    )
    from iris.application import request_sidecar_state

    native = request_sidecar_state._copy_tensor
    calls = 0

    def copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        if calls > 2 and rollback_fails:
            raise RuntimeError("rollback-secondary")
        native(target, source)

    cap = owner._reserve_bootstrap(wrapper)
    with (
        patch.object(request_sidecar_state, "_copy_tensor", side_effect=copy_tensor),
        pytest.raises(type(failure)) as caught,
    ):
        request._install_owned_sidecar(state, _authorization=cap)
    assert caught.value is failure and cap.consumed and owner._state == "poisoned"
    if not rollback_fails:
        assert module_state_sha256(request._sidecar_modules) == original
    else:
        assert failure.__cause__ is not None
    with pytest.raises(IrisRequestError):
        owner._reserve_bootstrap(wrapper)


def test_extracted_alias_equal_value_reinstall_is_detected_by_version_and_poisoned(
    bundle: IrisScientificAblationInferenceBundleV1,
) -> None:
    request, owner, wrapper = _ready(bundle)
    state = {
        f"{name}.{key}": value.detach().clone()
        for name, module in request._sidecar_modules
        for key, value in module.state_dict().items()
    }
    before = request.sidecar_state_sha256
    install_request_sidecar_state_v1(
        modules=request._sidecar_modules,
        state=state,
        expected_current_sha256=before,
        expected_installed_sha256=before,
    )
    assert request.sidecar_state_sha256 == before  # equal digest is NOT enough
    prompt = _rows()[0].prompt
    with patch.object(request.backbone, "forward_prompt") as forward:
        with pytest.raises(IrisRequestError, match="version/storage"):
            owner._reserve_execution(
                wrapper,
                prompt=prompt,
                session=_session(request, prompt, bundle.manifest.payload.ablation),
                strategy="greedy",
            )
        forward.assert_not_called()
    assert owner._state == "poisoned"
