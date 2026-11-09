"""Private neutral core parity, not persisted/owned/public inference acceptance."""

from dataclasses import fields
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch

from iris.ablations import CognitiveAblationPolicyV1, CognitiveAblationSessionV1
from iris.application.composition import build_request_orchestrator
from iris.application.errors import IrisRequestError
from iris.application.request import IrisRequestExecutionOutput, IrisRequestOrchestrator
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
from iris.utils.hashing import domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_scientific_ablation_inference_projection import _fresh, _source
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration
Routing = Literal["learned", "random", "uniform"]


def _request(bundle: IrisScientificAblationInferenceBundleV1) -> IrisRequestOrchestrator:
    # This is a trusted in-memory builder result, not a candidate-authentication
    # recipe or the future production owned loader. No legacy artifact cast.
    manifest, tensors = _checked(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.source_checkpoint_sha256,
        expected_lifecycle_sha256=bundle.manifest.payload.selection.lifecycle_sha256,
    )
    independent = _fresh()
    request = build_request_orchestrator(
        backbone=independent._backbone,
        cognitive_config=manifest.payload.model.cognitive,
        workspace_config=manifest.payload.model.workspace,
        stop_token_ids=tuple(range(independent._backbone.vocabulary_size)),
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )
    request.restore_teacher_sidecar_state_v1(
        {name: tensors[name] for name in manifest.payload.model.state_names},
        expected_installed_sha256=manifest.payload.model.sha256,
    )
    return request


def _session(
    request: IrisRequestOrchestrator,
    prompt: str,
    policy: CognitiveAblationPolicyV1,
) -> CognitiveAblationSessionV1:
    return CognitiveAblationSessionV1(
        policy,
        prompt_sha256=domain_sha256("iris/cognitive-ablation-prompt/v1", prompt.encode()),
        num_modules=request._num_modules,
        num_slots=request._num_slots,
        state_dim=request._state_dim,
        cycles=request._cycles,
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_trained_earlier_best_all12_c3_teacher_private_core_parity(
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    source, controller = _source(routing, goal=goal, broadcast=broadcast)
    before = build_scientific_ablation_checkpoint_v1(controller._run)
    restored: list[IrisScientificAblationRunV1] = []

    def factory() -> IrisScientificAblationRunV1:
        run = _fresh(routing, goal=goal, broadcast=broadcast)
        restored.append(run)
        return run

    bundle = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=factory
    )
    assert bundle.manifest.payload.completed_steps == 1 and controller.state.step == 3
    request = _request(bundle)
    model = restored[0]._optimizer._model
    assert request.backbone is not model.teacher.backbone
    assert request.backbone._model is not model.teacher.backbone._model
    model.eval()
    row = _rows()[0]
    options = WorkspaceTraceOptionsV1(
        mode="full",
        example_id=row.id,
        checkpoint_id=bundle.manifest.payload.source_checkpoint_sha256,
        seed=42,
    )
    teacher_logits: list[torch.Tensor] = []
    native_forward = model.teacher.backbone.forward_embeddings

    def teacher_forward(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        output = native_forward(*args, **kwargs)
        first_target = int(torch.nonzero(kwargs["labels"][0] != -100)[0, 0])
        teacher_logits.append(output.logits[:, first_target - 1, :].detach().clone())
        return output

    with patch.object(model.teacher.backbone, "forward_embeddings", side_effect=teacher_forward):
        reference = model.evaluation_forward(row, terminal_token_id=1, diagnostics=options)
    request_logits: list[torch.Tensor] = []
    independent_forward = request.backbone.forward_embeddings

    def request_forward(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        output = independent_forward(*args, **kwargs)
        request_logits.append(output.logits[:, -1, :].detach().clone())
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    for _ in range(2):
        session = _session(request, row.prompt, bundle.manifest.payload.ablation)
        with patch.object(request.backbone, "forward_embeddings", side_effect=request_forward):
            result = request._execute_core(
                row.prompt, strategy="greedy", ablation=session, diagnostics=options
            )
        assert not isinstance(result, IrisRequestExecutionOutput) and not hasattr(result, "receipt")
        assert session._finished and not session._failed
        assert (
            len(result.generation.workspace_trace)
            == len(reference.trace.teacher.workspace_trace)
            == 3
        )
        for actual, expected in zip(
            result.generation.workspace_trace, reference.trace.teacher.workspace_trace, strict=True
        ):
            for field in fields(actual):
                assert torch.equal(getattr(actual, field.name), getattr(expected, field.name))
        actual_diagnostic = result.diagnostic_trace
        expected_diagnostic = reference.trace.teacher.diagnostic_trace
        assert actual_diagnostic is not None and expected_diagnostic is not None
        assert actual_diagnostic == expected_diagnostic
        assert (
            actual_diagnostic.intervention_plan_sha256
            == bundle.manifest.payload.ablation_policy_sha256
        )
        assert result.generation.token_ids.tolist() == [teacher_logits[0].argmax(dim=-1).tolist()]
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        assert request.sidecar_state_sha256 == bundle.manifest.payload.model.sha256
    assert len(teacher_logits) == 1 and len(request_logits) == 2
    for logits in request_logits:
        torch.testing.assert_close(logits, teacher_logits[0], rtol=1e-5, atol=1e-6)
    assert build_scientific_ablation_checkpoint_v1(controller._run) == before


@pytest.fixture(scope="module")
def private_request() -> IrisRequestOrchestrator:
    source, _ = _source()
    bundle = build_scientific_ablation_inference_v1(
        source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=_fresh
    )
    return _request(bundle)


@pytest.mark.parametrize(
    "fault",
    (
        "foreign",
        "subclass",
        "prompt",
        "geometry",
        "policy_sha",
        "hidden_session",
        "hidden_policy",
        "phase",
        "bool_phase",
        "failed",
        "finished",
        "length",
        "mixing",
    ),
)
def test_invalid_session_rejects_before_forward_or_diagnostics(
    private_request: IrisRequestOrchestrator,
    fault: str,
) -> None:
    prompt = _rows()[0].prompt
    session: Any = _session(private_request, prompt, CognitiveAblationPolicyV1())
    intervention = None
    if fault == "foreign":
        session = object()
    elif fault == "subclass":

        class Foreign(CognitiveAblationSessionV1):
            pass

        session.__class__ = Foreign
    elif fault == "prompt":
        session.prompt_sha256 = domain_sha256("iris/request-execution/v1/prompt", prompt.encode())
    elif fault == "geometry":
        session.cycles = 2
    elif fault == "policy_sha":
        session.policy_sha256 = "0" * 64
    elif fault == "hidden_session":
        vars(session)["ghost"] = True
    elif fault == "hidden_policy":
        vars(session._policy)["ghost"] = True
    elif fault == "phase":
        session._phase = 1
    elif fault == "bool_phase":
        session._phase = False
    elif fault in ("failed", "finished"):
        setattr(session, "_" + fault, True)
    elif fault == "length":
        session._length = 7
    else:
        intervention = object()
    with (
        patch.object(private_request.backbone, "forward_prompt") as forward,
        patch("iris.application.request.WorkspaceTraceRecorderV1") as diagnostic,
        pytest.raises(IrisRequestError),
    ):
        private_request._execute_core(
            prompt,
            strategy="greedy",
            ablation=session,
            intervention=intervention,  # type: ignore[arg-type]
        )
    forward.assert_not_called()
    diagnostic.assert_not_called()


def test_default_policy_private_core_is_numerically_baseline_without_baseline_receipt(
    private_request: IrisRequestOrchestrator,
) -> None:
    prompt = _rows()[0].prompt
    baseline = private_request.execute(prompt, strategy="greedy")
    session = _session(private_request, prompt, CognitiveAblationPolicyV1())
    ablated = private_request._execute_core(prompt, strategy="greedy", ablation=session)
    assert baseline.receipt is not None and not hasattr(ablated, "receipt")
    assert torch.equal(baseline.generation.token_ids, ablated.generation.token_ids)
    for actual, expected in zip(
        baseline.generation.workspace_trace, ablated.generation.workspace_trace, strict=True
    ):
        for field in fields(actual):
            assert torch.equal(getattr(actual, field.name), getattr(expected, field.name))
    with (
        patch.object(private_request.backbone, "forward_prompt") as forward,
        pytest.raises(IrisRequestError),
    ):
        private_request._execute_core(prompt, strategy="greedy", ablation=session)
    forward.assert_not_called()
