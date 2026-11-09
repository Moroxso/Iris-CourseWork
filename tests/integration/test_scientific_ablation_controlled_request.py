"""Actual trained best1/C3 controlled generation; no task-quality/CLI claim."""

import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch

from iris.application.errors import IrisRequestError
from iris.application.request import IrisRequestOrchestrator
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_controlled_receipt import (
    IrisScientificAblationControlledRequestOutputV1,
    IrisScientificAblationControlledRequestReceiptV1,
)
from iris.application.scientific_ablation_controlled_request import (
    IrisScientificAblationControlledRequestV1,
    build_scientific_ablation_controlled_request_v1,
)
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    build_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_ablation_receipt import IrisScientificAblationRequestReceiptV1
from iris.application.scientific_ablation_request import build_scientific_ablation_request_v1
from iris.application.scientific_optimization import _same_runtime
from iris.backbone.adapter import BackboneAdapter
from iris.generation import DecodingStrategy, TemperatureTopPSamplingConfig
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1
from tests.integration.test_scientific_ablation_inference_projection import Routing, _fresh, _source
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration
ROUTINGS: tuple[Routing, ...] = ("learned", "random", "uniform")
MODES: tuple[tuple[Routing, str, str], ...] = tuple(
    (routing, goal, broadcast)
    for routing in ROUTINGS
    for goal in ("enabled", "disabled")
    for broadcast in ("enabled", "disabled")
)


@dataclass(frozen=True)
class Native:
    bundle: IrisScientificAblationInferenceBundleV1
    anchors: tuple[str, str, str]
    backbone: BackboneAdapter
    current: IrisScientificAblationLifecycleV1
    current_sha256: str


@pytest.fixture(scope="module", params=MODES, ids=lambda m: "-".join(m))
def native(request: pytest.FixtureRequest) -> Iterator[Native]:
    routing, goal, broadcast = request.param
    with (
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No model download/load"),
        ) as model_load,
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer download/load"),
        ) as tokenizer_load,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate execution"),
        ) as candidate,
    ):
        source, controller = _source(routing, goal=goal, broadcast=broadcast)
        factory = partial(_fresh, routing, goal=goal, broadcast=broadcast)
        bundle = build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=factory
        )
        assert source.manifest.payload.best is not None
        assert source.manifest.payload.best.payload.completed_steps == 1
        assert source.manifest.payload.current.payload.completed_steps == 3
        assert bundle.manifest.payload.selection.kind == "selected_best"
        anchors = (
            bundle.manifest.sha256,
            bundle.manifest.payload.source_checkpoint_sha256,
            source.manifest.sha256,
        )
        before = build_scientific_ablation_checkpoint_v1(controller._run).manifest.sha256
        yield Native(bundle, anchors, factory()._backbone, controller, before)
        assert build_scientific_ablation_checkpoint_v1(controller._run).manifest.sha256 == before
        model_load.assert_not_called()
        tokenizer_load.assert_not_called()
        candidate.assert_not_called()


def load(native: Native) -> IrisScientificAblationControlledRequestV1:
    return build_scientific_ablation_controlled_request_v1(
        native.bundle,
        expected_sha256=native.anchors[0],
        expected_source_checkpoint_sha256=native.anchors[1],
        expected_lifecycle_sha256=native.anchors[2],
        backbone=native.backbone,
        stop_token_ids=native.backbone.eos_token_ids,
        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
    )


def control(kind: Literal["empty", "zero", "shuffle"]) -> WorkspaceInterventionPlanV1:
    if kind == "empty":
        operations: tuple[WorkspaceInterventionSpecV1, ...] = ()
    elif kind == "zero":
        operations = (WorkspaceInterventionSpecV1(kind="zero_workspace"),)
    else:
        operations = (WorkspaceInterventionSpecV1(kind="shuffle_slots", seed=42),)
    return WorkspaceInterventionPlanV1(interventions=operations)


def generator(prompt: str, strategy: DecodingStrategy) -> torch.Generator | None:
    seed = (
        int(domain_sha256("iris-tests/input-only-controlled-sampling/v1", prompt.encode())[:16], 16)
        % 2**63
    )
    return None if strategy == "greedy" else torch.Generator(device="cpu").manual_seed(seed)


@pytest.mark.parametrize("strategy", ("greedy", "sampling"))
@pytest.mark.parametrize("kind", ("empty", "zero", "shuffle"))
def test_actual_best_all12_c3_full_cap_controlled_generation_and_noop_parity(
    native: Native,
    strategy: DecodingStrategy,
    kind: Literal["empty", "zero", "shuffle"],
    tmp_path: Path,
) -> None:
    prompt = _rows()[0].prompt
    caller = capture_runtime_state(torch.device("cpu"))
    request = load(native)
    options = WorkspaceTraceOptionsV1(
        mode="full", example_id="prompt-only", checkpoint_id=native.anchors[1], seed=42
    )
    sampling = generator(prompt, strategy)
    with (
        patch.object(
            native.backbone, "forward_prompt", wraps=native.backbone.forward_prompt
        ) as first_pass,
        patch.object(
            native.backbone, "forward_embeddings", wraps=native.backbone.forward_embeddings
        ) as decode,
    ):
        result = request.execute(
            prompt,
            control=control(kind),
            strategy=strategy,
            generator=sampling,
            diagnostics=options,
        )
        assert first_pass.call_count == 1 and decode.call_count >= 1
    assert type(result) is IrisScientificAblationControlledRequestOutputV1
    assert type(result.receipt) is IrisScientificAblationControlledRequestReceiptV1
    assert not isinstance(result.receipt, IrisScientificAblationRequestReceiptV1)
    assert result.receipt.artifact_sha256 == native.anchors[0]
    assert result.receipt.source_checkpoint_sha256 == native.anchors[1]
    assert result.receipt.lifecycle_sha256 == native.anchors[2]
    assert result.receipt.selection == "selected_best" and result.receipt.completed_steps == 1
    assert result.receipt.stop_token_ids == native.backbone.eos_token_ids
    assert result.receipt.max_new_tokens == native.backbone.max_new_tokens == 256
    assert result.diagnostic_trace is not None
    assert result.diagnostic_trace.intervention_plan_sha256 == result.receipt.control_plan_sha256
    generated = result.generation.token_ids.shape[1]
    if result.generation.stop_reason == "max_new_tokens":
        assert generated == 256
    else:
        assert result.generation.stop_reason == "stop_token"
        assert result.generation.matched_stop_token_id in native.backbone.eos_token_ids
        assert 1 <= generated <= 256
    if kind == "zero":
        assert all(
            not bool(state.slots.any()) and not bool(state.occupancy.any())
            for state in result.generation.workspace_trace
        )
    if kind == "empty":
        baseline = build_scientific_ablation_request_v1(
            native.bundle,
            expected_sha256=native.anchors[0],
            expected_source_checkpoint_sha256=native.anchors[1],
            expected_lifecycle_sha256=native.anchors[2],
            backbone=native.backbone,
            stop_token_ids=native.backbone.eos_token_ids,
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )
        original = baseline.execute(
            prompt, strategy=strategy, generator=generator(prompt, strategy)
        )
        assert torch.equal(result.generation.token_ids, original.generation.token_ids)
        assert result.generation.text == original.generation.text
        for left, right in zip(
            result.generation.workspace_trace, original.generation.workspace_trace, strict=True
        ):
            assert all(
                torch.equal(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)
            )
    assert request._state == request._owner._state == "ready" and request._owner._pending is None
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert (
        build_scientific_ablation_checkpoint_v1(native.current._run).manifest.sha256
        == native.current_sha256
    )
    (tmp_path / "actual-controlled-generation.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-tests-actual-controlled-generation-v1",
                receipt=asdict(result.receipt),
                token_ids=result.generation.token_ids.detach().cpu().tolist(),
                text=result.generation.text,
                actual_prompt_forwards=first_pass.call_count,
                actual_decode_prefills=decode.call_count,
                generated_tokens=generated,
                stop_reason=result.generation.stop_reason,
                source_current_step=3,
                noop_parity_checked=kind == "empty",
                caller_runtime_preserved=True,
                scientific_acceptance="not_assessed",
            )
        )
    )


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_actual_forward_signal_restores_caller_rng_and_poison_owner_without_retry(
    native: Native, signal: type[BaseException]
) -> None:
    request = load(native)
    prompt = _rows()[0].prompt
    sampling = generator(prompt, "sampling")
    assert sampling is not None
    initial = sampling.get_state().clone()
    caller = capture_runtime_state(torch.device("cpu"))
    failure = signal("actual forward boundary")

    def fail(*args: object, **kwargs: object) -> object:
        random.random()
        torch.rand(4)
        raise failure

    with (
        patch.object(native.backbone, "forward_prompt", side_effect=fail),
        pytest.raises(signal) as raised,
    ):
        request.execute(prompt, control=control("zero"), strategy="sampling", generator=sampling)
    assert raised.value is failure
    assert request._state == request._owner._state == "poisoned"
    assert torch.equal(initial, sampling.get_state())
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    with pytest.raises(IrisRequestError):
        request.execute(prompt, control=control("empty"), strategy="greedy")


def test_real_controlled_owner_cannot_use_ordinary_unbound_public_or_raw_core(
    native: Native,
) -> None:
    request = load(native)
    with patch.object(
        native.backbone, "forward_prompt", side_effect=AssertionError("No forward")
    ) as forward:
        with pytest.raises(IrisRequestError):
            IrisRequestOrchestrator.execute(request._request, _rows()[0].prompt, strategy="greedy")
        with pytest.raises(IrisRequestError):
            IrisRequestOrchestrator._execute_core(
                request._request, _rows()[0].prompt, strategy="greedy"
            )
        forward.assert_not_called()
