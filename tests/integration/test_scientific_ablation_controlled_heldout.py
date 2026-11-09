"""Actual controlled held-out Parquet/full cap; separate unmodified-mode NLL."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch

from iris.ablations.controls import (
    CognitiveAblationControlPlanV1,
    validate_ablation_control_plan_v1,
)
from iris.application import scientific_ablation_controlled_heldout as module
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_controlled_generation_context import (
    IrisScientificAblationControlledGenerationContextError,
    IrisScientificAblationControlledGenerationContextV1,
    build_scientific_ablation_controlled_generation_context_v1,
)
from iris.application.scientific_ablation_controlled_generation_context import (
    _restore_caller as restore_caller,
)
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutError,
    IrisScientificAblationControlledHeldoutV1,
    build_scientific_ablation_controlled_heldout_v1,
)
from iris.application.scientific_ablation_controlled_receipt import (
    IrisScientificAblationControlledRequestOutputV1,
)
from iris.application.scientific_ablation_controlled_request import (
    IrisScientificAblationControlledRequestV1,
)
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
)
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherOutputV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.generation import DecodingStrategy, TemperatureTopPSamplingConfig
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_controlled_generation_context import _fresh
from tests.integration.test_scientific_ablation_controlled_request import (
    MODES,
)
from tests.integration.test_scientific_ablation_controlled_request import (
    control as workspace_control,
)
from tests.integration.test_scientific_ablation_generation_source import Routing, _prepared, _read
from tests.integration.test_scientific_generation import _test_rows, _test_spec
from tests.integration.test_scientific_lifecycle import _validation

pytestmark = pytest.mark.integration
Error = IrisScientificAblationControlledHeldoutError


@pytest.fixture(scope="module", autouse=True)
def no_loaders_or_candidates_or_ordinary_owner() -> Iterator[None]:
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


def _decoding(*, sampling: bool = False) -> ScientificDecodingPolicyV1:
    return ScientificDecodingPolicyV1(
        strategy="sampling" if sampling else "greedy",
        seed=1337,
        temperature=0.7 if sampling else 1.0,
        top_p=0.9 if sampling else 1.0,
    )


def _context(
    bundle: IrisScientificAblationGenerationSourceBundleV1,
    *,
    routing: Routing = "learned",
    goal: str = "enabled",
    broadcast: str = "enabled",
    sampling: bool = False,
) -> IrisScientificAblationControlledGenerationContextV1:
    decoding = _decoding(sampling=sampling)
    return build_scientific_ablation_controlled_generation_context_v1(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_lifecycle_sha256=bundle.lifecycle.manifest.sha256,
        expected_inference_sha256=bundle.inference.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.best_checkpoint_sha256,
        make_fresh_run=lambda: _fresh(routing, goal=goal, broadcast=broadcast),
        validation=_validation(),
        sampling=TemperatureTopPSamplingConfig(decoding.temperature, decoding.top_p),
    )


def _plan(
    context: IrisScientificAblationControlledGenerationContextV1,
    path: Path,
    rows: tuple[IrisExample, ...],
    *,
    sampling: bool = False,
    control: WorkspaceInterventionPlanV1 | None = None,
) -> IrisScientificAblationControlledHeldoutV1:
    return build_scientific_ablation_controlled_heldout_v1(
        context,
        expected_reconstruction_sha256=context.reconstruction.sha256,
        test=_test_spec(path, rows),
        base=path.parent,
        decoding=_decoding(sampling=sampling),
        control=workspace_control("empty") if control is None else control,
    )


@pytest.fixture(scope="module")
def bound(
    tmp_path_factory: pytest.TempPathFactory,
) -> IrisScientificAblationGenerationSourceBundleV1:
    return _read(_prepared(tmp_path_factory.mktemp("heldout-source")))


@dataclass(frozen=True)
class ModeSource:
    bundle: IrisScientificAblationGenerationSourceBundleV1
    routing: Routing
    goal: str
    broadcast: str
    native_path: Path
    inference_path: Path


@pytest.fixture(scope="module", params=MODES, ids=lambda mode: "-".join(mode))
def mode_source(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ModeSource:
    routing, goal, broadcast = request.param
    prepared = _prepared(
        tmp_path_factory.mktemp("controlled-heldout-source"),
        routing,
        goal=goal,
        broadcast=broadcast,
    )
    return ModeSource(
        _read(prepared), routing, goal, broadcast, prepared.native_path, prepared.inference_path
    )


@pytest.mark.parametrize("kind", ("empty", "zero", "shuffle"))
@pytest.mark.parametrize("strategy", ("greedy", "sampling"))
def test_actual_all12_c3_full_cap_prompt_only_generation_and_independent_best_target_nll(
    tmp_path: Path,
    mode_source: ModeSource,
    kind: Literal["empty", "zero", "shuffle"],
    strategy: DecodingStrategy,
) -> None:
    bundle = mode_source.bundle
    sampled = strategy == "sampling"
    context = _context(
        bundle,
        routing=mode_source.routing,
        goal=mode_source.goal,
        broadcast=mode_source.broadcast,
        sampling=sampled,
    )
    row = _test_rows()[-1]
    before = build_scientific_ablation_checkpoint_v1(context._run)
    caller = capture_runtime_state(torch.device("cpu"))
    plan = _plan(
        context,
        tmp_path / "test.parquet",
        (row,),
        sampling=sampled,
        control=workspace_control(kind),
    )
    result = plan.execute_row(0)
    p = result.payload
    assert p.input == plan.manifest
    assert p.input.payload.selected_checkpoint_sha256 == before.manifest.sha256
    assert p.input.payload.max_new_tokens == 256 and p.input.payload.eos_token_ids == (1,)
    assert p.input.payload.ablation_policy_sha256 == bundle.manifest.payload.ablation_policy_sha256
    assert (
        p.input.payload.target_nll_semantics
        == "selected_best_cognitive_mode_without_workspace_control"
    )
    assert p.example_id == row.id and p.row_index == 0
    assert (p.sampling_seed is not None) is sampled
    assert p.output_text != row.expected_response and len(p.token_ids) > 1
    assert len(p.token_ids) == 256 if p.stop_reason == "max_new_tokens" else p.token_ids[-1] == 1
    assert result.sha256 == domain_sha256(
        module._ROW_DOMAIN, canonical_json_bytes(p.model_dump(mode="json"))
    )
    raw_receipt = json.loads(p.request_receipt_json)
    assert raw_receipt["generated_token_ids_sha256"] == tensor_sha256(
        torch.tensor([p.token_ids], dtype=torch.int64),
        domain=module._CORE_DOMAIN + "/generated-ids",
    )
    assert raw_receipt["cycles"] == 3 and raw_receipt["selection"] == "selected_best"
    assert raw_receipt["completed_steps"] == 1
    assert raw_receipt["control_plan_sha256"] == p.input.payload.generation_control_plan_sha256
    assert raw_receipt["control_plan_json"] == p.input.payload.generation_control_plan_json
    assert p.request_receipt_sha256 == domain_sha256(
        module._REQUEST_DOMAIN, p.request_receipt_json.encode()
    )
    model = context._run._optimizer._model
    model.eval()
    try:
        independent = model.evaluation_forward(row, terminal_token_id=1)
        assert p.target_nll_sum == float(independent.losses.language_model)
        assert p.target_token_count == independent.supervised_token_count
        assert p.teacher_invocation_sha256 == independent.invocation_sha256
        assert p.scientific_state_sha256 == independent.scientific_state_sha256
    finally:
        model.train(True)
    assert build_scientific_ablation_checkpoint_v1(context._run) == before
    assert context.reconstruction.payload.selected_step == 1
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert row.prompt not in repr(result) and row.expected_response not in repr(result)
    witness = dict(
        row=result.model_dump(mode="json"),
        source=bundle.manifest.model_dump(mode="json"),
        reconstruction=context.reconstruction.model_dump(mode="json"),
        source_lifecycle_path=str(mode_source.native_path),
        source_inference_path=str(mode_source.inference_path),
        test_parquet_path=str(tmp_path / "test.parquet"),
        scientific_acceptance="not_assessed",
        durable_controlled_capture=False,
    )
    with (tmp_path / "actual-controlled-heldout-row.json").open("xb") as stream:
        stream.write(canonical_json_bytes(witness) + b"\n")


@pytest.mark.parametrize("kind", ("empty", "zero", "shuffle"))
def test_full_four_family_parquet_preflight_and_varied_actual_target_counts(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    kind: Literal["empty", "zero", "shuffle"],
) -> None:
    context = _context(bound)
    rows = _test_rows()
    before = build_scientific_ablation_checkpoint_v1(context._run)
    with patch(
        "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No forward")
    ) as forward:
        plan = _plan(context, tmp_path / "test.parquet", rows, control=workspace_control(kind))
    forward.assert_not_called()
    assert plan.manifest.payload.example_count == 4
    assert len(set(plan._guard.target_counts)) > 1
    actual = [plan.execute_row(index).payload for index in range(4)]
    assert {row.task_type for row in actual} == {row.task_type for row in rows}
    assert sum(row.target_token_count for row in actual) == plan.manifest.payload.target_token_count
    assert build_scientific_ablation_checkpoint_v1(context._run) == before


def test_sampling_is_paired_input_only_reproducible_and_independent_of_order_and_reference(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound, sampling=True)
    rows = _test_rows()[-2:]
    plan = _plan(context, tmp_path / "test.parquet", rows, sampling=True)
    first = tuple(plan.execute_row(i).payload for i in (0, 1))
    again = plan.execute_row(0).payload
    assert again == first[0]
    reverse = _plan(context, tmp_path / "reverse.parquet", rows[::-1], sampling=True)
    for index, original in enumerate(first[::-1]):
        reordered = reverse.execute_row(index).payload
        assert (
            reordered.token_ids == original.token_ids
            and reordered.output_text == original.output_text
        )
        assert reordered.sampling_seed == original.sampling_seed
    raw = rows[0].model_dump(mode="python")
    reference = json.loads(raw["expected_response"])
    reference["content"] = "changed reference only"
    raw["expected_response"] = json.dumps(reference)
    updated = IrisExample.model_validate(raw)
    changed = (
        _plan(context, tmp_path / "changed.parquet", (updated, rows[1]), sampling=True)
        .execute_row(0)
        .payload
    )
    assert changed.sampling_seed == first[0].sampling_seed
    assert changed.token_ids == first[0].token_ids and changed.output_text == first[0].output_text
    assert changed.target_nll_sum != first[0].target_nll_sum
    assert changed.example_sha256 != first[0].example_sha256


@pytest.mark.parametrize(
    "fault",
    (
        "train",
        "validation",
        "train_overlap",
        "validation_overlap",
        "duplicate",
        "long_prompt",
        "long_target",
    ),
)
def test_actual_bad_late_parquet_row_denied_before_any_heldout_forward(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    context = _context(bound)
    rows = _test_rows()[-2:]
    raw = rows[-1].model_dump(mode="python")
    if fault in ("train", "validation"):
        raw["split"] = fault
        raw["id"] = f"{fault}-{raw['seed']}"
    elif fault in ("train_overlap", "validation_overlap"):
        raw["seed"] = 0 if fault == "train_overlap" else 1000000
        raw["id"] = f"test-{raw['seed']}"
    elif fault == "long_prompt":
        raw["prompt"] = "irrelevant " * 769
    elif fault == "long_target":
        reference = json.loads(raw["expected_response"])
        reference["content"] = "irrelevant " * 257
        raw["expected_response"] = json.dumps(reference)
    bad = rows[0] if fault == "duplicate" else IrisExample.model_validate(raw)
    path = tmp_path / "test.parquet"
    if fault == "duplicate":
        # The normal writer correctly refuses duplicates. Emit an adversarial
        # physical fixture with its actual framed logical digest; restore the
        # reader's duplicate guard BEFORE invoking the production builder.
        digest = sha256(b"iris/synthetic-01/logical-rows/v1\0")
        for row in (rows[0], bad):
            content = canonical_json_bytes(row.model_dump(mode="json"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        with patch("iris.data.parquet._rows_digest", return_value=digest.hexdigest()):
            spec = _test_spec(path, (rows[0], bad))
    else:
        spec = _test_spec(path, (rows[0], bad))
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(Error),
    ):
        build_scientific_ablation_controlled_heldout_v1(
            context,
            expected_reconstruction_sha256=context.reconstruction.sha256,
            test=spec,
            base=tmp_path,
            decoding=_decoding(),
            control=workspace_control("empty"),
        )
    forward.assert_not_called()
    assert context.reconstruction.payload.selected_step == 1


@pytest.mark.parametrize("fault", ("row_extra", "case_extra", "legacy", "scientific_subset"))
def test_complete_owned_fields_and_scientific_distribution_are_not_laundered(fault: str) -> None:
    rows = _test_rows()
    row = rows[0].model_copy(deep=True)
    if fault == "row_extra":
        vars(row)["untracked"] = True
    elif fault == "case_extra":
        vars(row.code_cases[0])["untracked"] = True
    elif fault == "legacy":
        vars(row)["template_id"] = "iris-synthetic-01-v1"
    with pytest.raises(ValueError):
        module._own_rows(
            (row, *rows[1:]), excluded_seeds=set(), scientific=fault == "scientific_subset"
        )


@pytest.mark.parametrize(
    "fault", ("reconstruction", "file_sha", "spec_extra", "decoding_extra", "decoding_policy")
)
def test_independent_trust_and_exact_preflight_policy_deny_before_forward(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    context = _context(bound)
    spec = _test_spec(tmp_path / "test.parquet", _test_rows()[-1:])
    decoding = _decoding(sampling=fault == "decoding_policy")
    if fault == "file_sha":
        vars(spec)["sha256"] = "0" * 64
    elif fault == "spec_extra":
        vars(spec)["untracked"] = True
    elif fault == "decoding_extra":
        vars(decoding)["untracked"] = True
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(Error),
    ):
        build_scientific_ablation_controlled_heldout_v1(
            context,
            expected_reconstruction_sha256="0" * 64
            if fault == "reconstruction"
            else context.reconstruction.sha256,
            test=spec,
            base=tmp_path,
            decoding=decoding,
            control=workspace_control("empty"),
        )
    forward.assert_not_called()


@pytest.mark.parametrize("index", (-1, 1, True))
def test_bad_index_cannot_execute_or_poison_valid_source(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    index: int,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    with (
        patch.object(IrisScientificAblationControlledRequestV1, "execute") as execute,
        pytest.raises(Error),
    ):
        plan.execute_row(index)
    execute.assert_not_called()
    assert plan.manifest.payload.example_count == 1 and not context._run.poisoned


def test_changed_original_parquet_is_denied_without_computation(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    path = tmp_path / "test.parquet"
    plan = _plan(context, path, _test_rows()[-1:])
    _test_spec(path, _test_rows()[-2:])
    with (
        patch.object(IrisScientificAblationControlledRequestV1, "execute") as execute,
        pytest.raises(Error),
    ):
        plan.execute_row(0)
    execute.assert_not_called()
    assert not context._run.poisoned and plan._state == "poisoned"


@pytest.mark.parametrize(
    "primary",
    (
        ValueError("generate"),
        KeyboardInterrupt("generate"),
        SystemExit(61),
        BaseExceptionGroup("generate", [ValueError("other"), KeyboardInterrupt("signal")]),
    ),
)
def test_primary_generation_failure_withholds_row_and_restores_caller_and_modes(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    primary: BaseException,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    caller = capture_runtime_state(torch.device("cpu"))
    expected = Error if isinstance(primary, Exception) else type(primary)
    with (
        patch.object(IrisScientificAblationControlledRequestV1, "execute", side_effect=primary),
        pytest.raises(expected) as caught,
    ):
        plan.execute_row(0)
    assert (
        (caught.value.__cause__ is primary)
        if isinstance(primary, Exception)
        else (caught.value is primary)
    )
    assert context._run.poisoned and context._request._state == "poisoned"
    assert all(model.training for model in context._run._optimizer._model.modules())
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "fault",
    (
        "teacher_count",
        "teacher_invocation",
        "teacher_state",
        "nan_nll",
        "request_anchor",
        "generated_tensor",
    ),
)
def test_inconsistent_actual_request_or_teacher_facts_never_become_row_evidence(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    model = context._run._optimizer._model
    original_teacher = model.evaluation_forward
    original_request = context._request.execute

    def teacher(
        row: IrisExample, *, terminal_token_id: int
    ) -> IrisScientificAblationTeacherOutputV1:
        output = original_teacher(row, terminal_token_id=terminal_token_id)
        if fault == "teacher_count":
            return replace(output, supervised_token_count=True)
        if fault == "teacher_invocation":
            return replace(output, invocation_sha256="0" * 64)
        if fault == "teacher_state":
            return replace(output, scientific_state_sha256="0" * 64)
        if fault == "nan_nll":
            return replace(
                output, losses=replace(output.losses, language_model=torch.tensor(float("nan")))
            )
        return output

    def request(
        self: IrisScientificAblationControlledRequestV1,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        control: WorkspaceInterventionPlanV1,
    ) -> IrisScientificAblationControlledRequestOutputV1:
        output = original_request(prompt, strategy=strategy, generator=generator, control=control)
        if fault == "request_anchor":
            return replace(
                output, receipt=replace(output.receipt, source_checkpoint_sha256="0" * 64)
            )
        if fault == "generated_tensor":
            tokens = output.generation.token_ids.clone()
            tokens[0, 0] = (tokens[0, 0] + 1) % context._run._backbone.vocabulary_size
            return replace(output, generation=replace(output.generation, token_ids=tokens))
        return output

    with (
        patch.object(model, "evaluation_forward", side_effect=teacher),
        patch.object(IrisScientificAblationControlledRequestV1, "execute", new=request),
        pytest.raises(Error),
    ):
        plan.execute_row(0)
    assert context._run.poisoned and plan._state == "poisoned"


def test_reentry_denial_is_retryable_inside_one_actual_request(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    original = context._request.execute

    def request(
        self: IrisScientificAblationControlledRequestV1,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        control: WorkspaceInterventionPlanV1,
    ) -> IrisScientificAblationControlledRequestOutputV1:
        with pytest.raises(Error, match="already busy"):
            plan.execute_row(0)
        return original(prompt, strategy=strategy, generator=generator, control=control)

    with patch.object(IrisScientificAblationControlledRequestV1, "execute", new=request):
        assert plan.execute_row(0).payload.example_id == _test_rows()[-1].id
    assert plan._state == "ready" and not context._run.poisoned


def test_caller_restoration_signal_withholds_otherwise_successful_actual_row(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    caller = capture_runtime_state(torch.device("cpu"))
    actual_restore = restore_caller
    signal = SystemExit(67)

    def restore(snapshot: RuntimeStateSnapshot, dtype: torch.dtype, device: torch.device) -> None:
        actual_restore(snapshot, dtype, device)
        raise signal

    with (
        patch.object(module, "_restore_caller", side_effect=restore),
        pytest.raises(SystemExit) as caught,
    ):
        plan.execute_row(0)
    assert caught.value is signal and context._run.poisoned
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_scientific_population_owns_complete_actual5000_source_family_rows() -> None:
    rows = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=5000),
            template_id="iris-synthetic-01-v2",
        )
    )
    owned, digest = module._own_rows(rows, excluded_seeds={0, 1000000}, scientific=True)
    assert len(owned) == 5000 and len(digest) == 64
    assert owned[0] == rows[0] and owned[0] is not rows[0]
    assert [
        sum(row.task_type == family for row in owned)
        for family in (
            "split_spec_repair",
            "delayed_goal",
            "conflict_resolution",
            "next_tool_prediction",
        )
    ] == [2000, 1000, 1000, 1000]


def test_heldout_preflight_and_execution_restore_ambient_meta_double_inference_context(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    caller = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.inference_mode(), torch.no_grad():
            plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
            assert plan.execute_row(0).payload.target_token_count >= 2
            assert torch.get_default_dtype() == torch.float64
            assert torch.empty(()).device.type == "meta"
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)
        restore_runtime_state(caller)


def test_other_plan_and_context_metadata_reentry_cannot_poison_active_valid_execution(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    other = _plan(context, tmp_path / "other.parquet", _test_rows()[-1:])
    original = context._request.execute

    def request(
        self: IrisScientificAblationControlledRequestV1,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        control: WorkspaceInterventionPlanV1,
    ) -> IrisScientificAblationControlledRequestOutputV1:
        with pytest.raises(Error, match="already busy"):
            other.execute_row(0)
        with pytest.raises(IrisScientificAblationControlledGenerationContextError):
            _ = context.reconstruction
        return original(prompt, strategy=strategy, generator=generator, control=control)

    with patch.object(IrisScientificAblationControlledRequestV1, "execute", new=request):
        assert plan.execute_row(0).payload.target_token_count >= 2
    assert other.manifest == plan.manifest
    assert context.reconstruction.payload.selected_step == 1 and not context._run.poisoned


@pytest.mark.parametrize(
    "fault",
    (
        "parent_extra",
        "nested_extra",
        "raw_kind",
        "slot",
        "cycle",
        "module",
        "capacity",
        "width",
        "winners",
    ),
)
def test_full_control_tree_and_actual_geometry_deny_before_forward(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    context = _context(bound)
    operation = WorkspaceInterventionSpecV1(kind="zero_workspace")
    if fault == "slot":
        operation = WorkspaceInterventionSpecV1(kind="zero_slot", slot=4)
    elif fault == "cycle":
        operation = WorkspaceInterventionSpecV1(kind="zero_workspace", cycles=(4,))
    elif fault == "module":
        operation = WorkspaceInterventionSpecV1(kind="block_module_read", module=4)
    elif fault == "capacity":
        operation = WorkspaceInterventionSpecV1(kind="reduce_workspace_capacity", capacity=5)
    elif fault == "width":
        operation = WorkspaceInterventionSpecV1(
            kind="replace_slot", slot=0, values=(0.0,), source_module=0, confidence=1.0, ttl=1
        )
    operations: tuple[WorkspaceInterventionSpecV1, ...] = (operation,)
    if fault == "winners":
        operations = tuple(
            WorkspaceInterventionSpecV1(kind="force_module_winner", module=index)
            for index in range(3)
        )
    candidate = WorkspaceInterventionPlanV1(interventions=operations)
    if fault == "parent_extra":
        vars(candidate)["untracked"] = True
    elif fault == "nested_extra":
        vars(candidate.interventions[0])["untracked"] = True
    elif fault == "raw_kind":
        vars(candidate.interventions[0])["kind"] = "not-a-control"
    before = build_scientific_ablation_checkpoint_v1(context._run)
    with patch("transformers.Qwen3ForCausalLM.forward") as forward, pytest.raises(Error):
        _plan(context, tmp_path / "test.parquet", _test_rows()[-1:], control=candidate)
    forward.assert_not_called()
    assert build_scientific_ablation_checkpoint_v1(context._run) == before
    assert context.reconstruction.payload.selected_step == 1


def test_full_receipt_byte_quota_denies_before_parquet_or_model_work(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    candidate = workspace_control("empty")
    combined = validate_ablation_control_plan_v1(
        CognitiveAblationControlPlanV1.model_construct(
            ablation=context._source.manifest.payload.ablation, control=candidate
        )
    )
    control_length = len(canonical_json_bytes(combined.model_dump(mode="json")))
    spec = _test_spec(tmp_path / "test.parquet", _test_rows()[-1:])
    with (
        patch.object(module, "_MAX_CONTROL_JSON", control_length + 1),
        patch.object(module, "_rows") as rows,
        patch("transformers.Qwen3ForCausalLM.forward") as forward,
        pytest.raises(Error),
    ):
        build_scientific_ablation_controlled_heldout_v1(
            context,
            expected_reconstruction_sha256=context.reconstruction.sha256,
            test=spec,
            base=tmp_path,
            decoding=_decoding(),
            control=candidate,
        )
    rows.assert_not_called()
    forward.assert_not_called()
    assert context.reconstruction.payload.selected_step == 1


def test_owned_control_alias_cannot_change_actual_input_or_sampling(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound, sampling=True)
    original = workspace_control("shuffle")
    rows = _test_rows()[-1:]
    plan = _plan(context, tmp_path / "test.parquet", rows, sampling=True, control=original)
    before = plan.manifest
    vars(original.interventions[0])["seed"] = 999
    vars(original)["untracked"] = True
    actual = plan.execute_row(0)
    reference = _plan(
        context,
        tmp_path / "reference.parquet",
        rows,
        sampling=True,
        control=workspace_control("shuffle"),
    ).execute_row(0)
    assert actual == reference
    assert plan.manifest == before
    assert (
        json.loads(actual.payload.request_receipt_json)["control_plan_json"]
        == before.payload.generation_control_plan_json
    )


def test_other_valid_control_output_is_rejected_before_target_nll(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    context = _context(bound)
    plan = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    original = context._request.execute

    def wrong_control(
        self: IrisScientificAblationControlledRequestV1,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        control: WorkspaceInterventionPlanV1,
    ) -> IrisScientificAblationControlledRequestOutputV1:
        return original(
            prompt, strategy=strategy, generator=generator, control=workspace_control("zero")
        )

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisScientificAblationControlledRequestV1, "execute", new=wrong_control),
        patch.object(module, "_target_nll") as target,
        pytest.raises(Error),
    ):
        plan.execute_row(0)
    target.assert_not_called()
    assert plan._state == context._request._state == "poisoned"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
