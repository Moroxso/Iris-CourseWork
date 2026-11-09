"""Labelled controller/wiring probes, NOT native/model or real-row evidence."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_workflow as owner
from iris.application import scientific_ablation_workflow_kernel as kernel
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    ScientificAblationWorkflowDataV1,
    ScientificAblationWorkflowStoreV1,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowCaptureDataV1,
    ScientificAblationWorkflowScoreDataV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1
from tests.unit._workflow_data_setup import intent
from tests.unit.test_scientific_ablation_workflow_expected_phase import job_and_phase

PLAN = "a" * 64


def chunk(**changes: Any) -> Chunk:
    return Chunk.model_validate(
        dict(
            kernel_plan_sha256=PLAN,
            acknowledgement_head_sha256=capture._genesis(PLAN),
            acknowledged_rows=0,
            total_rows=2,
            rows_per_call=1,
            stop_row_index=1,
        )
        | changes
    )


def call(job: Any, phase: Any, value: Any, rows: Any = 1) -> Any:
    return job.advance_expected(
        expected_phase=phase,
        expected_completed_phase_sha256=(),
        rows_per_call=rows,
        expected_chunk=value,
    )


@pytest.mark.parametrize("poison", ("object", "extra", "private", "unchecked", "bool", "subclass"))
def test_chunk_payload_is_checked_before_claim_or_preflight(tmp_path: Path, poison: str) -> None:
    job, phase = job_and_phase(tmp_path)
    value: Any = chunk()
    if poison == "object":
        value = object()
    elif poison == "extra":
        value.__dict__["hidden"] = True
    elif poison == "private":
        object.__setattr__(value, "__pydantic_private__", {"hidden": True})
    elif poison == "unchecked":
        value = Chunk.model_construct(**(value.model_dump() | {"stop_row_index": 2}))
    elif poison == "bool":
        value.__dict__["acknowledged_rows"] = False
    else:

        class Other(Chunk):
            pass

        value = Other.model_validate(value.model_dump())
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        patch.object(kernel, "_check_chunk_before_native") as read,
        pytest.raises((TypeError, ValueError)),
    ):
        call(job, phase, value)
    assert claim.call_count == read.call_count == 0 and not (tmp_path / "output").exists()


@pytest.mark.parametrize("rows", (None, True, False, 0, 2, "1", 1.0))
def test_exact_chunk_budget_is_not_default_or_coerced(tmp_path: Path, rows: Any) -> None:
    job, phase = job_and_phase(tmp_path)
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(ValueError),
    ):
        call(job, phase, chunk(), rows)
    assert claim.call_count == 0 and job._state == "idle"


@pytest.mark.parametrize("index", (2, 3, 4, 5))
def test_non_kernel_chunk_is_rejected_before_claim(tmp_path: Path, index: int) -> None:
    job, _ = job_and_phase(tmp_path)
    phase = scientific_ablation_workflow_phases_v1(job._seal.store.plan)[index]
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(ValueError),
    ):
        call(job, phase, chunk())
    assert claim.call_count == 0


def test_public_snapshot_survives_caller_mutation_and_rechecks_inside_claim(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    value = chunk()
    original = ScientificAblationWorkflowStoreV1.claim
    seen: list[Chunk] = []

    @contextmanager
    def claim(store: ScientificAblationWorkflowStoreV1) -> Any:
        with original(store) as held:
            value.__dict__["acknowledgement_head_sha256"] = "b" * 64
            yield held

    def execute(*args: Any, **kwargs: Any) -> None:
        assert job._state == "busy"
        seen.append(kwargs["expected_chunk"])

    with (
        patch.object(kernel, "_check_chunk_before_native") as preflight,
        patch.object(ScientificAblationWorkflowStoreV1, "claim", new=claim),
        patch.object(steps, "execute", side_effect=execute),
    ):
        result = call(job, phase, value)
    assert preflight.call_count == 2 and len(seen) == 1
    assert seen[0] == chunk() and seen[0] is not value
    assert result.completed_phase_sha256 == () and job._state == "idle"


def test_changed_prefix_inside_outer_claim_precedes_attempt_and_factory(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    stale = ValueError("stale")
    with (
        patch.object(kernel, "_check_chunk_before_native", side_effect=[None, stale]) as preflight,
        patch.object(ScientificAblationWorkflowStoreV1, "begin") as begin,
        patch.object(steps, "execute") as execute,
        pytest.raises(IrisScientificAblationWorkflowStorageError) as caught,
    ):
        call(job, phase, chunk())
    # Preserve the original claim's error envelope, including the actual cause.
    assert caught.value.__cause__ is stale and preflight.call_count == 2
    assert begin.call_count == execute.call_count == 0 and job._state == "idle"
    assert not tuple(job._seal.store.path.glob("phase-*.attempt.json"))


@pytest.mark.parametrize("control", (False, True))
@pytest.mark.parametrize("bad", ("head", "rows", "total"))
def test_fresh_capture_checks_genesis_and_actual_row_count_without_owner(
    tmp_path: Path, control: bool, bad: str
) -> None:
    p = build_scientific_ablation_workflow_plan_v1(
        intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    )
    store = ScientificAblationWorkflowStoreV1(tmp_path / "out", p)
    phase = scientific_ablation_workflow_phases_v1(p)[2 if control else 0]
    prefix: Any = (object(), object()) if control else ()
    module = controlled_capture if control else capture
    values: dict[str, Any] = {"acknowledgement_head_sha256": module._genesis(PLAN)}
    if bad == "head":
        values["acknowledgement_head_sha256"] = "b" * 64
    elif bad == "rows":
        values.update(acknowledged_rows=1, stop_row_index=2)
    else:
        values["total_rows"] = 3
    with (
        patch.object(steps, "source"),
        patch.object(kernel, "_rows", return_value=(object(), object())),
        patch.object(steps, "_owner") as factory,
        pytest.raises(ValueError),
    ):
        kernel._check_chunk_before_native(store, phase, prefix, chunk(**values), 1)
    assert factory.call_count == 0 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("control", (False, True))
def test_fresh_capture_valid_first_chunk_is_not_excluded(tmp_path: Path, control: bool) -> None:
    p = build_scientific_ablation_workflow_plan_v1(
        intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    )
    store = ScientificAblationWorkflowStoreV1(tmp_path / "out", p)
    phase = scientific_ablation_workflow_phases_v1(p)[2 if control else 0]
    prefix: Any = (object(), object()) if control else ()
    module = controlled_capture if control else capture
    with (
        patch.object(steps, "source"),
        patch.object(kernel, "_rows", return_value=(object(), object())),
    ):
        kernel._check_chunk_before_native(
            store, phase, prefix, chunk(acknowledgement_head_sha256=module._genesis(PLAN)), 1
        )
    assert not (tmp_path / "out").exists()


def test_stale_existing_capture_is_denied_before_outer_claim(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    target = steps._kernel(job._seal.store, phase) / capture._STORE / PLAN
    target.mkdir(parents=True)
    with (
        patch.object(owner, "_prefix", return_value=()),
        patch.object(kernel, "_capture", return_value=(PLAN, "b" * 64, 1, 2, None, 0)),
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        patch.object(steps, "execute") as execute,
        pytest.raises(ValueError),
    ):
        call(job, phase, chunk())
    assert claim.call_count == execute.call_count == 0 and job._state == "idle"


@pytest.mark.parametrize("kind", ("capture", "score"))
@pytest.mark.parametrize("control", (False, True))
def test_real_step_dispatch_forwards_exact_request_to_each_original_kernel(
    tmp_path: Path, kind: str, control: bool
) -> None:
    p = build_scientific_ablation_workflow_plan_v1(
        intent(tmp_path, (WorkspaceInterventionPlanV1(interventions=()),))
    )
    store = ScientificAblationWorkflowStoreV1(tmp_path / "out", p)
    phase = next(
        ph
        for ph in scientific_ablation_workflow_phases_v1(p)
        if ph.kind == kind and (ph.stream_index is not None) is control
    )
    # Constructible labelled DATA, not a claimed native execution or real row.
    # Use the actual ordered record interface consumed before kernel dispatch.
    labelled_prefix: tuple[ScientificAblationWorkflowDataV1, ...] = ()
    for prior in scientific_ablation_workflow_phases_v1(p)[: phase.index]:
        if prior.kind == "capture":
            files = {
                "capture.json": steps.encoded(
                    ScientificAblationWorkflowCaptureDataV1(
                        protocol="ordinary" if prior.stream_index is None else "controlled",
                        plan_sha256=PLAN,
                        input_sha256="b" * 64,
                        capture_sha256="c" * 64,
                        examples=2,
                    )
                )
            }
        else:
            assert prior.kind == "score"
            files = {
                "evaluation.json": steps.encoded(
                    ScientificAblationWorkflowScoreDataV1(
                        protocol="ordinary" if prior.stream_index is None else "controlled",
                        evaluation_plan_sha256=PLAN,
                        evaluation_sha256="b" * 64,
                        complete_sha256="c" * 64,
                        capture_plan_sha256=PLAN,
                        capture_sha256="c" * 64,
                        examples=2,
                    )
                )
            }
        labelled_prefix += (
            store.make_data(
                prior,
                labelled_prefix[-1].record.sha256 if labelled_prefix else None,
                origin="kernel_data",
                files=files,
            ),
        )
    module = (
        (controlled_capture if control else capture)
        if kind == "capture"
        else (controlled_score if control else score)
    )
    builder = (
        (
            "build_scientific_ablation_controlled_capture_plan_v1"
            if control
            else "build_scientific_ablation_capture_plan_v1"
        )
        if kind == "capture"
        else (
            "build_scientific_ablation_controlled_evaluation_plan_v1"
            if control
            else "build_scientific_ablation_evaluation_plan_v2"
        )
    )
    operation = (
        (
            "capture_scientific_ablation_controlled_heldout_v1"
            if control
            else "capture_scientific_ablation_heldout_v1"
        )
        if kind == "capture"
        else (
            "evaluate_scientific_ablation_controlled_capture_resumable_v1"
            if control
            else "evaluate_scientific_ablation_capture_resumable_v2"
        )
    )
    owner_type = (
        IrisScientificAblationControlledHeldoutV1 if control else IrisScientificAblationHeldoutV1
    )
    labelled_owner = object.__new__(owner_type)
    reached = RuntimeError("Reached the original kernel boundary, no native execution")
    with ExitStack() as stack:
        stack.enter_context(patch.object(steps, "budget", return_value=1024**2))
        stack.enter_context(patch.object(steps, "_owner", return_value=labelled_owner))
        stack.enter_context(patch.object(steps, "_score_args", return_value={}))
        stack.enter_context(
            patch.object(
                owner_type,
                "manifest",
                new_callable=PropertyMock,
                return_value=SimpleNamespace(sha256="d" * 64),
            )
        )
        stack.enter_context(
            patch.object(module, builder, return_value=SimpleNamespace(sha256=PLAN))
        )
        actual = stack.enter_context(patch.object(module, operation, side_effect=reached))
        with pytest.raises(RuntimeError) as caught:
            steps.execute(store, phase, labelled_prefix, None, Mock(), 1, expected_chunk=chunk())
    assert caught.value is reached
    assert actual.call_count == 1 and actual.call_args.kwargs["expected_chunk"] == chunk()
    assert actual.call_args.kwargs["rows_per_call"] == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("same", (False, True))
def test_process_local_ready_retry_binds_original_chunk_without_rescoring(
    tmp_path: Path, same: bool
) -> None:
    job, phase = job_and_phase(tmp_path)
    value = chunk()
    store = job._seal.store
    with store.claim() as claim:
        store.begin(phase, None, claim)
    labelled_data = store.make_data(
        phase,
        None,
        origin="kernel_data",
        files={
            "capture.json": steps.encoded(
                ScientificAblationWorkflowCaptureDataV1(
                    protocol="ordinary",
                    plan_sha256=PLAN,
                    input_sha256="b" * 64,
                    capture_sha256="c" * 64,
                    examples=2,
                )
            )
        },
    )
    job._seal = replace(
        job._seal,
        ready=owner._Ready(
            steps.encoded(labelled_data.record), labelled_data.files, steps.encoded(value)
        ),
    )
    requested = value if same else chunk(acknowledgement_head_sha256="f" * 64)
    with (
        patch.object(steps, "execute") as execute,
        patch.object(steps, "validate_data"),
        patch.object(kernel, "_check_chunk_before_native") as native_read,
    ):
        if same:
            result = call(job, phase, requested)
            assert len(result.completed_phase_sha256) == 1
        else:
            with pytest.raises(owner.IrisScientificAblationWorkflowError):
                call(job, phase, requested)
            assert job._seal.anchors == ()
    assert execute.call_count == native_read.call_count == 0 and job._state == "idle"
