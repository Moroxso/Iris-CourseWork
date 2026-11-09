"""Labelled controller probes, NOT physical source/native/durable-ACK evidence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow as owner
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    ScientificAblationWorkflowStoreV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_chunk_forwarding import chunk
from tests.unit.test_scientific_ablation_workflow_expected_phase import job_and_phase


def labelled_binding(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    """Real workflow controller; explicitly fake operator DATA for routing only."""
    job, _ = job_and_phase(tmp_path)
    descriptor = Mock()
    descriptor.sha256 = "a" * 64
    descriptor.model_dump.return_value = {"labelled_controller_only": True, "sha256": "a" * 64}
    source = SimpleNamespace(
        operator_sha256="a" * 64, execution_sha256="b" * 64, export_result_sha256="c" * 64
    )
    original_factory = job._seal.factory
    assert original_factory is not None
    binding = operator._OperatorWorkflowBinding(
        job=job,
        operator_path=tmp_path / "operator",
        request_json=canonical_json_file_bytes(_request().model_dump(mode="json")),
        export_root=tmp_path / "exports",
        base=tmp_path,
        descriptor_json=canonical_json_file_bytes(descriptor.model_dump(mode="json")),
        execution_sha256=source.execution_sha256,
        export_result_sha256=source.export_result_sha256,
        factory=original_factory,
    )
    job._seal = replace(job._seal, operator_binding=binding)
    return job, binding, descriptor, source


def invoke(job: Any, phase: Any, method: str) -> object:
    rows = 1 if phase.kind in ("capture", "score") else None
    if method == "advance":
        return job.advance(rows_per_call=rows)
    arguments: dict[str, Any] = dict(
        expected_phase=phase,
        expected_completed_phase_sha256=job._seal.anchors,
        rows_per_call=rows,
    )
    if method == "advance_expected_with_result" and rows is not None:
        arguments["expected_chunk"] = chunk()
    return getattr(job, method)(**arguments)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("method", ("advance", "advance_expected", "advance_expected_with_result"))
def test_every_mutating_entry_and_phase_rechecks_source_before_claim(
    tmp_path: Path, index: int, method: str
) -> None:
    job, _, _, _ = labelled_binding(tmp_path)
    job._seal = replace(job._seal, anchors=tuple(f"{i:064x}" for i in range(index)))
    phase = scientific_ablation_workflow_phases_v1(job._seal.store.plan)[index]
    with (
        patch.object(
            operator, "_bind", side_effect=ValueError("Changed source/test intent")
        ) as bind,
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        patch.object(steps, "execute") as execute,
        patch.object(steps, "execute_produced") as produced,
        pytest.raises(operator.IrisScientificAblationOperatorWorkflowError),
    ):
        invoke(job, phase, method)
    assert (
        bind.call_count == 1 and claim.call_count == execute.call_count == produced.call_count == 0
    )
    assert job._state == "idle" and not (tmp_path / "output").exists()


@pytest.mark.parametrize("fault", ("job", "factory", "type", "request"))
def test_changed_private_binding_denied_before_source_reads(tmp_path: Path, fault: str) -> None:
    job, binding, _, _ = labelled_binding(tmp_path)
    if fault == "job":
        other, _ = job_and_phase(tmp_path / "other")
        changed: Any = replace(binding, job=other)
    elif fault == "factory":
        changed = replace(binding, factory=lambda: None)
    elif fault == "type":
        changed = SimpleNamespace(**{name: getattr(binding, name) for name in binding.__slots__})
    else:
        changed = replace(binding, request_json=binding.request_json + b" ")
    job._seal = replace(job._seal, operator_binding=changed)
    with (
        patch.object(operator, "_bind") as bind,
        pytest.raises(operator.IrisScientificAblationOperatorWorkflowError),
    ):
        operator._revalidate_operator_binding(job, job._seal)
    assert bind.call_count == 0 and not (tmp_path / "output").exists()


@pytest.mark.parametrize("fault", ("descriptor", "execution", "export", "operator", "plan"))
def test_preflight_compares_original_pins_not_only_current_validity(
    tmp_path: Path, fault: str
) -> None:
    job, _, descriptor, source = labelled_binding(tmp_path)
    returned_plan = job._seal.store.plan
    if fault == "descriptor":
        descriptor.model_dump.return_value = {"changed": True}
    elif fault == "plan":
        returned_plan = returned_plan.model_copy(update={"sha256": "f" * 64})
    else:
        setattr(
            source,
            {
                "execution": "execution_sha256",
                "export": "export_result_sha256",
                "operator": "operator_sha256",
            }[fault],
            "f" * 64,
        )
    with (
        patch.object(operator, "_bind", return_value=(returned_plan, source)),
        patch.object(
            operator,
            "inspect_scientific_ablation_operator_v1",
            return_value=SimpleNamespace(descriptor=descriptor),
        ),
        pytest.raises(operator.IrisScientificAblationOperatorWorkflowError),
    ):
        operator._revalidate_operator_binding(job, job._seal)
    assert not (tmp_path / "output").exists()


def test_full_binding_repeated_inside_original_claim_before_attempt(tmp_path: Path) -> None:
    job, _, descriptor, source = labelled_binding(tmp_path)
    with (
        patch.object(
            operator,
            "_bind",
            side_effect=[(job._seal.store.plan, source), ValueError("Changed after outer check")],
        ) as bind,
        patch.object(
            operator,
            "inspect_scientific_ablation_operator_v1",
            return_value=SimpleNamespace(descriptor=descriptor),
        ),
        patch.object(ScientificAblationWorkflowStoreV1, "begin") as begin,
        patch.object(steps, "execute") as execute,
        pytest.raises(IrisScientificAblationWorkflowStorageError),
    ):
        job.advance()
    assert bind.call_count == 2 and begin.call_count == execute.call_count == 0
    assert {path.name for path in job._seal.path.iterdir()} == {"plan.json", ".iris-owner.lock"}
    assert job._state == "idle"


@pytest.mark.parametrize("fault", (KeyboardInterrupt(), SystemExit(9)))
def test_preflight_preserves_the_exact_primary_signal(tmp_path: Path, fault: BaseException) -> None:
    job, _, _, _ = labelled_binding(tmp_path)
    with patch.object(operator, "_bind", side_effect=fault), pytest.raises(type(fault)) as raised:
        job.advance()
    assert raised.value is fault and job._state == "idle"
    assert not (tmp_path / "output").exists()


def test_valid_labelled_controller_binding_is_read_only(tmp_path: Path) -> None:
    job, _, descriptor, source = labelled_binding(tmp_path)
    with (
        patch.object(operator, "_bind", return_value=(job._seal.store.plan, source)) as bind,
        patch.object(
            operator,
            "inspect_scientific_ablation_operator_v1",
            return_value=SimpleNamespace(descriptor=descriptor),
        ),
        patch.object(operator, "create_scientific_ablation_command_run_v1") as factory,
    ):
        operator._revalidate_operator_binding(job, job._seal)
    assert bind.call_count == 1 and factory.call_count == 0
    assert not (tmp_path / "output").exists()


def test_storage_only_ready_retry_still_revalidates_source(tmp_path: Path) -> None:
    job, _, _, _ = labelled_binding(tmp_path)
    # Deliberately unreadable fake ready bytes: preflight must run BEFORE use.
    job._seal = replace(job._seal, ready=owner._Ready(b"not physical DATA", ()))
    with (
        patch.object(operator, "_bind", side_effect=ValueError("Changed source")) as bind,
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(operator.IrisScientificAblationOperatorWorkflowError),
    ):
        job.advance()
    assert bind.call_count == 1 and claim.call_count == 0
    assert not (tmp_path / "output").exists()


def test_original_operator_creation_attaches_binding_and_factory_rechecks_it(
    tmp_path: Path,
) -> None:
    prototype, _, descriptor, source = labelled_binding(tmp_path)
    descriptor.payload.maximum_trainable_parameter_bytes = 1024**2
    source.export_path = tmp_path / "exports/receipt"
    source.lifecycle_path = tmp_path / "operator/lifecycle"
    source.inference_path = tmp_path / "operator/inference"
    with (
        patch.object(operator, "_bind", return_value=(prototype._seal.store.plan, source)),
        patch.object(
            operator,
            "inspect_scientific_ablation_operator_v1",
            return_value=SimpleNamespace(path=tmp_path / "operator/job", descriptor=descriptor),
        ),
        patch.object(operator, "_recipe", return_value=object()),
        patch.object(operator, "create_scientific_ablation_command_run_v1") as factory,
    ):
        job = operator.create_scientific_ablation_operator_workflow_v1(
            tmp_path / "operator/job",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=prototype._seal.store.plan,
            base=tmp_path,
            root=tmp_path / "output",
        )
    binding = job._seal.operator_binding
    assert type(binding) is operator._OperatorWorkflowBinding and binding.job is job
    assert binding.factory is job._seal.factory and factory.call_count == 0
    with (
        patch.object(operator, "_bind", side_effect=ValueError("Changed before native")) as bind,
        patch.object(operator, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(operator.IrisScientificAblationOperatorWorkflowError),
    ):
        binding.factory()
    assert bind.call_count == 1 and factory.call_count == 0
    assert not (tmp_path / "output").exists()
