"""Structural source/plan denials before filesystem/native authority; no fake positive."""

from pathlib import Path
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_operator_export as export
from iris.application import scientific_ablation_operator_workflow as workflow
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
    ScientificAblationExportRequestPayloadV1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit._workflow_data_setup import plan as data_plan
from tests.unit.test_scientific_ablation_operator_export_v1 import _request


def test_current_export_is_not_generation_eligibility_before_source_reads(tmp_path: Path) -> None:
    payload = ScientificAblationExportRequestPayloadV1.model_validate(
        _request().payload.model_dump(mode="python") | {"selection": "current"}
    )
    request = IrisScientificAblationExportRequestV1(
        sha256=domain_sha256(
            export._REQUEST, canonical_json_bytes(payload.model_dump(mode="json"))
        ),
        payload=payload,
    )
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.inspect_scientific_ablation_operator_v1",
            side_effect=AssertionError("No source read"),
        ) as inspect,
        pytest.raises(workflow.IrisScientificAblationOperatorWorkflowError),
    ):
        workflow.read_scientific_ablation_operator_generation_source_v1(
            tmp_path / "missing", request=request, export_root=tmp_path / "exports"
        )
    assert inspect.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ["digest", "hidden", "mutable", "private"])
def test_unchecked_workflow_plan_rejects_before_any_source_or_factory(
    tmp_path: Path, fault: str
) -> None:
    plan = data_plan(tmp_path)
    if fault == "digest":
        vars(plan)["sha256"] = "0" * 64
    elif fault == "hidden":
        vars(plan.payload.resources)["native_callback"] = "not authority"
    elif fault == "mutable":
        vars(plan.payload)["controls"] = []
    else:
        object.__setattr__(plan, "__pydantic_private__", {"native_owner": object()})
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.read_scientific_ablation_operator_generation_source_v1",
            side_effect=AssertionError("No source read"),
        ) as source,
        patch(
            "iris.application.scientific_ablation_operator_workflow.create_scientific_ablation_command_run_v1"
        ) as factory,
        pytest.raises(workflow.IrisScientificAblationOperatorWorkflowError),
    ):
        workflow.create_scientific_ablation_operator_workflow_v1(
            tmp_path / "missing",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=plan,
            base=tmp_path,
            root=tmp_path / "workflow",
        )
    assert source.call_count == factory.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", [KeyboardInterrupt(), SystemExit(9)])
def test_generation_binding_preserves_primary_signal(fault: BaseException, tmp_path: Path) -> None:
    with (
        patch(
            "iris.application.scientific_ablation_operator_workflow.inspect_scientific_ablation_operator_v1",
            side_effect=fault,
        ),
        pytest.raises(type(fault)),
    ):
        workflow.read_scientific_ablation_operator_generation_source_v1(
            tmp_path / "operator", request=_request(), export_root=tmp_path / "exports"
        )


def test_read_only_export_never_calls_publisher_or_model_on_missing_data(tmp_path: Path) -> None:
    with (
        patch(
            "iris.application.scientific_ablation_operator_export.export_scientific_ablation_operator_v1",
            side_effect=AssertionError("Inspection is not export"),
        ) as publisher,
        patch(
            "iris.application.scientific_ablation_operator_export.create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("No native factory"),
        ) as factory,
        pytest.raises(export.IrisScientificAblationOperatorExportError),
    ):
        export.inspect_scientific_ablation_operator_export_v1(
            tmp_path / "missing", request=_request(), root=tmp_path / "exports"
        )
    assert publisher.call_count == factory.call_count == 0 and list(tmp_path.iterdir()) == []
