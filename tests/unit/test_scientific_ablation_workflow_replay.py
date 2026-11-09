"""Pure constructible request DATA/denial fences, NOT native/replay positives."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from iris.application import scientific_ablation_workflow_replay as replay
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
    ScientificAblationWorkflowCommandIntentPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from tests.unit._workflow_data_setup import plan
from tests.unit.test_scientific_ablation_command_workflow_namespace import definition
from tests.unit.test_scientific_ablation_operator_export_v1 import _request
from tests.unit.test_scientific_ablation_workflow_chunk_forwarding import chunk


def payload(tmp_path: Path) -> ScientificAblationWorkflowCommandIntentPayloadV1:
    p = plan(tmp_path)
    return ScientificAblationWorkflowCommandIntentPayloadV1(
        namespace=definition(p),
        phase=scientific_ablation_workflow_phases_v1(p)[0],
        completed_phase_sha256=(),
        expected_chunk=chunk(),
        rows_per_call=1,
    )


def test_pure_request_is_defensively_snapshotted_not_an_owner(tmp_path: Path) -> None:
    original = payload(tmp_path)
    command = replay.build_scientific_ablation_workflow_command_intent_v1(original)
    raw = command.model_dump_json()
    vars(original.namespace)["sha256"] = "f" * 64
    assert command.model_dump_json() == raw
    assert command.payload.authority == "request_data_only" and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "fault", ("prefix", "chunk_missing", "budget", "rows_bool", "authority", "extra")
)
def test_invalid_request_semantics_reject_without_IO(tmp_path: Path, fault: str) -> None:
    raw = payload(tmp_path).model_dump(mode="python")
    if fault == "prefix":
        raw["completed_phase_sha256"] = ("f" * 64,)
    elif fault == "chunk_missing":
        raw["expected_chunk"] = None
    elif fault == "budget":
        raw["rows_per_call"] = 2
    elif fault == "rows_bool":
        raw["rows_per_call"] = True
    elif fault == "authority":
        raw["authority"] = "native_owner"
    else:
        raw["completion_callback"] = "not authority"
    with pytest.raises(ValidationError):
        ScientificAblationWorkflowCommandIntentPayloadV1.model_validate(raw)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ("hash", "hidden", "private", "namespace"))
def test_unchecked_request_denied_before_source_or_filesystem(tmp_path: Path, fault: str) -> None:
    command = replay.build_scientific_ablation_workflow_command_intent_v1(payload(tmp_path))
    if fault == "hash":
        vars(command)["sha256"] = "f" * 64
    elif fault == "hidden":
        vars(command.payload)["native_owner"] = object()
    elif fault == "private":
        object.__setattr__(command, "__pydantic_private__", {"callback": object()})
    else:
        vars(command.payload.namespace.payload)["authority"] = "native_owner"
    with (
        patch.object(replay, "_bind") as bind,
        pytest.raises(replay.IrisScientificAblationWorkflowReplayError),
    ):
        replay.read_scientific_ablation_operator_workflow_command_result_v1(
            tmp_path / "operator",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=plan(tmp_path),
            base=tmp_path,
            root=tmp_path / "workflow",
            command=command,
            expected_operator_ack_sha256="a" * 64,
        )
    assert bind.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", (None, True, "", "F" * 64, "a" * 63, "a" * 64 + "\n"))
def test_independent_ack_required_before_source_reads(tmp_path: Path, bad: Any) -> None:
    command = replay.build_scientific_ablation_workflow_command_intent_v1(payload(tmp_path))
    with (
        patch.object(replay, "_bind") as bind,
        pytest.raises(replay.IrisScientificAblationWorkflowReplayError),
    ):
        replay.read_scientific_ablation_operator_workflow_command_result_v1(
            tmp_path / "operator",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=plan(tmp_path),
            base=tmp_path,
            root=tmp_path / "workflow",
            command=command,
            expected_operator_ack_sha256=bad,
        )
    assert bind.call_count == 0 and list(tmp_path.iterdir()) == []


def test_schema_checksum_is_not_accepted_merely_because_fields_parse(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        IrisScientificAblationWorkflowCommandIntentV1(sha256="f" * 64, payload=payload(tmp_path))
