"""Real namespace/lock/quota tests; no native, durable-command or ACK evidence."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow as workflow
from iris.application import scientific_ablation_workflow_io as storage
from iris.artifacts.scientific_ablation_command_workflow_v1 import (
    COMMAND_WORKFLOW_DEFINITION_DOMAIN,
    COMMAND_WORKFLOW_NAMESPACE,
    IrisScientificAblationCommandWorkflowDefinitionV1,
    ScientificAblationCommandWorkflowDefinitionPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit._workflow_data_setup import intent, plan
from tests.unit.test_scientific_ablation_operator_export_v1 import _request


def definition(
    original: IrisScientificAblationWorkflowPlanV1, **changes: Any
) -> IrisScientificAblationCommandWorkflowDefinitionV1:
    payload = ScientificAblationCommandWorkflowDefinitionPayloadV1.model_validate(
        dict(
            operator_sha256="a" * 64,
            execution_sha256="b" * 64,
            export_request_sha256="c" * 64,
            export_result_sha256="d" * 64,
            workflow_plan_sha256=original.sha256,
        )
        | changes
    )
    return IrisScientificAblationCommandWorkflowDefinitionV1(
        sha256=domain_sha256(
            COMMAND_WORKFLOW_DEFINITION_DOMAIN,
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )


def command_store(tmp_path: Path) -> storage.ScientificAblationWorkflowStoreV1:
    original = plan(tmp_path)
    return storage.ScientificAblationWorkflowStoreV1._for_commands(
        tmp_path / "output", original, definition(original)
    )


@pytest.mark.parametrize(
    "field",
    (
        "operator_sha256",
        "execution_sha256",
        "export_request_sha256",
        "export_result_sha256",
        "workflow_plan_sha256",
    ),
)
def test_definition_binds_each_original_identity(tmp_path: Path, field: str) -> None:
    original = plan(tmp_path)
    first = definition(original)
    second = definition(original, **{field: "f" * 64})
    assert first.sha256 != second.sha256
    assert first.payload.authority == "namespace_data_only"
    assert list(tmp_path.iterdir()) == []


def test_opt_in_is_separate_read_only_constructor_and_never_reads_legacy_root(
    tmp_path: Path,
) -> None:
    original = plan(tmp_path)
    legacy = storage.ScientificAblationWorkflowStoreV1(tmp_path / "output", original)
    legacy.path.parent.mkdir(parents=True)
    legacy.path.write_bytes(b"foreign legacy file must remain untouched")
    before = legacy.path.read_bytes()
    store = command_store(tmp_path)
    assert (
        store.path == tmp_path / "output" / COMMAND_WORKFLOW_NAMESPACE / definition(original).sha256
    )
    assert not store.path.exists() and legacy.path.read_bytes() == before
    assert legacy.namespace_definition_json is None
    assert store.plan == legacy.plan
    with store.claim() as claim:
        claim.check()
    assert legacy.path.read_bytes() == before


def test_real_original_claim_initializes_exact_metadata_and_counts_all_definition_bytes(
    tmp_path: Path,
) -> None:
    store = command_store(tmp_path)
    with store.claim() as claim:
        raw = store.namespace_definition_json
        assert raw is not None and claim.namespace_definition == raw
        assert {p.name for p in store.path.iterdir()} == {
            ".iris-owner.lock",
            "plan.json",
            "definition.json",
        }
        assert (store.path / "definition.json").read_bytes() == raw
        assert storage._tree_size(store.path)[0] == 1 + len(storage._encoded(store.plan)) + len(raw)
        held = claim.with_reserved_bytes(123)
        assert held.lock is claim.lock and held.identity == claim.identity
        assert held.namespace_definition == raw and held.minimum_free_for(7) == 130
        held.reserve(0)
    assert (store.path / ".iris-owner.lock").read_bytes() == b"1"
    with pytest.raises(OSError):
        held.check()
    reopened = command_store(tmp_path)
    with reopened.claim() as second:
        second.check()


@pytest.mark.parametrize("commands", (False, True))
@pytest.mark.parametrize("member", ("commands", "intent.json", "result.json", "operator-ack.json"))
def test_no_generic_journal_or_result_member_is_silently_accepted(
    tmp_path: Path, commands: bool, member: str
) -> None:
    store = (
        command_store(tmp_path)
        if commands
        else storage.ScientificAblationWorkflowStoreV1(tmp_path / "output", plan(tmp_path))
    )
    with store.claim():
        pass
    foreign = store.path / member
    foreign.write_bytes(b"not authority")
    with pytest.raises(storage.IrisScientificAblationWorkflowStorageError), store.claim():
        raise AssertionError("Foreign layout must reject before yielding")
    assert foreign.read_bytes() == b"not authority"


def test_legacy_layout_still_rejects_namespace_definition(tmp_path: Path) -> None:
    store = storage.ScientificAblationWorkflowStoreV1(tmp_path / "output", plan(tmp_path))
    with store.claim():
        pass
    foreign = store.path / "definition.json"
    foreign.write_bytes(storage._encoded(definition(store.plan)))
    with pytest.raises(storage.IrisScientificAblationWorkflowStorageError), store.claim():
        raise AssertionError("Legacy namespace cannot adopt command DATA")
    assert foreign.exists()


def test_definition_cannot_grant_an_extra_storage_allowance(tmp_path: Path) -> None:
    raw = intent(tmp_path).model_dump(mode="python")
    raw["resources"]["maximum_storage_bytes"] = 1
    original = storage.build_scientific_ablation_workflow_plan_v1(
        ScientificAblationWorkflowIntentV1.model_validate(raw)
    )
    store = storage.ScientificAblationWorkflowStoreV1._for_commands(
        tmp_path / "output", original, definition(original)
    )
    with pytest.raises(storage.IrisScientificAblationWorkflowStorageError), store.claim():
        raise AssertionError("Plan plus definition must fit the ORIGINAL whole quota")
    assert {p.name for p in store.path.iterdir()} == {".iris-owner.lock"}
    assert store.plan.payload.resources.maximum_storage_bytes == 1


@pytest.mark.parametrize("signal", (KeyboardInterrupt(), SystemExit(17)))
def test_signal_between_metadata_writes_preserves_plan_and_completes_only_definition_on_retry(
    tmp_path: Path, signal: BaseException
) -> None:
    store = command_store(tmp_path)
    original_write = store._write_new

    def interrupted(path: Path, content: bytes, claim: storage._Claim) -> tuple[int, int]:
        if path.name == "definition.json":
            raise signal
        return original_write(path, content, claim)

    with (
        patch.object(store, "_write_new", interrupted),
        pytest.raises(type(signal)),
        store.claim(),
    ):
        raise AssertionError("Initialization never finished")
    assert {p.name for p in store.path.iterdir()} == {".iris-owner.lock", "plan.json"}
    saved = (store.path / "plan.json").read_bytes()
    with store.claim() as claim:
        claim.check()
    assert (store.path / "plan.json").read_bytes() == saved


def test_live_claim_detects_missing_definition_without_repair_or_adoption(tmp_path: Path) -> None:
    store = command_store(tmp_path)
    with pytest.raises(storage.IrisScientificAblationWorkflowStorageError), store.claim() as claim:
        (store.path / "definition.json").unlink()  # Deliberate fixture fault, not cleanup.
        claim.reserve(0)
    assert not (store.path / "definition.json").exists()


@pytest.mark.parametrize("fault", ("plan", "digest", "hidden", "private", "type"))
def test_unchecked_or_wrong_namespace_rejects_without_any_files(tmp_path: Path, fault: str) -> None:
    original = plan(tmp_path)
    value: Any = definition(original)
    if fault == "plan":
        value = definition(original, workflow_plan_sha256="f" * 64)
    elif fault == "digest":
        vars(value)["sha256"] = "f" * 64
    elif fault == "hidden":
        vars(value.payload)["native_owner"] = "not authority"
    elif fault == "private":
        object.__setattr__(value, "__pydantic_private__", {"owner": object()})
    else:
        value = value.model_dump(mode="json")
    with pytest.raises((ValueError, TypeError)):
        storage.ScientificAblationWorkflowStoreV1._for_commands(
            tmp_path / "output", original, value
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ("definition", "path", "root"))
def test_original_job_seals_namespace_before_any_claim_or_producer(
    tmp_path: Path, fault: str
) -> None:
    store = command_store(tmp_path)
    job = workflow._create_workflow_with_store(
        store, make_fresh_run=None, completed_phase_sha256=()
    )
    assert not store.path.exists()
    if fault == "definition":
        store._definition_bytes = None
    elif fault == "path":
        store.path = tmp_path / "different"
    else:
        store._root = tmp_path / "different"
    with (
        patch.object(store, "claim") as claim,
        pytest.raises((ValueError, workflow.IrisScientificAblationWorkflowError)),
    ):
        job.advance()
    assert claim.call_count == 0 and list(tmp_path.iterdir()) == []


def test_new_command_composition_preserves_invalid_plan_denial_before_source_or_factory(
    tmp_path: Path,
) -> None:
    original = plan(tmp_path)
    vars(original)["sha256"] = "f" * 64
    with (
        patch.object(operator, "read_scientific_ablation_operator_generation_source_v1") as source,
        patch.object(operator, "create_scientific_ablation_command_run_v1") as factory,
        pytest.raises(operator.IrisScientificAblationOperatorWorkflowError),
    ):
        operator._create_operator_workflow(
            tmp_path / "missing",
            request=_request(),
            export_root=tmp_path / "exports",
            plan=original,
            base=tmp_path,
            root=tmp_path / "output",
            command_namespace=True,
        )
    assert source.call_count == factory.call_count == 0 and list(tmp_path.iterdir()) == []


def test_seal_replacement_retains_explicit_namespace_bytes(tmp_path: Path) -> None:
    store = command_store(tmp_path)
    job = workflow._create_workflow_with_store(
        store, make_fresh_run=None, completed_phase_sha256=()
    )
    assert replace(job._seal, recovered=True).storage_definition == store.namespace_definition_json
