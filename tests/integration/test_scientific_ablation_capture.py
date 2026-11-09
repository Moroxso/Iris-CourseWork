"""Actual durable mode-bound row commits, cold-owner resume and adversarial IO."""

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_capture as module
from iris.application.scientific_ablation_capture import (
    IrisScientificAblationCaptureError,
    IrisScientificAblationCaptureProgressV1,
    build_scientific_ablation_capture_plan_v1,
    capture_scientific_ablation_heldout_v1,
    read_scientific_ablation_capture_v1,
)
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_inference_io import _write_bytes as write_bytes
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_capture_v1 import IrisScientificAblationCapturePlanV1
from iris.artifacts.scientific_ablation_heldout_v1 import IrisScientificAblationHeldoutRowV1
from iris.data.schemas import IrisExample
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_generation_source import Routing, _prepared, _read
from tests.integration.test_scientific_ablation_heldout import _context, _plan
from tests.integration.test_scientific_generation import _test_rows

pytestmark = pytest.mark.integration
Error = IrisScientificAblationCaptureError


def _capture(
    heldout: IrisScientificAblationHeldoutV1,
    root: Path,
    plan: IrisScientificAblationCapturePlanV1,
    *,
    rows_per_call: int | None = None,
    maximum: int = 32 * 1024**2,
    reserve: int = 0,
) -> IrisScientificAblationCaptureProgressV1:
    return capture_scientific_ablation_heldout_v1(
        heldout,
        root=root,
        expected_input_sha256=plan.payload.input.sha256,
        expected_plan_sha256=plan.sha256,
        rows_per_call=rows_per_call,
        maximum_storage_bytes=maximum,
        minimum_free_bytes=reserve,
    )


def _setup(
    bound: IrisScientificAblationGenerationSourceBundleV1,
    path: Path,
    *,
    rows: tuple[IrisExample, ...] | None = None,
    sampling: bool = False,
) -> tuple[IrisScientificAblationHeldoutV1, IrisScientificAblationCapturePlanV1]:
    context = _context(bound, sampling=sampling)
    heldout = _plan(context, path, _test_rows()[-1:] if rows is None else rows, sampling=sampling)
    plan = build_scientific_ablation_capture_plan_v1(
        heldout, expected_input_sha256=heldout.manifest.sha256
    )
    return heldout, plan


@pytest.fixture(scope="module")
def bound(
    tmp_path_factory: pytest.TempPathFactory,
) -> IrisScientificAblationGenerationSourceBundleV1:
    return _read(_prepared(tmp_path_factory.mktemp("capture-source")))


@pytest.fixture(scope="module")
def complete(
    tmp_path_factory: pytest.TempPathFactory,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> IrisScientificAblationCaptureProgressV1:
    root = tmp_path_factory.mktemp("actual-capture")
    heldout, plan = _setup(bound, root / "test.parquet")
    return _capture(heldout, root / "store", plan)


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_trained_native_best_full_cap_disk_ack_and_exact_retry(
    tmp_path: Path,
    routing: Routing,
    goal: str,
    broadcast: str,
) -> None:
    source = _read(_prepared(tmp_path / "source", routing, goal=goal, broadcast=broadcast))
    context = _context(source, routing=routing, goal=goal, broadcast=broadcast)
    heldout = _plan(context, tmp_path / "test.parquet", _test_rows()[-1:])
    caller = capture_runtime_state(torch.device("cpu"))
    plan = build_scientific_ablation_capture_plan_v1(
        heldout, expected_input_sha256=heldout.manifest.sha256
    )
    result = _capture(heldout, tmp_path / "store", plan)
    assert result.complete is not None and result.next_row_index == 1
    row, ack = result.rows[0], result.acknowledgements[0]
    assert row.payload.input.payload.max_new_tokens == 256
    assert (
        len(row.payload.token_ids) > 1
        and row.payload.output_text != _test_rows()[-1].expected_response
    )
    assert json.loads(row.payload.request_receipt_json)["cycles"] == 3
    assert (
        row.payload.input.payload.selected_checkpoint_sha256
        == source.manifest.payload.best_checkpoint_sha256
    )
    assert (
        plan.payload.source.payload.selected_step == 1
        and plan.payload.source.payload.terminal_step == 3
    )
    assert ack.payload.row_sha256 == row.sha256
    with patch.object(
        IrisScientificAblationHeldoutV1,
        "execute_row",
        side_effect=AssertionError("Already acknowledged"),
    ):
        assert _capture(heldout, tmp_path / "store", plan) == result
    trusted = read_scientific_ablation_capture_v1(
        result.path,
        expected_plan_sha256=plan.sha256,
        expected_capture_sha256=result.complete.sha256,
    )
    assert trusted == result
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_actual_four_family_partial_prefix_cold_owner_resume_and_weighted_target_nll(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet", rows=_test_rows())
    partial = _capture(heldout, tmp_path / "store", plan, rows_per_call=1)
    assert partial.next_row_index == 1 and partial.complete is None
    cold, cold_plan = _setup(bound, tmp_path / "test.parquet", rows=_test_rows())
    assert cold is not heldout and cold_plan == plan
    original = IrisScientificAblationHeldoutV1.execute_row
    calls: list[int] = []

    def execute(
        owner: IrisScientificAblationHeldoutV1, index: int
    ) -> IrisScientificAblationHeldoutRowV1:
        calls.append(index)
        assert index != 0
        return original(owner, index)

    with patch.object(IrisScientificAblationHeldoutV1, "execute_row", new=execute):
        result = _capture(cold, tmp_path / "store", cold_plan)
    assert calls == [1, 2, 3] and result.rows[0] == partial.rows[0]
    assert result.complete is not None and result.next_row_index == 4
    assert len({row.payload.target_token_count for row in result.rows}) > 1
    assert len({row.payload.task_type for row in result.rows}) == 4
    assert result.complete.payload.target_token_count == sum(
        row.payload.target_token_count for row in result.rows
    )
    assert result.complete.payload.target_nll_sum == sum(
        row.payload.target_nll_sum for row in result.rows
    )


def test_actual_sampling_disk_resume_uses_input_only_seed_and_skips_acked_generation(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet", rows=_test_rows()[-2:], sampling=True)
    partial = _capture(heldout, tmp_path / "store", plan, rows_per_call=1)
    cold, cold_plan = _setup(
        bound, tmp_path / "test.parquet", rows=_test_rows()[-2:], sampling=True
    )
    assert plan == cold_plan
    original = IrisScientificAblationHeldoutV1.execute_row

    def execute(
        owner: IrisScientificAblationHeldoutV1, index: int
    ) -> IrisScientificAblationHeldoutRowV1:
        assert index == 1
        return original(owner, index)

    with patch.object(IrisScientificAblationHeldoutV1, "execute_row", new=execute):
        result = _capture(cold, tmp_path / "store", cold_plan)
    assert result.rows[0] == partial.rows[0] and result.complete is not None
    assert all(row.payload.sampling_seed is not None for row in result.rows)
    assert all(
        expectation.sampling_initial_state_sha256 is not None for expectation in plan.payload.rows
    )


def test_complete_plan_preflight_constructs_no_forward_and_restores_ambient_caller(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    heldout, _ = _setup(bound, tmp_path / "test.parquet", rows=_test_rows())
    caller = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with (
            torch.device("meta"),
            torch.inference_mode(),
            torch.no_grad(),
            patch(
                "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No forward")
            ) as forward,
        ):
            plan = build_scientific_ablation_capture_plan_v1(
                heldout, expected_input_sha256=heldout.manifest.sha256
            )
            assert len(plan.payload.rows) == 4
            assert torch.get_default_dtype() == torch.float64 and torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled() and torch.empty(()).device.type == "meta"
        forward.assert_not_called()
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize(
    "fault", ("input", "plan", "boolean_quota", "boolean_reserve", "rows_zero", "rows_boolean")
)
def test_bad_independent_anchor_or_budget_precedes_any_store_write_or_generation(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    root = tmp_path / "absent"
    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as execute,
        pytest.raises(Error),
    ):
        capture_scientific_ablation_heldout_v1(
            heldout,
            root=root,
            expected_input_sha256="0" * 64 if fault == "input" else plan.payload.input.sha256,
            expected_plan_sha256="0" * 64 if fault == "plan" else plan.sha256,
            maximum_storage_bytes=True if fault == "boolean_quota" else 32 * 1024**2,
            minimum_free_bytes=True if fault == "boolean_reserve" else 0,
            rows_per_call=0 if fault == "rows_zero" else True if fault == "rows_boolean" else None,
        )
    execute.assert_not_called()
    assert not root.exists() and not heldout._guard.context._run.poisoned


def _copy(complete: IrisScientificAblationCaptureProgressV1, root: Path) -> Path:
    path = root / complete.plan.sha256
    shutil.copytree(complete.path, path)
    return path


@pytest.mark.parametrize(
    "fault",
    (
        "unknown",
        "row_bytes",
        "row_extra",
        "row_rehashed_teacher",
        "row_rehashed_request",
        "row_rehashed_token",
        "ack_chain",
        "complete_nll",
        "missing_row",
        "duplicate_json",
        "hardlink",
    ),
)
def test_model_free_capture_read_rejects_corruption_and_semantic_rehash_without_forward(
    tmp_path: Path,
    complete: IrisScientificAblationCaptureProgressV1,
    fault: str,
) -> None:
    path = _copy(complete, tmp_path)
    row_path = path / "row-00000" / "row.json"
    if fault == "unknown":
        (path / "foreign.txt").write_text("foreign", encoding="utf-8")
    elif fault == "row_bytes":
        row_path.write_bytes(row_path.read_bytes() + b" ")
    elif fault in {
        "row_extra",
        "row_rehashed_teacher",
        "row_rehashed_request",
        "row_rehashed_token",
    }:
        raw = json.loads(row_path.read_bytes())
        if fault == "row_extra":
            raw["payload"]["untracked"] = True
        elif fault == "row_rehashed_teacher":
            raw["payload"]["teacher_invocation_sha256"] = "0" * 64
        elif fault == "row_rehashed_token":
            raw["payload"]["token_ids"][0] = (raw["payload"]["token_ids"][0] + 1) % raw["payload"][
                "input"
            ]["payload"]["vocabulary_size"]
        else:
            receipt = json.loads(raw["payload"]["request_receipt_json"])
            receipt["source_checkpoint_sha256"] = "0" * 64
            raw["payload"]["request_receipt_json"] = canonical_json_bytes(receipt).decode()
            raw["payload"]["request_receipt_sha256"] = domain_sha256(
                module._REQUEST_DOMAIN, canonical_json_bytes(receipt)
            )
        raw["sha256"] = domain_sha256(module._ROW_DOMAIN, canonical_json_bytes(raw["payload"]))
        row_path.write_bytes(canonical_json_file_bytes(raw))
    elif fault == "ack_chain":
        ack_path = path / "row-00000" / "ack.json"
        raw = json.loads(ack_path.read_bytes())
        raw["payload"]["previous_ack_sha256"] = "0" * 64
        raw["sha256"] = domain_sha256(module._ACK_DOMAIN, canonical_json_bytes(raw["payload"]))
        ack_path.write_bytes(canonical_json_file_bytes(raw))
    elif fault == "complete_nll":
        complete_path = path / "complete" / "complete.json"
        raw = json.loads(complete_path.read_bytes())
        raw["payload"]["target_nll_sum"] += 1.0
        raw["sha256"] = domain_sha256(module._COMPLETE_DOMAIN, canonical_json_bytes(raw["payload"]))
        complete_path.write_bytes(canonical_json_file_bytes(raw))
    elif fault == "missing_row":
        (path / "row-00000").rename(path / "row-00001")
    elif fault == "duplicate_json":
        content = row_path.read_bytes()
        row_path.write_bytes(
            b'{"sha256":"' + complete.rows[0].sha256.encode() + b'",' + content[1:]
        )
    elif fault == "hardlink":
        os.link(row_path, tmp_path / "hardlink.json")
    with (
        patch(
            "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(Error),
    ):
        read_scientific_ablation_capture_v1(path, expected_plan_sha256=complete.plan.sha256)
    forward.assert_not_called()


@pytest.mark.parametrize("fault", ("plan", "complete", "quota"))
def test_complete_reader_requires_independent_trust_and_explicit_budget(
    complete: IrisScientificAblationCaptureProgressV1,
    fault: str,
) -> None:
    assert complete.complete is not None
    with pytest.raises(Error):
        read_scientific_ablation_capture_v1(
            complete.path,
            expected_plan_sha256="0" * 64 if fault == "plan" else complete.plan.sha256,
            expected_capture_sha256="0" * 64 if fault == "complete" else complete.complete.sha256,
            maximum_storage_bytes=1 if fault == "quota" else 32 * 1024**2,
        )


@pytest.mark.parametrize(
    "primary",
    (
        ValueError("generation"),
        KeyboardInterrupt("generation"),
        SystemExit(71),
        BaseExceptionGroup("generation", [ValueError("other"), KeyboardInterrupt("first signal")]),
    ),
)
def test_generation_failure_records_unacknowledged_status_and_preserves_first_signal(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    primary: BaseException,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row", side_effect=primary),
        pytest.raises(Error if isinstance(primary, Exception) else type(primary)) as caught,
    ):
        _capture(heldout, tmp_path / "store", plan)
    assert (
        caught.value.__cause__ is primary
        if isinstance(primary, Exception)
        else caught.value is primary
    )
    path = tmp_path / "store" / module._STORE / plan.sha256
    progress = read_scientific_ablation_capture_v1(path, expected_plan_sha256=plan.sha256)
    assert (
        progress.next_row_index == 0
        and progress.complete is None
        and progress.uncommitted_stage_count == 1
    )
    stage = next(path.glob(".iris-capture-*.tmp"))
    failure = json.loads((stage / "failure.json").read_bytes())
    assert failure["status"] == (
        "interrupted_unacknowledged"
        if not isinstance(primary, Exception)
        else "failed_unacknowledged"
    )
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "fault",
    (
        "short_row_write",
        "fsync",
        "ack_write",
        "rename_before",
        "rename_after_signal",
        "rename_after_oserror",
        "complete_after_signal",
    ),
)
def test_storage_failure_and_lost_ack_resume_preserve_actual_committed_prefix(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    complete: IrisScientificAblationCaptureProgressV1,
    fault: str,
) -> None:
    # IO-only faults replay genuinely captured row bytes, not claimed new model forwards.
    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    assert plan == complete.plan
    original_write, original_rename = (
        write_bytes,
        rename_directory_with_windows_retry,
    )
    signal = SystemExit(72)

    def write(descriptor: int, content: bytes) -> None:
        if fault == "short_row_write" and b'"iris-scientific-ablation-heldout-row-v1"' in content:
            os.write(descriptor, content[:19])
            raise OSError("partial physical row")
        if fault == "ack_write" and b'"iris-scientific-ablation-capture-row-ack-v1"' in content:
            raise OSError("ack write")
        if fault == "fsync" and b'"iris-scientific-ablation-heldout-row-v1"' in content:
            with patch.object(os, "fsync", side_effect=OSError("row fsync")):
                original_write(descriptor, content)
            return
        original_write(descriptor, content)

    def rename(source: Path, target: Path) -> None:
        row = target.name.startswith("row-")
        if fault == "rename_before" and row:
            raise OSError("before commit")
        original_rename(source, target)
        if fault == "rename_after_oserror" and row:
            raise OSError("lost ordinary ack")
        if (fault == "rename_after_signal" and row) or (
            fault == "complete_after_signal" and target.name == "complete"
        ):
            raise signal

    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row", return_value=complete.rows[0]),
        patch.object(module, "_write_bytes", side_effect=write),
        patch.object(module, "rename_directory_with_windows_retry", side_effect=rename),
    ):
        if fault == "rename_after_oserror":
            result = _capture(heldout, tmp_path / "store", plan)
            assert result.complete is not None
        else:
            with pytest.raises(SystemExit if "signal" in fault else Error) as caught:
                _capture(heldout, tmp_path / "store", plan)
            if "signal" in fault:
                assert caught.value is signal
    path = tmp_path / "store" / module._STORE / plan.sha256
    progress = read_scientific_ablation_capture_v1(path, expected_plan_sha256=plan.sha256)
    committed = fault in {"rename_after_signal", "rename_after_oserror", "complete_after_signal"}
    assert progress.next_row_index == int(committed)
    with patch.object(
        IrisScientificAblationHeldoutV1,
        "execute_row",
        side_effect=AssertionError("Acked") if committed else None,
        return_value=complete.rows[0],
    ) as execute:
        resumed = _capture(heldout, tmp_path / "store", plan)
    assert resumed.complete is not None and resumed.rows == complete.rows
    assert execute.call_count == (0 if committed else 1)


@pytest.mark.parametrize("fault", ("foreign", "quota", "reserve", "process_lock"))
def test_store_denial_precedes_any_generation_and_preserves_existing_entries(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    fault: str,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    store = tmp_path / "store" / module._STORE
    store.mkdir(parents=True)
    foreign = store / "foreign.txt"
    if fault == "foreign":
        foreign.write_bytes(b"keep me")
    with patch.object(IrisScientificAblationHeldoutV1, "execute_row") as execute:
        if fault == "process_lock":
            with process_file_lock(store / module._LOCK), pytest.raises(Error):
                _capture(heldout, tmp_path / "store", plan)
        else:
            with (
                patch.object(
                    shutil, "disk_usage", return_value=shutil._ntuple_diskusage(100, 99, 1)
                )
                if fault == "reserve"
                else nullcontext(),
                pytest.raises(Error),
            ):
                _capture(
                    heldout,
                    tmp_path / "store",
                    plan,
                    maximum=1 if fault == "quota" else 32 * 1024**2,
                    reserve=100 if fault == "reserve" else 0,
                )
        execute.assert_not_called()
    if fault == "foreign":
        assert foreign.read_bytes() == b"keep me"


def test_same_process_reentry_is_denied_before_duplicate_forward_and_outer_commit_succeeds(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    complete: IrisScientificAblationCaptureProgressV1,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet")

    def execute(
        owner: IrisScientificAblationHeldoutV1, index: int
    ) -> IrisScientificAblationHeldoutRowV1:
        with pytest.raises(Error):
            _capture(heldout, tmp_path / "store", plan)
        return complete.rows[index]

    with patch.object(IrisScientificAblationHeldoutV1, "execute_row", new=execute):
        result = _capture(heldout, tmp_path / "store", plan)
    assert result.complete is not None and result.next_row_index == 1


def test_worst_case_row_reservation_denies_before_forward_after_valid_plan_creation(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    maximum = len(canonical_json_file_bytes(plan.model_dump(mode="json"))) + 4096
    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as execute,
        pytest.raises(Error),
    ):
        _capture(heldout, tmp_path / "store", plan, maximum=maximum)
    execute.assert_not_called()
    path = tmp_path / "store" / module._STORE / plan.sha256
    progress = read_scientific_ablation_capture_v1(path, expected_plan_sha256=plan.sha256)
    assert progress.next_row_index == 0 and progress.complete is None
    assert progress.uncommitted_stage_count == 0


@pytest.mark.parametrize("fault", ("foreign_stage", "replaced_stage"))
def test_uncertain_stage_ownership_never_publishes_or_removes_foreign_or_detached_entries(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
    complete: IrisScientificAblationCaptureProgressV1,
    fault: str,
) -> None:
    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    original = module._publish
    retained: list[Path] = []

    def publish(
        stage: module._Stage, target: Path, files: dict[str, bytes], claim: module._Claim
    ) -> None:
        if target.name.startswith("row-"):
            if fault == "foreign_stage":
                foreign = stage.path / "foreign.txt"
                foreign.write_bytes(b"keep foreign bytes")
                retained.append(foreign)
            else:
                detached = stage.path.with_name(".iris-capture-" + "f" * 32 + ".tmp")
                assert stage.path.parent == target.parent == detached.parent
                assert not detached.exists()
                stage.path.rename(detached)
                stage.path.mkdir()
                retained.append(detached / "row.json")
        original(stage, target, files, claim)

    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row", return_value=complete.rows[0]),
        patch.object(module, "_publish", side_effect=publish),
        pytest.raises(Error),
    ):
        _capture(heldout, tmp_path / "store", plan)
    assert len(retained) == 1 and retained[0].is_file()
    path = tmp_path / "store" / module._STORE / plan.sha256
    assert not (path / "row-00000").exists() and not (path / "complete").exists()
    if fault == "foreign_stage":
        assert retained[0].read_bytes() == b"keep foreign bytes"
        with pytest.raises(Error):
            read_scientific_ablation_capture_v1(path, expected_plan_sha256=plan.sha256)
    else:
        assert json.loads(retained[0].read_bytes())["sha256"] == complete.rows[0].sha256
        progress = read_scientific_ablation_capture_v1(path, expected_plan_sha256=plan.sha256)
        assert progress.next_row_index == 0 and progress.uncommitted_stage_count == 2


def test_lock_exit_secondary_signal_cannot_replace_first_capture_signal(
    tmp_path: Path,
    bound: IrisScientificAblationGenerationSourceBundleV1,
) -> None:

    original = process_file_lock
    primary, secondary = KeyboardInterrupt("first"), SystemExit(73)

    @contextmanager
    def lock(path: Path) -> Iterator[ProcessFileLock]:
        with original(path) as claim:
            yield claim
            raise secondary

    heldout, plan = _setup(bound, tmp_path / "test.parquet")
    with (
        patch.object(IrisScientificAblationHeldoutV1, "execute_row", side_effect=primary),
        patch.object(module, "process_file_lock", side_effect=lock),
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        _capture(heldout, tmp_path / "store", plan)
    assert caught.value is primary and isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert secondary in caught.value.__cause__.exceptions
