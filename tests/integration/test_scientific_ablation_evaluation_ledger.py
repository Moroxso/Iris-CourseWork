"""Actual owned capture/scorer ledger tests; no candidate code executes on the host."""

import json
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_evaluation_ledger as ledger
from iris.application import scientific_ablation_evaluation_v2 as scoring
from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_ablation_evaluation_v2 import _score_row
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_inference_io import _finish_descriptor
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationPlanV2,
    IrisScientificAblationEvaluationRowV2,
    ScientificAblationEvaluationPlanPayloadV2,
)
from iris.integrations.python_sandbox import _worker
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_evaluation_v2 import (
    Case,
    _copy,
    _evaluate,
    _make,
)
from tests.integration.test_scientific_ablation_generation_source import Routing
from tests.integration.test_scientific_generation import _test_rows
from tests.unit.test_synthetic_scoring_v2 import _POLICY

pytestmark = pytest.mark.integration
Error = scoring.IrisScientificAblationEvaluationError


def _arguments(case: Case) -> dict[str, Any]:
    assert case.capture.complete is not None
    return {
        "capture_path": case.capture.path,
        "expected_plan_sha256": case.capture.plan.sha256,
        "expected_capture_sha256": case.capture.complete.sha256,
        "lifecycle_path": case.source.native_path,
        "inference_path": case.source.inference_path,
        "inputs_root": case.root,
        "training": case.training,
        "validation": case.validation,
        "test": case.test,
        "python_policy": _POLICY if any(row.executable for row in case.rows) else None,
    }


def _plan(case: Case) -> IrisScientificAblationEvaluationPlanV2:
    return ledger.build_scientific_ablation_evaluation_plan_v2(**_arguments(case))


def _run(
    case: Case,
    plan: IrisScientificAblationEvaluationPlanV2,
    root: Path,
    **changes: Any,
) -> ledger.IrisScientificAblationEvaluationProgressV2:
    values = {
        **_arguments(case),
        "root": root,
        "expected_evaluation_plan_sha256": plan.sha256,
        "minimum_free_bytes": 0,
        **changes,
    }
    return ledger.evaluate_scientific_ablation_capture_resumable_v2(**values)


def _read(
    plan: IrisScientificAblationEvaluationPlanV2, path: Path, **changes: Any
) -> ledger.IrisScientificAblationEvaluationProgressV2:
    return ledger.read_scientific_ablation_evaluation_progress_v2(
        path,
        **{
            "expected_evaluation_plan_sha256": plan.sha256,
            "expected_plan_sha256": plan.payload.generation_plan.sha256,
            "expected_capture_sha256": plan.payload.capture.sha256,
            **changes,
        },
    )


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    return _make(tmp_path_factory.mktemp("actual-score-ledger-source"))


@pytest.fixture(scope="module")
def plan(case: Case) -> IrisScientificAblationEvaluationPlanV2:
    return _plan(case)


def _snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_ino,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def test_read_only_plan_binds_actual_owned_rows_and_never_scores(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2
) -> None:
    before = _snapshot(case.root)
    with patch.object(scoring, "score_synthetic_answer_v2") as score:
        assert _plan(case) == plan
    score.assert_not_called()
    assert _snapshot(case.root) == before
    assert plan.payload.generation_plan == case.capture.plan
    assert plan.payload.capture == case.capture.complete
    assert tuple(row.capture_row_sha256 for row in plan.payload.rows) == tuple(
        row.sha256 for row in case.capture.rows
    )
    assert tuple(row.capture_ack_sha256 for row in plan.payload.rows) == tuple(
        ack.sha256 for ack in case.capture.acknowledgements
    )
    content = ledger._file(plan)
    for private in (b'"output_text"', b'"expected_response"', b'"prompt"', b'"hidden_facts"'):
        assert private not in content


@pytest.mark.parametrize("default", ("cpu", "meta"))
@pytest.mark.parametrize("inference", (False, True))
def test_partial_resume_exact_retry_skips_ack_matches_uninterrupted_and_restores_caller(
    case: Case,
    plan: IrisScientificAblationEvaluationPlanV2,
    tmp_path: Path,
    default: str,
    inference: bool,
) -> None:
    expected = _evaluate(case)
    caller = capture_runtime_state(torch.device("cpu"))
    (tmp_path / "unrelated-user.txt").write_bytes(b"keep")
    with (
        torch.device(default),
        torch.inference_mode(inference),
        patch.object(IrisScientificAblationHeldoutV1, "execute_row") as generate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
        patch.object(ledger, "_score_row", wraps=_score_row) as score,
    ):
        partial = _run(case, plan, tmp_path, rows_per_call=1)
        assert (
            partial.next_row_index == 1 and partial.evaluation is None and partial.complete is None
        )
        assert score.call_count == 1
        assert _read(plan, partial.path) == partial
        prefix = _snapshot(partial.path / "score-00000")
        score.reset_mock()
        resumed = _run(case, plan, tmp_path)
        assert [call.args[1] for call in score.call_args_list] == [1, 2, 3]
        assert resumed.rows[0] == partial.rows[0]
        assert _snapshot(partial.path / "score-00000") == prefix
        assert resumed.evaluation == expected and resumed.complete is not None
        before = _snapshot(tmp_path)
        score.reset_mock()
        with patch.object(ledger, "_write") as write:
            assert _run(case, plan, tmp_path) == resumed
        write.assert_not_called()
        score.assert_not_called()
        assert _snapshot(tmp_path) == before
        assert torch.get_default_device() == torch.device(default)
        assert torch.is_inference_mode_enabled() is inference
    generate.assert_not_called()
    model.assert_not_called()
    tokenizer.assert_not_called()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert (tmp_path / "unrelated-user.txt").read_bytes() == b"keep"
    assert resumed.plan.payload.generation_plan.payload.cycles == 3


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_actual_all12_c3_earlier_best_owned_score_complete_parity(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    actual = _make(
        tmp_path / "source", routing, goal=goal, broadcast=broadcast, rows=_test_rows()[-1:]
    )
    plan = _plan(actual)
    with patch.object(ledger, "_score_row", wraps=_score_row) as score:
        progress = _run(actual, plan, tmp_path / "scores")
        assert _run(actual, plan, tmp_path / "scores") == progress
    assert score.call_count == 1
    assert progress.evaluation == _evaluate(actual)
    source = plan.payload.generation_plan.payload.source.payload
    assert (source.selected_step, source.terminal_step) == (1, 3)
    assert (source.ablation.routing, source.ablation.goal, source.ablation.broadcast) == (
        routing,
        goal,
        broadcast,
    )
    assert plan.payload.generation_plan.payload.cycles == 3
    assert _read(plan, progress.path) == progress


@pytest.mark.parametrize(
    "field",
    ("expected_evaluation_plan_sha256", "expected_plan_sha256", "expected_capture_sha256"),
)
def test_wrong_anchor_fails_before_mkdir_or_score(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, field: str
) -> None:
    root = tmp_path / "must-not-exist"
    with patch.object(ledger, "_score_row") as score, pytest.raises(Error):
        _run(case, plan, root, **{field: "0" * 64})
    score.assert_not_called()
    assert not root.exists()


@pytest.mark.parametrize("chunk", (True, False, 0, -1, 5001, 1.0, "1"))
def test_chunk_bound_is_exact_and_checked_before_write(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, chunk: object
) -> None:
    root = tmp_path / "not-created"
    with patch.object(ledger, "_score_row") as score, pytest.raises(Error):
        _run(case, plan, root, rows_per_call=chunk)
    score.assert_not_called()
    assert not root.exists()


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group", "ordinary", "infra"))
def test_failed_owned_attempt_is_not_adopted_and_first_error_survives(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, kind: str
) -> None:
    signal: BaseException = (
        KeyboardInterrupt("first")
        if kind == "interrupt"
        else SystemExit(79)
        if kind == "exit"
        else BaseExceptionGroup("first", [SystemExit(81), ValueError("ordinary")])
        if kind == "group"
        else IrisCodeEvaluationUnavailableError("private infrastructure details")
        if kind == "infra"
        else ValueError("private answer and hidden label")
    )
    with (
        patch.object(ledger, "_score_row", side_effect=signal),
        pytest.raises(BaseException) as caught,
    ):
        _run(case, plan, tmp_path, rows_per_call=1)
    if not isinstance(signal, Exception):
        assert caught.value is signal
    else:
        assert isinstance(
            caught.value, IrisCodeEvaluationUnavailableError if kind == "infra" else Error
        )
        assert "private" not in str(caught.value)
    path = tmp_path / ledger._STORE / plan.sha256
    partial = _read(plan, path)
    assert partial.next_row_index == 0 and partial.uncommitted_stage_count == 1
    stages = [entry for entry in path.iterdir() if ledger._STAGE.fullmatch(entry.name)]
    before = _snapshot(stages[0])
    assert json.loads((stages[0] / "attempt.json").read_bytes())["status"] == "owned_before_scoring"
    assert (
        json.loads((stages[0] / "failure.json").read_bytes())["exception_type"]
        == type(signal).__name__
    )
    with patch.object(ledger, "_score_row", wraps=_score_row) as score:
        complete = _run(case, plan, tmp_path)
    assert [call.args[1] for call in score.call_args_list] == [0, 1, 2, 3]
    assert complete.complete is not None and complete.uncommitted_stage_count == 1
    assert _snapshot(stages[0]) == before


@pytest.mark.parametrize("phase", ("plan", "row", "complete"))
def test_ordinary_lost_publication_ack_accepts_exact_committed_target_without_rescore(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, phase: str
) -> None:
    original = rename_directory_with_windows_retry

    def lost(source: Path, target: Path) -> None:
        original(source, target)
        if (
            target.name
            == {"plan": plan.sha256, "row": "score-00000", "complete": "complete"}[phase]
        ):
            raise OSError("lost ordinary return acknowledgement")

    with (
        patch.object(ledger, "rename_directory_with_windows_retry", side_effect=lost),
        patch.object(ledger, "_score_row", wraps=_score_row) as score,
    ):
        actual = _run(case, plan, tmp_path)
    assert score.call_count == 4 and actual.evaluation == _evaluate(case)
    assert _read(plan, actual.path) == actual


@pytest.mark.parametrize("kind", ("interrupt", "exit", "group"))
def test_signal_after_actual_row_commit_is_never_laundered_into_success(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, kind: str
) -> None:
    signal: BaseException = (
        KeyboardInterrupt("postrename")
        if kind == "interrupt"
        else SystemExit(83)
        if kind == "exit"
        else BaseExceptionGroup("signal", [SystemExit(84), ValueError()])
    )
    original = rename_directory_with_windows_retry

    def die(source: Path, target: Path) -> None:
        original(source, target)
        if target.name == "score-00000":
            raise signal

    with (
        patch.object(ledger, "rename_directory_with_windows_retry", side_effect=die),
        pytest.raises(BaseException) as caught,
    ):
        _run(case, plan, tmp_path)
    assert caught.value is signal
    partial = _read(plan, tmp_path / ledger._STORE / plan.sha256)
    assert partial.next_row_index == 1 and partial.complete is None
    with patch.object(ledger, "_score_row", wraps=_score_row) as score:
        result = _run(case, plan, tmp_path)
    assert [call.args[1] for call in score.call_args_list] == [1, 2, 3]
    assert result.evaluation == _evaluate(case)


@pytest.mark.parametrize(
    "kind",
    (
        "row_bytes",
        "ack_bytes",
        "attempt_extra",
        "attempt_pretty",
        "gap",
        "extra",
        "report",
        "complete",
        "hidden",
        "rehashed_nll",
    ),
)
def test_corrupt_or_rehashed_ledger_never_becomes_cache(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, kind: str
) -> None:
    complete = _run(case, plan, tmp_path)
    path = complete.path
    row_dir = path / "score-00000"
    if kind in {"row_bytes", "ack_bytes"}:
        file = row_dir / ("row.json" if kind == "row_bytes" else "ack.json")
        file.write_bytes(file.read_bytes().replace(b'"sha256":', b'"broken":', 1))
    elif kind in {"attempt_extra", "attempt_pretty"}:
        file = row_dir / "attempt.json"
        payload = json.loads(file.read_bytes())
        if kind == "attempt_extra":
            payload["foreign"] = True
            file.write_bytes(canonical_json_file_bytes(payload))
        else:
            file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif kind == "gap":
        (path / "score-00001").rename(path / "score-00004")
    elif kind == "extra":
        (row_dir / "failure.json").write_bytes(b"{}\n")
    elif kind in {"report", "complete"}:
        file = path / "complete" / ("evaluation.json" if kind == "report" else "complete.json")
        file.write_bytes(file.read_bytes()[:-1])
    else:
        payload = complete.rows[0].payload.model_dump(mode="json")
        if kind == "hidden":
            payload["measurements"]["output_text"] = "not allowed"
            row_dir.joinpath("row.json").write_bytes(
                canonical_json_file_bytes({"sha256": "0" * 64, "payload": payload})
            )
        else:
            payload["measurements"]["target_nll_sum"] += 1.0
            parsed = complete.rows[0].payload.__class__.model_validate(payload)
            row = IrisScientificAblationEvaluationRowV2(
                sha256=ledger._sha(ledger._ROW_DOMAIN, parsed), payload=parsed
            )
            row_dir.joinpath("row.json").write_bytes(ledger._file(row))
            ack = ledger._ack(plan.sha256, 0, ledger._genesis(plan.sha256), row)
            row_dir.joinpath("ack.json").write_bytes(ledger._file(ack))
    before = _snapshot(tmp_path)
    with pytest.raises(Error):
        _read(plan, path)
    with patch.object(ledger, "_score_row") as score, pytest.raises(Error):
        _run(case, plan, tmp_path)
    score.assert_not_called()
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ("byte", "file", "directory", "free"))
def test_quota_and_foreign_stage_are_checked_before_expensive_scoring(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, kind: str
) -> None:
    store = tmp_path / ledger._STORE
    stage = store / (".iris-score-" + "b" * 32 + ".tmp")
    stage.mkdir(parents=True)
    (stage / "failure.json").write_bytes(b"x" * 4096)
    before = _snapshot(stage)
    with (
        patch.object(ledger, "_score_row") as score,
        patch.object(ledger, "_MAX_FILES", 0 if kind == "file" else 60000),
        patch.object(ledger, "_MAX_DIRS", 0 if kind == "directory" else 20000),
        patch.object(
            shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=0 if kind == "free" else 1024**4),
        ),
        pytest.raises(Error),
    ):
        _run(case, plan, tmp_path, maximum_storage_bytes=4096 if kind == "byte" else 512 * 1024**2)
    score.assert_not_called()
    assert _snapshot(stage) == before


def test_real_nonblocking_process_lock_rejects_competing_writer(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path
) -> None:
    partial = _run(case, plan, tmp_path, rows_per_call=1)
    before = _snapshot(tmp_path)
    with (
        process_file_lock(partial.path.parent / ledger._LOCK),
        patch.object(ledger, "_score_row") as score,
        pytest.raises(Error),
    ):
        _run(case, plan, tmp_path)
    score.assert_not_called()
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ("capture", "test", "native", "inference", "worker", "hidden"))
def test_source_or_current_row_drift_withholds_ack_and_completion(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path, kind: str
) -> None:
    local = _copy(case, tmp_path / "source")
    root = tmp_path / "scores"
    original = _score_row
    worker_original = _worker

    def drift(inputs: scoring._EvaluationInputs, index: int) -> Any:
        result = original(inputs, index)
        if index == 0:
            if kind == "capture":
                file = local.capture.path / "row-00000" / "row.json"
            elif kind == "test":
                file = local.root / "test.parquet"
            elif kind == "native":
                file = local.source.native_path / "best.safetensors"
            elif kind == "inference":
                file = local.source.inference_path / "state.safetensors"
            elif kind == "hidden":
                inputs.rows[0].hidden_facts["foreign"] = "changed"
                return result
            else:
                return result
            file.write_bytes(file.read_bytes() + b"drift")
        return result

    calls = 0

    def worker(name: str) -> bytes:
        nonlocal calls
        calls += 1
        return worker_original(name) + (b"drift" if kind == "worker" and calls > 20 else b"")

    with (
        patch.object(ledger, "_score_row", side_effect=drift),
        patch.object(scoring, "_worker", side_effect=worker),
        pytest.raises(Error),
    ):
        _run(local, plan, root)
    path = root / ledger._STORE / plan.sha256
    assert not (path / "complete").exists()
    if kind in {"capture", "test", "native", "inference"}:
        assert not (path / "score-00000").exists()


def test_normal_chunk_deep_audit_and_global_inventory_are_per_call_not_per_row(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path
) -> None:
    original = ledger._scan
    scans: list[int] = []

    def scan(path: Path, depth: int, maximum: int) -> dict[Path, tuple[int, ...]]:
        scans.append(depth)
        return original(path, depth, maximum)

    with (
        patch.object(ledger, "_scan", side_effect=scan),
        patch.object(scoring, "_audit", wraps=scoring._audit) as audit,
    ):
        result = _run(case, plan, tmp_path)
    assert result.next_row_index == 4 and result.complete is not None
    assert scans == [0, 1, 1, 0, 1, 1]
    assert audit.call_count == 2


def test_first_signal_survives_failure_journal_and_lock_exit_signals(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path
) -> None:
    first = KeyboardInterrupt("first actual work signal")
    failure_signal = SystemExit(92)
    lock_signal = SystemExit(93)
    original_write = ledger._write

    def write(stage: ledger._Stage, name: str, content: bytes, budget: ledger._Budget) -> None:
        if name == "failure.json":
            raise failure_signal
        original_write(stage, name, content, budget)

    @contextmanager
    def close_signal(path: Path) -> Iterator[ProcessFileLock]:
        with process_file_lock(path) as lock:
            yield lock
        raise lock_signal

    with (
        patch.object(ledger, "_score_row", side_effect=first),
        patch.object(ledger, "_write", side_effect=write),
        patch.object(ledger, "process_file_lock", side_effect=close_signal),
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        _run(case, plan, tmp_path, rows_per_call=1)
    assert caught.value is first
    assert isinstance(first.__cause__, BaseExceptionGroup)
    assert first.__cause__.exceptions == (failure_signal, lock_signal)


def test_same_byte_file_replacement_after_descriptor_close_is_rejected(
    case: Case, plan: IrisScientificAblationEvaluationPlanV2, tmp_path: Path
) -> None:
    original = _finish_descriptor
    replaced = False

    def close(primary: BaseException | None, closer: Callable[[], None]) -> None:
        nonlocal replaced
        original(primary, closer)
        if not replaced:
            stages = list((tmp_path / ledger._STORE).glob(".iris-score-*.tmp"))
            file = stages[0] / "plan.json"
            content = file.read_bytes()
            file.rename(tmp_path / "displaced-plan.json")
            file.write_bytes(content)
            replaced = True

    with (
        patch.object(ledger, "_finish_descriptor", side_effect=close),
        patch.object(ledger, "_score_row") as score,
        pytest.raises(Error),
    ):
        _run(case, plan, tmp_path)
    score.assert_not_called()
    assert replaced and not (tmp_path / ledger._STORE / plan.sha256).exists()


@pytest.mark.parametrize(
    "kind", ("tiny_code_cases", "extra", "missing_worker", "nll", "capture_head")
)
def test_exact_plan_schema_rejects_forged_expectations(
    plan: IrisScientificAblationEvaluationPlanV2, kind: str
) -> None:
    payload = plan.payload.model_dump(mode="json")
    if kind == "tiny_code_cases":
        repair = next(
            index
            for index, row in enumerate(plan.payload.generation_plan.payload.rows)
            if row.task_type == "split_spec_repair"
        )
        payload["rows"][repair]["case_count"] = 1
    elif kind == "extra":
        payload["unowned"] = True
    elif kind == "missing_worker":
        payload["worker_sha256"] = None
    elif kind == "nll":
        payload["rows"][0]["target_nll_sum"] += 1
    else:
        payload["rows"][-1]["capture_ack_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ScientificAblationEvaluationPlanPayloadV2.model_validate(payload)
