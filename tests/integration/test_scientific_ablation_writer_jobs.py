"""Actual tiny CPU writer jobs plus labelled controller faults; no cold/science claim."""

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as controls
from iris.application import scientific_ablation_writer_io as storage
from iris.application import scientific_ablation_writer_jobs as jobs
from iris.application import scientific_ablation_writer_reports as native
from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
    _source_transaction,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_writer_reports import WriterCompleted
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    build_scientific_ablation_causal_observed_budget_v1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    read_scientific_ablation_writer_report_v1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_ablation_writer_source_v1 import (
    WriterSourceProtocol,
    read_scientific_ablation_writer_source_v1,
    scientific_ablation_writer_source_bytes_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_complete_source_preflight import ready
from tests.integration.test_scientific_ablation_diagnostics import Experiment, _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make
from tests.integration.test_scientific_ablation_writer_io import tree

pytestmark = pytest.mark.integration
Error = jobs.IrisScientificAblationWriterJobError


@dataclass(frozen=True)
class Case:
    experiment: Experiment
    owner: IrisScientificAblationHeldoutV1
    protocol: WriterSourceProtocol
    job: jobs.IrisScientificAblationWriterJobV1
    completed: WriterCompleted
    actual_native_calls: int


def fresh(case: Case) -> jobs.IrisScientificAblationWriterJobV1:
    return jobs.create_scientific_ablation_writer_job_v1(
        case.owner,
        input=_input(case.experiment),
        policy=_policy(case.protocol == "message"),
        limits=ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
        expected_protocol=case.protocol,
    )


def phase(job: jobs.IrisScientificAblationWriterJobV1) -> jobs.WriterJobPhase:
    return job.phase


@pytest.fixture(scope="module", params=(False, True), ids=("workspace", "message"))
def case(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Case:
    message = bool(request.param)
    protocol: WriterSourceProtocol = "message" if message else "workspace"
    root = tmp_path_factory.mktemp("actual-native-writer-job")
    rows = _rows()
    assert not any(row.executable for row in rows)  # No generated Python host execution.
    experiment = _experiment(_make(root / "case", rows=rows), root / "scores")
    owner = _heldout(experiment)
    context, run = owner._guard.context, owner._guard.context._run
    model = run._optimizer._model
    job = jobs.create_scientific_ablation_writer_job_v1(
        owner,
        input=_input(experiment),
        policy=_policy(message),
        limits=ScientificAblationCausalCompleteLimitsV1(maximum_observation_scalars=336),
        expected_protocol=protocol,
    )
    assert phase(job) == "pending"
    caller = capture_runtime_state(torch.device("cpu"))
    private = run._runtime
    modes = tuple((child, child.training) for child in model.modules())
    original_init = torch.nn.Module.__init__
    original_budget = build_scientific_ablation_causal_observed_budget_v1
    original_preflight = layout.build_scientific_ablation_causal_report_byte_preflight_v1
    original_reservations: list[Any] = []
    before_observations: list[dict[str, Any]] = []

    def initialize(module: torch.nn.Module) -> None:
        assert type(module) is torch.nn.ModuleList
        original_init(module)

    def before(stage: str) -> None:
        ready(owner)
        assert execute.call_count == 0
        assert all(child.training is mode for child, mode in modes)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        assert _same_runtime(private, run._runtime)
        before_observations.append(
            dict(
                stage=stage,
                states=[owner._state, context._state, context._request._state],
                run_busy=run._busy,
                original_modes_restored=all(child.training is mode for child, mode in modes),
                caller_rng_unchanged=_same_runtime(
                    caller, capture_runtime_state(torch.device("cpu"))
                ),
                private_rng_unchanged=_same_runtime(private, run._runtime),
            )
        )

    def numeric(*args: Any, **kwargs: Any) -> Any:
        before("numeric")
        result = original_budget(*args, **kwargs)
        assert execute.call_count == 0
        return result

    def preflight(*args: Any, **kwargs: Any) -> Any:
        before("bytes")
        reserved = original_preflight(*args, **kwargs)
        original_reservations.append(reserved)
        original_payload = dict(reserved._shape.members)["payload"]
        assert isinstance(original_payload, layout._Object)
        payload = dict(original_payload.members)
        fixed = payload["source_metadata"]
        assert isinstance(fixed, layout._Fixed)
        witness = dict(
            kind="iris-tests-original-complete-source-before-busy-v1",
            scope="trusted-ready-numeric-and-json-reservation-not-full-effects",
            actual_preflight_teacher_executions=execute.call_count,
            before_observations=before_observations,
            complete_layout=layout._descriptor(reserved._shape),
            source_metadata=json.loads(fixed.encoded),
            **{
                field.name: getattr(reserved, field.name)
                for field in fields(reserved)
                if field.name.endswith("_bound") or field.name.endswith("_bytes")
            },
            layout_sha256=reserved.layout_sha256,
            scientific_acceptance="not_assessed",
            full_r015_complete=False,
        )
        (root / ("preflight-" + protocol + ".json")).write_bytes(canonical_json_file_bytes(witness))
        assert execute.call_count == 0
        return reserved

    with (
        patch.object(
            jobs,
            "measure_scientific_ablation_workspace_writer_completed_v1",
            wraps=native.measure_scientific_ablation_workspace_writer_completed_v1,
        ) as workspace,
        patch.object(
            jobs,
            "measure_scientific_ablation_message_writer_completed_v1",
            wraps=native.measure_scientific_ablation_message_writer_completed_v1,
        ) as paired,
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(native, "_source_transaction", wraps=_source_transaction) as transaction,
        patch.object(controls, "build_scientific_ablation_causal_observed_budget_v1", new=numeric),
        patch.object(
            controls, "build_scientific_ablation_causal_report_byte_preflight_v1", new=preflight
        ),
        patch.object(torch.nn.Module, "__init__", new=initialize),
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        completed = job.measure()
        assert workspace.call_count == int(not message)
        assert paired.call_count == int(message)
        assert (
            execute.call_count
            == completed.report.payload.actual_teacher_executions
            == (20 if message else 24)
        )
        assert transaction.call_count == 1 and len(original_reservations) == 1
        assert [event["stage"] for event in before_observations] == ["numeric", "bytes"]
        candidate.assert_not_called()
        loader.assert_not_called()
        tokenizer.assert_not_called()
    ready(owner)
    assert job.phase == "measured"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert _same_runtime(private, owner._guard.context._run._runtime)
    assert all(child.training is mode for child, mode in modes)
    source = completed.report.payload.source_metadata.evaluation.payload.plan.payload.source.payload
    assert (source.selected_step, source.terminal_step) == (1, 3)
    (root / ("native-writer-" + protocol + ".json")).write_bytes(
        scientific_ablation_writer_report_bytes_v1(
            completed.report, preflight=original_reservations[0], runtime=completed.runtime
        )
    )
    (root / ("native-source-" + protocol + ".json")).write_bytes(
        scientific_ablation_writer_source_bytes_v1(
            completed.source, expected_sha256=completed.source.sha256
        )
    )
    (root / "native-job-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-tests-actual-native-writer-job-v1",
                protocol=protocol,
                pid=os.getpid(),
                parent_pid=os.getppid(),
                actual_native_transactions=transaction.call_count,
                actual_teacher_executions=execute.call_count,
                original_preflight_teacher_executions=0,
                workspace_entry_calls=workspace.call_count,
                message_entry_calls=paired.call_count,
                selected_step=source.selected_step,
                terminal_step=source.terminal_step,
                source_sha256=completed.source.sha256,
                report_sha256=completed.report.sha256,
                owner_ready_after_measurement=True,
                caller_rng_unchanged=True,
                private_rng_unchanged=True,
                original_modes_restored=True,
                model_or_tokenizer_loads=0,
                candidate_executions=0,
                scientific_acceptance="not_assessed",
                full_r015_complete=False,
            )
        )
    )
    assert type(job._seal) is jobs._Measured
    assert not hasattr(job._seal, "heldout")  # Completed handle holds DATA, not the native owner.
    return Case(experiment, owner, protocol, job, completed, execute.call_count)


def test_actual_single_native_stage_is_retained_as_defensive_data(case: Case) -> None:
    assert case.actual_native_calls == (20 if case.protocol == "message" else 24)
    copy = case.job.data
    assert copy == case.completed and copy is not case.completed
    assert copy.report is not case.completed.report
    assert copy.source is not case.completed.source
    object.__setattr__(copy.report, "sha256", "f" * 64)
    assert case.job.data.report.sha256 == case.completed.report.sha256


def test_measure_cache_and_real_publication_retry_never_call_model_again(
    case: Case, tmp_path: Path
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(
            jobs,
            "measure_scientific_ablation_workspace_writer_completed_v1",
            side_effect=AssertionError("No native replay"),
        ),
        patch.object(
            jobs,
            "measure_scientific_ablation_message_writer_completed_v1",
            side_effect=AssertionError("No native replay"),
        ),
        patch.object(
            torch.nn.Module, "__init__", side_effect=AssertionError("No model construction")
        ),
        patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No model call")),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate execution"),
        ),
    ):
        assert case.job.measure() == case.completed
        path = case.job.publish(tmp_path)
        before = tree(path)
        with patch.object(storage, "_write", side_effect=AssertionError("No retry file writes")):
            assert case.job.publish(tmp_path) == path
            assert case.job.measure() == case.completed
        assert tree(path) == before
    assert case.job.phase == "published"
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("signal", (ValueError, KeyboardInterrupt, SystemExit, GeneratorExit))
def test_real_after_commit_signal_preserves_cache_and_exact_no_write_retry(
    case: Case,
    tmp_path: Path,
    signal: type[BaseException],
) -> None:
    with (
        patch.object(
            jobs, "read_scientific_ablation_writer_bundle_v1", side_effect=signal("late read")
        ),
        pytest.raises(signal),
    ):
        case.job.publish(tmp_path)
    assert case.job.phase == "measured"
    target = (
        tmp_path / f"scientific-ablation-{case.protocol}-writer-v1" / case.completed.report.sha256
    )
    assert target.is_dir()
    before = tree(target)
    with (
        patch.object(storage, "_write", side_effect=AssertionError("No after-commit retry writes")),
        patch.object(
            jobs,
            "measure_scientific_ablation_workspace_writer_completed_v1",
            side_effect=AssertionError("No native replay"),
        ),
        patch.object(
            jobs,
            "measure_scientific_ablation_message_writer_completed_v1",
            side_effect=AssertionError("No native replay"),
        ),
    ):
        assert case.job.publish(tmp_path) == target
    assert tree(target) == before and phase(case.job) == "published"


def test_prepublication_quota_failure_retains_data_without_implicit_measure(
    case: Case, tmp_path: Path
) -> None:
    with pytest.raises(storage.IrisScientificAblationWriterStorageError):
        case.job.publish(tmp_path, maximum_storage_bytes=1)
    assert case.job.phase == "measured" and not list(tmp_path.iterdir())
    assert case.job.data == case.completed


def test_pending_publish_and_data_never_implicitly_run_native(case: Case, tmp_path: Path) -> None:
    pending = fresh(case)
    with (
        patch.object(
            jobs, "measure_scientific_ablation_workspace_writer_completed_v1"
        ) as workspace,
        patch.object(jobs, "measure_scientific_ablation_message_writer_completed_v1") as paired,
    ):
        with pytest.raises(Error):
            pending.publish(tmp_path)
        with pytest.raises(Error):
            _ = pending.data
        workspace.assert_not_called()
        paired.assert_not_called()
    assert pending.phase == "pending" and not list(tmp_path.iterdir())


@pytest.mark.parametrize("signal", (ValueError, KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("point", ("native-entry", "late-cache"))
def test_controller_fault_never_exposes_partial_result_or_allows_remeasurement(
    case: Case,
    tmp_path: Path,
    signal: type[BaseException],
    point: str,
) -> None:
    # Controller unit injection: returned fixture DATA is NOT another native witness.
    pending = fresh(case)
    target = "measure_scientific_ablation_" + case.protocol + "_writer_completed_v1"
    with (
        patch.object(
            jobs,
            target,
            **(
                {"side_effect": signal("native")}
                if point == "native-entry"
                else {"return_value": case.completed}
            ),
        ) as measure,
        patch.object(jobs, "_capture", side_effect=signal("cache")) as capture,
        pytest.raises(signal),
    ):
        pending.measure()
    assert measure.call_count == 1
    assert capture.call_count == int(point == "late-cache")
    assert pending.phase == "failed"
    with (
        patch.object(jobs, target, side_effect=AssertionError("No failed job replay")),
        pytest.raises(Error),
    ):
        pending.measure()
    with pytest.raises(Error):
        pending.publish(tmp_path)
    with pytest.raises(Error):
        _ = pending.data
    assert not list(tmp_path.iterdir())


def test_busy_job_reentry_is_denied_without_implicit_second_native_stage(case: Case) -> None:
    pending = fresh(case)
    target = "measure_scientific_ablation_" + case.protocol + "_writer_completed_v1"

    def body(*args: Any, **kwargs: Any) -> WriterCompleted:
        assert pending.phase == "measuring"
        with pytest.raises(Error):
            pending.measure()
        with pytest.raises(Error):
            _ = pending.data
        return case.completed  # Controller-only injection; not another native claim.

    with patch.object(jobs, target, side_effect=body) as measure:
        assert pending.measure() == case.completed
    assert measure.call_count == 1 and pending.phase == "measured"


def test_changed_private_cached_bytes_never_trigger_fallback_measurement(
    case: Case, tmp_path: Path
) -> None:
    pending = fresh(case)
    target = "measure_scientific_ablation_" + case.protocol + "_writer_completed_v1"
    with patch.object(jobs, target, return_value=case.completed):
        pending.measure()  # Controller unit injection, NOT an extra native witness.
    assert type(pending._seal) is jobs._Measured
    object.__setattr__(pending._seal, "source", b"changed")
    with patch.object(jobs, target, side_effect=AssertionError("No native fallback")):
        with pytest.raises(Error):
            pending.measure()
        with pytest.raises(Error):
            pending.publish(tmp_path)
    assert not list(tmp_path.iterdir())


def controller_cache(case: Case) -> jobs.IrisScientificAblationWriterJobV1:
    # Controller-only DATA injection, never another actual native measurement.
    job = fresh(case)
    target = "measure_scientific_ablation_" + case.protocol + "_writer_completed_v1"
    with patch.object(jobs, target, return_value=case.completed):
        job.measure()
    return job


@pytest.mark.parametrize("operation", ("data", "publish"))
def test_decode_reentry_is_already_busy_before_any_public_data_escapes(
    case: Case,
    tmp_path: Path,
    operation: str,
) -> None:
    job = controller_cache(case)
    original = read_scientific_ablation_writer_source_v1

    def decode(*args: Any, **kwargs: Any) -> Any:
        assert phase(job) == ("reading" if operation == "data" else "publishing")
        with pytest.raises(Error):
            job.measure()
        with pytest.raises(Error):
            _ = job.data
        with pytest.raises(Error):
            job.publish(tmp_path / "never-created")
        return original(*args, **kwargs)

    with patch.object(
        jobs, "read_scientific_ablation_writer_source_v1", side_effect=decode
    ) as reader:
        if operation == "data":
            assert job.data == case.completed
        else:
            assert job.publish(tmp_path).is_dir()
    assert reader.call_count == 1
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("signal", (ValueError, KeyboardInterrupt, SystemExit, GeneratorExit))
def test_malformed_cache_cleanup_cannot_mask_the_original_after_commit_signal(
    case: Case,
    tmp_path: Path,
    signal: type[BaseException],
) -> None:
    job = controller_cache(case)
    primary = signal("first original signal")

    def late(*args: Any, **kwargs: Any) -> Any:
        assert type(job._seal) is jobs._Measured
        object.__setattr__(job._seal, "runtime", object())
        raise primary

    with (
        patch.object(jobs, "read_scientific_ablation_writer_bundle_v1", side_effect=late),
        pytest.raises(signal) as caught,
    ):
        job.publish(tmp_path)
    assert caught.value is primary and phase(job) == "failed"
    target = (
        tmp_path / f"scientific-ablation-{case.protocol}-writer-v1" / case.completed.report.sha256
    )
    assert target.is_dir()  # Real commit is retained; do not delete/rewrite it.


def test_late_data_byte_drift_returns_no_detached_result(case: Case) -> None:
    job = controller_cache(case)
    original = read_scientific_ablation_writer_report_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert type(job._seal) is jobs._Measured
        object.__setattr__(job._seal, "source", b"changed-after-decode")
        return result

    with (
        patch.object(jobs, "read_scientific_ablation_writer_report_v1", side_effect=changed),
        pytest.raises(Error, match="changed during data read"),
    ):
        _ = job.data
    assert phase(job) == "failed"


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_wrong_protocol_policy_pair_fails_before_original_owner_audit(
    protocol: WriterSourceProtocol,
) -> None:
    bad: Any = object()
    with (
        patch.object(jobs, "_heldout_entry", side_effect=AssertionError("No owner touch")),
        pytest.raises(ValueError, match="protocol/policy"),
    ):
        jobs.create_scientific_ablation_writer_job_v1(
            bad,
            input=bad,
            policy=_policy(protocol != "message"),
            limits=ScientificAblationCausalCompleteLimitsV1(),
            expected_protocol=protocol,
        )


def test_direct_construction_or_unbound_foreign_receiver_is_not_an_owned_job() -> None:
    with pytest.raises(Error):
        jobs.IrisScientificAblationWriterJobV1()
    bad: Any = object()
    with pytest.raises(Error):
        jobs.IrisScientificAblationWriterJobV1.measure(bad)
