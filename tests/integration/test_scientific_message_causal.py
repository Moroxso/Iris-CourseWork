"""Actual selected-best tiny-Qwen paired messages, audited source and private CLI."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application import scientific_command as command
from iris.application import scientific_generation as generation
from iris.application import scientific_message_causal as messages
from iris.application import scientific_message_command as message_command
from iris.application.scientific_answer_command import IrisScientificAnswerCommandError
from iris.application.scientific_evaluation import IrisScientificEvaluationError
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_teacher import IrisScientificTeacherV1
from iris.application.workspace_message_causal import measure_workspace_message_causal_v1
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.artifacts.scientific_message_causal_v1 import (
    IrisScientificMessageCausalV1,
    ScientificMessageCausalPolicyV1,
)
from iris.cli import app
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from iris.workspace.message_causal import (
    WorkspaceMessageCausalCaptureV1,
    WorkspaceMessageCausalPolicyV1,
)
from tests.integration.test_scientific_command import _edit, _files, _recipe
from tests.integration.test_scientific_evaluation import CaptureCase, _recapture
from tests.integration.test_scientific_generation import _generate, _test_spec
from tests.integration.test_scientific_workspace import local_loaders as local_loaders

pytestmark = pytest.mark.integration

_POLICY = ScientificMessageCausalPolicyV1(
    selection_seed=42,
    examples_per_stratum=2,
    causal=WorkspaceMessageCausalPolicyV1(shuffle_seed=1337),
)


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory, local_loaders: None) -> CaptureCase:
    root = tmp_path_factory.mktemp("message-source")
    recipe = _recipe(root)
    _edit(recipe, lambda raw: raw["workspace"].update(cycles=2))
    _edit(
        recipe,
        lambda raw: raw["lifecycle"].update(minimum_improvement=1e6, early_stopping_patience=2),
    )
    created = command.create_scientific_command_v1(config_path=recipe, output_root=root / "runs")
    done = command.resume_scientific_command_v1(
        config_path=recipe, execution_path=created.path, phases=1000
    )
    assert (done.state.step, done.state.best_step, done.state.phase) == (3, 1, "stopped")
    available = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000100, count=60),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = tuple(
        row
        for task in TASKS
        for row in tuple(r for r in available if r.task_type == task and r.difficulty == 1)[:3]
    )
    assert len(rows) == 12
    test = _test_spec(root / "test.parquet", rows)
    capture = _generate((recipe, done.path), test)
    path = generation.write_scientific_generation_v1(capture, root=root / "captures")
    return CaptureCase(recipe, done.path, test, path, capture, rows)


@pytest.fixture
def case(tmp_path: Path, captured: CaptureCase) -> CaptureCase:
    source, target = captured.recipe.parent, tmp_path / "source"
    shutil.copytree(source, target)
    return replace(
        captured,
        recipe=target / captured.recipe.name,
        execution=target / captured.execution.relative_to(source),
        generation_path=target / captured.generation_path.relative_to(source),
    )


def measure(
    case: CaptureCase, policy: ScientificMessageCausalPolicyV1 = _POLICY
) -> IrisScientificMessageCausalV1:
    return messages.measure_scientific_messages_v1(
        generation_path=case.generation_path,
        execution_path=case.execution,
        config_path=case.recipe,
        test=case.test,
        policy=policy,
    )


@pytest.fixture(scope="module")
def reported(captured: CaptureCase) -> IrisScientificMessageCausalV1:
    return measure(captured)


def test_actual_selected_best_pair_report_preserves_source_rng_and_has_all_effects(
    case: CaptureCase,
    reported: IrisScientificMessageCausalV1,
) -> None:
    before, rng = _files(case.recipe.parent), capture_runtime_state(torch.device("cpu"))
    with (
        patch(
            "iris.application.request.IrisRequestOrchestrator.generate",
            side_effect=AssertionError("no generation"),
        ),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("no execution"),
        ),
    ):
        result = measure(case)
    assert result == reported
    assert _files(case.recipe.parent) == before
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    p = result.payload
    assert (p.selected_step, p.terminal_step, p.executions, p.trace_scalar_bound) == (
        1,
        3,
        48,
        18144,
    )
    assert len(p.members) == 12 and len(p.rows) == 8 and len(p.causal.donor_pairs) == 4
    assert p.causal.executions == 40 and len(p.causal.effects) == 48
    assert p.best_checkpoint_sha256 == case.capture.payload.best_checkpoint_sha256
    assert p.scientific_acceptance == p.causal.scientific_acceptance == "not_assessed"
    saved = {r.example_id: r for r in case.capture.payload.rows}
    assert all(r.control_nll_sum == saved[r.example_id].target_nll_sum for r in p.rows)
    assert all(r.causal is None and all(c.full is None for c in r.trace.cycles) for r in p.rows)
    assert any(e.genuine_write_distance > 1e-6 for e in p.causal.effects)
    assert all(not s.significant_positive_effect for s in p.causal.statistics)
    assert [(g.available_examples, g.selected_examples) for g in p.groups[1:5]] == [(3, 2)] * 4
    for forbidden in (
        '"prompt"',
        '"values"',
        '"expected_response"',
        '"output_text"',
        '"hidden_facts"',
    ):
        assert forbidden not in result.model_dump_json()


def test_full_stored_vectors_are_opt_in_without_changing_effects(
    case: CaptureCase, reported: IrisScientificMessageCausalV1
) -> None:
    result = measure(case, _POLICY.model_copy(update={"trace_mode": "full"}))
    assert all(c.full is not None for r in result.payload.rows for c in r.trace.cycles)
    assert result.payload.causal == reported.payload.causal
    assert result.payload.groups == reported.payload.groups
    assert (result.payload.executions, result.payload.trace_scalar_bound) == (48, 19680)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_executions": 47},
        {"maximum_total_trace_scalars": 18143},
        {"maximum_trace_scalars": 1},
        {"maximum_message_scalars": 1},
    ],
)
def test_whole_budget_rejects_before_model_loading(
    case: CaptureCase, changes: dict[str, int]
) -> None:
    policy = _POLICY.model_copy(
        update={
            "causal": WorkspaceMessageCausalPolicyV1.model_validate(
                {**_POLICY.causal.model_dump(), **changes}
            )
        }
    )
    with (
        patch.object(messages, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case, policy)
    compose.assert_not_called()


def test_unselected_source_member_is_audited_before_loading(
    case: CaptureCase, reported: IrisScientificMessageCausalV1
) -> None:
    selected = {r.example_id for r in reported.payload.rows}
    wrong = next(r.example_id for r in case.capture.payload.rows if r.example_id not in selected)
    rows = tuple(
        r.model_copy(update={"example_sha256": "0" * 64}) if r.example_id == wrong else r
        for r in case.capture.payload.rows
    )
    case = _recapture(case, rows=rows)
    with (
        patch.object(messages, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    compose.assert_not_called()


@pytest.mark.parametrize(
    "field", ["best_checkpoint_sha256", "test_corpus_sha256", "teacher_configuration_sha256"]
)
def test_rehashed_wrong_source_identity_fails_before_loading(case: CaptureCase, field: str) -> None:
    case = _recapture(case, **{field: "0" * 64})
    with (
        patch.object(messages, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    compose.assert_not_called()


def test_selected_control_nll_is_actually_reproduced(case: CaptureCase) -> None:
    case = _recapture(
        case,
        rows=tuple(
            r.model_copy(update={"target_nll_sum": r.target_nll_sum + 1.0})
            for r in case.capture.payload.rows
        ),
    )
    with (
        patch.object(messages, "measure_workspace_message_causal_v1") as causal,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    causal.assert_not_called()


def test_changed_initial_recipe_fails_before_any_teacher(case: CaptureCase) -> None:
    _edit(case.recipe, lambda raw: raw.update(initialization_seed=99))
    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward") as forward,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    forward.assert_not_called()


def test_failure_restores_rng_and_redacts_private_content(case: CaptureCase) -> None:
    rng, before = capture_runtime_state(torch.device("cpu")), _files(case.recipe.parent)

    def fail(*args: Any, **kwargs: Any) -> Any:
        torch.rand(9)
        raise RuntimeError("secret prompt/target")

    with (
        patch.object(messages, "measure_workspace_message_causal_v1", side_effect=fail),
        pytest.raises(IrisScientificEvaluationError) as error,
    ):
        measure(case)
    assert "secret" not in str(error.value) and error.value.__suppress_context__
    assert _same_runtime(rng, capture_runtime_state(torch.device("cpu")))
    assert _files(case.recipe.parent) == before


def test_roundtrip_idempotence_and_read_only_inspection(
    reported: IrisScientificMessageCausalV1, tmp_path: Path
) -> None:
    path = messages.write_scientific_messages_v1(reported, root=tmp_path)
    assert messages.write_scientific_messages_v1(reported, root=tmp_path) == path
    with patch.object(messages, "_composed", side_effect=AssertionError("no model")):
        assert messages.read_scientific_messages_v1(path) == reported


@pytest.mark.parametrize("corruption", ["duplicate", "extra", "noncanonical", "groups", "writer"])
def test_rehashed_invalid_design_or_corrupted_store_rejects(
    reported: IrisScientificMessageCausalV1, tmp_path: Path, corruption: str
) -> None:
    path = messages.write_scientific_messages_v1(reported, root=tmp_path)
    file = path / "messages.json"
    if corruption == "duplicate":
        file.write_bytes(b'{"sha256":"' + reported.sha256.encode() + b'",' + file.read_bytes()[1:])
    elif corruption == "extra":
        (path / "foreign").write_bytes(b"preserve")
    elif corruption == "noncanonical":
        file.write_bytes(file.read_bytes() + b" ")
    else:
        raw = json.loads(file.read_bytes())
        if corruption == "groups":
            raw["payload"]["groups"][0]["selected_examples"] = 999
        else:
            for effect in raw["payload"]["causal"]["effects"]:
                effect["slot"] = 1 - effect["slot"]
            WorkspaceMessageCausalCaptureV1.model_validate_json(
                canonical_json_bytes(raw["payload"]["causal"])
            )
        raw["sha256"] = domain_sha256(
            "iris/scientific-message-causal/v1", canonical_json_bytes(raw["payload"])
        )
        file.write_bytes(canonical_json_bytes(raw) + b"\n")
        new = path.with_name(raw["sha256"])
        rename_directory_with_windows_retry(path, new)
        path = new
    with pytest.raises(IrisScientificEvaluationError):
        messages.read_scientific_messages_v1(path)


def test_quota_reserve_and_short_write_never_publish_partial_report(
    reported: IrisScientificMessageCausalV1, tmp_path: Path
) -> None:
    with pytest.raises(IrisScientificEvaluationError):
        messages.write_scientific_messages_v1(
            reported, root=tmp_path / "quota", maximum_storage_bytes=1
        )
    with (
        patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
        pytest.raises(IrisScientificEvaluationError),
    ):
        messages.write_scientific_messages_v1(reported, root=tmp_path / "reserve")

    def short(path: Path, content: bytes) -> None:
        path.write_bytes(content[:-1])

    with (
        patch.object(messages, "_write_exact_file", side_effect=short),
        pytest.raises(IrisScientificEvaluationError),
    ):
        messages.write_scientific_messages_v1(reported, root=tmp_path / "short")
    assert not tuple((tmp_path / "short" / messages._STORE).iterdir())


def command_files(case: CaptureCase) -> tuple[Path, Path]:
    policy, descriptor = (
        case.recipe.parent / "message-policy.json",
        case.recipe.parent / "test-input.json",
    )
    policy.write_text(_POLICY.model_dump_json(), encoding="utf-8")
    descriptor.write_text(case.test.model_dump_json(), encoding="utf-8")
    return policy, descriptor


def test_actual_cli_and_model_free_inspection(case: CaptureCase, tmp_path: Path) -> None:
    policy, descriptor = command_files(case)
    output = tmp_path / "results"
    runner = CliRunner()
    response = runner.invoke(
        app,
        [
            "scientific",
            "diagnose-messages",
            "--config",
            str(case.recipe),
            "--execution",
            str(case.execution),
            "--generation",
            str(case.generation_path),
            "--test-input",
            str(descriptor),
            "--policy",
            str(policy),
            "--output-root",
            str(output),
        ],
    )
    assert response.exit_code == 0, response.output
    (path,) = (output / messages._STORE).iterdir()
    assert "not_assessed" in response.output
    with patch.object(messages, "_composed", side_effect=AssertionError("no model")):
        response = runner.invoke(app, ["scientific", "inspect-messages", str(path)])
    assert response.exit_code == 0 and "source not re-audited" in response.output
    assert case.rows[0].prompt not in response.output


def test_changed_policy_prevents_publication(
    case: CaptureCase, reported: IrisScientificMessageCausalV1, tmp_path: Path
) -> None:
    policy, descriptor = command_files(case)

    def changed(**kwargs: Any) -> IrisScientificMessageCausalV1:
        policy.write_bytes(policy.read_bytes() + b" ")
        return reported

    with (
        patch.object(message_command, "measure_scientific_messages_v1", side_effect=changed),
        patch.object(message_command, "write_scientific_messages_v1") as write,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        message_command.diagnose_scientific_messages_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=tmp_path / "out",
        )
    write.assert_not_called()


def test_output_store_cannot_overlap_source(case: CaptureCase) -> None:
    policy, descriptor = command_files(case)
    with (
        patch.object(message_command, "measure_scientific_messages_v1") as measure_call,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        message_command.diagnose_scientific_messages_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=case.execution,
        )
    measure_call.assert_not_called()


def test_public_api_is_canonical() -> None:
    import iris.application as application

    assert application.measure_scientific_messages_v1 is messages.measure_scientific_messages_v1
    assert application.read_scientific_messages_v1 is messages.read_scientific_messages_v1
    assert application.write_scientific_messages_v1 is messages.write_scientific_messages_v1
    assert (
        application.diagnose_scientific_messages_command_v1
        is message_command.diagnose_scientific_messages_command_v1
    )


def test_selected_inference_export_must_match_before_any_teacher(case: CaptureCase) -> None:
    case = _recapture(case, inference_sha256="0" * 64)
    with (
        patch.object(IrisScientificTeacherV1, "evaluation_forward") as forward,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    forward.assert_not_called()


def test_actual_model_mutation_after_causal_execution_is_not_publishable(case: CaptureCase) -> None:
    actual = measure_workspace_message_causal_v1

    def changed(model: Any, *args: Any, **kwargs: Any) -> Any:
        result = actual(model, *args, **kwargs)
        with torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).add_(1.0)
        return result

    with (
        patch.object(messages, "measure_workspace_message_causal_v1", side_effect=changed),
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)


def test_recipe_bytes_are_rechecked_after_real_measurement(case: CaptureCase) -> None:
    actual = measure_workspace_message_causal_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = actual(*args, **kwargs)
        case.recipe.write_bytes(case.recipe.read_bytes() + b" ")
        return result

    with (
        patch.object(messages, "measure_workspace_message_causal_v1", side_effect=changed),
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)


def test_audited_singleton_strata_fail_before_model_loading(case: CaptureCase) -> None:
    rows = case.rows[::3]
    assert len(rows) == 4 and len({r.task_type for r in rows}) == 4
    spec = _test_spec(case.recipe.parent / "singletons.parquet", rows)
    capture = _generate((case.recipe, case.execution), spec)
    path = generation.write_scientific_generation_v1(
        capture, root=case.recipe.parent / "singletons"
    )
    case = replace(case, rows=rows, test=spec, capture=capture, generation_path=path)
    with (
        patch.object(messages, "_composed") as compose,
        pytest.raises(IrisScientificEvaluationError),
    ):
        measure(case)
    compose.assert_not_called()


def test_lost_rename_acknowledgement_reuses_complete_report(
    reported: IrisScientificMessageCausalV1,
    tmp_path: Path,
) -> None:
    def lost(source: Path, target: Path) -> None:
        rename_directory_with_windows_retry(source, target)
        raise OSError("lost acknowledgement")

    with patch.object(messages, "rename_directory_with_windows_retry", side_effect=lost):
        path = messages.write_scientific_messages_v1(reported, root=tmp_path)
    assert messages.read_scientific_messages_v1(path) == reported
    assert tuple(path.parent.iterdir()) == (path,)


def test_foreign_staging_collision_is_preserved(
    reported: IrisScientificMessageCausalV1,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / messages._STORE / ".iris-message-fixed.tmp"
    candidate.mkdir(parents=True)
    sentinel = candidate / "foreign"
    sentinel.write_bytes(b"preserve")
    with (
        patch.object(messages, "uuid4", return_value=SimpleNamespace(hex="fixed")),
        pytest.raises(IrisScientificEvaluationError),
    ):
        messages.write_scientific_messages_v1(reported, root=tmp_path)
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize("invalid", ["duplicate", "odd-cap", "missing-causal", "unknown"])
def test_invalid_explicit_policy_rejects_before_command_dispatch(
    case: CaptureCase,
    tmp_path: Path,
    invalid: str,
) -> None:
    policy, descriptor = command_files(case)
    raw = _POLICY.model_dump(mode="json")
    if invalid == "duplicate":
        policy.write_bytes(b'{"selection_seed":42,' + canonical_json_bytes(raw)[1:])
    else:
        if invalid == "odd-cap":
            raw["examples_per_stratum"] = 3
        elif invalid == "missing-causal":
            del raw["causal"]
        else:
            raw["unknown"] = True
        policy.write_bytes(canonical_json_bytes(raw))
    with (
        patch.object(message_command, "measure_scientific_messages_v1") as measure_call,
        pytest.raises(IrisScientificAnswerCommandError),
    ):
        message_command.diagnose_scientific_messages_command_v1(
            config_path=case.recipe,
            execution_path=case.execution,
            generation_path=case.generation_path,
            test_input_path=descriptor,
            policy_path=policy,
            output_root=tmp_path / "out",
        )
    measure_call.assert_not_called()
