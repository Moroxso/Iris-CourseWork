"""Genuine ALL4 source-bound planning/CLI rows, explicit original/recovered DATA.

Real trained selected export + original factory/claim/owners/rows. Wrappers only
observe original calls/results or inject ONE primary signal AFTER an actual ACK.
4non-Python rows/one nonempty control/original16MiB; no generated Python, cold
OS process, installed package, CLI training/diagnostics/writer or science claim.
"""

import json
import traceback
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
from typer.testing import CliRunner

from iris.application import scientific_ablation_workflow_command as public
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_ablation_workflow_io import (
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_ablation_workflow_planning import (
    build_scientific_ablation_workflow_kernel_request_v1,
)
from iris.artifacts.scientific_ablation_workflow_planning_v1 import (
    IrisScientificAblationWorkflowKernelRequestV1,
    ScientificAblationWorkflowKernelRequestPayloadV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.cli import app
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_execution import _tree
from tests.integration.test_scientific_ablation_operator_export import _intent
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import _row_pins
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    cache_ready as cache_ready,
)
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    completed as completed,
)
from tests.integration.test_scientific_ablation_workflow_multirow_recovery import (
    multirow_bound as multirow_bound,
)
from tests.integration.test_scientific_ablation_workflow_public_chunk import _binding

pytestmark = pytest.mark.integration
runner = CliRunner()
ACTUAL_CLI_OBSERVATIONS: list[dict[str, Any]] = []


def _write(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_file_bytes(value.model_dump(mode="json")))


def _delta(now: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: value - before[key] for key, value in now.items()}


@contextmanager
def _observe(stack: ExitStack, errors: list[dict[str, Any]]) -> Iterator[dict[str, int]]:
    counts = dict(graph=0, qwen=0, teacher=0, model=0, tokenizer=0)

    def observing(original: Any, label: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            counts[label] += 1
            return original(*args, **kwargs)

        return call

    for owner, field, label in (
        (torch.nn.Module, "__init__", "graph"),
        (Qwen3ForCausalLM, "forward", "qwen"),
        (IrisScientificAblationTeacherV1, "evaluation_forward", "teacher"),
        (AutoModelForCausalLM, "from_pretrained", "model"),
        (AutoTokenizer, "from_pretrained", "tokenizer"),
    ):
        stack.enter_context(patch.object(owner, field, new=observing(getattr(owner, field), label)))
    for command_owner, name in (
        (public.IrisScientificAblationOperatorCommandWorkflowV1, "plan_kernel"),
        (public.IrisScientificAblationOperatorCommandWorkflowV1, "dispatch"),
        (public.IrisScientificAblationOperatorCommandWorkflowV1, "recover"),
        (public, "create_scientific_ablation_operator_command_workflow_v1"),
    ):
        original = getattr(command_owner, name)

        def measured(
            self: Any, *args: Any, _original: Any = original, _name: str = name, **kwargs: Any
        ) -> Any:
            try:
                return _original(self, *args, **kwargs)
            except BaseException as primary:
                chain: list[dict[str, Any]] = []
                error: BaseException | None = primary
                seen: set[int] = set()
                while error is not None and id(error) not in seen and len(chain) < 12:
                    seen.add(id(error))
                    chain.append(
                        dict(
                            type=type(error).__name__,
                            message=str(error)[:4000],
                            frames=[
                                dict(file=f.filename, line=f.lineno, function=f.name)
                                for f in traceback.extract_tb(error.__traceback__)
                            ],
                        )
                    )
                    error = error.__cause__ or error.__context__
                errors.append(dict(method=_name, exception_chain=chain))
                raise

        stack.enter_context(patch.object(command_owner, name, new=measured))
    yield counts


def _call(
    common: list[str],
    arguments: list[str],
    logs: list[dict[str, Any]],
    counts: dict[str, int],
    expected: int = 0,
) -> dict[str, Any] | None:
    before = counts.copy()
    argv = ["scientific-ablation", "workflow", *common, *arguments]
    result = runner.invoke(app, argv)
    value = json.loads(result.stdout) if result.exit_code == 0 else None
    record = dict(
        argv=argv,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        response=value,
        actual_counts=_delta(counts, before),
    )
    logs.append(record)
    assert result.exit_code == expected, record
    if expected == 0:
        assert type(value) is dict and result.stdout.encode() == canonical_json_file_bytes(value)
    return value


def _common(
    path: Path,
    request: Path,
    exports: Path,
    plan: Path,
    base: Path,
    root: Path,
    anchors: tuple[str, ...],
) -> list[str]:
    result = [
        "--operator",
        str(path),
        "--export-request",
        str(request),
        "--export-root",
        str(exports),
        "--plan",
        str(plan),
        "--base",
        str(base),
        "--root",
        str(root),
    ]
    for digest in anchors:
        result.extend(("--completed-phase-sha256", digest))
    return result


def _assert_zero(counts: dict[str, int], before: dict[str, int]) -> None:
    assert not any(_delta(counts, before).values())


def _negative_initial_planning(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    destination: Path,
    request_file: Path,
    logs: list[dict[str, Any]],
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Real separate source-bound namespaces; never contaminate/clean positive DATA."""
    path, base, exports, plan = bound
    observations: list[dict[str, Any]] = []
    before = counts.copy()
    for label in ("source", "quota", "unknown-kernel", "unknown-attempt", "unknown-phase"):
        negative = destination / label
        negative.mkdir()
        payload = plan.payload.model_dump(mode="python")
        if label == "source":
            assert payload["source_sha256"] != "f" * 64
            payload["source_sha256"] = "f" * 64
        elif label == "quota":
            payload["resources"] = payload["resources"] | dict(maximum_storage_bytes=2 * 1024**2)
        chosen = build_scientific_ablation_workflow_plan_v1(
            ScientificAblationWorkflowIntentV1.model_validate(payload)
        )
        plan_file = negative / "plan.json"
        _write(plan_file, chosen)
        root = negative / "commands"
        common = _common(path, request_file, exports, plan_file, base, root, ())
        if label == "source":
            _call(common, ["status"], logs, counts, expected=2)
            assert not root.exists()
            observations.append(dict(label=label, plan=chosen.model_dump(mode="json")))
            _assert_zero(counts, before)
            continue
        request = negative / "request.json"
        phase = scientific_ablation_workflow_phases_v1(chosen)[0]
        saved = _call(
            common,
            [
                "kernel-request",
                "--phase",
                phase.name,
                "--initial",
                "--rows-per-call",
                "3",
                "--out",
                str(request),
            ],
            logs,
            counts,
        )
        assert saved is not None
        facade = public.create_scientific_ablation_operator_command_workflow_v1(
            path, request=_intent(path), export_root=exports, plan=chosen, base=base, root=root
        )
        store = facade._seal.job._seal.store
        # Original claim initializes original metadata; no fake factory/owner/ACK.
        with store.claim():
            pass
        marker: Path | None = None
        if label.startswith("unknown-"):
            if label == "unknown-attempt":
                marker = store.path / "phase-000.attempt.json"
            else:
                parent = (
                    store.path / "kernels" / phase.name
                    if label == "unknown-kernel"
                    else store.path / "phase-000"
                )
                parent.mkdir(parents=True)
                marker = parent / "unknown-must-remain.bin"
            with marker.open("xb") as stream:
                stream.write(b"Unknown original namespace DATA; preserve, never adopt.\n")
        tree = _tree(store.path)
        output = negative / "must-not-exist.json"
        _call(
            common,
            ["kernel-plan", "--request", str(request), "--out", str(output)],
            logs,
            counts,
            expected=2,
        )
        _assert_zero(counts, before)
        assert not output.exists() and _tree(store.path) == tree
        assert marker is None or marker.is_file()
        observations.append(
            dict(
                label=label,
                plan=chosen.model_dump(mode="json"),
                request=saved,
                workflow_path=str(store.path),
                unknown_marker=None if marker is None else str(marker),
                actual_tree_unchanged=True,
                actual_counts=_delta(counts, before),
            )
        )
    return observations


@pytest.mark.parametrize("scenario", ("original", "recovery"))
def test_actual_all4_planning_and_CLI_original_or_recovered_rows(
    multirow_bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1],
    tmp_path: Path,
    scenario: str,
) -> None:
    path, base, exports, plan = multirow_bound
    assert len(plan.payload.controls) == 1 and plan.payload.controls[0].interventions
    assert (
        plan.payload.python_policy is None
        and plan.payload.resources.maximum_storage_bytes == 16 * 1024**2
    )
    request_file = tmp_path / "export-request.json"
    _write(request_file, _intent(path))
    payload_file = tmp_path / "workflow-payload.json"
    _write(payload_file, plan.payload)
    plan_file = tmp_path / "workflow-plan.json"
    root = tmp_path / "commands"
    logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    anchors: tuple[str, ...] = ()
    with ExitStack() as stack, _observe(stack, errors) as counts:
        ACTUAL_CLI_OBSERVATIONS.append(
            dict(
                scenario=scenario,
                path=str(tmp_path),
                actual_CLI_calls=logs,
                actual_method_errors=errors,
                actual_kernel_cases=cases,
                actual_native_counts=counts,
            )
        )
        common = _common(path, request_file, exports, plan_file, base, root, anchors)
        before = counts.copy()
        planned = _call(common, ["plan-request", "--payload", str(payload_file)], logs, counts)
        assert planned == plan.model_dump(mode="json")
        _assert_zero(counts, before)
        status = _call(common, ["status"], logs, counts)
        assert status is not None
        definition = status["definition"]
        store = root / "scientific-ablation-command-workflow-v1" / definition["sha256"]
        assert not store.exists()
        _assert_zero(counts, before)
        negatives = _negative_initial_planning(multirow_bound, tmp_path, request_file, logs, counts)
        for phase in scientific_ablation_workflow_phases_v1(plan)[:4]:
            assert phase.index == len(anchors)
            common = _common(path, request_file, exports, plan_file, base, root, anchors)
            name = f"phase-{phase.index:02d}"
            planning_request = tmp_path / (name + ".initial-planning.json")
            chunk_file = tmp_path / (name + ".initial-chunk.json")
            command_file = tmp_path / (name + ".original-command.json")
            kernel_root = store / "kernels" / phase.name
            before = counts.copy()
            _call(
                common,
                [
                    "kernel-request",
                    "--phase",
                    phase.name,
                    "--initial",
                    "--rows-per-call",
                    "3",
                    "--out",
                    str(planning_request),
                ],
                logs,
                counts,
            )
            _assert_zero(counts, before)
            chunk = _call(
                common,
                ["kernel-plan", "--request", str(planning_request), "--out", str(chunk_file)],
                logs,
                counts,
            )
            assert (
                chunk is not None
                and chunk["acknowledged_rows"] == 0
                and chunk["total_rows"] == 4
                and chunk["stop_row_index"] == 3
            )
            planning_counts = _delta(counts, before)
            assert planning_counts["qwen"] == planning_counts["teacher"] == 0
            if phase.kind == "capture":
                assert (
                    planning_counts["graph"] > 0
                    and planning_counts["model"] == planning_counts["tokenizer"] > 0
                )
            else:
                assert not any(planning_counts.values())
            assert (
                not kernel_root.exists()
                and not (store / f"phase-{phase.index:03d}.attempt.json").exists()
            )
            module, _, _ = _binding(phase)
            kernel = kernel_root / module._STORE / chunk["kernel_plan_sha256"]
            row_prefix = "row" if phase.kind == "capture" else "score"
            operation = "generate" if phase.kind == "capture" else "score"
            before = counts.copy()
            command = _call(
                common,
                [
                    "command-request",
                    "--phase",
                    phase.name,
                    "--expected-chunk",
                    str(chunk_file),
                    "--rows-per-call",
                    "3",
                    "--out",
                    str(command_file),
                ],
                logs,
                counts,
            )
            assert command is not None
            _assert_zero(counts, before)
            called: list[int] = []
            published: list[int] = []
            aggregates: list[str] = []
            signalled: list[int] = []
            publish = module._publish

            def published_original(
                stage: Any,
                target: Path,
                files: Any,
                claim: Any,
                *,
                _original: Any = publish,
                _row_prefix: str = row_prefix,
                published: list[int] = published,
                signalled: list[int] = signalled,
                aggregates: list[str] = aggregates,
            ) -> None:
                _original(stage, target, files, claim)
                if target.name.startswith(_row_prefix + "-"):
                    index = int(target.name.split("-")[1])
                    published.append(index)
                    if scenario == "recovery" and index == 0 and not signalled:
                        assert (target / "ack.json").is_file()
                        signalled.append(index)
                        raise KeyboardInterrupt(
                            "Genuine CLI: first row ACK committed before interruption"
                        )
                elif target.name == "complete":
                    aggregates.append(str(target))

            with ExitStack() as phase_stack:
                phase_stack.enter_context(patch.object(module, "_publish", new=published_original))
                if phase.kind == "capture":
                    cls = (
                        IrisScientificAblationHeldoutV1
                        if phase.stream_index is None
                        else IrisScientificAblationControlledHeldoutV1
                    )
                    execute = cls.execute_row

                    def row(
                        self: Any,
                        index: int,
                        *,
                        _original: Any = execute,
                        called: list[int] = called,
                    ) -> Any:
                        called.append(index)
                        return _original(self, index)

                    phase_stack.enter_context(patch.object(cls, "execute_row", new=row))
                else:
                    execute_score = module._score_row

                    def score_row(
                        inputs: Any,
                        index: int,
                        *,
                        _original: Any = execute_score,
                        called: list[int] = called,
                    ) -> Any:
                        called.append(index)
                        return _original(inputs, index)

                    phase_stack.enter_context(patch.object(module, "_score_row", new=score_row))
                original = _call(
                    common,
                    [operation, "--command", str(command_file)],
                    logs,
                    counts,
                    expected=130 if scenario == "recovery" else 0,
                )
                if scenario == "recovery":
                    assert (
                        original is None
                        and called == published == signalled == [0]
                        and not aggregates
                    )
                    pins0 = _row_pins(kernel / (row_prefix + "-00000"))
                    before = counts.copy()
                    tree_before = _tree(store)
                    _call(
                        common,
                        [operation, "--command", str(command_file)],
                        logs,
                        counts,
                        expected=2,
                    )
                    _assert_zero(counts, before)
                    assert _tree(store) == tree_before
                    partial = _call(
                        common,
                        ["recover", "--operation", operation, "--command", str(command_file)],
                        logs,
                        counts,
                    )
                    assert (
                        partial is not None
                        and partial["original_return"] == "unavailable_not_reconstructed"
                    )
                    assert _row_pins(kernel / (row_prefix + "-00000")) == pins0
                    assert {
                        p.name
                        for p in (
                            store / "handoffs" / command["sha256"][:2] / command["sha256"]
                        ).iterdir()
                    } == {"intent.json"}
                else:
                    assert original is not None
                    partial = original
                    pins0 = _row_pins(kernel / (row_prefix + "-00000"))
                snapshot = partial["kernel_progress"]
                assert snapshot["acknowledged_rows"] == 3 and snapshot["total_rows"] == 4
                assert called == published == [0, 1, 2] and not aggregates
                assert not (kernel / (row_prefix + "-00003")).exists()
                before = counts.copy()
                tree_before = _tree(store)
                if scenario == "recovery":
                    same = _call(
                        common,
                        [
                            "recover",
                            "--operation",
                            operation,
                            "--command",
                            str(command_file),
                            "--expected-operator-ack-sha256",
                            partial["operator_ack"]["sha256"],
                        ],
                        logs,
                        counts,
                    )
                    assert same == partial
                else:
                    same = _call(
                        common,
                        [
                            operation,
                            "--command",
                            str(command_file),
                            "--expected-operator-ack-sha256",
                            partial["operator_ack"]["sha256"],
                        ],
                        logs,
                        counts,
                    )
                    assert (
                        same is not None
                        and same["delivery"] == "saved_original_data"
                        and same["kernel_progress"] == snapshot
                    )
                _assert_zero(counts, before)
                assert _tree(store) == tree_before
                retained_file = tmp_path / (name + ".retained-planning.json")
                final_chunk = tmp_path / (name + ".final-chunk.json")
                final_command = tmp_path / (name + ".final-command.json")
                before = counts.copy()
                retained = _call(
                    common,
                    [
                        "kernel-request",
                        "--phase",
                        phase.name,
                        "--rows-per-call",
                        "1",
                        "--expected-kernel-plan-sha256",
                        chunk["kernel_plan_sha256"],
                        "--expected-acknowledgement-head-sha256",
                        snapshot["acknowledgement_head_sha256"],
                        "--expected-acknowledged-rows",
                        "3",
                        "--out",
                        str(retained_file),
                    ],
                    logs,
                    counts,
                )
                assert retained is not None
                complete_chunk = _call(
                    common,
                    ["kernel-plan", "--request", str(retained_file), "--out", str(final_chunk)],
                    logs,
                    counts,
                )
                assert (
                    complete_chunk is not None
                    and complete_chunk["acknowledged_rows"] == 3
                    and complete_chunk["stop_row_index"] == 4
                )
                _assert_zero(counts, before)
                assert _tree(store) == tree_before
                # Well-formed wrong independent head: actual DATA validation,
                # not a checksum/parser failure or a mocked refusal.
                raw_request = IrisScientificAblationWorkflowKernelRequestV1.model_validate_json(
                    retained_file.read_bytes()
                )
                bad_request = build_scientific_ablation_workflow_kernel_request_v1(
                    ScientificAblationWorkflowKernelRequestPayloadV1.model_validate(
                        raw_request.payload.model_dump(mode="python")
                        | dict(expected_acknowledgement_head_sha256="f" * 64)
                    )
                )
                bad_file = tmp_path / (name + ".wrong-head.json")
                _write(bad_file, bad_request)
                before = counts.copy()
                _call(
                    common,
                    [
                        "kernel-plan",
                        "--request",
                        str(bad_file),
                        "--out",
                        str(tmp_path / (name + ".must-not-exist.json")),
                    ],
                    logs,
                    counts,
                    expected=2,
                )
                _assert_zero(counts, before)
                assert _tree(store) == tree_before
                final_intent = _call(
                    common,
                    [
                        "command-request",
                        "--phase",
                        phase.name,
                        "--expected-chunk",
                        str(final_chunk),
                        "--rows-per-call",
                        "1",
                        "--out",
                        str(final_command),
                    ],
                    logs,
                    counts,
                )
                assert final_intent is not None and final_intent["sha256"] != command["sha256"]
                finished = _call(common, [operation, "--command", str(final_command)], logs, counts)
                assert (
                    finished is not None and finished["kernel_progress"]["acknowledged_rows"] == 4
                )
                assert called == published == [0, 1, 2, 3] and aggregates == [
                    str(kernel / "complete")
                ]
                assert _row_pins(kernel / (row_prefix + "-00000")) == pins0
                old_anchors = anchors
                anchors = tuple(finished["progress"]["completed_phase_sha256"])
                assert anchors[:-1] == old_anchors and len(anchors) == phase.index + 1
                current = _common(path, request_file, exports, plan_file, base, root, anchors)
                before = counts.copy()
                final_tree = _tree(store)
                if scenario == "recovery":
                    replay = _call(
                        current,
                        [
                            "recover",
                            "--operation",
                            operation,
                            "--command",
                            str(command_file),
                            "--expected-operator-ack-sha256",
                            partial["operator_ack"]["sha256"],
                        ],
                        logs,
                        counts,
                    )
                    assert replay == partial
                else:
                    replay = _call(
                        current,
                        [
                            operation,
                            "--command",
                            str(command_file),
                            "--expected-operator-ack-sha256",
                            partial["operator_ack"]["sha256"],
                        ],
                        logs,
                        counts,
                    )
                    assert (
                        replay is not None
                        and replay["delivery"] == "saved_original_data"
                        and replay["kernel_progress"] == snapshot
                        and replay["progress"] == partial["progress"]
                    )
                _assert_zero(counts, before)
                assert _tree(store) == final_tree
                current_status = _call(current, ["status"], logs, counts)
                assert (
                    current_status is not None
                    and tuple(current_status["progress"]["completed_phase_sha256"]) == anchors
                )
            cases.append(
                dict(
                    phase=phase.model_dump(mode="json"),
                    initial_request=json.loads(planning_request.read_bytes()),
                    initial_chunk=chunk,
                    original_command=command,
                    partial_reply=partial,
                    initial_planning_counts=planning_counts,
                    retained_request=retained,
                    final_chunk=complete_chunk,
                    final_command=final_intent,
                    completed_reply=finished,
                    old_reply_after_physical_head4=replay,
                    actual_row_calls=called,
                    actual_published_rows=published,
                    actual_aggregate_publications=aggregates,
                    original_row0_pins=pins0,
                )
            )
        assert len(cases) == len(anchors) == 4
        (tmp_path / "actual-all4-workflow-planning-cli-witness.json").write_bytes(
            canonical_json_file_bytes(
                dict(
                    operator_path=str(path),
                    base=str(base),
                    export_root=str(exports),
                    workflow_path=str(store),
                    plan=plan.model_dump(mode="json"),
                    phase_anchors=anchors,
                    scenario=scenario,
                    actual_kernel_cases=cases,
                    actual_initial_planning_refusals=negatives,
                    actual_CLI_calls=logs,
                    actual_method_errors=errors,
                    actual_native_counts=counts,
                    real_original_source_factory_owners_rows=True,
                    primary_signal_only_not_hard_exit_cold_OS_or_race=True,
                    fixture_training_export_not_CLI_training_or_installed_or_scientific_acceptance=True,
                )
            )
        )
