"""5000-row FORMAT fixtures, not 5000 teacher calls, quality or scientific results.

A genuine tiny four-row measurement supplies schema-shaped headers only. All
expanded identities, NLLs, traces and probe classes below are explicitly synthetic.
No fabricated report is passed to the actual measurement service or published.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from iris.application import scientific_ablation_collapse as compact
from iris.application import scientific_ablation_diagnostics as ordinary
from iris.artifacts.scientific_ablation_capture_v1 import (
    IrisScientificAblationCaptureCompleteV1,
    IrisScientificAblationCapturePlanV1,
)
from iris.artifacts.scientific_ablation_collapse_v1 import (
    IrisScientificAblationCollapseV1,
    ScientificAblationCollapseMemberV1,
    ScientificAblationCollapsePayloadV1,
    ScientificAblationCollapseRowV1,
    scientific_ablation_collapse_groups_v1,
    scientific_ablation_collapse_witness_v1,
)
from iris.artifacts.scientific_ablation_diagnostics_v1 import (
    PROBE_NAMES,
    IrisScientificAblationDiagnosticsV1,
    ScientificAblationDiagnosticRowV1,
    ScientificAblationProbeObservationV1,
)
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
    IrisScientificAblationEvaluationRowV2,
    ScientificAblationEvaluationAckPayloadV2,
    ScientificAblationEvaluationRowPayloadV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import (
    IrisScientificAblationEvaluationV2,
    ScientificAblationScoredRowV2,
)
from iris.artifacts.scientific_ablation_generation_reconstruction_v1 import (
    IrisScientificAblationGenerationReconstructionV1,
)
from iris.artifacts.scientific_ablation_generation_source_v1 import (
    IrisScientificAblationGenerationSourceManifestV1,
)
from iris.artifacts.scientific_ablation_heldout_v1 import IrisScientificAblationHeldoutInputV1
from iris.artifacts.scientific_collapse_v1 import scientific_collapse_criterion_v1
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    ScientificFamilyQualityV1,
    scientific_quality_counts_v1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.workspace.trace import (
    WorkspaceCycleMetadataV1,
    WorkspaceTraceOptionsV1,
    WorkspaceTraceV1,
    summarize_workspace_traces_v1,
)
from tests.integration.test_scientific_ablation_collapse import _measure
from tests.integration.test_scientific_ablation_diagnostics import _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make

pytestmark = pytest.mark.integration
N = 5000


def _manifest[M: BaseModel](model: type[M], raw: dict[str, Any], domain: str) -> M:
    return model.model_validate_json(
        canonical_json_bytes(
            {"sha256": domain_sha256(domain, canonical_json_bytes(raw)), "payload": raw}
        )
    )


def _identity(label: str, index: int = 0) -> str:
    return domain_sha256(
        "iris/tests/synthetic-collapse-scale/v1",
        canonical_json_bytes({"label": label, "index": index}),
    )


def _trace(
    index: int, checkpoint: str, mode: str, seed: int, *, dense: bool = False
) -> WorkspaceTraceV1:
    """Valid numeric DTO stress; ordinary variant obeys empty/2/4 read occupancy.

    Not a neural trace. Dense variant stress-tests the broader DTO contract only,
    and is NOT called a reachable empty-initialized C3/R2 teacher trajectory.
    """
    cycles = []
    precise = 0.12345678901234568
    for number in (1, 2, 3):
        before, after = (8, 8) if dense else (2 * (number - 1), 2 * number)
        winners = tuple(
            sorted(((index + 2 * (number - 1)) % 4, (index + 2 * (number - 1) + 1) % 4))
        )
        slots = (2 * (number - 1), 2 * (number - 1) + 1)
        weights = (
            (precise,) * (before - 1) + (1 - precise * (before - 1),) + (0.0,) * (8 - before)
            if before
            else (0.0,) * 8
        )
        cycles.append(
            WorkspaceCycleMetadataV1(
                cycle=number,
                module_bids=(1.2345678901234568e-35,) * 8,
                module_confidence=(precise,) * 8,
                selected_modules=winners,
                write_modules=winners,
                selected_slots=slots,
                overwritten_slots=slots if dense else (),
                slot_ages=(number - 1,) * after + (0,) * (8 - after),
                slot_ttl=(1.2345678901234568e-10,) * after + (0.0,) * (8 - after),
                slot_norms=(1.2345678901234568e-25,) * after + (0.0,) * (8 - after),
                slot_sources=tuple(i % 4 if i < after else -1 for i in range(8)),
                slot_occupancy=tuple(i < after for i in range(8)),
                slot_write_counts=(1,) * after + (0,) * (8 - after),
                module_state_norms=(1.2345678901234568e-25,) * 8,
                workspace_cosine_matrix=tuple(
                    tuple(
                        0.0 if i >= after or j >= after else 1.0 if i == j else precise
                        for j in range(8)
                    )
                    for i in range(8)
                ),
                read_slot_sources=tuple(i % 4 if i < before else -1 for i in range(8)),
                read_slot_occupancy=tuple(i < before for i in range(8)),
                read_attention=(weights,) * 8,
            )
        )
    return WorkspaceTraceV1(
        options=WorkspaceTraceOptionsV1(
            mode="metadata", example_id=f"test-{index}", checkpoint_id=checkpoint, seed=seed
        ),
        num_modules=8,
        num_slots=8,
        state_dim=256,
        cycle_count=3,
        intervention_plan_sha256=mode,
        cycles=tuple(cycles),
    )


@pytest.fixture(scope="module")
def seed(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1]:
    root = tmp_path_factory.mktemp("real-four-row-header-seed")
    experiment = _experiment(_make(root / "case"), root / "scores")
    actual: list[IrisScientificAblationDiagnosticsV1] = []
    original = ordinary.measure_scientific_ablation_diagnostics_v1

    def capture(*args: Any, **kwargs: Any) -> IrisScientificAblationDiagnosticsV1:
        result = original(*args, **kwargs)
        actual.append(result)
        return result

    with patch.object(compact, "measure_scientific_ablation_diagnostics_v1", new=capture):
        measured = _measure(experiment, _heldout(experiment))
    assert len(actual) == 1 and measured.payload.teacher_executions == 4
    return measured, actual[0]


def _expanded(
    seed: tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1],
    *,
    dense: bool = False,
) -> tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1]:
    """Rebind ALL5000 format commitments, including exact score-file ack chain."""
    compact_seed, diagnostic_seed = seed
    old = compact_seed.payload
    cp = old.evaluation.payload.plan.payload
    source_raw = cp.source.payload.model_dump(mode="json")
    source_raw.update(purpose="scientific", training_examples=50000, validation_examples=5000)
    source = _manifest(
        IrisScientificAblationGenerationSourceManifestV1,
        source_raw,
        "iris/scientific-ablation-generation-source/v1/payload",
    )
    reconstruction_raw = cp.reconstruction.payload.model_dump(mode="json")
    reconstruction_raw.update(
        source_sha256=source.sha256, training_examples=50000, validation_examples=5000
    )
    reconstruction = _manifest(
        IrisScientificAblationGenerationReconstructionV1,
        reconstruction_raw,
        "iris/scientific-ablation-generation-reconstruction/v1/payload",
    )
    input_raw = cp.input.payload.model_dump(mode="json")
    input_raw.update(
        purpose="scientific",
        example_count=N,
        target_token_count=2 * N,
        source_sha256=source.sha256,
        reconstruction_sha256=reconstruction.sha256,
    )
    heldout = _manifest(
        IrisScientificAblationHeldoutInputV1, input_raw, "iris/scientific-ablation-heldout-input/v1"
    )
    expectations = []
    members = []
    task_indices = [0] * 2000 + [1] * 1000 + [2] * 1000 + [3] * 1000
    for index, family in enumerate(task_indices):
        expected = cp.rows[family].model_dump(mode="json")
        expected.update(
            row_index=index,
            example_id=f"test-{index}",
            example_sha256=_identity("example", index),
            prompt_sha256=_identity("prompt", index),
            cognitive_prompt_sha256=_identity("cognitive-prompt", index),
            target_token_count=2,
        )
        expectations.append(expected)
        members.append(
            ScientificAblationCollapseMemberV1(
                row_index=index,
                example_id=f"test-{index}",
                example_sha256=expected["example_sha256"],
                prompt_sha256=expected["prompt_sha256"],
                cognitive_prompt_sha256=expected["cognitive_prompt_sha256"],
                task_type=TASKS[family],
                difficulty=index % 3 + 1,
            )
        )
    capture_plan_raw = cp.model_dump(mode="json")
    capture_plan_raw.update(
        input=heldout.model_dump(mode="json"),
        source=source.model_dump(mode="json"),
        reconstruction=reconstruction.model_dump(mode="json"),
        teacher_graph_sha256=_identity("synthetic-geometry"),
        rows=expectations,
    )
    capture_plan = _manifest(
        IrisScientificAblationCapturePlanV1,
        capture_plan_raw,
        "iris/scientific-ablation-capture-plan/v1",
    )
    capture_raw = old.evaluation.payload.capture.payload.model_dump(mode="json")
    capture_raw.update(
        plan_sha256=capture_plan.sha256,
        input_sha256=heldout.sha256,
        example_count=N,
        target_token_count=2 * N,
        target_nll_sum=10.125 * N,
        acknowledgement_head_sha256=_identity("capture-ack", N - 1),
    )
    capture = _manifest(
        IrisScientificAblationCaptureCompleteV1,
        capture_raw,
        "iris/scientific-ablation-capture-complete/v1",
    )
    scores = []
    plan_expectations = []
    for index, family in enumerate(task_indices):
        raw = old.evaluation.payload.rows[family].model_dump(mode="json")
        raw.update(
            row_index=index,
            example_id=f"test-{index}",
            generation_example_sha256=members[index].example_sha256,
            capture_row_sha256=_identity("capture-row", index),
            capture_ack_sha256=_identity("capture-ack", index),
            target_token_count=2,
            target_nll_sum=10.125,
        )
        scores.append(ScientificAblationScoredRowV2.model_validate_json(canonical_json_bytes(raw)))
        plan_row = old.evaluation_plan.payload.rows[family].model_dump(mode="json")
        plan_row.update(
            row_index=index,
            capture_row_sha256=raw["capture_row_sha256"],
            capture_ack_sha256=raw["capture_ack_sha256"],
            target_nll_sum=10.125,
        )
        plan_expectations.append(plan_row)
    plan_raw = old.evaluation_plan.payload.model_dump(mode="json")
    plan_raw.update(
        generation_plan=capture_plan.model_dump(mode="json"),
        capture=capture.model_dump(mode="json"),
        rows=plan_expectations,
    )
    plan = _manifest(
        IrisScientificAblationEvaluationPlanV2,
        plan_raw,
        "iris/scientific-ablation-evaluation-plan/v2",
    )
    report_raw = old.evaluation.payload.model_dump(mode="json")
    report_raw.update(
        plan=capture_plan.model_dump(mode="json"),
        capture=capture.model_dump(mode="json"),
        rows=[row.model_dump(mode="json") for row in scores],
        overall=scientific_quality_counts_v1(tuple(scores)).model_dump(mode="json"),
        families=[
            ScientificFamilyQualityV1(
                task_type=task,
                counts=scientific_quality_counts_v1(
                    tuple(row for row in scores if row.task_type == task)
                ),
            ).model_dump(mode="json")
            for task in TASKS
        ],
    )
    report = _manifest(
        IrisScientificAblationEvaluationV2, report_raw, "iris/scientific-ablation-evaluation/v2"
    )
    diagnostic_rows = []
    compact_rows = []
    traces = []
    probe_vocabulary_sha = diagnostic_seed.payload.probe_vocabulary_sha256
    head = domain_sha256("iris/scientific-ablation-evaluation-ack/v2/genesis", plan.sha256.encode())
    for index, score in enumerate(scores):
        row_payload = ScientificAblationEvaluationRowPayloadV2(
            plan_sha256=plan.sha256, measurements=score
        )
        row_sha = domain_sha256(
            "iris/scientific-ablation-evaluation-row/v2",
            canonical_json_bytes(row_payload.model_dump(mode="json")),
        )
        content = canonical_json_file_bytes(
            IrisScientificAblationEvaluationRowV2(sha256=row_sha, payload=row_payload).model_dump(
                mode="json"
            )
        )
        ack = ScientificAblationEvaluationAckPayloadV2(
            plan_sha256=plan.sha256,
            row_index=index,
            previous_ack_sha256=head,
            row_sha256=row_sha,
            row_file_sha256=sha256(content).hexdigest(),
            row_file_bytes=len(content),
        )
        head = domain_sha256(
            "iris/scientific-ablation-evaluation-ack/v2",
            canonical_json_bytes(ack.model_dump(mode="json")),
        )
        trace = _trace(
            index,
            source.payload.best_checkpoint_sha256,
            source.payload.ablation_policy_sha256,
            old.run_seed,
            dense=dense,
        )
        traces.append(trace)
        probes = []
        for name, classes in zip(
            PROBE_NAMES, diagnostic_seed.payload.probe_class_counts, strict=True
        ):
            probes.append(
                ScientificAblationProbeObservationV1(
                    field=name,
                    class_count=classes,
                    predicted_class=0,
                    correct=True,
                    target_sha256=domain_sha256(
                        "iris/scientific-ablation-probe-target/v1",
                        canonical_json_bytes(
                            {
                                "example_sha256": score.generation_example_sha256,
                                "vocabulary_sha256": probe_vocabulary_sha,
                                "field": name,
                                "class_index": 0,
                            }
                        ),
                    ),
                )
            )
        diagnostic_rows.append(
            ScientificAblationDiagnosticRowV1(
                row_index=index,
                capture_row_sha256=score.capture_row_sha256,
                evaluation_row_sha256=row_sha,
                evaluation_ack_sha256=head,
                teacher_invocation_sha256=score.teacher_invocation_sha256,
                teacher_target_sha256=score.teacher_target_sha256,
                target_token_count=2,
                target_nll_sum=10.125,
                trace=trace,
                probes=tuple(probes),
            )
        )
        trace_sha, context, cycles = scientific_ablation_collapse_witness_v1(
            trace, expected_ablation_policy_sha256=source.payload.ablation_policy_sha256
        )
        compact_rows.append(
            ScientificAblationCollapseRowV1(
                row_index=index,
                capture_row_sha256=score.capture_row_sha256,
                evaluation_row_sha256=row_sha,
                evaluation_ack_sha256=head,
                teacher_invocation_sha256=score.teacher_invocation_sha256,
                teacher_target_sha256=score.teacher_target_sha256,
                target_token_count=2,
                target_nll_sum=10.125,
                metadata_trace_sha256=trace_sha,
                context=context,
                cycles=cycles,
            )
        )
    complete_raw = old.evaluation_complete.payload.model_dump(mode="json")
    complete_raw.update(
        plan_sha256=plan.sha256,
        capture_sha256=capture.sha256,
        evaluation_sha256=report.sha256,
        acknowledgement_head_sha256=head,
        example_count=N,
    )
    complete = _manifest(
        IrisScientificAblationEvaluationCompleteV2,
        complete_raw,
        "iris/scientific-ablation-evaluation-complete/v2",
    )
    diagnostic_raw = diagnostic_seed.payload.model_dump(mode="json")
    diagnostic_raw.update(
        evaluation_plan=plan.model_dump(mode="json"),
        evaluation=report.model_dump(mode="json"),
        evaluation_complete=complete.model_dump(mode="json"),
        teacher_graph_sha256=capture_plan.payload.teacher_graph_sha256,
        teacher_executions=N,
        trace_scalar_bound=5_535_000,
        rows=[row.model_dump(mode="json") for row in diagnostic_rows],
        workspace=summarize_workspace_traces_v1(tuple(traces)).model_dump(mode="json"),
        probe_correct_counts=[N] * 4,
        source_20_4="met",
    )
    diagnostic = _manifest(
        IrisScientificAblationDiagnosticsV1,
        diagnostic_raw,
        "iris/scientific-ablation-diagnostics/v1",
    )
    groups = scientific_ablation_collapse_groups_v1(
        tuple(members), tuple(compact_rows), modules=8, slots=8
    )
    summary = groups[0].summary
    assert summary is not None
    payload = ScientificAblationCollapsePayloadV1(
        evaluation_plan=plan,
        evaluation=report,
        evaluation_complete=complete,
        policy=old.policy,
        policy_sha256=old.policy_sha256,
        membership_sha256=domain_sha256(
            compact._MEMBERS_DOMAIN,
            canonical_json_bytes(tuple(member.model_dump(mode="json") for member in members)),
        ),
        ordinary_diagnostics_sha256=diagnostic.sha256,
        run_seed=old.run_seed,
        teacher_graph_sha256=capture_plan.payload.teacher_graph_sha256,
        num_modules=8,
        num_slots=8,
        state_dim=256,
        cycle_count=3,
        writers_per_cycle=2,
        members=tuple(members),
        rows=tuple(compact_rows),
        teacher_executions=N,
        trace_scalar_bound=5_535_000,
        groups=groups,
        source_20_4=scientific_collapse_criterion_v1(purpose="scientific", summary=summary),
    )
    result = IrisScientificAblationCollapseV1(
        sha256=domain_sha256(
            compact._DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
        ),
        payload=payload,
    )
    return result, diagnostic


@pytest.fixture(scope="module")
def expanded(
    seed: tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1],
) -> tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1]:
    return _expanded(seed)


def test_full_5000_schema_ack_chain_groups_and_compact_byte_envelope(
    expanded: tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1],
    tmp_path: Path,
) -> None:
    collapse, diagnostic = expanded
    p = collapse.payload
    assert len(p.rows) == len(p.members) == p.teacher_executions == 5000
    assert p.trace_scalar_bound == 5_535_000 and len(p.groups) == 20
    assert tuple(group.examples for group in p.groups[1:5]) == (2000, 1000, 1000, 1000)
    assert (
        p.evaluation_complete.payload.acknowledgement_head_sha256
        == p.rows[-1].evaluation_ack_sha256
    )
    content = compact._checked(collapse)
    assert len(content) <= 64 * 1024**2
    assert IrisScientificAblationCollapseV1.model_validate_json(content) == collapse
    ordinary_content = canonical_json_file_bytes(diagnostic.model_dump(mode="json"))
    ordinary_fits = len(ordinary_content) <= 64 * 1024**2
    if ordinary_fits:
        assert ordinary._checked(diagnostic) == ordinary_content
    else:
        with pytest.raises(ValueError):
            ordinary._checked(diagnostic)
    (tmp_path / "synthetic-collapse.json").write_bytes(content)
    # Raw FORMAT witness for the independent observer, not production publication.
    (tmp_path / "synthetic-ordinary.json").write_bytes(ordinary_content)
    metrics = {
        "kind": "iris-synthetic-5000-collapse-envelope-v1",
        "examples": 5000,
        "teacher_forwards_for_expanded_rows": 0,
        "compact_bytes": len(content),
        "ordinary_metadata_bytes": len(ordinary_content),
        "cap_bytes": 64 * 1024**2,
        "ordinary_fits": ordinary_fits,
        "scientific_acceptance": "not_assessed",
        "trace_shape": "synthetic-empty-2-4-read-and-2-4-6-postwrite",
        "full_capture_physical_execution_proof": False,
    }
    (tmp_path / "envelope.json").write_bytes(canonical_json_file_bytes(metrics))
    print(json.dumps(metrics))


@pytest.mark.parametrize(
    "problem", ("last_ack", "last_row", "last_nll", "last_difficulty", "last_context", "threshold")
)
def test_late_5000th_row_and_fixed_gate_forgeries_are_not_hidden(
    expanded: tuple[IrisScientificAblationCollapseV1, IrisScientificAblationDiagnosticsV1],
    problem: str,
) -> None:
    raw = expanded[0].payload.model_dump(mode="json")
    if problem == "last_ack":
        raw["rows"][-1]["evaluation_ack_sha256"] = "0" * 64
    elif problem == "last_row":
        raw["rows"] = raw["rows"][:-1]
    elif problem == "last_nll":
        raw["rows"][-1]["target_nll_sum"] += 1
    elif problem == "last_difficulty":
        raw["members"][-1]["difficulty"] = 4
    elif problem == "last_context":
        raw["rows"][-1]["context"]["ablation_policy_sha256"] = "0" * 64
    else:
        raw["source_20_4"]["minimum_effective_slots"] = 1
    with pytest.raises(ValueError):
        ScientificAblationCollapsePayloadV1.model_validate_json(canonical_json_bytes(raw))
