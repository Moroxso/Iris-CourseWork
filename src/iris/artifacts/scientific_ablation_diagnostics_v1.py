"""Actual ordinary ablation diagnostic witnesses, never baseline or causal evidence."""

from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
    IrisScientificAblationEvaluationRowV2,
    ScientificAblationEvaluationAckPayloadV2,
    ScientificAblationEvaluationRowPayloadV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.workspace.trace import (
    WorkspaceDiagnosticsV1,
    WorkspaceTraceV1,
    _scalar_bound,
    summarize_workspace_traces_v1,
)

type ProbeName = Literal["active_constraint", "operation", "conflicting_conditions", "result_type"]
PROBE_NAMES: tuple[ProbeName, ...] = (
    "active_constraint",
    "operation",
    "conflicting_conditions",
    "result_type",
)


class ScientificAblationDiagnosticPolicyV1(_Strict):
    trace_mode: Literal["metadata", "full"] = "metadata"
    coverage: Literal["complete_original_test_order"] = "complete_original_test_order"
    maximum_teacher_executions: int = Field(default=5000, ge=1, le=5000)
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_total_trace_scalars: int = Field(default=16_000_000, ge=1, le=64_000_000)


class ScientificAblationProbeObservationV1(_Strict):
    field: Literal["active_constraint", "operation", "conflicting_conditions", "result_type"]
    class_count: int = Field(ge=1, le=128)
    predicted_class: int = Field(ge=0, le=127)
    target_sha256: Digest
    correct: bool

    @model_validator(mode="after")
    def bounded_class(self) -> Self:
        if self.predicted_class >= self.class_count:
            raise ValueError("Actual probe prediction exceeds original class domain")
        return self


class ScientificAblationDiagnosticRowV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    capture_row_sha256: Digest
    evaluation_row_sha256: Digest
    evaluation_ack_sha256: Digest
    teacher_invocation_sha256: Digest
    teacher_target_sha256: Digest
    target_token_count: int = Field(ge=2, le=65536)
    target_nll_sum: float = Field(ge=0.0)
    trace: WorkspaceTraceV1
    probes: tuple[ScientificAblationProbeObservationV1, ...] = Field(min_length=4, max_length=4)


class ScientificAblationDiagnosticsPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-diagnostics-v1"] = (
        "iris-scientific-ablation-diagnostics-v1"
    )
    scope: Literal["ordinary_teacher_controls_not_causal_message_or_launch_approval"] = (
        "ordinary_teacher_controls_not_causal_message_or_launch_approval"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    probe_scope: Literal["frozen_native_probes_final_cycle_mean_all_slots"] = (
        "frozen_native_probes_final_cycle_mean_all_slots"
    )
    evaluation_plan: IrisScientificAblationEvaluationPlanV2
    evaluation: IrisScientificAblationEvaluationV2
    evaluation_complete: IrisScientificAblationEvaluationCompleteV2
    policy: ScientificAblationDiagnosticPolicyV1
    policy_sha256: Digest
    run_seed: int = Field(ge=0, le=2**63 - 1)
    teacher_graph_sha256: Digest
    probe_configuration_sha256: Digest
    probe_vocabulary_sha256: Digest
    probe_class_counts: tuple[int, int, int, int]
    teacher_executions: int = Field(ge=1, le=5000)
    trace_scalar_bound: int = Field(ge=1)
    rows: tuple[ScientificAblationDiagnosticRowV1, ...] = Field(min_length=1, max_length=5000)
    workspace: WorkspaceDiagnosticsV1
    probe_correct_counts: tuple[int, int, int, int]
    source_20_4: Literal["not_assessed_smoke", "not_applicable_geometry", "met", "not_met"]

    @model_validator(mode="after")
    def complete_actual_controls(self) -> Self:
        plan, report = self.evaluation_plan.payload, self.evaluation.payload
        if (
            self.evaluation_plan.sha256
            != domain_sha256(
                "iris/scientific-ablation-evaluation-plan/v2",
                canonical_json_bytes(plan.model_dump(mode="json")),
            )
            or self.evaluation.sha256
            != domain_sha256(
                "iris/scientific-ablation-evaluation/v2",
                canonical_json_bytes(report.model_dump(mode="json")),
            )
            or self.policy_sha256
            != domain_sha256(
                "iris/scientific-ablation-diagnostic-policy/v1",
                canonical_json_bytes(self.policy.model_dump(mode="json")),
            )
            or report.plan != plan.generation_plan
            or report.capture != plan.capture
            or len(self.rows) != len(report.rows)
            or self.teacher_executions != len(self.rows)
            or self.teacher_executions > self.policy.maximum_teacher_executions
            or any(not 1 <= count <= 128 for count in self.probe_class_counts)
            or self.probe_class_counts[2] != 2
        ):
            raise ValueError("Complete source/mode/capture/evaluation/policy inventory differs")
        source = report.plan.payload.source.payload
        head = domain_sha256(
            "iris/scientific-ablation-evaluation-ack/v2/genesis",
            self.evaluation_plan.sha256.encode(),
        )
        for index, (row, score) in enumerate(zip(self.rows, report.rows, strict=True)):
            record_payload = ScientificAblationEvaluationRowPayloadV2(
                plan_sha256=self.evaluation_plan.sha256,
                measurements=score,
            )
            record_sha = domain_sha256(
                "iris/scientific-ablation-evaluation-row/v2",
                canonical_json_bytes(record_payload.model_dump(mode="json")),
            )
            content = canonical_json_file_bytes(
                IrisScientificAblationEvaluationRowV2(
                    sha256=record_sha,
                    payload=record_payload,
                ).model_dump(mode="json")
            )
            ack_payload = ScientificAblationEvaluationAckPayloadV2(
                plan_sha256=self.evaluation_plan.sha256,
                row_index=index,
                previous_ack_sha256=head,
                row_sha256=record_sha,
                row_file_sha256=sha256(content).hexdigest(),
                row_file_bytes=len(content),
            )
            head = domain_sha256(
                "iris/scientific-ablation-evaluation-ack/v2",
                canonical_json_bytes(ack_payload.model_dump(mode="json")),
            )
            options = row.trace.options
            if (
                row.row_index != index
                or row.evaluation_row_sha256 != record_sha
                or row.evaluation_ack_sha256 != head
                or row.capture_row_sha256 != score.capture_row_sha256
                or row.teacher_invocation_sha256 != score.teacher_invocation_sha256
                or row.teacher_target_sha256 != score.teacher_target_sha256
                or row.target_token_count != score.target_token_count
                or row.target_nll_sum != score.target_nll_sum
                or options.example_id != score.example_id
                or options.checkpoint_id != source.best_checkpoint_sha256
                or options.seed != self.run_seed
                or options.mode != self.policy.trace_mode
                or options.maximum_scalars != self.policy.maximum_trace_scalars
                or row.trace.intervention_plan_sha256 != source.ablation_policy_sha256
                or row.trace.cycle_count != report.plan.payload.cycles
                or tuple(probe.field for probe in row.probes) != PROBE_NAMES
                or tuple(probe.class_count for probe in row.probes) != self.probe_class_counts
            ):
                raise ValueError("Actual mode-bound trace/probe/EOS NLL row differs")
            for probe in row.probes:
                targets = tuple(
                    target
                    for target in range(probe.class_count)
                    if probe.target_sha256
                    == domain_sha256(
                        "iris/scientific-ablation-probe-target/v1",
                        canonical_json_bytes(
                            {
                                "example_sha256": score.generation_example_sha256,
                                "vocabulary_sha256": self.probe_vocabulary_sha256,
                                "field": probe.field,
                                "class_index": target,
                            }
                        ),
                    )
                )
                if len(targets) != 1 or probe.correct != (probe.predicted_class == targets[0]):
                    raise ValueError("Probe target class domain/correctness differs")
        complete = self.evaluation_complete.payload
        if (
            self.evaluation_complete.sha256
            != domain_sha256(
                "iris/scientific-ablation-evaluation-complete/v2",
                canonical_json_bytes(complete.model_dump(mode="json")),
            )
            or complete.plan_sha256 != self.evaluation_plan.sha256
            or complete.capture_sha256 != plan.capture.sha256
            or complete.evaluation_sha256 != self.evaluation.sha256
            or complete.acknowledgement_head_sha256 != head
            or complete.example_count != len(self.rows)
        ):
            raise ValueError("Original complete score acknowledgement chain differs")
        first = self.rows[0].trace
        bound = len(self.rows) * _scalar_bound(
            first.num_modules,
            first.num_slots,
            first.state_dim,
            first.cycle_count,
            self.policy.trace_mode == "full",
        )
        if (
            bound != self.trace_scalar_bound
            or bound > self.policy.maximum_total_trace_scalars
            or self.workspace
            != summarize_workspace_traces_v1(tuple(row.trace for row in self.rows))
            or self.probe_correct_counts
            != tuple(sum(row.probes[index].correct for row in self.rows) for index in range(4))
        ):
            raise ValueError("Actual pooled counts/probe denominators/budget differ")
        outcome = (
            "not_assessed_smoke"
            if source.purpose == "smoke"
            else "not_applicable_geometry"
            if (first.num_modules, first.num_slots) != (8, 8)
            else "met"
            if self.workspace.effective_active_modules >= 4
            and self.workspace.effective_active_slots >= 4
            else "not_met"
        )
        if self.source_20_4 != outcome:
            raise ValueError("Fixed full-test source20.4 assessment differs")
        return self


class IrisScientificAblationDiagnosticsV1(_Strict):
    sha256: Digest
    payload: ScientificAblationDiagnosticsPayloadV1


__all__ = [
    "IrisScientificAblationDiagnosticsV1",
    "ScientificAblationDiagnosticPolicyV1",
    "ScientificAblationDiagnosticRowV1",
    "ScientificAblationDiagnosticsPayloadV1",
    "ScientificAblationProbeObservationV1",
]
