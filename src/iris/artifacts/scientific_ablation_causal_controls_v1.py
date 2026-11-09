"""Source-bound actual empty causal controls, NOT writer effects or science.

This is the shared ordinary/combined-control phase of the complete experiment.
Integrity references alone do not attest execution or original Parquet contents.
"""

from hashlib import sha256
from math import isclose
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.ablations.controls import CognitiveAblationControlPlanV1, ablation_control_plan_sha256_v1
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    IrisScientificAblationCausalDesignV1,
    ScientificAblationMessageCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
    IrisScientificAblationEvaluationRowV2,
    ScientificAblationEvaluationAckPayloadV2,
    ScientificAblationEvaluationRowPayloadV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.interventions.messages import ModuleMessageSnapshotV1, WorkspaceMessagePlanV1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.training.ablation_controls import controlled_teacher_invocation_sha256_v1
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.workspace.trace import WorkspaceTraceV1

DOMAIN = "iris/scientific-ablation-causal-controls/v1"


def causal_control_report_byte_bound_v1(
    design: IrisScientificAblationCausalDesignV1,
    plan: IrisScientificAblationEvaluationPlanV2,
    evaluation: IrisScientificAblationEvaluationV2,
    complete: IrisScientificAblationEvaluationCompleteV2,
) -> int:
    """Conservative canonical REPORT bytes, separate from all model-pass budgets."""
    p, g = design.payload, design.payload.geometry
    base = len(
        canonical_json_bytes(
            dict(
                design=design.model_dump(mode="json"),
                evaluation_plan=plan.model_dump(mode="json"),
                evaluation=evaluation.model_dump(mode="json"),
                evaluation_complete=complete.model_dump(mode="json"),
            )
        )
    )
    numeric = (
        p.budget.ordinary_trace_scalars
        + p.budget.full_trace_scalars
        + p.budget.message_scalars_per_collection
    )
    proposals = (
        g.num_modules * g.cycle_count
        if type(p.policy) is ScientificAblationMessageCausalPolicyV1
        else 0
    )
    # 32 bytes covers finite float/int/bool scalars and separators; explicit
    # structural/context/snapshot reserves cover bounded field names/IDs/hashes.
    return base + 65536 + len(p.selected_row_indices) * (32768 + 32 * numeric + 1024 * proposals)


def _empty_plan(design: IrisScientificAblationCausalDesignV1) -> CognitiveAblationControlPlanV1:
    p = design.payload
    if type(p.policy) is ScientificAblationMessageCausalPolicyV1:
        return CognitiveAblationControlPlanV1(
            ablation=p.ablation,
            control=WorkspaceMessagePlanV1(),
            maximum_message_scalars=p.policy.maximum_message_scalars,
        )
    return CognitiveAblationControlPlanV1(
        ablation=p.ablation, control=WorkspaceInterventionPlanV1(interventions=())
    )


def _score_bindings(
    plan: IrisScientificAblationEvaluationPlanV2,
    evaluation: IrisScientificAblationEvaluationV2,
    complete: IrisScientificAblationEvaluationCompleteV2,
) -> tuple[tuple[str, str], ...]:
    """Reconstruct the COMPLETE physical score chain, not just selected rows."""
    for item, domain in (
        (plan, "iris/scientific-ablation-evaluation-plan/v2"),
        (evaluation, "iris/scientific-ablation-evaluation/v2"),
        (complete, "iris/scientific-ablation-evaluation-complete/v2"),
    ):
        if item.sha256 != domain_sha256(
            domain, canonical_json_bytes(item.payload.model_dump(mode="json"))
        ):
            raise ValueError("Original score plan/report/complete checksum differs")
    if (
        evaluation.payload.plan != plan.payload.generation_plan
        or evaluation.payload.capture != plan.payload.capture
    ):
        raise ValueError("Original complete score source/capture differs")
    head = domain_sha256("iris/scientific-ablation-evaluation-ack/v2/genesis", plan.sha256.encode())
    result = []
    for index, score in enumerate(evaluation.payload.rows):
        payload = ScientificAblationEvaluationRowPayloadV2(
            plan_sha256=plan.sha256, measurements=score
        )
        row_sha = domain_sha256(
            "iris/scientific-ablation-evaluation-row/v2",
            canonical_json_bytes(payload.model_dump(mode="json")),
        )
        content = canonical_json_file_bytes(
            IrisScientificAblationEvaluationRowV2(sha256=row_sha, payload=payload).model_dump(
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
        result.append((row_sha, head))
    c = complete.payload
    if (
        c.plan_sha256,
        c.capture_sha256,
        c.evaluation_sha256,
        c.acknowledgement_head_sha256,
        c.example_count,
    ) != (plan.sha256, plan.payload.capture.sha256, evaluation.sha256, head, len(result)):
        raise ValueError("Entire original score acknowledgement inventory differs")
    return tuple(result)


class ScientificAblationCausalControlRowV1(_Strict):
    row_index: int = Field(ge=0, le=4999)
    capture_row_sha256: Digest
    evaluation_row_sha256: Digest
    evaluation_ack_sha256: Digest
    ordinary_teacher_invocation_sha256: Digest
    target_sha256: Digest
    target_token_count: int = Field(ge=2, le=65536)
    target_nll_sum: float = Field(ge=0.0)
    ordinary_mean_target_nll: float = Field(ge=0.0)
    controlled_mean_target_nll: float = Field(ge=0.0)
    ordinary_trace: WorkspaceTraceV1
    controlled_trace: WorkspaceTraceV1
    controlled_invocation_sha256: Digest
    messages: tuple[ModuleMessageSnapshotV1, ...] = Field(max_length=4096)


class ScientificAblationCausalControlsPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-causal-controls-v1"] = (
        "iris-scientific-ablation-causal-controls-v1"
    )
    scope: Literal["source-owned-empty-controls-not-causal-effects-or-answer-quality"] = (
        "source-owned-empty-controls-not-causal-effects-or-answer-quality"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    design: IrisScientificAblationCausalDesignV1
    evaluation_plan: IrisScientificAblationEvaluationPlanV2
    evaluation: IrisScientificAblationEvaluationV2
    evaluation_complete: IrisScientificAblationEvaluationCompleteV2
    run_seed: int = Field(ge=0, le=2**63 - 1)
    ordinary_teacher_graph_sha256: Digest
    controlled_teacher_graph_sha256: Digest
    sidecar_sha256: Digest
    empty_plan_sha256: Digest
    actual_teacher_executions: int = Field(ge=2, le=1536)
    actual_trace_proposal_scalar_bound: int = Field(ge=1)
    maximum_report_bytes: int = Field(ge=1, le=64 * 1024**2)
    report_byte_preflight_bound: int = Field(ge=1)
    rows: tuple[ScientificAblationCausalControlRowV1, ...] = Field(min_length=1, max_length=768)

    @model_validator(mode="after")
    def original_controls(self) -> Self:
        d, g = self.design.payload, self.design.payload.geometry
        source = self.evaluation.payload.plan.payload.source.payload
        bindings = _score_bindings(self.evaluation_plan, self.evaluation, self.evaluation_complete)
        if (
            len(d.members) != len(self.evaluation.payload.rows)
            or d.purpose != source.purpose
            or d.ablation != source.ablation
            or d.ablation_policy_sha256 != source.ablation_policy_sha256
            or tuple(r.row_index for r in self.rows) != d.selected_row_indices
            or self.empty_plan_sha256 != ablation_control_plan_sha256_v1(_empty_plan(self.design))
            or self.actual_teacher_executions != 2 * len(self.rows)
            or self.report_byte_preflight_bound
            != causal_control_report_byte_bound_v1(
                self.design, self.evaluation_plan, self.evaluation, self.evaluation_complete
            )
            or self.report_byte_preflight_bound > self.maximum_report_bytes
        ):
            raise ValueError("Original native mode/source/selected control inventory differs")
        message = type(d.policy) is ScientificAblationMessageCausalPolicyV1
        scalar_bound = len(self.rows) * (
            d.budget.ordinary_trace_scalars
            + d.budget.full_trace_scalars
            + d.budget.message_scalars_per_collection
        )
        if self.actual_trace_proposal_scalar_bound != scalar_bound:
            raise ValueError(
                "Actual control phase scalar count differs from complete planned budget"
            )
        for member, expected in zip(
            d.members, self.evaluation.payload.plan.payload.rows, strict=True
        ):
            if (
                member.example_id,
                member.example_sha256,
                member.prompt_sha256,
                member.task_type,
            ) != (
                expected.example_id,
                expected.example_sha256,
                expected.prompt_sha256,
                expected.task_type,
            ):
                raise ValueError("ALL original design members must match generation source")
        geometry = (g.num_modules, g.num_slots, g.state_dim, g.cycle_count)
        for row in self.rows:
            index = row.row_index
            member, score = d.members[index], self.evaluation.payload.rows[index]
            if (
                (row.evaluation_row_sha256, row.evaluation_ack_sha256) != bindings[index]
                or row.capture_row_sha256 != score.capture_row_sha256
                or row.ordinary_teacher_invocation_sha256 != score.teacher_invocation_sha256
                or row.target_sha256 != score.teacher_target_sha256
                or row.target_token_count != score.target_token_count
                or row.target_nll_sum != score.target_nll_sum
                or row.ordinary_mean_target_nll != row.controlled_mean_target_nll
                # Saved scientific NLL is the float32 mean-times-EOS-count product.
                # This is a rounding identity bound, not an effect tolerance.
                or not isclose(
                    row.target_nll_sum,
                    row.ordinary_mean_target_nll * row.target_token_count,
                    rel_tol=2e-7,
                    abs_tol=1e-7,
                )
                or row.controlled_invocation_sha256
                != controlled_teacher_invocation_sha256_v1(
                    plan_sha256=self.empty_plan_sha256,
                    prompt_sha256=member.cognitive_prompt_sha256,
                    target_sha256=row.target_sha256,
                    teacher_graph_sha256=self.controlled_teacher_graph_sha256,
                    sidecar_state_sha256=self.sidecar_sha256,
                    supervised_token_count=row.target_token_count,
                )
            ):
                raise ValueError("Actual ordinary/controlled EOS NLL and score binding differs")
            for trace, mode, plan_sha in (
                (row.ordinary_trace, d.policy.ordinary_trace_mode, d.ablation_policy_sha256),
                (row.controlled_trace, "full", self.empty_plan_sha256),
            ):
                if (
                    trace.options.example_id != member.example_id
                    or trace.options.checkpoint_id != source.best_checkpoint_sha256
                    or trace.options.seed != self.run_seed
                    or trace.options.mode != mode
                    or trace.options.maximum_scalars != d.policy.maximum_trace_scalars
                    or trace.intervention_plan_sha256 != plan_sha
                    or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
                    != geometry
                    or any(len(c.write_modules) != g.writers_per_cycle for c in trace.cycles)
                ):
                    raise ValueError("Actual control trace has another mode/graph/checkpoint")
            if tuple(c.model_dump(exclude={"full"}) for c in row.ordinary_trace.cycles) != tuple(
                c.model_dump(exclude={"full"}) for c in row.controlled_trace.cycles
            ) or (
                d.policy.ordinary_trace_mode == "full"
                and row.ordinary_trace.cycles != row.controlled_trace.cycles
            ):
                raise ValueError("Empty control is not bitwise ordinary cycle parity")
            if (message and len(row.messages) != g.num_modules * g.cycle_count) or (
                not message and row.messages
            ):
                raise ValueError("Complete actual control proposal coverage required")
            for offset, snapshot in enumerate(row.messages):
                if (
                    (snapshot.cycle, snapshot.module)
                    != (offset // g.num_modules + 1, offset % g.num_modules)
                    or snapshot.example_id != member.example_id
                    or snapshot.prompt_sha256 != member.cognitive_prompt_sha256
                    or snapshot.teacher_graph_sha256 != self.controlled_teacher_graph_sha256
                    or snapshot.sidecar_state_sha256 != self.sidecar_sha256
                    or len(snapshot.values) != g.state_dim
                ):
                    raise ValueError("Actual same-mode control message context differs")
        return self


class IrisScientificAblationCausalControlsV1(_Strict):
    sha256: Digest
    payload: ScientificAblationCausalControlsPayloadV1

    @model_validator(mode="after")
    def checksum(self) -> Self:
        if self.sha256 != domain_sha256(
            DOMAIN, canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Distinct source-owned causal control checksum differs")
        return self
