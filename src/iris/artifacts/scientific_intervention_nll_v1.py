"""Hash-only paired teacher NLL companion; never a generated-answer quality score."""

from math import fsum
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.schemas import SyntheticTask


class ScientificInterventionNllRowV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    example_sha256: Digest
    prompt_sha256: Digest
    target_sha256: Digest
    bound_plan_sha256: Digest
    task_type: SyntheticTask
    difficulty: int = Field(ge=1, le=3)
    target_token_count: int = Field(ge=1)
    control_nll_sum: float = Field(ge=0)
    intervened_nll_sum: float = Field(ge=0)


class ScientificPairedNllV1(_Strict):
    examples: int = Field(ge=0)
    target_token_count: int = Field(ge=0)
    control_nll_sum: float = Field(ge=0)
    intervened_nll_sum: float = Field(ge=0)
    control_mean_nll: float | None = Field(ge=0)
    intervened_mean_nll: float | None = Field(ge=0)
    delta_mean_nll: float | None

    @model_validator(mode="after")
    def exact_rates(self) -> Self:
        count = self.target_token_count
        control = self.control_nll_sum / count if count else None
        changed = self.intervened_nll_sum / count if count else None
        if (
            (self.examples == 0) != (count == 0)
            or count < self.examples
            or (count == 0 and (self.control_nll_sum != 0 or self.intervened_nll_sum != 0))
            or self.control_mean_nll != control
            or self.intervened_mean_nll != changed
            or self.delta_mean_nll
            != (None if control is None or changed is None else changed - control)
        ):
            raise ValueError("Paired NLL must use target-token sums/counts; empty rates are null")
        return self


def scientific_paired_nll_v1(
    rows: tuple[ScientificInterventionNllRowV1, ...],
) -> ScientificPairedNllV1:
    count = sum(row.target_token_count for row in rows)
    control = fsum(row.control_nll_sum for row in rows)
    changed = fsum(row.intervened_nll_sum for row in rows)
    return ScientificPairedNllV1(
        examples=len(rows),
        target_token_count=count,
        control_nll_sum=control,
        intervened_nll_sum=changed,
        control_mean_nll=control / count if count else None,
        intervened_mean_nll=changed / count if count else None,
        delta_mean_nll=changed / count - control / count if count else None,
    )


class ScientificNllGroupV1(_Strict):
    task_type: SyntheticTask | None
    difficulty: int | None = Field(ge=1, le=3)
    nll: ScientificPairedNllV1


def scientific_nll_groups_v1(
    rows: tuple[ScientificInterventionNllRowV1, ...],
) -> tuple[ScientificNllGroupV1, ...]:
    """Overall, four families, three difficulties, then twelve intersections."""
    selectors: tuple[tuple[SyntheticTask | None, int | None], ...] = (
        (None, None),
        *((task, None) for task in TASKS),
        *((None, level) for level in (1, 2, 3)),
        *((task, level) for task in TASKS for level in (1, 2, 3)),
    )
    return tuple(
        ScientificNllGroupV1(
            task_type=task,
            difficulty=level,
            nll=scientific_paired_nll_v1(
                tuple(
                    row
                    for row in rows
                    if (task is None or row.task_type == task)
                    and (level is None or row.difficulty == level)
                )
            ),
        )
        for task, level in selectors
    )


class ScientificInterventionNllPayloadV1(_Strict):
    kind: Literal["iris-scientific-intervention-nll-v1"] = "iris-scientific-intervention-nll-v1"
    arm: Literal["D"] = "D"
    purpose: Literal["smoke", "scientific"]
    semantics: Literal["iris-workspace-intervention-semantics-v1"] = (
        "iris-workspace-intervention-semantics-v1"
    )
    operator_rng_policy: Literal["iris-scientific-intervention-row-seed-v1"] = (
        "iris-scientific-intervention-row-seed-v1"
    )
    nll_scope: Literal["reference_target_natural_log_token_weighted"] = (
        "reference_target_natural_log_token_weighted"
    )
    scientific_launch_status: Literal["not_assessed"] = "not_assessed"
    intervention_sha256: Digest
    baseline_generation_sha256: Digest
    plan_sha256: Digest
    execution_sha256: Digest
    terminal_event_sha256: Digest
    best_checkpoint_sha256: Digest
    selected_step: int = Field(ge=1)
    teacher_configuration_sha256: Digest
    teacher_graph_sha256: Digest
    teacher_sidecar_sha256: Digest
    backbone_runtime_sha256: Digest
    test_parquet_sha256: Digest
    test_corpus_sha256: Digest
    rows: tuple[ScientificInterventionNllRowV1, ...] = Field(min_length=1, max_length=5000)
    groups: tuple[ScientificNllGroupV1, ...] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def exact_aggregation(self) -> Self:
        if len({row.example_id for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate measured rows")
        if self.groups != scientific_nll_groups_v1(self.rows):
            raise ValueError("Paired NLL groups differ from measured rows")
        if self.purpose == "scientific" and tuple(
            group.nll.examples for group in self.groups[1:5]
        ) != (2000, 1000, 1000, 1000):
            raise ValueError("Scientific NLL requires all 5000 held-out rows in source proportions")
        return self


class IrisScientificInterventionNllV1(_Strict):
    sha256: Digest
    payload: ScientificInterventionNllPayloadV1
