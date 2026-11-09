"""Next-cycle slot-ablation effects and a paired, example-clustered sign test.

Slot permutation is a negative control, not semantic corruption of memory.
These reports do not establish task benefit or complete scientific acceptance.
"""

from collections import defaultdict
from math import comb, fsum, hypot, isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.workspace.trace import _Strict

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Distance = Annotated[float, Field(ge=0, le=1)]


class WorkspaceCausalPolicyV1(_Strict):
    kind: Literal["slot-ablation-next-cycle-v1"] = "slot-ablation-next-cycle-v1"
    shuffle_seed: int = Field(ge=0, le=2**63 - 1)
    maximum_executions: int = Field(default=64, ge=1, le=257)
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)


class WorkspaceCausalEffectV1(_Strict):
    write_cycle: int = Field(ge=1)
    source_module: int = Field(ge=0)
    slot: int = Field(ge=0)
    target_module: int = Field(ge=0)
    survives_to_next_read: bool
    overwrote_existing_record: bool
    ablated_distance: Distance
    shuffled_distance: Distance
    ablation_plan_sha256: Digest
    shuffle_plan_sha256: Digest


class WorkspaceCausalCaptureV1(_Strict):
    kind: Literal["iris-workspace-causal-capture-v1"] = "iris-workspace-causal-capture-v1"
    endpoint: Literal["next-cycle-symmetric-relative-module-l2"] = (
        "next-cycle-symmetric-relative-module-l2"
    )
    shuffle_control: Literal["whole-slot-record-permutation-not-content-corruption"] = (
        "whole-slot-record-permutation-not-content-corruption"
    )
    attribution: Literal["last-writer-associated-whole-record-not-isolated-proposal"] = (
        "last-writer-associated-whole-record-not-isolated-proposal"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    policy: WorkspaceCausalPolicyV1
    example_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$")
    prompt_sha256: Digest
    teacher_graph_sha256: Digest
    sidecar_state_sha256: Digest
    seed: int = Field(ge=0, le=2**63 - 1)
    num_modules: int = Field(ge=2, le=64)
    num_slots: int = Field(ge=1)
    cycle_count: int = Field(ge=2)
    writers_per_cycle: int = Field(ge=1)
    executions: int = Field(ge=1)
    effects: tuple[WorkspaceCausalEffectV1, ...]

    @model_validator(mode="after")
    def complete_design(self) -> Self:
        m, r, c = self.num_modules, self.writers_per_cycle, self.cycle_count
        expected_calls = 1 + (c - 1) * (r + 1)
        if (
            r > min(m, self.num_slots)
            or self.executions != expected_calls
            or self.executions > self.policy.maximum_executions
            or len(self.effects) != (c - 1) * r * (m - 1)
        ):
            raise ValueError("Incomplete or oversized next-cycle intervention design")
        trials: dict[tuple[int, int, int], list[WorkspaceCausalEffectV1]] = defaultdict(list)
        for effect in self.effects:
            if (
                effect.write_cycle >= c
                or effect.source_module >= m
                or effect.target_module >= m
                or effect.source_module == effect.target_module
                or effect.slot >= self.num_slots
            ):
                raise ValueError("Invalid next-cycle source/target/slot domain")
            trials[effect.write_cycle, effect.source_module, effect.slot].append(effect)
        for cycle in range(1, c):
            keys = [key for key in trials if key[0] == cycle]
            if (
                len(keys) != r
                or len({key[1] for key in keys}) != r
                or len({key[2] for key in keys}) != r
                or len({e.shuffle_plan_sha256 for e in self.effects if e.write_cycle == cycle}) != 1
            ):
                raise ValueError("Missing or duplicated source write in causal design")
        for (_, source, _), effects in trials.items():
            if (
                len(effects) != m - 1
                or {e.target_module for e in effects} != set(range(m)) - {source}
                or len({e.ablation_plan_sha256 for e in effects}) != 1
                or len({e.survives_to_next_read for e in effects}) != 1
                or len({e.overwrote_existing_record for e in effects}) != 1
            ):
                raise ValueError("Ablation must observe every other module exactly once")
        return self


def relative_state_distance_v1(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """||a-b|| / (||a||+||b||), with zero/zero=0 and scaled intermediates."""
    if not left or len(left) != len(right) or not all(isfinite(v) for v in (*left, *right)):
        raise ValueError("Matched nonempty state vectors required")
    scale = max(map(abs, (*left, *right)))
    if scale == 0:
        return 0.0
    a, b = tuple(v / scale for v in left), tuple(v / scale for v in right)
    result = hypot(*(x - y for x, y in zip(a, b, strict=True))) / (hypot(*a) + hypot(*b))
    if not 0 <= result <= 1 + 1e-12:
        raise ValueError("Finite module states required")
    return min(1.0, result)


class WorkspaceCausalPairV1(_Strict):
    source_module: int
    target_module: int
    examples: int
    write_trials: int
    overwritten_record_trials: int
    expired_before_read_trials: int
    mean_ablation_distance: Distance | None
    mean_shuffle_distance: Distance | None
    mean_excess: float | None
    positive_examples: int
    negative_examples: int
    tied_examples: int
    sign_test_p: Distance
    holm_adjusted_p: Distance
    significant_positive_effect: bool


class WorkspaceCausalSummaryV1(_Strict):
    kind: Literal["iris-workspace-causal-summary-v1"] = "iris-workspace-causal-summary-v1"
    test: Literal["one-sided-paired-sign-test-example-means-holm-all-directed-pairs"] = (
        "one-sided-paired-sign-test-example-means-holm-all-directed-pairs"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    examples: int
    teacher_graph_sha256: Digest
    sidecar_state_sha256: Digest
    seed: int
    policy: WorkspaceCausalPolicyV1
    alpha: Annotated[float, Field(gt=0, lt=1)]
    numerical_tolerance: Annotated[float, Field(ge=0, lt=1)]
    pairs: tuple[WorkspaceCausalPairV1, ...]


def summarize_workspace_causal_v1(
    captures: tuple[WorkspaceCausalCaptureV1, ...],
    *,
    alpha: float = 0.05,
    numerical_tolerance: float = 1e-6,
) -> WorkspaceCausalSummaryV1:
    """Average cycles within each example BEFORE testing; no writer selection by effect.

    Alpha/tolerance and the held-out diagnostic subset must be chosen before results.
    Ties are excluded from the sign test, not from descriptive means. Missing
    source-target pairs stay in the Holm family with p=1. Independent examples
    are an experiment-design assumption, not something these hashes can prove.
    """
    if type(captures) is not tuple or not 1 <= len(captures) <= 1024:
        raise ValueError("Require one to 1024 bounded exact captures")
    if any(type(c) is not WorkspaceCausalCaptureV1 for c in captures):
        raise TypeError("Exact causal captures required")
    checked = tuple(WorkspaceCausalCaptureV1.model_validate(c) for c in captures)
    first = checked[0]
    for field in ("example_id", "prompt_sha256"):
        if len({getattr(c, field) for c in checked}) != len(checked):
            raise ValueError("Repeated example/prompt would duplicate experimental units")
    identity = (
        "teacher_graph_sha256",
        "sidecar_state_sha256",
        "seed",
        "policy",
        "num_modules",
        "num_slots",
        "cycle_count",
        "writers_per_cycle",
    )
    if any(any(getattr(c, field) != getattr(first, field) for field in identity) for c in checked):
        raise ValueError("Cannot pool distinct model/seed/policy/geometry contexts")
    if not 0 < alpha < 1 or not 0 <= numerical_tolerance < 1:
        raise ValueError("Invalid predeclared significance/tolerance policy")
    records: list[WorkspaceCausalPairV1] = []
    for source in range(first.num_modules):
        for target in range(first.num_modules):
            if source == target:
                continue
            a: list[float] = []
            s: list[float] = []
            trials = overwritten = expired = 0
            for capture in checked:
                rows = [
                    e
                    for e in capture.effects
                    if (e.source_module, e.target_module) == (source, target)
                ]
                if rows:
                    trials += len(rows)
                    overwritten += sum(e.overwrote_existing_record for e in rows)
                    expired += sum(not e.survives_to_next_read for e in rows)
                    a.append(fsum(e.ablated_distance for e in rows) / len(rows))
                    s.append(fsum(e.shuffled_distance for e in rows) / len(rows))
            differences = [x - y for x, y in zip(a, s, strict=True)]
            positive = sum(d > numerical_tolerance for d in differences)
            negative = sum(d < -numerical_tolerance for d in differences)
            n = positive + negative
            p = sum(comb(n, k) for k in range(positive, n + 1)) / 2**n if n else 1.0
            records.append(
                WorkspaceCausalPairV1(
                    source_module=source,
                    target_module=target,
                    examples=len(a),
                    write_trials=trials,
                    overwritten_record_trials=overwritten,
                    expired_before_read_trials=expired,
                    mean_ablation_distance=fsum(a) / len(a) if a else None,
                    mean_shuffle_distance=fsum(s) / len(s) if s else None,
                    mean_excess=fsum(differences) / len(differences) if differences else None,
                    positive_examples=positive,
                    negative_examples=negative,
                    tied_examples=len(a) - n,
                    sign_test_p=p,
                    holm_adjusted_p=1.0,
                    significant_positive_effect=False,
                )
            )
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(records)), key=lambda i: records[i].sign_test_p)):
        row = records[index]
        adjusted = max(previous, min(1.0, row.sign_test_p * (len(records) - rank)))
        records[index] = WorkspaceCausalPairV1.model_validate(
            {
                **row.model_dump(),
                "holm_adjusted_p": adjusted,
                "significant_positive_effect": adjusted <= alpha,
            }
        )
        previous = adjusted
    return WorkspaceCausalSummaryV1(
        examples=len(checked),
        teacher_graph_sha256=first.teacher_graph_sha256,
        sidecar_state_sha256=first.sidecar_state_sha256,
        seed=first.seed,
        policy=first.policy,
        alpha=alpha,
        numerical_tolerance=numerical_tolerance,
        pairs=tuple(records),
    )
