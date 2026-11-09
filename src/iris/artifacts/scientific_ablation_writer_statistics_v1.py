"""Native-ablation writer-effect metadata and complete clustered sign statistics.

This is not an executor, source attestation or scientific/generated-answer
acceptance. The trusted complete service must derive EVERY effect from its native
passes and compare the resulting statistics before publication. Neutral capture/
report containers are never cast into these distinct domains. No model call occurs.
"""

from collections import defaultdict
from math import comb, fsum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_ablation_causal_design_v1 import (
    IrisScientificAblationCausalDesignV1,
    ScientificAblationMessageCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import _own
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.hashing import canonical_json_file_bytes

Distance = Annotated[float, Field(ge=0, le=1)]
Excess = Annotated[float, Field(ge=-1, le=1)]


def _sign_tail(positive: int, negative: int) -> float:
    """Exact binomial one-sided tail; numerical-tolerance ties are absent here."""
    if any(type(n) is not int or not 0 <= n <= 768 for n in (positive, negative)):
        raise ValueError("Exact bounded sign counts required")
    n = positive + negative
    if n > 768:
        raise ValueError("Complete independent-unit bound exceeded")
    return sum(comb(n, k) for k in range(positive, n + 1)) / 2**n if n else 1.0


def _holm(values: tuple[float, ...]) -> tuple[float, ...]:
    # All directed pairs, including missing pairs with p=1, belong to the family.
    result = [1.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
        result[index] = previous
    return tuple(result)


class _WriterEffectFields(_Strict):
    row_index: int = Field(ge=0, le=4999)
    write_cycle: int = Field(ge=1, le=63)
    source_module: int = Field(ge=0, le=63)
    target_module: int = Field(ge=0, le=63)
    slot: int = Field(ge=0, le=63)
    survives_to_next_read: bool
    overwrote_existing_record: bool
    genuine_distance: Distance
    negative_control_distance: Distance
    empty_invocation_sha256: Digest
    intervention_invocation_sha256: Digest
    negative_control_invocation_sha256: Digest

    @model_validator(mode="after")
    def other_receiver(self) -> Self:
        if self.source_module == self.target_module:
            raise ValueError("Next-cycle receiver must differ from the actual writer")
        return self


class ScientificAblationWorkspaceWriterEffectV1(_WriterEffectFields):
    kind: Literal["iris-scientific-ablation-workspace-writer-effect-v1"] = (
        "iris-scientific-ablation-workspace-writer-effect-v1"
    )
    recipient_message_sha256: None = None
    donor_message_sha256: None = None
    donor_pair_sha256: None = None
    donor_row_index: None = None


class ScientificAblationMessageWriterEffectV1(_WriterEffectFields):
    kind: Literal["iris-scientific-ablation-message-writer-effect-v1"] = (
        "iris-scientific-ablation-message-writer-effect-v1"
    )
    recipient_message_sha256: Digest
    donor_message_sha256: Digest
    donor_pair_sha256: Digest
    donor_row_index: int = Field(ge=0, le=4999)


type WriterEffect = (
    ScientificAblationWorkspaceWriterEffectV1 | ScientificAblationMessageWriterEffectV1
)


class ScientificAblationWriterPairStatisticsV1(_Strict):
    source_module: int = Field(ge=0, le=63)
    target_module: int = Field(ge=0, le=63)
    independent_units: int = Field(ge=0, le=768)
    recipient_examples: int = Field(ge=0, le=768)
    write_trials: int = Field(ge=0, le=8192)
    overwritten_trials: int = Field(ge=0, le=8192)
    expired_trials: int = Field(ge=0, le=8192)
    mean_genuine_distance: Distance | None
    mean_negative_control_distance: Distance | None
    mean_excess: Excess | None
    positive_units: int = Field(ge=0, le=768)
    negative_units: int = Field(ge=0, le=768)
    tied_units: int = Field(ge=0, le=768)
    sign_test_p: Distance
    holm_adjusted_p: Distance
    significant_positive_effect: bool

    @model_validator(mode="after")
    def coherent_counts(self) -> Self:
        if (
            self.source_module == self.target_module
            or self.positive_units + self.negative_units + self.tied_units != self.independent_units
            or not self.independent_units <= self.recipient_examples <= self.write_trials
            or max(self.overwritten_trials, self.expired_trials) > self.write_trials
            or self.sign_test_p != _sign_tail(self.positive_units, self.negative_units)
        ):
            raise ValueError("Incoherent complete writer pair counts/exact sign tail")
        means = (self.mean_genuine_distance, self.mean_negative_control_distance, self.mean_excess)
        if self.independent_units == 0:
            if any(v is not None for v in means) or self.write_trials != 0:
                raise ValueError("Missing directed pair must retain p1/null means/zero counts")
        elif any(v is None for v in means):
            raise ValueError("Observed directed pair requires all descriptive means")
        return self


class _WriterStatisticsFields(_Strict):
    alpha: float = Field(gt=0, lt=1)
    numerical_tolerance: float = Field(ge=0, lt=1)
    independence: Literal["input-uniqueness-does-not-prove-task-independence"] = (
        "input-uniqueness-does-not-prove-task-independence"
    )
    pairs: tuple[ScientificAblationWriterPairStatisticsV1, ...] = Field(
        min_length=2, max_length=4032
    )

    @model_validator(mode="after")
    def full_family(self) -> Self:
        m = 1 + max(max(p.source_module, p.target_module) for p in self.pairs)
        if tuple((p.source_module, p.target_module) for p in self.pairs) != tuple(
            (s, t) for s in range(m) for t in range(m) if s != t
        ):
            raise ValueError("ALL directed M(M-1) pairs required in canonical order")
        corrected = _holm(tuple(p.sign_test_p for p in self.pairs))
        if any(
            p.holm_adjusted_p != adjusted
            or p.significant_positive_effect is not (adjusted <= self.alpha)
            for p, adjusted in zip(self.pairs, corrected, strict=True)
        ):
            raise ValueError("Holm family includes missing pairs and every directed pair")
        return self


class ScientificAblationWorkspaceWriterStatisticsV1(_WriterStatisticsFields):
    kind: Literal["iris-scientific-ablation-workspace-writer-statistics-v1"] = (
        "iris-scientific-ablation-workspace-writer-statistics-v1"
    )
    inferential_unit: Literal["example-cycle-mean"] = "example-cycle-mean"


class ScientificAblationMessageWriterStatisticsV1(_WriterStatisticsFields):
    kind: Literal["iris-scientific-ablation-message-writer-statistics-v1"] = (
        "iris-scientific-ablation-message-writer-statistics-v1"
    )
    inferential_unit: Literal["disjoint-donor-pair"] = "disjoint-donor-pair"


type WriterStatistics = (
    ScientificAblationWorkspaceWriterStatisticsV1 | ScientificAblationMessageWriterStatisticsV1
)


def _effect(value: WriterEffect, message: bool) -> WriterEffect:
    if message:
        if type(value) is not ScientificAblationMessageWriterEffectV1:
            raise ValueError("Exact distinct native-ablation message effect required")
        return _own(value, ScientificAblationMessageWriterEffectV1)
    if type(value) is not ScientificAblationWorkspaceWriterEffectV1:
        raise ValueError("Exact distinct native-ablation workspace effect required")
    return _own(value, ScientificAblationWorkspaceWriterEffectV1)


def _inventory(
    design: IrisScientificAblationCausalDesignV1, effects: tuple[WriterEffect, ...], message: bool
) -> dict[tuple[int, int, int], list[WriterEffect]]:
    p, g = design.payload, design.payload.geometry
    if type(effects) is not tuple or len(effects) != p.budget.maximum_effect_rows:
        raise ValueError("EVERY actual write/other receiver effect required, no subset")
    indexed: dict[tuple[int, int, int], list[WriterEffect]] = defaultdict(list)
    trials: dict[tuple[int, int, int], list[WriterEffect]] = defaultdict(list)
    donors = {
        i: (other, pair.sha256)
        for pair in p.donor_pairs
        for i, other in (
            (pair.first_row_index, pair.second_row_index),
            (pair.second_row_index, pair.first_row_index),
        )
    }
    ordinary: dict[int, set[str]] = defaultdict(set)
    shuffle: dict[tuple[int, int], set[str]] = defaultdict(set)
    for value in effects:
        e = _effect(value, message)
        if (
            e.row_index not in p.selected_row_indices
            or e.write_cycle >= g.cycle_count
            or max(e.source_module, e.target_module) >= g.num_modules
            or e.slot >= g.num_slots
        ):
            raise ValueError("Effect lies outside complete original selection/geometry")
        if message and (e.donor_row_index, e.donor_pair_sha256) != donors[e.row_index]:
            raise ValueError("Message effect requires its original input-only disjoint donor pair")
        trials[e.row_index, e.write_cycle, e.source_module].append(e)
        indexed[e.source_module, e.target_module, e.row_index].append(e)
        ordinary[e.row_index].add(e.empty_invocation_sha256)
        shuffle[e.row_index, e.write_cycle].add(e.negative_control_invocation_sha256)
    for row in p.selected_row_indices:
        if len(ordinary[row]) != 1:
            raise ValueError("ONE original EMPTY pass per selected row, not a repeated baseline")
        for cycle in range(1, g.cycle_count):
            groups = [rows for (i, c, _), rows in trials.items() if (i, c) == (row, cycle)]
            if (
                len(groups) != g.writers_per_cycle
                or len({rows[0].slot for rows in groups}) != g.writers_per_cycle
                or (not message and len(shuffle[row, cycle]) != 1)
            ):
                raise ValueError(
                    "Every actual distinct writer/slot and one whole-record SHUFFLE required"
                )
            for rows in groups:
                first = rows[0]
                if (
                    len(rows) != g.num_modules - 1
                    or {e.target_module for e in rows}
                    != set(range(g.num_modules)) - {first.source_module}
                    or any(
                        any(
                            getattr(e, name) != getattr(first, name)
                            for name in (
                                "slot",
                                "survives_to_next_read",
                                "overwrote_existing_record",
                                "intervention_invocation_sha256",
                                "negative_control_invocation_sha256",
                                "recipient_message_sha256",
                                "donor_message_sha256",
                            )
                        )
                        for e in rows
                    )
                ):
                    raise ValueError("Every other receiver must share the SAME actual writer trial")
    return indexed


def build_scientific_ablation_writer_statistics_v1(
    design: IrisScientificAblationCausalDesignV1, effects: tuple[WriterEffect, ...]
) -> WriterStatistics:
    """Cluster before sign testing; hashes/closed metadata do NOT attest execution."""
    owned = _own(design, IrisScientificAblationCausalDesignV1)
    p = owned.payload
    message = type(p.policy) is ScientificAblationMessageCausalPolicyV1
    indexed = _inventory(owned, effects, message)
    units = (
        tuple((pair.first_row_index, pair.second_row_index) for pair in p.donor_pairs)
        if message
        else tuple((row,) for row in p.selected_row_indices)
    )
    records = []
    for source in range(p.geometry.num_modules):
        for target in range(p.geometry.num_modules):
            if source == target:
                continue
            genuine: list[float] = []
            negative: list[float] = []
            recipient_count = trial_count = overwritten = expired = 0
            for unit in units:
                unit_g: list[float] = []
                unit_n: list[float] = []
                for row in unit:
                    selected = indexed.get((source, target, row))
                    if not selected:
                        continue
                    recipient_count += 1
                    trial_count += len(selected)
                    overwritten += sum(e.overwrote_existing_record for e in selected)
                    expired += sum(not e.survives_to_next_read for e in selected)
                    unit_g.append(fsum(e.genuine_distance for e in selected) / len(selected))
                    unit_n.append(
                        fsum(e.negative_control_distance for e in selected) / len(selected)
                    )
                # A donor need not have won: use the AVAILABLE recipient means.
                if unit_g:
                    genuine.append(fsum(unit_g) / len(unit_g))
                    negative.append(fsum(unit_n) / len(unit_n))
            excess = [g - n for g, n in zip(genuine, negative, strict=True)]
            positive = sum(v > p.policy.numerical_tolerance for v in excess)
            negative_units = sum(v < -p.policy.numerical_tolerance for v in excess)
            records.append(
                ScientificAblationWriterPairStatisticsV1(
                    source_module=source,
                    target_module=target,
                    independent_units=len(genuine),
                    recipient_examples=recipient_count,
                    write_trials=trial_count,
                    overwritten_trials=overwritten,
                    expired_trials=expired,
                    mean_genuine_distance=fsum(genuine) / len(genuine) if genuine else None,
                    mean_negative_control_distance=fsum(negative) / len(negative)
                    if negative
                    else None,
                    mean_excess=fsum(excess) / len(excess) if excess else None,
                    positive_units=positive,
                    negative_units=negative_units,
                    tied_units=len(genuine) - positive - negative_units,
                    sign_test_p=_sign_tail(positive, negative_units),
                    holm_adjusted_p=1.0,
                    significant_positive_effect=False,
                )
            )
    adjusted = _holm(tuple(r.sign_test_p for r in records))
    checked = tuple(
        ScientificAblationWriterPairStatisticsV1.model_validate(
            r.model_dump(mode="python")
            | dict(holm_adjusted_p=v, significant_positive_effect=v <= p.policy.alpha)
        )
        for r, v in zip(records, adjusted, strict=True)
    )
    if message:
        return ScientificAblationMessageWriterStatisticsV1(
            alpha=p.policy.alpha, numerical_tolerance=p.policy.numerical_tolerance, pairs=checked
        )
    return ScientificAblationWorkspaceWriterStatisticsV1(
        alpha=p.policy.alpha, numerical_tolerance=p.policy.numerical_tolerance, pairs=checked
    )


def scientific_ablation_writer_statistics_bytes_v1(value: WriterStatistics) -> bytes:
    """Canonical embedded statistics ONLY, not a complete causal report/root SHA."""
    owned: WriterStatistics
    if type(value) is ScientificAblationWorkspaceWriterStatisticsV1:
        owned = _own(value, ScientificAblationWorkspaceWriterStatisticsV1)
    elif type(value) is ScientificAblationMessageWriterStatisticsV1:
        owned = _own(value, ScientificAblationMessageWriterStatisticsV1)
    else:
        raise ValueError("Exact distinct native-ablation writer statistics required")
    return canonical_json_file_bytes(owned.model_dump(mode="json"))
