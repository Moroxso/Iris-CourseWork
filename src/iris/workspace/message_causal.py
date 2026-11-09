"""Input-matched write effects against disjoint-pair donor-content controls."""

from collections import defaultdict
from math import comb, fsum
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.causal import Digest, Distance
from iris.workspace.trace import _scalar_bound, _Strict


class WorkspaceMessageCausalPolicyV1(_Strict):
    kind: Literal["disjoint-pair-message-write-effect-v1"] = "disjoint-pair-message-write-effect-v1"
    shuffle_seed: int = Field(ge=0, le=2**63 - 1)
    maximum_examples: int = Field(default=64, ge=2, le=512)
    maximum_executions: int = Field(default=1024, ge=1, le=8192)
    maximum_trace_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_message_scalars: int = Field(default=65536, ge=1, le=2_000_000)
    maximum_total_trace_scalars: int = Field(default=16_000_000, ge=1, le=64_000_000)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    numerical_tolerance: float = Field(default=1e-6, ge=0, lt=1)


class MessageCausalSubjectV1(_Strict):
    example_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$")
    prompt_sha256: Digest
    stratum: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:/-]+$")


class MessageDonorPairV1(_Strict):
    sha256: Digest
    stratum: str
    first_example_id: str
    second_example_id: str


def pair_message_subjects_v1(
    subjects: tuple[MessageCausalSubjectV1, ...], policy: WorkspaceMessageCausalPolicyV1
) -> tuple[MessageDonorPairV1, ...]:
    """Sort input hashes inside each stratum; adjacent examples swap content.

    Each example belongs to exactly one pair. Odd/singleton strata reject, rather
    than dropping examples or reusing a donor and pretending observations independent.
    """
    policy = WorkspaceMessageCausalPolicyV1.model_validate(policy)
    subjects = tuple(MessageCausalSubjectV1.model_validate(s) for s in subjects)
    if not 2 <= len(subjects) <= policy.maximum_examples or any(
        len({getattr(s, field) for s in subjects}) != len(subjects)
        for field in ("example_id", "prompt_sha256")
    ):
        raise ValueError("Distinct bounded recipient/donor examples and prompts required")
    groups: dict[str, list[MessageCausalSubjectV1]] = defaultdict(list)
    for subject in subjects:
        groups[subject.stratum].append(subject)
    pairs = []
    for stratum in sorted(groups):
        rows = groups[stratum]
        if len(rows) % 2:
            raise ValueError("Every donor stratum must have an even number of independent inputs")
        rows.sort(
            key=lambda s: (
                domain_sha256(
                    "iris/message-donor-rank/v1",
                    canonical_json_bytes(
                        {"seed": policy.shuffle_seed, "subject": s.model_dump(mode="json")}
                    ),
                ),
                s.example_id,
            )
        )
        for left, right in zip(rows[::2], rows[1::2], strict=True):
            pairs.append(
                MessageDonorPairV1(
                    sha256=domain_sha256(
                        "iris/message-donor-pair/v1",
                        canonical_json_bytes(
                            {
                                "seed": policy.shuffle_seed,
                                "left": left.model_dump(mode="json"),
                                "right": right.model_dump(mode="json"),
                            }
                        ),
                    ),
                    stratum=stratum,
                    first_example_id=left.example_id,
                    second_example_id=right.example_id,
                )
            )
    return tuple(pairs)


def message_causal_budget_v1(
    policy: WorkspaceMessageCausalPolicyV1,
    *,
    examples: int,
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    writers: int,
) -> tuple[int, int]:
    for value in (examples, modules, slots, width, cycles, writers):
        if type(value) is not int or value < 1:
            raise ValueError("Exact positive message experiment geometry required")
    full = _scalar_bound(modules, slots, width, cycles, True)
    messages = cycles * modules * width
    trials = (cycles - 1) * writers
    calls = examples * (1 + 2 * trials)
    volume = calls * full + examples * (1 + trials) * messages
    if (
        not 2 <= examples <= policy.maximum_examples
        or examples % 2
        or not 2 <= modules <= 64
        or cycles < 2
        or width > 4096
        or writers > min(modules, slots)
        or full > policy.maximum_trace_scalars
        or messages > policy.maximum_message_scalars
        or calls > policy.maximum_executions
        or volume > policy.maximum_total_trace_scalars
    ):
        raise ValueError("Message experiment exceeds geometry/execution/scalar budget")
    return calls, volume


class MessageCausalEffectV1(_Strict):
    example_id: str
    donor_example_id: str
    donor_pair_sha256: Digest
    write_cycle: int = Field(ge=1)
    source_module: int = Field(ge=0)
    slot: int = Field(ge=0)
    target_module: int = Field(ge=0)
    overwrote_existing_record: bool
    survives_to_next_read: bool
    genuine_write_distance: Distance
    donor_write_distance: Distance
    recipient_message_sha256: Digest
    donor_message_sha256: Digest
    block_plan_sha256: Digest
    transfer_plan_sha256: Digest


class MessageCausalPairStatsV1(_Strict):
    source_module: int
    target_module: int
    independent_donor_pairs: int
    recipient_examples: int
    write_trials: int
    overwritten_trials: int
    expired_trials: int
    mean_genuine_distance: Distance | None
    mean_donor_distance: Distance | None
    mean_excess: float | None
    positive_pairs: int
    negative_pairs: int
    tied_pairs: int
    sign_test_p: Distance
    holm_adjusted_p: Distance
    significant_positive_effect: bool


def message_causal_statistics_v1(
    pairs: tuple[MessageDonorPairV1, ...],
    effects: tuple[MessageCausalEffectV1, ...],
    *,
    modules: int,
    policy: WorkspaceMessageCausalPolicyV1,
) -> tuple[MessageCausalPairStatsV1, ...]:
    """Average cycles per recipient, THEN recipients per disjoint donor pair."""
    records = []
    indexed: dict[tuple[int, int, str], list[MessageCausalEffectV1]] = defaultdict(list)
    for effect in effects:
        indexed[effect.source_module, effect.target_module, effect.example_id].append(effect)
    for source in range(modules):
        for target in range(modules):
            if source == target:
                continue
            genuine: list[float] = []
            donor: list[float] = []
            trials = recipients = overwritten = expired = 0
            for pair in pairs:
                g: list[float] = []
                d: list[float] = []
                for example in (pair.first_example_id, pair.second_example_id):
                    rows = indexed[source, target, example]
                    if rows:
                        recipients += 1
                        trials += len(rows)
                        overwritten += sum(e.overwrote_existing_record for e in rows)
                        expired += sum(not e.survives_to_next_read for e in rows)
                        g.append(fsum(e.genuine_write_distance for e in rows) / len(rows))
                        d.append(fsum(e.donor_write_distance for e in rows) / len(rows))
                if g:
                    genuine.append(fsum(g) / len(g))
                    donor.append(fsum(d) / len(d))
            excess = [a - b for a, b in zip(genuine, donor, strict=True)]
            positive = sum(v > policy.numerical_tolerance for v in excess)
            negative = sum(v < -policy.numerical_tolerance for v in excess)
            n = positive + negative
            p = sum(comb(n, k) for k in range(positive, n + 1)) / 2**n if n else 1.0
            records.append(
                MessageCausalPairStatsV1(
                    source_module=source,
                    target_module=target,
                    independent_donor_pairs=len(genuine),
                    recipient_examples=recipients,
                    write_trials=trials,
                    overwritten_trials=overwritten,
                    expired_trials=expired,
                    mean_genuine_distance=fsum(genuine) / len(genuine) if genuine else None,
                    mean_donor_distance=fsum(donor) / len(donor) if donor else None,
                    mean_excess=fsum(excess) / len(excess) if excess else None,
                    positive_pairs=positive,
                    negative_pairs=negative,
                    tied_pairs=len(genuine) - n,
                    sign_test_p=p,
                    holm_adjusted_p=1.0,
                    significant_positive_effect=False,
                )
            )
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(records)), key=lambda i: records[i].sign_test_p)):
        row = records[index]
        adjusted = max(previous, min(1.0, row.sign_test_p * (len(records) - rank)))
        records[index] = MessageCausalPairStatsV1.model_validate(
            {
                **row.model_dump(),
                "holm_adjusted_p": adjusted,
                "significant_positive_effect": adjusted <= policy.alpha,
            }
        )
        previous = adjusted
    return tuple(records)


class WorkspaceMessageCausalCaptureV1(_Strict):
    kind: Literal["iris-workspace-message-causal-v1"] = "iris-workspace-message-causal-v1"
    attribution: Literal["isolated-current-write-conditional-on-recipient-allocation"] = (
        "isolated-current-write-conditional-on-recipient-allocation"
    )
    endpoint: Literal["next-cycle-write-versus-blocked-symmetric-relative-module-l2"] = (
        "next-cycle-write-versus-blocked-symmetric-relative-module-l2"
    )
    shuffle_control: Literal[
        "same-module-cycle-paired-donor-content-recipient-routing-retained"
    ] = "same-module-cycle-paired-donor-content-recipient-routing-retained"
    test: Literal["one-sided-sign-test-disjoint-donor-pair-means-holm-all-directed-pairs"] = (
        "one-sided-sign-test-disjoint-donor-pair-means-holm-all-directed-pairs"
    )
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"
    policy: WorkspaceMessageCausalPolicyV1
    teacher_graph_sha256: Digest
    sidecar_state_sha256: Digest
    seed: int = Field(ge=0, le=2**63 - 1)
    num_modules: int = Field(ge=2, le=64)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1, le=4096)
    cycle_count: int = Field(ge=2)
    writers_per_cycle: int = Field(ge=1)
    executions: int = Field(ge=1)
    trace_scalar_bound: int = Field(ge=1)
    subjects: tuple[MessageCausalSubjectV1, ...] = Field(min_length=2, max_length=512)
    donor_pairs: tuple[MessageDonorPairV1, ...] = Field(min_length=1, max_length=256)
    effects: tuple[MessageCausalEffectV1, ...] = Field(max_length=500000)
    statistics: tuple[MessageCausalPairStatsV1, ...]

    @model_validator(mode="after")
    def complete_design(self) -> Self:
        m, r, c = self.num_modules, self.writers_per_cycle, self.cycle_count
        if tuple(
            sorted(self.subjects, key=lambda s: s.example_id)
        ) != self.subjects or self.donor_pairs != pair_message_subjects_v1(
            self.subjects, self.policy
        ):
            raise ValueError("Message donor pairing differs from declared input-only design")
        budget = message_causal_budget_v1(
            self.policy,
            examples=len(self.subjects),
            modules=m,
            slots=self.num_slots,
            width=self.state_dim,
            cycles=c,
            writers=r,
        )
        if (self.executions, self.trace_scalar_bound) != budget or len(self.effects) != len(
            self.subjects
        ) * (c - 1) * r * (m - 1):
            raise ValueError("Incomplete or oversized message intervention design")
        donor_map = {}
        for pair in self.donor_pairs:
            donor_map[pair.first_example_id] = (pair.second_example_id, pair.sha256)
            donor_map[pair.second_example_id] = (pair.first_example_id, pair.sha256)
        trials: dict[tuple[str, int, int, int], list[MessageCausalEffectV1]] = defaultdict(list)
        for effect in self.effects:
            if (
                donor_map.get(effect.example_id)
                != (effect.donor_example_id, effect.donor_pair_sha256)
                or not effect.write_cycle < c
                or effect.source_module >= m
                or effect.target_module >= m
                or effect.target_module == effect.source_module
                or effect.slot >= self.num_slots
            ):
                raise ValueError("Message trial source/receiver/donor domain differs")
            trials[effect.example_id, effect.write_cycle, effect.source_module, effect.slot].append(
                effect
            )
        for subject in self.subjects:
            for cycle in range(1, c):
                keys = [k for k in trials if k[:2] == (subject.example_id, cycle)]
                if (
                    len(keys) != r
                    or len({k[2] for k in keys}) != r
                    or len({k[3] for k in keys}) != r
                ):
                    raise ValueError("Every actual writer must be measured once")
        for (_, _, source, _), rows in trials.items():
            if len(rows) != m - 1 or {e.target_module for e in rows} != set(range(m)) - {source}:
                raise ValueError("Every other module must be observed for each writer")
            for field in (
                "overwrote_existing_record",
                "survives_to_next_read",
                "recipient_message_sha256",
                "donor_message_sha256",
                "block_plan_sha256",
                "transfer_plan_sha256",
            ):
                if len({getattr(e, field) for e in rows}) != 1:
                    raise ValueError("One message trial has conflicting metadata/receipts")
        if self.statistics != message_causal_statistics_v1(
            self.donor_pairs, self.effects, modules=m, policy=self.policy
        ):
            raise ValueError("Donor-pair clustered causal statistics differ from raw effects")
        return self
