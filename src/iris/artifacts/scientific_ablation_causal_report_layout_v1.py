"""Closed complete-report JSON layout and pre-forward byte reservation.

This is a SERIALIZATION contract, not an effect executor or execution attestation.
The trusted source service must bind the metadata, validate native tensors and
all measurements/statistics, and call this before BUSY/any model forward. Layout
checking alone deliberately cannot certify those semantics. Every controlled
pass includes all three distinct ALL7 phases and the actual proposal collections;
donor data embedded in transfer plans is reserved again, not silently deduplicated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from math import isfinite
from re import fullmatch
from typing import Literal, TypeAliasType, get_args

from pydantic import BaseModel

from iris.artifacts.scientific_ablation_causal_controls_v1 import _score_bindings
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    IrisScientificAblationCausalObservedBudgetV1,
    scientific_ablation_causal_observed_budget_bytes_v1,
)
from iris.artifacts.scientific_ablation_evaluation_ledger_v2 import (
    IrisScientificAblationEvaluationCompleteV2,
    IrisScientificAblationEvaluationPlanV2,
)
from iris.artifacts.scientific_ablation_evaluation_v2 import IrisScientificAblationEvaluationV2
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

LAYOUT_DOMAIN = "iris/scientific-ablation-causal-report-layout/v1"
MAX_REPORT_BYTES = 512 * 1024**2


def _json_types(value: object, depth: int = 0) -> None:
    if depth > 48:
        raise ValueError("Closed report JSON depth exceeded")
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("Exact JSON object keys required")
            _json_types(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _json_types(child, depth + 1)
    elif not any(type(value) is primitive for primitive in (str, int, float, bool, type(None))):
        raise ValueError("Exact JSON primitives/containers required")


def _model_types(expected: type[BaseModel]) -> frozenset[type[BaseModel]]:
    """Resolve only trusted DECLARED annotations, never inspect a candidate class."""
    models: set[type[BaseModel]] = set()
    aliases: set[int] = set()

    def visit(annotation: object) -> None:
        if isinstance(annotation, TypeAliasType):
            if id(annotation) not in aliases:
                aliases.add(id(annotation))
                visit(annotation.__value__)
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation not in models:
                models.add(annotation)
                for field in annotation.model_fields.values():
                    visit(field.annotation)
        else:
            for child in get_args(annotation):
                visit(child)

    visit(expected)
    return frozenset(models)


def _closed_models(
    value: object,
    owned: object | None = None,
    depth: int = 0,
    allowed: frozenset[type[BaseModel]] | None = None,
) -> None:
    if depth > 48:
        raise ValueError("Closed source metadata depth exceeded")
    if allowed is None:
        # Internal generic tests may supply their own trusted root. Public _own
        # always supplies the declared root's closure AFTER its exact root check.
        if not isinstance(value, BaseModel):
            raise ValueError("Declared source metadata model root required")
        allowed = _model_types(type(value))
    # A foreign metaclass may override __hash__/__eq__. Identity alone must
    # decide membership BEFORE any candidate attributes or callbacks execute.
    if any(type(value) is expected for expected in allowed):
        assert isinstance(value, BaseModel)
        data = vars(value)
        if type(data) is not dict or any(type(name) is not str for name in data):
            raise ValueError("Exact source model dictionary/plain field keys required")
        if (
            set(data) != set(type(value).model_fields)
            or value.__pydantic_extra__ is not None
            or value.__pydantic_private__ is not None
            or (owned is not None and type(value) is not type(owned))
        ):
            raise ValueError("Exact closed source metadata tree required")
        for name, child in data.items():
            _closed_models(
                child, None if owned is None else getattr(owned, name), depth + 1, allowed
            )
    elif type(value) is tuple:
        if len(value) > 8192 or (owned is not None and type(owned) is not tuple):
            raise ValueError("Bounded immutable source metadata required")
        for index, child in enumerate(value):
            if owned is not None:
                assert type(owned) is tuple
            _closed_models(child, None if owned is None else owned[index], depth + 1, allowed)
    elif not any(type(value) is primitive for primitive in (str, int, float, bool, type(None))) or (
        owned is not None and type(value) is not type(owned)
    ):
        raise ValueError("Exact immutable source metadata primitives required")


def _own[T: BaseModel](value: T, expected: type[T]) -> T:
    if type(value) is not expected:
        raise ValueError("Exact original report source root required")
    allowed = _model_types(expected)
    _closed_models(value, allowed=allowed)
    content = canonical_json_file_bytes(value.model_dump(mode="json", warnings=False))
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("Original report source metadata exceeds storage limit")
    owned = expected.model_validate_json(content)
    _closed_models(value, owned, allowed=allowed)
    return owned


@dataclass(frozen=True, slots=True)
class _Fixed:
    encoded: bytes

    @property
    def bound(self) -> int:
        return len(self.encoded)

    def check(self, value: object) -> None:
        _json_types(value)
        if canonical_json_bytes(value) != self.encoded:
            raise ValueError("Frozen report metadata or literal differs")


@dataclass(frozen=True, slots=True)
class _Scalar:
    kind: Literal["float", "int", "bool", "digest", "text"]
    limit: int = 256

    @property
    def bound(self) -> int:
        # Finite binary64: <=17 significant digits, sign/point/exponent; signed
        # int64: <=20 chars. 32 covers each, excluding separately counted commas.
        # ensure_ascii=False: each codepoint is <=4 UTF8 or <=6 escaped bytes.
        return {"float": 32, "int": 32, "bool": 5, "digest": 66}.get(self.kind, 2 + 6 * self.limit)

    def check(self, value: object) -> None:
        valid = (
            (self.kind == "float" and type(value) is float and isfinite(value))
            or (self.kind == "int" and type(value) is int and -(2**63) <= value < 2**63)
            or (self.kind == "bool" and type(value) is bool)
            or (
                self.kind == "digest"
                and type(value) is str
                and fullmatch(r"[0-9a-f]{64}", value) is not None
            )
            or (self.kind == "text" and type(value) is str and len(value) <= self.limit)
        )
        if not valid:
            raise ValueError("Report scalar exceeds its exact type/encoding contract")
        # Exact finite built-in numbers use their ASCII repr in canonical JSON.
        # Keep the explicit encoding-bound check without allocating an encoder
        # for each numeric leaf. Strings still traverse UTF-8/escape validation,
        # including lone-surrogate rejection; no candidate repr can run here.
        size = (
            len(repr(value))
            if type(value) is float or type(value) is int
            else len(canonical_json_bytes(value))
        )
        if size > self.bound:
            raise ValueError("Report scalar encoding exceeds reservation")


@dataclass(frozen=True, slots=True)
class _Array:
    item: _Shape
    count: int
    minimum: int | None = None

    @property
    def bound(self) -> int:
        return 2 + self.count * self.item.bound + max(0, self.count - 1)

    def check(self, value: object) -> None:
        minimum = self.count if self.minimum is None else self.minimum
        if type(value) is not list or not minimum <= len(value) <= self.count:
            raise ValueError("Complete report array geometry/inventory differs")
        for child in value:
            self.item.check(child)


@dataclass(frozen=True, slots=True)
class _Object:
    members: tuple[tuple[str, _Shape], ...]

    @property
    def bound(self) -> int:
        return (
            2
            + max(0, len(self.members) - 1)
            + sum(len(canonical_json_bytes(name)) + 1 + child.bound for name, child in self.members)
        )

    def check(self, value: object) -> None:
        if type(value) is not dict or set(value) != {name for name, _ in self.members}:
            raise ValueError("Exact closed report object fields required")
        for name, child in self.members:
            child.check(value[name])


@dataclass(frozen=True, slots=True)
class _Choice:
    choices: tuple[_Shape, ...]

    @property
    def bound(self) -> int:
        return max(child.bound for child in self.choices)

    def check(self, value: object) -> None:
        for child in self.choices:
            try:
                child.check(value)
            except (ValueError, UnicodeError):
                continue
            return
        raise ValueError("Report value is outside its closed alternatives")


type _Shape = _Fixed | _Scalar | _Array | _Object | _Choice


def _descriptor(shape: object, depth: int = 0) -> object:
    """Bind ALL layout field names/types/lengths, not just aggregate byte counts."""
    if depth > 48:
        raise ValueError("Closed layout descriptor depth exceeded")
    if type(shape) is _Fixed:
        if type(shape.encoded) is not bytes:
            raise ValueError("Exact fixed layout encoding required")
        return dict(kind="fixed", bytes=len(shape.encoded), encoded=shape.encoded.decode())
    if type(shape) is _Scalar:
        if (
            type(shape.kind) is not str
            or shape.kind not in ("float", "int", "bool", "digest", "text")
            or type(shape.limit) is not int
            or not 0 <= shape.limit <= 4096
        ):
            raise ValueError("Exact bounded layout scalar required")
        return dict(kind="scalar", scalar=shape.kind, limit=shape.limit)
    if type(shape) is _Array:
        if (
            type(shape.count) is not int
            or not 0 <= shape.count < 2**31
            or (
                shape.minimum is not None
                and (type(shape.minimum) is not int or not 0 <= shape.minimum <= shape.count)
            )
        ):
            raise ValueError("Exact bounded layout array required")
        return dict(
            kind="array",
            count=shape.count,
            minimum=shape.minimum,
            item=_descriptor(shape.item, depth + 1),
        )
    if type(shape) is _Object:
        if (
            type(shape.members) is not tuple
            or len(shape.members) > 128
            or any(
                type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                for pair in shape.members
            )
            or len({pair[0] for pair in shape.members}) != len(shape.members)
        ):
            raise ValueError("Exact unique layout object fields required")
        return dict(
            kind="object",
            members=[(name, _descriptor(child, depth + 1)) for name, child in shape.members],
        )
    if type(shape) is _Choice:
        if type(shape.choices) is not tuple or not 1 <= len(shape.choices) <= 16:
            raise ValueError("Exact closed layout alternatives required")
        return dict(
            kind="choice", choices=[_descriptor(child, depth + 1) for child in shape.choices]
        )
    raise ValueError("Foreign layout nodes are forbidden")


def _fixed(value: object) -> _Fixed:
    return _Fixed(canonical_json_bytes(value))


def _object(**members: _Shape) -> _Object:
    return _Object(tuple(members.items()))


def _optional(value: _Shape) -> _Choice:
    return _Choice((_fixed(None), value))


def _matrix(rows: int, columns: int, scalar: _Shape) -> _Array:
    return _Array(_Array(scalar, columns), rows)


F, N, B, H, T = (
    _Scalar("float"),
    _Scalar("int"),
    _Scalar("bool"),
    _Scalar("digest"),
    _Scalar("text"),
)


def _trace(m: int, k: int, d: int, c: int, r: int, *, full: bool) -> _Object:
    vectors: _Shape = _fixed(None)
    if full:
        vectors = _object(
            read_workspace_slots=_matrix(k, d, F),
            workspace_slots=_matrix(k, d, F),
            module_states=_matrix(m, d, F),
            workspace_reads=_matrix(m, d, F),
        )
    cycle = _object(
        cycle=N,
        module_bids=_Array(F, m),
        module_confidence=_Array(F, m),
        selected_modules=_Array(N, r),
        selected_slots=_Array(N, r),
        write_modules=_Array(N, r),
        overwritten_slots=_Array(N, r, 0),
        slot_ages=_Array(N, k),
        slot_ttl=_Array(F, k),
        slot_norms=_Array(F, k),
        slot_sources=_Array(N, k),
        slot_occupancy=_Array(B, k),
        slot_write_counts=_Array(N, k),
        module_state_norms=_Array(F, m),
        workspace_cosine_matrix=_matrix(k, k, F),
        read_slot_sources=_Array(N, k),
        read_slot_occupancy=_Array(B, k),
        read_attention=_matrix(m, k, F),
        full=vectors,
    )
    return _object(
        kind=_fixed("iris-workspace-diagnostic-trace-v1"),
        semantics=_fixed("post-intervention-read-and-post-write-v1"),
        options=_object(
            mode=_fixed("full" if full else "metadata"),
            example_id=T,
            checkpoint_id=T,
            seed=N,
            maximum_scalars=N,
        ),
        num_modules=_fixed(m),
        num_slots=_fixed(k),
        state_dim=_fixed(d),
        cycle_count=_fixed(c),
        intervention_plan_sha256=_optional(H),
        cycles=_Array(cycle, c),
    )


def _tensor(shape: tuple[int, ...], scalar: _Shape, dtypes: tuple[str, ...]) -> _Object:
    values = scalar
    for count in reversed(shape):
        values = _Array(values, count)
    return _object(
        dtype=_Choice(tuple(_fixed(t) for t in dtypes)),
        shape=_fixed(list(shape)),
        tensor_sha256=H,
        values=values,
    )


def _state(k: int, d: int) -> _Object:
    floats = ("float16", "bfloat16", "float32", "float64")
    return _object(
        slots=_tensor((1, k, d), F, floats),
        age=_tensor((1, k), N, ("int64",)),
        confidence=_tensor((1, k), F, floats),
        ttl=_tensor((1, k), F, floats),
        occupancy=_tensor((1, k), B, ("bool",)),
        source_module=_tensor((1, k), N, ("int64",)),
        write_count=_tensor((1, k), N, ("int64",)),
    )


def _message(d: int) -> _Object:
    return _object(
        kind=_fixed("iris-module-message-snapshot-v1"),
        example_id=T,
        prompt_sha256=H,
        teacher_graph_sha256=H,
        sidecar_state_sha256=H,
        cycle=N,
        module=N,
        values=_Array(F, d),
        content_sha256=H,
    )


def _operation(kind: str) -> _Object:
    values: dict[str, _Shape] = dict.fromkeys(
        (
            "slot",
            "other_slot",
            "module",
            "seed",
            "sigma",
            "factor",
            "capacity",
            "values",
            "source_module",
            "confidence",
            "ttl",
        ),
        _fixed(None),
    )
    for name in {
        "zero_slot": ("slot",),
        "shuffle_slots": ("seed",),
        "block_module_write": ("module",),
    }[kind]:
        values[name] = N
    return _object(kind=_fixed(kind), cycles=_Array(N, 1), **values)


def _plan(ablation: object, role: str, d: int, message_limit: int | None) -> _Object:
    if role == "donor" or (role == "empty" and message_limit is not None):
        control = _object(
            kind=_fixed("iris-workspace-message-plan-v1"),
            donor=_message(d) if role == "donor" else _fixed(None),
        )
    else:
        control = _object(
            kind=_fixed("iris-workspace-intervention-plan-v1"),
            interventions=_Array(
                _fixed(None)
                if role == "empty"
                else _operation(
                    {
                        "zero": "zero_slot",
                        "shuffle": "shuffle_slots",
                        "block": "block_module_write",
                    }[role]
                ),
                0 if role == "empty" else 1,
            ),
        )
        message_limit = None
    return _object(
        kind=_fixed("iris-cognitive-ablation-control-plan-v1"),
        ablation=_fixed(ablation),
        control=control,
        maximum_message_scalars=_fixed(message_limit),
    )


@dataclass(frozen=True, slots=True)
class ScientificAblationCausalReportBytePreflightV1:
    """Planned closed JSON layout. NOT measured effects, native provenance or peak RAM."""

    observed_budget: IrisScientificAblationCausalObservedBudgetV1
    evaluation_plan: IrisScientificAblationEvaluationPlanV2
    evaluation: IrisScientificAblationEvaluationV2
    evaluation_complete: IrisScientificAblationEvaluationCompleteV2
    run_seed: int
    ordinary_teacher_graph_sha256: str
    controlled_teacher_graph_sha256: str
    sidecar_sha256: str
    maximum_report_bytes: int
    report_byte_preflight_bound: int
    source_metadata_bytes: int
    ordinary_rows_byte_bound: int
    controlled_passes_byte_bound: int
    effects_byte_bound: int
    statistics_byte_bound: int
    framing_byte_bound: int
    post_intervention_scalar_bound: int
    layout_sha256: str
    _shape: _Object


def build_scientific_ablation_causal_report_byte_preflight_v1(
    observed_budget: IrisScientificAblationCausalObservedBudgetV1,
    *,
    evaluation_plan: IrisScientificAblationEvaluationPlanV2,
    evaluation: IrisScientificAblationEvaluationV2,
    evaluation_complete: IrisScientificAblationEvaluationCompleteV2,
    run_seed: int,
    ordinary_teacher_graph_sha256: str,
    controlled_teacher_graph_sha256: str,
    sidecar_sha256: str,
    maximum_report_bytes: int,
) -> ScientificAblationCausalReportBytePreflightV1:
    """Reserve the ENTIRE future schema without allocating geometry-sized arrays.

    Source binding here checks closed metadata and complete score/membership
    consistency, not actual original Parquet/state/capture/score files. A trusted
    source transaction must supply its authenticated values before any forward.
    No quota-induced subset/cycle/writer/receiver shrink or model call occurs.
    """
    if type(maximum_report_bytes) is not int or not 1 <= maximum_report_bytes <= MAX_REPORT_BYTES:
        raise ValueError("Exact bounded complete report byte limit required")
    N.check(run_seed)
    if run_seed < 0:
        raise ValueError("Original nonnegative run seed required")
    for digest in (ordinary_teacher_graph_sha256, controlled_teacher_graph_sha256, sidecar_sha256):
        H.check(digest)
    budget = IrisScientificAblationCausalObservedBudgetV1.model_validate_json(
        scientific_ablation_causal_observed_budget_bytes_v1(observed_budget)
    )
    plan = _own(evaluation_plan, IrisScientificAblationEvaluationPlanV2)
    report = _own(evaluation, IrisScientificAblationEvaluationV2)
    complete = _own(evaluation_complete, IrisScientificAblationEvaluationCompleteV2)
    _score_bindings(plan, report, complete)
    design, g = budget.payload.design.payload, budget.payload.design.payload.geometry
    generation = report.payload.plan.payload
    source = generation.source.payload
    if (
        (design.ablation, design.ablation_policy_sha256, design.purpose)
        != (source.ablation, source.ablation_policy_sha256, source.purpose)
        or g.cycle_count != generation.cycles
        or ordinary_teacher_graph_sha256 != generation.teacher_graph_sha256
        or sidecar_sha256 != source.teacher_sidecar_sha256
        or len(design.members) != len(generation.rows)
        or any(
            (
                member.row_index,
                member.example_id,
                member.example_sha256,
                member.prompt_sha256,
                member.cognitive_prompt_sha256,
                member.task_type,
            )
            != (
                row.row_index,
                row.example_id,
                row.example_sha256,
                row.prompt_sha256,
                row.cognitive_prompt_sha256,
                row.task_type,
            )
            for member, row in zip(design.members, generation.rows, strict=True)
        )
    ):
        raise ValueError("Whole original source/design membership differs")
    m, k, d, c, r = (g.num_modules, g.num_slots, g.state_dim, g.cycle_count, g.writers_per_cycle)
    n = design.budget.examples
    message = type(design.policy) is ScientificAblationMessageCausalPolicyV1
    protocol = "message" if message else "workspace"
    full_trace, state = _trace(m, k, d, c, r, full=True), _state(k, d)
    message_limit = (
        design.policy.maximum_message_scalars
        if type(design.policy) is ScientificAblationMessageCausalPolicyV1
        else None
    )
    ordinary = _object(
        row_index=N,
        capture_row_sha256=H,
        evaluation_row_sha256=H,
        evaluation_ack_sha256=H,
        ordinary_teacher_invocation_sha256=H,
        target_sha256=H,
        target_token_count=N,
        target_nll_sum=F,
        ordinary_mean_target_nll=F,
        ordinary_trace=_trace(m, k, d, c, r, full=design.policy.ordinary_trace_mode == "full"),
    )

    def control(role: str) -> _Object:
        collect = message and role in ("empty", "donor")
        controlled = _object(
            kind=_fixed("iris-ablation-controlled-teacher-output-v1"),
            loss=F,
            supervised_token_count=N,
            plan=_plan(design.ablation.model_dump(mode="json"), role, d, message_limit),
            plan_sha256=H,
            ablation_policy_sha256=H,
            control_plan_sha256=H,
            prompt_sha256=H,
            target_sha256=H,
            teacher_graph_sha256=H,
            sidecar_state_sha256=H,
            invocation_sha256=H,
            trace=full_trace,
            messages=_Array(_message(d), c * m if collect else 0),
            states=_Array(state, c),
        )
        observation = _object(
            cycle=N,
            before_write=state,
            native_written=state,
            before_write_sha256=H,
            native_written_sha256=H,
        )
        return _object(
            row_index=N,
            role=_fixed(role),
            write_cycle=_fixed(None) if role == "empty" else N,
            writer=_fixed(None) if role in ("empty", "shuffle") else N,
            slot=_fixed(None) if role in ("empty", "shuffle") else N,
            donor_row_index=N if role == "donor" else _fixed(None),
            donor_pair_sha256=H if role == "donor" else _fixed(None),
            output=_object(
                kind=_fixed("iris-ablation-observed-teacher-output-v1"),
                controlled=controlled,
                maximum_observation_scalars=_fixed(budget.payload.maximum_observation_scalars),
                actual_observation_scalars=_fixed(budget.payload.observation_scalar_bound_per_pass),
                invocation_sha256=H,
                observations=_Array(observation, c),
            ),
        )

    trials = (
        _object(
            block=_Array(control("block"), n * (c - 1) * r),
            donor=_Array(control("donor"), n * (c - 1) * r),
        )
        if message
        else _object(
            shuffle=_Array(control("shuffle"), n * (c - 1)),
            zero=_Array(control("zero"), n * (c - 1) * r),
        )
    )
    controls = _object(empty=_Array(control("empty"), n), trials=trials)
    effect = _object(
        kind=_fixed(f"iris-scientific-ablation-{protocol}-writer-effect-v1"),
        row_index=N,
        write_cycle=N,
        source_module=N,
        target_module=N,
        slot=N,
        survives_to_next_read=B,
        overwrote_existing_record=B,
        genuine_distance=F,
        negative_control_distance=F,
        empty_invocation_sha256=H,
        intervention_invocation_sha256=H,
        negative_control_invocation_sha256=H,
        recipient_message_sha256=H if message else _fixed(None),
        donor_message_sha256=H if message else _fixed(None),
        donor_pair_sha256=H if message else _fixed(None),
        donor_row_index=N if message else _fixed(None),
    )
    pair = _object(
        source_module=N,
        target_module=N,
        independent_units=N,
        recipient_examples=N,
        write_trials=N,
        overwritten_trials=N,
        expired_trials=N,
        mean_genuine_distance=_optional(F),
        mean_negative_control_distance=_optional(F),
        mean_excess=_optional(F),
        positive_units=N,
        negative_units=N,
        tied_units=N,
        sign_test_p=F,
        holm_adjusted_p=F,
        significant_positive_effect=B,
    )
    statistics = _object(
        kind=_fixed(f"iris-scientific-ablation-{protocol}-writer-statistics-v1"),
        inferential_unit=_fixed("disjoint-donor-pair" if message else "example-cycle-mean"),
        alpha=_fixed(design.policy.alpha),
        numerical_tolerance=_fixed(design.policy.numerical_tolerance),
        independence=_fixed("input-uniqueness-does-not-prove-task-independence"),
        pairs=_Array(pair, m * (m - 1)),
    )
    metadata = dict(
        observed_budget=budget.model_dump(mode="json"),
        evaluation_plan=plan.model_dump(mode="json"),
        evaluation=report.model_dump(mode="json"),
        evaluation_complete=complete.model_dump(mode="json"),
        run_seed=run_seed,
        ordinary_teacher_graph_sha256=ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256=controlled_teacher_graph_sha256,
        sidecar_sha256=sidecar_sha256,
    )
    ordinary_rows, effects = _Array(ordinary, n), _Array(effect, design.budget.maximum_effect_rows)
    post_scalars = budget.payload.planned_controlled_teacher_executions * c * k * (d + 6)
    payload = _object(
        kind=_fixed(f"iris-scientific-ablation-{protocol}-writer-report-v1"),
        scope=_fixed("complete-teacher-writer-effects-not-generated-answer-quality"),
        scientific_acceptance=_fixed("not_assessed"),
        source_metadata=_fixed(metadata),
        byte_preflight=_object(
            layout_sha256=H,
            report_byte_preflight_bound=N,
            maximum_report_bytes=_fixed(maximum_report_bytes),
        ),
        actual_teacher_executions=_fixed(budget.payload.planned_total_teacher_executions),
        actual_trace_proposal_observation_scalar_bound=_fixed(
            budget.payload.total_trace_proposal_observation_scalar_bound
        ),
        actual_post_intervention_scalar_bound=_fixed(post_scalars),
        ordinary_rows=ordinary_rows,
        controlled_passes=controls,
        effects=effects,
        statistics=statistics,
    )
    shape = _object(sha256=H, payload=payload)
    bound = shape.bound + 1  # final LF; every object/array/key/comma/colon is counted above
    if bound > maximum_report_bytes:
        raise ValueError("Complete canonical report byte budget exceeded before any forward")
    base = len(canonical_json_bytes(metadata))
    framing = bound - base - ordinary_rows.bound - controls.bound - effects.bound - statistics.bound
    layout_sha = domain_sha256(
        LAYOUT_DOMAIN,
        canonical_json_bytes(
            dict(
                protocol=protocol,
                source_metadata=metadata,
                maximum_report_bytes=maximum_report_bytes,
                ordinary_rows_byte_bound=ordinary_rows.bound,
                controlled_passes_byte_bound=controls.bound,
                effects_byte_bound=effects.bound,
                statistics_byte_bound=statistics.bound,
                framing_byte_bound=framing,
                post_intervention_scalar_bound=post_scalars,
                report_byte_preflight_bound=bound,
                complete_layout=_descriptor(shape),
            )
        ),
    )
    return ScientificAblationCausalReportBytePreflightV1(
        budget,
        plan,
        report,
        complete,
        run_seed,
        ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256,
        sidecar_sha256,
        maximum_report_bytes,
        bound,
        base,
        ordinary_rows.bound,
        controls.bound,
        effects.bound,
        statistics.bound,
        framing,
        post_scalars,
        layout_sha,
        shape,
    )


def validate_scientific_ablation_causal_report_byte_preflight_v1(
    preflight: ScientificAblationCausalReportBytePreflightV1,
) -> ScientificAblationCausalReportBytePreflightV1:
    """Own the entire closed reservation again, without a model or result JSON."""
    if type(preflight) is not ScientificAblationCausalReportBytePreflightV1:
        raise ValueError("Exact closed complete report byte preflight required")
    _descriptor(preflight._shape)
    owned = build_scientific_ablation_causal_report_byte_preflight_v1(
        preflight.observed_budget,
        evaluation_plan=preflight.evaluation_plan,
        evaluation=preflight.evaluation,
        evaluation_complete=preflight.evaluation_complete,
        run_seed=preflight.run_seed,
        ordinary_teacher_graph_sha256=preflight.ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256=preflight.controlled_teacher_graph_sha256,
        sidecar_sha256=preflight.sidecar_sha256,
        maximum_report_bytes=preflight.maximum_report_bytes,
    )
    if any(
        type(getattr(preflight, f.name)) is not type(getattr(owned, f.name))
        or getattr(preflight, f.name) != getattr(owned, f.name)
        for f in fields(owned)
    ):
        raise ValueError("Complete report byte preflight changed")
    return owned


def causal_report_layout_bytes_v1(
    value: object,
    preflight: ScientificAblationCausalReportBytePreflightV1,
) -> bytes:
    """Own/check the complete serialization layout; NOT measurement acceptance.

    A future semantic formatter must validate original observed tensors, schedule,
    effects, source identities/statistics and the distinct report checksum FIRST.
    This size guard is mandatory even after the trusted pre-forward reservation.
    """
    owned = validate_scientific_ablation_causal_report_byte_preflight_v1(preflight)
    owned._shape.check(value)
    assert type(value) is dict
    byte_metadata = value["payload"]["byte_preflight"]
    if byte_metadata["layout_sha256"] != owned.layout_sha256 or (
        byte_metadata["report_byte_preflight_bound"] != owned.report_byte_preflight_bound
    ):
        raise ValueError("Actual report byte reservation binding differs")
    content = canonical_json_file_bytes(value)
    if len(content) > min(owned.report_byte_preflight_bound, owned.maximum_report_bytes):
        raise ValueError("Complete canonical report exceeded its byte reservation")
    # Detach from caller containers and repeat the closed shape check on owned JSON.
    owned._shape.check(json.loads(content))
    return content


__all__ = [
    "ScientificAblationCausalReportBytePreflightV1",
    "build_scientific_ablation_causal_report_byte_preflight_v1",
    "causal_report_layout_bytes_v1",
    "validate_scientific_ablation_causal_report_byte_preflight_v1",
]
