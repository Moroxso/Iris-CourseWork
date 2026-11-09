"""Explicit causal intervention intent; not a baseline or scientific-acceptance receipt."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.utils.hashing import canonical_json_bytes, domain_sha256

InterventionKindV1 = Literal[
    "zero_workspace",
    "zero_slot",
    "freeze_slot",
    "swap_slots",
    "shuffle_slots",
    "gaussian_noise",
    "replace_slot",
    "block_module_write",
    "block_module_read",
    "force_module_winner",
    "scale_module_bid",
    "reduce_workspace_capacity",
]


class WorkspaceInterventionSpecV1(BaseModel):
    """Closed, hashable parameters; cycles are one-based, None means every cycle."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False, revalidate_instances="always"
    )

    kind: InterventionKindV1
    cycles: tuple[int, ...] | None = None
    slot: int | None = Field(default=None, ge=0)
    other_slot: int | None = Field(default=None, ge=0)
    module: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    sigma: float | None = Field(default=None, ge=0)
    factor: float | None = Field(default=None, ge=0)
    capacity: int | None = Field(default=None, ge=1)
    values: tuple[float, ...] | None = Field(default=None, min_length=1)
    source_module: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    ttl: float | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def exact_parameters(self) -> Self:
        required = {
            "zero_workspace": set(),
            "zero_slot": {"slot"},
            "freeze_slot": {"slot"},
            "swap_slots": {"slot", "other_slot"},
            "shuffle_slots": {"seed"},
            "gaussian_noise": {"seed", "sigma"},
            "replace_slot": {"slot", "values", "source_module", "confidence", "ttl"},
            "block_module_write": {"module"},
            "block_module_read": {"module"},
            "force_module_winner": {"module"},
            "scale_module_bid": {"module", "factor"},
            "reduce_workspace_capacity": {"capacity"},
        }[self.kind]
        present = set(self.model_dump(exclude_none=True)) - {"kind", "cycles"}
        if present != required:
            raise ValueError("Intervention parameters must exactly match its kind")
        if self.cycles is not None and (
            not self.cycles
            or any(type(c) is not int or c < 1 for c in self.cycles)
            or tuple(sorted(set(self.cycles))) != self.cycles
        ):
            raise ValueError("Intervention cycles must be unique increasing positive integers")
        if self.kind == "swap_slots" and self.slot == self.other_slot:
            raise ValueError("Swap requires two different slots")
        return self

    def active(self, cycle: int) -> bool:
        if type(cycle) is not int or cycle < 1:
            raise ValueError("Intervention cycle must be a positive integer")
        return self.cycles is None or cycle in self.cycles


class WorkspaceInterventionPlanV1(BaseModel):
    """Ordered operations within each phase; an empty plan is the paired control."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False, revalidate_instances="always"
    )

    kind: Literal["iris-workspace-intervention-plan-v1"] = "iris-workspace-intervention-plan-v1"
    interventions: tuple[WorkspaceInterventionSpecV1, ...] = Field(max_length=16)

    def validate_dimensions(
        self,
        *,
        num_slots: int,
        num_modules: int,
        state_dim: int,
        cycles: int,
        winners_per_cycle: int,
    ) -> None:
        # Pydantic model_copy/model_construct can bypass construction validation.
        plan = validate_intervention_plan_v1(self)
        for dimension in (num_slots, num_modules, state_dim, cycles, winners_per_cycle):
            if type(dimension) is not int or dimension < 1:
                raise ValueError("Positive exact graph dimensions required")
        if winners_per_cycle > min(num_slots, num_modules):
            raise ValueError("Winner count exceeds the original graph capacity")
        for operation in plan.interventions:
            for slot in (operation.slot, operation.other_slot):
                if slot is not None and slot >= num_slots:
                    raise ValueError("Intervention slot is outside the actual graph")
            for module in (operation.module, operation.source_module):
                if module is not None and module >= num_modules:
                    raise ValueError("Intervention module is outside the actual graph")
            if operation.values is not None and len(operation.values) != state_dim:
                raise ValueError("Replacement must specify the complete latent vector")
            if operation.capacity is not None and operation.capacity > num_slots:
                raise ValueError("Capacity cannot increase the original workspace")
            if operation.cycles is not None and operation.cycles[-1] > cycles:
                raise ValueError("Intervention cycle is outside the actual graph")
        # Only scheduled transitions can change the forced set; graph cycle counts
        # need not make this validation perform an unbounded range scan.
        check_cycles = {1}
        for operation in plan.interventions:
            if operation.kind == "force_module_winner" and operation.cycles is not None:
                check_cycles.update(operation.cycles)
        for cycle in check_cycles:
            forced = {
                op.module
                for op in plan.interventions
                if op.kind == "force_module_winner" and op.active(cycle)
            }
            if len(forced) > winners_per_cycle:
                raise ValueError("Too many distinct forced winners in one cycle")


def validate_intervention_plan_v1(value: object) -> WorkspaceInterventionPlanV1:
    """Take a freshly validated, exact-type snapshot at an execution boundary."""
    if type(value) is not WorkspaceInterventionPlanV1:
        raise TypeError("An exact WorkspaceInterventionPlanV1 is required")
    if any(type(op) is not WorkspaceInterventionSpecV1 for op in value.interventions):
        raise TypeError("Plan entries must be exact WorkspaceInterventionSpecV1 instances")
    return WorkspaceInterventionPlanV1.model_validate(value)


def intervention_plan_sha256_v1(value: WorkspaceInterventionPlanV1) -> str:
    """Bind order, schedules and every declared parameter in a separate domain."""
    plan = validate_intervention_plan_v1(value)
    return domain_sha256(
        "iris/workspace-intervention-plan/v1", canonical_json_bytes(plan.model_dump(mode="json"))
    )


__all__ = [
    "InterventionKindV1",
    "WorkspaceInterventionPlanV1",
    "WorkspaceInterventionSpecV1",
    "intervention_plan_sha256_v1",
    "validate_intervention_plan_v1",
]
