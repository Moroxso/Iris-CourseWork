"""Functional state-phase interventions; graph-phase controls are separate.

Operations validate even inactive inputs and never modify or return aliases of
caller-owned tensors. Randomness is local to (kind, seed, one-based cycle).
Standalone states do not declare a module count: the graph owner must additionally
bind the intervention plan to its actual dimensions and module domain.
"""

from dataclasses import fields
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch

from iris.interventions.policy import InterventionKindV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.state import WorkspaceState

if TYPE_CHECKING:
    from iris.workspace.outputs import SlotAllocatorOutput


class WorkspaceInterventionError(RuntimeError):
    """An intervention cannot preserve the workspace state contract."""


@runtime_checkable
class WorkspaceIntervention(Protocol):
    """Source section18's state-phase interface, with one-based cycle numbers."""

    def apply(self, state: WorkspaceState, cycle: int) -> WorkspaceState: ...


def validate_intervention_state(state: WorkspaceState) -> None:
    """Require a canonical dense workspace, without changing the old validators."""
    if type(state) is not WorkspaceState:
        raise WorkspaceInterventionError("An exact WorkspaceState is required")
    slots = state.slots
    if (
        type(slots) is not torch.Tensor
        or slots.layout != torch.strided
        or slots.ndim != 3
        or any(dimension < 1 for dimension in slots.shape)
        or slots.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        or slots.device.type == "meta"
    ):
        raise WorkspaceInterventionError("Workspace slots must be dense finite [B,K,D] floats")
    shape = tuple(slots.shape[:2])
    for name, dtype in (
        ("age", torch.int64),
        ("occupancy", torch.bool),
        ("source_module", torch.int64),
        ("confidence", slots.dtype),
        ("ttl", slots.dtype),
        ("write_count", torch.int64),
    ):
        tensor = getattr(state, name)
        if (
            type(tensor) is not torch.Tensor
            or tensor.layout != torch.strided
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != slots.device
        ):
            raise WorkspaceInterventionError(f"Workspace {name} has invalid shape/dtype/device")
    for tensor in (slots, state.confidence, state.ttl):
        if not bool(torch.isfinite(tensor).all()):
            raise WorkspaceInterventionError("Workspace floating tensors must be finite")
    for tensor in (state.age, state.write_count, state.ttl):
        if not bool((tensor >= 0).all()):
            raise WorkspaceInterventionError("Workspace counters and TTL must be non-negative")
    if not bool(((state.confidence >= 0) & (state.confidence <= 1)).all()):
        raise WorkspaceInterventionError("Workspace confidence must lie in [0,1]")
    if not bool((state.source_module.masked_select(state.occupancy) >= 0).all()):
        raise WorkspaceInterventionError("Occupied slots must identify a non-negative module")
    empty = ~state.occupancy
    for field in fields(WorkspaceState):
        if field.name == "occupancy":
            continue
        tensor = getattr(state, field.name)
        mask = empty.unsqueeze(-1) if field.name == "slots" else empty
        expected = -1 if field.name == "source_module" else 0
        if not bool((tensor.masked_select(mask) == expected).all()):
            raise WorkspaceInterventionError(f"Empty slots must have canonical {field.name}")


def _clone(state: WorkspaceState) -> WorkspaceState:
    return WorkspaceState(
        **{field.name: getattr(state, field.name).clone() for field in fields(state)}
    )


def _clear(state: WorkspaceState, selection: torch.Tensor) -> WorkspaceState:
    return WorkspaceState(
        **{
            field.name: getattr(state, field.name).masked_fill(
                selection.unsqueeze(-1) if field.name == "slots" else selection,
                -1 if field.name == "source_module" else 0,
            )
            for field in fields(state)
        }
    )


def _permute(state: WorkspaceState, indices: torch.Tensor) -> WorkspaceState:
    return WorkspaceState(
        **{
            field.name: getattr(state, field.name).gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, state.slots.shape[-1])
                if field.name == "slots"
                else indices,
            )
            for field in fields(state)
        }
    )


def _generator(spec: WorkspaceInterventionSpecV1, cycle: int) -> torch.Generator:
    digest = domain_sha256(
        "iris/workspace-intervention-rng/v1",
        canonical_json_bytes({"kind": spec.kind, "seed": spec.seed, "cycle": cycle}),
    )
    seed = int(digest[:16], 16) & (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


class _StateOperation:
    _kind: ClassVar[InterventionKindV1]

    def __init__(self, spec: WorkspaceInterventionSpecV1) -> None:
        self._spec = spec

    @property
    def spec(self) -> WorkspaceInterventionSpecV1:
        """Return a validated immutable intent snapshot, never runtime state."""
        if type(self._spec) is not WorkspaceInterventionSpecV1:
            raise WorkspaceInterventionError("An exact intervention spec is required")
        spec = WorkspaceInterventionSpecV1.model_validate(self._spec)
        if spec.kind != self._kind:
            raise WorkspaceInterventionError("Operator kind does not match its specification")
        return spec

    def _check(self, state: WorkspaceState, cycle: int) -> WorkspaceInterventionSpecV1:
        spec = self.spec
        spec.active(cycle)
        validate_intervention_state(state)
        for slot in (spec.slot, spec.other_slot):
            if slot is not None and slot >= state.slots.shape[1]:
                raise WorkspaceInterventionError("Intervention slot is outside the workspace")
        if spec.capacity is not None and spec.capacity > state.slots.shape[1]:
            raise WorkspaceInterventionError("Capacity cannot increase the workspace")
        if spec.values is not None and len(spec.values) != state.slots.shape[-1]:
            raise WorkspaceInterventionError("Replacement must specify the complete latent vector")
        return spec

    def apply(self, state: WorkspaceState, cycle: int) -> WorkspaceState:
        spec = self._check(state, cycle)
        output = self._apply_active(state, spec, cycle) if spec.active(cycle) else _clone(state)
        validate_intervention_state(output)
        return output

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        raise NotImplementedError


class ZeroWorkspace(_StateOperation):
    """Clear values and all metadata, retaining the full tensor/prefix shape."""

    _kind = "zero_workspace"

    def __init__(self, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, cycles=cycles))

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        return _clear(state, torch.ones_like(state.occupancy))


class ZeroSlot(_StateOperation):
    """Clear one complete slot record for every batch row."""

    _kind = "zero_slot"

    def __init__(self, slot: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, slot=slot, cycles=cycles))

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        assert spec.slot is not None
        selection = torch.zeros_like(state.occupancy)
        selection[:, spec.slot] = True
        return _clear(state, selection)


class SwapSlots(_StateOperation):
    """Swap complete records, including occupancy, source, TTL and counters."""

    _kind = "swap_slots"

    def __init__(
        self, slot: int, other_slot: int, *, cycles: tuple[int, ...] | None = None
    ) -> None:
        super().__init__(
            WorkspaceInterventionSpecV1(
                kind=self._kind, slot=slot, other_slot=other_slot, cycles=cycles
            )
        )

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        assert spec.slot is not None and spec.other_slot is not None
        indices = torch.arange(state.slots.shape[1], device=state.slots.device)
        indices[spec.slot], indices[spec.other_slot] = spec.other_slot, spec.slot
        return _permute(state, indices.expand(state.slots.shape[0], -1))


class ShuffleSlots(_StateOperation):
    """Independently permute each row with a private CPU generator.

    An identity permutation is valid: do not condition the RNG on its outcome.
    The result is deterministic for the same shape, seed and cycle; callers that
    change dataset batching must bind per-row seeds in their evaluation protocol.
    """

    _kind = "shuffle_slots"

    def __init__(self, seed: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, seed=seed, cycles=cycles))

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        generator = _generator(spec, cycle)
        indices = torch.stack(
            [
                torch.randperm(state.slots.shape[1], generator=generator, device="cpu")
                for _ in range(state.slots.shape[0])
            ]
        ).to(device=state.slots.device)
        return _permute(state, indices)


class GaussianNoise(_StateOperation):
    """Add N(0,sigma**2) only to occupied vectors; empty records stay canonical.

    Draw the full [B,K,D] tensor independently of occupancy. Draw/scale in CPU
    float64 before casting to the state device/dtype, and reject finite overflow.
    The random stream excludes sigma so paired amplitudes use the same noise.
    """

    _kind = "gaussian_noise"

    def __init__(self, sigma: float, seed: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(
            WorkspaceInterventionSpecV1(kind=self._kind, sigma=sigma, seed=seed, cycles=cycles)
        )

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        assert spec.sigma is not None
        if spec.sigma == 0:
            return _clone(state)
        noise = torch.randn(
            tuple(state.slots.shape),
            dtype=torch.float64,
            device="cpu",
            generator=_generator(spec, cycle),
        )
        perturbation = (noise * spec.sigma).to(dtype=state.slots.dtype, device=state.slots.device)
        output = _clone(state)
        output.slots.add_(perturbation.masked_fill(~state.occupancy.unsqueeze(-1), 0))
        return output


class ReplaceSlot(_StateOperation):
    """Inject an explicit vector and metadata, with age=0 and one counted write.

    No implicit donor, source module or TTL is inferred. A zero vector may be
    occupied, just as with a regular write; the later decay policy decides expiry.
    """

    _kind = "replace_slot"

    def __init__(
        self,
        slot: int,
        values: tuple[float, ...],
        *,
        source_module: int,
        confidence: float,
        ttl: float,
        cycles: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            WorkspaceInterventionSpecV1(
                kind=self._kind,
                slot=slot,
                values=values,
                source_module=source_module,
                confidence=confidence,
                ttl=ttl,
                cycles=cycles,
            )
        )

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        assert spec.slot is not None and spec.values is not None and spec.source_module is not None
        assert spec.confidence is not None and spec.ttl is not None
        maximum = torch.finfo(state.slots.dtype).max
        if (
            any(abs(value) > maximum for value in spec.values)
            or spec.ttl > maximum
            or spec.source_module > torch.iinfo(torch.int64).max
        ):
            raise WorkspaceInterventionError(
                "Replacement metadata/vector cannot fit the state dtype"
            )
        if bool((state.write_count[:, spec.slot] == torch.iinfo(torch.int64).max).any()):
            raise WorkspaceInterventionError("Replacement write count would overflow")
        output = _clone(state)
        output.slots[:, spec.slot] = torch.tensor(
            spec.values, dtype=state.slots.dtype, device=state.slots.device
        )
        output.age[:, spec.slot] = 0
        output.occupancy[:, spec.slot] = True
        output.source_module[:, spec.slot] = spec.source_module
        output.confidence[:, spec.slot] = spec.confidence
        output.ttl[:, spec.slot] = spec.ttl
        output.write_count[:, spec.slot] += 1
        return output


class FreezeSlot(_StateOperation):
    """Capture the first selected post-write record and reuse it on selected cycles.

    This is a per-request session, not a reusable singleton. apply calls must
    strictly advance the cycle; reset starts a new request. The graph must also
    call restore_before_read after decay on later selected cycles, or expiry
    would still affect what modules read. The stored record is detached/private.
    """

    _kind = "freeze_slot"

    def __init__(self, slot: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(WorkspaceInterventionSpecV1(kind=self._kind, slot=slot, cycles=cycles))
        self._snapshot: WorkspaceState | None = None
        self._snapshot_shape: tuple[int, ...] | None = None
        self._snapshot_spec: WorkspaceInterventionSpecV1 | None = None
        self._last_cycle = 0

    def reset(self) -> None:
        self._snapshot = None
        self._snapshot_shape = None
        self._snapshot_spec = None
        self._last_cycle = 0

    def _check(self, state: WorkspaceState, cycle: int) -> WorkspaceInterventionSpecV1:
        spec = super()._check(state, cycle)
        if self._snapshot is not None and (
            tuple(state.slots.shape) != self._snapshot_shape
            or state.slots.dtype != self._snapshot.slots.dtype
            or state.slots.device != self._snapshot.slots.device
            or spec != self._snapshot_spec
        ):
            raise WorkspaceInterventionError(
                "FreezeSlot request geometry/intent changed; reset required"
            )
        return spec

    def apply(self, state: WorkspaceState, cycle: int) -> WorkspaceState:
        spec = self._check(state, cycle)
        if cycle <= self._last_cycle:
            raise WorkspaceInterventionError(
                "FreezeSlot needs increasing cycles or an explicit reset"
            )
        output = super().apply(state, cycle)
        self._last_cycle = cycle
        # No new snapshot is committed until input/output validation has succeeded.
        if spec.active(cycle) and self._snapshot is None:
            assert spec.slot is not None
            self._snapshot = WorkspaceState(
                **{
                    field.name: getattr(output, field.name)[:, spec.slot : spec.slot + 1]
                    .detach()
                    .clone()
                    for field in fields(output)
                }
            )
            self._snapshot_shape = tuple(state.slots.shape)
            self._snapshot_spec = spec
        return output

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        return self._restore(state, spec)

    def restore_before_read(self, state: WorkspaceState, cycle: int) -> WorkspaceState:
        """Restore the private record after ordinary decay, without recapturing it."""
        spec = self._check(state, cycle)
        if cycle <= self._last_cycle:
            raise WorkspaceInterventionError("Pre-read restore must precede this cycle's apply")
        output = self._restore(state, spec) if spec.active(cycle) else _clone(state)
        validate_intervention_state(output)
        return output

    def _restore(self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1) -> WorkspaceState:
        output = _clone(state)
        snapshot = self._snapshot
        if snapshot is None:
            return output
        assert spec.slot is not None
        for field in fields(state):
            getattr(output, field.name)[:, spec.slot : spec.slot + 1] = getattr(
                snapshot, field.name
            )
        return output


class ReduceWorkspaceCapacity(_StateOperation):
    """Clear the disabled suffix and explicitly constrain allocator targets.

    The graph must apply state clearing before reads and after writes, and call
    apply_allocation between ordinary allocation and writing. If capacity<R the
    first capacity ranked winners commit; the other writes are discarded in the
    disabled suffix. Original R/K/prefix geometry and competition are unchanged.
    """

    _kind = "reduce_workspace_capacity"

    def __init__(self, capacity: int, *, cycles: tuple[int, ...] | None = None) -> None:
        super().__init__(
            WorkspaceInterventionSpecV1(kind=self._kind, capacity=capacity, cycles=cycles)
        )

    def _apply_active(
        self, state: WorkspaceState, spec: WorkspaceInterventionSpecV1, cycle: int
    ) -> WorkspaceState:
        assert spec.capacity is not None
        selection = torch.zeros_like(state.occupancy)
        selection[:, spec.capacity :] = True
        return _clear(state, selection)

    def apply_allocation(
        self, state: WorkspaceState, allocation: "SlotAllocatorOutput", cycle: int
    ) -> "SlotAllocatorOutput":
        # Lazy import keeps pure state operations independent of phase validators.
        from iris.interventions.capacity import restrict_allocation_capacity

        spec = self._check(state, cycle)
        assert spec.capacity is not None
        capacity = spec.capacity if spec.active(cycle) else state.slots.shape[1]
        return restrict_allocation_capacity(state, allocation, capacity=capacity)


__all__ = [
    "FreezeSlot",
    "GaussianNoise",
    "ReduceWorkspaceCapacity",
    "ReplaceSlot",
    "ShuffleSlots",
    "SwapSlots",
    "WorkspaceIntervention",
    "WorkspaceInterventionError",
    "ZeroSlot",
    "ZeroWorkspace",
    "validate_intervention_state",
]
