"""Bounded genuine prewrite/native-write observations, separate from V1 math.

These hashes bind retained tensors to a controlled invocation. They do not
attest execution or prove an intervention's correctness on their own.
"""

from dataclasses import dataclass, fields
from re import fullmatch

import torch

from iris.interventions.state import validate_intervention_state
from iris.training.ablation_controls import IrisAblationControlledTeacherOutputV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.workspace.state import WorkspaceState


def ablation_workspace_observation_scalars_v1(*, cycles: int, slots: int, width: int) -> int:
    """Two complete seven-field states per cycle; NOT a peak-memory bound."""
    if any(type(n) is not int or n < 1 for n in (cycles, slots, width)):
        raise ValueError("Exact positive observation geometry required")
    return 2 * cycles * slots * (width + 6)


def _state(state: WorkspaceState, *, slots: int, width: int, modules: int) -> None:
    validate_intervention_state(state)
    if tuple(state.slots.shape) != (1, slots, width) or any(
        type(getattr(state, f.name)) is not torch.Tensor
        or getattr(state, f.name).requires_grad
        or getattr(state, f.name).grad_fn is not None
        or getattr(state, f.name).is_inference()
        for f in fields(WorkspaceState)
    ):
        raise ValueError("Exact detached batch-one observation state required")
    if not bool((state.source_module.masked_select(state.occupancy) < modules).all()):
        raise ValueError("Observed workspace source is outside its actual graph")


def _copy(state: WorkspaceState) -> WorkspaceState:
    return WorkspaceState(
        **{f.name: getattr(state, f.name).detach().clone() for f in fields(state)}
    )


def _sha256(digest: object) -> bool:
    return type(digest) is str and fullmatch(r"[0-9a-f]{64}", digest) is not None


def ablation_workspace_state_sha256_v1(state: WorkspaceState) -> str:
    if type(state) is not WorkspaceState:
        raise TypeError("Exact seven-field workspace required")
    return domain_sha256(
        "iris/ablation-workspace-observed-state/v1",
        canonical_json_bytes(
            {f.name: tensor_sha256(getattr(state, f.name)) for f in fields(state)}
        ),
    )


@dataclass(frozen=True, slots=True)
class AblationWorkspaceWriteObservationV1:
    cycle: int
    before_write: WorkspaceState
    native_written: WorkspaceState
    before_write_sha256: str
    native_written_sha256: str


def _invocation(
    controlled: IrisAblationControlledTeacherOutputV1,
    observations: tuple[AblationWorkspaceWriteObservationV1, ...],
    maximum_scalars: int,
    actual_scalars: int,
) -> str:
    return domain_sha256(
        "iris/ablation-workspace-observed-teacher-invocation/v1",
        canonical_json_bytes(
            dict(
                controlled_invocation_sha256=controlled.invocation_sha256,
                maximum_observation_scalars=maximum_scalars,
                actual_observation_scalars=actual_scalars,
                observations=[
                    dict(
                        cycle=o.cycle,
                        before_write_sha256=o.before_write_sha256,
                        native_written_sha256=o.native_written_sha256,
                    )
                    for o in observations
                ],
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class IrisAblationObservedTeacherOutputV1:
    controlled: IrisAblationControlledTeacherOutputV1
    observations: tuple[AblationWorkspaceWriteObservationV1, ...]
    maximum_observation_scalars: int
    actual_observation_scalars: int
    invocation_sha256: str
    kind: str = "iris-ablation-observed-teacher-output-v1"

    def __post_init__(self) -> None:
        if (
            type(self) is not IrisAblationObservedTeacherOutputV1
            or type(self.kind) is not str
            or self.kind != "iris-ablation-observed-teacher-output-v1"
            or type(self.controlled) is not IrisAblationControlledTeacherOutputV1
            or type(self.controlled.kind) is not str
            or self.controlled.kind != "iris-ablation-controlled-teacher-output-v1"
            or not _sha256(self.invocation_sha256)
            or any(
                not _sha256(getattr(self.controlled, name))
                for name in (
                    "plan_sha256",
                    "ablation_policy_sha256",
                    "control_plan_sha256",
                    "prompt_sha256",
                    "target_sha256",
                    "teacher_graph_sha256",
                    "sidecar_state_sha256",
                    "invocation_sha256",
                )
            )
            or type(self.observations) is not tuple
            or type(self.maximum_observation_scalars) is not int
            or not 1 <= self.maximum_observation_scalars <= 2_000_000
            or type(self.actual_observation_scalars) is not int
        ):
            raise ValueError("Exact bounded distinct observed teacher receipt required")
        self.controlled.__post_init__()
        trace = self.controlled.diagnostic_trace
        expected = ablation_workspace_observation_scalars_v1(
            cycles=trace.cycle_count, slots=trace.num_slots, width=trace.state_dim
        )
        if (
            trace.options.mode != "full"
            or self.actual_observation_scalars != expected
            or expected > self.maximum_observation_scalars
            or len(self.observations) != trace.cycle_count
        ):
            raise ValueError("Observation geometry or scalar preflight differs")
        storage: set[tuple[torch.device, int]] = set()
        for cycle, observation in enumerate(self.observations, 1):
            if type(observation) is not AblationWorkspaceWriteObservationV1 or (
                type(observation.cycle) is not int or observation.cycle != cycle
            ):
                raise ValueError("Complete ordered actual writer observations required")
            for state, digest in (
                (observation.before_write, observation.before_write_sha256),
                (observation.native_written, observation.native_written_sha256),
            ):
                if not _sha256(digest):
                    raise ValueError("Exact lowercase writer observation SHA256 required")
                _state(
                    state, slots=trace.num_slots, width=trace.state_dim, modules=trace.num_modules
                )
                if digest != ablation_workspace_state_sha256_v1(state):
                    raise ValueError("Retained writer observation tensors changed")
                for f in fields(state):
                    tensor = getattr(state, f.name)
                    identity = (tensor.device, tensor.untyped_storage().data_ptr())
                    if identity in storage:
                        raise ValueError("Writer observations must own independent tensor storage")
                    storage.add(identity)
        for state in self.controlled.workspace_trace:
            if any(
                (getattr(state, f.name).device, getattr(state, f.name).untyped_storage().data_ptr())
                in storage
                for f in fields(state)
            ):
                raise ValueError("Observations cannot alias controlled postwrite tensors")
        if self.invocation_sha256 != _invocation(
            self.controlled, self.observations, self.maximum_observation_scalars, expected
        ):
            raise ValueError("Distinct observed invocation binding differs")


class _AblationWorkspaceObservationSessionV1:
    """Native-only two-phase recorder; no user callback or module hooks."""

    def __init__(self, *, cycles: int, slots: int, width: int, modules: int, maximum: int) -> None:
        scalars = ablation_workspace_observation_scalars_v1(cycles=cycles, slots=slots, width=width)
        if (
            type(modules) is not int
            or modules < 1
            or type(maximum) is not int
            or not scalars <= maximum <= 2_000_000
        ):
            raise ValueError("Complete writer observation budget exceeded before execution")
        self._geometry = (cycles, slots, width, modules)
        self._maximum, self._scalars = maximum, scalars
        self._context: tuple[str, ...] | None = None
        self._observations: list[AblationWorkspaceWriteObservationV1] = []
        self._pending: WorkspaceState | None = None
        self._original: WorkspaceState | None = None
        self._versions: tuple[tuple[int, int], ...] = ()
        self._failed = False
        self._finished = False

    def claim(self, *, context: tuple[str, ...], geometry: tuple[int, int, int, int]) -> None:
        try:
            if (
                type(self) is not _AblationWorkspaceObservationSessionV1
                or self._context is not None
                or self._failed
                or self._finished
                or geometry != self._geometry
                or self._observations
                or self._pending is not None
                or self._original is not None
                or self._versions
                or set(vars(self))
                != {
                    "_geometry",
                    "_maximum",
                    "_scalars",
                    "_context",
                    "_observations",
                    "_pending",
                    "_original",
                    "_versions",
                    "_failed",
                    "_finished",
                }
            ):
                raise ValueError("Writer observation session is foreign or consumed")
            self._context = context
        except BaseException:
            self._failed = True
            raise

    def _check(self, state: WorkspaceState, cycle: int, *, pending: bool) -> None:
        c, k, d, m = self._geometry
        if (
            self._context is None
            or self._failed
            or self._finished
            or type(cycle) is not int
            or cycle != len(self._observations) + 1
            or cycle > c
            or (self._pending is not None) != pending
        ):
            raise ValueError("Writer observation phase is invalid or closed")
        _state(state, slots=k, width=d, modules=m)

    def before_write(self, state: WorkspaceState, cycle: int) -> None:
        try:
            self._check(state, cycle, pending=False)
            self._pending, self._original = _copy(state), state
            self._versions = tuple(
                (id(getattr(state, f.name)), getattr(state, f.name)._version) for f in fields(state)
            )
        except BaseException:
            self._failed = True
            raise

    def native_written(self, before: WorkspaceState, written: WorkspaceState, cycle: int) -> None:
        try:
            self._check(written, cycle, pending=True)
            pending = self._pending
            assert pending is not None
            if (
                before is not self._original
                or any(
                    not torch.equal(getattr(before, f.name), getattr(pending, f.name))
                    for f in fields(before)
                )
                or self._versions
                != tuple(
                    (id(getattr(before, f.name)), getattr(before, f.name)._version)
                    for f in fields(before)
                )
            ):
                raise ValueError("Native writer mutated its original prewrite input")
            owned = _copy(written)
            self._observations.append(
                AblationWorkspaceWriteObservationV1(
                    cycle=cycle,
                    before_write=pending,
                    native_written=owned,
                    before_write_sha256=ablation_workspace_state_sha256_v1(pending),
                    native_written_sha256=ablation_workspace_state_sha256_v1(owned),
                )
            )
            self._pending = self._original = None
            self._versions = ()
        except BaseException:
            self._failed = True
            raise

    def finish(
        self, controlled: IrisAblationControlledTeacherOutputV1
    ) -> IrisAblationObservedTeacherOutputV1:
        try:
            if (
                self._failed
                or self._finished
                or self._pending is not None
                or self._original is not None
                or self._versions
                or len(self._observations) != self._geometry[0]
                or type(controlled) is not IrisAblationControlledTeacherOutputV1
                or self._context
                != (
                    controlled.plan_sha256,
                    controlled.prompt_sha256,
                    controlled.target_sha256,
                    controlled.teacher_graph_sha256,
                    controlled.sidecar_state_sha256,
                )
            ):
                raise ValueError("Writer observation did not complete its original invocation")
            observations = tuple(self._observations)
            result = IrisAblationObservedTeacherOutputV1(
                controlled=controlled,
                observations=observations,
                maximum_observation_scalars=self._maximum,
                actual_observation_scalars=self._scalars,
                invocation_sha256=_invocation(
                    controlled, observations, self._maximum, self._scalars
                ),
            )
            self._finished = True
            return result
        except BaseException:
            self._failed = True
            raise


__all__ = [
    "AblationWorkspaceWriteObservationV1",
    "IrisAblationObservedTeacherOutputV1",
    "ablation_workspace_observation_scalars_v1",
    "ablation_workspace_state_sha256_v1",
]
