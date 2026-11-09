"""Owned cognitive-ablation cadence/best controller; durable lifecycle IO is separate."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from weakref import ReferenceType, WeakKeyDictionary, ref

from pydantic import BaseModel

from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointBundleV1,
    _verify,
    build_scientific_ablation_checkpoint_v1,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    IrisScientificAblationLifecycleReceiptV1,
    _bundle_parts,
    build_scientific_ablation_lifecycle_checkpoint_v1,
    write_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
)
from iris.application.scientific_ablation_run import (
    IrisScientificAblationRunV1,
    _ScientificAblationControllerAttachment,
    _ScientificAblationDelegation,
)
from iris.application.scientific_ablation_validation import (
    IrisScientificAblationValidationV1,
    _evaluate_scientific_ablation_validation_v1,
    _has_signal,
    _snapshot,
    _training_corpus_sha256,
    own_scientific_ablation_validation_rows_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    IrisScientificAblationLifecyclePolicyV1,
    ScientificAblationLifecycleStateV1,
)
from iris.data.schemas import IrisExample
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state

_CONFIG_DOMAIN = "iris/scientific-ablation-lifecycle-configuration/v1"
_OWNERS: WeakKeyDictionary[IrisScientificAblationRunV1, tuple[object, ReferenceType[object]]] = (
    WeakKeyDictionary()
)


class IrisScientificAblationLifecycleError(RuntimeError):
    """Mode-bound controller ownership, cadence or a phase operation failed."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationLifecycleBoundaryV1:
    """Actual in-memory current/best boundary, NOT a durable publication receipt."""

    configuration_json: str
    configuration_sha256: str
    state: ScientificAblationLifecycleStateV1
    current: IrisScientificAblationCheckpointBundleV1
    best: IrisScientificAblationCheckpointBundleV1 | None


def _exact(instance: BaseModel, expected: type[BaseModel]) -> None:
    if (
        type(instance) is not expected
        or set(vars(instance)) != set(expected.model_fields)
        or instance.__pydantic_extra__ is not None
    ):
        raise ValueError("Exact complete controller instance fields required")


def _validate_cadence(
    state: ScientificAblationLifecycleStateV1,
    policy: IrisScientificAblationLifecyclePolicyV1,
    maximum: int,
) -> None:
    """Validate phase counters against the declared schedule, including pending phases."""
    _exact(state, ScientificAblationLifecycleStateV1)
    _exact(policy, IrisScientificAblationLifecyclePolicyV1)
    state = ScientificAblationLifecycleStateV1.model_validate(state.model_dump(mode="python"))
    policy = IrisScientificAblationLifecyclePolicyV1.model_validate(
        policy.model_dump(mode="python")
    )
    step, phase = state.step, state.phase
    terminal = phase == "stopped"
    pending_validation = phase == "validation"
    pending_checkpoint = phase in ("validation", "checkpoint")
    if (
        step > maximum
        or (not step and phase != "train")
        or (phase == "train" and step == maximum)
        or (pending_validation and step % policy.validate_every_steps != 0 and step != maximum)
        or (
            phase == "checkpoint"
            and step % policy.checkpoint_every_steps != 0
            and state.stop_reason is None
        )
    ):
        raise ValueError("Controller phase differs from declared cadence/budget")
    validated_through = step - int(pending_validation)
    checkpointed_through = step - int(pending_checkpoint)
    validation_count = validated_through // policy.validate_every_steps
    last_validation = validation_count * policy.validate_every_steps
    if not pending_validation and step == maximum and step % policy.validate_every_steps:
        validation_count += 1
        last_validation = step
    checkpoint_count = checkpointed_through // policy.checkpoint_every_steps
    last_checkpoint = checkpoint_count * policy.checkpoint_every_steps
    if terminal and step % policy.checkpoint_every_steps:
        checkpoint_count += 1
        last_checkpoint = step
    reason = (
        "early_stopping"
        if state.nonimproving_validations >= policy.early_stopping_patience
        else "step_budget"
        if step == maximum and not pending_validation
        else None
    )
    if (
        state.validation_count != validation_count
        or state.last_validation_step != last_validation
        or state.checkpoint_count != checkpoint_count
        or state.last_checkpoint_step != last_checkpoint
        or state.stop_reason != reason
        or state.nonimproving_validations > policy.early_stopping_patience
        or (
            (reason is not None)
            != (phase in ("checkpoint", "stopped") and state.stop_reason is not None)
        )
    ):
        raise ValueError("Controller counters/stopping differ from declared cadence")
    if state.best_step is not None:
        best_index = state.best_step // policy.validate_every_steps
        if state.best_step % policy.validate_every_steps:
            if state.best_step != maximum:
                raise ValueError("Selected best is not a validation step")
            best_index += 1
        if state.nonimproving_validations != state.validation_count - best_index:
            raise ValueError("Controller staleness differs from selected best")


class IrisScientificAblationLifecycleV1:
    """One owned phase per advance, with exclusive delegated run execution.

    Default checkpoint phases return immutable in-memory boundaries. An explicit
    checkpoint_root publishes durable boundaries and returns verified receipts;
    failed publication retains exact pending bytes and cannot retrain or evaluate.
    """

    def __init__(
        self,
        *,
        run: IrisScientificAblationRunV1,
        validation: tuple[IrisExample, ...],
        policy: IrisScientificAblationLifecyclePolicyV1,
        checkpoint_root: Path | None = None,
    ) -> None:
        if checkpoint_root is not None and (
            not isinstance(checkpoint_root, Path) or not checkpoint_root.name
        ):
            raise IrisScientificAblationLifecycleError("Explicit lifecycle storage root required")
        if type(run) is not IrisScientificAblationRunV1 or run._busy or run.completed_steps:
            raise IrisScientificAblationLifecycleError("Fresh idle exact ablation run required")
        existing = _OWNERS.get(run)
        if (
            existing is not None
            or run._controller_attachment is not None
            or run._controller_attachment_guard is not None
        ):
            raise IrisScientificAblationLifecycleError("Ablation run already has a controller")
        self._run = self._owned_run = run
        self._attachment_token = object()
        self._run_attachment = _ScientificAblationControllerAttachment(
            self._attachment_token, ref(self)
        )
        self._delegation: _ScientificAblationDelegation | None = None
        self._poisoned = self._busy = False
        self._best: IrisScientificAblationCheckpointBundleV1 | None = None
        self._checkpoint_root = None if checkpoint_root is None else checkpoint_root.absolute()
        self._pending: IrisScientificAblationLifecycleBundleV1 | None = None
        self._state = ScientificAblationLifecycleStateV1()
        caller = capture_runtime_state(run._optimizer._device)
        primary: BaseException | None = None
        cleanup_signal: BaseException | None = None
        prepared = False
        cleanup_failed = False
        before: tuple[object, ...] | None = None
        previous_attachment = run._controller_attachment
        previous_guard = run._controller_attachment_guard

        def rollback_attachment() -> None:
            # Restore partial installations independently, but never erase a
            # foreign replacement or silently legitimize it with a new seal.
            entry = _OWNERS.get(run)
            if entry is not None and (
                entry[0] is self._attachment_token and entry[1] is self._run_attachment.owner
            ):
                if existing is None:
                    del _OWNERS[run]
                else:
                    _OWNERS[run] = existing
            elif entry is not existing:
                self._poison()
            if run._controller_attachment is self._run_attachment:
                run._controller_attachment = previous_attachment
            elif run._controller_attachment is not previous_attachment:
                self._poison()
            if run._controller_attachment_guard is self._run_attachment:
                run._controller_attachment_guard = previous_guard
            elif run._controller_attachment_guard is not previous_guard:
                self._poison()

        run._busy = True
        try:
            _OWNERS[run] = self._attachment_token, self._run_attachment.owner
            run._controller_attachment = run._controller_attachment_guard = self._run_attachment
            run._validate_boundary()
            before = _snapshot(run)
            _training_corpus_sha256(run)
            _exact(policy, IrisScientificAblationLifecyclePolicyV1)
            self._policy = IrisScientificAblationLifecyclePolicyV1.model_validate(
                policy.model_dump(mode="python")
            )
            if self._policy.purpose != run._optimizer._policy.purpose:
                raise ValueError("Controller and optimizer purposes differ")
            rows, digest = own_scientific_ablation_validation_rows_v1(
                validation, training_rows=run._rows
            )
            if self._policy.purpose == "scientific" and (
                len(run._rows) != 50000 or len(rows) != 5000
            ):
                raise ValueError("Scientific controller requires source50000/5000splits")
            self._validation = self._owned_validation = rows
            self._validation_digest = digest
            self._validation_target_tokens = sum(
                int(
                    prepare_teacher_forcing_input(
                        backbone=run._backbone,
                        prompt=row.prompt,
                        target_text=row.expected_response,
                        terminal_token_id=run._terminal_token_id,
                    ).target_attention_mask.sum()
                )
                for row in rows
            )
            self._configuration_json = canonical_json_bytes(
                {
                    "kind": "iris-scientific-ablation-lifecycle-v1",
                    "run_sha256": run.configuration_sha256,
                    "ablation_policy_sha256": run._ablation_identity,
                    "validation_sha256": digest,
                    "validation_examples": len(rows),
                    "validation_target_tokens": self._validation_target_tokens,
                    "policy": self._policy.model_dump(mode="json"),
                }
            ).decode()
            self._configuration_sha256 = domain_sha256(
                _CONFIG_DOMAIN, self._configuration_json.encode()
            )
            self._run_snapshot = before
            self._private_runtime = run._clone_runtime(run._runtime)
            self._control_seal = self._control()
            self._validate_locked()
            prepared = True
        except BaseException as error:
            if _has_signal(error):
                primary = error
                raise
            raise IrisScientificAblationLifecycleError("Controller could not attach") from None
        finally:
            # A failed provisional install may contain only the registry claim;
            # undo it before boundary validation so our own partial claim is not
            # mistaken for an externally corrupted, stateful run mutation.
            if not prepared:
                try:
                    rollback_attachment()
                except BaseException as error:
                    cleanup_failed = True
                    self._poison()
                    if _has_signal(error):
                        cleanup_signal = error
            try:
                run._validate_boundary()
                if before is not None and _snapshot(run) != before:
                    raise ValueError("Controller attachment mutated its run")
            except BaseException as error:
                cleanup_failed = True
                self._poison()
                if _has_signal(error) and cleanup_signal is None:
                    cleanup_signal = error
            try:
                self._restore_caller(caller, primary if primary is not None else cleanup_signal)
            except BaseException as error:
                cleanup_failed = True
                if _has_signal(error) and cleanup_signal is None:
                    cleanup_signal = error
            finally:
                run._busy = False
            # Attachment is committed only after the caller cleanup has succeeded.
            if not prepared or cleanup_failed or self.poisoned:
                try:
                    rollback_attachment()
                except BaseException as error:
                    cleanup_failed = True
                    self._poison()
                    if _has_signal(error) and cleanup_signal is None:
                        cleanup_signal = error
            if primary is None and cleanup_signal is not None:
                raise cleanup_signal
            if prepared and (cleanup_failed or self.poisoned):
                raise IrisScientificAblationLifecycleError("Controller attachment cleanup failed")

    @property
    def poisoned(self) -> bool:
        return self._poisoned or self._owned_run.poisoned

    @property
    def configuration_json(self) -> str:
        return self._configuration_json

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    def _poison(self) -> None:
        self._poisoned = self._owned_run._poisoned = self._owned_run._optimizer._poisoned = True

    def _restore_caller(self, caller: RuntimeStateSnapshot, primary: BaseException | None) -> None:
        try:
            restore_runtime_state(caller)
            if not _same_runtime(caller, capture_runtime_state(self._owned_run._optimizer._device)):
                raise ValueError("Caller runtime was not restored")
        except BaseException as error:
            self._poison()
            if primary is None:
                if _has_signal(error):
                    raise
                raise IrisScientificAblationLifecycleError(
                    "Controller runtime cleanup failed"
                ) from None

    def _control(
        self,
        state: ScientificAblationLifecycleStateV1 | None = None,
        best: IrisScientificAblationCheckpointBundleV1 | None = None,
        *,
        replacement: bool = False,
        pending: IrisScientificAblationLifecycleBundleV1 | None = None,
        pending_replacement: bool = False,
    ) -> tuple[object, ...]:
        current_state = self._state if state is None else state
        current_best = best if replacement else self._best
        current_pending = pending if pending_replacement else self._pending
        _exact(self._policy, IrisScientificAblationLifecyclePolicyV1)
        _exact(current_state, ScientificAblationLifecycleStateV1)
        _validate_cadence(
            current_state, self._policy, self._owned_run._optimizer._policy.max_optimizer_steps
        )
        _, digest = own_scientific_ablation_validation_rows_v1(
            self._validation, training_rows=self._owned_run._rows
        )
        if digest != self._validation_digest or self._validation is not self._owned_validation:
            raise ValueError("Owned validation corpus changed")
        configuration = canonical_json_bytes(
            {
                "kind": "iris-scientific-ablation-lifecycle-v1",
                "run_sha256": self._owned_run.configuration_sha256,
                "ablation_policy_sha256": self._owned_run._ablation_identity,
                "validation_sha256": digest,
                "validation_examples": len(self._validation),
                "validation_target_tokens": self._validation_target_tokens,
                "policy": self._policy.model_dump(mode="json"),
            }
        ).decode()
        if (
            configuration != self._configuration_json
            or domain_sha256(_CONFIG_DOMAIN, configuration.encode()) != self._configuration_sha256
        ):
            raise ValueError("Controller configuration changed")
        best_identity = None
        if current_best is not None:
            manifest, _ = _verify(current_best)
            if (
                manifest.payload.configuration_sha256 != self._owned_run.configuration_sha256
                or manifest.payload.completed_steps != current_state.best_step
            ):
                raise ValueError("Selected best differs from its controller")
            best_identity = manifest.sha256
        if (current_state.best_step is None) != (current_best is None):
            raise ValueError("Selected best presence differs from its controller")
        pending_identity = None
        if current_pending is not None:
            pending_manifest, _, _ = _bundle_parts(current_pending)
            payload = pending_manifest.payload
            if (
                self._checkpoint_root is None
                or current_state.phase != "checkpoint"
                or payload.state != self._checkpoint_state(current_state)
                or payload.configuration_json != self._configuration_json
                or payload.configuration_sha256 != self._configuration_sha256
                or payload.current.payload.completed_steps != current_state.step
                or payload.current.payload.scientific_state_sha256
                != self._owned_run._optimizer._committed_scientific_state_sha256
                or (None if payload.best is None else payload.best.sha256) != best_identity
            ):
                raise ValueError("Pending publication differs from its owned checkpoint phase")
            pending_identity = pending_manifest.sha256
        return (
            id(self._run),
            id(self._attachment_token),
            id(self._run_attachment),
            self._configuration_json,
            self._configuration_sha256,
            canonical_json_bytes(current_state.model_dump(mode="json")),
            best_identity,
            digest,
            self._checkpoint_root,
            pending_identity,
        )

    def _validate_locked(self) -> None:
        owner = _OWNERS.get(self._owned_run)
        if (
            self.poisoned
            or self._run is not self._owned_run
            or owner is None
            or (owner[0] is not self._attachment_token or owner[1]() is not self)
            or self._owned_run._controller_attachment is not self._run_attachment
            or self._owned_run._controller_attachment_guard is not self._run_attachment
            or self._run_attachment.token is not self._attachment_token
            or self._run_attachment.owner is not owner[1]
            or self._delegation is not None
        ):
            raise ValueError("Controller/run attachment ownership changed")
        self._owned_run._validate_boundary()
        _training_corpus_sha256(self._owned_run)
        if (
            self._control() != self._control_seal
            or self._state.step != self._owned_run.completed_steps
            or _snapshot(self._owned_run) != self._run_snapshot
            or not _same_runtime(self._private_runtime, self._owned_run._runtime)
        ):
            raise ValueError("Controller no longer owns the exact run/control boundary")

    @contextmanager
    def _delegate(self, operation: str) -> Iterator[_ScientificAblationDelegation]:
        if not self._busy or operation not in ("train", "validation", "restore"):
            raise ValueError("Active exact controller operation required for delegation")
        self._validate_locked()
        capability = _ScientificAblationDelegation(self._run_attachment, operation)
        self._delegation = capability
        try:
            yield capability
        finally:
            # Child entry consumes this exact one-call capability. An untouched
            # failure/mock may leave it unused; neither case grants future access.
            if self._delegation is capability:
                self._delegation = None
            elif self._delegation is not None:
                self._poison()

    @contextmanager
    def _operation(self) -> Iterator[None]:
        run = self._owned_run
        if self._busy or run._busy:
            raise IrisScientificAblationLifecycleError("Reentrant controller operation forbidden")
        caller = capture_runtime_state(run._optimizer._device)
        self._busy = run._busy = True
        primary: BaseException | None = None
        cleanup_failed = False
        cleanup_signal: BaseException | None = None
        try:
            self._validate_locked()
            yield
        except BaseException as error:
            if _has_signal(error):
                primary = error
                raise
            raise IrisScientificAblationLifecycleError(
                "Controller phase could not complete"
            ) from None
        finally:
            run._busy = True
            try:
                self._validate_locked()
            except BaseException as error:
                cleanup_failed = True
                self._poison()
                if _has_signal(error):
                    cleanup_signal = error
            try:
                self._restore_caller(caller, primary if primary is not None else cleanup_signal)
            finally:
                self._busy = run._busy = False
            if cleanup_failed and primary is None:
                if cleanup_signal is not None:
                    raise cleanup_signal
                raise IrisScientificAblationLifecycleError(
                    "Controller owned-state cleanup failed"
                ) from None

    @property
    def state(self) -> ScientificAblationLifecycleStateV1:
        with self._operation():
            return deepcopy(self._state)

    @property
    def best_checkpoint(self) -> IrisScientificAblationCheckpointBundleV1 | None:
        with self._operation():
            return deepcopy(self._best)

    @property
    def publication_pending(self) -> bool:
        with self._operation():
            return self._pending is not None

    @staticmethod
    def _checkpoint_state(
        state: ScientificAblationLifecycleStateV1,
    ) -> ScientificAblationLifecycleStateV1:
        return ScientificAblationLifecycleStateV1.model_validate(
            state.model_dump(mode="python")
            | {
                "phase": "stopped" if state.stop_reason is not None else "train",
                "checkpoint_count": state.checkpoint_count + 1,
                "last_checkpoint_step": state.step,
            }
        )

    def _publish_pending(self) -> IrisScientificAblationLifecycleReceiptV1:
        if self._pending is None or self._checkpoint_root is None:
            raise ValueError("Owned pending lifecycle publication required")
        pending = self._pending
        manifest, _, _ = _bundle_parts(pending)
        path = write_scientific_ablation_lifecycle_checkpoint_v1(
            pending, root=self._checkpoint_root
        )
        receipt = IrisScientificAblationLifecycleReceiptV1(path, deepcopy(manifest))
        self._commit(deepcopy(manifest.payload.state), self._best, clear_pending=True)
        return receipt

    def _commit(
        self,
        state: ScientificAblationLifecycleStateV1,
        best: IrisScientificAblationCheckpointBundleV1 | None,
        *,
        clear_pending: bool = False,
    ) -> None:
        if self._control() != self._control_seal:
            raise ValueError("Controller state changed during its owned phase")
        self._owned_run._validate_boundary()
        _training_corpus_sha256(self._owned_run)
        following = self._control(state, best, replacement=True, pending_replacement=clear_pending)
        snapshot = _snapshot(self._owned_run)
        runtime = self._owned_run._clone_runtime(self._owned_run._runtime)
        (
            self._state,
            self._best,
            self._control_seal,
            self._run_snapshot,
            self._private_runtime,
            self._pending,
        ) = (
            state,
            best,
            following,
            snapshot,
            runtime,
            None if clear_pending else self._pending,
        )

    def advance(
        self,
    ) -> (
        IrisScientificAblationOptimizationOutputV1
        | IrisScientificAblationValidationV1
        | IrisScientificAblationLifecycleBoundaryV1
        | IrisScientificAblationLifecycleReceiptV1
    ):
        with self._operation():
            state, policy, run = self._state, self._policy, self._owned_run
            if self._pending is not None:
                return self._publish_pending()
            if state.phase == "stopped":
                raise ValueError("Scientific ablation controller has stopped")
            run._busy = False
            try:
                if state.phase == "train":
                    with self._delegate("train") as capability:
                        result = run._step_authorized(capability)
                    if (
                        type(result) is not IrisScientificAblationOptimizationOutputV1
                        or type(result.step_index) is not int
                        or result.step_index != state.step + 1
                        or result.step_index != run.completed_steps
                        or result.ablation_policy_sha256 != run._ablation_identity
                        or result.model_configuration_sha256 != run._model_configuration_identity
                        or result.scientific_state_after_sha256
                        != run._optimizer._committed_scientific_state_sha256
                    ):
                        raise ValueError(
                            "Controller train result does not describe its committed run"
                        )
                    step = run.completed_steps
                    final = step == run._optimizer._policy.max_optimizer_steps
                    phase = (
                        "validation"
                        if final or step % policy.validate_every_steps == 0
                        else (
                            "checkpoint" if step % policy.checkpoint_every_steps == 0 else "train"
                        )
                    )
                    following = ScientificAblationLifecycleStateV1.model_validate(
                        state.model_dump(mode="python") | {"phase": phase, "step": step}
                    )
                    run._busy = True
                    self._commit(following, self._best)
                    return result
                if state.phase == "validation":
                    with self._delegate("validation") as capability:
                        report = _evaluate_scientific_ablation_validation_v1(
                            run, self._validation, capability=capability
                        )
                    run._busy = True
                    if (
                        type(report) is not IrisScientificAblationValidationV1
                        or report.dataset_sha256 != self._validation_digest
                        or type(report.example_count) is not int
                        or report.example_count != len(self._validation)
                        or type(report.target_token_count) is not int
                        or report.target_token_count != self._validation_target_tokens
                        or type(report.at_optimizer_step) is not int
                        or report.at_optimizer_step != state.step
                        or report.run_configuration_sha256 != run.configuration_sha256
                        or report.model_configuration_sha256 != run._model_configuration_identity
                        or report.ablation_policy_sha256 != run._ablation_identity
                        or report.scientific_state_sha256
                        != run._optimizer._committed_scientific_state_sha256
                        or type(report.target_nll_sum) is not float
                        or type(report.target_nll_per_token) is not float
                        or not math.isfinite(report.target_nll_sum)
                        or report.target_nll_sum < 0
                        or report.target_nll_per_token
                        != report.target_nll_sum / report.target_token_count
                    ):
                        raise ValueError("Controller validation report provenance differs")
                    if _snapshot(run) != self._run_snapshot or not _same_runtime(
                        self._private_runtime, run._runtime
                    ):
                        raise ValueError("Validation changed the controlled run")
                    better = state.best_nll_per_token is None or report.target_nll_per_token < (
                        state.best_nll_per_token - policy.minimum_improvement
                    )
                    run._busy = False
                    best = build_scientific_ablation_checkpoint_v1(run) if better else self._best
                    run._busy = True
                    misses = 0 if better else state.nonimproving_validations + 1
                    reason = (
                        "early_stopping"
                        if misses >= policy.early_stopping_patience
                        else (
                            "step_budget"
                            if state.step == run._optimizer._policy.max_optimizer_steps
                            else None
                        )
                    )
                    following = ScientificAblationLifecycleStateV1.model_validate(
                        state.model_dump(mode="python")
                        | {
                            "phase": "checkpoint"
                            if reason is not None or state.step % policy.checkpoint_every_steps == 0
                            else "train",
                            "validation_count": state.validation_count + 1,
                            "last_validation_step": state.step,
                            "best_step": state.step if better else state.best_step,
                            "best_nll_per_token": report.target_nll_per_token
                            if better
                            else state.best_nll_per_token,
                            "nonimproving_validations": misses,
                            "stop_reason": reason,
                        }
                    )
                    self._commit(following, best)
                    return report
                current = build_scientific_ablation_checkpoint_v1(run)
                run._busy = True
                if _snapshot(run) != self._run_snapshot:
                    raise ValueError("Checkpoint changed the controlled run")
                following = self._checkpoint_state(state)
                best = (
                    current
                    if self._best is not None and self._best == current
                    else deepcopy(self._best)
                )
                boundary = IrisScientificAblationLifecycleBoundaryV1(
                    self._configuration_json,
                    self._configuration_sha256,
                    deepcopy(following),
                    current,
                    best,
                )
                if self._checkpoint_root is not None:
                    pending = build_scientific_ablation_lifecycle_checkpoint_v1(boundary)
                    control = self._control(pending=pending, pending_replacement=True)
                    self._pending, self._control_seal = pending, control
                    return self._publish_pending()
                self._commit(following, self._best)
                return boundary
            finally:
                run._busy = True


__all__ = [
    "IrisScientificAblationLifecycleBoundaryV1",
    "IrisScientificAblationLifecycleError",
    "IrisScientificAblationLifecycleV1",
]
