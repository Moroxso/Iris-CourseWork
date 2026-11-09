"""Mode-bound scientific cognitive-ablation corpus, cursor and runtime ownership.

This in-memory stream is not a durable checkpoint, launcher or study result.
Historical scientific-D run/checkpoint/delivery APIs retain their exact-type guards.
"""

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from weakref import ReferenceType

from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
    IrisScientificAblationOptimizerV1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunError, _ScientificRunCore
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.data.schemas import IrisExample
from iris.training.intervention import teacher_target_sha256_v1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state


class IrisScientificAblationRunError(IrisScientificRunError):
    """A mode-bound stream failed; uncertain committed state cannot be reused."""


@dataclass(frozen=True, slots=True)
class _ScientificAblationControllerAttachment:
    token: object
    owner: ReferenceType[object]


@dataclass(frozen=True, slots=True)
class _ScientificAblationDelegation:
    attachment: _ScientificAblationControllerAttachment
    operation: str


class IrisScientificAblationRunV1(
    _ScientificRunCore[
        IrisScientificAblationOptimizerV1, IrisScientificAblationOptimizationOutputV1
    ]
):
    """Exclusive fixed-order corpus/runtime owner of an exact fresh ablation optimizer."""

    def __init__(
        self,
        *,
        optimizer: IrisScientificAblationOptimizerV1,
        examples: tuple[IrisExample, ...],
        terminal_token_id: int,
        seed: int,
    ) -> None:
        if type(optimizer) is not IrisScientificAblationOptimizerV1:
            raise IrisScientificAblationRunError("Exact scientific ablation optimizer required")
        if optimizer._attached_run_token is not None:
            raise IrisScientificAblationRunError(
                "Scientific ablation optimizer is already attached"
            )
        self._attachment_token = object()
        try:
            optimizer._validate_contract()
            model = optimizer._model
            self._ablation_identity = model.ablation_policy_sha256
            self._model_configuration_identity = model.configuration_sha256
            self._optimizer_configuration_identity = optimizer.configuration_sha256
            self._committed_scientific_state_identity = optimizer._committed_scientific_state_sha256
            configuration = json.loads(model.base.configuration_json)
            self._teacher_graph_identity = domain_sha256(
                "iris/cognitive-ablation-teacher-graph/v1",
                canonical_json_bytes(
                    {
                        "backbone": configuration["backbone_configuration_json"],
                        "cognitive": configuration["cognitive_configuration_json"],
                        "workspace": configuration["workspace_configuration_json"],
                        "policy": model.ablation.model_dump(mode="json"),
                    }
                ),
            )
            runtime = capture_runtime_state(optimizer._device)
            super().__init__(
                optimizer=optimizer,
                examples=examples,
                terminal_token_id=terminal_token_id,
                seed=seed,
                configuration_kind="iris-scientific-ablation-run-v1",
                configuration_domain="iris/scientific-ablation-run-configuration/v1",
                history_domain="iris/scientific-ablation-run-history/v1",
                configuration_extra={
                    "ablation": model.ablation.model_dump(mode="json"),
                    "ablation_policy_sha256": self._ablation_identity,
                    "model_configuration_sha256": self._model_configuration_identity,
                    "initial_scientific_state_sha256": self._committed_scientific_state_identity,
                    "runtime_policy": {
                        "deterministic_algorithms": runtime.deterministic_algorithms,
                        "deterministic_warn_only": runtime.deterministic_warn_only,
                        "cudnn_deterministic": runtime.cudnn_deterministic,
                        "cudnn_benchmark": runtime.cudnn_benchmark,
                    },
                },
            )
        except BaseException as error:
            if optimizer._attached_run_token is self._attachment_token:
                optimizer._attached_run_token = None
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise IrisScientificAblationRunError(
                "Scientific ablation run attachment failed"
            ) from None

    @property
    def configuration_json(self) -> str:
        if (
            self.configuration_sha256 != self._configuration_guard
            or domain_sha256(
                "iris/scientific-ablation-run-configuration/v1",
                self._configuration_json.encode("utf-8"),
            )
            != self._configuration_guard
        ):
            raise IrisScientificAblationRunError("Scientific ablation run configuration changed")
        return self._configuration_json

    @property
    def history_sha256(self) -> str:
        return self._history_sha256

    def _cursor(self) -> tuple[int, int, int, str]:
        return (
            self._completed_steps,
            self._examples_seen,
            self._target_tokens_seen,
            self._history_sha256,
        )

    @staticmethod
    def _clone_runtime(runtime: RuntimeStateSnapshot) -> RuntimeStateSnapshot:
        return replace(
            runtime,
            cpu_rng_state=runtime.cpu_rng_state.clone(),
            cuda_rng_state=None
            if runtime.cuda_rng_state is None
            else runtime.cuda_rng_state.clone(),
        )

    def _initialize_owner(self) -> None:
        self._optimizer._attached_run_token = self._attachment_token
        self._configuration_guard = self.configuration_sha256
        self._owned_optimizer = self._optimizer
        self._owned_rows = self._rows
        self._owned_row_fingerprints = self._row_fingerprints
        self._controller_attachment: _ScientificAblationControllerAttachment | None = None
        self._controller_attachment_guard: _ScientificAblationControllerAttachment | None = None
        self._commit_owner()

    def _validate_controller_attachment(self) -> None:
        # Deliberately not part of _commit_owner: train/restore may not reseal an
        # externally replaced controller attachment as a legitimate cursor commit.
        from iris.application.scientific_ablation_lifecycle import _OWNERS

        attachment = self._controller_attachment
        registered = _OWNERS.get(self)
        if attachment is not self._controller_attachment_guard or (
            (attachment is None and registered is not None)
            or (
                attachment is not None
                and (
                    type(attachment) is not _ScientificAblationControllerAttachment
                    or type(registered) is not tuple
                    or len(registered) != 2
                    or registered[0] is not attachment.token
                    or registered[1] is not attachment.owner
                )
            )
        ):
            self._poisoned = self._optimizer._poisoned = True
            raise IrisScientificAblationRunError("Scientific ablation controller ownership changed")

    def _authorize_operation(self, operation: str, capability: object | None) -> None:
        from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1

        self._validate_controller_attachment()
        attachment = self._controller_attachment
        if attachment is None:
            if capability is not None:
                raise IrisScientificAblationRunError("Unexpected ablation controller delegation")
            return
        owner = attachment.owner()
        if (
            type(capability) is not _ScientificAblationDelegation
            or capability.attachment is not attachment
            or capability.operation != operation
            or type(owner) is not IrisScientificAblationLifecycleV1
            or owner._delegation is not capability
            or not owner._busy
            or owner._owned_run is not self
            or owner._run is not self
            or owner._attachment_token is not attachment.token
            or owner._pending is not None
            or (
                operation in ("train", "validation")
                and owner._state.phase != ("train" if operation == "train" else "validation")
            )
            or (operation == "restore" and owner._state.step != 0)
        ):
            raise IrisScientificAblationRunError(
                "Attached ablation run requires controller delegation"
            )
        # This exact call consumes its capability before runtime/model side effects.
        owner._delegation = None

    def _commit_owner(self) -> None:
        self._committed_scientific_state_identity = (
            self._optimizer._committed_scientific_state_sha256
        )
        self._committed_cursor = self._cursor()
        self._owned_runtime = self._clone_runtime(self._runtime)

    def _validate_owner(self) -> None:
        self._validate_controller_attachment()
        if (
            self._optimizer is not self._owned_optimizer
            or self._optimizer._attached_run_token is not self._attachment_token
            or self._rows is not self._owned_rows
            or self._row_fingerprints is not self._owned_row_fingerprints
            or self._cursor() != self._committed_cursor
            or not _same_runtime(self._runtime, self._owned_runtime)
            or self._history_domain != "iris/scientific-ablation-run-history/v1"
        ):
            self._poisoned = True
            raise IrisScientificAblationRunError(
                "Scientific ablation cursor/runtime ownership changed"
            )
        try:
            _ = self.configuration_json
        except Exception:
            self._poisoned = True
            raise IrisScientificAblationRunError(
                "Scientific ablation configuration ownership changed"
            ) from None

    def _validate_boundary(self) -> None:
        self._validate_owner()
        if self._optimizer.completed_steps != self._completed_steps:
            self._poisoned = True
            raise IrisScientificAblationRunError(
                "Scientific ablation optimizer advanced externally"
            )
        super()._validate_boundary()
        model = self._optimizer._model
        if (
            model.ablation_policy_sha256 != self._ablation_identity
            or model.configuration_sha256 != self._model_configuration_identity
            or self._optimizer.configuration_sha256 != self._optimizer_configuration_identity
            or self._optimizer._teacher_graph_identity != self._teacher_graph_identity
            or self._optimizer._committed_scientific_state_sha256
            != self._committed_scientific_state_identity
        ):
            self._poisoned = True
            raise IrisScientificAblationRunError(
                "Scientific ablation graph/optimizer identity changed"
            )

    def _window_identity(
        self, window: tuple[IrisExample, ...], target_counts: tuple[int, ...]
    ) -> str:
        model = self._optimizer._model
        teacher_state = model.teacher.checkpoint_provenance.sidecar_state_sha256
        rows: list[str] = []
        invocations: list[str] = []
        for offset, (row, count) in enumerate(zip(window, target_counts, strict=True)):
            encoded = canonical_json_bytes(row.model_dump(mode="json"))
            if (
                sha256(encoded).hexdigest()
                != self._row_fingerprints[(self._examples_seen + offset) % len(self._rows)]
            ):
                self._poisoned = True
                raise IrisScientificAblationRunError("Selected immutable corpus row changed")
            prepared = prepare_teacher_forcing_input(
                backbone=self._backbone,
                prompt=row.prompt,
                target_text=row.expected_response,
                terminal_token_id=self._terminal_token_id,
            )
            if int(prepared.target_attention_mask.sum()) != count:
                raise IrisScientificAblationRunError("Selected target count changed")
            identity = domain_sha256(
                "iris/scientific-ablation-example/v1",
                encoded,
            )
            invocation = domain_sha256(
                "iris/cognitive-ablation-teacher-invocation/v1",
                canonical_json_bytes(
                    {
                        "mode": "training",
                        "policy_sha256": self._ablation_identity,
                        "prompt_sha256": domain_sha256(
                            "iris/cognitive-ablation-prompt/v1", row.prompt.encode("utf-8")
                        ),
                        "target_sha256": teacher_target_sha256_v1(
                            prepared.target_input_ids, prepared.target_attention_mask
                        ),
                        "teacher_graph_sha256": self._teacher_graph_identity,
                        "sidecar_state_sha256": teacher_state,
                    }
                ),
            )
            rows.append(identity)
            invocations.append(
                domain_sha256(
                    "iris/scientific-ablation-invocation/v1",
                    canonical_json_bytes(
                        {
                            "mode": "training",
                            "example_sha256": identity,
                            "configuration_sha256": self._model_configuration_identity,
                            "teacher_invocation_sha256": invocation,
                            "scientific_state_sha256": self._committed_scientific_state_identity,
                        }
                    ),
                )
            )
        return domain_sha256(
            "iris/scientific-ablation-optimizer-window/v1",
            canonical_json_bytes(
                {
                    "rows": rows,
                    "target_counts": list(target_counts),
                    "invocations": invocations,
                    "terminal_token_id": self._terminal_token_id,
                    "model_configuration_sha256": self._model_configuration_identity,
                    "ablation_policy_sha256": self._ablation_identity,
                    "scientific_state_before_sha256": self._committed_scientific_state_identity,
                }
            ),
        )

    def _execute_window(
        self, window: tuple[IrisExample, ...], target_counts: tuple[int, ...]
    ) -> IrisScientificAblationOptimizationOutputV1:
        expected_window = self._window_identity(window, target_counts)
        self._validate_boundary()
        result = self._optimizer.step(window, terminal_token_id=self._terminal_token_id)
        self._validate_owner()  # Child execution must not rewrite the run cursor or private RNG.
        if (
            type(result) is not IrisScientificAblationOptimizationOutputV1
            or result.step_index != self._completed_steps + 1
            or result.step_index != self._optimizer.completed_steps
            or result.example_count != len(window)
            or result.supervised_token_count != sum(target_counts)
            or result.purpose != self._optimizer._policy.purpose
            or result.configuration_sha256 != self._optimizer_configuration_identity
            or result.model_configuration_sha256 != self._model_configuration_identity
            or result.ablation_policy_sha256 != self._ablation_identity
            or result.scientific_state_before_sha256 != self._committed_scientific_state_identity
            or result.scientific_state_after_sha256
            != self._optimizer._committed_scientific_state_sha256
            or result.window_sha256 != expected_window
        ):
            raise IrisScientificAblationRunError("Executed ablation window provenance differs")
        return result

    def step(self) -> IrisScientificAblationOptimizationOutputV1:
        return self._step_authorized(None)

    def _step_authorized(
        self, capability: object | None
    ) -> IrisScientificAblationOptimizationOutputV1:
        try:
            return super()._step_authorized(capability)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise IrisScientificAblationRunError(
                "Scientific ablation run step could not complete"
            ) from None


__all__ = ["IrisScientificAblationRunError", "IrisScientificAblationRunV1"]
