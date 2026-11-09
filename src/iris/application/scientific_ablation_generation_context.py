"""Trusted fresh reconstruction before any mode-bound held-out forward.

Original train/model initialization and complete validation are rechecked before
restoring native best. No lifecycle is attached, no training/evaluation is performed,
and no files are written. The private context exposes defensive metadata only;
complete held-out input/capture/ack-resume/NLL/scoring remain a separate boundary.
Runtime/default-dtype ownership is exclusive and process-global, not thread-safe.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from iris.application.scientific_ablation_checkpoint import (
    build_scientific_ablation_checkpoint_v1,
    restore_scientific_ablation_checkpoint_v1,
)
from iris.application.scientific_ablation_generation_source import (
    IrisScientificAblationGenerationSourceBundleV1,
    verify_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_inference import IrisScientificAblationInferenceBundleV1
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    _bundle_parts,
    _Configuration,
)
from iris.application.scientific_ablation_request import (
    IrisScientificAblationRequestV1,
    _compatible_backbone,
    build_scientific_ablation_request_v1,
)
from iris.application.scientific_ablation_request import (
    _entry as _request_entry,
)
from iris.application.scientific_ablation_request import (
    _poison as _poison_request,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_validation import (
    _complete_row_fields,
    _has_signal,
    _snapshot,
    _training_corpus_sha256,
    own_scientific_ablation_validation_rows_v1,
)
from iris.application.scientific_checkpoint import _json
from iris.application.scientific_optimization import _same_runtime
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.scientific_ablation_generation_reconstruction_v1 import (
    IrisScientificAblationGenerationReconstructionV1,
    ScientificAblationGenerationReconstructionPayloadV1,
)
from iris.artifacts.scientific_ablation_generation_source_v1 import (
    IrisScientificAblationGenerationSourceManifestV1,
)
from iris.data.schemas import IrisExample
from iris.generation import TemperatureTopPSamplingConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state

_DOMAIN = "iris/scientific-ablation-generation-reconstruction/v1/payload"


class IrisScientificAblationGenerationContextError(RuntimeError):
    """Original fresh context, source state or caller restoration did not verify."""


@dataclass(frozen=True, slots=True, eq=False)
class _ContextSeal:
    context: "IrisScientificAblationGenerationContextV1"
    run: IrisScientificAblationRunV1
    request: IrisScientificAblationRequestV1
    source: IrisScientificAblationGenerationSourceBundleV1
    validation: tuple[IrisExample, ...]
    source_metadata: bytes
    reconstruction_metadata: bytes
    run_snapshot: tuple[object, ...]
    model_modes: tuple[tuple[str, bool], ...]
    gradients: tuple[tuple[str, int | None, int | None, str | None], ...]
    private_runtime: RuntimeStateSnapshot


class IrisScientificAblationGenerationContextV1:
    """Private reconstructed native-best teacher plus independently owned request.

    Factory code is trusted and may retain references: boundary checks detect
    ordinary alias/state corruption, not hostile introspection or monkeypatching.
    No public raw run/request, retransfer or held-out execution API is exposed here.
    """

    __slots__ = ("_guard", "_request", "_run", "_source", "_state", "_validation")
    _guard: _ContextSeal
    _request: IrisScientificAblationRequestV1
    _run: IrisScientificAblationRunV1
    _source: IrisScientificAblationGenerationSourceBundleV1
    _state: str
    _validation: tuple[IrisExample, ...]

    def __init__(self) -> None:
        raise IrisScientificAblationGenerationContextError("Use the verified fresh context builder")

    @property
    def reconstruction(self) -> IrisScientificAblationGenerationReconstructionV1:
        seal = _entry(self)
        return IrisScientificAblationGenerationReconstructionV1.model_validate_json(
            seal.reconstruction_metadata
        )

    @property
    def source_manifest(self) -> IrisScientificAblationGenerationSourceManifestV1:
        seal = _entry(self)
        return IrisScientificAblationGenerationSourceManifestV1.model_validate_json(
            seal.source_metadata
        )


def _gradients(
    run: IrisScientificAblationRunV1,
) -> tuple[tuple[str, int | None, int | None, str | None], ...]:
    return tuple(
        (name, None, None, None)
        if parameter.grad is None
        else (name, id(parameter.grad), parameter.grad._version, tensor_sha256(parameter.grad))
        for name, parameter in run._optimizer._model.named_parameters()
    )


def _poison(context: IrisScientificAblationGenerationContextV1) -> None:
    context._state = "poisoned"
    # Poison the sealed owners, not a foreign owner substituted through an alias.
    seal = getattr(context, "_guard", None)
    if type(seal) is not _ContextSeal or seal.context is not context:
        return
    run, request = seal.run, seal.request
    if type(run) is IrisScientificAblationRunV1:
        run._poisoned = run._optimizer._poisoned = True
    if type(request) is IrisScientificAblationRequestV1:
        _poison_request(request)


def _entry(context: IrisScientificAblationGenerationContextV1) -> _ContextSeal:
    reentry_denied = False
    try:
        if type(context) is not IrisScientificAblationGenerationContextV1:
            raise ValueError("Exact reconstructed context required")
        seal = context._guard
        if (
            type(seal) is not _ContextSeal
            or seal.context is not context
            or context._run is not seal.run
            or context._request is not seal.request
            or context._source is not seal.source
            or context._validation is not seal.validation
        ):
            raise ValueError("Reconstructed context anchor/binding/state changed")
        if context._state == "busy":
            reentry_denied = True
            raise IrisScientificAblationGenerationContextError(
                "Reconstructed context is already busy"
            )
        if context._state != "ready":
            raise ValueError("Reconstructed context is not ready")
        with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
            run = seal.run
            run._validate_boundary()
            request_seal = _request_entry(seal.request)
            request_manifest = seal.request.manifest
            request_seal.owner._live(seal.request)
            request_seal.owner._check_sidecar(request_manifest.payload.model.sha256)
            _compatible_backbone(request_manifest, run._backbone)
            if (
                _snapshot(run) != seal.run_snapshot
                or tuple(
                    (name, module.training)
                    for name, module in run._optimizer._model.named_modules()
                )
                != seal.model_modes
                or _gradients(run) != seal.gradients
                or not _same_runtime(run._runtime, seal.private_runtime)
            ):
                raise ValueError("Private native-best state/gradient/mode/runtime changed")
        return seal
    except BaseException as error:
        if type(context) is IrisScientificAblationGenerationContextV1 and not reentry_denied:
            _poison(context)
        if _has_signal(error):
            raise
        raise IrisScientificAblationGenerationContextError(
            "Reconstructed context is not ready"
        ) from error


def _validation_snapshot(examples: tuple[IrisExample, ...]) -> tuple[IrisExample, ...]:
    if type(examples) is not tuple or not 1 <= len(examples) <= 5000:
        raise ValueError("Explicit bounded complete original validation required")
    rows: list[IrisExample] = []
    logical_bytes = 0
    for row in examples:
        if not _complete_row_fields(row) or row.split != "validation":
            raise ValueError("Exact original validation rows required before factory")
        owned = IrisExample.model_validate(row.model_dump(mode="python", warnings=False))
        logical_bytes += len(canonical_json_bytes(owned.model_dump(mode="json")))
        if logical_bytes > 64 * 1024**2:
            raise ValueError("Original validation byte budget exceeded")
        rows.append(owned)
    return tuple(rows)


def _restore_caller(
    snapshot: RuntimeStateSnapshot, dtype: torch.dtype, device: torch.device
) -> None:
    failures: list[BaseException] = []
    operations: tuple[Callable[[], None], ...] = (
        lambda: torch.set_default_dtype(dtype),
        lambda: restore_runtime_state(snapshot),
    )
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            failures.append(error)
    try:
        if torch.get_default_dtype() != dtype or not _same_runtime(
            snapshot, capture_runtime_state(device)
        ):
            raise ValueError("Caller context restoration returned without exact restoration")
    except BaseException as error:
        failures.append(error)
    if failures:
        raise BaseExceptionGroup("Generation context caller restoration failed", failures)


def build_scientific_ablation_generation_context_v1(
    bundle: IrisScientificAblationGenerationSourceBundleV1,
    *,
    expected_sha256: str,
    expected_lifecycle_sha256: str,
    expected_inference_sha256: str,
    expected_source_checkpoint_sha256: str,
    make_fresh_run: Callable[[], IrisScientificAblationRunV1],
    validation: tuple[IrisExample, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisScientificAblationGenerationContextV1:
    """Reproduce original train/model/validation BEFORE any selected-best copy.

    Factory must genuinely reproduce the original standalone initial graph/run;
    it is trusted explicit caller code, never reconstruction from untrusted JSON.
    Full caller runtime is captured before factory, construction uses explicit
    device/normal tensors/grad/FP32 dtype, and full source bytes remain read-only.
    No controller is attached and no best/current selection is re-run on test data.
    """
    caller: RuntimeStateSnapshot | None = None
    caller_dtype = torch.get_default_dtype()
    device: torch.device | None = None
    fresh: IrisScientificAblationRunV1 | None = None
    context: IrisScientificAblationGenerationContextV1 | None = None
    request: IrisScientificAblationRequestV1 | None = None
    mutation_started = False
    primary: BaseException | None = None
    cleanup: BaseException | None = None
    with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
        try:
            manifest = verify_scientific_ablation_generation_source_v1(
                bundle,
                expected_sha256=expected_sha256,
                expected_lifecycle_sha256=expected_lifecycle_sha256,
                expected_inference_sha256=expected_inference_sha256,
                expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
            )
            if not callable(make_fresh_run) or type(sampling) is not TemperatureTopPSamplingConfig:
                raise ValueError("Explicit original factory and exact sampling policy required")
            sampling = TemperatureTopPSamplingConfig(sampling.temperature, sampling.top_p)
            validation = _validation_snapshot(validation)
            private = IrisScientificAblationGenerationSourceBundleV1(
                manifest,
                IrisScientificAblationLifecycleBundleV1(
                    bundle.lifecycle.manifest.model_copy(deep=True),
                    bundle.lifecycle.current_state_bytes,
                    bundle.lifecycle.best_state_bytes,
                ),
                IrisScientificAblationInferenceBundleV1(
                    bundle.inference.manifest.model_copy(deep=True), bundle.inference.state_bytes
                ),
            )
            device = torch.device(private.inference.manifest.payload.model.runtime.device)
            caller = capture_runtime_state(device)
            torch.set_default_dtype(torch.float32)
            with torch.device(device):
                candidate = make_fresh_run()
                if type(candidate) is not IrisScientificAblationRunV1 or candidate._busy:
                    raise ValueError("Exact idle fresh standalone factory result required")
                candidate._authorize_operation("restore", None)
                candidate._validate_boundary()
                if (
                    candidate.completed_steps
                    or candidate.examples_seen
                    or candidate.target_tokens_seen
                ):
                    raise ValueError("Original reconstruction cannot use progressed/attached run")
                fresh = candidate
                torch.set_default_dtype(torch.float32)
                initial = build_scientific_ablation_checkpoint_v1(fresh)
                source = manifest.payload
                if (
                    initial.manifest.payload.configuration_sha256 != source.run_configuration_sha256
                    or initial.manifest.payload.scientific_state_sha256
                    != source.initial_scientific_state_sha256
                    or fresh._optimizer._model.configuration_sha256
                    != source.model_configuration_sha256
                    or _training_corpus_sha256(fresh) != source.training_corpus_sha256
                    or len(fresh._rows) != source.training_examples
                ):
                    raise ValueError("Factory does not reproduce original initial graph/train/run")
                rows, validation_digest = own_scientific_ablation_validation_rows_v1(
                    validation, training_rows=fresh._rows
                )
                target_count = sum(
                    int(
                        prepare_teacher_forcing_input(
                            backbone=fresh._backbone,
                            prompt=row.prompt,
                            target_text=row.expected_response,
                            terminal_token_id=fresh._terminal_token_id,
                        ).target_attention_mask.sum()
                    )
                    for row in rows
                )
                original = _Configuration.model_validate(
                    _json(private.lifecycle.manifest.payload.configuration_json.encode())
                )
                reconstructed = _Configuration(
                    kind="iris-scientific-ablation-lifecycle-v1",
                    run_sha256=fresh.configuration_sha256,
                    ablation_policy_sha256=fresh._ablation_identity,
                    validation_sha256=validation_digest,
                    validation_examples=len(rows),
                    validation_target_tokens=target_count,
                    policy=original.policy,
                )
                configuration_json = canonical_json_bytes(
                    reconstructed.model_dump(mode="json")
                ).decode()
                if (
                    configuration_json != private.lifecycle.manifest.payload.configuration_json
                    or domain_sha256(
                        "iris/scientific-ablation-lifecycle-configuration/v1",
                        configuration_json.encode(),
                    )
                    != source.lifecycle_configuration_sha256
                ):
                    raise ValueError(
                        "Factory/validation does not reproduce original lifecycle context"
                    )
                _, _, best = _bundle_parts(private.lifecycle)
                if best is None:
                    raise ValueError("Actual selected best required")
                mutation_started = True
                restore_scientific_ablation_checkpoint_v1(best, run=fresh)
                if build_scientific_ablation_checkpoint_v1(fresh) != best:
                    raise ValueError("Actual native-best restore differs from saved source")
                restored_snapshot = _snapshot(fresh)
                request = build_scientific_ablation_request_v1(
                    private.inference,
                    expected_sha256=expected_inference_sha256,
                    expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
                    expected_lifecycle_sha256=expected_lifecycle_sha256,
                    backbone=fresh._backbone,
                    stop_token_ids=fresh._backbone.eos_token_ids,
                    sampling=sampling,
                )
                if _snapshot(fresh) != restored_snapshot:
                    raise ValueError("Owned request construction changed private native best")
                payload = ScientificAblationGenerationReconstructionPayloadV1(
                    source_sha256=manifest.sha256,
                    lifecycle_sha256=source.lifecycle_sha256,
                    lifecycle_configuration_sha256=source.lifecycle_configuration_sha256,
                    run_configuration_sha256=source.run_configuration_sha256,
                    model_configuration_sha256=source.model_configuration_sha256,
                    ablation_policy_sha256=source.ablation_policy_sha256,
                    fresh_initial_checkpoint_sha256=initial.manifest.sha256,
                    initial_scientific_state_sha256=initial.manifest.payload.scientific_state_sha256,
                    training_corpus_sha256=source.training_corpus_sha256,
                    training_examples=len(fresh._rows),
                    validation_corpus_sha256=validation_digest,
                    validation_examples=len(rows),
                    validation_target_tokens=target_count,
                    selected_checkpoint_sha256=best.manifest.sha256,
                    selected_step=fresh.completed_steps,
                    inference_sha256=source.inference_sha256,
                    backbone_runtime_sha256=fresh._backbone.checkpoint_runtime_sha256,
                    request_sidecar_sha256=source.request_sidecar_sha256,
                )
                reconstruction = IrisScientificAblationGenerationReconstructionV1(
                    sha256=domain_sha256(
                        _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                    ),
                    payload=payload,
                )
                context = object.__new__(IrisScientificAblationGenerationContextV1)
                context._run, context._request = fresh, request
                context._source, context._validation, context._state = private, rows, "ready"
                context._guard = _ContextSeal(
                    context,
                    fresh,
                    request,
                    private,
                    rows,
                    canonical_json_bytes(manifest.model_dump(mode="json")),
                    canonical_json_bytes(reconstruction.model_dump(mode="json")),
                    _snapshot(fresh),
                    tuple(
                        (name, module.training)
                        for name, module in fresh._optimizer._model.named_modules()
                    ),
                    _gradients(fresh),
                    fresh._clone_runtime(fresh._runtime),
                )
                _entry(context)
        except BaseException as error:
            primary = error
        finally:
            if caller is not None and device is not None:
                try:
                    _restore_caller(caller, caller_dtype, device)
                except BaseException as error:
                    cleanup = error
            if primary is not None or cleanup is not None:
                if context is not None:
                    _poison(context)
                elif fresh is not None and (mutation_started or cleanup is not None):
                    fresh._poisoned = fresh._optimizer._poisoned = True
                    if request is not None:
                        _poison_request(request)
    if primary is not None:
        if _has_signal(primary):
            raise primary from cleanup
        if cleanup is not None and _has_signal(cleanup):
            raise cleanup from primary
        cause = (
            primary
            if cleanup is None
            else BaseExceptionGroup("Preparation/restoration failed", [primary, cleanup])
        )
        raise IrisScientificAblationGenerationContextError(
            "Generation context preparation failed"
        ) from cause
    if cleanup is not None:
        if _has_signal(cleanup):
            raise cleanup
        raise IrisScientificAblationGenerationContextError(
            "Generation context caller restoration failed"
        ) from cleanup
    if context is None:
        raise IrisScientificAblationGenerationContextError("No verified reconstructed context")
    return context


__all__ = [
    "IrisScientificAblationGenerationContextError",
    "IrisScientificAblationGenerationContextV1",
    "build_scientific_ablation_generation_context_v1",
]
