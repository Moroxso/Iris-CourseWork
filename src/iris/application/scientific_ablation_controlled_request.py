"""Verified in-memory mode-bound loader and exclusive caller-state transaction.

Backbone loading is external/explicit. Three expected hashes MUST come from a
trusted handoff, not this candidate. No training owner is needed at load time.
Runtime globals are process-global and selected-device scoped, not thread-safe
or all-GPU isolation. This module is not persisted publication or CUDA evidence.
"""

from contextlib import suppress
from dataclasses import dataclass
from typing import NoReturn

import torch

from iris.ablations.generation_controls import _GenerationWorkspaceControlSessionV1
from iris.application._scientific_ablation_controlled_request_owner import (
    _binding_values,
    _ControlledRequestBinding,
    _ControlledRequestOwner,
    _phase_binding,
)
from iris.application.composition import _compose_request_orchestrator
from iris.application.errors import IrisRequestError
from iris.application.request import (
    IrisRequestOrchestrator,
    _CoreRequestExecutionOutput,
    _ScientificAblationControlledOwnedRequest,
)
from iris.application.scientific_ablation_controlled_receipt import (
    _MAX_CONTROL_JSON,
    IrisScientificAblationControlledRequestOutputV1,
    IrisScientificAblationControlledRequestReceiptV1,
)
from iris.application.scientific_ablation_inference import (
    _MAX_MANIFEST,
    IrisScientificAblationInferenceBundleV1,
    _checked,
)
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_inference_v1 import (
    IrisScientificAblationInferenceManifestV1,
)
from iris.backbone.adapter import BackboneAdapter
from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.generation.sampling import _clone_owned_generator
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1

_RECEIPT_DOMAIN = "iris/scientific-ablation-controlled-request-execution/v1"


@dataclass(frozen=True, slots=True, eq=False)
class _WrapperSeal:
    wrapper: "IrisScientificAblationControlledRequestV1"
    request: _ScientificAblationControlledOwnedRequest
    owner: _ControlledRequestOwner
    binding: _ControlledRequestBinding
    binding_values: tuple[str, ...]
    metadata: bytes
    device: torch.device
    device_value: str


class IrisScientificAblationControlledRequestV1:
    """Exclusive fresh mode-bound graph; no public attach/restore/raw graph API."""

    __slots__ = ("_binding", "_guard", "_metadata", "_owner", "_request", "_state")

    _binding: _ControlledRequestBinding
    _guard: _WrapperSeal
    _metadata: bytes
    _owner: _ControlledRequestOwner
    _request: _ScientificAblationControlledOwnedRequest
    _state: str

    def __init__(self) -> None:
        raise IrisRequestError(
            "Use build_scientific_ablation_controlled_request_v1 with trusted bindings"
        )

    @property
    def manifest(self) -> IrisScientificAblationInferenceManifestV1:
        """Defensive metadata, never a mutable owned policy or training state."""
        seal = _entry(self)
        return IrisScientificAblationInferenceManifestV1.model_validate_json(seal.metadata)

    def generate(
        self,
        prompt: str,
        *,
        control: WorkspaceInterventionPlanV1,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        # Direct dispatch, including unbound calls; not an overridable execute hook.
        return _execute(
            self, prompt, control=control, strategy=strategy, generator=generator
        ).generation

    def execute(
        self,
        prompt: str,
        *,
        control: WorkspaceInterventionPlanV1,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificAblationControlledRequestOutputV1:
        return _execute(
            self,
            prompt,
            control=control,
            strategy=strategy,
            generator=generator,
            diagnostics=diagnostics,
        )


def _poison(
    wrapper: IrisScientificAblationControlledRequestV1,
    *,
    trusted_owner: _ControlledRequestOwner | None = None,
) -> None:
    """Best effort; never invoke a foreign guard's properties or mask primary."""
    owners: list[object] = [trusted_owner]
    if type(wrapper) is IrisScientificAblationControlledRequestV1:
        with suppress(BaseException):
            owners.append(object.__getattribute__(wrapper, "_owner"))
        with suppress(BaseException):
            guard = object.__getattribute__(wrapper, "_guard")
            if type(guard) is _WrapperSeal:
                owners.append(guard.owner)
        with suppress(BaseException):
            object.__setattr__(wrapper, "_state", "poisoned")
    for owner in owners:
        if type(owner) is _ControlledRequestOwner:
            with suppress(BaseException):
                _ControlledRequestOwner._poison(owner)


def _structure(wrapper: IrisScientificAblationControlledRequestV1) -> _WrapperSeal:
    """No model getters, contexts, cloning, graph hashing or global RNG effects."""
    try:
        if type(wrapper) is not IrisScientificAblationControlledRequestV1:
            raise IrisRequestError("Exact verified mode-bound wrapper required")
        seal = wrapper._guard
        if (
            type(seal) is not _WrapperSeal
            or seal.wrapper is not wrapper
            or wrapper._request is not seal.request
            or wrapper._owner is not seal.owner
            or wrapper._binding is not seal.binding
            or wrapper._metadata is not seal.metadata
            or type(seal.metadata) is not bytes
            or not 1 <= len(seal.metadata) <= _MAX_MANIFEST
            or type(seal.request) is not _ScientificAblationControlledOwnedRequest
            or type(seal.owner) is not _ControlledRequestOwner
            or seal.request._controlled_anchor is not seal.owner
            or seal.request._controlled_binding is not seal.binding
            or seal.owner._request is not seal.request
            or seal.owner._wrapper is not wrapper
            or seal.owner._binding is not seal.binding
            or seal.owner._wrapper_seal is not seal
            or _binding_values(seal.binding) != seal.binding_values
            or type(seal.device) is not torch.device
            or str(seal.device) != seal.device_value
            or type(wrapper._state) is not str
            or wrapper._state not in {"building", "busy", "ready", "poisoned"}
        ):
            raise IrisRequestError("Mode-bound wrapper/request/owner/binding seal changed")
        return seal
    except BaseException as error:
        _poison(wrapper)
        if isinstance(error, Exception) and not isinstance(error, IrisRequestError):
            raise IrisRequestError("Mode-bound wrapper is missing or has corrupt fields") from error
        raise


def _entry(wrapper: IrisScientificAblationControlledRequestV1) -> _WrapperSeal:
    seal = _structure(wrapper)
    if wrapper._state == "busy":
        # Reentry denial must not poison an otherwise valid call that catches it.
        raise IrisRequestError("Mode-bound wrapper is already busy")
    if (
        wrapper._state != "ready"
        or seal.owner._state != "ready"
        or seal.owner._pending is not None
        or seal.owner._entered is not False
    ):
        _poison(wrapper)
        raise IrisRequestError("Mode-bound wrapper/owner is not ready")
    return seal


def _compatible_backbone(
    manifest: IrisScientificAblationInferenceManifestV1, backbone: BackboneAdapter
) -> None:
    payload, model = manifest.payload, manifest.payload.model
    if (
        model.backbone != BackboneConfig.model_validate_json(backbone.configuration_json)
        or model.runtime.device != str(backbone.device)
        or model.runtime.dtype != str(backbone.dtype).removeprefix("torch.")
        or (
            model.runtime.hidden_size,
            model.runtime.num_hidden_layers,
            model.runtime.vocabulary_size,
            model.runtime.max_position_embeddings,
        )
        != (
            backbone.hidden_size,
            backbone.num_hidden_layers,
            backbone.vocabulary_size,
            backbone.max_position_embeddings,
        )
        or payload.backbone_runtime_sha256 != backbone.checkpoint_runtime_sha256
    ):
        raise IrisRequestError("Mode-bound backbone/tokenizer/configuration/runtime differs")


def _boundary(
    wrapper: IrisScientificAblationControlledRequestV1,
) -> IrisScientificAblationInferenceManifestV1:
    seal = _structure(wrapper)
    if wrapper._state not in {"building", "busy"}:
        raise IrisRequestError("Mode-bound transaction is not exclusive")
    manifest = IrisScientificAblationInferenceManifestV1.model_validate_json(seal.metadata)
    request, model = seal.request, manifest.payload.model
    _compatible_backbone(manifest, request._backbone)
    if model.cognitive != CognitiveConfig.model_validate_json(
        request._cognitive_config_snapshot
    ) or model.workspace != WorkspaceConfig.model_validate_json(request._workspace_config_snapshot):
        raise IrisRequestError("Mode-bound cognitive/workspace configuration differs")
    seal.owner._live(wrapper)
    seal.owner._check_sidecar(seal.binding.sidecar_sha256)
    return manifest


def _restore_context(
    snapshot: RuntimeStateSnapshot, device: torch.device, caller_dtype: torch.dtype
) -> None:
    # RuntimeStateSnapshot deliberately covers RNG/determinism, not default dtype.
    # Attempt BOTH restorations even if the first raises; preserve real signals.
    failures: list[BaseException] = []
    try:
        torch.set_default_dtype(caller_dtype)
    except BaseException as error:
        failures.append(error)
    try:
        restore_runtime_state(snapshot)
    except BaseException as error:
        failures.append(error)
    try:
        if torch.get_default_dtype() != caller_dtype or not _same_runtime(
            snapshot, capture_runtime_state(device)
        ):
            raise IrisRequestError(
                "Caller dtype/RNG restoration returned without exact restoration"
            )
    except BaseException as error:
        failures.append(error)
    if failures:
        cause = BaseExceptionGroup("Controlled caller restoration failures", failures)
        for failure in failures:
            if not isinstance(failure, Exception):
                raise failure from cause
        raise IrisRequestError("Controlled caller restoration is uncertain") from cause


def _restore_sampling_state(generator: torch.Generator, state: torch.Tensor) -> None:
    generator.set_state(state.detach().clone())
    if not torch.equal(generator.get_state(), state):
        raise IrisRequestError("Caller sampling rollback returned without exact restoration")


def _commit_sampling_state(generator: torch.Generator, state: torch.Tensor) -> None:
    generator.set_state(state.detach().clone())
    if not torch.equal(generator.get_state(), state):
        raise IrisRequestError("Caller sampling commit returned without exact final state")


def _raise_failed(primary: BaseException, secondary: list[BaseException]) -> NoReturn:
    if not secondary:
        if isinstance(primary, IrisRequestError) or not isinstance(primary, Exception):
            raise primary
        raise IrisRequestError("Mode-bound caller transaction failed; output withheld") from primary
    cause = BaseExceptionGroup(
        "Mode-bound transaction and restoration failures", [primary, *secondary]
    )
    if not isinstance(primary, Exception):
        raise primary from BaseExceptionGroup("Secondary restoration failures", secondary)
    for error in secondary:
        if not isinstance(error, Exception):
            raise error from cause
    raise IrisRequestError(
        "Mode-bound transaction restoration is uncertain; output withheld"
    ) from cause


def build_scientific_ablation_controlled_request_v1(
    bundle: IrisScientificAblationInferenceBundleV1,
    *,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisScientificAblationControlledRequestV1:
    """Verify/authenticate BEFORE decode/construction; privately compose and copy.

    This consumes an in-memory verified bundle, not a file-system publisher.
    Candidate-derived expected hashes defeat authentication. The supplied already
    loaded frozen backbone remains externally owned; sidecar graph/copies are fresh.
    """
    wrapper: IrisScientificAblationControlledRequestV1 | None = None
    caller: RuntimeStateSnapshot | None = None
    caller_dtype: torch.dtype | None = None
    device: torch.device | None = None
    try:
        with torch.inference_mode(False), torch.device("cpu"):
            manifest, decoded = _checked(
                bundle,
                expected_sha256=expected_sha256,
                expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
                expected_lifecycle_sha256=expected_lifecycle_sha256,
            )
        # Explicit trusted device from authenticated metadata, before ANY adapter getter.
        device = torch.device(manifest.payload.model.runtime.device)
        caller_dtype = torch.get_default_dtype()
        caller = capture_runtime_state(device)
        _compatible_backbone(manifest, backbone)
        if type(sampling) is not TemperatureTopPSamplingConfig:
            raise IrisRequestError("Exact mode-bound sampling policy required")
        sampling = TemperatureTopPSamplingConfig(sampling.temperature, sampling.top_p)
        wrapper = object.__new__(IrisScientificAblationControlledRequestV1)
        wrapper._state = "building"
        # Do not expose or retain caller-owned mutable DTOs.
        metadata = canonical_json_bytes(manifest.model_dump(mode="json"))
        private = IrisScientificAblationInferenceManifestV1.model_validate_json(metadata)
        request = _compose_request_orchestrator(
            backbone=backbone,
            cognitive_config=private.payload.model.cognitive,
            workspace_config=private.payload.model.workspace,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            constructor=_ScientificAblationControlledOwnedRequest,
        )
        payload = private.payload
        binding = _ControlledRequestBinding(
            manifest.sha256,
            payload.source_checkpoint_sha256,
            payload.selection.lifecycle_sha256,
            payload.ablation_policy_sha256,
            payload.model.sha256,
        )
        owner = _ControlledRequestOwner(request, wrapper=wrapper, binding=binding)
        wrapper._request, wrapper._owner, wrapper._binding = request, owner, binding
        wrapper._metadata = metadata
        wrapper._guard = _WrapperSeal(
            wrapper,
            request,
            owner,
            binding,
            _binding_values(binding),
            metadata,
            device,
            str(device),
        )
        owner._wrapper_seal = wrapper._guard
        cap = owner._reserve_bootstrap(wrapper)
        # Installer order is traversal order, NOT lexical safetensors decode order.
        state = {name: decoded[name] for name in payload.model.state_names}
        request._install_owned_sidecar(state, _authorization=cap)
        _boundary(wrapper)
        _restore_context(caller, device, caller_dtype)
        _boundary(wrapper)  # Restoration callbacks cannot change the bound graph.
        if torch.get_default_dtype() != caller_dtype or not _same_runtime(
            caller, capture_runtime_state(device)
        ):
            raise IrisRequestError("Loader postchecks changed caller runtime")
        wrapper._state = "ready"
        return wrapper
    except BaseException as primary:
        if wrapper is not None:
            _poison(wrapper)
        secondary: list[BaseException] = []
        if caller is not None and device is not None and caller_dtype is not None:
            try:
                _restore_context(caller, device, caller_dtype)
            except BaseException as cleanup:
                secondary.append(cleanup)
        _raise_failed(primary, secondary)


def _receipt(
    manifest: IrisScientificAblationInferenceManifestV1,
    request: _ScientificAblationControlledOwnedRequest,
    core: _CoreRequestExecutionOutput,
    session: _GenerationWorkspaceControlSessionV1,
    prompt: str,
    strategy: DecodingStrategy,
    seed: int | None,
    initial: torch.Tensor | None,
    final: torch.Tensor | None,
) -> IrisScientificAblationControlledRequestReceiptV1:
    cognitive = session.ablation
    if (
        type(core) is not _CoreRequestExecutionOutput
        or type(core.generation) is not IrisGeneratorOutput
        or type(cognitive._length) is not int
        or cognitive._length != core.prompt_attention_mask.shape[1]
        or len(core.generation.workspace_trace) != request._cycles
        or (
            core.diagnostic_trace is not None
            and core.diagnostic_trace.intervention_plan_sha256 != session.plan_sha256
        )
    ):
        raise IrisRequestError("Completed core facts/session/diagnostic binding differs")
    payload = manifest.payload
    return IrisScientificAblationControlledRequestReceiptV1(
        artifact_sha256=manifest.sha256,
        source_checkpoint_sha256=payload.source_checkpoint_sha256,
        lifecycle_sha256=payload.selection.lifecycle_sha256,
        source_run_configuration_sha256=payload.source_run_configuration_sha256,
        source_model_configuration_sha256=payload.source_model_configuration_sha256,
        backbone_runtime_sha256=payload.backbone_runtime_sha256,
        ablation_policy_sha256=payload.ablation_policy_sha256,
        ablation_policy_json=canonical_json_bytes(
            payload.ablation.model_dump(mode="json")
        ).decode(),
        control_plan_sha256=session.plan_sha256,
        control_plan_json=session._plan_json.decode("utf-8"),
        sidecar_state_sha256=payload.model.sha256,
        selection=payload.selection.kind,
        completed_steps=payload.completed_steps,
        purpose=payload.purpose,
        prompt_sha256=domain_sha256(_RECEIPT_DOMAIN + "/prompt", prompt.encode("utf-8")),
        cognitive_prompt_sha256=cognitive.prompt_sha256,
        prompt_input_ids_sha256=tensor_sha256(
            core.prompt_input_ids, domain=_RECEIPT_DOMAIN + "/ids"
        ),
        prompt_attention_mask_sha256=tensor_sha256(
            core.prompt_attention_mask, domain=_RECEIPT_DOMAIN + "/mask"
        ),
        generated_token_ids_sha256=tensor_sha256(
            core.generation.token_ids, domain=_RECEIPT_DOMAIN + "/generated-ids"
        ),
        strategy=strategy,
        stop_token_ids=request._iris_generator.stop_token_ids,
        sampling=request._iris_generator.sampling,
        max_new_tokens=request._iris_generator.max_new_tokens,
        cycles=request._cycles,
        sampling_seed=seed,
        sampling_initial_state_sha256=None
        if initial is None
        else tensor_sha256(initial, domain=_RECEIPT_DOMAIN + "/sampling-state"),
        sampling_final_state_sha256=None
        if final is None
        else tensor_sha256(final, domain=_RECEIPT_DOMAIN + "/sampling-state"),
    )


def _execute(
    wrapper: IrisScientificAblationControlledRequestV1,
    prompt: str,
    *,
    control: WorkspaceInterventionPlanV1,
    strategy: DecodingStrategy,
    generator: torch.Generator | None = None,
    diagnostics: WorkspaceTraceOptionsV1 | None = None,
) -> IrisScientificAblationControlledRequestOutputV1:
    seal = _entry(wrapper)  # Reject missing fields/reentry BEFORE context/RNG/model getters.
    IrisRequestOrchestrator._validate_request(prompt, strategy=strategy, generator=generator)
    caller_dtype = torch.get_default_dtype()
    caller: RuntimeStateSnapshot | None = None
    initial: torch.Tensor | None = None
    wrapper._state = "busy"
    try:
        caller = capture_runtime_state(seal.device)
        local: torch.Generator | None = None
        seed: int | None = None
        if generator is not None:
            initial = generator.get_state().detach().clone()
            seed = generator.initial_seed()
            local = _clone_owned_generator(generator, expected_device=seal.device)
        manifest = _boundary(wrapper)
        request, owner = seal.request, seal.owner
        session = _GenerationWorkspaceControlSessionV1(
            manifest.payload.ablation,
            control,
            binding=_phase_binding(request, prompt),
        )
        if len(session._plan_json) > _MAX_CONTROL_JSON:
            raise IrisRequestError("Controlled intent exceeds receipt byte quota before model work")
        cap = owner._reserve_execution(
            wrapper,
            prompt=prompt,
            session=session,
            strategy=strategy,
            generator=local,
            diagnostics=diagnostics,
        )
        # Tokenizer/library allocations must not inherit a caller's meta default.
        # Scope exits before outer restoration, including primary signals.
        with torch.device(seal.device):
            core = request._execute_core(
                prompt,
                strategy=strategy,
                generator=local,
                control=session,
                diagnostics=diagnostics,
                _authorization=cap,
            )
        owner._verify_execution(wrapper, cap)  # Keep busy through ALL outer postchecks.
        manifest = _boundary(wrapper)
        final = None if local is None else local.get_state().detach().clone()
        if local is not None and local.initial_seed() != seed:
            raise IrisRequestError("Private sampling generator was reseeded")
        receipt = _receipt(manifest, request, core, session, prompt, strategy, seed, initial, final)
        output = IrisScientificAblationControlledRequestOutputV1(
            core.generation,
            core.prompt_input_ids,
            core.prompt_attention_mask,
            receipt,
            core.diagnostic_trace,
        )
        if (
            generator is not None
            and initial is not None
            and (
                generator.initial_seed() != seed or not torch.equal(generator.get_state(), initial)
            )
        ):
            raise IrisRequestError("Caller sampling generator changed before commit")
        if torch.get_default_dtype() != caller_dtype or not _same_runtime(
            caller, capture_runtime_state(seal.device)
        ):
            raise IrisRequestError("Mode-bound execution changed caller runtime globals")
        _restore_context(caller, seal.device, caller_dtype)
        owner._verify_execution(wrapper, cap)
        manifest = _boundary(wrapper)
        if (
            _receipt(manifest, request, core, session, prompt, strategy, seed, initial, final)
            != receipt
        ):
            raise IrisRequestError("Mode-bound result/receipt changed during restoration")
        owner._complete_execution(wrapper, cap)
        if generator is not None and final is not None:
            # Check AGAIN after restoration; only now may caller-visible sampling advance.
            assert initial is not None
            if generator.initial_seed() != seed or not torch.equal(generator.get_state(), initial):
                raise IrisRequestError("Caller sampling generator changed during restoration")
            _commit_sampling_state(generator, final)
            if generator.initial_seed() != seed or not torch.equal(generator.get_state(), final):
                raise IrisRequestError(
                    "Caller sampling commit did not produce the exact receipt state"
                )
        _boundary(wrapper)
        if (
            generator is not None
            and final is not None
            and (generator.initial_seed() != seed or not torch.equal(generator.get_state(), final))
        ):
            raise IrisRequestError("Caller sampling state changed during final postchecks")
        if torch.get_default_dtype() != caller_dtype or not _same_runtime(
            caller, capture_runtime_state(seal.device)
        ):
            raise IrisRequestError("Mode-bound finalization changed caller runtime globals")
        wrapper._state = "ready"
        return output
    except BaseException as primary:
        _poison(wrapper, trusted_owner=seal.owner)
        secondary: list[BaseException] = []
        if generator is not None and initial is not None:
            try:
                _restore_sampling_state(generator, initial)
            except BaseException as cleanup:
                secondary.append(cleanup)
        if caller is not None:
            try:
                _restore_context(caller, seal.device, caller_dtype)
            except BaseException as cleanup:
                secondary.append(cleanup)
        _raise_failed(primary, secondary)


__all__ = [
    "IrisScientificAblationControlledRequestV1",
    "build_scientific_ablation_controlled_request_v1",
]
