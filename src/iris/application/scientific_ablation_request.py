"""Verified in-memory mode-bound loader and exclusive caller-state transaction.

Backbone loading is external/explicit. Three expected hashes MUST come from a
trusted handoff, not this candidate. No training owner is needed at load time.
Runtime globals are process-global and selected-device scoped, not thread-safe
or all-GPU isolation. This module is not persisted publication or CUDA evidence.
"""

from dataclasses import dataclass
from typing import NoReturn

import torch

from iris.ablations.cognitive import CognitiveAblationSessionV1
from iris.application._scientific_ablation_request_owner import (
    _AblationRequestBinding,
    _AblationRequestOwner,
    _binding_values,
)
from iris.application.composition import _compose_request_orchestrator
from iris.application.errors import IrisRequestError
from iris.application.request import (
    IrisRequestOrchestrator,
    _CoreRequestExecutionOutput,
    _ScientificAblationOwnedRequest,
)
from iris.application.scientific_ablation_inference import (
    _MAX_MANIFEST,
    IrisScientificAblationInferenceBundleV1,
    _checked,
)
from iris.application.scientific_ablation_receipt import (
    IrisScientificAblationRequestOutputV1,
    IrisScientificAblationRequestReceiptV1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_ablation_inference_v1 import (
    IrisScientificAblationInferenceManifestV1,
)
from iris.backbone.adapter import BackboneAdapter
from iris.generation import DecodingStrategy, IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.generation.sampling import _clone_owned_generator
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1

_RECEIPT_DOMAIN = "iris/scientific-ablation-request-execution/v1"


@dataclass(frozen=True, slots=True, eq=False)
class _WrapperSeal:
    wrapper: "IrisScientificAblationRequestV1"
    request: _ScientificAblationOwnedRequest
    owner: _AblationRequestOwner
    binding: _AblationRequestBinding
    binding_values: tuple[str, ...]
    metadata: bytes
    device: torch.device
    device_value: str


class IrisScientificAblationRequestV1:
    """Exclusive fresh mode-bound graph; no public attach/restore/raw graph API."""

    __slots__ = ("_binding", "_guard", "_metadata", "_owner", "_request", "_state")

    _binding: _AblationRequestBinding
    _guard: _WrapperSeal
    _metadata: bytes
    _owner: _AblationRequestOwner
    _request: _ScientificAblationOwnedRequest
    _state: str

    def __init__(self) -> None:
        raise IrisRequestError("Use build_scientific_ablation_request_v1 with trusted bindings")

    @property
    def manifest(self) -> IrisScientificAblationInferenceManifestV1:
        """Defensive metadata, never a mutable owned policy or training state."""
        seal = _entry(self)
        return IrisScientificAblationInferenceManifestV1.model_validate_json(seal.metadata)

    def generate(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        # Direct dispatch, including unbound calls; not an overridable execute hook.
        return _execute(self, prompt, strategy=strategy, generator=generator).generation

    def execute(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificAblationRequestOutputV1:
        return _execute(
            self, prompt, strategy=strategy, generator=generator, diagnostics=diagnostics
        )


def _poison(wrapper: IrisScientificAblationRequestV1) -> None:
    # Best effort on corrupt/missing fields; never masks the primary failure.
    for owner in (
        getattr(wrapper, "_owner", None),
        getattr(getattr(wrapper, "_guard", None), "owner", None),
    ):
        if type(owner) is _AblationRequestOwner:
            owner._poison()
    if isinstance(wrapper, IrisScientificAblationRequestV1):
        wrapper._state = "poisoned"


def _structure(wrapper: IrisScientificAblationRequestV1) -> _WrapperSeal:
    """No model getters, contexts, cloning, graph hashing or global RNG effects."""
    try:
        if type(wrapper) is not IrisScientificAblationRequestV1:
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
            or type(seal.request) is not _ScientificAblationOwnedRequest
            or type(seal.owner) is not _AblationRequestOwner
            or seal.request._ablation_anchor is not seal.owner
            or seal.request._ablation_binding is not seal.binding
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


def _entry(wrapper: IrisScientificAblationRequestV1) -> _WrapperSeal:
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
    wrapper: IrisScientificAblationRequestV1,
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


def _restore_context(snapshot: RuntimeStateSnapshot, device: torch.device) -> None:
    restore_runtime_state(snapshot)
    if not _same_runtime(snapshot, capture_runtime_state(device)):
        raise IrisRequestError("Caller runtime restoration returned without exact restoration")


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


def build_scientific_ablation_request_v1(
    bundle: IrisScientificAblationInferenceBundleV1,
    *,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
    backbone: BackboneAdapter,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisScientificAblationRequestV1:
    """Verify/authenticate BEFORE decode/construction; privately compose and copy.

    This consumes an in-memory verified bundle, not a file-system publisher.
    Candidate-derived expected hashes defeat authentication. The supplied already
    loaded frozen backbone remains externally owned; sidecar graph/copies are fresh.
    """
    wrapper: IrisScientificAblationRequestV1 | None = None
    caller: RuntimeStateSnapshot | None = None
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
        caller = capture_runtime_state(device)
        _compatible_backbone(manifest, backbone)
        if type(sampling) is not TemperatureTopPSamplingConfig:
            raise IrisRequestError("Exact mode-bound sampling policy required")
        sampling = TemperatureTopPSamplingConfig(sampling.temperature, sampling.top_p)
        wrapper = object.__new__(IrisScientificAblationRequestV1)
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
            constructor=_ScientificAblationOwnedRequest,
        )
        payload = private.payload
        binding = _AblationRequestBinding(
            manifest.sha256,
            payload.source_checkpoint_sha256,
            payload.selection.lifecycle_sha256,
            payload.ablation_policy_sha256,
            payload.model.sha256,
        )
        owner = _AblationRequestOwner(request, wrapper=wrapper, binding=binding)
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
        _restore_context(caller, device)
        _boundary(wrapper)  # Restoration callbacks cannot change the bound graph.
        if not _same_runtime(caller, capture_runtime_state(device)):
            raise IrisRequestError("Loader postchecks changed caller runtime")
        wrapper._state = "ready"
        return wrapper
    except BaseException as primary:
        if wrapper is not None:
            _poison(wrapper)
        secondary: list[BaseException] = []
        if caller is not None and device is not None:
            try:
                _restore_context(caller, device)
            except BaseException as cleanup:
                secondary.append(cleanup)
        _raise_failed(primary, secondary)


def _receipt(
    manifest: IrisScientificAblationInferenceManifestV1,
    request: _ScientificAblationOwnedRequest,
    core: _CoreRequestExecutionOutput,
    session: CognitiveAblationSessionV1,
    prompt: str,
    strategy: DecodingStrategy,
    seed: int | None,
    initial: torch.Tensor | None,
    final: torch.Tensor | None,
) -> IrisScientificAblationRequestReceiptV1:
    if (
        type(core) is not _CoreRequestExecutionOutput
        or type(core.generation) is not IrisGeneratorOutput
        or type(session._length) is not int
        or session._length != core.prompt_attention_mask.shape[1]
        or len(core.generation.workspace_trace) != request._cycles
        or (
            core.diagnostic_trace is not None
            and core.diagnostic_trace.intervention_plan_sha256 != session.policy_sha256
        )
    ):
        raise IrisRequestError("Completed core facts/session/diagnostic binding differs")
    payload = manifest.payload
    return IrisScientificAblationRequestReceiptV1(
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
        sidecar_state_sha256=payload.model.sha256,
        selection=payload.selection.kind,
        completed_steps=payload.completed_steps,
        purpose=payload.purpose,
        prompt_sha256=domain_sha256(_RECEIPT_DOMAIN + "/prompt", prompt.encode("utf-8")),
        cognitive_prompt_sha256=session.prompt_sha256,
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
    wrapper: IrisScientificAblationRequestV1,
    prompt: str,
    *,
    strategy: DecodingStrategy,
    generator: torch.Generator | None = None,
    diagnostics: WorkspaceTraceOptionsV1 | None = None,
) -> IrisScientificAblationRequestOutputV1:
    seal = _entry(wrapper)  # Reject missing fields/reentry BEFORE context/RNG/model getters.
    IrisRequestOrchestrator._validate_request(prompt, strategy=strategy, generator=generator)
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
        session = CognitiveAblationSessionV1(
            manifest.payload.ablation,
            prompt_sha256=domain_sha256(
                "iris/cognitive-ablation-prompt/v1", prompt.encode("utf-8")
            ),
            num_modules=request._num_modules,
            num_slots=request._num_slots,
            state_dim=request._state_dim,
            cycles=request._cycles,
        )
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
                ablation=session,
                diagnostics=diagnostics,
                _authorization=cap,
            )
        owner._verify_execution(wrapper, cap)  # Keep busy through ALL outer postchecks.
        manifest = _boundary(wrapper)
        final = None if local is None else local.get_state().detach().clone()
        if local is not None and local.initial_seed() != seed:
            raise IrisRequestError("Private sampling generator was reseeded")
        receipt = _receipt(manifest, request, core, session, prompt, strategy, seed, initial, final)
        output = IrisScientificAblationRequestOutputV1(
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
        if not _same_runtime(caller, capture_runtime_state(seal.device)):
            raise IrisRequestError("Mode-bound execution changed caller runtime globals")
        _restore_context(caller, seal.device)
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
            if generator.initial_seed() != seed:
                raise IrisRequestError("Caller sampling generator seed changed during commit")
        _boundary(wrapper)
        if not _same_runtime(caller, capture_runtime_state(seal.device)):
            raise IrisRequestError("Mode-bound finalization changed caller runtime globals")
        wrapper._state = "ready"
        return output
    except BaseException as primary:
        _poison(wrapper)
        secondary: list[BaseException] = []
        if generator is not None and initial is not None:
            try:
                _restore_sampling_state(generator, initial)
            except BaseException as cleanup:
                secondary.append(cleanup)
        if caller is not None:
            try:
                _restore_context(caller, seal.device)
            except BaseException as cleanup:
                secondary.append(cleanup)
        _raise_failed(primary, secondary)


__all__ = ["IrisScientificAblationRequestV1", "build_scientific_ablation_request_v1"]
