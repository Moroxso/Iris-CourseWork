"""Validated construction of the complete Iris request graph."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from iris.application.errors import IrisCompositionError
from iris.application.request import IrisRequestOrchestrator
from iris.backbone import BackboneAdapter
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import (
    IrisGenerator,
    SecondPassGenerationPrefixAssembler,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import (
    CognitiveConfig,
    LayerSamplerConfig,
    SlotAllocatorConfig,
    WorkspaceArbiterConfig,
    WorkspaceConfig,
)
from iris.workspace import (
    LatentBridge,
    SlotAllocator,
    WorkspaceArbiter,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceWriter,
)


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    """One stable public metadata snapshot from an already-loaded backbone."""

    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_new_tokens: int
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _CompositionInputs:
    """Immutable scalar and nested-schema snapshot used for all construction."""

    backbone: _BackboneMetadata
    cognitive_snapshot: str
    workspace_snapshot: str
    state_dim: int
    num_modules: int
    layer_sampler: LayerSamplerConfig
    num_slots: int
    cycles: int
    winners_per_cycle: int
    decay_rate: float
    empty_threshold: float
    arbiter: WorkspaceArbiterConfig
    allocator: SlotAllocatorConfig


def _stage[ResultT](name: str, operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except IrisCompositionError:
        raise
    except Exception as error:
        message = f"Iris request composition failed during {name}"
        raise IrisCompositionError(message) from error


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        message = f"Backbone {name} must be a positive integer"
        raise ValueError(message)
    return value


def _read_backbone_metadata(backbone: BackboneAdapter) -> _BackboneMetadata:
    hidden_size = _positive_integer("hidden_size", backbone.hidden_size)
    num_hidden_layers = _positive_integer(
        "num_hidden_layers",
        backbone.num_hidden_layers,
    )
    vocabulary_size = _positive_integer("vocabulary_size", backbone.vocabulary_size)
    max_position_embeddings = _positive_integer(
        "max_position_embeddings",
        backbone.max_position_embeddings,
    )
    max_prompt_tokens = _positive_integer(
        "max_prompt_tokens",
        backbone.max_prompt_tokens,
    )
    max_new_tokens = _positive_integer("max_new_tokens", backbone.max_new_tokens)
    device = backbone.device
    dtype = backbone.dtype
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        message = "Backbone device must be an explicit CPU or CUDA torch.device"
        raise ValueError(message)
    if device.index is not None and device.index < 0:
        message = "Backbone device index cannot be negative"
        raise ValueError(message)
    if device.type == "cuda" and device.index is None:
        message = "Backbone CUDA device must have one concrete index"
        raise ValueError(message)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        message = "Backbone dtype must be fp16, bf16, or fp32"
        raise ValueError(message)
    if device.type == "cpu" and dtype != torch.float32:
        message = "Iris-GW-0.1 requires fp32 sidecar composition on CPU"
        raise ValueError(message)
    return _BackboneMetadata(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocabulary_size=vocabulary_size,
        max_position_embeddings=max_position_embeddings,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
    )


def _validate_preflight(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    workspace_config: WorkspaceConfig,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> _CompositionInputs:
    if not isinstance(backbone, BackboneAdapter):
        message = "backbone must be an already-loaded BackboneAdapter"
        raise TypeError(message)
    if not isinstance(cognitive_config, CognitiveConfig):
        message = "cognitive_config must be a validated CognitiveConfig"
        raise TypeError(message)
    if not isinstance(workspace_config, WorkspaceConfig):
        message = "workspace_config must be a validated WorkspaceConfig"
        raise TypeError(message)
    if not isinstance(sampling, TemperatureTopPSamplingConfig):
        message = "sampling must be TemperatureTopPSamplingConfig"
        raise TypeError(message)
    if type(stop_token_ids) is not tuple:
        message = "stop_token_ids must be an explicit tuple"
        raise TypeError(message)
    if any(type(token_id) is not int for token_id in stop_token_ids):
        message = "stop_token_ids must contain only integers"
        raise ValueError(message)
    if len(set(stop_token_ids)) != len(stop_token_ids):
        message = "stop_token_ids must be unique"
        raise ValueError(message)
    cognitive_snapshot = cognitive_config.model_dump_json()
    workspace_snapshot = workspace_config.model_dump_json()
    state_dim = cognitive_config.state_dim
    num_modules = cognitive_config.num_modules
    layer_sampler = cognitive_config.layer_sampler
    num_slots = workspace_config.num_slots
    cycles = workspace_config.cycles
    winners_per_cycle = workspace_config.winners_per_cycle
    decay_rate = workspace_config.decay_rate
    empty_threshold = workspace_config.empty_threshold
    arbiter = workspace_config.arbiter
    allocator = workspace_config.allocator
    for name, value in {
        "cognitive state_dim": state_dim,
        "cognitive num_modules": num_modules,
        "workspace num_slots": num_slots,
        "workspace cycles": cycles,
        "workspace winners_per_cycle": winners_per_cycle,
    }.items():
        if type(value) is not int or value <= 0:
            message = f"{name} must be a positive integer"
            raise ValueError(message)
    if not isinstance(layer_sampler, LayerSamplerConfig):
        message = "cognitive layer_sampler must be a validated LayerSamplerConfig"
        raise TypeError(message)
    if not isinstance(arbiter, WorkspaceArbiterConfig):
        message = "workspace arbiter must be a validated WorkspaceArbiterConfig"
        raise TypeError(message)
    if not isinstance(allocator, SlotAllocatorConfig):
        message = "workspace allocator must be a validated SlotAllocatorConfig"
        raise TypeError(message)
    if type(decay_rate) is not float or not isfinite(decay_rate) or not 0.0 < decay_rate < 1.0:
        message = "workspace decay_rate must be a float in (0, 1)"
        raise ValueError(message)
    if (
        type(empty_threshold) is not float
        or not isfinite(empty_threshold)
        or empty_threshold <= 0.0
    ):
        message = "workspace empty_threshold must be a positive float"
        raise ValueError(message)
    if winners_per_cycle > num_modules:
        message = "winners_per_cycle cannot exceed cognitive num_modules"
        raise ValueError(message)

    metadata = _read_backbone_metadata(backbone)
    LayerSampler.resolve_layer_indices(
        layer_sampler.layer_fractions,
        metadata.num_hidden_layers,
    )
    if any(token_id < 0 or token_id >= metadata.vocabulary_size for token_id in stop_token_ids):
        message = "stop_token_ids contains an out-of-vocabulary ID"
        raise ValueError(message)
    if (
        metadata.max_prompt_tokens + num_slots + metadata.max_new_tokens
        > metadata.max_position_embeddings
    ):
        message = "Prompt, workspace, and generation maxima exceed backbone context"
        raise ValueError(message)
    if cognitive_config.model_dump_json() != cognitive_snapshot:
        message = "Backbone metadata access mutated cognitive_config"
        raise ValueError(message)
    if workspace_config.model_dump_json() != workspace_snapshot:
        message = "Backbone metadata access mutated workspace_config"
        raise ValueError(message)
    return _CompositionInputs(
        backbone=metadata,
        cognitive_snapshot=cognitive_snapshot,
        workspace_snapshot=workspace_snapshot,
        state_dim=state_dim,
        num_modules=num_modules,
        layer_sampler=layer_sampler,
        num_slots=num_slots,
        cycles=cycles,
        winners_per_cycle=winners_per_cycle,
        decay_rate=decay_rate,
        empty_threshold=empty_threshold,
        arbiter=arbiter,
        allocator=allocator,
    )


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


def _validate_module(
    name: str,
    module: nn.Module,
    *,
    metadata: _BackboneMetadata,
) -> None:
    if any(child.training for child in module.modules()):
        message = f"{name} and all children must be in evaluation mode"
        raise ValueError(message)
    for parameter in module.parameters():
        if parameter.is_inference():
            message = f"{name} parameters must be ordinary trainable tensors"
            raise ValueError(message)
        if not parameter.requires_grad:
            message = f"{name} parameters must remain trainable"
            raise ValueError(message)
        if not _device_matches(parameter.device, metadata.device):
            message = f"{name} parameters are not on the backbone device"
            raise ValueError(message)
        if torch.is_floating_point(parameter) and parameter.dtype != metadata.dtype:
            message = f"{name} parameters do not use the backbone dtype"
            raise ValueError(message)
        if torch.is_floating_point(parameter) and not bool(torch.isfinite(parameter).all()):
            message = f"{name} parameters contain NaN or Inf"
            raise ValueError(message)
    for buffer in module.buffers():
        if buffer.is_inference():
            message = f"{name} buffers must be ordinary tensors"
            raise ValueError(message)
        if not _device_matches(buffer.device, metadata.device):
            message = f"{name} buffers are not on the backbone device"
            raise ValueError(message)
        if torch.is_floating_point(buffer) and buffer.dtype != metadata.dtype:
            message = f"{name} floating buffers do not use the backbone dtype"
            raise ValueError(message)
        if torch.is_floating_point(buffer) and not bool(torch.isfinite(buffer).all()):
            message = f"{name} buffers contain NaN or Inf"
            raise ValueError(message)


def _prepare_module[ModuleT: nn.Module](
    name: str,
    module: ModuleT,
    *,
    metadata: _BackboneMetadata,
) -> ModuleT:
    if not isinstance(module, nn.Module):
        message = f"{name} constructor did not return a torch.nn.Module"
        raise TypeError(message)
    placed = module.to(device=metadata.device, dtype=metadata.dtype)
    if placed is not module:
        message = f"{name}.to() must preserve module identity"
        raise ValueError(message)
    evaluated = module.eval()
    if evaluated is not module:
        message = f"{name}.eval() must preserve module identity"
        raise ValueError(message)
    _validate_module(name, module, metadata=metadata)
    return module


def _validate_stable_inputs(
    *,
    backbone: BackboneAdapter,
    metadata: _BackboneMetadata,
    cognitive_config: CognitiveConfig,
    cognitive_snapshot: str,
    workspace_config: WorkspaceConfig,
    workspace_snapshot: str,
    modules: tuple[tuple[str, nn.Module], ...],
) -> None:
    if _read_backbone_metadata(backbone) != metadata:
        message = "Backbone metadata changed during request composition"
        raise ValueError(message)
    if cognitive_config.model_dump_json() != cognitive_snapshot:
        message = "cognitive_config changed during request composition"
        raise ValueError(message)
    if workspace_config.model_dump_json() != workspace_snapshot:
        message = "workspace_config changed during request composition"
        raise ValueError(message)
    for name, module in modules:
        _validate_module(name, module, metadata=metadata)


def _validate_generator(
    generator: IrisGenerator,
    *,
    backbone: BackboneAdapter,
    metadata: _BackboneMetadata,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> None:
    if not isinstance(generator, IrisGenerator):
        message = "IrisGenerator construction returned an invalid object"
        raise TypeError(message)
    if generator.backbone is not backbone:
        message = "IrisGenerator must own the exact composed backbone"
        raise ValueError(message)
    if type(generator.max_new_tokens) is not int or (
        generator.max_new_tokens != metadata.max_new_tokens
    ):
        message = "IrisGenerator max_new_tokens changed during composition"
        raise ValueError(message)
    if generator.stop_token_ids is not stop_token_ids:
        message = "IrisGenerator must preserve the exact stop-token tuple"
        raise ValueError(message)
    if generator.sampling is not sampling:
        message = "IrisGenerator must preserve the exact sampling policy"
        raise ValueError(message)


def build_request_orchestrator(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    workspace_config: WorkspaceConfig,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> IrisRequestOrchestrator:
    """Compose a placed eval-only request graph around one loaded backbone.

    Configuration parsing, backbone loading, seed ownership, checkpoints, and
    sampling-generator creation deliberately remain outside this factory.
    """
    return _compose_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive_config,
        workspace_config=workspace_config,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        constructor=IrisRequestOrchestrator,
    )


def _compose_request_orchestrator[RequestT: IrisRequestOrchestrator](
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    workspace_config: WorkspaceConfig,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
    constructor: Callable[..., RequestT],
) -> RequestT:
    """Private final-constructor seam; public factory and placement order unchanged."""
    inputs = _stage(
        "preflight validation",
        lambda: _validate_preflight(
            backbone=backbone,
            cognitive_config=cognitive_config,
            workspace_config=workspace_config,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        ),
    )
    metadata = inputs.backbone
    state_dim = inputs.state_dim
    num_modules = inputs.num_modules
    num_slots = inputs.num_slots
    cycles = inputs.cycles
    winners_per_cycle = inputs.winners_per_cycle

    # Sidecar initialization deliberately consumes the caller-owned CPU RNG in
    # one stable order, independent of a process-wide default device.
    with torch.inference_mode(False), torch.device("cpu"):
        layer_sampler = _stage(
            "LayerSampler construction and placement",
            lambda: _prepare_module(
                "layer_sampler",
                LayerSampler(
                    inputs.layer_sampler,
                    backbone_hidden_size=metadata.hidden_size,
                    state_dim=state_dim,
                ),
                metadata=metadata,
            ),
        )
        goal_encoder = _stage(
            "GoalEncoder construction and placement",
            lambda: _prepare_module(
                "goal_encoder",
                GoalEncoder(state_dim=state_dim),
                metadata=metadata,
            ),
        )
        module_input_router = _stage(
            "ModuleInputRouter construction and placement",
            lambda: _prepare_module(
                "module_input_router",
                ModuleInputRouter(state_dim=state_dim, num_modules=num_modules),
                metadata=metadata,
            ),
        )
        workspace_reader = _stage(
            "WorkspaceReader construction and placement",
            lambda: _prepare_module(
                "workspace_reader",
                WorkspaceReader(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    num_slots=num_slots,
                ),
                metadata=metadata,
            ),
        )
        cognitive_modules = _stage(
            "CognitiveModule construction and placement",
            lambda: _prepare_module(
                "cognitive_modules",
                CognitiveModule(state_dim=state_dim, num_modules=num_modules),
                metadata=metadata,
            ),
        )
        proposal_head = _stage(
            "ProposalHead construction and placement",
            lambda: _prepare_module(
                "proposal_head",
                ProposalHead(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    cycles=cycles,
                ),
                metadata=metadata,
            ),
        )
        workspace_arbiter = _stage(
            "WorkspaceArbiter construction and placement",
            lambda: _prepare_module(
                "workspace_arbiter",
                WorkspaceArbiter(
                    inputs.arbiter,
                    state_dim=state_dim,
                    num_modules=num_modules,
                    num_slots=num_slots,
                    winners_per_cycle=winners_per_cycle,
                ),
                metadata=metadata,
            ),
        )
        slot_allocator = _stage(
            "SlotAllocator construction and placement",
            lambda: _prepare_module(
                "slot_allocator",
                SlotAllocator(
                    inputs.allocator,
                    state_dim=state_dim,
                    num_modules=num_modules,
                    num_slots=num_slots,
                    winners_per_cycle=winners_per_cycle,
                ),
                metadata=metadata,
            ),
        )
        workspace_writer = _stage(
            "WorkspaceWriter construction and placement",
            lambda: _prepare_module(
                "workspace_writer",
                WorkspaceWriter(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    num_slots=num_slots,
                    winners_per_cycle=winners_per_cycle,
                ),
                metadata=metadata,
            ),
        )
        workspace_decay = _stage(
            "WorkspaceDecay construction and placement",
            lambda: _prepare_module(
                "workspace_decay",
                WorkspaceDecay(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    num_slots=num_slots,
                    decay_rate=inputs.decay_rate,
                    empty_threshold=inputs.empty_threshold,
                ),
                metadata=metadata,
            ),
        )
        latent_bridge = _stage(
            "LatentBridge construction and placement",
            lambda: _prepare_module(
                "latent_bridge",
                LatentBridge(
                    state_dim=state_dim,
                    backbone_hidden_size=metadata.hidden_size,
                    num_slots=num_slots,
                ),
                metadata=metadata,
            ),
        )
        prefix_assembler = _stage(
            "generation-prefix construction and placement",
            lambda: _prepare_module(
                "prefix_assembler",
                SecondPassGenerationPrefixAssembler(
                    hidden_size=metadata.hidden_size,
                    num_slots=num_slots,
                    max_prompt_tokens=metadata.max_prompt_tokens,
                    max_new_tokens=metadata.max_new_tokens,
                    max_context_tokens=metadata.max_position_embeddings,
                ),
                metadata=metadata,
            ),
        )

    modules: tuple[tuple[str, nn.Module], ...] = (
        ("layer_sampler", layer_sampler),
        ("goal_encoder", goal_encoder),
        ("module_input_router", module_input_router),
        ("workspace_reader", workspace_reader),
        ("cognitive_modules", cognitive_modules),
        ("proposal_head", proposal_head),
        ("workspace_arbiter", workspace_arbiter),
        ("slot_allocator", slot_allocator),
        ("workspace_writer", workspace_writer),
        ("workspace_decay", workspace_decay),
        ("latent_bridge", latent_bridge),
        ("prefix_assembler", prefix_assembler),
    )
    _stage(
        "post-placement validation",
        lambda: _validate_stable_inputs(
            backbone=backbone,
            metadata=metadata,
            cognitive_config=cognitive_config,
            cognitive_snapshot=inputs.cognitive_snapshot,
            workspace_config=workspace_config,
            workspace_snapshot=inputs.workspace_snapshot,
            modules=modules,
        ),
    )
    iris_generator = _stage(
        "IrisGenerator construction",
        lambda: IrisGenerator(
            backbone,
            max_new_tokens=metadata.max_new_tokens,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        ),
    )
    _stage(
        "IrisGenerator validation",
        lambda: _validate_generator(
            iris_generator,
            backbone=backbone,
            metadata=metadata,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        ),
    )
    orchestrator = _stage(
        "IrisRequestOrchestrator wiring",
        lambda: constructor(
            backbone=backbone,
            cognitive_config=cognitive_config,
            workspace_config=workspace_config,
            layer_sampler=layer_sampler,
            goal_encoder=goal_encoder,
            module_input_router=module_input_router,
            workspace_reader=workspace_reader,
            cognitive_modules=cognitive_modules,
            proposal_head=proposal_head,
            workspace_arbiter=workspace_arbiter,
            slot_allocator=slot_allocator,
            workspace_writer=workspace_writer,
            workspace_decay=workspace_decay,
            latent_bridge=latent_bridge,
            prefix_assembler=prefix_assembler,
            iris_generator=iris_generator,
        ),
    )
    _stage(
        "final composition validation",
        lambda: _validate_stable_inputs(
            backbone=backbone,
            metadata=metadata,
            cognitive_config=cognitive_config,
            cognitive_snapshot=inputs.cognitive_snapshot,
            workspace_config=workspace_config,
            workspace_snapshot=inputs.workspace_snapshot,
            modules=modules,
        ),
    )
    _stage(
        "final IrisGenerator validation",
        lambda: _validate_generator(
            iris_generator,
            backbone=backbone,
            metadata=metadata,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        ),
    )
    return orchestrator
