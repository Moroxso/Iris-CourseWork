"""Validated construction of the Iris teacher-forced training graph."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from iris.application.errors import IrisTeacherCompositionError
from iris.backbone import BackboneAdapter
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import SecondPassSequenceAssembler
from iris.schemas import (
    CognitiveConfig,
    LayerSamplerConfig,
    SlotAllocatorConfig,
    WorkspaceArbiterConfig,
    WorkspaceConfig,
)
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import module_state_sha256
from iris.workspace import (
    LatentBridge,
    SlotAllocator,
    WorkspaceArbiter,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceWriter,
)

_IRIS_TEACHER_FORCING_MODEL_TYPE = IrisTeacherForcingModel


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    """One stable public snapshot from an already-loaded frozen backbone."""

    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_target_tokens: int
    device: torch.device
    dtype: torch.dtype
    trainable_parameter_count: int
    has_parameter_gradients: bool


@dataclass(frozen=True, slots=True)
class _CompositionInputs:
    """Immutable configuration and backbone values used for construction."""

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
    except IrisTeacherCompositionError:
        raise
    except Exception as error:
        message = f"Iris teacher composition failed during {name}"
        raise IrisTeacherCompositionError(message) from error


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
    max_target_tokens = _positive_integer(
        "max_new_tokens",
        backbone.max_new_tokens,
    )
    device = backbone.device
    dtype = backbone.dtype
    trainable_parameter_count = backbone.trainable_parameter_count
    has_parameter_gradients = backbone.has_parameter_gradients

    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        message = "Backbone device must be an explicit CPU or CUDA torch.device"
        raise ValueError(message)
    if device.type == "cpu":
        if device != torch.device("cpu"):
            message = "CPU backbone must use the canonical indexless CPU device"
            raise ValueError(message)
    elif device.index is None or device.index < 0:
        message = "CUDA backbone must use one concrete nonnegative device index"
        raise ValueError(message)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        message = "Backbone dtype must be fp16, bf16, or fp32"
        raise ValueError(message)
    if device.type == "cpu" and dtype != torch.float32:
        message = "Iris-GW-0.1 requires fp32 sidecar composition on CPU"
        raise ValueError(message)
    if type(trainable_parameter_count) is not int or trainable_parameter_count != 0:
        message = "Loaded backbone must not contain trainable parameters"
        raise ValueError(message)
    if type(has_parameter_gradients) is not bool or has_parameter_gradients:
        message = "Loaded backbone parameters must not contain gradients"
        raise ValueError(message)

    return _BackboneMetadata(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocabulary_size=vocabulary_size,
        max_position_embeddings=max_position_embeddings,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        device=device,
        dtype=dtype,
        trainable_parameter_count=trainable_parameter_count,
        has_parameter_gradients=has_parameter_gradients,
    )


def _validate_preflight(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    workspace_config: WorkspaceConfig,
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
    if winners_per_cycle > num_slots:
        message = "winners_per_cycle cannot exceed workspace num_slots"
        raise ValueError(message)

    metadata = _read_backbone_metadata(backbone)
    LayerSampler.resolve_layer_indices(
        layer_sampler.layer_fractions,
        metadata.num_hidden_layers,
    )
    if (
        metadata.max_prompt_tokens + num_slots + metadata.max_target_tokens
        > metadata.max_position_embeddings
    ):
        message = "Prompt, workspace, and target maxima exceed backbone context"
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


def _validate_module(
    name: str,
    module: nn.Module,
    *,
    metadata: _BackboneMetadata,
) -> None:
    if not isinstance(module, nn.Module):
        message = f"{name} must be a torch.nn.Module"
        raise TypeError(message)
    non_training_child = next((child for child in module.modules() if not child.training), None)
    if non_training_child is not None:
        message = f"{name} and all children must be in training mode"
        raise ValueError(message)
    for parameter_name, parameter in module.named_parameters():
        if parameter.is_inference():
            message = f"{name} parameter {parameter_name!r} must be an ordinary tensor"
            raise ValueError(message)
        if not parameter.requires_grad:
            message = f"{name} parameter {parameter_name!r} must remain trainable"
            raise ValueError(message)
        if parameter.device != metadata.device:
            message = f"{name} parameters are not on the exact backbone device"
            raise ValueError(message)
        if torch.is_floating_point(parameter) and parameter.dtype != metadata.dtype:
            message = f"{name} parameters do not use the backbone dtype"
            raise ValueError(message)
        if torch.is_floating_point(parameter) and not bool(torch.isfinite(parameter).all()):
            message = f"{name} parameter {parameter_name!r} contains NaN or Inf"
            raise ValueError(message)
    for buffer_name, buffer in module.named_buffers():
        if buffer.is_inference():
            message = f"{name} buffer {buffer_name!r} must be an ordinary tensor"
            raise ValueError(message)
        if buffer.device != metadata.device:
            message = f"{name} buffers are not on the exact backbone device"
            raise ValueError(message)
        if torch.is_floating_point(buffer) and buffer.dtype != metadata.dtype:
            message = f"{name} floating buffers do not use the backbone dtype"
            raise ValueError(message)
        if torch.is_floating_point(buffer) and not bool(torch.isfinite(buffer).all()):
            message = f"{name} buffer {buffer_name!r} contains NaN or Inf"
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
    trained = module.train(True)
    if trained is not module:
        message = f"{name}.train() must preserve module identity"
        raise ValueError(message)
    _validate_module(name, module, metadata=metadata)
    return module


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            return False
        return all(
            _same_typed_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _validate_component_contracts(
    components: tuple[tuple[str, nn.Module], ...],
    *,
    inputs: _CompositionInputs,
) -> None:
    metadata = inputs.backbone
    expected: dict[str, dict[str, object]] = {
        "layer_sampler": {
            "backbone_hidden_size": metadata.hidden_size,
            "state_dim": inputs.state_dim,
            "layer_fractions": inputs.layer_sampler.layer_fractions,
        },
        "goal_encoder": {"state_dim": inputs.state_dim},
        "module_input_router": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
        },
        "workspace_reader": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
            "num_slots": inputs.num_slots,
        },
        "cognitive_modules": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
        },
        "proposal_head": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
            "cycles": inputs.cycles,
        },
        "workspace_arbiter": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
            "num_slots": inputs.num_slots,
            "winners_per_cycle": inputs.winners_per_cycle,
            "confidence_weight": inputs.arbiter.confidence_weight,
            "novelty_weight": inputs.arbiter.novelty_weight,
            "relevance_weight": inputs.arbiter.relevance_weight,
        },
        "slot_allocator": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
            "num_slots": inputs.num_slots,
            "winners_per_cycle": inputs.winners_per_cycle,
            "empty_bonus": inputs.allocator.empty_bonus,
            "allocation_age_penalty": inputs.allocator.allocation_age_penalty,
            "retention_age_penalty": inputs.allocator.retention_age_penalty,
        },
        "workspace_writer": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
            "num_slots": inputs.num_slots,
            "winners_per_cycle": inputs.winners_per_cycle,
        },
        "workspace_decay": {
            "state_dim": inputs.state_dim,
            "num_modules": inputs.num_modules,
            "num_slots": inputs.num_slots,
            "decay_rate": inputs.decay_rate,
            "empty_threshold": inputs.empty_threshold,
        },
        "latent_bridge": {
            "state_dim": inputs.state_dim,
            "backbone_hidden_size": metadata.hidden_size,
            "num_slots": inputs.num_slots,
        },
        "sequence_assembler": {
            "hidden_size": metadata.hidden_size,
            "num_slots": inputs.num_slots,
            "vocabulary_size": metadata.vocabulary_size,
            "max_prompt_tokens": metadata.max_prompt_tokens,
            "max_target_tokens": metadata.max_target_tokens,
            "max_context_tokens": metadata.max_position_embeddings,
        },
    }
    if tuple(name for name, _ in components) != tuple(expected):
        message = "Teacher sidecar components do not follow the canonical order"
        raise ValueError(message)
    for component_name, component in components:
        for attribute, expected_value in expected[component_name].items():
            try:
                actual = getattr(component, attribute)
            except Exception as error:
                message = f"Failed to read {component_name}.{attribute}"
                raise ValueError(message) from error
            if not _same_typed_value(actual, expected_value):
                message = (
                    f"{component_name}.{attribute}={actual!r} does not match "
                    f"teacher configuration {expected_value!r}"
                )
                raise ValueError(message)


def _validate_stable_inputs(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    workspace_config: WorkspaceConfig,
    inputs: _CompositionInputs,
    components: tuple[tuple[str, nn.Module], ...],
    expected_state_sha256: str | None = None,
) -> str:
    if _read_backbone_metadata(backbone) != inputs.backbone:
        message = "Backbone metadata changed during teacher composition"
        raise ValueError(message)
    if cognitive_config.model_dump_json() != inputs.cognitive_snapshot:
        message = "cognitive_config changed during teacher composition"
        raise ValueError(message)
    if workspace_config.model_dump_json() != inputs.workspace_snapshot:
        message = "workspace_config changed during teacher composition"
        raise ValueError(message)
    for name, module in components:
        _validate_module(name, module, metadata=inputs.backbone)
    _validate_component_contracts(components, inputs=inputs)
    state_sha256 = module_state_sha256(components)
    if expected_state_sha256 is not None and state_sha256 != expected_state_sha256:
        message = "Teacher sidecar module state changed during model construction"
        raise ValueError(message)
    return state_sha256


def _validate_model(
    model: IrisTeacherForcingModel,
    *,
    components: tuple[tuple[str, nn.Module], ...],
) -> None:
    if not isinstance(model, _IRIS_TEACHER_FORCING_MODEL_TYPE):
        message = "Teacher model construction returned an invalid object"
        raise TypeError(message)
    if not model.training or any(not child.training for child in model.modules()):
        message = "Teacher model and all registered children must be in training mode"
        raise ValueError(message)
    for name, component in components:
        try:
            owned = getattr(model, name)
        except Exception as error:
            message = f"Failed to read teacher model component {name!r}"
            raise ValueError(message) from error
        if owned is not component:
            message = f"Teacher model must preserve the exact {name} instance"
            raise ValueError(message)


def build_teacher_forcing_model(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    workspace_config: WorkspaceConfig,
) -> IrisTeacherForcingModel:
    """Compose a recursively-training teacher graph around one frozen backbone.

    The caller owns configuration parsing, backbone loading, initialization RNG,
    optimizer policy, target tokenization, checkpointing, and evaluation mode.
    This factory never executes the tokenizer or either frozen-model pass.
    """

    inputs = _stage(
        "preflight validation",
        lambda: _validate_preflight(
            backbone=backbone,
            cognitive_config=cognitive_config,
            workspace_config=workspace_config,
        ),
    )
    metadata = inputs.backbone

    # Constructor allocations consume the caller-owned CPU RNG in the same
    # stable sidecar order as request composition. The nested boundary prevents
    # ambient inference mode from creating immutable parameter tensors.
    with torch.inference_mode(False), torch.device("cpu"):
        layer_sampler = _stage(
            "LayerSampler construction and placement",
            lambda: _prepare_module(
                "layer_sampler",
                LayerSampler(
                    inputs.layer_sampler,
                    backbone_hidden_size=metadata.hidden_size,
                    state_dim=inputs.state_dim,
                ),
                metadata=metadata,
            ),
        )
        goal_encoder = _stage(
            "GoalEncoder construction and placement",
            lambda: _prepare_module(
                "goal_encoder",
                GoalEncoder(state_dim=inputs.state_dim),
                metadata=metadata,
            ),
        )
        module_input_router = _stage(
            "ModuleInputRouter construction and placement",
            lambda: _prepare_module(
                "module_input_router",
                ModuleInputRouter(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                ),
                metadata=metadata,
            ),
        )
        workspace_reader = _stage(
            "WorkspaceReader construction and placement",
            lambda: _prepare_module(
                "workspace_reader",
                WorkspaceReader(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    num_slots=inputs.num_slots,
                ),
                metadata=metadata,
            ),
        )
        cognitive_modules = _stage(
            "CognitiveModule construction and placement",
            lambda: _prepare_module(
                "cognitive_modules",
                CognitiveModule(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                ),
                metadata=metadata,
            ),
        )
        proposal_head = _stage(
            "ProposalHead construction and placement",
            lambda: _prepare_module(
                "proposal_head",
                ProposalHead(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    cycles=inputs.cycles,
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
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    num_slots=inputs.num_slots,
                    winners_per_cycle=inputs.winners_per_cycle,
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
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    num_slots=inputs.num_slots,
                    winners_per_cycle=inputs.winners_per_cycle,
                ),
                metadata=metadata,
            ),
        )
        workspace_writer = _stage(
            "WorkspaceWriter construction and placement",
            lambda: _prepare_module(
                "workspace_writer",
                WorkspaceWriter(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    num_slots=inputs.num_slots,
                    winners_per_cycle=inputs.winners_per_cycle,
                ),
                metadata=metadata,
            ),
        )
        workspace_decay = _stage(
            "WorkspaceDecay construction and placement",
            lambda: _prepare_module(
                "workspace_decay",
                WorkspaceDecay(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    num_slots=inputs.num_slots,
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
                    state_dim=inputs.state_dim,
                    backbone_hidden_size=metadata.hidden_size,
                    num_slots=inputs.num_slots,
                ),
                metadata=metadata,
            ),
        )
        sequence_assembler = _stage(
            "teacher sequence construction and placement",
            lambda: _prepare_module(
                "sequence_assembler",
                SecondPassSequenceAssembler(
                    hidden_size=metadata.hidden_size,
                    num_slots=inputs.num_slots,
                    vocabulary_size=metadata.vocabulary_size,
                    max_prompt_tokens=metadata.max_prompt_tokens,
                    max_target_tokens=metadata.max_target_tokens,
                    max_context_tokens=metadata.max_position_embeddings,
                ),
                metadata=metadata,
            ),
        )

    components: tuple[tuple[str, nn.Module], ...] = (
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
        ("sequence_assembler", sequence_assembler),
    )
    state_sha256 = _stage(
        "post-placement validation",
        lambda: _validate_stable_inputs(
            backbone=backbone,
            cognitive_config=cognitive_config,
            workspace_config=workspace_config,
            inputs=inputs,
            components=components,
        ),
    )
    with torch.inference_mode(False), torch.device("cpu"):
        model = _stage(
            "IrisTeacherForcingModel wiring",
            lambda: IrisTeacherForcingModel(
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
                sequence_assembler=sequence_assembler,
            ),
        )
    _stage(
        "teacher model validation",
        lambda: _validate_model(model, components=components),
    )
    _stage(
        "final composition validation",
        lambda: _validate_stable_inputs(
            backbone=backbone,
            cognitive_config=cognitive_config,
            workspace_config=workspace_config,
            inputs=inputs,
            components=components,
            expected_state_sha256=state_sha256,
        ),
    )
    return model
