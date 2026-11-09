"""Composition of the trainable no-workspace arm-B teacher graph."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from iris.application.errors import IrisTeacherCompositionError
from iris.backbone import BackboneAdapter
from iris.cognitive import (
    NO_WORKSPACE_COGNITIVE_CYCLES_V1,
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ModuleLatentBridge,
)
from iris.generation import ModuleSecondPassSequenceAssembler
from iris.schemas import CognitiveConfig, LayerSamplerConfig
from iris.training import IrisNoWorkspaceTeacherForcingModel
from iris.utils.hashing import module_state_sha256


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_target_tokens: int
    device: torch.device
    dtype: torch.dtype


def _stage[ResultT](name: str, operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except IrisTeacherCompositionError:
        raise
    except Exception as error:
        message = f"Arm-B teacher composition failed during {name}"
        raise IrisTeacherCompositionError(message) from error


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        message = f"Backbone {name} must be a positive integer"
        raise ValueError(message)
    return value


def _read_backbone_metadata(backbone: BackboneAdapter) -> _BackboneMetadata:
    if not isinstance(backbone, BackboneAdapter):
        message = "backbone must be an already-loaded BackboneAdapter"
        raise TypeError(message)
    metadata = _BackboneMetadata(
        hidden_size=_positive_integer("hidden_size", backbone.hidden_size),
        num_hidden_layers=_positive_integer(
            "num_hidden_layers",
            backbone.num_hidden_layers,
        ),
        vocabulary_size=_positive_integer("vocabulary_size", backbone.vocabulary_size),
        max_position_embeddings=_positive_integer(
            "max_position_embeddings",
            backbone.max_position_embeddings,
        ),
        max_prompt_tokens=_positive_integer(
            "max_prompt_tokens",
            backbone.max_prompt_tokens,
        ),
        max_target_tokens=_positive_integer(
            "max_target_tokens",
            backbone.max_new_tokens,
        ),
        device=backbone.device,
        dtype=backbone.dtype,
    )
    if metadata.device.type not in {"cpu", "cuda"}:
        message = "Arm-B teacher backbone device must be CPU or CUDA"
        raise ValueError(message)
    if metadata.device.index is not None and metadata.device.index < 0:
        message = "Arm-B teacher backbone device index cannot be negative"
        raise ValueError(message)
    if metadata.device.type == "cuda" and metadata.device.index is None:
        message = "Arm-B teacher CUDA device must have a concrete index"
        raise ValueError(message)
    if metadata.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        message = "Arm-B teacher backbone dtype is unsupported"
        raise ValueError(message)
    if metadata.device.type == "cpu" and metadata.dtype != torch.float32:
        message = "Arm-B teacher CPU execution requires fp32"
        raise ValueError(message)
    if type(backbone.trainable_parameter_count) is not int or (
        backbone.trainable_parameter_count != 0
    ):
        message = "Loaded arm-B teacher backbone must be frozen"
        raise ValueError(message)
    if type(backbone.has_parameter_gradients) is not bool or (backbone.has_parameter_gradients):
        message = "Loaded arm-B teacher backbone must not carry gradients"
        raise ValueError(message)
    return metadata


def _validate_module(
    name: str,
    module: nn.Module,
    *,
    metadata: _BackboneMetadata,
) -> None:
    if not isinstance(module, nn.Module):
        message = f"{name} must be a torch.nn.Module"
        raise TypeError(message)
    if any(not child.training for child in module.modules()):
        message = f"{name} and all children must be in training mode"
        raise ValueError(message)
    for parameter_name, parameter in module.named_parameters():
        if (
            type(parameter) is not nn.Parameter
            or parameter.is_inference()
            or not parameter.requires_grad
            or parameter.device != metadata.device
            or (torch.is_floating_point(parameter) and parameter.dtype != metadata.dtype)
            or not bool(torch.isfinite(parameter).all())
        ):
            message = f"{name} parameter {parameter_name!r} violates placement ownership"
            raise ValueError(message)
    for buffer_name, buffer in module.named_buffers():
        if (
            type(buffer) is not torch.Tensor
            or buffer.is_inference()
            or buffer.device != metadata.device
            or (torch.is_floating_point(buffer) and buffer.dtype != metadata.dtype)
            or (torch.is_floating_point(buffer) and not bool(torch.isfinite(buffer).all()))
        ):
            message = f"{name} buffer {buffer_name!r} violates placement ownership"
            raise ValueError(message)


def _prepare_module[ModuleT: nn.Module](
    name: str,
    module: ModuleT,
    *,
    metadata: _BackboneMetadata,
) -> ModuleT:
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


def _validate_preflight(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
) -> tuple[_BackboneMetadata, str, int, int, LayerSamplerConfig]:
    if type(cognitive_config) is not CognitiveConfig:
        message = "cognitive_config must be exactly CognitiveConfig"
        raise TypeError(message)
    snapshot = cognitive_config.model_dump_json()
    state_dim = cognitive_config.state_dim
    num_modules = cognitive_config.num_modules
    layer_sampler = cognitive_config.layer_sampler
    if (
        type(state_dim) is not int
        or state_dim <= 0
        or type(num_modules) is not int
        or num_modules <= 0
        or type(layer_sampler) is not LayerSamplerConfig
    ):
        message = "cognitive_config contains invalid arm-B teacher topology"
        raise ValueError(message)
    metadata = _read_backbone_metadata(backbone)
    LayerSampler.resolve_layer_indices(
        layer_sampler.layer_fractions,
        metadata.num_hidden_layers,
    )
    if (
        metadata.max_prompt_tokens + num_modules + metadata.max_target_tokens
        > metadata.max_position_embeddings
    ):
        message = "Prompt, module, and target maxima exceed backbone context"
        raise ValueError(message)
    if cognitive_config.model_dump_json() != snapshot:
        message = "Backbone metadata access mutated cognitive_config"
        raise ValueError(message)
    return metadata, snapshot, state_dim, num_modules, layer_sampler


def build_no_workspace_teacher_forcing_model(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
) -> IrisNoWorkspaceTeacherForcingModel:
    """Compose the exact train-mode arm-B sidecar around one frozen backbone."""
    metadata, snapshot, state_dim, num_modules, layer_sampler_config = _stage(
        "preflight validation",
        lambda: _validate_preflight(
            backbone=backbone,
            cognitive_config=cognitive_config,
        ),
    )
    with torch.inference_mode(False), torch.device("cpu"):
        layer_sampler = _stage(
            "LayerSampler construction and placement",
            lambda: _prepare_module(
                "layer_sampler",
                LayerSampler(
                    layer_sampler_config,
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
        cognitive_modules = _stage(
            "CognitiveModule construction and placement",
            lambda: _prepare_module(
                "cognitive_modules",
                CognitiveModule(state_dim=state_dim, num_modules=num_modules),
                metadata=metadata,
            ),
        )
        module_latent_bridge = _stage(
            "ModuleLatentBridge construction and placement",
            lambda: _prepare_module(
                "module_latent_bridge",
                ModuleLatentBridge(
                    state_dim=state_dim,
                    backbone_hidden_size=metadata.hidden_size,
                    num_modules=num_modules,
                ),
                metadata=metadata,
            ),
        )
        sequence_assembler = _stage(
            "module teacher sequence construction and placement",
            lambda: _prepare_module(
                "sequence_assembler",
                ModuleSecondPassSequenceAssembler(
                    hidden_size=metadata.hidden_size,
                    num_modules=num_modules,
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
        ("cognitive_modules", cognitive_modules),
        ("module_latent_bridge", module_latent_bridge),
        ("sequence_assembler", sequence_assembler),
    )
    initial_state_sha256 = _stage(
        "pre-wiring state ownership",
        lambda: module_state_sha256(components),
    )
    with torch.inference_mode(False), torch.device("cpu"):
        model = _stage(
            "arm-B teacher model wiring",
            lambda: IrisNoWorkspaceTeacherForcingModel(
                backbone=backbone,
                cognitive_config=cognitive_config,
                layer_sampler=layer_sampler,
                goal_encoder=goal_encoder,
                module_input_router=module_input_router,
                cognitive_modules=cognitive_modules,
                module_latent_bridge=module_latent_bridge,
                sequence_assembler=sequence_assembler,
            ),
        )
    if cognitive_config.model_dump_json() != snapshot:
        message = "cognitive_config changed during arm-B teacher composition"
        raise IrisTeacherCompositionError(message)
    if _read_backbone_metadata(backbone) != metadata:
        message = "Backbone metadata changed during arm-B teacher composition"
        raise IrisTeacherCompositionError(message)
    if tuple(model._modules) != tuple(name for name, _ in components):
        message = "Arm-B teacher model does not own the canonical components"
        raise IrisTeacherCompositionError(message)
    if any(getattr(model, name) is not component for name, component in components):
        message = "Arm-B teacher model changed component identity"
        raise IrisTeacherCompositionError(message)
    for name, component in components:
        _validate_module(name, component, metadata=metadata)
    if module_state_sha256(components) != initial_state_sha256:
        message = "Arm-B teacher component state changed during wiring"
        raise IrisTeacherCompositionError(message)
    if (
        model.cycles != NO_WORKSPACE_COGNITIVE_CYCLES_V1
        or tuple(model._modules) != tuple(name for name, _ in components)
        or module_state_sha256(components) != initial_state_sha256
    ):
        message = "Arm-B teacher component ownership is inconsistent"
        raise IrisTeacherCompositionError(message)
    return model


__all__ = ["build_no_workspace_teacher_forcing_model"]
