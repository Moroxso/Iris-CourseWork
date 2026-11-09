"""Validated composition of the append-only shared-buffer arm-C teacher."""

from collections.abc import Callable
from dataclasses import dataclass

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
from iris.schemas import CognitiveConfig, LayerSamplerConfig
from iris.shared_buffer import (
    IrisSharedBufferPolicyV1,
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferWriter,
    build_shared_buffer_policy_v1,
    shared_buffer_policy_sha256_v1,
)
from iris.training.shared_buffer_model import IrisSharedBufferTeacherForcingModel
from iris.utils.hashing import module_state_sha256

_IRIS_SHARED_BUFFER_TEACHER_TYPE = IrisSharedBufferTeacherForcingModel


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    """One immutable public snapshot of the caller-owned frozen backbone."""

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
    """Canonical values that must remain stable across model construction."""

    backbone: _BackboneMetadata
    cognitive_snapshot: str
    policy: IrisSharedBufferPolicyV1
    policy_snapshot: str
    policy_sha256: str
    state_dim: int
    num_modules: int
    cycles: int
    layer_sampler: LayerSamplerConfig


def _stage[ResultT](name: str, operation: Callable[[], ResultT]) -> ResultT:
    try:
        return operation()
    except IrisTeacherCompositionError:
        raise
    except Exception as error:
        message = f"Arm-C teacher composition failed during {name}"
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
        message = "Arm-C teacher backbone device must be explicit CPU or CUDA"
        raise ValueError(message)
    if device.type == "cpu":
        if device != torch.device("cpu"):
            message = "Arm-C CPU backbone must use the canonical indexless device"
            raise ValueError(message)
    elif device.index is None or device.index < 0:
        message = "Arm-C CUDA backbone must use one concrete nonnegative index"
        raise ValueError(message)
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        message = "Arm-C teacher backbone dtype must be fp16, bf16, or fp32"
        raise ValueError(message)
    if device.type == "cpu" and dtype != torch.float32:
        message = "Arm-C teacher CPU composition requires fp32"
        raise ValueError(message)
    if type(trainable_parameter_count) is not int or trainable_parameter_count != 0:
        message = "Arm-C teacher backbone must remain frozen"
        raise ValueError(message)
    if type(has_parameter_gradients) is not bool or has_parameter_gradients:
        message = "Arm-C teacher backbone must not carry gradients"
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
) -> _CompositionInputs:
    if not isinstance(backbone, BackboneAdapter):
        message = "backbone must be an already-loaded BackboneAdapter"
        raise TypeError(message)
    if type(cognitive_config) is not CognitiveConfig:
        message = "cognitive_config must be exactly CognitiveConfig"
        raise TypeError(message)

    cognitive_snapshot = cognitive_config.model_dump_json()
    policy = build_shared_buffer_policy_v1()
    if type(policy) is not IrisSharedBufferPolicyV1:
        message = "Arm-C policy builder must return exactly IrisSharedBufferPolicyV1"
        raise TypeError(message)
    policy_snapshot = policy.model_dump_json()
    policy_sha256 = shared_buffer_policy_sha256_v1()
    state_dim = cognitive_config.state_dim
    num_modules = cognitive_config.num_modules
    layer_sampler = cognitive_config.layer_sampler
    cycles = policy.cognitive_cycles

    if type(state_dim) is not int or state_dim <= 0:
        message = "Arm-C cognitive state_dim must be a positive integer"
        raise ValueError(message)
    if type(num_modules) is not int or num_modules <= 0:
        message = "Arm-C cognitive num_modules must be a positive integer"
        raise ValueError(message)
    if type(layer_sampler) is not LayerSamplerConfig:
        message = "Arm-C layer_sampler must be exactly LayerSamplerConfig"
        raise TypeError(message)
    if type(cycles) is not int or cycles <= 0:
        message = "Arm-C policy cycles must be a positive integer"
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
        message = "Prompt, M shared-buffer latents, and target maxima exceed context"
        raise ValueError(message)
    if cognitive_config.model_dump_json() != cognitive_snapshot:
        message = "Backbone metadata access mutated cognitive_config"
        raise ValueError(message)
    if (
        policy.model_dump_json() != policy_snapshot
        or policy != build_shared_buffer_policy_v1()
        or policy_sha256 != shared_buffer_policy_sha256_v1()
    ):
        message = "Canonical arm-C shared-buffer policy is unstable"
        raise ValueError(message)

    return _CompositionInputs(
        backbone=metadata,
        cognitive_snapshot=cognitive_snapshot,
        policy=policy,
        policy_snapshot=policy_snapshot,
        policy_sha256=policy_sha256,
        state_dim=state_dim,
        num_modules=num_modules,
        cycles=cycles,
        layer_sampler=layer_sampler,
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
            or (torch.is_floating_point(parameter) and not bool(torch.isfinite(parameter).all()))
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


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            return False
        return all(
            _same_typed_value(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _validate_component_contracts(
    components: tuple[tuple[str, nn.Module], ...],
    *,
    inputs: _CompositionInputs,
) -> None:
    metadata = inputs.backbone
    max_entries = inputs.num_modules * inputs.cycles
    expected: tuple[tuple[str, type[nn.Module], dict[str, object]], ...] = (
        (
            "layer_sampler",
            LayerSampler,
            {
                "backbone_hidden_size": metadata.hidden_size,
                "state_dim": inputs.state_dim,
                "layer_fractions": inputs.layer_sampler.layer_fractions,
            },
        ),
        ("goal_encoder", GoalEncoder, {"state_dim": inputs.state_dim}),
        (
            "module_input_router",
            ModuleInputRouter,
            {"state_dim": inputs.state_dim, "num_modules": inputs.num_modules},
        ),
        (
            "shared_buffer_reader",
            SharedBufferReader,
            {
                "state_dim": inputs.state_dim,
                "num_modules": inputs.num_modules,
                "cycles": inputs.cycles,
                "max_entries": max_entries,
            },
        ),
        (
            "cognitive_modules",
            CognitiveModule,
            {"state_dim": inputs.state_dim, "num_modules": inputs.num_modules},
        ),
        (
            "proposal_head",
            ProposalHead,
            {
                "state_dim": inputs.state_dim,
                "num_modules": inputs.num_modules,
                "cycles": inputs.cycles,
            },
        ),
        (
            "shared_buffer_writer",
            SharedBufferWriter,
            {
                "state_dim": inputs.state_dim,
                "num_modules": inputs.num_modules,
                "cycles": inputs.cycles,
                "max_entries": max_entries,
            },
        ),
        (
            "shared_buffer_latent_bridge",
            SharedBufferLatentBridge,
            {
                "state_dim": inputs.state_dim,
                "backbone_hidden_size": metadata.hidden_size,
                "num_modules": inputs.num_modules,
                "cycles": inputs.cycles,
                "final_entry_count": max_entries,
            },
        ),
        (
            "sequence_assembler",
            SecondPassSequenceAssembler,
            {
                "hidden_size": metadata.hidden_size,
                "num_slots": inputs.num_modules,
                "vocabulary_size": metadata.vocabulary_size,
                "max_prompt_tokens": metadata.max_prompt_tokens,
                "max_target_tokens": metadata.max_target_tokens,
                "max_context_tokens": metadata.max_position_embeddings,
            },
        ),
    )
    if tuple(name for name, _ in components) != tuple(name for name, _, _ in expected):
        message = "Arm-C teacher components do not follow the canonical order"
        raise ValueError(message)
    for (name, component), (_, expected_type, attributes) in zip(
        components,
        expected,
        strict=True,
    ):
        if type(component) is not expected_type:
            message = f"{name} must be exactly {expected_type.__name__}"
            raise TypeError(message)
        for attribute, expected_value in attributes.items():
            try:
                actual = getattr(component, attribute)
            except Exception as error:
                message = f"Failed to read {name}.{attribute}"
                raise ValueError(message) from error
            if not _same_typed_value(actual, expected_value):
                message = (
                    f"{name}.{attribute}={actual!r} does not match "
                    f"arm-C teacher configuration {expected_value!r}"
                )
                raise ValueError(message)


def _validate_stable_inputs(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    inputs: _CompositionInputs,
    components: tuple[tuple[str, nn.Module], ...],
    expected_state_sha256: str | None = None,
) -> str:
    if _read_backbone_metadata(backbone) != inputs.backbone:
        message = "Backbone metadata changed during arm-C teacher composition"
        raise ValueError(message)
    if cognitive_config.model_dump_json() != inputs.cognitive_snapshot:
        message = "cognitive_config changed during arm-C teacher composition"
        raise ValueError(message)
    if (
        type(inputs.policy) is not IrisSharedBufferPolicyV1
        or inputs.policy.model_dump_json() != inputs.policy_snapshot
        or inputs.policy != build_shared_buffer_policy_v1()
        or inputs.policy_sha256 != shared_buffer_policy_sha256_v1()
    ):
        message = "Arm-C shared-buffer policy changed during teacher composition"
        raise ValueError(message)
    for name, component in components:
        _validate_module(name, component, metadata=inputs.backbone)
    _validate_component_contracts(components, inputs=inputs)
    state_sha256 = module_state_sha256(components)
    if expected_state_sha256 is not None and state_sha256 != expected_state_sha256:
        message = "Arm-C teacher component state changed during model wiring"
        raise ValueError(message)
    return state_sha256


def _validate_model(
    model: IrisSharedBufferTeacherForcingModel,
    *,
    backbone: BackboneAdapter,
    inputs: _CompositionInputs,
    components: tuple[tuple[str, nn.Module], ...],
) -> None:
    if type(model) is not _IRIS_SHARED_BUFFER_TEACHER_TYPE:
        message = "Arm-C teacher construction returned an invalid model"
        raise TypeError(message)
    if model.backbone is not backbone:
        message = "Arm-C teacher must preserve the caller-owned backbone identity"
        raise ValueError(message)
    if not model.training or any(not child.training for child in model.modules()):
        message = "Arm-C teacher and all registered children must remain in training mode"
        raise ValueError(message)
    if model.cycles != inputs.cycles:
        message = "Arm-C teacher does not own the canonical cycle count"
        raise ValueError(message)
    if model.shared_buffer_policy_sha256 != inputs.policy_sha256:
        message = "Arm-C teacher does not bind the canonical shared-buffer policy"
        raise ValueError(message)
    if tuple(model._modules) != tuple(name for name, _ in components):
        message = "Arm-C teacher registered component order is not canonical"
        raise ValueError(message)
    for name, component in components:
        if getattr(model, name, None) is not component:
            message = f"Arm-C teacher must preserve the exact {name} instance"
            raise ValueError(message)


def build_shared_buffer_teacher_forcing_model(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
) -> IrisSharedBufferTeacherForcingModel:
    """Compose the exact train-mode arm-C sidecar around one frozen backbone.

    The factory owns the canonical shared-buffer policy. The caller owns RNG,
    backbone loading, target preparation, optimizer policy, and checkpointing.
    No tokenizer or backbone forward method is executed during composition.
    """
    inputs = _stage(
        "preflight validation",
        lambda: _validate_preflight(
            backbone=backbone,
            cognitive_config=cognitive_config,
        ),
    )
    metadata = inputs.backbone

    # Keep constructor order equal to the arm-C request graph and defeat an
    # ambient inference-mode context without changing the caller-owned RNG.
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
        shared_buffer_reader = _stage(
            "SharedBufferReader construction and placement",
            lambda: _prepare_module(
                "shared_buffer_reader",
                SharedBufferReader(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    cycles=inputs.cycles,
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
        shared_buffer_writer = _stage(
            "SharedBufferWriter construction and placement",
            lambda: _prepare_module(
                "shared_buffer_writer",
                SharedBufferWriter(
                    state_dim=inputs.state_dim,
                    num_modules=inputs.num_modules,
                    cycles=inputs.cycles,
                ),
                metadata=metadata,
            ),
        )
        shared_buffer_latent_bridge = _stage(
            "SharedBufferLatentBridge construction and placement",
            lambda: _prepare_module(
                "shared_buffer_latent_bridge",
                SharedBufferLatentBridge(
                    state_dim=inputs.state_dim,
                    backbone_hidden_size=metadata.hidden_size,
                    num_modules=inputs.num_modules,
                    cycles=inputs.cycles,
                ),
                metadata=metadata,
            ),
        )
        sequence_assembler = _stage(
            "arm-C teacher sequence construction and placement",
            lambda: _prepare_module(
                "sequence_assembler",
                SecondPassSequenceAssembler(
                    hidden_size=metadata.hidden_size,
                    num_slots=inputs.num_modules,
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
        ("shared_buffer_reader", shared_buffer_reader),
        ("cognitive_modules", cognitive_modules),
        ("proposal_head", proposal_head),
        ("shared_buffer_writer", shared_buffer_writer),
        ("shared_buffer_latent_bridge", shared_buffer_latent_bridge),
        ("sequence_assembler", sequence_assembler),
    )
    state_sha256 = _stage(
        "post-placement validation",
        lambda: _validate_stable_inputs(
            backbone=backbone,
            cognitive_config=cognitive_config,
            inputs=inputs,
            components=components,
        ),
    )
    with torch.inference_mode(False), torch.device("cpu"):
        model = _stage(
            "arm-C teacher model wiring",
            lambda: IrisSharedBufferTeacherForcingModel(
                backbone=backbone,
                cognitive_config=cognitive_config,
                policy=inputs.policy,
                layer_sampler=layer_sampler,
                goal_encoder=goal_encoder,
                module_input_router=module_input_router,
                shared_buffer_reader=shared_buffer_reader,
                cognitive_modules=cognitive_modules,
                proposal_head=proposal_head,
                shared_buffer_writer=shared_buffer_writer,
                shared_buffer_latent_bridge=shared_buffer_latent_bridge,
                sequence_assembler=sequence_assembler,
            ),
        )
    _stage(
        "arm-C teacher model validation",
        lambda: _validate_model(
            model,
            backbone=backbone,
            inputs=inputs,
            components=components,
        ),
    )
    _stage(
        "final composition validation",
        lambda: _validate_stable_inputs(
            backbone=backbone,
            cognitive_config=cognitive_config,
            inputs=inputs,
            components=components,
            expected_state_sha256=state_sha256,
        ),
    )
    return model


__all__ = ["build_shared_buffer_teacher_forcing_model"]
