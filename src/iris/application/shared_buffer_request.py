"""Arm-C request execution with an append-only all-proposals shared buffer."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

import torch
from torch import nn

from iris.application.errors import IrisCompositionError, IrisSharedBufferRequestError
from iris.application.request import (
    IrisRequestExecutionReceiptV1,
    _capture_request_execution_receipt_v1,
)
from iris.application.request_sidecar_state import (
    IrisRequestSidecarInstallResultV1,
    install_request_sidecar_state_v1,
)
from iris.backbone import BackboneAdapter, BackboneFirstPassOutput
from iris.cognitive import (
    CognitiveModule,
    CognitiveModuleOutput,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import (
    DecodingStrategy,
    IrisGenerator,
    IrisGeneratorOutput,
    SecondPassGenerationPrefixAssembler,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig, LayerSamplerConfig
from iris.shared_buffer import (
    IrisSharedBufferPolicyV1,
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferState,
    SharedBufferWriter,
    build_shared_buffer_policy_v1,
    shared_buffer_policy_sha256_v1,
)
from iris.shared_buffer.state import validate_shared_buffer_state
from iris.utils.hashing import module_state_sha256
from iris.workspace import WorkspaceReaderOutput, WorkspaceState

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_new_tokens: int
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _PromptProvenance:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    expected_input_ids: torch.Tensor
    expected_attention_mask: torch.Tensor
    expected_versions: tuple[int, int]
    expected_storage: tuple[tuple[torch.device, int], tuple[torch.device, int]]


@dataclass(frozen=True, slots=True)
class IrisSharedBufferRequestExecutionOutputV1:
    """One arm-C generation plus native buffer and reader provenance."""

    generation: IrisGeneratorOutput
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    receipt: IrisRequestExecutionReceiptV1
    shared_buffer_trace: tuple[SharedBufferState, SharedBufferState, SharedBufferState]
    pre_update_reads: tuple[
        WorkspaceReaderOutput,
        WorkspaceReaderOutput,
        WorkspaceReaderOutput,
    ]
    final_output_read: WorkspaceReaderOutput


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        message = f"Backbone {name} must be a positive integer"
        raise ValueError(message)
    return value


def _read_backbone_metadata(backbone: BackboneAdapter) -> _BackboneMetadata:
    metadata = _BackboneMetadata(
        hidden_size=_positive_integer("hidden_size", backbone.hidden_size),
        num_hidden_layers=_positive_integer("num_hidden_layers", backbone.num_hidden_layers),
        vocabulary_size=_positive_integer("vocabulary_size", backbone.vocabulary_size),
        max_position_embeddings=_positive_integer(
            "max_position_embeddings",
            backbone.max_position_embeddings,
        ),
        max_prompt_tokens=_positive_integer("max_prompt_tokens", backbone.max_prompt_tokens),
        max_new_tokens=_positive_integer("max_new_tokens", backbone.max_new_tokens),
        device=backbone.device,
        dtype=backbone.dtype,
    )
    if not isinstance(metadata.device, torch.device) or metadata.device.type not in {
        "cpu",
        "cuda",
    }:
        message = "Backbone device must be an explicit CPU or CUDA torch.device"
        raise ValueError(message)
    if metadata.device.index is not None and metadata.device.index < 0:
        message = "Backbone device index cannot be negative"
        raise ValueError(message)
    if metadata.device.type == "cuda" and metadata.device.index is None:
        message = "Backbone CUDA device must have one concrete index"
        raise ValueError(message)
    if metadata.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        message = "Backbone dtype must be fp16, bf16, or fp32"
        raise ValueError(message)
    if metadata.device.type == "cpu" and metadata.dtype != torch.float32:
        message = "Iris-GW-0.1 requires fp32 shared-buffer execution on CPU"
        raise ValueError(message)
    if backbone.trainable_parameter_count != 0 or backbone.has_parameter_gradients:
        message = "Arm-C backbone must remain frozen and gradient-free"
        raise ValueError(message)
    return metadata


def _validate_module(name: str, module: nn.Module, *, metadata: _BackboneMetadata) -> None:
    if any(child.training for child in module.modules()):
        message = f"{name} and all children must be in evaluation mode"
        raise ValueError(message)
    for parameter_name, parameter in module.named_parameters():
        if (
            parameter.is_inference()
            or not parameter.requires_grad
            or parameter.device != metadata.device
            or (torch.is_floating_point(parameter) and parameter.dtype != metadata.dtype)
            or (torch.is_floating_point(parameter) and not bool(torch.isfinite(parameter).all()))
        ):
            message = f"{name} parameter {parameter_name!r} is invalid"
            raise ValueError(message)
    for buffer_name, buffer in module.named_buffers():
        if (
            buffer.is_inference()
            or buffer.device != metadata.device
            or (torch.is_floating_point(buffer) and buffer.dtype != metadata.dtype)
            or (torch.is_floating_point(buffer) and not bool(torch.isfinite(buffer).all()))
        ):
            message = f"{name} buffer {buffer_name!r} is invalid"
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


class CognitiveSharedBufferRequestOrchestrator:
    """Execute arm C with prior-cycle broadcast and append-all writes."""

    arm_id = "C"
    arm_kind = "llm_modules_unlimited_shared_buffer"
    request_graph = "append_only_all_proposals_prior_cycle_shared_buffer_v1"

    def __init__(
        self,
        *,
        backbone: BackboneAdapter,
        cognitive_config: CognitiveConfig,
        policy: IrisSharedBufferPolicyV1,
        layer_sampler: LayerSampler,
        goal_encoder: GoalEncoder,
        module_input_router: ModuleInputRouter,
        shared_buffer_reader: SharedBufferReader,
        cognitive_modules: CognitiveModule,
        proposal_head: ProposalHead,
        shared_buffer_writer: SharedBufferWriter,
        shared_buffer_latent_bridge: SharedBufferLatentBridge,
        prefix_assembler: SecondPassGenerationPrefixAssembler,
        iris_generator: IrisGenerator,
    ) -> None:
        if not isinstance(backbone, BackboneAdapter):
            message = "backbone must be an already-loaded BackboneAdapter"
            raise TypeError(message)
        if type(cognitive_config) is not CognitiveConfig:
            message = "cognitive_config must be exactly CognitiveConfig"
            raise TypeError(message)
        if type(policy) is not IrisSharedBufferPolicyV1:
            message = "policy must be exactly IrisSharedBufferPolicyV1"
            raise TypeError(message)
        components: tuple[tuple[str, nn.Module, type[nn.Module]], ...] = (
            ("layer_sampler", layer_sampler, LayerSampler),
            ("goal_encoder", goal_encoder, GoalEncoder),
            ("module_input_router", module_input_router, ModuleInputRouter),
            ("shared_buffer_reader", shared_buffer_reader, SharedBufferReader),
            ("cognitive_modules", cognitive_modules, CognitiveModule),
            ("proposal_head", proposal_head, ProposalHead),
            ("shared_buffer_writer", shared_buffer_writer, SharedBufferWriter),
            (
                "shared_buffer_latent_bridge",
                shared_buffer_latent_bridge,
                SharedBufferLatentBridge,
            ),
            ("prefix_assembler", prefix_assembler, SecondPassGenerationPrefixAssembler),
        )
        for name, component, expected_type in components:
            if not isinstance(component, expected_type):
                message = f"{name} must be {expected_type.__name__}"
                raise TypeError(message)
        if not isinstance(iris_generator, IrisGenerator):
            message = "iris_generator must be IrisGenerator"
            raise TypeError(message)

        self._backbone = backbone
        self._cognitive_config = cognitive_config
        self._cognitive_config_snapshot = cognitive_config.model_dump_json()
        self._policy = policy
        self._policy_snapshot = policy.model_dump_json()
        self._policy_sha256 = shared_buffer_policy_sha256_v1()
        self._metadata = _read_backbone_metadata(backbone)
        self._state_dim = cognitive_config.state_dim
        self._num_modules = cognitive_config.num_modules
        self._layer_fractions = cognitive_config.layer_sampler.layer_fractions
        self._cycles = policy.cognitive_cycles
        self._max_entries = self._num_modules * self._cycles
        self._layer_sampler = layer_sampler
        self._goal_encoder = goal_encoder
        self._module_input_router = module_input_router
        self._shared_buffer_reader = shared_buffer_reader
        self._cognitive_modules = cognitive_modules
        self._proposal_head = proposal_head
        self._shared_buffer_writer = shared_buffer_writer
        self._shared_buffer_latent_bridge = shared_buffer_latent_bridge
        self._prefix_assembler = prefix_assembler
        self._iris_generator = iris_generator
        self._sidecar_modules = tuple((name, component) for name, component, _ in components)
        self._validate_wiring()
        self._sidecar_state_sha256 = module_state_sha256(self._sidecar_modules)

    @property
    def backbone(self) -> BackboneAdapter:
        return self._backbone

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def shared_buffer_policy_sha256(self) -> str:
        self._validate_runtime_boundary()
        return self._policy_sha256

    @property
    def sidecar_state_sha256(self) -> str:
        try:
            self._validate_runtime_boundary()
            current = module_state_sha256(self._sidecar_modules)
        except Exception as error:
            message = "Failed to verify arm-C sidecar state provenance"
            raise IrisSharedBufferRequestError(message) from error
        if current != self._sidecar_state_sha256:
            message = "Arm-C sidecar state changed after composition"
            raise IrisSharedBufferRequestError(message)
        return current

    def restore_teacher_sidecar_state_v1(
        self,
        state: dict[str, torch.Tensor],
        *,
        expected_installed_sha256: str | None = None,
    ) -> IrisRequestSidecarInstallResultV1:
        """Transactionally replace C request tensors from one validated checkpoint."""
        self._validate_runtime_boundary()
        result = install_request_sidecar_state_v1(
            modules=self._sidecar_modules,
            state=state,
            expected_current_sha256=self._sidecar_state_sha256,
            expected_installed_sha256=expected_installed_sha256,
        )
        self._sidecar_state_sha256 = result.installed_state_sha256
        return result

    def generate(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        return self.execute(prompt, strategy=strategy, generator=generator).generation

    @torch.inference_mode()
    def execute(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisSharedBufferRequestExecutionOutputV1:
        """Run exactly three recurrent cycles and one output-only final read."""
        self._validate_request(prompt, strategy=strategy, generator=generator)
        self._validate_runtime_boundary()
        first_pass = self._run_stage(
            "frozen first pass",
            lambda: self._backbone.forward_prompt(prompt),
        )
        if type(first_pass) is not BackboneFirstPassOutput:
            message = "Backbone did not return exact first-pass prompt provenance"
            raise IrisSharedBufferRequestError(message)
        provenance = self._capture_prompt_provenance(first_pass)
        sampler_input = BackboneFirstPassOutput(
            logits=first_pass.logits,
            hidden_states=first_pass.hidden_states,
            attention_mask=provenance.attention_mask,
            input_ids=provenance.input_ids,
        )
        sampled = self._run_stage(
            "backbone layer sampling",
            lambda: self._layer_sampler(sampler_input),
            provenance=provenance,
        )
        goal = self._run_stage(
            "request goal encoding",
            lambda: self._goal_encoder(sampled.context_features, sampled.attention_mask),
            provenance=provenance,
        )
        routed = self._run_stage(
            "module input routing",
            lambda: self._module_input_router(
                sampled.context_features,
                sampled.attention_mask,
            ),
            provenance=provenance,
        )
        batch_size, state_dtype, state_device = self._context_metadata(sampled.context_features)
        module_states = self._run_stage(
            "cognitive state initialization",
            lambda: self._cognitive_modules.initial_state(
                batch_size=batch_size,
                dtype=state_dtype,
                device=state_device,
            ),
            provenance=provenance,
        )
        shared_buffer = self._run_stage(
            "shared-buffer initialization",
            lambda: SharedBufferState.empty(
                batch_size=batch_size,
                state_dim=self._state_dim,
                num_modules=self._num_modules,
                cycles=self._cycles,
                dtype=state_dtype,
                device=state_device,
            ),
            provenance=provenance,
        )

        trace_states: list[SharedBufferState] = []
        read_states: list[WorkspaceReaderOutput] = []
        for cycle_number in range(1, self._cycles + 1):
            reads = self._run_stage(
                f"shared-buffer pre-update read for cycle {cycle_number}",
                partial(self._shared_buffer_reader, shared_buffer),
                provenance=provenance,
            )
            read_states.append(reads)
            module_output = self._run_stage(
                f"cognitive modules for cycle {cycle_number}",
                partial(
                    self._cognitive_modules,
                    routed.module_inputs,
                    reads.workspace_reads,
                    goal.goal_state,
                    previous_states=module_states,
                ),
                provenance=provenance,
            )
            module_states = self._run_stage(
                f"local state extraction for cycle {cycle_number}",
                partial(self._extract_module_states, module_output),
                provenance=provenance,
            )
            proposal = self._run_stage(
                f"proposal head for cycle {cycle_number}",
                partial(self._proposal_head, module_states),
                provenance=provenance,
            )
            shared_buffer = self._run_stage(
                f"append-all shared-buffer write for cycle {cycle_number}",
                partial(
                    self._shared_buffer_writer,
                    proposal,
                    shared_buffer,
                    goal.goal_state,
                    cycle_number=cycle_number,
                ),
                provenance=provenance,
            )
            trace_states.append(shared_buffer)

        if len(trace_states) != 3 or len(read_states) != 3:
            message = "Arm-C execution did not preserve exactly three cognitive cycles"
            raise IrisSharedBufferRequestError(message)
        native_trace = (trace_states[0], trace_states[1], trace_states[2])
        pre_update_reads = (read_states[0], read_states[1], read_states[2])
        final_output_read = self._run_stage(
            "output-only final all-buffer read",
            partial(self._shared_buffer_reader, native_trace[-1]),
            provenance=provenance,
        )
        self._validate_native_trace(
            native_trace,
            pre_update_reads=pre_update_reads,
            final_output_read=final_output_read,
            dtype=state_dtype,
            device=state_device,
        )
        public_trace = self._run_stage(
            "public shared-buffer trace projection",
            lambda: tuple(state.to_workspace_state() for state in native_trace),
            provenance=provenance,
        )
        prompt_embeddings = self._run_stage(
            "exact prompt token embedding",
            lambda: self._backbone.embed_tokens(provenance.input_ids),
            provenance=provenance,
        )
        latent = self._run_stage(
            "token-matched final-read latent bridge",
            lambda: self._shared_buffer_latent_bridge(
                final_output_read,
                provenance.attention_mask,
            ),
            provenance=provenance,
        )
        prefix = self._run_stage(
            "shared-buffer generation prefix",
            lambda: self._prefix_assembler(
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=provenance.attention_mask,
                workspace=latent,
            ),
            provenance=provenance,
        )
        try:
            generation = self._iris_generator.generate(
                prefix,
                strategy=strategy,
                workspace_trace=public_trace,
                generator=generator,
            )
        except Exception as error:
            message = "Arm-C text generation failed"
            raise IrisSharedBufferRequestError(message) from error
        self._validate_prompt_provenance(provenance)
        self._validate_runtime_boundary()
        if (
            type(generation) is not IrisGeneratorOutput
            or generation.workspace_trace is not public_trace
        ):
            message = "Arm-C generation did not preserve the exact projected trace"
            raise IrisSharedBufferRequestError(message)
        self._validate_public_trace(public_trace, native_trace=native_trace)
        return IrisSharedBufferRequestExecutionOutputV1(
            generation=generation,
            prompt_input_ids=provenance.input_ids,
            prompt_attention_mask=provenance.attention_mask,
            receipt=_capture_request_execution_receipt_v1(
                prompt=prompt,
                strategy=strategy,
                generator=generator,
                iris_generator=self._iris_generator,
            ),
            shared_buffer_trace=native_trace,
            pre_update_reads=pre_update_reads,
            final_output_read=final_output_read,
        )

    def _validate_native_trace(
        self,
        trace: tuple[SharedBufferState, SharedBufferState, SharedBufferState],
        *,
        pre_update_reads: tuple[
            WorkspaceReaderOutput,
            WorkspaceReaderOutput,
            WorkspaceReaderOutput,
        ],
        final_output_read: WorkspaceReaderOutput,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        for index, state in enumerate(trace):
            contract = validate_shared_buffer_state(
                state,
                state_dim=self._state_dim,
                num_modules=self._num_modules,
                cycles=self._cycles,
                error_type=IrisSharedBufferRequestError,
                component=f"Arm-C trace cycle {index + 1}",
            )
            if (
                contract.batch_size != 1
                or contract.entry_count != (index + 1) * self._num_modules
                or contract.dtype != dtype
                or contract.device != device
            ):
                message = "Arm-C native trace cardinality or placement is invalid"
                raise IrisSharedBufferRequestError(message)
        expected_read_widths = (0, self._num_modules, 2 * self._num_modules)
        for read, width in zip(pre_update_reads, expected_read_widths, strict=True):
            self._validate_read_snapshot(
                read,
                entry_count=width,
                dtype=dtype,
                device=device,
            )
        self._validate_read_snapshot(
            final_output_read,
            entry_count=self._max_entries,
            dtype=dtype,
            device=device,
        )

    def _validate_read_snapshot(
        self,
        read: WorkspaceReaderOutput,
        *,
        entry_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if type(read) is not WorkspaceReaderOutput:
            message = "Arm-C trace requires exact reader outputs"
            raise IrisSharedBufferRequestError(message)
        if (
            type(read.workspace_reads) is not torch.Tensor
            or tuple(read.workspace_reads.shape) != (1, self._num_modules, self._state_dim)
            or read.workspace_reads.dtype != dtype
            or read.workspace_reads.device != device
            or not bool(torch.isfinite(read.workspace_reads).all())
        ):
            message = "Arm-C reader snapshot values are invalid"
            raise IrisSharedBufferRequestError(message)
        expected_attention_dtype = (
            torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        )
        attention = read.attention_weights
        if (
            type(attention) is not torch.Tensor
            or tuple(attention.shape) != (1, self._num_modules, entry_count)
            or attention.dtype != expected_attention_dtype
            or attention.device != device
            or not bool(torch.isfinite(attention).all())
        ):
            message = "Arm-C reader snapshot attention is invalid"
            raise IrisSharedBufferRequestError(message)
        if entry_count == 0 and not bool((read.workspace_reads == 0).all()):
            message = "Arm-C first cognitive cycle must receive exact zero shared reads"
            raise IrisSharedBufferRequestError(message)

    def _validate_public_trace(
        self,
        public_trace: tuple[WorkspaceState, ...],
        *,
        native_trace: tuple[SharedBufferState, SharedBufferState, SharedBufferState],
    ) -> None:
        if type(public_trace) is not tuple or len(public_trace) != self._cycles:
            message = "Arm-C public trace must contain one projection per cycle"
            raise IrisSharedBufferRequestError(message)
        for projected, native in zip(public_trace, native_trace, strict=True):
            if (
                type(projected) is not WorkspaceState
                or not torch.equal(projected.slots, native.entries)
                or not torch.equal(projected.source_module, native.source_module)
                or not bool(projected.occupancy.all())
                or not bool((projected.confidence == 1).all())
                or not bool((projected.ttl == 1).all())
                or not bool((projected.write_count == 1).all())
            ):
                message = "Arm-C public trace is not a faithful labelled projection"
                raise IrisSharedBufferRequestError(message)

    def _validate_wiring(self) -> None:
        if self._cognitive_config.model_dump_json() != self._cognitive_config_snapshot:
            message = "Arm-C cognitive configuration changed after composition"
            raise ValueError(message)
        if (
            self._policy.model_dump_json() != self._policy_snapshot
            or self._policy != build_shared_buffer_policy_v1()
            or self._policy_sha256 != shared_buffer_policy_sha256_v1()
        ):
            message = "Arm-C shared-buffer policy changed after composition"
            raise ValueError(message)
        metadata = _read_backbone_metadata(self._backbone)
        if metadata != self._metadata:
            message = "Arm-C backbone metadata changed after composition"
            raise ValueError(message)
        expected_components: tuple[tuple[str, object, dict[str, object]], ...] = (
            (
                "layer_sampler",
                self._layer_sampler,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": metadata.hidden_size,
                    "layer_fractions": self._layer_fractions,
                },
            ),
            ("goal_encoder", self._goal_encoder, {"state_dim": self._state_dim}),
            (
                "module_input_router",
                self._module_input_router,
                {"state_dim": self._state_dim, "num_modules": self._num_modules},
            ),
            (
                "shared_buffer_reader",
                self._shared_buffer_reader,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                    "max_entries": self._max_entries,
                },
            ),
            (
                "cognitive_modules",
                self._cognitive_modules,
                {"state_dim": self._state_dim, "num_modules": self._num_modules},
            ),
            (
                "proposal_head",
                self._proposal_head,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                },
            ),
            (
                "shared_buffer_writer",
                self._shared_buffer_writer,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                    "max_entries": self._max_entries,
                },
            ),
            (
                "shared_buffer_latent_bridge",
                self._shared_buffer_latent_bridge,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": metadata.hidden_size,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                    "final_entry_count": self._max_entries,
                },
            ),
            (
                "prefix_assembler",
                self._prefix_assembler,
                {
                    "hidden_size": metadata.hidden_size,
                    "num_slots": self._num_modules,
                    "max_prompt_tokens": metadata.max_prompt_tokens,
                    "max_new_tokens": metadata.max_new_tokens,
                    "max_context_tokens": metadata.max_position_embeddings,
                },
            ),
        )
        if tuple(name for name, _ in self._sidecar_modules) != tuple(
            name for name, _, _ in expected_components
        ):
            message = "Arm-C sidecar components do not follow canonical order"
            raise ValueError(message)
        if self._iris_generator.backbone is not self._backbone:
            message = "Arm-C generator must own the exact request backbone"
            raise ValueError(message)
        if (
            type(self._iris_generator.stop_token_ids) is not tuple
            or any(
                type(token_id) is not int or token_id < 0 or token_id >= metadata.vocabulary_size
                for token_id in self._iris_generator.stop_token_ids
            )
            or len(set(self._iris_generator.stop_token_ids))
            != len(self._iris_generator.stop_token_ids)
            or type(self._iris_generator.sampling) is not TemperatureTopPSamplingConfig
        ):
            message = "Arm-C generator decoding policy is invalid"
            raise ValueError(message)
        for component_name, component, expected_attributes in expected_components:
            for attribute, expected in expected_attributes.items():
                actual = getattr(component, attribute, None)
                if not self._same_typed_value(actual, expected):
                    message = (
                        f"{component_name}.{attribute}={actual!r} does not match "
                        f"arm-C configuration {expected!r}"
                    )
                    raise ValueError(message)

    @classmethod
    def _same_typed_value(cls, actual: object, expected: object) -> bool:
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, tuple):
            if not isinstance(actual, tuple) or len(actual) != len(expected):
                return False
            return all(
                cls._same_typed_value(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        return bool(actual == expected)

    @staticmethod
    def _validate_request(
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None,
    ) -> None:
        if type(prompt) is not str or not prompt.strip():
            message = "prompt must contain non-whitespace text"
            raise IrisSharedBufferRequestError(message)
        try:
            prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            message = "prompt must be valid scalar Unicode encodable as UTF-8"
            raise IrisSharedBufferRequestError(message) from error
        if type(strategy) is not str or strategy not in {"greedy", "sampling"}:
            message = "strategy must be either 'greedy' or 'sampling'"
            raise IrisSharedBufferRequestError(message)
        if strategy == "greedy" and generator is not None:
            message = "greedy generation does not accept a random generator"
            raise IrisSharedBufferRequestError(message)
        if strategy == "sampling" and not isinstance(generator, torch.Generator):
            message = "sampling generation requires an explicit torch.Generator"
            raise IrisSharedBufferRequestError(message)

    def _capture_prompt_provenance(
        self,
        first_pass: BackboneFirstPassOutput,
    ) -> _PromptProvenance:
        try:
            expected_input_ids = first_pass.input_ids.detach().clone()
            expected_attention_mask = first_pass.attention_mask.detach().clone()
            with torch.inference_mode(False):
                input_ids = expected_input_ids.detach().clone()
                attention_mask = expected_attention_mask.detach().clone()
            provenance = _PromptProvenance(
                input_ids=input_ids,
                attention_mask=attention_mask,
                expected_input_ids=expected_input_ids,
                expected_attention_mask=expected_attention_mask,
                expected_versions=(input_ids._version, attention_mask._version),
                expected_storage=(
                    (input_ids.device, input_ids.untyped_storage().data_ptr()),
                    (attention_mask.device, attention_mask.untyped_storage().data_ptr()),
                ),
            )
            storage_identities = {
                (tensor.device, tensor.untyped_storage().data_ptr())
                for tensor in (
                    first_pass.input_ids,
                    first_pass.attention_mask,
                    expected_input_ids,
                    expected_attention_mask,
                    input_ids,
                    attention_mask,
                )
            }
        except Exception as error:
            message = "Failed to capture exact arm-C prompt provenance"
            raise IrisSharedBufferRequestError(message) from error
        if len(storage_identities) != 6:
            message = "Arm-C prompt provenance tensors must use independent storage"
            raise IrisSharedBufferRequestError(message)
        self._validate_prompt_provenance(provenance)
        return provenance

    @staticmethod
    def _validate_prompt_provenance(provenance: _PromptProvenance) -> None:
        for name, actual, expected, expected_version, expected_storage in (
            (
                "input_ids",
                provenance.input_ids,
                provenance.expected_input_ids,
                provenance.expected_versions[0],
                provenance.expected_storage[0],
            ),
            (
                "attention_mask",
                provenance.attention_mask,
                provenance.expected_attention_mask,
                provenance.expected_versions[1],
                provenance.expected_storage[1],
            ),
        ):
            try:
                unchanged = torch.equal(actual, expected)
                actual_storage = (actual.device, actual.untyped_storage().data_ptr())
            except Exception as error:
                message = f"Failed to verify arm-C prompt {name} provenance"
                raise IrisSharedBufferRequestError(message) from error
            if (
                type(actual) is not torch.Tensor
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.device != expected.device
                or actual.is_inference()
                or actual.requires_grad
                or actual.grad_fn is not None
                or actual._version != expected_version
                or actual_storage != expected_storage
                or not unchanged
            ):
                message = f"Arm-C stage mutated exact first-pass {name}"
                raise IrisSharedBufferRequestError(message)

    def _context_metadata(
        self,
        context_features: torch.Tensor,
    ) -> tuple[int, torch.dtype, torch.device]:
        if (
            type(context_features) is not torch.Tensor
            or context_features.ndim != 3
            or context_features.shape[0] != 1
            or context_features.shape[1] <= 0
            or context_features.shape[2] != self._state_dim
            or not torch.is_floating_point(context_features)
            or not bool(torch.isfinite(context_features).all())
        ):
            message = "Layer sampler returned invalid arm-C context features"
            raise IrisSharedBufferRequestError(message)
        return context_features.shape[0], context_features.dtype, context_features.device

    @staticmethod
    def _extract_module_states(output: CognitiveModuleOutput) -> torch.Tensor:
        if (
            type(output) is not CognitiveModuleOutput
            or type(output.module_states) is not torch.Tensor
        ):
            message = "Cognitive modules returned invalid arm-C local state"
            raise IrisSharedBufferRequestError(message)
        return output.module_states

    def _validate_runtime_boundary(self) -> None:
        try:
            self._validate_wiring()
            for component_name, component in self._sidecar_modules:
                if any(child.training for child in component.modules()):
                    message = f"{component_name} must remain in evaluation mode"
                    raise ValueError(message)
        except Exception as error:
            message = "Arm-C request runtime boundary is not ready"
            raise IrisSharedBufferRequestError(message) from error

    def _run_stage(
        self,
        name: str,
        operation: Callable[[], _ResultT],
        *,
        provenance: _PromptProvenance | None = None,
    ) -> _ResultT:
        try:
            self._validate_runtime_boundary()
            if provenance is not None:
                self._validate_prompt_provenance(provenance)
            result = operation()
            if provenance is not None:
                self._validate_prompt_provenance(provenance)
            self._validate_runtime_boundary()
            return result
        except IrisSharedBufferRequestError:
            raise
        except Exception as error:
            message = f"Arm-C request failed during {name}"
            raise IrisSharedBufferRequestError(message) from error


def build_shared_buffer_request_orchestrator(
    *,
    backbone: BackboneAdapter,
    cognitive_config: CognitiveConfig,
    stop_token_ids: tuple[int, ...],
    sampling: TemperatureTopPSamplingConfig,
) -> CognitiveSharedBufferRequestOrchestrator:
    """Compose the placed eval-only arm-C graph around one frozen backbone."""
    try:
        if not isinstance(backbone, BackboneAdapter):
            message = "backbone must be an already-loaded BackboneAdapter"
            raise TypeError(message)
        if type(cognitive_config) is not CognitiveConfig:
            message = "cognitive_config must be exactly CognitiveConfig"
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
        if type(sampling) is not TemperatureTopPSamplingConfig:
            message = "sampling must be exactly TemperatureTopPSamplingConfig"
            raise TypeError(message)
        cognitive_snapshot = cognitive_config.model_dump_json()
        policy = build_shared_buffer_policy_v1()
        policy_snapshot = policy.model_dump_json()
        state_dim = cognitive_config.state_dim
        num_modules = cognitive_config.num_modules
        cycles = policy.cognitive_cycles
        layer_sampler_config = cognitive_config.layer_sampler
        if (
            type(state_dim) is not int
            or state_dim <= 0
            or type(num_modules) is not int
            or num_modules <= 0
            or type(layer_sampler_config) is not LayerSamplerConfig
        ):
            message = "cognitive_config contains invalid arm-C dimensions"
            raise ValueError(message)
        metadata = _read_backbone_metadata(backbone)
        LayerSampler.resolve_layer_indices(
            layer_sampler_config.layer_fractions,
            metadata.num_hidden_layers,
        )
        if any(token_id < 0 or token_id >= metadata.vocabulary_size for token_id in stop_token_ids):
            message = "stop_token_ids contains an out-of-vocabulary ID"
            raise ValueError(message)
        if (
            metadata.max_prompt_tokens + num_modules + metadata.max_new_tokens
            > metadata.max_position_embeddings
        ):
            message = "Prompt, token-matched buffer latents, and generation maxima exceed context"
            raise ValueError(message)

        with torch.inference_mode(False), torch.device("cpu"):
            layer_sampler = _prepare_module(
                "layer_sampler",
                LayerSampler(
                    layer_sampler_config,
                    backbone_hidden_size=metadata.hidden_size,
                    state_dim=state_dim,
                ),
                metadata=metadata,
            )
            goal_encoder = _prepare_module(
                "goal_encoder",
                GoalEncoder(state_dim=state_dim),
                metadata=metadata,
            )
            module_input_router = _prepare_module(
                "module_input_router",
                ModuleInputRouter(state_dim=state_dim, num_modules=num_modules),
                metadata=metadata,
            )
            shared_buffer_reader = _prepare_module(
                "shared_buffer_reader",
                SharedBufferReader(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    cycles=cycles,
                ),
                metadata=metadata,
            )
            cognitive_modules = _prepare_module(
                "cognitive_modules",
                CognitiveModule(state_dim=state_dim, num_modules=num_modules),
                metadata=metadata,
            )
            proposal_head = _prepare_module(
                "proposal_head",
                ProposalHead(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    cycles=cycles,
                ),
                metadata=metadata,
            )
            shared_buffer_writer = _prepare_module(
                "shared_buffer_writer",
                SharedBufferWriter(
                    state_dim=state_dim,
                    num_modules=num_modules,
                    cycles=cycles,
                ),
                metadata=metadata,
            )
            shared_buffer_latent_bridge = _prepare_module(
                "shared_buffer_latent_bridge",
                SharedBufferLatentBridge(
                    state_dim=state_dim,
                    backbone_hidden_size=metadata.hidden_size,
                    num_modules=num_modules,
                    cycles=cycles,
                ),
                metadata=metadata,
            )
            prefix_assembler = _prepare_module(
                "prefix_assembler",
                SecondPassGenerationPrefixAssembler(
                    hidden_size=metadata.hidden_size,
                    num_slots=num_modules,
                    max_prompt_tokens=metadata.max_prompt_tokens,
                    max_new_tokens=metadata.max_new_tokens,
                    max_context_tokens=metadata.max_position_embeddings,
                ),
                metadata=metadata,
            )
            iris_generator = IrisGenerator(
                backbone,
                max_new_tokens=metadata.max_new_tokens,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
            )
            orchestrator = CognitiveSharedBufferRequestOrchestrator(
                backbone=backbone,
                cognitive_config=cognitive_config,
                policy=policy,
                layer_sampler=layer_sampler,
                goal_encoder=goal_encoder,
                module_input_router=module_input_router,
                shared_buffer_reader=shared_buffer_reader,
                cognitive_modules=cognitive_modules,
                proposal_head=proposal_head,
                shared_buffer_writer=shared_buffer_writer,
                shared_buffer_latent_bridge=shared_buffer_latent_bridge,
                prefix_assembler=prefix_assembler,
                iris_generator=iris_generator,
            )
        if cognitive_config.model_dump_json() != cognitive_snapshot:
            message = "cognitive_config changed during arm-C composition"
            raise ValueError(message)
        if policy.model_dump_json() != policy_snapshot or policy != build_shared_buffer_policy_v1():
            message = "shared-buffer policy changed during arm-C composition"
            raise ValueError(message)
        if _read_backbone_metadata(backbone) != metadata:
            message = "Backbone metadata changed during arm-C composition"
            raise ValueError(message)
        return orchestrator
    except IrisCompositionError:
        raise
    except Exception as error:
        message = "Arm-C shared-buffer request composition did not complete"
        raise IrisCompositionError(message) from error


__all__ = [
    "CognitiveSharedBufferRequestOrchestrator",
    "IrisSharedBufferRequestExecutionOutputV1",
    "build_shared_buffer_request_orchestrator",
]
