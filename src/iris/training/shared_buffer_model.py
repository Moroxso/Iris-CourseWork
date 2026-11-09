"""Teacher-forced arm-C sidecar with an append-only shared buffer."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, TypeVar

import torch
from torch import nn

from iris.backbone import BackboneAdapter, BackboneCausalOutput, BackboneFirstPassOutput
from iris.cognitive import (
    CognitiveModule,
    CognitiveModuleOutput,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import SecondPassSequenceAssembler
from iris.schemas import CognitiveConfig
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
from iris.training.outputs import (
    IrisSharedBufferTeacherCheckpointProvenanceV1,
    IrisTeacherForcedOutput,
    IrisTeacherForcingInput,
)
from iris.utils.hashing import module_state_sha256
from iris.workspace import WorkspaceReaderOutput, WorkspaceState

_ResultT = TypeVar("_ResultT")


class IrisSharedBufferTeacherForcingError(RuntimeError):
    """Raised when the exact arm-C teacher boundary cannot complete safely."""


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


@dataclass(frozen=True, slots=True)
class _TokenProvenance:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    expected_input_ids: torch.Tensor
    expected_attention_mask: torch.Tensor
    expected_versions: tuple[int, int]
    expected_storage: tuple[tuple[torch.device, int], tuple[torch.device, int]]


class IrisSharedBufferTeacherForcingModel(nn.Module):
    """Train the exact nine-component arm-C sidecar around a frozen backbone."""

    _CANONICAL_COMPONENT_NAMES = (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "shared_buffer_reader",
        "cognitive_modules",
        "proposal_head",
        "shared_buffer_writer",
        "shared_buffer_latent_bridge",
        "sequence_assembler",
    )
    _TEACHER_GRAPH: Literal[
        "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"
    ] = "append_only_all_proposals_prior_cycle_shared_buffer_teacher_v1"
    _backbone: BackboneAdapter
    _cognitive_config: CognitiveConfig
    _policy: IrisSharedBufferPolicyV1
    _cognitive_config_snapshot: str
    _policy_snapshot: str
    _backbone_metadata: _BackboneMetadata

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
        sequence_assembler: SecondPassSequenceAssembler,
    ) -> None:
        super().__init__()
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
            ("sequence_assembler", sequence_assembler, SecondPassSequenceAssembler),
        )
        for name, component, expected_type in components:
            if not isinstance(component, expected_type):
                message = f"{name} must be {expected_type.__name__}"
                raise TypeError(message)
        if len({id(component) for _, component, _ in components}) != len(components):
            message = "Arm-C teacher components must be distinct module instances"
            raise ValueError(message)
        try:
            cognitive_snapshot = cognitive_config.model_dump_json()
            policy_snapshot = policy.model_dump_json()
            metadata = self._read_backbone_metadata(backbone)
        except Exception as error:
            message = "Failed to snapshot arm-C teacher configuration and backbone"
            raise ValueError(message) from error

        object.__setattr__(self, "_backbone", backbone)
        object.__setattr__(self, "_cognitive_config", cognitive_config)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_cognitive_config_snapshot", cognitive_snapshot)
        object.__setattr__(self, "_policy_snapshot", policy_snapshot)
        object.__setattr__(self, "_backbone_metadata", metadata)
        self._policy_sha256 = shared_buffer_policy_sha256_v1()
        self._state_dim = cognitive_config.state_dim
        self._num_modules = cognitive_config.num_modules
        self._layer_fractions = cognitive_config.layer_sampler.layer_fractions
        self._cycles = policy.cognitive_cycles
        self._max_entries = self._num_modules * self._cycles

        self.layer_sampler = layer_sampler
        self.goal_encoder = goal_encoder
        self.module_input_router = module_input_router
        self.shared_buffer_reader = shared_buffer_reader
        self.cognitive_modules = cognitive_modules
        self.proposal_head = proposal_head
        self.shared_buffer_writer = shared_buffer_writer
        self.shared_buffer_latent_bridge = shared_buffer_latent_bridge
        self.sequence_assembler = sequence_assembler
        self._sidecar_modules = tuple(
            (name, getattr(self, name)) for name in self._CANONICAL_COMPONENT_NAMES
        )

        self.train()
        self._validate_wiring()
        self._validate_mode(expected_training=True)

    @property
    def backbone(self) -> BackboneAdapter:
        return self._backbone

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def shared_buffer_policy_sha256(self) -> str:
        return self._policy_sha256

    def forward(self, example: IrisTeacherForcingInput) -> IrisTeacherForcedOutput:
        return self.training_forward(example)

    def training_forward(self, example: IrisTeacherForcingInput) -> IrisTeacherForcedOutput:
        if torch.is_inference_mode_enabled():
            message = "Arm-C teacher training cannot run under torch.inference_mode"
            raise IrisSharedBufferTeacherForcingError(message)
        if not torch.is_grad_enabled():
            message = "Arm-C teacher training requires autograd"
            raise IrisSharedBufferTeacherForcingError(message)
        self._validate_runtime_boundary(expected_training=True)
        return self._execute(example, expected_training=True)

    def evaluation_forward(self, example: IrisTeacherForcingInput) -> IrisTeacherForcedOutput:
        if torch.is_inference_mode_enabled():
            message = "Arm-C teacher evaluation cannot run under torch.inference_mode"
            raise IrisSharedBufferTeacherForcingError(message)
        self._validate_runtime_boundary(expected_training=False)
        with torch.no_grad():
            return self._execute(example, expected_training=False)

    @property
    def checkpoint_provenance(self) -> IrisSharedBufferTeacherCheckpointProvenanceV1:
        if torch.is_inference_mode_enabled():
            message = "Arm-C checkpoint provenance cannot run under torch.inference_mode"
            raise IrisSharedBufferTeacherForcingError(message)
        self._validate_runtime_boundary(expected_training=True)
        try:
            state_sha256 = module_state_sha256(self._sidecar_modules)
            backbone_configuration_json = self._backbone.configuration_json
        except Exception as error:
            message = "Failed to capture arm-C teacher checkpoint provenance"
            raise IrisSharedBufferTeacherForcingError(message) from error
        metadata = self._backbone_metadata
        return IrisSharedBufferTeacherCheckpointProvenanceV1(
            arm_id="C",
            teacher_graph=self._TEACHER_GRAPH,
            shared_buffer_policy_sha256=self._policy_sha256,
            cognitive_cycles=self._cycles,
            pre_update_read_count=self._policy.pre_update_read_count,
            final_output_read_count=self._policy.final_output_read_count,
            entries_per_cycle=self._num_modules,
            final_entry_count=self._max_entries,
            latent_token_count=self._num_modules,
            component_names=self._CANONICAL_COMPONENT_NAMES,
            backbone_configuration_json=backbone_configuration_json,
            cognitive_configuration_json=self._cognitive_config_snapshot,
            hidden_size=metadata.hidden_size,
            num_hidden_layers=metadata.num_hidden_layers,
            vocabulary_size=metadata.vocabulary_size,
            max_position_embeddings=metadata.max_position_embeddings,
            max_prompt_tokens=metadata.max_prompt_tokens,
            max_target_tokens=metadata.max_target_tokens,
            device=str(metadata.device),
            dtype=str(metadata.dtype).removeprefix("torch."),
            sidecar_state_sha256=state_sha256,
        )

    def _execute(
        self,
        example: IrisTeacherForcingInput,
        *,
        expected_training: bool,
    ) -> IrisTeacherForcedOutput:
        self._validate_example(example)
        target_provenance = self._capture_token_provenance(
            example.target_input_ids,
            example.target_attention_mask,
            segment="target",
        )
        first_pass = self._run_stage(
            "frozen first pass",
            lambda: self._backbone.forward_prompt(example.prompt),
            expected_training=expected_training,
            target_provenance=target_provenance,
        )
        if type(first_pass) is not BackboneFirstPassOutput:
            message = "Backbone did not return exact first-pass prompt provenance"
            raise IrisSharedBufferTeacherForcingError(message)
        prompt_provenance = self._capture_token_provenance(
            first_pass.input_ids,
            first_pass.attention_mask,
            segment="prompt",
        )
        sampler_input = BackboneFirstPassOutput(
            logits=first_pass.logits,
            hidden_states=first_pass.hidden_states,
            attention_mask=prompt_provenance.attention_mask,
            input_ids=prompt_provenance.input_ids,
        )
        sampled = self._run_stage(
            "backbone layer sampling",
            lambda: self.layer_sampler(sampler_input),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        goal = self._run_stage(
            "request goal encoding",
            lambda: self.goal_encoder(sampled.context_features, sampled.attention_mask),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        routed = self._run_stage(
            "module input routing",
            lambda: self.module_input_router(
                sampled.context_features,
                sampled.attention_mask,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        batch_size, state_dtype, state_device = self._context_metadata(
            sampled.context_features
        )
        module_states = self._run_stage(
            "cognitive state initialization",
            lambda: self.cognitive_modules.initial_state(
                batch_size=batch_size,
                dtype=state_dtype,
                device=state_device,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
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
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )

        trace_states: list[SharedBufferState] = []
        pre_reads: list[WorkspaceReaderOutput] = []
        for cycle_number in range(1, self._cycles + 1):
            reads = self._run_stage(
                f"shared-buffer pre-update read for cycle {cycle_number}",
                partial(self.shared_buffer_reader, shared_buffer),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            pre_reads.append(reads)
            module_output = self._run_stage(
                f"cognitive modules for cycle {cycle_number}",
                partial(
                    self.cognitive_modules,
                    routed.module_inputs,
                    reads.workspace_reads,
                    goal.goal_state,
                    previous_states=module_states,
                ),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            module_states = self._run_stage(
                f"local state extraction for cycle {cycle_number}",
                partial(self._extract_module_states, module_output),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            proposal = self._run_stage(
                f"proposal head for cycle {cycle_number}",
                partial(self.proposal_head, module_states),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            shared_buffer = self._run_stage(
                f"append-all shared-buffer write for cycle {cycle_number}",
                partial(
                    self.shared_buffer_writer,
                    proposal,
                    shared_buffer,
                    goal.goal_state,
                    cycle_number=cycle_number,
                ),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            trace_states.append(shared_buffer)

        if len(trace_states) != 3 or len(pre_reads) != 3:
            message = "Arm-C teacher did not preserve exactly three cognitive cycles"
            raise IrisSharedBufferTeacherForcingError(message)
        native_trace = (trace_states[0], trace_states[1], trace_states[2])
        pre_update_reads = (pre_reads[0], pre_reads[1], pre_reads[2])
        final_output_read = self._run_stage(
            "output-only final all-buffer read",
            partial(self.shared_buffer_reader, native_trace[-1]),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        self._validate_native_execution(
            native_trace,
            pre_update_reads=pre_update_reads,
            final_output_read=final_output_read,
            dtype=state_dtype,
            device=state_device,
        )
        prompt_embeddings = self._run_stage(
            "exact prompt token embedding",
            lambda: self._backbone.embed_tokens(prompt_provenance.input_ids),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        target_embeddings = self._run_stage(
            "exact target token embedding",
            lambda: self._backbone.embed_tokens(target_provenance.input_ids),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        latent = self._run_stage(
            "token-matched final-read latent bridge",
            lambda: self.shared_buffer_latent_bridge(
                final_output_read,
                prompt_provenance.attention_mask,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        sequence = self._run_stage(
            "shared-buffer teacher sequence",
            lambda: self.sequence_assembler(
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=prompt_provenance.attention_mask,
                workspace=latent,
                target_input_ids=target_provenance.input_ids,
                target_embeddings=target_embeddings,
                target_attention_mask=target_provenance.attention_mask,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        causal_output = self._run_stage(
            "frozen teacher-forced causal pass",
            lambda: self._backbone.forward_embeddings(
                sequence.inputs_embeds,
                sequence.attention_mask,
                sequence.position_ids,
                labels=sequence.labels,
                past_key_values=None,
                use_cache=False,
                logits_to_keep=0,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        self._validate_causal_output(
            causal_output,
            total_length=sequence.inputs_embeds.shape[1],
            expected_training=expected_training,
        )
        loss = causal_output.loss
        if loss is None:
            message = "Arm-C teacher causal pass did not return a loss"
            raise IrisSharedBufferTeacherForcingError(message)
        if not expected_training:
            loss = loss.detach().clone()
        public_trace = tuple(state.to_workspace_state() for state in native_trace)
        output = IrisTeacherForcedOutput(
            loss=loss,
            supervised_token_count=int(
                target_provenance.attention_mask.to(dtype=torch.int64).sum().item()
            ),
            workspace_trace=public_trace,
        )
        self._validate_output(output, expected_training=expected_training)
        self._validate_token_provenance(prompt_provenance, segment="prompt")
        self._validate_token_provenance(target_provenance, segment="target")
        return output

    def _validate_native_execution(
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
                error_type=IrisSharedBufferTeacherForcingError,
                component=f"Arm-C teacher trace cycle {index + 1}",
            )
            if (
                contract.batch_size != 1
                or contract.entry_count != (index + 1) * self._num_modules
                or contract.dtype != dtype
                or contract.device != device
            ):
                message = "Arm-C teacher native trace is invalid"
                raise IrisSharedBufferTeacherForcingError(message)
        widths = (0, self._num_modules, 2 * self._num_modules)
        for read, width in zip(pre_update_reads, widths, strict=True):
            self._validate_read_snapshot(read, entry_count=width, dtype=dtype, device=device)
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
        attention_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        if (
            type(read) is not WorkspaceReaderOutput
            or type(read.workspace_reads) is not torch.Tensor
            or tuple(read.workspace_reads.shape) != (1, self._num_modules, self._state_dim)
            or read.workspace_reads.dtype != dtype
            or read.workspace_reads.device != device
            or not bool(torch.isfinite(read.workspace_reads).all())
            or type(read.attention_weights) is not torch.Tensor
            or tuple(read.attention_weights.shape) != (1, self._num_modules, entry_count)
            or read.attention_weights.dtype != attention_dtype
            or read.attention_weights.device != device
            or not bool(torch.isfinite(read.attention_weights).all())
        ):
            message = "Arm-C teacher reader snapshot is invalid"
            raise IrisSharedBufferTeacherForcingError(message)
        if entry_count == 0 and not bool((read.workspace_reads == 0).all()):
            message = "Arm-C teacher first cycle must receive exact zero shared reads"
            raise IrisSharedBufferTeacherForcingError(message)

    def _validate_wiring(self) -> None:
        if self._cognitive_config.model_dump_json() != self._cognitive_config_snapshot:
            message = "Arm-C teacher cognitive configuration changed"
            raise ValueError(message)
        if (
            self._policy.model_dump_json() != self._policy_snapshot
            or self._policy != build_shared_buffer_policy_v1()
            or self._policy_sha256 != shared_buffer_policy_sha256_v1()
        ):
            message = "Arm-C teacher shared-buffer policy changed"
            raise ValueError(message)
        metadata = self._read_backbone_metadata(self._backbone)
        if metadata != self._backbone_metadata:
            message = "Arm-C teacher backbone metadata changed"
            raise ValueError(message)
        if (
            metadata.max_prompt_tokens + self._num_modules + metadata.max_target_tokens
            > metadata.max_position_embeddings
        ):
            message = "Prompt, shared-buffer latents, and target maxima exceed backbone context"
            raise ValueError(message)
        LayerSampler.resolve_layer_indices(
            self._layer_fractions,
            metadata.num_hidden_layers,
        )
        registered = tuple(self._modules.items())
        if tuple(name for name, _ in registered) != self._CANONICAL_COMPONENT_NAMES:
            message = "Arm-C teacher module registration is not canonical"
            raise ValueError(message)
        if any(module is not getattr(self, name) for name, module in registered):
            message = "Arm-C teacher registered component identity changed"
            raise ValueError(message)
        if len(self._sidecar_modules) != len(self._CANONICAL_COMPONENT_NAMES) or any(
            name != expected or component is not getattr(self, expected)
            for (name, component), expected in zip(
                self._sidecar_modules,
                self._CANONICAL_COMPONENT_NAMES,
                strict=True,
            )
        ):
            message = "Arm-C teacher component ownership changed"
            raise ValueError(message)
        expected_components: tuple[tuple[str, nn.Module, dict[str, object]], ...] = (
            (
                "layer_sampler",
                self.layer_sampler,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": metadata.hidden_size,
                    "layer_fractions": self._layer_fractions,
                },
            ),
            ("goal_encoder", self.goal_encoder, {"state_dim": self._state_dim}),
            (
                "module_input_router",
                self.module_input_router,
                {"state_dim": self._state_dim, "num_modules": self._num_modules},
            ),
            (
                "shared_buffer_reader",
                self.shared_buffer_reader,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                    "max_entries": self._max_entries,
                },
            ),
            (
                "cognitive_modules",
                self.cognitive_modules,
                {"state_dim": self._state_dim, "num_modules": self._num_modules},
            ),
            (
                "proposal_head",
                self.proposal_head,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                },
            ),
            (
                "shared_buffer_writer",
                self.shared_buffer_writer,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                    "max_entries": self._max_entries,
                },
            ),
            (
                "shared_buffer_latent_bridge",
                self.shared_buffer_latent_bridge,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": metadata.hidden_size,
                    "num_modules": self._num_modules,
                    "cycles": self._cycles,
                    "final_entry_count": self._max_entries,
                },
            ),
            (
                "sequence_assembler",
                self.sequence_assembler,
                {
                    "hidden_size": metadata.hidden_size,
                    "num_slots": self._num_modules,
                    "vocabulary_size": metadata.vocabulary_size,
                    "max_prompt_tokens": metadata.max_prompt_tokens,
                    "max_target_tokens": metadata.max_target_tokens,
                    "max_context_tokens": metadata.max_position_embeddings,
                },
            ),
        )
        if tuple(name for name, _ in self._sidecar_modules) != tuple(
            name for name, _, _ in expected_components
        ):
            message = "Arm-C teacher components do not follow canonical order"
            raise ValueError(message)
        for component_name, component, expected_attributes in expected_components:
            for attribute, expected in expected_attributes.items():
                actual = getattr(component, attribute, None)
                if not self._same_typed_value(actual, expected):
                    message = (
                        f"{component_name}.{attribute}={actual!r} does not match "
                        f"arm-C teacher configuration {expected!r}"
                    )
                    raise ValueError(message)
            self._validate_module(component_name, component, metadata=metadata)

    @staticmethod
    def _read_backbone_metadata(backbone: BackboneAdapter) -> _BackboneMetadata:
        values = {
            "hidden_size": backbone.hidden_size,
            "num_hidden_layers": backbone.num_hidden_layers,
            "vocabulary_size": backbone.vocabulary_size,
            "max_position_embeddings": backbone.max_position_embeddings,
            "max_prompt_tokens": backbone.max_prompt_tokens,
            "max_target_tokens": backbone.max_new_tokens,
        }
        for name, value in values.items():
            if type(value) is not int or value <= 0:
                message = f"Backbone {name} must be a positive integer"
                raise ValueError(message)
        device = backbone.device
        dtype = backbone.dtype
        if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
            message = "Arm-C teacher backbone device must be explicit CPU or CUDA"
            raise ValueError(message)
        if device.index is not None and device.index < 0:
            message = "Arm-C teacher backbone device index cannot be negative"
            raise ValueError(message)
        if device.type == "cuda" and device.index is None:
            message = "Arm-C teacher CUDA device must have a concrete index"
            raise ValueError(message)
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            message = "Arm-C teacher backbone dtype is unsupported"
            raise ValueError(message)
        if device.type == "cpu" and dtype != torch.float32:
            message = "Arm-C teacher CPU execution requires fp32"
            raise ValueError(message)
        if type(backbone.trainable_parameter_count) is not int or (
            backbone.trainable_parameter_count != 0
        ):
            message = "Arm-C teacher backbone must remain frozen"
            raise ValueError(message)
        if type(backbone.has_parameter_gradients) is not bool or backbone.has_parameter_gradients:
            message = "Arm-C teacher backbone must not carry gradients"
            raise ValueError(message)
        return _BackboneMetadata(
            hidden_size=values["hidden_size"],
            num_hidden_layers=values["num_hidden_layers"],
            vocabulary_size=values["vocabulary_size"],
            max_position_embeddings=values["max_position_embeddings"],
            max_prompt_tokens=values["max_prompt_tokens"],
            max_target_tokens=values["max_target_tokens"],
            device=device,
            dtype=dtype,
        )

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
    def _validate_module(
        name: str,
        module: nn.Module,
        *,
        metadata: _BackboneMetadata,
    ) -> None:
        for parameter in module.parameters():
            if (
                type(parameter) is not nn.Parameter
                or parameter.is_inference()
                or not parameter.requires_grad
                or parameter.device != metadata.device
                or (torch.is_floating_point(parameter) and parameter.dtype != metadata.dtype)
                or not bool(torch.isfinite(parameter.detach()).all())
            ):
                message = f"{name} contains an invalid trainable parameter"
                raise ValueError(message)
        for buffer in module.buffers():
            if (
                type(buffer) is not torch.Tensor
                or buffer.is_inference()
                or buffer.device != metadata.device
                or (torch.is_floating_point(buffer) and buffer.dtype != metadata.dtype)
                or (torch.is_floating_point(buffer) and not bool(torch.isfinite(buffer).all()))
            ):
                message = f"{name} contains an invalid buffer"
                raise ValueError(message)

    def _validate_example(self, example: IrisTeacherForcingInput) -> None:
        if type(example) is not IrisTeacherForcingInput:
            message = "example must be exactly IrisTeacherForcingInput"
            raise IrisSharedBufferTeacherForcingError(message)
        if type(example.prompt) is not str or not example.prompt.strip():
            message = "Arm-C teacher prompt must contain non-whitespace text"
            raise IrisSharedBufferTeacherForcingError(message)
        try:
            example.prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            message = "Arm-C teacher prompt must be valid scalar Unicode"
            raise IrisSharedBufferTeacherForcingError(message) from error
        input_ids = example.target_input_ids
        attention_mask = example.target_attention_mask
        if (
            type(input_ids) is not torch.Tensor
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] <= 0
            or input_ids.shape[1] > self._backbone_metadata.max_target_tokens
            or input_ids.dtype != torch.int64
            or input_ids.device != self._backbone_metadata.device
        ):
            message = "Arm-C teacher target IDs violate the single-example contract"
            raise IrisSharedBufferTeacherForcingError(message)
        if not bool(
            torch.logical_and(
                input_ids >= 0,
                input_ids < self._backbone_metadata.vocabulary_size,
            ).all()
        ):
            message = "Arm-C teacher target IDs contain an out-of-vocabulary value"
            raise IrisSharedBufferTeacherForcingError(message)
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if (
            type(attention_mask) is not torch.Tensor
            or attention_mask.shape != input_ids.shape
            or attention_mask.dtype not in integer_dtypes
            or attention_mask.device != self._backbone_metadata.device
            or not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all())
        ):
            message = "Arm-C teacher target attention mask is invalid"
            raise IrisSharedBufferTeacherForcingError(message)
        canonical_mask = attention_mask.to(dtype=torch.bool)
        transitions = canonical_mask[:, 1:].to(dtype=torch.int8) - canonical_mask[:, :-1].to(
            dtype=torch.int8
        )
        if not bool(canonical_mask[:, 0].all()) or not bool((transitions <= 0).all()):
            message = "Arm-C teacher target mask must use non-empty right padding"
            raise IrisSharedBufferTeacherForcingError(message)

    def _capture_token_provenance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        segment: str,
    ) -> _TokenProvenance:
        try:
            with torch.inference_mode(False):
                expected_input_ids = input_ids.detach().clone()
                expected_attention_mask = attention_mask.detach().clone()
                owned_input_ids = expected_input_ids.detach().clone()
                owned_attention_mask = expected_attention_mask.detach().clone()
            provenance = _TokenProvenance(
                input_ids=owned_input_ids,
                attention_mask=owned_attention_mask,
                expected_input_ids=expected_input_ids,
                expected_attention_mask=expected_attention_mask,
                expected_versions=(owned_input_ids._version, owned_attention_mask._version),
                expected_storage=(
                    (owned_input_ids.device, owned_input_ids.untyped_storage().data_ptr()),
                    (
                        owned_attention_mask.device,
                        owned_attention_mask.untyped_storage().data_ptr(),
                    ),
                ),
            )
            source_storage = {
                (input_ids.device, input_ids.untyped_storage().data_ptr()),
                (attention_mask.device, attention_mask.untyped_storage().data_ptr()),
            }
            private_tensors = (
                expected_input_ids,
                expected_attention_mask,
                owned_input_ids,
                owned_attention_mask,
            )
            private_storage = {
                (tensor.device, tensor.untyped_storage().data_ptr()) for tensor in private_tensors
            }
        except Exception as error:
            message = f"Failed to capture exact arm-C {segment} token provenance"
            raise IrisSharedBufferTeacherForcingError(message) from error
        if len(private_storage) != len(private_tensors) or private_storage & source_storage:
            message = f"Arm-C {segment} provenance must use independent storage"
            raise IrisSharedBufferTeacherForcingError(message)
        self._validate_token_provenance(provenance, segment=segment)
        return provenance

    @staticmethod
    def _validate_token_provenance(
        provenance: _TokenProvenance,
        *,
        segment: str,
    ) -> None:
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
                message = f"Failed to verify exact arm-C {segment} {name}"
                raise IrisSharedBufferTeacherForcingError(message) from error
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
                message = f"Arm-C teacher stage mutated exact {segment} {name}"
                raise IrisSharedBufferTeacherForcingError(message)

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
            or context_features.dtype != self._backbone_metadata.dtype
            or context_features.device != self._backbone_metadata.device
            or context_features.is_inference()
            or not bool(torch.isfinite(context_features).all())
        ):
            message = "Layer sampler returned invalid arm-C teacher context"
            raise IrisSharedBufferTeacherForcingError(message)
        return context_features.shape[0], context_features.dtype, context_features.device

    @staticmethod
    def _extract_module_states(output: CognitiveModuleOutput) -> torch.Tensor:
        if (
            type(output) is not CognitiveModuleOutput
            or type(output.module_states) is not torch.Tensor
        ):
            message = "Cognitive modules returned invalid arm-C local state"
            raise IrisSharedBufferTeacherForcingError(message)
        return output.module_states

    def _validate_mode(self, *, expected_training: bool) -> None:
        if self.training is not expected_training:
            expected = "training" if expected_training else "evaluation"
            message = f"Arm-C teacher model must be in {expected} mode"
            raise IrisSharedBufferTeacherForcingError(message)
        for component_name, component in self._sidecar_modules:
            if any(child.training is not expected_training for child in component.modules()):
                expected = "training" if expected_training else "evaluation"
                message = f"{component_name} must recursively remain in {expected} mode"
                raise IrisSharedBufferTeacherForcingError(message)

    def _validate_runtime_boundary(
        self,
        *,
        expected_training: bool,
        prompt_provenance: _TokenProvenance | None = None,
        target_provenance: _TokenProvenance | None = None,
    ) -> None:
        try:
            self._validate_wiring()
        except Exception as error:
            message = "Arm-C teacher wiring is not stable"
            raise IrisSharedBufferTeacherForcingError(message) from error
        self._validate_mode(expected_training=expected_training)
        if prompt_provenance is not None:
            self._validate_token_provenance(prompt_provenance, segment="prompt")
        if target_provenance is not None:
            self._validate_token_provenance(target_provenance, segment="target")

    def _run_stage(
        self,
        name: str,
        operation: Callable[[], _ResultT],
        *,
        expected_training: bool,
        prompt_provenance: _TokenProvenance | None = None,
        target_provenance: _TokenProvenance | None = None,
    ) -> _ResultT:
        try:
            self._validate_runtime_boundary(
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            result = operation()
            self._validate_runtime_boundary(
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            return result
        except IrisSharedBufferTeacherForcingError:
            raise
        except Exception as error:
            message = f"Arm-C teacher forcing failed during {name}"
            raise IrisSharedBufferTeacherForcingError(message) from error

    def _validate_causal_output(
        self,
        output: object,
        *,
        total_length: int,
        expected_training: bool,
    ) -> None:
        if type(output) is not BackboneCausalOutput:
            message = "Backbone returned invalid arm-C teacher causal output"
            raise IrisSharedBufferTeacherForcingError(message)
        logits = output.logits
        if type(logits) is not torch.Tensor or tuple(logits.shape) != (
            1,
            total_length,
            self._backbone_metadata.vocabulary_size,
        ):
            message = "Arm-C teacher causal logits have an invalid shape"
            raise IrisSharedBufferTeacherForcingError(message)
        if (
            not torch.is_floating_point(logits)
            or logits.dtype != self._backbone_metadata.dtype
            or logits.device != self._backbone_metadata.device
            or not bool(torch.isfinite(logits).all())
            or logits.requires_grad is not expected_training
            or (expected_training and logits.grad_fn is None)
        ):
            message = "Arm-C teacher causal logits have an invalid boundary"
            raise IrisSharedBufferTeacherForcingError(message)
        if output.past_key_values is not None:
            message = "Arm-C teacher causal pass must not return a cache"
            raise IrisSharedBufferTeacherForcingError(message)
        loss = output.loss
        if (
            type(loss) is not torch.Tensor
            or loss.ndim != 0
            or not torch.is_floating_point(loss)
            or loss.device != self._backbone_metadata.device
            or not bool(torch.isfinite(loss))
            or loss.requires_grad is not expected_training
            or (expected_training and loss.grad_fn is None)
        ):
            message = "Arm-C teacher causal loss has an invalid boundary"
            raise IrisSharedBufferTeacherForcingError(message)

    def _validate_output(
        self,
        output: IrisTeacherForcedOutput,
        *,
        expected_training: bool,
    ) -> None:
        if type(output) is not IrisTeacherForcedOutput:
            message = "Arm-C teacher returned an invalid output"
            raise IrisSharedBufferTeacherForcingError(message)
        loss = output.loss
        if (
            type(loss) is not torch.Tensor
            or loss.ndim != 0
            or not torch.is_floating_point(loss)
            or loss.device != self._backbone_metadata.device
            or not bool(torch.isfinite(loss))
            or loss.requires_grad is not expected_training
            or (expected_training and loss.grad_fn is None)
        ):
            message = "Arm-C teacher output loss has an invalid boundary"
            raise IrisSharedBufferTeacherForcingError(message)
        if type(output.supervised_token_count) is not int or output.supervised_token_count <= 0:
            message = "Arm-C teacher token count must be a positive integer"
            raise IrisSharedBufferTeacherForcingError(message)
        trace = output.workspace_trace
        if type(trace) is not tuple or len(trace) != self._cycles:
            message = "Arm-C teacher trace must contain three projected snapshots"
            raise IrisSharedBufferTeacherForcingError(message)
        for index, state in enumerate(trace):
            expected_entries = (index + 1) * self._num_modules
            if (
                type(state) is not WorkspaceState
                or tuple(state.slots.shape) != (1, expected_entries, self._state_dim)
                or state.slots.requires_grad
                or state.slots.grad_fn is not None
                or state.slots.is_inference()
                or not bool(torch.isfinite(state.slots).all())
                or not bool(state.occupancy.all())
                or not bool((state.confidence == 1).all())
                or not bool((state.ttl == 1).all())
                or not bool((state.write_count == 1).all())
            ):
                message = "Arm-C teacher projected trace is invalid"
                raise IrisSharedBufferTeacherForcingError(message)


__all__ = [
    "IrisSharedBufferTeacherForcingError",
    "IrisSharedBufferTeacherForcingModel",
]
