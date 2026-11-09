"""Teacher-forced Iris-GW execution with a frozen language-model backbone."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

import torch
from torch import nn

from iris.ablations import CognitiveAblationPolicyV1, CognitiveAblationSessionV1
from iris.ablations.controls import (
    CognitiveAblationControlPlanV1,
    _AblationControlSessionV1,
    ablation_control_graph_sha256_v1,
    ablation_control_plan_sha256_v1,
    validate_ablation_control_plan_v1,
)
from iris.ablations.policy import validate_cognitive_ablation_policy_v1
from iris.backbone import BackboneAdapter, BackboneCausalOutput, BackboneFirstPassOutput
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import SecondPassSequenceAssembler
from iris.interventions.messages import WorkspaceMessagePlanV1, WorkspaceMessageSessionV1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.ablation_controls import (
    IrisAblationControlledTeacherOutputV1,
    controlled_teacher_invocation_sha256_v1,
)
from iris.training.ablation_observation import (
    IrisAblationObservedTeacherOutputV1,
    _AblationWorkspaceObservationSessionV1,
)
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.intervention import IrisIntervenedTeacherOutputV1, teacher_target_sha256_v1
from iris.training.messages import IrisMessageTeacherOutputV1
from iris.training.outputs import (
    IrisTeacherCycleTraceV1,
    IrisTeacherForcedOutput,
    IrisTeacherForcingInput,
    IrisTeacherModelCheckpointProvenance,
    IrisTeacherTracedOutputV1,
)
from iris.utils.hashing import (
    canonical_json_bytes,
    domain_sha256,
    module_state_sha256,
    tensor_sha256,
)
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace import (
    LatentBridge,
    SlotAllocator,
    WorkspaceArbiter,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceState,
    WorkspaceWriter,
)
from iris.workspace.trace import WorkspaceTraceOptionsV1, WorkspaceTraceRecorderV1

_ResultT = TypeVar("_ResultT")


class IrisTeacherForcingError(RuntimeError):
    """Raised when one teacher-forced Iris execution cannot complete safely."""


@dataclass(frozen=True, slots=True)
class _BackboneMetadata:
    """Stable public metadata required by the teacher-forcing graph."""

    hidden_size: int
    num_hidden_layers: int
    vocabulary_size: int
    max_position_embeddings: int
    max_prompt_tokens: int
    max_new_tokens: int
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _TokenProvenance:
    """Owned token tensors with immutable value and storage expectations."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    expected_input_ids: torch.Tensor
    expected_attention_mask: torch.Tensor
    expected_versions: tuple[int, int]
    expected_storage: tuple[tuple[torch.device, int], tuple[torch.device, int]]


class IrisTeacherForcingModel(nn.Module):
    """Run one prompt/target pair through the trainable Iris-GW sidecar."""

    _backbone: BackboneAdapter
    _cognitive_config: CognitiveConfig
    _workspace_config: WorkspaceConfig
    _cognitive_config_snapshot: str
    _workspace_config_snapshot: str
    _backbone_metadata: _BackboneMetadata

    _CANONICAL_COMPONENT_NAMES = (
        "layer_sampler",
        "goal_encoder",
        "module_input_router",
        "workspace_reader",
        "cognitive_modules",
        "proposal_head",
        "workspace_arbiter",
        "slot_allocator",
        "workspace_writer",
        "workspace_decay",
        "latent_bridge",
        "sequence_assembler",
    )

    def __init__(
        self,
        *,
        backbone: BackboneAdapter,
        cognitive_config: CognitiveConfig,
        workspace_config: WorkspaceConfig,
        layer_sampler: LayerSampler,
        goal_encoder: GoalEncoder,
        module_input_router: ModuleInputRouter,
        workspace_reader: WorkspaceReader,
        cognitive_modules: CognitiveModule,
        proposal_head: ProposalHead,
        workspace_arbiter: WorkspaceArbiter,
        slot_allocator: SlotAllocator,
        workspace_writer: WorkspaceWriter,
        workspace_decay: WorkspaceDecay,
        latent_bridge: LatentBridge,
        sequence_assembler: SecondPassSequenceAssembler,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, BackboneAdapter):
            message = "backbone must be an already-loaded BackboneAdapter"
            raise TypeError(message)
        if not isinstance(cognitive_config, CognitiveConfig):
            message = "cognitive_config must be a validated CognitiveConfig"
            raise TypeError(message)
        if not isinstance(workspace_config, WorkspaceConfig):
            message = "workspace_config must be a validated WorkspaceConfig"
            raise TypeError(message)
        if workspace_config.winners_per_cycle > cognitive_config.num_modules:
            message = "winners_per_cycle cannot exceed cognitive num_modules"
            raise ValueError(message)

        components: tuple[tuple[str, nn.Module, type[nn.Module]], ...] = (
            ("layer_sampler", layer_sampler, LayerSampler),
            ("goal_encoder", goal_encoder, GoalEncoder),
            ("module_input_router", module_input_router, ModuleInputRouter),
            ("workspace_reader", workspace_reader, WorkspaceReader),
            ("cognitive_modules", cognitive_modules, CognitiveModule),
            ("proposal_head", proposal_head, ProposalHead),
            ("workspace_arbiter", workspace_arbiter, WorkspaceArbiter),
            ("slot_allocator", slot_allocator, SlotAllocator),
            ("workspace_writer", workspace_writer, WorkspaceWriter),
            ("workspace_decay", workspace_decay, WorkspaceDecay),
            ("latent_bridge", latent_bridge, LatentBridge),
            ("sequence_assembler", sequence_assembler, SecondPassSequenceAssembler),
        )
        for name, component, expected_type in components:
            if not isinstance(component, expected_type):
                message = f"{name} must be {expected_type.__name__}"
                raise TypeError(message)
        if len({id(component) for _, component, _ in components}) != len(components):
            message = "Teacher-forcing components must be distinct module instances"
            raise ValueError(message)

        try:
            cognitive_snapshot = cognitive_config.model_dump_json()
            workspace_snapshot = workspace_config.model_dump_json()
            backbone_metadata = self._read_backbone_metadata(backbone)
        except Exception as error:
            message = "Failed to snapshot teacher-forcing configuration and backbone metadata"
            raise ValueError(message) from error

        # BackboneAdapter deliberately remains outside the registered nn.Module graph.
        object.__setattr__(self, "_backbone", backbone)
        object.__setattr__(self, "_cognitive_config", cognitive_config)
        object.__setattr__(self, "_workspace_config", workspace_config)
        object.__setattr__(self, "_cognitive_config_snapshot", cognitive_snapshot)
        object.__setattr__(self, "_workspace_config_snapshot", workspace_snapshot)
        object.__setattr__(self, "_backbone_metadata", backbone_metadata)

        self._state_dim = cognitive_config.state_dim
        self._num_modules = cognitive_config.num_modules
        self._layer_fractions = cognitive_config.layer_sampler.layer_fractions
        self._num_slots = workspace_config.num_slots
        self._cycles = workspace_config.cycles
        self._winners_per_cycle = workspace_config.winners_per_cycle
        self._decay_rate = workspace_config.decay_rate
        self._empty_threshold = workspace_config.empty_threshold
        self._arbiter_weights = (
            workspace_config.arbiter.confidence_weight,
            workspace_config.arbiter.novelty_weight,
            workspace_config.arbiter.relevance_weight,
        )
        self._allocator_weights = (
            workspace_config.allocator.empty_bonus,
            workspace_config.allocator.allocation_age_penalty,
            workspace_config.allocator.retention_age_penalty,
        )

        self.layer_sampler = layer_sampler
        self.goal_encoder = goal_encoder
        self.module_input_router = module_input_router
        self.workspace_reader = workspace_reader
        self.cognitive_modules = cognitive_modules
        self.proposal_head = proposal_head
        self.workspace_arbiter = workspace_arbiter
        self.slot_allocator = slot_allocator
        self.workspace_writer = workspace_writer
        self.workspace_decay = workspace_decay
        self.latent_bridge = latent_bridge
        self.sequence_assembler = sequence_assembler
        self._sidecar_modules = tuple(
            (name, getattr(self, name)) for name in self._CANONICAL_COMPONENT_NAMES
        )

        self.train()
        self._validate_wiring()
        self._validate_mode(expected_training=True)

    @property
    def backbone(self) -> BackboneAdapter:
        """Return the exact frozen backbone used for deterministic preparation."""
        return self._backbone

    def forward(self, example: IrisTeacherForcingInput) -> IrisTeacherForcedOutput:
        """Delegate the ordinary module call to the explicit training boundary."""
        return self.training_forward(example)

    def training_forward(
        self,
        example: IrisTeacherForcingInput,
        *,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisTeacherForcedOutput:
        """Return a graph-attached teacher loss without taking an optimizer step."""
        if torch.is_inference_mode_enabled():
            message = "Teacher-forced training cannot run under torch.inference_mode"
            raise IrisTeacherForcingError(message)
        if not torch.is_grad_enabled():
            message = "Teacher-forced training requires autograd to be enabled"
            raise IrisTeacherForcingError(message)
        self._validate_runtime_boundary(expected_training=True)
        return self._execute(example, expected_training=True, diagnostics=diagnostics)

    def evaluation_forward(
        self,
        example: IrisTeacherForcingInput,
        *,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisTeacherForcedOutput:
        """Evaluate one teacher pair without changing module mode or existing gradients."""
        if torch.is_inference_mode_enabled():
            message = "Teacher-forced evaluation cannot run under torch.inference_mode"
            raise IrisTeacherForcingError(message)
        self._validate_runtime_boundary(expected_training=False)
        with torch.no_grad():
            return self._execute(example, expected_training=False, diagnostics=diagnostics)

    def training_forward_with_trace(
        self,
        example: IrisTeacherForcingInput,
        *,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisTeacherTracedOutputV1:
        """Run the same teacher once, retaining differentiable pre-write usage inputs."""
        if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
            raise IrisTeacherForcingError("Traced training requires normal autograd")
        self._validate_runtime_boundary(expected_training=True)
        cycles: list[IrisTeacherCycleTraceV1] = []
        output = self._execute(
            example, expected_training=True, cycle_trace=cycles, diagnostics=diagnostics
        )
        return IrisTeacherTracedOutputV1(teacher=output, cycles=tuple(cycles))

    def evaluation_forward_with_trace(
        self,
        example: IrisTeacherForcingInput,
        *,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisTeacherTracedOutputV1:
        """Return the same eval result and no-grad usage inputs without mode changes."""
        if torch.is_inference_mode_enabled():
            raise IrisTeacherForcingError("Traced evaluation cannot run in inference_mode")
        self._validate_runtime_boundary(expected_training=False)
        cycles: list[IrisTeacherCycleTraceV1] = []
        with torch.no_grad():
            output = self._execute(
                example, expected_training=False, cycle_trace=cycles, diagnostics=diagnostics
            )
        return IrisTeacherTracedOutputV1(teacher=output, cycles=tuple(cycles))

    def evaluation_forward_intervened(
        self,
        example: IrisTeacherForcingInput,
        *,
        plan: WorkspaceInterventionPlanV1,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisIntervenedTeacherOutputV1:
        """Measure genuine target NLL under the same workspace phases as inference.

        Explicit eval mode is required; this method never trains, changes modes,
        computes auxiliary surrogate losses or returns an ordinary baseline row.
        Each call owns a fresh intervention session. Caller runtime is restored
        on success/failure; detected weight/input mutation aborts, not repairs, it.
        """
        if torch.is_inference_mode_enabled():
            raise IrisTeacherForcingError("Intervened evaluation requires ordinary no-grad tensors")
        if torch.is_autocast_enabled(self._backbone.device.type):
            raise IrisTeacherForcingError("Intervened evaluation requires explicit model precision")
        self._validate_runtime_boundary(expected_training=False)
        self._validate_example(example)
        caller = capture_runtime_state(self._backbone.device)
        try:
            session = WorkspaceInterventionSessionV1(
                plan,
                num_slots=self._num_slots,
                num_modules=self._num_modules,
                state_dim=self._state_dim,
                cycles=self._cycles,
                winners_per_cycle=self._winners_per_cycle,
            )
            sidecar = module_state_sha256(self._sidecar_modules)
            prompt = example.prompt
            target = teacher_target_sha256_v1(
                example.target_input_ids, example.target_attention_mask
            )
            graph = domain_sha256(
                "iris/intervened-teacher-graph/v1",
                canonical_json_bytes(
                    {
                        "backbone": self._backbone.configuration_json,
                        "cognitive": self._cognitive_config_snapshot,
                        "workspace": self._workspace_config_snapshot,
                    }
                ),
            )
            with torch.no_grad():
                output = self._execute(
                    example, expected_training=False, intervention=session, diagnostics=diagnostics
                )
            if (
                module_state_sha256(self._sidecar_modules) != sidecar
                or example.prompt != prompt
                or teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask)
                != target
            ):
                raise IrisTeacherForcingError("Intervened teacher weights or target changed")
            return IrisIntervenedTeacherOutputV1(
                loss=output.loss,
                supervised_token_count=output.supervised_token_count,
                workspace_trace=output.workspace_trace,
                prompt_sha256=domain_sha256(
                    "iris/intervened-teacher-prompt/v1", prompt.encode("utf-8")
                ),
                target_sha256=target,
                intervention_plan_sha256=session.plan_sha256,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
                diagnostic_trace=output.diagnostic_trace,
            )
        except Exception:
            raise IrisTeacherForcingError(
                "Intervened teacher evaluation did not complete"
            ) from None
        finally:
            restore_runtime_state(caller)

    def evaluation_forward_messages(
        self,
        example: IrisTeacherForcingInput,
        *,
        plan: WorkspaceMessagePlanV1,
        diagnostics: WorkspaceTraceOptionsV1,
        maximum_message_scalars: int = 65536,
    ) -> IrisMessageTeacherOutputV1:
        """Capture [C,M,D] real proposals; optionally transplant one content vector.

        Requires explicit eval/full trace; controls remain the recipient's own.
        The new plan/receipt domain does not change the legacy intervention contract.
        """
        self._validate_runtime_boundary(expected_training=False)
        self._validate_example(example)
        if torch.is_inference_mode_enabled() or torch.is_autocast_enabled(
            self._backbone.device.type
        ):
            raise IrisTeacherForcingError("Message evaluation requires ordinary explicit precision")
        if type(diagnostics) is not WorkspaceTraceOptionsV1 or diagnostics.mode != "full":
            raise IrisTeacherForcingError("Message evaluation requires explicit full diagnostics")
        caller = capture_runtime_state(self._backbone.device)
        try:
            sidecar = module_state_sha256(self._sidecar_modules)
            prompt = example.prompt
            prompt_sha = domain_sha256("iris/intervened-teacher-prompt/v1", prompt.encode("utf-8"))
            target = teacher_target_sha256_v1(
                example.target_input_ids, example.target_attention_mask
            )
            graph = domain_sha256(
                "iris/intervened-teacher-graph/v1",
                canonical_json_bytes(
                    {
                        "backbone": self._backbone.configuration_json,
                        "cognitive": self._cognitive_config_snapshot,
                        "workspace": self._workspace_config_snapshot,
                    }
                ),
            )
            session = WorkspaceMessageSessionV1(
                plan,
                example_id=diagnostics.example_id,
                prompt_sha256=prompt_sha,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
                num_slots=self._num_slots,
                num_modules=self._num_modules,
                state_dim=self._state_dim,
                cycles=self._cycles,
                winners_per_cycle=self._winners_per_cycle,
                maximum_message_scalars=maximum_message_scalars,
            )
            with torch.no_grad():
                output = self._execute(
                    example, expected_training=False, intervention=session, diagnostics=diagnostics
                )
            if (
                module_state_sha256(self._sidecar_modules) != sidecar
                or example.prompt != prompt
                or teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask)
                != target
                or output.diagnostic_trace is None
            ):
                raise IrisTeacherForcingError("Message teacher weights/input changed")
            return IrisMessageTeacherOutputV1(
                loss=output.loss,
                supervised_token_count=output.supervised_token_count,
                workspace_trace=output.workspace_trace,
                diagnostic_trace=output.diagnostic_trace,
                messages=session.messages,
                prompt_sha256=prompt_sha,
                target_sha256=target,
                message_plan_sha256=session.plan_sha256,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
            )
        except Exception:
            raise IrisTeacherForcingError("Message teacher evaluation did not complete") from None
        finally:
            restore_runtime_state(caller)

    def evaluation_forward_ablation_controlled(
        self,
        example: IrisTeacherForcingInput,
        *,
        plan: CognitiveAblationControlPlanV1,
        diagnostics: WorkspaceTraceOptionsV1,
    ) -> IrisAblationControlledTeacherOutputV1:
        """Actual eval-only mode+control math; never the five-term training objective.

        Legacy ablation/intervention entry points still cannot be combined. This
        distinct plan owns fresh trusted engines and a combined trace identity.
        Caller labels are diagnostic context, not selected-best/source attestation.
        """
        return self._evaluation_forward_ablation_controlled(
            example, plan=plan, diagnostics=diagnostics, observation=None
        )

    def evaluation_forward_ablation_observed(
        self,
        example: IrisTeacherForcingInput,
        *,
        plan: CognitiveAblationControlPlanV1,
        diagnostics: WorkspaceTraceOptionsV1,
        maximum_observation_scalars: int,
    ) -> IrisAblationObservedTeacherOutputV1:
        """Distinct bounded actual prewrite/native-write receipt, not causal effects.

        Extra seven-field snapshots are counted separately from diagnostic trace
        and proposal budgets. Whole-experiment consumers must budget all three.
        """
        try:
            if type(diagnostics) is not WorkspaceTraceOptionsV1 or diagnostics.mode != "full":
                raise ValueError("Writer observations require explicit full diagnostics")
            observation = _AblationWorkspaceObservationSessionV1(
                cycles=self._cycles,
                slots=self._num_slots,
                width=self._state_dim,
                modules=self._num_modules,
                maximum=maximum_observation_scalars,
            )
            output = self._evaluation_forward_ablation_controlled(
                example, plan=plan, diagnostics=diagnostics, observation=observation
            )
            return observation.finish(output)
        except Exception:
            raise IrisTeacherForcingError(
                "Observed mode/control evaluation did not complete"
            ) from None

    def _evaluation_forward_ablation_controlled(
        self,
        example: IrisTeacherForcingInput,
        *,
        plan: CognitiveAblationControlPlanV1,
        diagnostics: WorkspaceTraceOptionsV1,
        observation: _AblationWorkspaceObservationSessionV1 | None,
    ) -> IrisAblationControlledTeacherOutputV1:
        if torch.is_inference_mode_enabled() or torch.is_autocast_enabled(
            self._backbone.device.type
        ):
            raise IrisTeacherForcingError("Mode/control evaluation requires explicit precision")
        self._validate_runtime_boundary(expected_training=False)
        self._validate_example(example)
        caller = capture_runtime_state(self._backbone.device)
        failure: BaseException | None = None
        try:
            owned = validate_ablation_control_plan_v1(plan)
            if type(diagnostics) is not WorkspaceTraceOptionsV1:
                raise TypeError("Exact explicit diagnostic options required")
            diagnostics = WorkspaceTraceOptionsV1.model_validate(diagnostics)
            if diagnostics.mode == "none" or (
                type(owned.control) is WorkspaceMessagePlanV1 and diagnostics.mode != "full"
            ):
                raise ValueError("Messages require full diagnostics; controls require a trace")
            prompt = example.prompt
            prompt_sha = domain_sha256("iris/cognitive-ablation-prompt/v1", prompt.encode("utf-8"))
            target = teacher_target_sha256_v1(
                example.target_input_ids, example.target_attention_mask
            )
            original_targets = (example.target_input_ids, example.target_attention_mask)
            target_versions = tuple((id(t), t._version) for t in original_targets)
            sidecar = module_state_sha256(self._sidecar_modules)
            tensors = tuple(self.parameters()) + tuple(self.buffers())
            versions = tuple(tensor._version for tensor in tensors)

            def gradients() -> tuple[tuple[int, int | None, int | None, str | None], ...]:
                return tuple(
                    (
                        id(parameter),
                        None if parameter.grad is None else id(parameter.grad),
                        None if parameter.grad is None else parameter.grad._version,
                        None if parameter.grad is None else tensor_sha256(parameter.grad),
                    )
                    for parameter in self.parameters()
                )

            before_gradients = gradients()
            graph = ablation_control_graph_sha256_v1(
                backbone=self._backbone.configuration_json,
                cognitive=self._cognitive_config_snapshot,
                workspace=self._workspace_config_snapshot,
                ablation=owned.ablation,
            )
            session = _AblationControlSessionV1(
                owned,
                example_id=diagnostics.example_id,
                prompt_sha256=prompt_sha,
                target_sha256=target,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
                num_modules=self._num_modules,
                num_slots=self._num_slots,
                state_dim=self._state_dim,
                cycles=self._cycles,
                winners_per_cycle=self._winners_per_cycle,
            )
            with torch.no_grad():
                output = self._execute(
                    example,
                    expected_training=False,
                    controlled=session,
                    diagnostics=diagnostics,
                    observation=observation,
                )
            session.check_intent()
            if (
                module_state_sha256(self._sidecar_modules) != sidecar
                or example.prompt != prompt
                or tuple(tensor._version for tensor in tensors) != versions
                or gradients() != before_gradients
                or tuple(
                    (id(t), t._version)
                    for t in (example.target_input_ids, example.target_attention_mask)
                )
                != target_versions
                or teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask)
                != target
                or output.diagnostic_trace is None
                or ablation_control_plan_sha256_v1(session.plan) != session.plan_sha256
            ):
                raise ValueError("Mode/control evaluation changed state, inputs or intent")
            invocation = controlled_teacher_invocation_sha256_v1(
                plan_sha256=session.plan_sha256,
                prompt_sha256=prompt_sha,
                target_sha256=target,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
                supervised_token_count=output.supervised_token_count,
            )
            return IrisAblationControlledTeacherOutputV1(
                loss=output.loss,
                supervised_token_count=output.supervised_token_count,
                workspace_trace=output.workspace_trace,
                diagnostic_trace=output.diagnostic_trace,
                messages=session.control.messages
                if type(session.control) is WorkspaceMessageSessionV1
                else (),
                plan=session.plan,
                plan_sha256=session.plan_sha256,
                ablation_policy_sha256=session.policy_sha256,
                control_plan_sha256=session.control_sha256,
                prompt_sha256=prompt_sha,
                target_sha256=target,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
                invocation_sha256=invocation,
            )
        except Exception:
            failure = IrisTeacherForcingError("Mode/control teacher evaluation did not complete")
            raise failure from None
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                restore_runtime_state(caller)
            except BaseException:
                if failure is not None:
                    raise failure from None
                raise IrisTeacherForcingError("Mode/control runtime restoration failed") from None

    def training_forward_ablated(
        self,
        example: IrisTeacherForcingInput,
        *,
        policy: CognitiveAblationPolicyV1,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisAblatedTeacherOutputV1:
        """Actual graph-attached ablation math, not a legacy scientific run receipt."""
        return self._forward_ablated(
            example, policy=policy, expected_training=True, diagnostics=diagnostics
        )

    def evaluation_forward_ablated(
        self,
        example: IrisTeacherForcingInput,
        *,
        policy: CognitiveAblationPolicyV1,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisAblatedTeacherOutputV1:
        """Actual ordinary no-grad ablation, retaining the final workspace LM bridge."""
        return self._forward_ablated(
            example, policy=policy, expected_training=False, diagnostics=diagnostics
        )

    def _forward_ablated(
        self,
        example: IrisTeacherForcingInput,
        *,
        policy: CognitiveAblationPolicyV1,
        expected_training: bool,
        diagnostics: WorkspaceTraceOptionsV1 | None,
    ) -> IrisAblatedTeacherOutputV1:
        if torch.is_inference_mode_enabled() or torch.is_autocast_enabled(
            self._backbone.device.type
        ):
            raise IrisTeacherForcingError("Ablations require ordinary explicit-precision tensors")
        if expected_training and not torch.is_grad_enabled():
            raise IrisTeacherForcingError("Training ablations require autograd")
        self._validate_runtime_boundary(expected_training=expected_training)
        self._validate_example(example)
        caller = capture_runtime_state(self._backbone.device)
        try:
            policy = validate_cognitive_ablation_policy_v1(policy)
            prompt = example.prompt
            prompt_sha = domain_sha256("iris/cognitive-ablation-prompt/v1", prompt.encode("utf-8"))
            target_sha = teacher_target_sha256_v1(
                example.target_input_ids, example.target_attention_mask
            )
            sidecar = module_state_sha256(self._sidecar_modules)
            graph = domain_sha256(
                "iris/cognitive-ablation-teacher-graph/v1",
                canonical_json_bytes(
                    {
                        "backbone": self._backbone.configuration_json,
                        "cognitive": self._cognitive_config_snapshot,
                        "workspace": self._workspace_config_snapshot,
                        "policy": policy.model_dump(mode="json"),
                    }
                ),
            )
            session = CognitiveAblationSessionV1(
                policy,
                prompt_sha256=prompt_sha,
                num_modules=self._num_modules,
                num_slots=self._num_slots,
                state_dim=self._state_dim,
                cycles=self._cycles,
            )
            cycles: list[IrisTeacherCycleTraceV1] = []
            if expected_training:
                output = self._execute(
                    example,
                    expected_training=True,
                    cycle_trace=cycles,
                    ablation=session,
                    diagnostics=diagnostics,
                )
            else:
                with torch.no_grad():
                    output = self._execute(
                        example,
                        expected_training=False,
                        cycle_trace=cycles,
                        ablation=session,
                        diagnostics=diagnostics,
                    )
            if (
                module_state_sha256(self._sidecar_modules) != sidecar
                or example.prompt != prompt
                or teacher_target_sha256_v1(example.target_input_ids, example.target_attention_mask)
                != target_sha
            ):
                raise IrisTeacherForcingError("Ablation changed model weights or input")
            invocation = domain_sha256(
                "iris/cognitive-ablation-teacher-invocation/v1",
                canonical_json_bytes(
                    {
                        "mode": "training" if expected_training else "evaluation",
                        "policy_sha256": session.policy_sha256,
                        "prompt_sha256": prompt_sha,
                        "target_sha256": target_sha,
                        "teacher_graph_sha256": graph,
                        "sidecar_state_sha256": sidecar,
                    }
                ),
            )
            return IrisAblatedTeacherOutputV1(
                mode="training" if expected_training else "evaluation",
                traced=IrisTeacherTracedOutputV1(teacher=output, cycles=tuple(cycles)),
                policy=policy,
                policy_sha256=session.policy_sha256,
                invocation_sha256=invocation,
                prompt_sha256=prompt_sha,
                target_sha256=target_sha,
                teacher_graph_sha256=graph,
                sidecar_state_sha256=sidecar,
            )
        except Exception:
            raise IrisTeacherForcingError("Cognitive ablation did not complete") from None
        finally:
            restore_runtime_state(caller)

    @property
    def checkpoint_provenance(self) -> IrisTeacherModelCheckpointProvenance:
        """Return current sidecar identity after validating the training graph."""
        return self._execution_provenance(expected_training=True)

    @property
    def evaluation_provenance(self) -> IrisTeacherModelCheckpointProvenance:
        """Read the same model identity in eval mode without changing any mode."""
        return self._execution_provenance(expected_training=False)

    def _execution_provenance(
        self, *, expected_training: bool
    ) -> IrisTeacherModelCheckpointProvenance:
        if torch.is_inference_mode_enabled():
            message = "Checkpoint provenance cannot be captured under torch.inference_mode"
            raise IrisTeacherForcingError(message)
        self._validate_runtime_boundary(expected_training=expected_training)
        try:
            backbone_configuration_json = self._backbone.configuration_json
            state_sha256 = module_state_sha256(self._sidecar_modules)
        except Exception as error:
            message = "Failed to capture teacher checkpoint provenance"
            raise IrisTeacherForcingError(message) from error
        metadata = self._backbone_metadata
        return IrisTeacherModelCheckpointProvenance(
            backbone_configuration_json=backbone_configuration_json,
            cognitive_configuration_json=self._cognitive_config_snapshot,
            workspace_configuration_json=self._workspace_config_snapshot,
            hidden_size=metadata.hidden_size,
            num_hidden_layers=metadata.num_hidden_layers,
            vocabulary_size=metadata.vocabulary_size,
            max_position_embeddings=metadata.max_position_embeddings,
            max_prompt_tokens=metadata.max_prompt_tokens,
            max_target_tokens=metadata.max_new_tokens,
            device=str(metadata.device),
            dtype=str(metadata.dtype).removeprefix("torch."),
            sidecar_state_sha256=state_sha256,
        )

    def _execute(
        self,
        example: IrisTeacherForcingInput,
        *,
        expected_training: bool,
        cycle_trace: list[IrisTeacherCycleTraceV1] | None = None,
        intervention: WorkspaceInterventionSessionV1 | WorkspaceMessageSessionV1 | None = None,
        ablation: CognitiveAblationSessionV1 | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
        controlled: _AblationControlSessionV1 | None = None,
        observation: _AblationWorkspaceObservationSessionV1 | None = None,
    ) -> IrisTeacherForcedOutput:
        if observation is not None and (
            type(observation) is not _AblationWorkspaceObservationSessionV1
            or controlled is None
            or expected_training
            or type(diagnostics) is not WorkspaceTraceOptionsV1
            or diagnostics.mode != "full"
        ):
            raise IrisTeacherForcingError(
                "Only exact native controlled writer observations allowed"
            )
        if intervention is not None and (expected_training or cycle_trace is not None):
            raise IrisTeacherForcingError("Interventions require a separate evaluation-only path")
        if ablation is not None and (
            type(ablation) is not CognitiveAblationSessionV1 or intervention is not None
        ):
            raise IrisTeacherForcingError(
                "Cognitive ablations cannot share legacy intervention sessions"
            )
        self._validate_example(example)
        if controlled is not None:
            if (
                type(controlled) is not _AblationControlSessionV1
                or expected_training
                or cycle_trace is not None
                or ablation is not None
                or intervention is not None
                or type(diagnostics) is not WorkspaceTraceOptionsV1
            ):
                raise IrisTeacherForcingError(
                    "Only the distinct fresh eval mode/control path is allowed"
                )
            controlled.claim(
                example_id=diagnostics.example_id,
                prompt_sha256=domain_sha256(
                    "iris/cognitive-ablation-prompt/v1", example.prompt.encode()
                ),
                target_sha256=teacher_target_sha256_v1(
                    example.target_input_ids, example.target_attention_mask
                ),
                teacher_graph_sha256=ablation_control_graph_sha256_v1(
                    backbone=self._backbone.configuration_json,
                    cognitive=self._cognitive_config_snapshot,
                    workspace=self._workspace_config_snapshot,
                    ablation=controlled.plan.ablation,
                ),
                sidecar_state_sha256=module_state_sha256(self._sidecar_modules),
                num_modules=self._num_modules,
                num_slots=self._num_slots,
                state_dim=self._state_dim,
                cycles=self._cycles,
                winners_per_cycle=self._winners_per_cycle,
            )
            # Original legacy-composition rejection ran ABOVE. Only this exact new
            # private session can now expose its fresh owned trusted phase engines.
            ablation, intervention = controlled.ablation, controlled.control
            if observation is not None:
                observation.claim(
                    context=(
                        controlled.plan_sha256,
                        controlled._context[1],
                        controlled._context[2],
                        controlled._context[3],
                        controlled._context[4],
                    ),
                    geometry=(self._cycles, self._num_slots, self._state_dim, self._num_modules),
                )
        diagnostic = WorkspaceTraceRecorderV1(
            diagnostics,
            num_modules=self._num_modules,
            num_slots=self._num_slots,
            state_dim=self._state_dim,
            cycles=self._cycles,
            intervention_plan_sha256=controlled.plan_sha256
            if controlled is not None
            else ablation.policy_sha256
            if ablation is not None
            else None
            if intervention is None
            else intervention.plan_sha256,
        )
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
        if not isinstance(first_pass, BackboneFirstPassOutput):
            message = "Backbone did not return first-pass token provenance"
            raise IrisTeacherForcingError(message)
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
            lambda: self.goal_encoder(
                sampled.context_features,
                sampled.attention_mask,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        if ablation is not None:
            goal = self._run_stage(
                "cognitive ablation goal",
                partial(ablation.after_goal, goal),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
        routed = self._run_stage(
            "module input routing",
            lambda: (
                self.module_input_router(
                    sampled.context_features,
                    sampled.attention_mask,
                )
                if ablation is None
                else ablation.route(
                    self.module_input_router, sampled.context_features, sampled.attention_mask
                )
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )

        batch_size, state_dtype, state_device = self._run_stage(
            "sampled context metadata validation",
            lambda: self._context_metadata(sampled.context_features),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
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
        workspace = self._run_stage(
            "workspace initialization",
            lambda: WorkspaceState.empty(
                batch_size=batch_size,
                num_slots=self._num_slots,
                state_dim=self._state_dim,
                dtype=state_dtype,
                device=state_device,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )

        trace_states: list[WorkspaceState] = []
        for cycle_index in range(self._cycles):
            cycle_number = cycle_index + 1
            if cycle_index > 0:
                workspace = self._run_stage(
                    f"workspace decay before cycle {cycle_number}",
                    partial(self.workspace_decay, workspace),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            if intervention is not None:
                workspace = self._run_stage(
                    f"intervention before read for cycle {cycle_number}",
                    partial(intervention.before_read, workspace, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            read_state = workspace
            reads = self._run_stage(
                f"workspace read for cycle {cycle_number}",
                partial(self.workspace_reader, workspace),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            if ablation is not None:
                reads = self._run_stage(
                    f"cognitive ablation broadcast for cycle {cycle_number}",
                    partial(ablation.after_read, reads, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            if intervention is not None:
                reads = self._run_stage(
                    f"intervention after read for cycle {cycle_number}",
                    partial(intervention.after_read, reads, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
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
                f"cognitive state extraction for cycle {cycle_number}",
                partial(getattr, module_output, "module_states"),
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
            if intervention is not None:
                proposal = self._run_stage(
                    f"intervention after proposal for cycle {cycle_number}",
                    partial(intervention.after_proposal, proposal, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            arbitration = self._run_stage(
                f"workspace arbitration for cycle {cycle_number}",
                partial(
                    self.workspace_arbiter,
                    proposal,
                    workspace,
                    goal.goal_state,
                ),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            if intervention is not None:
                arbitration = self._run_stage(
                    f"intervention after arbitration for cycle {cycle_number}",
                    partial(intervention.after_arbitration, arbitration, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            allocation = self._run_stage(
                f"slot allocation for cycle {cycle_number}",
                partial(self.slot_allocator, proposal, workspace, arbitration),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            if intervention is not None:
                allocation = self._run_stage(
                    f"intervention after allocation for cycle {cycle_number}",
                    partial(intervention.after_allocation, workspace, allocation, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            if type(intervention) is WorkspaceMessageSessionV1:
                proposal = self._run_stage(
                    f"message transfer before writer for cycle {cycle_number}",
                    partial(intervention.prepare_write, proposal, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            if cycle_trace is not None:
                # Clone before the writer can consume/mutate its inputs; do not
                # detach probabilities/scores needed by the auxiliary graph.
                cycle_trace.append(
                    IrisTeacherCycleTraceV1(
                        module_probabilities=arbitration.soft_probabilities.clone(),
                        allocation_scores=allocation.allocation_scores.clone(),
                        winner_mask=arbitration.winner_mask.detach().clone(),
                        assignment_mask=allocation.assignment_mask.detach().clone(),
                    )
                )
            before_write = workspace
            if observation is not None:
                self._run_stage(
                    f"observe actual prewrite for cycle {cycle_number}",
                    partial(observation.before_write, before_write, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            workspace = self._run_stage(
                f"workspace write for cycle {cycle_number}",
                partial(
                    self.workspace_writer,
                    proposal,
                    workspace,
                    allocation,
                    goal.goal_state,
                ),
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
            if observation is not None:
                self._run_stage(
                    f"observe actual native write for cycle {cycle_number}",
                    partial(observation.native_written, before_write, workspace, cycle_number),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            if intervention is not None:
                workspace = self._run_stage(
                    f"intervention after write for cycle {cycle_number}",
                    partial(
                        intervention.after_write, before_write, workspace, allocation, cycle_number
                    ),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            if diagnostic.enabled:
                self._run_stage(
                    f"diagnostic trace for cycle {cycle_number}",
                    partial(
                        diagnostic.record,
                        cycle=cycle_number,
                        read_state=read_state,
                        before_write=before_write,
                        workspace=workspace,
                        reads=reads,
                        module_states=module_states,
                        proposal=proposal,
                        arbitration=arbitration,
                        allocation=allocation,
                    ),
                    expected_training=expected_training,
                    prompt_provenance=prompt_provenance,
                    target_provenance=target_provenance,
                )
            trace_states.append(workspace)

        if intervention is not None:
            self._run_stage(
                "intervention completion",
                intervention.finish,
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
            )
        workspace_trace = tuple(trace_states)
        if ablation is not None:
            self._run_stage(
                "cognitive ablation completion",
                ablation.finish,
                expected_training=expected_training,
                prompt_provenance=prompt_provenance,
                target_provenance=target_provenance,
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
            "workspace latent bridge",
            lambda: self.latent_bridge(
                workspace_trace[-1],
                prompt_provenance.attention_mask,
            ),
            expected_training=expected_training,
            prompt_provenance=prompt_provenance,
            target_provenance=target_provenance,
        )
        sequence = self._run_stage(
            "teacher-forced second-pass sequence",
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
            message = "Teacher-forced causal pass did not return a loss"
            raise IrisTeacherForcingError(message)
        if not expected_training:
            loss = loss.detach().clone()
        supervised_token_count = int(
            target_provenance.attention_mask.to(dtype=torch.int64).sum().item()
        )
        output = IrisTeacherForcedOutput(
            loss=loss,
            supervised_token_count=supervised_token_count,
            workspace_trace=workspace_trace,
            diagnostic_trace=diagnostic.finish(),
        )
        self._validate_output(output, expected_training=expected_training)
        self._validate_token_provenance(prompt_provenance, segment="prompt")
        self._validate_token_provenance(target_provenance, segment="target")
        return output

    def _validate_wiring(self) -> None:
        try:
            configuration_unchanged = (
                self._cognitive_config.model_dump_json() == self._cognitive_config_snapshot
                and self._workspace_config.model_dump_json() == self._workspace_config_snapshot
            )
        except Exception as error:
            message = "Failed to validate teacher-forcing configuration snapshots"
            raise ValueError(message) from error
        if not configuration_unchanged:
            message = "Validated teacher-forcing configuration changed after construction"
            raise ValueError(message)
        try:
            metadata = self._read_backbone_metadata(self._backbone)
        except Exception as error:
            message = "Failed to validate teacher-forcing backbone metadata"
            raise ValueError(message) from error
        if metadata != self._backbone_metadata:
            message = "Backbone metadata changed after teacher-forcing construction"
            raise ValueError(message)
        if self._num_slots + metadata.max_prompt_tokens + metadata.max_new_tokens > (
            metadata.max_position_embeddings
        ):
            message = "Prompt, workspace, and target maxima exceed the backbone context"
            raise ValueError(message)
        LayerSampler.resolve_layer_indices(
            self._layer_fractions,
            metadata.num_hidden_layers,
        )

        registered = tuple(self._modules.items())
        if tuple(name for name, _ in registered) != self._CANONICAL_COMPONENT_NAMES:
            message = "Teacher-forcing model registration is not canonical"
            raise ValueError(message)
        if any(module is not getattr(self, name) for name, module in registered):
            message = "Teacher-forcing registered module identity changed"
            raise ValueError(message)
        if len(self._sidecar_modules) != len(self._CANONICAL_COMPONENT_NAMES) or any(
            name != expected_name or component is not getattr(self, expected_name)
            for (name, component), expected_name in zip(
                self._sidecar_modules,
                self._CANONICAL_COMPONENT_NAMES,
                strict=True,
            )
        ):
            message = "Teacher-forcing component ownership changed"
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
                "workspace_reader",
                self.workspace_reader,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
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
                "workspace_arbiter",
                self.workspace_arbiter,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
                    "winners_per_cycle": self._winners_per_cycle,
                    "confidence_weight": self._arbiter_weights[0],
                    "novelty_weight": self._arbiter_weights[1],
                    "relevance_weight": self._arbiter_weights[2],
                },
            ),
            (
                "slot_allocator",
                self.slot_allocator,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
                    "winners_per_cycle": self._winners_per_cycle,
                    "empty_bonus": self._allocator_weights[0],
                    "allocation_age_penalty": self._allocator_weights[1],
                    "retention_age_penalty": self._allocator_weights[2],
                },
            ),
            (
                "workspace_writer",
                self.workspace_writer,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
                    "winners_per_cycle": self._winners_per_cycle,
                },
            ),
            (
                "workspace_decay",
                self.workspace_decay,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
                    "decay_rate": self._decay_rate,
                    "empty_threshold": self._empty_threshold,
                },
            ),
            (
                "latent_bridge",
                self.latent_bridge,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": metadata.hidden_size,
                    "num_slots": self._num_slots,
                },
            ),
            (
                "sequence_assembler",
                self.sequence_assembler,
                {
                    "hidden_size": metadata.hidden_size,
                    "num_slots": self._num_slots,
                    "vocabulary_size": metadata.vocabulary_size,
                    "max_prompt_tokens": metadata.max_prompt_tokens,
                    "max_target_tokens": metadata.max_new_tokens,
                    "max_context_tokens": metadata.max_position_embeddings,
                },
            ),
        )
        for component_name, component, expected_attributes in expected_components:
            for attribute, expected in expected_attributes.items():
                try:
                    actual = getattr(component, attribute, None)
                    matches_expected = self._same_typed_value(actual, expected)
                except Exception as error:
                    message = (
                        f"Failed to read or compare {component_name}.{attribute} "
                        "for teacher-forcing wiring"
                    )
                    raise ValueError(message) from error
                if not matches_expected:
                    message = (
                        f"{component_name}.{attribute}={actual!r} does not match "
                        f"teacher-forcing configuration {expected!r}"
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
            "max_new_tokens": backbone.max_new_tokens,
        }
        for name, value in values.items():
            if type(value) is not int or value <= 0:
                message = f"Backbone {name} must be a positive integer"
                raise ValueError(message)
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
            message = "Iris-GW-0.1 supports teacher forcing only in fp32 on CPU"
            raise ValueError(message)
        trainable_count = backbone.trainable_parameter_count
        if type(trainable_count) is not int or trainable_count != 0:
            message = "Teacher-forcing backbone must not contain trainable parameters"
            raise ValueError(message)
        has_gradients = backbone.has_parameter_gradients
        if type(has_gradients) is not bool or has_gradients:
            message = "Teacher-forcing backbone parameters must not carry gradients"
            raise ValueError(message)
        return _BackboneMetadata(
            hidden_size=values["hidden_size"],
            num_hidden_layers=values["num_hidden_layers"],
            vocabulary_size=values["vocabulary_size"],
            max_position_embeddings=values["max_position_embeddings"],
            max_prompt_tokens=values["max_prompt_tokens"],
            max_new_tokens=values["max_new_tokens"],
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
                cls._same_typed_value(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
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
            if parameter.is_inference():
                message = f"{name} parameters must be ordinary trainable tensors"
                raise ValueError(message)
            if not parameter.requires_grad:
                message = f"{name} parameters must remain trainable"
                raise ValueError(message)
            if parameter.device != metadata.device:
                message = f"{name} parameters are not on the backbone device"
                raise ValueError(message)
            if torch.is_floating_point(parameter) and parameter.dtype != metadata.dtype:
                message = f"{name} parameters do not use the backbone dtype"
                raise ValueError(message)
            if torch.is_floating_point(parameter) and not bool(
                torch.isfinite(parameter.detach()).all()
            ):
                message = f"{name} parameters contain NaN or Inf"
                raise ValueError(message)
        for buffer in module.buffers():
            if buffer.is_inference():
                message = f"{name} buffers must be ordinary tensors"
                raise ValueError(message)
            if buffer.device != metadata.device:
                message = f"{name} buffers are not on the backbone device"
                raise ValueError(message)
            if torch.is_floating_point(buffer) and buffer.dtype != metadata.dtype:
                message = f"{name} floating buffers do not use the backbone dtype"
                raise ValueError(message)
            if torch.is_floating_point(buffer) and not bool(torch.isfinite(buffer).all()):
                message = f"{name} buffers contain NaN or Inf"
                raise ValueError(message)

    def _validate_example(self, example: IrisTeacherForcingInput) -> None:
        if not isinstance(example, IrisTeacherForcingInput):
            message = "example must be IrisTeacherForcingInput"
            raise IrisTeacherForcingError(message)
        if type(example.prompt) is not str or not example.prompt.strip():
            message = "Teacher-forcing prompt must contain non-whitespace text"
            raise IrisTeacherForcingError(message)
        try:
            example.prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            message = "Teacher-forcing prompt must contain valid UTF-8 scalar text"
            raise IrisTeacherForcingError(message) from error
        input_ids = example.target_input_ids
        attention_mask = example.target_attention_mask
        if type(input_ids) is not torch.Tensor or input_ids.ndim != 2:
            message = "Target input IDs must have shape [1, target]"
            raise IrisTeacherForcingError(message)
        target_length = input_ids.shape[1]
        if input_ids.shape[0] != 1 or target_length <= 0:
            message = "Target input IDs must contain one non-empty sequence"
            raise IrisTeacherForcingError(message)
        if target_length > self._backbone_metadata.max_new_tokens:
            message = "Target input IDs exceed backbone max_new_tokens"
            raise IrisTeacherForcingError(message)
        if input_ids.dtype != torch.int64:
            message = "Target input IDs must use torch.int64"
            raise IrisTeacherForcingError(message)
        if input_ids.device != self._backbone_metadata.device:
            message = "Target input IDs must be on the exact backbone device"
            raise IrisTeacherForcingError(message)
        if not bool(
            torch.logical_and(
                input_ids >= 0,
                input_ids < self._backbone_metadata.vocabulary_size,
            ).all()
        ):
            message = "Target input IDs contain an out-of-vocabulary index"
            raise IrisTeacherForcingError(message)

        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if type(attention_mask) is not torch.Tensor or attention_mask.shape != input_ids.shape:
            message = "Target attention mask must match target input IDs"
            raise IrisTeacherForcingError(message)
        if attention_mask.dtype not in integer_dtypes:
            message = "Target attention mask must use a boolean or integer dtype"
            raise IrisTeacherForcingError(message)
        if attention_mask.device != self._backbone_metadata.device:
            message = "Target attention mask must be on the exact backbone device"
            raise IrisTeacherForcingError(message)
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            message = "Target attention mask must contain only zeros and ones"
            raise IrisTeacherForcingError(message)
        canonical_mask = attention_mask.to(dtype=torch.bool)
        if not bool(canonical_mask[:, 0].all()):
            message = "Target attention mask must start with an attended token"
            raise IrisTeacherForcingError(message)
        transitions = canonical_mask[:, 1:].to(dtype=torch.int8) - canonical_mask[:, :-1].to(
            dtype=torch.int8
        )
        if not bool((transitions <= 0).all()):
            message = "Target attention mask must use contiguous right padding"
            raise IrisTeacherForcingError(message)

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
                    (
                        owned_input_ids.device,
                        owned_input_ids.untyped_storage().data_ptr(),
                    ),
                    (
                        owned_attention_mask.device,
                        owned_attention_mask.untyped_storage().data_ptr(),
                    ),
                ),
            )
            source_storage = {
                (input_ids.device, input_ids.untyped_storage().data_ptr()),
                (
                    attention_mask.device,
                    attention_mask.untyped_storage().data_ptr(),
                ),
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
            message = f"Failed to capture exact {segment} token provenance"
            raise IrisTeacherForcingError(message) from error
        if len(private_storage) != len(private_tensors) or private_storage & source_storage:
            message = f"Exact {segment} token provenance must use independent storage"
            raise IrisTeacherForcingError(message)
        self._validate_token_provenance(provenance, segment=segment)
        return provenance

    @staticmethod
    def _validate_token_provenance(
        provenance: _TokenProvenance,
        *,
        segment: str,
    ) -> None:
        pairs = (
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
        )
        for name, actual, expected, expected_version, expected_storage in pairs:
            try:
                unchanged = torch.equal(actual, expected)
                actual_version = actual._version
                actual_storage = (
                    actual.device,
                    actual.untyped_storage().data_ptr(),
                )
            except Exception as error:
                message = f"Failed to verify exact {segment} {name} provenance"
                raise IrisTeacherForcingError(message) from error
            if (
                type(actual) is not torch.Tensor
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.device != expected.device
                or actual.is_inference()
                or actual.requires_grad
                or actual.grad_fn is not None
                or actual_version != expected_version
                or actual_storage != expected_storage
                or not unchanged
            ):
                message = f"Teacher-forcing stage mutated exact {segment} {name}"
                raise IrisTeacherForcingError(message)

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
            message = "Layer sampler returned invalid teacher-forcing context features"
            raise IrisTeacherForcingError(message)
        return context_features.shape[0], context_features.dtype, context_features.device

    def _validate_mode(self, *, expected_training: bool) -> None:
        if self.training is not expected_training:
            expected_name = "training" if expected_training else "evaluation"
            message = f"Teacher-forcing model must be in {expected_name} mode"
            raise IrisTeacherForcingError(message)
        for component_name, component in self._sidecar_modules:
            mismatched_child = next(
                (child for child in component.modules() if child.training is not expected_training),
                None,
            )
            if mismatched_child is not None:
                expected_name = "training" if expected_training else "evaluation"
                message = (
                    f"{component_name} and all children must be in {expected_name} mode "
                    "for teacher forcing"
                )
                raise IrisTeacherForcingError(message)

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
            message = "Teacher-forcing wiring is not stable"
            raise IrisTeacherForcingError(message) from error
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
        except Exception as error:
            message = f"Iris teacher forcing failed during {name}"
            raise IrisTeacherForcingError(message) from error

    def _validate_causal_output(
        self,
        output: object,
        *,
        total_length: int,
        expected_training: bool,
    ) -> None:
        if not isinstance(output, BackboneCausalOutput):
            message = "Backbone returned an invalid teacher-forced causal output"
            raise IrisTeacherForcingError(message)
        logits = output.logits
        expected_shape = (
            1,
            total_length,
            self._backbone_metadata.vocabulary_size,
        )
        if type(logits) is not torch.Tensor or tuple(logits.shape) != expected_shape:
            message = "Teacher-forced causal logits have an invalid shape"
            raise IrisTeacherForcingError(message)
        if (
            not torch.is_floating_point(logits)
            or logits.dtype != self._backbone_metadata.dtype
            or logits.device != self._backbone_metadata.device
            or not bool(torch.isfinite(logits).all())
        ):
            message = "Teacher-forced causal logits have an invalid dtype, device, or value"
            raise IrisTeacherForcingError(message)
        if logits.requires_grad is not expected_training or (
            expected_training and logits.grad_fn is None
        ):
            message = "Teacher-forced causal logits have an invalid gradient state"
            raise IrisTeacherForcingError(message)
        if output.past_key_values is not None:
            message = "Teacher-forced causal output must not contain cache state"
            raise IrisTeacherForcingError(message)
        loss = output.loss
        if type(loss) is not torch.Tensor or loss.ndim != 0:
            message = "Teacher-forced causal output must contain one scalar loss"
            raise IrisTeacherForcingError(message)
        if (
            not torch.is_floating_point(loss)
            or loss.device != self._backbone_metadata.device
            or not bool(torch.isfinite(loss))
        ):
            message = "Teacher-forced causal loss has an invalid dtype, device, or value"
            raise IrisTeacherForcingError(message)
        if loss.requires_grad is not expected_training or (
            expected_training and loss.grad_fn is None
        ):
            message = "Teacher-forced causal loss has an invalid gradient state"
            raise IrisTeacherForcingError(message)

    def _validate_output(
        self,
        output: IrisTeacherForcedOutput,
        *,
        expected_training: bool,
    ) -> None:
        if not isinstance(output, IrisTeacherForcedOutput):
            message = "Teacher-forced execution returned an invalid output"
            raise IrisTeacherForcingError(message)
        loss = output.loss
        if (
            type(loss) is not torch.Tensor
            or loss.ndim != 0
            or not torch.is_floating_point(loss)
            or loss.device != self._backbone_metadata.device
            or not bool(torch.isfinite(loss))
        ):
            message = "Teacher-forced output loss is invalid"
            raise IrisTeacherForcingError(message)
        if loss.requires_grad is not expected_training or (
            expected_training and loss.grad_fn is None
        ):
            message = "Teacher-forced output loss has an invalid gradient state"
            raise IrisTeacherForcingError(message)
        if type(output.supervised_token_count) is not int or output.supervised_token_count <= 0:
            message = "Teacher-forced supervised token count must be positive"
            raise IrisTeacherForcingError(message)
        if type(output.workspace_trace) is not tuple or len(output.workspace_trace) != self._cycles:
            message = "Teacher-forced workspace trace must contain exactly one state per cycle"
            raise IrisTeacherForcingError(message)
        if any(not isinstance(state, WorkspaceState) for state in output.workspace_trace):
            message = "Teacher-forced workspace trace contains an invalid state"
            raise IrisTeacherForcingError(message)
