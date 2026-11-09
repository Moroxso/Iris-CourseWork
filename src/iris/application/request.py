"""Inference-only orchestration for one Iris global-workspace request."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from re import fullmatch
from typing import TypeVar

import torch

from iris.ablations.cognitive import CognitiveAblationSessionV1
from iris.ablations.generation_controls import _GenerationWorkspaceControlSessionV1
from iris.application.errors import IrisRequestError
from iris.application.intervention_receipt import (
    IrisIntervenedRequestExecutionOutput,
    IrisIntervenedRequestExecutionReceiptV1,
)
from iris.application.request_sidecar_state import (
    IrisRequestSidecarInstallResultV1,
    install_request_sidecar_state_v1,
)
from iris.backbone import BackboneAdapter, BackboneFirstPassOutput
from iris.cognitive import (
    CognitiveModule,
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
from iris.generation.sampling import _clone_owned_generator
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.interventions.session import WorkspaceInterventionSessionV1
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.utils.hashing import (
    canonical_json_bytes,
    domain_sha256,
    module_state_sha256,
    tensor_sha256,
)
from iris.workspace import (
    LatentBridge,
    SlotAllocator,
    WorkspaceArbiter,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceState,
    WorkspaceWriter,
)
from iris.workspace.trace import WorkspaceTraceOptionsV1, WorkspaceTraceRecorderV1, WorkspaceTraceV1

_ResultT = TypeVar("_ResultT")
_EXECUTION_PROMPT_DOMAIN = "iris/request-execution/v1/prompt"
_COGNITIVE_ABLATION_PROMPT_DOMAIN = "iris/cognitive-ablation-prompt/v1"


def _baseline_request_only[**P, R](
    method: Callable[P, R],
) -> Callable[P, R]:
    """Intrinsic outer guard, including unbound calls; no overridable owner flag."""

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        receiver = args[0] if args else kwargs.get("self")
        if isinstance(receiver, _ScientificAblationOwnedRequest):
            raise IrisRequestError("Owned ablation request cannot use baseline execution/restore")
        return method(*args, **kwargs)

    return guarded


def _owned_core_boundary[**P, R](
    method: Callable[P, R],
) -> Callable[P, R]:
    """Consume explicit owned authorization BEFORE inference context or graph effects."""

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        receiver = args[0] if args else kwargs.get("self")
        if type(receiver) is _ScientificAblationControlledOwnedRequest:
            from iris.application._scientific_ablation_controlled_request_owner import (
                _consume_execution as consume_controlled,
            )

            controlled_positional = args[1:] if args else ()
            if len(controlled_positional) > 1:
                raise IrisRequestError("Controlled core requires one explicit prompt")
            controlled_owner = consume_controlled(
                receiver,
                kwargs.get("_authorization"),
                prompt=controlled_positional[0] if controlled_positional else kwargs.get("prompt"),
                strategy=kwargs.get("strategy"),
                generator=kwargs.get("generator"),
                session=kwargs.get("control"),
                ablation=kwargs.get("ablation"),
                intervention=kwargs.get("intervention"),
                diagnostics=kwargs.get("diagnostics"),
            )
            try:
                return method(*args, **kwargs)
            except BaseException:
                controlled_owner._poison()
                raise
        if kwargs.get("control") is not None:
            raise IrisRequestError("Distinct owned controlled request/capability required")
        if not isinstance(receiver, _ScientificAblationOwnedRequest):
            if kwargs.get("_authorization") is not None:
                raise IrisRequestError("Owned authorization cannot execute an ordinary request")
            return method(*args, **kwargs)
        from iris.application._scientific_ablation_request_owner import _consume_execution

        positional = args[1:] if args else ()
        if len(positional) > 1:
            raise IrisRequestError("Owned core requires one explicit prompt")
        owner = _consume_execution(
            receiver,
            kwargs.get("_authorization"),
            prompt=positional[0] if positional else kwargs.get("prompt"),
            strategy=kwargs.get("strategy"),
            generator=kwargs.get("generator"),
            session=kwargs.get("ablation"),
            intervention=kwargs.get("intervention"),
            diagnostics=kwargs.get("diagnostics"),
        )
        try:
            return method(*args, **kwargs)
        except BaseException:
            owner._poison()
            raise

    return guarded


@dataclass(frozen=True, slots=True)
class _PromptProvenance:
    """Owned version-tracked prompt metadata and immutable value snapshots."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    expected_input_ids: torch.Tensor
    expected_attention_mask: torch.Tensor
    expected_versions: tuple[int, int]
    expected_storage: tuple[tuple[torch.device, int], tuple[torch.device, int]]


@dataclass(frozen=True, slots=True)
class IrisRequestExecutionReceiptV1:
    """In-memory receipt binding delivery provenance to the executed request inputs."""

    prompt_sha256: str
    strategy: DecodingStrategy
    stop_token_ids: tuple[int, ...]
    sampling: TemperatureTopPSamplingConfig
    sampling_seed: int | None

    def __post_init__(self) -> None:
        if (
            type(self.prompt_sha256) is not str
            or fullmatch(r"[0-9a-f]{64}", self.prompt_sha256) is None
        ):
            message = "execution receipt prompt SHA-256 must be lowercase hexadecimal"
            raise ValueError(message)
        if type(self.strategy) is not str or self.strategy not in {"greedy", "sampling"}:
            message = "execution receipt strategy must be greedy or sampling"
            raise ValueError(message)
        if (
            type(self.stop_token_ids) is not tuple
            or any(type(token_id) is not int or token_id < 0 for token_id in self.stop_token_ids)
            or len(set(self.stop_token_ids)) != len(self.stop_token_ids)
        ):
            message = "execution receipt stop-token IDs must be unique non-negative integers"
            raise ValueError(message)
        if type(self.sampling) is not TemperatureTopPSamplingConfig:
            message = "execution receipt sampling policy must be exact"
            raise TypeError(message)
        if self.strategy == "greedy" and self.sampling_seed is not None:
            message = "greedy execution receipt cannot contain a sampling seed"
            raise ValueError(message)
        if self.strategy == "sampling" and (
            type(self.sampling_seed) is not int
            or self.sampling_seed < 0
            or self.sampling_seed > 2**63 - 1
        ):
            message = "sampling execution receipt requires a non-negative 63-bit seed"
            raise ValueError(message)


def _capture_request_execution_receipt_v1(
    *,
    prompt: str,
    strategy: DecodingStrategy,
    generator: torch.Generator | None,
    iris_generator: IrisGenerator,
) -> IrisRequestExecutionReceiptV1:
    """Capture exact execution inputs from the graph that consumed them."""
    if type(prompt) is not str:
        message = "execution receipt prompt must be an exact string"
        raise TypeError(message)
    prompt_bytes = prompt.encode("utf-8", errors="strict")
    sampling_seed: int | None = None
    if strategy == "sampling":
        if not isinstance(generator, torch.Generator):
            message = "sampling execution receipt requires the used torch.Generator"
            raise TypeError(message)
        sampling_seed = generator.initial_seed()
    return IrisRequestExecutionReceiptV1(
        prompt_sha256=domain_sha256(_EXECUTION_PROMPT_DOMAIN, prompt_bytes),
        strategy=strategy,
        stop_token_ids=iris_generator.stop_token_ids,
        sampling=iris_generator.sampling,
        sampling_seed=sampling_seed,
    )


@dataclass(frozen=True, slots=True)
class IrisRequestExecutionOutput:
    """One generation plus exact first-pass token provenance for delivery."""

    generation: IrisGeneratorOutput
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    receipt: IrisRequestExecutionReceiptV1 | None = None
    diagnostic_trace: WorkspaceTraceV1 | None = None


@dataclass(frozen=True, slots=True)
class _CoreRequestExecutionOutput:
    """Neutral baseline/intervention/ablation facts; deliberately no public receipt."""

    generation: IrisGeneratorOutput
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    diagnostic_trace: WorkspaceTraceV1 | None = None


class IrisRequestOrchestrator:
    """Run one frozen prompt through the complete Iris-GW inference graph."""

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
        prefix_assembler: SecondPassGenerationPrefixAssembler,
        iris_generator: IrisGenerator,
    ) -> None:
        if not isinstance(cognitive_config, CognitiveConfig):
            message = "cognitive_config must be a validated CognitiveConfig"
            raise TypeError(message)
        if not isinstance(workspace_config, WorkspaceConfig):
            message = "workspace_config must be a validated WorkspaceConfig"
            raise TypeError(message)
        if workspace_config.winners_per_cycle > cognitive_config.num_modules:
            message = "winners_per_cycle cannot exceed cognitive num_modules"
            raise ValueError(message)

        self._backbone = backbone
        self._cognitive_config = cognitive_config
        self._workspace_config = workspace_config
        self._cognitive_config_snapshot = cognitive_config.model_dump_json()
        self._workspace_config_snapshot = workspace_config.model_dump_json()
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
        self._layer_sampler = layer_sampler
        self._goal_encoder = goal_encoder
        self._module_input_router = module_input_router
        self._workspace_reader = workspace_reader
        self._cognitive_modules = cognitive_modules
        self._proposal_head = proposal_head
        self._workspace_arbiter = workspace_arbiter
        self._slot_allocator = slot_allocator
        self._workspace_writer = workspace_writer
        self._workspace_decay = workspace_decay
        self._latent_bridge = latent_bridge
        self._prefix_assembler = prefix_assembler
        self._iris_generator = iris_generator
        self._sidecar_modules = (
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
        self._validate_wiring()
        self._sidecar_state_sha256 = module_state_sha256(self._sidecar_modules)

    @property
    def backbone(self) -> BackboneAdapter:
        """Return the exact frozen backbone owned by the arm-D request graph."""
        return self._backbone

    @property
    def sidecar_state_sha256(self) -> str:
        """Return the bootstrap sidecar fingerprint after proving it is unchanged."""
        try:
            self._validate_runtime_boundary()
            current = module_state_sha256(self._sidecar_modules)
        except Exception as error:
            message = "Failed to verify Iris sidecar state provenance"
            raise IrisRequestError(message) from error
        if current != self._sidecar_state_sha256:
            message = "Iris sidecar state changed after request composition"
            raise IrisRequestError(message)
        return self._sidecar_state_sha256

    @_baseline_request_only
    def restore_teacher_sidecar_state_v1(
        self,
        state: dict[str, torch.Tensor],
        *,
        expected_installed_sha256: str | None = None,
    ) -> IrisRequestSidecarInstallResultV1:
        """Transactionally replace D request tensors from one validated checkpoint."""
        self._validate_runtime_boundary()
        result = install_request_sidecar_state_v1(
            modules=self._sidecar_modules,
            state=state,
            expected_current_sha256=self._sidecar_state_sha256,
            expected_installed_sha256=expected_installed_sha256,
        )
        self._sidecar_state_sha256 = result.installed_state_sha256
        return result

    @_baseline_request_only
    def generate(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        """Generate one response and preserve all post-cycle workspace states."""
        return self.execute(
            prompt,
            strategy=strategy,
            generator=generator,
        ).generation

    @_baseline_request_only
    @torch.inference_mode()
    def execute(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisRequestExecutionOutput:
        """Execute once and retain exact token provenance for artifact delivery."""
        result = self._execute_core(
            prompt, strategy=strategy, generator=generator, diagnostics=diagnostics
        )
        return IrisRequestExecutionOutput(
            generation=result.generation,
            prompt_input_ids=result.prompt_input_ids,
            prompt_attention_mask=result.prompt_attention_mask,
            receipt=_capture_request_execution_receipt_v1(
                prompt=prompt,
                strategy=strategy,
                generator=generator,
                iris_generator=self._iris_generator,
            ),
            diagnostic_trace=result.diagnostic_trace,
        )

    @_baseline_request_only
    @torch.inference_mode()
    def execute_intervened(
        self,
        prompt: str,
        *,
        plan: WorkspaceInterventionPlanV1,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisIntervenedRequestExecutionOutput:
        """Run an explicitly labelled counterfactual with a fresh owned session.

        Existing baseline artifact/capture writers intentionally cannot consume
        this output type. An empty plan is the paired numerical control, but still
        has the explicit intervention receipt and phase-semantics identity.
        """
        self._validate_request(prompt, strategy=strategy, generator=generator)
        try:
            session = WorkspaceInterventionSessionV1(
                plan,
                num_slots=self._num_slots,
                num_modules=self._num_modules,
                state_dim=self._state_dim,
                cycles=self._cycles,
                winners_per_cycle=self._winners_per_cycle,
            )
            sidecar_sha256 = self.sidecar_state_sha256
            local_generator = (
                _clone_owned_generator(generator, expected_device=self._backbone.device)
                if generator is not None
                else None
            )
            initial_rng_state = (
                local_generator.get_state().detach().clone()
                if local_generator is not None
                else None
            )
            sampling_seed = local_generator.initial_seed() if local_generator is not None else None
            sampling_policy = self._iris_generator.sampling
            if type(sampling_policy) is not TemperatureTopPSamplingConfig:
                raise TypeError("Intervened execution requires an exact sampling policy")
            receipt = IrisIntervenedRequestExecutionReceiptV1(
                prompt_sha256=domain_sha256(
                    _EXECUTION_PROMPT_DOMAIN, prompt.encode("utf-8", errors="strict")
                ),
                strategy=strategy,
                stop_token_ids=self._iris_generator.stop_token_ids,
                sampling=TemperatureTopPSamplingConfig(
                    temperature=sampling_policy.temperature, top_p=sampling_policy.top_p
                ),
                sampling_seed=sampling_seed,
                sampling_rng_state_sha256=(
                    tensor_sha256(
                        initial_rng_state, domain="iris/intervened-request/sampling-rng/v1"
                    )
                    if initial_rng_state is not None
                    else None
                ),
                intervention_plan_sha256=session.plan_sha256,
                graph_sha256=domain_sha256(
                    "iris/intervened-request-graph/v1",
                    canonical_json_bytes(
                        {
                            "cognitive": self._cognitive_config.model_dump(mode="json"),
                            "workspace": self._workspace_config.model_dump(mode="json"),
                        }
                    ),
                ),
                sidecar_state_sha256=sidecar_sha256,
                cycles=self._cycles,
            )
            result = self._execute_core(
                prompt,
                strategy=strategy,
                generator=local_generator,
                intervention=session,
                diagnostics=diagnostics,
            )
            if self.sidecar_state_sha256 != sidecar_sha256:
                raise IrisRequestError("Sidecar weights changed during intervened execution")
            if local_generator is not None and local_generator.initial_seed() != sampling_seed:
                raise IrisRequestError(
                    "Sampling generator was reseeded during intervened execution"
                )
            if (
                type(self._iris_generator.sampling) is not TemperatureTopPSamplingConfig
                or self._iris_generator.sampling != receipt.sampling
                or self._iris_generator.stop_token_ids != receipt.stop_token_ids
            ):
                raise IrisRequestError("Decoding policy changed during intervened execution")
            output = IrisIntervenedRequestExecutionOutput(
                generation=result.generation,
                prompt_input_ids=result.prompt_input_ids,
                prompt_attention_mask=result.prompt_attention_mask,
                receipt=receipt,
                diagnostic_trace=result.diagnostic_trace,
            )
            if generator is not None:
                assert local_generator is not None and initial_rng_state is not None
                if not torch.equal(generator.get_state(), initial_rng_state):
                    raise IrisRequestError(
                        "Caller sampling RNG changed outside the intervened request transaction"
                    )
                generator.set_state(local_generator.get_state().detach().clone())
        except Exception as error:
            raise IrisRequestError("Intervened Iris request failed") from error
        return output

    @_owned_core_boundary
    @torch.inference_mode()
    def _execute_core(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        intervention: WorkspaceInterventionSessionV1 | None = None,
        ablation: CognitiveAblationSessionV1 | None = None,
        control: _GenerationWorkspaceControlSessionV1 | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
        _authorization: object | None = None,
    ) -> _CoreRequestExecutionOutput:
        """Neutral private graph; explicit cognitive hooks never mint a baseline receipt.

        Public generate/execute remain baseline-only. A mode-bound public runtime
        still requires its separate owned request/capability/receipt transaction.
        """
        self._validate_request(prompt, strategy=strategy, generator=generator)
        if ablation is not None:
            if type(ablation) is not CognitiveAblationSessionV1 or intervention is not None:
                raise IrisRequestError(
                    "Exact cognitive session cannot mix with legacy intervention"
                )
            try:
                ablation._validate_start(
                    prompt_sha256=domain_sha256(
                        _COGNITIVE_ABLATION_PROMPT_DOMAIN, prompt.encode("utf-8")
                    ),
                    num_modules=self._num_modules,
                    num_slots=self._num_slots,
                    state_dim=self._state_dim,
                    cycles=self._cycles,
                )
            except Exception as error:
                raise IrisRequestError(
                    "Cognitive ablation session boundary is not ready"
                ) from error
        if control is not None:
            if (
                type(self) is not _ScientificAblationControlledOwnedRequest
                or type(control) is not _GenerationWorkspaceControlSessionV1
                or ablation is not None
                or intervention is not None
            ):
                raise IrisRequestError("Distinct controlled core cannot mix legacy sessions")
            from iris.application._scientific_ablation_controlled_request_owner import (
                _phase_binding,
            )

            control.claim(binding=_phase_binding(self, prompt))
            ablation, intervention = control.ablation, control.workspace
        diagnostic = WorkspaceTraceRecorderV1(
            diagnostics,
            num_modules=self._num_modules,
            num_slots=self._num_slots,
            state_dim=self._state_dim,
            cycles=self._cycles,
            intervention_plan_sha256=control.plan_sha256
            if control is not None
            else ablation.policy_sha256
            if ablation is not None
            else None
            if intervention is None
            else intervention.plan_sha256,
        )
        try:
            self._validate_runtime_boundary()
        except Exception as error:
            message = f"Iris request runtime boundary is not ready: {error}"
            raise IrisRequestError(message) from error

        first_pass = self._run_stage(
            "frozen first pass",
            lambda: self._backbone.forward_prompt(prompt),
        )
        if not isinstance(first_pass, BackboneFirstPassOutput):
            message = "Backbone did not return first-pass token provenance"
            raise IrisRequestError(message)
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
            lambda: self._goal_encoder(
                sampled.context_features,
                sampled.attention_mask,
            ),
            provenance=provenance,
        )
        if ablation is not None:
            goal = self._run_stage(
                "cognitive ablation goal",
                partial(ablation.after_goal, goal),
                provenance=provenance,
            )
        routed = self._run_stage(
            "module input routing",
            lambda: (
                self._module_input_router(sampled.context_features, sampled.attention_mask)
                if ablation is None
                else ablation.route(
                    self._module_input_router, sampled.context_features, sampled.attention_mask
                )
            ),
            provenance=provenance,
        )

        batch_size, state_dtype, state_device = self._run_stage(
            "sampled context metadata validation",
            lambda: self._context_metadata(sampled.context_features),
            provenance=provenance,
        )
        module_states = self._run_stage(
            "cognitive state initialization",
            lambda: self._cognitive_modules.initial_state(
                batch_size=batch_size,
                dtype=state_dtype,
                device=state_device,
            ),
            provenance=provenance,
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
            provenance=provenance,
        )

        trace_states: list[WorkspaceState] = []
        for cycle_index in range(self._cycles):
            cycle_number = cycle_index + 1
            if cycle_index > 0:
                workspace = self._run_stage(
                    f"workspace decay before cycle {cycle_number}",
                    partial(self._workspace_decay, workspace),
                    provenance=provenance,
                )
            if intervention is not None:
                workspace = self._run_stage(
                    f"intervention before read for cycle {cycle_number}",
                    partial(intervention.before_read, workspace, cycle_number),
                    provenance=provenance,
                )
            read_state = workspace
            reads = self._run_stage(
                f"workspace read for cycle {cycle_number}",
                partial(self._workspace_reader, workspace),
                provenance=provenance,
            )
            if ablation is not None:
                reads = self._run_stage(
                    f"cognitive ablation broadcast for cycle {cycle_number}",
                    partial(ablation.after_read, reads, cycle_number),
                    provenance=provenance,
                )
            if intervention is not None:
                reads = self._run_stage(
                    f"intervention after read for cycle {cycle_number}",
                    partial(intervention.after_read, reads, cycle_number),
                    provenance=provenance,
                )
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
                f"cognitive state extraction for cycle {cycle_number}",
                partial(getattr, module_output, "module_states"),
                provenance=provenance,
            )
            proposal = self._run_stage(
                f"proposal head for cycle {cycle_number}",
                partial(self._proposal_head, module_states),
                provenance=provenance,
            )
            if intervention is not None:
                proposal = self._run_stage(
                    f"intervention after proposal for cycle {cycle_number}",
                    partial(intervention.after_proposal, proposal, cycle_number),
                    provenance=provenance,
                )
            arbitration = self._run_stage(
                f"workspace arbitration for cycle {cycle_number}",
                partial(
                    self._workspace_arbiter,
                    proposal,
                    workspace,
                    goal.goal_state,
                ),
                provenance=provenance,
            )
            if intervention is not None:
                arbitration = self._run_stage(
                    f"intervention after arbitration for cycle {cycle_number}",
                    partial(intervention.after_arbitration, arbitration, cycle_number),
                    provenance=provenance,
                )
            allocation = self._run_stage(
                f"slot allocation for cycle {cycle_number}",
                partial(self._slot_allocator, proposal, workspace, arbitration),
                provenance=provenance,
            )
            if intervention is not None:
                allocation = self._run_stage(
                    f"intervention after allocation for cycle {cycle_number}",
                    partial(intervention.after_allocation, workspace, allocation, cycle_number),
                    provenance=provenance,
                )
            before_write = workspace
            workspace = self._run_stage(
                f"workspace write for cycle {cycle_number}",
                partial(
                    self._workspace_writer,
                    proposal,
                    workspace,
                    allocation,
                    goal.goal_state,
                ),
                provenance=provenance,
            )
            if intervention is not None:
                workspace = self._run_stage(
                    f"intervention after write for cycle {cycle_number}",
                    partial(
                        intervention.after_write, before_write, workspace, allocation, cycle_number
                    ),
                    provenance=provenance,
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
                    provenance=provenance,
                )
            trace_states.append(workspace)

        if intervention is not None:
            self._run_stage("intervention completion", intervention.finish, provenance=provenance)
        workspace_trace = tuple(trace_states)
        if ablation is not None:
            self._run_stage("cognitive ablation completion", ablation.finish, provenance=provenance)
        if control is not None:
            self._run_stage(
                "controlled generation phase completion", control.complete, provenance=provenance
            )
        prompt_embeddings = self._run_stage(
            "exact prompt token embedding",
            lambda: self._backbone.embed_tokens(provenance.input_ids),
            provenance=provenance,
        )
        latent = self._run_stage(
            "workspace latent bridge",
            lambda: self._latent_bridge(
                workspace_trace[-1],
                provenance.attention_mask,
            ),
            provenance=provenance,
        )
        prefix = self._run_stage(
            "second-pass generation prefix",
            lambda: self._prefix_assembler(
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=provenance.attention_mask,
                workspace=latent,
            ),
            provenance=provenance,
        )
        try:
            self._validate_runtime_boundary(provenance=provenance)
        except Exception as error:
            message = f"Final Iris request boundary validation failed: {error}"
            raise IrisRequestError(message) from error
        try:
            generation = self._iris_generator.generate(
                prefix,
                strategy=strategy,
                workspace_trace=workspace_trace,
                generator=generator,
            )
        except Exception as error:
            message = "Iris text generation failed"
            raise IrisRequestError(message) from error
        self._validate_prompt_provenance(provenance)
        return _CoreRequestExecutionOutput(
            generation=generation,
            prompt_input_ids=provenance.input_ids,
            prompt_attention_mask=provenance.attention_mask,
            diagnostic_trace=diagnostic.finish(),
        )

    def _validate_wiring(self) -> None:
        try:
            configuration_unchanged = (
                self._cognitive_config.model_dump_json() == self._cognitive_config_snapshot
                and self._workspace_config.model_dump_json() == self._workspace_config_snapshot
            )
        except Exception as error:
            message = "Failed to validate request configuration snapshot"
            raise ValueError(message) from error
        if not configuration_unchanged:
            message = "Validated request configuration changed after construction"
            raise ValueError(message)
        try:
            backbone_hidden_size = self._backbone.hidden_size
            backbone_context = self._backbone.max_position_embeddings
            backbone_prompt_limit = self._backbone.max_prompt_tokens
            backbone_generation_limit = self._backbone.max_new_tokens
        except Exception as error:
            message = "Failed to read backbone dimensions for request wiring"
            raise ValueError(message) from error
        for name, value in {
            "hidden_size": backbone_hidden_size,
            "max_position_embeddings": backbone_context,
            "max_prompt_tokens": backbone_prompt_limit,
            "max_new_tokens": backbone_generation_limit,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"backbone {name} must be a positive integer"
                raise ValueError(message)
        try:
            generator_backbone = getattr(self._iris_generator, "backbone", None)
        except Exception as error:
            message = "Failed to read iris_generator.backbone for request wiring"
            raise ValueError(message) from error
        if generator_backbone is not self._backbone:
            message = "iris_generator must own the exact request backbone adapter"
            raise ValueError(message)

        expected_components: tuple[tuple[str, object, dict[str, object]], ...] = (
            (
                "layer_sampler",
                self._layer_sampler,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": backbone_hidden_size,
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
                "workspace_reader",
                self._workspace_reader,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
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
                "workspace_arbiter",
                self._workspace_arbiter,
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
                self._slot_allocator,
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
                self._workspace_writer,
                {
                    "state_dim": self._state_dim,
                    "num_modules": self._num_modules,
                    "num_slots": self._num_slots,
                    "winners_per_cycle": self._winners_per_cycle,
                },
            ),
            (
                "workspace_decay",
                self._workspace_decay,
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
                self._latent_bridge,
                {
                    "state_dim": self._state_dim,
                    "backbone_hidden_size": backbone_hidden_size,
                    "num_slots": self._num_slots,
                },
            ),
            (
                "prefix_assembler",
                self._prefix_assembler,
                {
                    "hidden_size": backbone_hidden_size,
                    "num_slots": self._num_slots,
                    "max_prompt_tokens": backbone_prompt_limit,
                    "max_new_tokens": backbone_generation_limit,
                    "max_context_tokens": backbone_context,
                },
            ),
            (
                "iris_generator",
                self._iris_generator,
                {"max_new_tokens": backbone_generation_limit},
            ),
        )
        for component_name, component, expected_attributes in expected_components:
            for attribute, expected in expected_attributes.items():
                try:
                    actual = getattr(component, attribute, None)
                    matches_expected = self._same_typed_value(actual, expected)
                except Exception as error:
                    message = (
                        f"Failed to read or compare {component_name}.{attribute} for request wiring"
                    )
                    raise ValueError(message) from error
                if not matches_expected:
                    message = (
                        f"{component_name}.{attribute}={actual!r} does not match "
                        f"request configuration {expected!r}"
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
                cls._same_typed_value(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
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
            raise IrisRequestError(message)
        if type(strategy) is not str or strategy not in {"greedy", "sampling"}:
            message = "strategy must be either 'greedy' or 'sampling'"
            raise IrisRequestError(message)
        if strategy == "greedy" and generator is not None:
            message = "greedy requests do not accept a random generator"
            raise IrisRequestError(message)
        if strategy == "sampling" and not isinstance(generator, torch.Generator):
            message = "sampling requests require an explicit torch.Generator"
            raise IrisRequestError(message)
        if strategy == "sampling" and isinstance(generator, torch.Generator):
            try:
                is_global_default = generator is torch.default_generator or any(
                    generator is default_generator
                    for default_generator in torch.cuda.default_generators
                )
            except Exception as error:
                message = "failed to validate sampling generator ownership"
                raise IrisRequestError(message) from error
            if is_global_default:
                message = "sampling requests require a caller-owned generator"
                raise IrisRequestError(message)

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
                    (
                        attention_mask.device,
                        attention_mask.untyped_storage().data_ptr(),
                    ),
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
            message = "Failed to capture exact first-pass prompt provenance"
            raise IrisRequestError(message) from error
        if len(storage_identities) != 6:
            message = "Request prompt provenance tensors must use independent storage"
            raise IrisRequestError(message)
        self._validate_prompt_provenance(provenance)
        return provenance

    @staticmethod
    def _validate_prompt_provenance(provenance: _PromptProvenance) -> None:
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
                message = f"Failed to verify request prompt {name} provenance"
                raise IrisRequestError(message) from error
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
                message = f"Request stage mutated exact first-pass {name}"
                raise IrisRequestError(message)

    def _context_metadata(
        self,
        context_features: torch.Tensor,
    ) -> tuple[int, torch.dtype, torch.device]:
        if (
            not isinstance(context_features, torch.Tensor)
            or context_features.ndim != 3
            or context_features.shape[0] != 1
            or context_features.shape[1] <= 0
            or context_features.shape[2] != self._state_dim
            or not torch.is_floating_point(context_features)
            or not bool(torch.isfinite(context_features).all())
        ):
            message = "Layer sampler returned invalid request context features"
            raise IrisRequestError(message)
        return context_features.shape[0], context_features.dtype, context_features.device

    def _validate_inference_modes(self) -> None:
        for component_name, component in self._sidecar_modules:
            if not isinstance(component, torch.nn.Module):
                message = f"{component_name} must be a torch.nn.Module"
                raise IrisRequestError(message)
            trained_submodule = next(
                (module for module in component.modules() if module.training),
                None,
            )
            if trained_submodule is not None:
                message = (
                    f"{component_name} and all of its children must be in evaluation mode "
                    "for request generation"
                )
                raise IrisRequestError(message)

    def _validate_runtime_boundary(
        self,
        *,
        provenance: _PromptProvenance | None = None,
    ) -> None:
        self._validate_wiring()
        if provenance is not None:
            self._validate_prompt_provenance(provenance)
        self._validate_inference_modes()

    def _run_stage(
        self,
        name: str,
        operation: Callable[[], _ResultT],
        *,
        provenance: _PromptProvenance | None = None,
    ) -> _ResultT:
        try:
            self._validate_runtime_boundary(provenance=provenance)
            result = operation()
            self._validate_runtime_boundary(provenance=provenance)
            return result
        except Exception as error:
            message = f"Iris request failed during {name}"
            raise IrisRequestError(message) from error


class _ScientificAblationOwnedRequest(IrisRequestOrchestrator):
    """Intrinsic private owned subtype; never handed out by the baseline factory.

    Slots stay absent until exclusive attachment. Missing/corrupt slots NEVER
    become a baseline fallback. This is not a Python introspection sandbox.
    """

    __slots__ = ("_ablation_anchor", "_ablation_binding")
    _ablation_anchor: object
    _ablation_binding: object

    def _install_owned_sidecar(
        self, state: dict[str, torch.Tensor], *, _authorization: object
    ) -> IrisRequestSidecarInstallResultV1:
        from iris.application._scientific_ablation_request_owner import (
            _consume_bootstrap,
            _raise_bootstrap_failure,
        )

        owner = _consume_bootstrap(self, _authorization)
        try:
            self._validate_runtime_boundary()
            result = install_request_sidecar_state_v1(
                modules=self._sidecar_modules,
                state=state,
                expected_current_sha256=self._sidecar_state_sha256,
                expected_installed_sha256=owner._binding.sidecar_sha256,
            )
            self._sidecar_state_sha256 = result.installed_state_sha256
            owner._complete_bootstrap(_authorization)
            return result
        except BaseException as error:
            owner._poison()
            _raise_bootstrap_failure(error)


class _ScientificAblationControlledOwnedRequest(_ScientificAblationOwnedRequest):
    """Distinct fresh subtype; inherited intrinsic baseline denials stay active."""

    __slots__ = ("_controlled_anchor", "_controlled_binding")
    _controlled_anchor: object
    _controlled_binding: object

    def _install_owned_sidecar(
        self, state: dict[str, torch.Tensor], *, _authorization: object
    ) -> IrisRequestSidecarInstallResultV1:
        from iris.application._scientific_ablation_controlled_request_owner import (
            _consume_bootstrap,
            _raise_bootstrap_failure,
        )

        owner = _consume_bootstrap(self, _authorization)
        try:
            self._validate_runtime_boundary()
            result = install_request_sidecar_state_v1(
                modules=self._sidecar_modules,
                state=state,
                expected_current_sha256=self._sidecar_state_sha256,
                expected_installed_sha256=owner._binding.sidecar_sha256,
            )
            self._sidecar_state_sha256 = result.installed_state_sha256
            owner._complete_bootstrap(_authorization)
            return result
        except BaseException as error:
            owner._poison()
            _raise_bootstrap_failure(error)
