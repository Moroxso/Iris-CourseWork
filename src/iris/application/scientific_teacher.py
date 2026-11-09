"""Row-bound five-term arm-D teacher objective with registered diagnostic heads.

This new application graph composes the historical teacher and target preparer.
It does not reuse the old LM-only optimizer/checkpoint/run meaning.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.data.schemas import IrisExample
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.training.intervention import IrisIntervenedTeacherOutputV1
from iris.training.losses import (
    IrisLossBreakdown,
    IrisLossWeights,
    combine_iris_losses,
    uniform_utilization_loss,
    workspace_diversity_loss,
    workspace_norm_loss,
)
from iris.training.model import IrisTeacherForcingModel
from iris.training.outputs import IrisTeacherTracedOutputV1
from iris.training.probes import (
    IrisProbeTargets,
    IrisProbeVocabulary,
    WorkspaceProbeHeads,
    prepare_probe_targets,
    workspace_probe_objective,
)
from iris.training.usage import IrisWorkspaceUsageV1, aggregate_workspace_usage_v1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, module_state_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from iris.workspace.trace import WorkspaceTraceOptionsV1


class IrisScientificTeacherError(RuntimeError):
    """Scientific row, objective or registered teacher ownership is incompatible."""


class IrisScientificObjectivePolicyV1(BaseModel):
    """Explicit per-example objective; changes require a distinct future run identity."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )
    kind: Literal["iris-scientific-objective-v1"] = "iris-scientific-objective-v1"
    lm_reduction: Literal["sum_attended_target_tokens"] = "sum_attended_target_tokens"
    probe_head_reduction: Literal["sum"] = "sum"
    workspace_reduction: Literal["final_post_write_state"] = "final_post_write_state"
    utilization_estimator: Literal["cycle_mean_soft_full_slot_mixture_v1"] = (
        "cycle_mean_soft_full_slot_mixture_v1"
    )
    diversity_margin: float = Field(default=0.30, ge=-1.0, le=1.0)
    weights: IrisLossWeights = Field(default_factory=IrisLossWeights)


@dataclass(frozen=True, slots=True)
class IrisScientificTeacherOutputV1:
    """One owned row's graph-attached objective, not a persisted training receipt."""

    example_id: str
    example_sha256: str
    configuration_sha256: str
    supervised_token_count: int
    losses: IrisLossBreakdown
    usage: IrisWorkspaceUsageV1
    trace: IrisTeacherTracedOutputV1


def _objective_from_trace_v1(
    trace: IrisTeacherTracedOutputV1,
    *,
    cycles: int,
    probes: WorkspaceProbeHeads,
    labels: IrisProbeTargets,
    policy: IrisScientificObjectivePolicyV1,
) -> tuple[IrisLossBreakdown, IrisWorkspaceUsageV1]:
    """Shared unchanged five-term arithmetic; no graph identity or receipt creation."""
    if len(trace.cycles) != cycles or len(trace.teacher.workspace_trace) != cycles:
        raise IrisScientificTeacherError("Incomplete scientific teacher trace")
    usage = aggregate_workspace_usage_v1(trace.cycles)
    final_slots = trace.teacher.workspace_trace[-1].slots
    with torch.autocast(device_type=final_slots.device.type, enabled=False):
        # The legacy adapter's loss is target-token MEAN; scientific policy is SUM.
        lm_sum = trace.teacher.loss * trace.teacher.supervised_token_count
    terms = combine_iris_losses(
        language_model_loss=lm_sum,
        probe_loss=workspace_probe_objective(
            probes, final_slots, labels, head_reduction=policy.probe_head_reduction
        ),
        diversity_loss=workspace_diversity_loss(final_slots, margin=policy.diversity_margin),
        utilization_loss=uniform_utilization_loss(
            usage.module_probabilities, usage.slot_probabilities
        ),
        norm_loss=workspace_norm_loss(final_slots),
        weights=policy.weights,
    )
    return terms, usage


@dataclass(frozen=True, slots=True)
class IrisScientificIntervenedTeacherOutputV1:
    """Row-bound actual intervened NLL sum, not the five-term baseline objective."""

    example_id: str
    example_sha256: str
    configuration_sha256: str
    target_nll_sum: torch.Tensor
    teacher: IrisIntervenedTeacherOutputV1

    @property
    def supervised_token_count(self) -> int:
        return self.teacher.supervised_token_count


class IrisScientificTeacherV1(nn.Module):
    """Own the D sidecar plus probes; the frozen backbone remains unregistered.

    Only a complete IrisExample is accepted. The same owned row supplies the LM
    prompt/full answer and all four labels; callers cannot pair unrelated labels
    or target tensors with a workspace. Training rejects held-out rows. This is
    a microbatch-one objective, not yet token accumulation, AMP or a managed run.
    """

    configuration_identity: torch.Tensor

    def __init__(
        self,
        *,
        teacher: IrisTeacherForcingModel,
        vocabulary: IrisProbeVocabulary,
        probe_init_seed: int,
        policy: IrisScientificObjectivePolicyV1,
    ) -> None:
        super().__init__()
        if type(teacher) is not IrisTeacherForcingModel:
            raise IrisScientificTeacherError("An exact arm-D teacher is required")
        if type(policy) is not IrisScientificObjectivePolicyV1:
            raise IrisScientificTeacherError("An explicit exact objective policy is required")
        self._policy = IrisScientificObjectivePolicyV1.model_validate(
            policy.model_dump(mode="python", warnings=False)
        )
        provenance = teacher.checkpoint_provenance
        heads = WorkspaceProbeHeads(
            vocabulary=vocabulary,
            state_dim=teacher.cognitive_modules.state_dim,
            pooling="mean_all_slots",
            init_seed=probe_init_seed,
            dtype=teacher.backbone.dtype,
            device=teacher.backbone.device,
        )
        configuration = {
            "kind": "iris-scientific-teacher-v1",
            "arm_id": "D",
            "policy": self._policy.model_dump(mode="json"),
            "probe_configuration_sha256": heads.configuration_sha256,
            "backbone_configuration_json": provenance.backbone_configuration_json,
            "cognitive_configuration_json": provenance.cognitive_configuration_json,
            "workspace_configuration_json": provenance.workspace_configuration_json,
            "hidden_size": provenance.hidden_size,
            "num_hidden_layers": provenance.num_hidden_layers,
            "vocabulary_size": provenance.vocabulary_size,
            "max_position_embeddings": provenance.max_position_embeddings,
            "device": provenance.device,
            "dtype": provenance.dtype,
        }
        self._policy_json = self._policy.model_dump_json()
        self._configuration_json = canonical_json_bytes(configuration).decode("utf-8")
        self._configuration_sha256 = domain_sha256(
            "iris/scientific-teacher-configuration/v1", canonical_json_bytes(configuration)
        )
        self.teacher = teacher
        self.probes = heads
        self._component_ids = (id(teacher), id(heads))
        self._parameter_identity = tuple(
            (name, id(value)) for name, value in self.named_parameters()
        )
        self._cycles = teacher.proposal_head.cycles
        self.register_buffer(
            "configuration_identity",
            torch.tensor(
                list(bytes.fromhex(self._configuration_sha256)),
                dtype=torch.uint8,
                device=teacher.backbone.device,
            ),
        )
        self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
            self._check_loading_identity
        )
        self._validate_boundary(training=True)

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def configuration_json(self) -> str:
        """Complete immutable scientific graph policy for checkpoint reconstruction."""
        if (
            domain_sha256(
                "iris/scientific-teacher-configuration/v1", self._configuration_json.encode("utf-8")
            )
            != self.configuration_sha256
        ):
            raise IrisScientificTeacherError("Scientific configuration snapshot changed")
        return self._configuration_json

    @property
    def policy(self) -> IrisScientificObjectivePolicyV1:
        return IrisScientificObjectivePolicyV1.model_validate_json(self._policy_json)

    @staticmethod
    def _matches_identity(value: object, expected: str) -> bool:
        return (
            type(value) is torch.Tensor
            and value.dtype == torch.uint8
            and value.layout == torch.strided
            and tuple(value.shape) == (32,)
            and value.device.type in ("cpu", "cuda")
            and bytes(value.detach().cpu().tolist()).hex() == expected
        )

    def _check_loading_identity(
        self,
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Check both root and probe identity before *any* teacher child is copied.
        # This is not a transactional replacement for a future full checkpoint IO.
        if (
            module is not self
            or not self._matches_identity(
                state_dict.get(prefix + "configuration_identity"), self._configuration_sha256
            )
            or not self._matches_identity(
                state_dict.get(prefix + "probes.configuration_identity"),
                self.probes.configuration_sha256,
            )
        ):
            raise IrisScientificTeacherError(
                "Scientific objective/probe checkpoint identity differs"
            )

    def _validate_boundary(self, *, training: bool) -> None:
        if (
            torch.is_inference_mode_enabled()
            or (training and not torch.is_grad_enabled())
            or any(module.training is not training for module in self.modules())
            or torch.is_autocast_enabled(self.teacher.backbone.device.type)
        ):
            raise IrisScientificTeacherError(
                "Explicit train/eval mode without ambient AMP required"
            )
        if (
            tuple(self._modules) != ("teacher", "probes")
            or (id(self.teacher), id(self.probes)) != self._component_ids
            or self._policy.model_dump_json() != self._policy_json
            or not self._matches_identity(self.configuration_identity, self._configuration_sha256)
            or not self._matches_identity(
                self.probes.configuration_identity, self.probes.configuration_sha256
            )
        ):
            raise IrisScientificTeacherError("Scientific teacher ownership or policy changed")
        entries = tuple(self.named_parameters(remove_duplicate=False))
        if (
            tuple((name, id(value)) for name, value in entries) != self._parameter_identity
            or len({id(value) for _, value in entries}) != len(entries)
            or any(
                not value.requires_grad or not bool(torch.isfinite(value).all())
                for _, value in entries
            )
            or self.teacher.backbone.trainable_parameter_count != 0
            or self.teacher.backbone.has_parameter_gradients
        ):
            raise IrisScientificTeacherError(
                "Scientific teacher parameter graph is not stable/frozen"
            )

    def forward(
        self, example: IrisExample, *, terminal_token_id: int
    ) -> IrisScientificTeacherOutputV1:
        return self.training_forward(example, terminal_token_id=terminal_token_id)

    def training_forward(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificTeacherOutputV1:
        try:
            self._validate_boundary(training=True)
            return self._execute(
                example, terminal_token_id=terminal_token_id, training=True, diagnostics=diagnostics
            )
        except Exception:
            raise IrisScientificTeacherError(
                "Scientific training forward could not complete"
            ) from None

    def evaluation_forward(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificTeacherOutputV1:
        try:
            self._validate_boundary(training=False)
            with torch.no_grad():
                return self._execute(
                    example,
                    terminal_token_id=terminal_token_id,
                    training=False,
                    diagnostics=diagnostics,
                )
        except Exception:
            raise IrisScientificTeacherError(
                "Scientific evaluation forward could not complete"
            ) from None

    def _execute(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        training: bool,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificTeacherOutputV1:
        if type(example) is not IrisExample:
            raise IrisScientificTeacherError("A complete exact IrisExample is required")
        row = IrisExample.model_validate(example.model_dump(mode="python", warnings=False))
        if diagnostics is not None:
            diagnostics = WorkspaceTraceOptionsV1.model_validate(diagnostics)
            if diagnostics.example_id != row.id:
                raise IrisScientificTeacherError("Diagnostic example label differs from owned row")
        if training and row.split != "train":
            raise IrisScientificTeacherError(
                "Validation/test rows cannot train the scientific teacher"
            )
        identity = domain_sha256(
            "iris/scientific-teacher-example/v1", canonical_json_bytes(row.model_dump(mode="json"))
        )
        versions = tuple(parameter._version for parameter in self.parameters())
        labels = prepare_probe_targets(
            (row,),
            vocabulary=self.probes.vocabulary,
            purpose="training" if training else "analysis",
            device=self.teacher.backbone.device,
        )
        prepared = prepare_teacher_forcing_input(
            backbone=self.teacher.backbone,
            prompt=row.prompt,
            target_text=row.expected_response,
            terminal_token_id=terminal_token_id,
        )
        self._validate_boundary(training=training)
        if tuple(parameter._version for parameter in self.parameters()) != versions:
            raise IrisScientificTeacherError("Target preparation mutated registered parameters")
        diagnostic_args = {} if diagnostics is None else {"diagnostics": diagnostics}
        trace = (
            self.teacher.training_forward_with_trace(prepared, **diagnostic_args)
            if training
            else self.teacher.evaluation_forward_with_trace(prepared, **diagnostic_args)
        )
        terms, usage = _objective_from_trace_v1(
            trace,
            cycles=self._cycles,
            probes=self.probes,
            labels=labels,
            policy=self._policy,
        )
        self._validate_boundary(training=training)
        if tuple(parameter._version for parameter in self.parameters()) != versions:
            raise IrisScientificTeacherError("Scientific forward mutated registered parameters")
        if terms.total.requires_grad is not training or (training and terms.total.grad_fn is None):
            raise IrisScientificTeacherError("Scientific objective gradient mode is invalid")
        return IrisScientificTeacherOutputV1(
            example_id=row.id,
            example_sha256=identity,
            configuration_sha256=self._configuration_sha256,
            supervised_token_count=trace.teacher.supervised_token_count,
            losses=terms,
            usage=usage,
            trace=trace,
        )

    def evaluation_forward_intervened(
        self,
        example: IrisExample,
        *,
        terminal_token_id: int,
        plan: WorkspaceInterventionPlanV1,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> IrisScientificIntervenedTeacherOutputV1:
        """Measure this owned row's target NLL with an explicit already-bound plan.

        Corpus callers bind stochastic operators to each row exactly as paired
        inference does. Reference tokens do not enter workspace analysis. No probe
        loss or baseline utilization surrogate is presented as a changed NLL.
        """
        self._validate_boundary(training=False)
        caller = capture_runtime_state(self.teacher.backbone.device)
        try:
            if type(example) is not IrisExample:
                raise ValueError("An exact complete IrisExample is required")
            row = IrisExample.model_validate(example.model_dump(mode="python", warnings=False))
            if diagnostics is not None:
                diagnostics = WorkspaceTraceOptionsV1.model_validate(diagnostics)
                if diagnostics.example_id != row.id:
                    raise ValueError("Diagnostic example label differs from owned row")
            identity = domain_sha256(
                "iris/scientific-teacher-example/v1",
                canonical_json_bytes(row.model_dump(mode="json")),
            )
            before = module_state_sha256((("scientific_teacher", self),))
            with torch.no_grad():
                prepared = prepare_teacher_forcing_input(
                    backbone=self.teacher.backbone,
                    prompt=row.prompt,
                    target_text=row.expected_response,
                    terminal_token_id=terminal_token_id,
                )
                self._validate_boundary(training=False)
                diagnostic_args = {} if diagnostics is None else {"diagnostics": diagnostics}
                output = self.teacher.evaluation_forward_intervened(
                    prepared, plan=plan, **diagnostic_args
                )
                if type(output) is not IrisIntervenedTeacherOutputV1:
                    raise ValueError("Exact intervened teacher result required")
                with torch.autocast(device_type=self.teacher.backbone.device.type, enabled=False):
                    nll_sum = output.loss * output.supervised_token_count
            self._validate_boundary(training=False)
            if module_state_sha256((("scientific_teacher", self),)) != before or not bool(
                torch.isfinite(nll_sum)
            ):
                raise ValueError("Intervened measurement changed registered state or is nonfinite")
            return IrisScientificIntervenedTeacherOutputV1(
                example_id=row.id,
                example_sha256=identity,
                configuration_sha256=self._configuration_sha256,
                target_nll_sum=nll_sum,
                teacher=output,
            )
        except Exception:
            raise IrisScientificTeacherError(
                "Scientific intervened evaluation did not complete"
            ) from None
        finally:
            restore_runtime_state(caller)


__all__ = [
    "IrisScientificIntervenedTeacherOutputV1",
    "IrisScientificObjectivePolicyV1",
    "IrisScientificTeacherError",
    "IrisScientificTeacherOutputV1",
    "IrisScientificTeacherV1",
]
