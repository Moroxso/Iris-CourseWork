"""Explicit diagnostic fact labels and workspace-only training/analysis heads.

No vocabulary fitting, model loader, inference integration or optimizer ownership.
The initial representation is explicitly mean_all_slots, not a source mandate.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Annotated, Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from torch import nn
from torch.nn import functional

from iris.data.schemas import IrisExample
from iris.training.losses import PROBE_FACT_FIELDS, workspace_probe_loss
from iris.utils.hashing import canonical_json_bytes, domain_sha256

type ProbeCategory = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class IrisProbeError(ValueError):
    """Probe vocabulary, labels, parameters or workspace input is incompatible."""


class IrisProbeVocabulary(BaseModel):
    """Ordered immutable class identities; changing order changes their meaning."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )
    schema_version: Literal["iris-probe-vocabulary-v1"] = "iris-probe-vocabulary-v1"
    active_constraint: tuple[ProbeCategory, ...] = Field(min_length=1, max_length=128)
    operation: tuple[ProbeCategory, ...] = Field(min_length=1, max_length=128)
    conflicting_conditions: tuple[StrictBool, StrictBool] = (False, True)
    result_type: tuple[ProbeCategory, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_classes(self) -> Self:
        for values in (self.active_constraint, self.operation, self.result_type):
            if len(set(values)) != len(values):
                raise ValueError("Probe class identities must be unique within each head")
        if self.conflicting_conditions != (False, True):
            raise ValueError("Conflict classes are explicitly false=0 and true=1")
        return self


def _own_vocabulary(vocabulary: IrisProbeVocabulary) -> IrisProbeVocabulary:
    if type(vocabulary) is not IrisProbeVocabulary:
        raise IrisProbeError("Exact IrisProbeVocabulary required")
    try:
        return IrisProbeVocabulary.model_validate(
            vocabulary.model_dump(mode="python", warnings=False)
        )
    except ValueError:
        raise IrisProbeError("Invalid probe vocabulary") from None


def probe_vocabulary_sha256(vocabulary: IrisProbeVocabulary) -> str:
    owned = _own_vocabulary(vocabulary)
    return domain_sha256(
        "iris/probe-vocabulary/v1", canonical_json_bytes(owned.model_dump(mode="json"))
    )


def synthetic_probe_vocabulary_v1() -> IrisProbeVocabulary:
    """Declared generator-v1/v2 fact domains, not a scan or fit of any dataset split."""
    return IrisProbeVocabulary(
        active_constraint=(
            "ordered_workflow_preconditions",
            "original_goal",
            "range_and_remainder",
            "source_priority_then_recency",
        ),
        operation=(
            "count",
            "edit_file",
            "finish",
            "inspect_error",
            "maximum",
            "minimum",
            "read_file",
            "resolve_conflict",
            "run_tests",
            "search_symbol",
            "squared_sum",
            "sum",
        ),
        result_type=("conflict_resolution", "job_id", "python_function", "tool_prediction"),
    )


def _encode_facts(facts: dict[str, Any], vocabulary: IrisProbeVocabulary) -> dict[str, int]:
    if type(facts) is not dict or not set(PROBE_FACT_FIELDS) <= facts.keys():
        raise IrisProbeError("All four probe facts must be supplied")
    encoded = {}
    for name in PROBE_FACT_FIELDS:
        value = facts[name]
        if name == "conflicting_conditions":
            if type(value) is not bool:
                raise IrisProbeError("Conflict fact must be an exact boolean")
            encoded[name] = int(value)
        else:
            values = getattr(vocabulary, name)
            if type(value) is not str or value not in values:
                raise IrisProbeError("Probe fact is outside the declared vocabulary")
            encoded[name] = values.index(value)
    return encoded


@dataclass(frozen=True, slots=True)
class IrisProbeTargets:
    """Caller-owned tensors and ordered IDs; not a tamper-proof provenance receipt."""

    vocabulary_sha256: str
    example_ids: tuple[str, ...]
    purpose: Literal["training", "analysis"]
    labels: dict[str, torch.Tensor]


def prepare_probe_targets(
    examples: tuple[IrisExample, ...],
    *,
    vocabulary: IrisProbeVocabulary,
    purpose: Literal["training", "analysis"],
    device: torch.device,
) -> IrisProbeTargets:
    """Encode explicit row facts; training rejects every non-train example.

    Analysis may inspect validation/test labels, but this function neither fits a
    vocabulary nor updates any model. No prompt, reference or hidden fact is an
    input to the heads. At most 1024 rows per call, with owned int64 label tensors.
    """
    owned = _own_vocabulary(vocabulary)
    if type(examples) is not tuple or not 1 <= len(examples) <= 1024:
        raise IrisProbeError("Probe targets require one to 1024 explicit examples")
    if type(purpose) is not str or purpose not in ("training", "analysis"):
        raise IrisProbeError("Probe target purpose must be explicit")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise IrisProbeError("Probe target device must be explicit CPU or CUDA")
    encoded = []
    identifiers = []
    try:
        for example in examples:
            if type(example) is not IrisExample:
                raise IrisProbeError("Probe labels require exact IrisExample rows")
            row = IrisExample.model_validate(example.model_dump(mode="python", warnings=False))
            if purpose == "training" and row.split != "train":
                raise IrisProbeError("Training probe labels cannot come from validation/test")
            encoded.append(_encode_facts(row.hidden_facts, owned))
            identifiers.append(row.id)
    except (TypeError, ValueError):
        raise IrisProbeError("Probe target rows could not be validated and encoded") from None
    # Allocate after validating the entire batch, including any bad late row.
    with torch.inference_mode(False):
        labels = {
            name: torch.tensor([row[name] for row in encoded], dtype=torch.int64, device=device)
            for name in PROBE_FACT_FIELDS
        }
    return IrisProbeTargets(probe_vocabulary_sha256(owned), tuple(identifiers), purpose, labels)


class _LinearProbe(nn.Module):
    def __init__(
        self,
        width: int,
        classes: int,
        generator: torch.Generator,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        initial_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        bound = 1.0 / sqrt(width)
        self.weight = nn.Parameter(
            torch.empty(classes, width, dtype=initial_dtype, device="cpu")
            .uniform_(-bound, bound, generator=generator)
            .to(device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(classes, dtype=initial_dtype, device="cpu")
            .uniform_(-bound, bound, generator=generator)
            .to(device=device, dtype=dtype)
        )


class WorkspaceProbeHeads(nn.Module):
    """Four registered linear heads on a mean of workspace slots, never raw facts.

    Local CPU-generator initialization does not consume the ambient RNG. Parameter
    count is independent of K; no semantic slot/module roles are assigned. This
    module must still be included in a future training optimizer/checkpoint owner.
    """

    configuration_identity: torch.Tensor

    def __init__(
        self,
        *,
        vocabulary: IrisProbeVocabulary,
        state_dim: int,
        pooling: Literal["mean_all_slots"],
        init_seed: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        owned = _own_vocabulary(vocabulary)
        if torch.is_inference_mode_enabled():
            raise IrisProbeError("Construct trainable probe parameters outside inference_mode")
        if type(state_dim) is not int or not 1 <= state_dim <= 4096:
            raise IrisProbeError("Probe state width must be an integer in [1, 4096]")
        if type(init_seed) is not int or not 0 <= init_seed < 2**63:
            raise IrisProbeError("Probe initialization seed must be an exact bounded integer")
        if type(pooling) is not str or pooling != "mean_all_slots":
            raise IrisProbeError("This probe profile requires explicit mean_all_slots pooling")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise IrisProbeError("Probe parameter dtype must be an explicit supported float")
        if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
            raise IrisProbeError("Probe device must be explicit CPU or CUDA")
        configuration = {
            "kind": "iris-workspace-probe-heads-v1",
            "vocabulary_sha256": probe_vocabulary_sha256(owned),
            "state_dim": state_dim,
            "pooling": pooling,
            "init_seed": init_seed,
            "initialization_dtype": str(dtype).removeprefix("torch."),
            "initialization": "cpu_local_generator_linear_uniform_v1",
        }
        identity = bytes.fromhex(
            domain_sha256("iris/workspace-probe-heads/v1", canonical_json_bytes(configuration))
        )
        self._configuration_identity = identity
        self._configuration_json = canonical_json_bytes(configuration).decode("utf-8")
        self._state_dim = state_dim
        self._vocabulary = owned
        self._vocabulary_identity = configuration["vocabulary_sha256"]
        self.register_buffer(
            "configuration_identity", torch.tensor(list(identity), dtype=torch.uint8, device=device)
        )
        generator = torch.Generator(device="cpu").manual_seed(init_seed)
        self.heads = nn.ModuleDict(
            {
                name: _LinearProbe(
                    state_dim, len(getattr(owned, name)), generator, dtype=dtype, device=device
                )
                for name in PROBE_FACT_FIELDS
            }
        )
        self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
            self._check_loading_identity
        )

    @property
    def vocabulary(self) -> IrisProbeVocabulary:
        return _own_vocabulary(self._vocabulary)

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_identity.hex()

    @property
    def configuration_json(self) -> str:
        """Complete immutable initialization policy for source-facing checkpoints."""
        if (
            domain_sha256("iris/workspace-probe-heads/v1", self._configuration_json.encode("utf-8"))
            != self.configuration_sha256
        ):
            raise IrisProbeError("Probe configuration snapshot changed")
        return self._configuration_json

    def _identity_matches(self, value: object) -> bool:
        return (
            type(value) is torch.Tensor
            and value.dtype == torch.uint8
            and value.layout == torch.strided
            and tuple(value.shape) == (32,)
            and value.device.type in ("cpu", "cuda")
            and bytes(value.detach().to(device="cpu").tolist()) == self._configuration_identity
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
        # Raise before loading any of this module's buffers/children, even if the
        # caller requests strict=False. A parent load is not promised to be atomic.
        if module is not self or not self._identity_matches(
            state_dict.get(prefix + "configuration_identity")
        ):
            raise IrisProbeError("Probe checkpoint configuration or class ordering differs")

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw logits from [batch, slots, width], with no labels accepted."""
        if (
            not isinstance(slots, torch.Tensor)
            or slots.layout != torch.strided
            or slots.ndim != 3
            or slots.shape[-1] != self._state_dim
            or slots.shape[0] < 1
            or slots.shape[1] < 1
            or slots.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or slots.device.type not in ("cpu", "cuda")
            or not bool(torch.isfinite(slots).all())
        ):
            raise IrisProbeError("Invalid workspace tensor for diagnostic probes")
        if (
            not self._identity_matches(self.configuration_identity)
            or probe_vocabulary_sha256(self._vocabulary) != self._vocabulary_identity
            or tuple(self.heads) != PROBE_FACT_FIELDS
        ):
            raise IrisProbeError("Probe registered configuration changed")
        validated_heads: dict[str, _LinearProbe] = {}
        for name in PROBE_FACT_FIELDS:
            head = self.heads[name]
            if type(head) is not _LinearProbe:
                raise IrisProbeError("Probe head registration changed")
            classes = len(getattr(self._vocabulary, name))
            for parameter, shape in (
                (head.weight, (classes, self._state_dim)),
                (head.bias, (classes,)),
            ):
                if (
                    tuple(parameter.shape) != shape
                    or parameter.dtype != slots.dtype
                    or parameter.device != slots.device
                    or not bool(torch.isfinite(parameter).all())
                ):
                    raise IrisProbeError("Probe parameters must match the workspace boundary")
            validated_heads[name] = head
        accumulation_dtype = torch.float64 if slots.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=slots.device.type, enabled=False):
            pooled = slots.to(dtype=accumulation_dtype).mean(dim=1)
            logits = {
                name: functional.linear(
                    pooled,
                    head.weight.to(dtype=accumulation_dtype),
                    head.bias.to(dtype=accumulation_dtype),
                )
                for name, head in validated_heads.items()
            }
        if any(not bool(torch.isfinite(value).all()) for value in logits.values()):
            raise IrisProbeError("Probe arithmetic produced non-finite logits")
        return logits


def workspace_probe_objective(
    heads: WorkspaceProbeHeads,
    slots: torch.Tensor,
    targets: IrisProbeTargets,
    *,
    head_reduction: Literal["sum", "mean"],
) -> torch.Tensor:
    """Connect explicit target identities to workspace-only heads and four-head CE.

    Training requires train mode and autograd. Analysis requires eval mode and runs
    under no_grad without changing modes/parameters/existing gradients. This does
    not associate a workspace with an LM example; the future teacher owner must.
    """
    if type(heads) is not WorkspaceProbeHeads or type(targets) is not IrisProbeTargets:
        raise IrisProbeError("Exact heads and prepared probe targets required")
    if (
        targets.vocabulary_sha256 != probe_vocabulary_sha256(heads.vocabulary)
        or type(targets.example_ids) is not tuple
        or not targets.example_ids
        or any(type(value) is not str or not value for value in targets.example_ids)
        or not isinstance(slots, torch.Tensor)
        or slots.ndim != 3
        or slots.shape[0] != len(targets.example_ids)
        or type(targets.labels) is not dict
        or type(targets.purpose) is not str
    ):
        raise IrisProbeError("Probe targets do not match the declared vocabulary and batch")
    if targets.purpose == "training":
        if (
            not heads.training
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or any(not parameter.requires_grad for parameter in heads.parameters())
        ):
            raise IrisProbeError("Probe training requires train mode with autograd")
        loss = workspace_probe_loss(heads(slots), targets.labels, head_reduction=head_reduction)
        if not loss.requires_grad or loss.grad_fn is None:
            raise IrisProbeError("Probe training objective must remain attached to its graph")
        return loss
    if targets.purpose != "analysis" or heads.training:
        raise IrisProbeError("Probe analysis requires explicit analysis targets and eval mode")
    with torch.no_grad():
        return workspace_probe_loss(heads(slots), targets.labels, head_reduction=head_reduction)


__all__ = [
    "IrisProbeError",
    "IrisProbeTargets",
    "IrisProbeVocabulary",
    "WorkspaceProbeHeads",
    "prepare_probe_targets",
    "probe_vocabulary_sha256",
    "synthetic_probe_vocabulary_v1",
    "workspace_probe_objective",
]
