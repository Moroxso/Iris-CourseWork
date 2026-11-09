"""Explicit offline CPU/FP32 composition and bounded scientific phase commands.

Candidate data and the scientific-size optimizer policy are not a release or a
scientific acceptance decision. CUDA/AMP remain outside this command version.
"""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.application.errors import ApplicationError
from iris.application.scientific_checkpoint import (
    _bounded_read,
    _regular,
    _unique,
    build_scientific_checkpoint_v1,
)
from iris.application.scientific_execution import (
    IrisScientificExecutionSnapshotV1,
    create_scientific_execution_v1,
    read_scientific_execution_v1,
    resume_scientific_execution_v1,
)
from iris.application.scientific_lifecycle import IrisScientificLifecycleV1
from iris.application.scientific_optimization import (
    IrisScientificOptimizerPolicyV1,
    IrisScientificOptimizerV1,
)
from iris.application.scientific_publication import publish_scientific_execution_deliveries_v1
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_teacher import (
    IrisScientificObjectivePolicyV1,
    IrisScientificTeacherV1,
)
from iris.application.scientific_validation import own_scientific_validation_rows_v1
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.scientific_execution_v1 import ScientificExecutionCheckpointV1
from iris.artifacts.scientific_lifecycle_v1 import (
    IrisScientificLifecyclePolicyV1,
    ScientificLifecycleStateV1,
)
from iris.backbone import BackboneAdapter
from iris.data.parquet import decode_synthetic_parquet
from iris.data.schemas import IrisExample
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.training.probes import prepare_probe_targets, synthetic_probe_vocabulary_v1
from iris.utils.devices import inspect_device
from iris.utils.hashing import canonical_json_bytes
from iris.utils.seeds import apply_runtime_policy, capture_runtime_state, restore_runtime_state


class IrisScientificCommandError(ApplicationError):
    """Content-redacted expected scientific command failure."""


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )


class ScientificParquetInputV1(_Strict):
    """Explicit local file identity and allocation limits; paths may be relocated."""

    path: str = Field(min_length=1, max_length=4096)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_rows: int = Field(ge=1, le=50000)
    maximum_file_bytes: int = Field(ge=1, le=512 * 1024**2)
    maximum_logical_bytes: int = Field(ge=1, le=512 * 1024**2)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        windows_path = PureWindowsPath(self.path)
        if (
            not self.path.strip()
            or "\0" in self.path
            or "://" in self.path
            or self.path.startswith(("//", "\\\\"))
            or (windows_path.drive and not windows_path.root)
        ):
            raise ValueError("An explicit local Parquet path is required")
        return self


class IrisScientificCommandConfigV1(_Strict):
    """A separate recipe, not a reinterpretation of legacy experiment profiles."""

    kind: Literal["iris-scientific-command-v1"] = "iris-scientific-command-v1"
    data_scope: Literal["candidate_not_launch_approval"]
    device: Literal["cpu"] = "cpu"
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    train: ScientificParquetInputV1
    validation: ScientificParquetInputV1
    initialization_seed: int = Field(ge=0, le=2**63 - 1)
    probe_initialization_seed: int = Field(ge=0, le=2**63 - 1)
    run_seed: int = Field(ge=0, le=2**63 - 1)
    deterministic: bool
    terminal_token_id: int = Field(ge=0)
    objective: IrisScientificObjectivePolicyV1
    optimizer: IrisScientificOptimizerPolicyV1
    lifecycle: IrisScientificLifecyclePolicyV1
    maximum_journal_bytes: int = Field(default=128 * 1024**2, ge=1, le=1024**3)

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        backbone = self.backbone
        if (
            not backbone.local_files_only
            or not backbone.tokenizer.use_fast
            or backbone.dtype != "fp32"
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", backbone.repository
            )
            is None
            or any(
                re.fullmatch(r"[0-9a-f]{40}", revision) is None
                for revision in (backbone.revision, backbone.tokenizer_revision)
            )
        ):
            raise ValueError("CPU/FP32, fast tokenizer, offline-only pinned HF repository required")
        if self.workspace.winners_per_cycle > self.cognitive.num_modules:
            raise ValueError("Workspace winners exceed cognitive modules")
        if self.optimizer.purpose != self.lifecycle.purpose:
            raise ValueError("Optimizer/lifecycle purposes differ")
        if self.validation.maximum_rows > 5000:
            raise ValueError("Validation row budget exceeds 5000")
        if self.optimizer.purpose == "scientific" and (
            self.train.maximum_rows != 50000 or self.validation.maximum_rows != 5000
        ):
            raise ValueError("Scientific-size policy requires 50000/5000 input budgets")
        return self


@dataclass(frozen=True, slots=True)
class IrisScientificCommandReportV1:
    path: Path
    execution_sha256: str
    run_configuration_sha256: str
    sequence: int
    state: ScientificLifecycleStateV1
    examples_seen: int
    target_tokens_seen: int
    purpose: Literal["scientific", "smoke"]


def _local_path(path: Path) -> Path:
    if not isinstance(path, Path) or str(path).startswith(("//", "\\\\")):
        raise ValueError("An explicit local filesystem path is required")
    windows_path = PureWindowsPath(str(path))
    if windows_path.drive and not windows_path.root:
        raise ValueError("Drive-relative paths are not reproducible")
    return path.absolute()


def _input_bytes(path: Path, limit: int) -> bytes:
    path = _local_path(path)
    for parent in path.parents:
        _regular(parent, directory=True)
    return _bounded_read(path, limit)


def read_scientific_command_config_v1(path: Path) -> IrisScientificCommandConfigV1:
    """Read bounded human-formatted JSON without duplicate fields or coercion."""
    try:
        raw = json.loads(_input_bytes(path, 128 * 1024).decode("utf-8"), object_pairs_hook=_unique)
        return IrisScientificCommandConfigV1.model_validate_json(canonical_json_bytes(raw))
    except Exception:
        raise IrisScientificCommandError("Scientific command configuration is invalid") from None


def _rows(spec: ScientificParquetInputV1, base: Path) -> tuple[IrisExample, ...]:
    content = _input_bytes(base / spec.path, spec.maximum_file_bytes)
    if sha256(content).hexdigest() != spec.sha256:
        raise ValueError("Parquet file identity differs")
    return decode_synthetic_parquet(
        content,
        max_rows=spec.maximum_rows,
        max_file_bytes=spec.maximum_file_bytes,
        max_logical_bytes=spec.maximum_logical_bytes,
    )


def _preflight(
    backbone: BackboneAdapter,
    config: IrisScientificCommandConfigV1,
    train: tuple[IrisExample, ...],
    validation: tuple[IrisExample, ...],
) -> None:
    """No forward pass: every raw prompt, full answer/EOS and static probe label."""
    vocabulary = synthetic_probe_vocabulary_v1()
    splits: tuple[tuple[tuple[IrisExample, ...], Literal["training", "analysis"]], ...] = (
        (train, "training"),
        (validation, "analysis"),
    )
    for rows, purpose in splits:
        for row in rows:
            raw = backbone._tokenizer(
                row.prompt,
                return_tensors="pt",
                truncation=False,
                padding=False,
                return_attention_mask=True,
            )
            ids, mask = raw["input_ids"], raw["attention_mask"]
            if (
                type(ids) is not torch.Tensor
                or ids.ndim != 2
                or ids.shape[0] != 1
                or not 1 <= ids.shape[1] <= backbone.max_prompt_tokens
                or type(mask) is not torch.Tensor
                or mask.shape != ids.shape
                or not bool((mask == 1).all())
            ):
                raise ValueError("A complete prompt does not fit the declared token budget")
            actual = backbone.tokenize(row.prompt)
            if not torch.equal(ids, actual["input_ids"].cpu()) or not torch.equal(
                mask, actual["attention_mask"].cpu()
            ):
                raise ValueError("Prompt tokenization would change or truncate source content")
            prepared = prepare_teacher_forcing_input(
                backbone=backbone,
                prompt=row.prompt,
                target_text=row.expected_response,
                terminal_token_id=config.terminal_token_id,
            )
            if (
                ids.shape[1] + config.workspace.num_slots + prepared.target_input_ids.shape[1]
                > backbone.max_position_embeddings
            ):
                raise ValueError("Prompt/workspace/full-target context does not fit")
            prepare_probe_targets(
                (row,), vocabulary=vocabulary, purpose=purpose, device=backbone.device
            )


@contextmanager
def _composed(
    config: IrisScientificCommandConfigV1, *, base: Path
) -> Iterator[IrisScientificLifecycleV1]:
    # Snapshot/revalidate even nested mutable models before data or model loading.
    config = IrisScientificCommandConfigV1.model_validate_json(config.model_dump_json())
    if Path(config.backbone.repository).exists():
        raise ValueError("A local directory cannot shadow the pinned HF repository identity")
    train, validation = _rows(config.train, base), _rows(config.validation, base)
    if any(row.split != "train" for row in train):
        raise ValueError("Training input contains held-out rows")
    own_scientific_validation_rows_v1(validation, training_rows=train)
    if config.optimizer.purpose == "scientific" and (len(train), len(validation)) != (50000, 5000):
        raise ValueError("Scientific-size lifecycle requires complete 50000/5000 splits")
    device = inspect_device(config.device)
    if torch.is_inference_mode_enabled() or torch.is_autocast_enabled(device.device.type):
        raise ValueError("Scientific command forbids ambient inference mode or autocast")
    caller = capture_runtime_state(device.device)
    try:
        apply_runtime_policy(
            seed=config.initialization_seed,
            deterministic=config.deterministic,
            device=device.device,
        )
        torch.backends.cudnn.benchmark = False
        backbone = BackboneAdapter(config.backbone, device)
        _preflight(backbone, config, train, validation)
        # A loader/tokenizer may consume RNG. Sidecar initialization is independently seeded.
        apply_runtime_policy(
            seed=config.initialization_seed,
            deterministic=config.deterministic,
            device=device.device,
        )
        torch.backends.cudnn.benchmark = False
        with torch.enable_grad():
            teacher = build_teacher_forcing_model(
                backbone=backbone,
                cognitive_config=config.cognitive,
                workspace_config=config.workspace,
            )
            model = IrisScientificTeacherV1(
                teacher=teacher,
                vocabulary=synthetic_probe_vocabulary_v1(),
                probe_init_seed=config.probe_initialization_seed,
                policy=config.objective,
            )
            optimizer = IrisScientificOptimizerV1(model=model, policy=config.optimizer)
            run = IrisScientificRunV1(
                optimizer=optimizer,
                examples=train,
                terminal_token_id=config.terminal_token_id,
                seed=config.run_seed,
            )
            yield IrisScientificLifecycleV1(run=run, validation=validation, policy=config.lifecycle)
    finally:
        restore_runtime_state(caller)


def _report(snapshot: IrisScientificExecutionSnapshotV1) -> IrisScientificCommandReportV1:
    head = snapshot.events[-1] if snapshot.events else None
    initial = snapshot.header.payload.initial.payload
    policy = IrisScientificOptimizerPolicyV1.model_validate(
        json.loads(initial.configuration_json)["optimizer_policy"]
    )
    return IrisScientificCommandReportV1(
        snapshot.path,
        snapshot.header.sha256,
        initial.configuration_sha256,
        0 if head is None else head.payload.sequence,
        ScientificLifecycleStateV1() if head is None else head.payload.after,
        0 if head is None else head.payload.examples_seen,
        0 if head is None else head.payload.target_tokens_seen,
        policy.purpose,
    )


def create_scientific_command_v1(
    *, config_path: Path, output_root: Path
) -> IrisScientificCommandReportV1:
    """Load local dependencies, preflight all rows and publish only the initial state."""
    config = read_scientific_command_config_v1(config_path)
    try:
        output_root = _local_path(output_root)
        with _composed(config, base=config_path.absolute().parent) as lifecycle:
            owner = create_scientific_execution_v1(
                lifecycle, root=output_root, maximum_journal_bytes=config.maximum_journal_bytes
            )
            return _report(read_scientific_execution_v1(owner.path))
    except Exception:
        raise IrisScientificCommandError(
            "Scientific creation failed; no training was requested. "
            "Check local inputs/configuration."
        ) from None


def resume_scientific_command_v1(
    *,
    config_path: Path,
    execution_path: Path,
    phases: int,
    delivery_root: Path | None = None,
    maximum_delivery_storage_bytes: int = 1024**3,
) -> IrisScientificCommandReportV1:
    """Replay committed tail, then append at most the explicitly requested phase count."""
    if type(phases) is not int or not 1 <= phases <= 1000:
        raise IrisScientificCommandError("Resume requires an explicit phase budget from 1 to 1000")
    config = read_scientific_command_config_v1(config_path)
    if config.optimizer.purpose == "scientific" and delivery_root is None:
        raise IrisScientificCommandError(
            "Scientific-purpose resume requires an explicit separate delivery root"
        )
    try:
        execution_path = _local_path(execution_path)
        if delivery_root is not None:
            delivery_root = _local_path(delivery_root)
            if delivery_root.is_relative_to(execution_path):
                raise ValueError("Delivery output must be outside the execution directory")
        if (
            type(maximum_delivery_storage_bytes) is not int
            or not 1 <= maximum_delivery_storage_bytes <= 16 * 1024**3
        ):
            raise ValueError("Explicit bounded delivery store quota required")
        snapshot = read_scientific_execution_v1(execution_path)  # Audit BEFORE loading the model.
        if config.maximum_journal_bytes != snapshot.header.payload.maximum_journal_bytes:
            raise ValueError("Journal budget differs from the original execution")
        with _composed(config, base=config_path.absolute().parent) as lifecycle:
            initial = build_scientific_checkpoint_v1(lifecycle._run)
            if (
                initial.manifest != snapshot.header.payload.initial
                or initial.state_bytes != snapshot.initial_state_bytes
            ):
                raise ValueError("Recipe does not reproduce the exact original initial state")

            def publish() -> None:
                if delivery_root is not None:
                    publish_scientific_execution_deliveries_v1(
                        execution_path=execution_path,
                        context=lifecycle._run,
                        output_root=delivery_root,
                        maximum_delivery_storage_bytes=maximum_delivery_storage_bytes,
                    )

            # Catch up previous publication failures BEFORE replay or any new phase.
            publish()
            owner = resume_scientific_execution_v1(execution_path, lifecycle=lifecycle)
            for _ in range(phases):
                if lifecycle.state.phase == "stopped":
                    break
                event = owner.advance()
                if type(event.payload.record) is ScientificExecutionCheckpointV1:
                    publish()
            return _report(read_scientific_execution_v1(owner.path))
    except Exception:
        raise IrisScientificCommandError(
            "Scientific resume failed; committed progress is retained. Inspect before retrying."
        ) from None


def inspect_scientific_command_v1(*, execution_path: Path) -> IrisScientificCommandReportV1:
    """Read-only full journal/checkpoint audit, without loading any backbone/tokenizer."""
    try:
        return _report(read_scientific_execution_v1(_local_path(execution_path)))
    except Exception:
        raise IrisScientificCommandError("Scientific execution inspection failed") from None


__all__ = [
    "IrisScientificCommandConfigV1",
    "IrisScientificCommandError",
    "IrisScientificCommandReportV1",
    "ScientificParquetInputV1",
    "create_scientific_command_v1",
    "inspect_scientific_command_v1",
    "read_scientific_command_config_v1",
    "resume_scientific_command_v1",
]
