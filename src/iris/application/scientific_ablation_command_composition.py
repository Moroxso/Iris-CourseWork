"""Actual offline CPU ablation owners from the distinct operator recipe.

No training/evaluation is requested by construction. This is not a durable phase
journal, installed command, scientific launch approval or a serialized factory.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch

from iris.application.errors import ApplicationError
from iris.application.scientific_ablation_command_config import (
    IrisScientificAblationCommandConfigV1,
)
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_ablation_optimization import IrisScientificAblationOptimizerV1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_ablation_validation import (
    _has_signal,
    own_scientific_ablation_validation_rows_v1,
)
from iris.application.scientific_checkpoint import _regular
from iris.application.scientific_command import _local_path, _rows
from iris.application.scientific_teacher import IrisScientificTeacherV1
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.training_composition import build_teacher_forcing_model
from iris.backbone.adapter import BackboneAdapter
from iris.data.schemas import IrisExample
from iris.training.losses import PROBE_FACT_FIELDS
from iris.training.probes import prepare_probe_targets, synthetic_probe_vocabulary_v1
from iris.utils.devices import inspect_device
from iris.utils.seeds import apply_runtime_policy, capture_runtime_state, restore_runtime_state

_DEFAULT_PARAMETER_BYTES = 256 * 1024**2


class IrisScientificAblationCommandCompositionError(ApplicationError):
    """Content-redacted construction failure, never a native completion receipt."""


def _snapshot(
    config: IrisScientificAblationCommandConfigV1,
) -> IrisScientificAblationCommandConfigV1:
    if (
        type(config) is not IrisScientificAblationCommandConfigV1
        or set(vars(config)) != set(IrisScientificAblationCommandConfigV1.model_fields)
        or config.__pydantic_extra__ is not None
    ):
        raise ValueError("Exact complete ablation operator recipe required")
    result = IrisScientificAblationCommandConfigV1.model_validate_json(
        config.model_dump_json(), strict=True
    )
    if not (
        2 <= result.cognitive.num_modules <= 64
        and 1 <= result.cognitive.state_dim <= 4096
        and 1 <= result.workspace.num_slots <= 64
        and 1 <= result.workspace.cycles <= 64
    ):
        raise ValueError("Geometry exceeds the supported mode-bound runtime profile")
    return result


def scientific_ablation_command_parameter_count_v1(
    *, config: IrisScientificAblationCommandConfigV1, backbone_hidden_size: int
) -> int:
    """Exact allocated sidecar+probe count, before constructing any parameters.

    All modes keep their registered parameters, including disconnected ones.
    This FP32 parameter allocation bound is NOT total RAM/VRAM/checkpoint bytes,
    and does not assert that the scientific under20M criterion has been met.
    """
    config = _snapshot(config)
    if type(backbone_hidden_size) is not int or backbone_hidden_size < 1:
        raise ValueError("Exact positive frozen backbone width required")
    width, modules, hidden = (
        config.cognitive.state_dim,
        config.cognitive.num_modules,
        backbone_hidden_size,
    )
    vocabulary = synthetic_probe_vocabulary_v1()
    probe_classes = sum(len(getattr(vocabulary, field)) for field in PROBE_FACT_FIELDS)
    return (
        4 * hidden * width
        + 4  # Four bias-free sampled-depth projections and mixing logits.
        + width  # Goal pooling query.
        + 2 * modules * width  # Module routing and workspace-reading queries.
        + modules * (11 * width**2 + 17 * width)  # Identity, norm, projections, GRUCell.
        + 2 * width**2
        + 3 * width  # Shared bias-free proposal heads.
        + width**2  # SlotAllocator's learned bias-free slot-key projection.
        + 5 * width**2
        + 2 * width  # Writer projections and LayerNorm.
        + hidden * width
        + 3 * hidden  # Latent bridge projection, norm, type embedding.
        + probe_classes * (width + 1)  # All four original diagnostic heads.
    )


def _parameter_budget(
    config: IrisScientificAblationCommandConfigV1, hidden: int, limit: int
) -> int:
    if type(limit) is not int or not 1 <= limit <= 512 * 1024**2:
        raise ValueError("Explicit positive FP32 parameter-byte ceiling up to512MiB required")
    count = scientific_ablation_command_parameter_count_v1(
        config=config, backbone_hidden_size=hidden
    )
    if 4 * count > limit:
        raise ValueError("Requested trainable parameter allocation exceeds its byte ceiling")
    return count


def _corpora(
    config: IrisScientificAblationCommandConfigV1, base: Path
) -> tuple[tuple[IrisExample, ...], tuple[IrisExample, ...]]:
    base = _local_path(base)
    _regular(base, directory=True)
    train, validation = _rows(config.train, base), _rows(config.validation, base)
    if (
        not train
        or any(row.split != "train" for row in train)
        or len({r.id for r in train}) != len(train)
    ):
        raise ValueError("Complete unique training rows required")
    validation, _ = own_scientific_ablation_validation_rows_v1(validation, training_rows=train)
    if config.purpose == "scientific" and (len(train), len(validation)) != (50000, 5000):
        raise ValueError("Scientific composition requires full50000/5000actual corpora")
    return train, validation


def _preflight(
    backbone: BackboneAdapter,
    config: IrisScientificAblationCommandConfigV1,
    train: tuple[IrisExample, ...],
    validation: tuple[IrisExample, ...],
) -> None:
    """Full original prompt/target/EOS/labels and actual routing/context bounds, no forward."""
    if config.terminal_token_id not in backbone.eos_token_ids:
        raise ValueError("Terminal token must be an actual frozen-tokenizer EOS")
    if (
        backbone.max_prompt_tokens + config.workspace.num_slots + backbone.max_new_tokens
        > backbone.max_position_embeddings
    ):
        raise ValueError("Declared full prompt/workspace/target maxima do not fit")
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
                raise ValueError("A complete prompt does not fit the token budget")
            actual = backbone.tokenize(row.prompt)
            if not torch.equal(ids, actual["input_ids"].cpu()) or not torch.equal(
                mask, actual["attention_mask"].cpu()
            ):
                raise ValueError("Prompt tokenization would truncate or change source content")
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
                raise ValueError("Complete prompt/workspace/full-target context does not fit")
            if (
                3 * config.cognitive.num_modules * ids.shape[1]
                + config.cognitive.num_modules * config.cognitive.state_dim
                > config.ablation.maximum_routing_scalars
            ):
                raise ValueError("Actual ablation routing scalar budget is insufficient")
            prepare_probe_targets(
                (row,), vocabulary=vocabulary, purpose=purpose, device=backbone.device
            )


def _fresh_run(
    config: IrisScientificAblationCommandConfigV1,
    train: tuple[IrisExample, ...],
    validation: tuple[IrisExample, ...],
    maximum_trainable_parameter_bytes: int,
) -> IrisScientificAblationRunV1:
    if Path(config.backbone.repository).exists():
        raise ValueError("A local directory cannot shadow the pinned HF repository")
    device = inspect_device(config.device)
    apply_runtime_policy(
        seed=config.initialization_seed, deterministic=config.deterministic, device=device.device
    )
    torch.backends.cudnn.benchmark = False
    backbone = BackboneAdapter(config.backbone, device)
    expected = _parameter_budget(config, backbone.hidden_size, maximum_trainable_parameter_bytes)
    _preflight(backbone, config, train, validation)
    # Model/tokenizer loaders may consume RNG; sidecar initialization may not depend on them.
    apply_runtime_policy(
        seed=config.initialization_seed, deterministic=config.deterministic, device=device.device
    )
    torch.backends.cudnn.benchmark = False
    with torch.enable_grad():
        teacher = build_teacher_forcing_model(
            backbone=backbone, cognitive_config=config.cognitive, workspace_config=config.workspace
        )
        base = IrisScientificTeacherV1(
            teacher=teacher,
            vocabulary=synthetic_probe_vocabulary_v1(),
            probe_init_seed=config.probe_initialization_seed,
            policy=config.objective,
        )
        model = IrisScientificAblationTeacherV1(base=base, ablation=config.ablation)
        if sum(p.numel() for p in model.parameters()) != expected or any(
            p.dtype != torch.float32 for p in model.parameters()
        ):
            raise ValueError("Allocated ablation parameters differ from preflight accounting")
        optimizer = IrisScientificAblationOptimizerV1(model=model, policy=config.optimizer)
        return IrisScientificAblationRunV1(
            optimizer=optimizer,
            examples=train,
            terminal_token_id=config.terminal_token_id,
            seed=config.run_seed,
        )


@contextmanager
def _caller_runtime() -> Iterator[None]:
    caller = capture_runtime_state(torch.device("cpu"))
    primary: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            restore_runtime_state(caller)
        except BaseException as error:
            if primary is not None and _has_signal(primary):
                raise primary from None
            if _has_signal(error):
                raise
            raise IrisScientificAblationCommandCompositionError(
                "Ablation composition could not restore the caller runtime."
            ) from None


def _construct(
    config: IrisScientificAblationCommandConfigV1, base: Path, maximum_parameter_bytes: int
) -> tuple[
    IrisScientificAblationCommandConfigV1, IrisScientificAblationRunV1, tuple[IrisExample, ...]
]:
    config = _snapshot(config)
    if torch.is_inference_mode_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Ambient inference mode/autocast forbidden")
    # Hidden width1 is a strict lower bound, checked BEFORE any model loading.
    _parameter_budget(config, 1, maximum_parameter_bytes)
    train, validation = _corpora(config, base)
    return config, _fresh_run(config, train, validation, maximum_parameter_bytes), validation


def create_scientific_ablation_command_run_v1(
    *,
    config: IrisScientificAblationCommandConfigV1,
    base: Path,
    maximum_trainable_parameter_bytes: int = _DEFAULT_PARAMETER_BYTES,
) -> IrisScientificAblationRunV1:
    """Actual fresh unattached native run; caller runtime restored, zero updates/forwards."""
    with _caller_runtime():
        try:
            _, run, _ = _construct(config, base, maximum_trainable_parameter_bytes)
            return run
        except BaseException as error:
            if _has_signal(error):
                raise
            raise IrisScientificAblationCommandCompositionError(
                "Ablation run construction failed; check recipe, corpora and resource limits. "
                "No training or evaluation was requested."
            ) from None


def create_scientific_ablation_command_lifecycle_v1(
    *,
    config: IrisScientificAblationCommandConfigV1,
    base: Path,
    maximum_trainable_parameter_bytes: int = _DEFAULT_PARAMETER_BYTES,
    checkpoint_root: Path | None = None,
) -> IrisScientificAblationLifecycleV1:
    """Actual fresh owned ablation controller, not a baseline execution journal."""
    with _caller_runtime():
        try:
            if checkpoint_root is not None and (
                not isinstance(checkpoint_root, Path) or not checkpoint_root.name
            ):
                raise ValueError("Explicit local checkpoint root required")
            config, run, validation = _construct(config, base, maximum_trainable_parameter_bytes)
            return IrisScientificAblationLifecycleV1(
                run=run,
                validation=validation,
                policy=config.lifecycle,
                checkpoint_root=checkpoint_root,
            )
        except BaseException as error:
            if _has_signal(error):
                raise
            raise IrisScientificAblationCommandCompositionError(
                "Ablation lifecycle construction failed; check inputs and owned configuration."
            ) from None


__all__ = [
    "IrisScientificAblationCommandCompositionError",
    "create_scientific_ablation_command_lifecycle_v1",
    "create_scientific_ablation_command_run_v1",
    "scientific_ablation_command_parameter_count_v1",
]
