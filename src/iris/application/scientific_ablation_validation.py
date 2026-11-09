"""Mode-bound complete target NLL with checked owner, state and RNG preservation."""

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Never

import torch

from iris.ablations import cognitive_ablation_policy_sha256_v1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherOutputV1
from iris.application.scientific_optimization import _same_runtime
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.data.schemas import IrisExample, SyntheticPythonCase
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.outputs import IrisTeacherForcedOutput, IrisTeacherTracedOutputV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state


class IrisScientificAblationValidationError(RuntimeError):
    """Mode-bound validation provenance or non-mutation failed."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationValidationV1:
    dataset_sha256: str
    example_count: int
    target_token_count: int
    target_nll_sum: float
    target_nll_per_token: float
    at_optimizer_step: int
    run_configuration_sha256: str
    model_configuration_sha256: str
    ablation_policy_sha256: str
    scientific_state_sha256: str


def _complete_row_fields(example: IrisExample) -> bool:
    return (
        type(example) is IrisExample
        and set(vars(example)) == set(IrisExample.model_fields)
        and example.__pydantic_extra__ is None
        and type(example.code_cases) is tuple
        and all(
            type(case) is SyntheticPythonCase
            and set(vars(case)) == set(SyntheticPythonCase.model_fields)
            and case.__pydantic_extra__ is None
            for case in example.code_cases
        )
    )


def own_scientific_ablation_validation_rows_v1(
    examples: tuple[IrisExample, ...], *, training_rows: tuple[IrisExample, ...]
) -> tuple[tuple[IrisExample, ...], str]:
    try:
        if (
            type(examples) is not tuple
            or not 1 <= len(examples) <= 5000
            or type(training_rows) is not tuple
            or not 1 <= len(training_rows) <= 200000
            or any(not _complete_row_fields(row) or row.split != "train" for row in training_rows)
        ):
            raise ValueError("Explicit bounded train/validation corpora required")
        train_seeds = {row.seed for row in training_rows}
        rows: list[IrisExample] = []
        identities: set[str] = set()
        logical_bytes = 0
        digest = sha256(b"iris/scientific-ablation-validation-corpus/v1\0")
        for example in examples:
            if not _complete_row_fields(example):
                raise ValueError("Exact complete validation row fields required")
            row = IrisExample.model_validate(example.model_dump(mode="python", warnings=False))
            if row.split != "validation" or row.id in identities or row.seed in train_seeds:
                raise ValueError("Validation must be unique and seed-disjoint from train")
            content = canonical_json_bytes(row.model_dump(mode="json"))
            logical_bytes += len(content)
            if logical_bytes > 64 * 1024**2:
                raise ValueError("Validation corpus byte budget exceeded")
            identities.add(row.id)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            rows.append(row)
        return tuple(rows), digest.hexdigest()
    except Exception:
        raise IrisScientificAblationValidationError("Ablation validation corpus rejected") from None


def _training_corpus_sha256(run: IrisScientificAblationRunV1) -> str:
    """Recheck the original sealed corpus, not mutable current IDs/seed labels."""
    configuration = json.loads(run.configuration_json)
    digest = sha256(b"iris/scientific-train-corpus/v1\0")
    logical_bytes = 0

    def reject() -> Never:
        run._poisoned = run._optimizer._poisoned = True
        raise ValueError("Owned training corpus differs from its original identity")

    if len(run._rows) != configuration["dataset_examples"] or len(run._row_fingerprints) != len(
        run._rows
    ):
        reject()
    for row, fingerprint in zip(run._rows, run._row_fingerprints, strict=True):
        if not _complete_row_fields(row) or row.split != "train":
            reject()
        try:
            content = canonical_json_bytes(row.model_dump(mode="json", warnings=False))
        except MemoryError:
            raise
        except Exception:
            reject()
        if sha256(content).hexdigest() != fingerprint:
            reject()
        logical_bytes += len(content)
        if logical_bytes > 512 * 1024**2:
            reject()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    result = digest.hexdigest()
    if (
        logical_bytes != configuration["dataset_logical_bytes"]
        or result != configuration["dataset_sha256"]
    ):
        reject()
    return result


def _snapshot(run: IrisScientificAblationRunV1) -> tuple[object, ...]:
    optimizer = run._optimizer
    model = optimizer._model
    frozen_sha256 = run._backbone.checkpoint_runtime_sha256
    if frozen_sha256 != run._backbone_identity:
        run._poisoned = optimizer._poisoned = True
        raise ValueError("Frozen backbone runtime differs from the attached run")
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    moments = []
    for index, parameter in enumerate(optimizer._topology.parameters):
        if parameter in optimizer._optimizer.state:
            for name in ("step", "exp_avg", "exp_avg_sq"):
                value = optimizer._optimizer.state[parameter][name]
                moments.append(
                    (index, name, id(value), value._version, tensor_sha256(value.reshape(-1)))
                )
    return (
        id(optimizer),
        id(model),
        id(optimizer._optimizer),
        id(optimizer._topology),
        id(run._attachment_token),
        id(optimizer._attached_run_token),
        id(run._rows),
        run.configuration_json,
        run.configuration_sha256,
        run._cursor(),
        optimizer.configuration_json,
        optimizer._parameter_steps,
        optimizer.completed_steps,
        optimizer._current_lr,
        optimizer._committed_scientific_state_sha256,
        optimizer._state_sha256(),
        frozen_sha256,
        tuple((id(value), value._version) for value in tensors),
        tuple((name, id(module)) for name, module in model.named_modules()),
        tuple(moments),
    )


def _has_signal(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, SystemExit)) or (
        isinstance(error, BaseExceptionGroup)
        and any(_has_signal(item) for item in error.exceptions)
    )


def evaluate_scientific_ablation_validation_v1(
    run: IrisScientificAblationRunV1, examples: tuple[IrisExample, ...]
) -> IrisScientificAblationValidationV1:
    return _evaluate_scientific_ablation_validation_v1(run, examples, capability=None)


def _evaluate_scientific_ablation_validation_v1(
    run: IrisScientificAblationRunV1,
    examples: tuple[IrisExample, ...],
    *,
    capability: object | None,
) -> IrisScientificAblationValidationV1:
    caller = None
    locked = False
    primary_signal: BaseException | None = None
    before: tuple[object, ...] | None = None
    try:
        if type(run) is not IrisScientificAblationRunV1 or run._busy:
            raise ValueError("Exact idle attached ablation run required")
        run._authorize_operation("validation", capability)
        run._validate_boundary()
        optimizer, model = run._optimizer, run._optimizer._model
        caller = capture_runtime_state(optimizer._device)
        runtime_before = run._clone_runtime(run._runtime)
        modes = tuple((module, module.training) for module in model.modules())
        run._busy = locked = True
        before = _snapshot(run)
        _training_corpus_sha256(run)
        rows, digest = own_scientific_ablation_validation_rows_v1(examples, training_rows=run._rows)
        restore_runtime_state(run._runtime)
        model.eval()
        state_sha256 = optimizer._committed_scientific_state_sha256
        sums: list[float] = []
        token_count = 0
        with torch.no_grad():
            for row in rows:
                prepared = prepare_teacher_forcing_input(
                    backbone=run._backbone,
                    prompt=row.prompt,
                    target_text=row.expected_response,
                    terminal_token_id=run._terminal_token_id,
                )
                expected_count = int(prepared.target_attention_mask.sum())
                prompt_sha256 = domain_sha256(
                    "iris/cognitive-ablation-prompt/v1", row.prompt.encode()
                )
                target_sha256 = teacher_target_sha256_v1(
                    prepared.target_input_ids, prepared.target_attention_mask
                )
                sidecar_sha256 = model.teacher.evaluation_provenance.sidecar_state_sha256
                expected_teacher_invocation = domain_sha256(
                    "iris/cognitive-ablation-teacher-invocation/v1",
                    canonical_json_bytes(
                        {
                            "mode": "evaluation",
                            "policy_sha256": model.ablation_policy_sha256,
                            "prompt_sha256": prompt_sha256,
                            "target_sha256": target_sha256,
                            "teacher_graph_sha256": run._teacher_graph_identity,
                            "sidecar_state_sha256": sidecar_sha256,
                        }
                    ),
                )
                row_sha256 = domain_sha256(
                    "iris/scientific-ablation-example/v1",
                    canonical_json_bytes(row.model_dump(mode="json")),
                )
                expected_invocation = domain_sha256(
                    "iris/scientific-ablation-invocation/v1",
                    canonical_json_bytes(
                        {
                            "mode": "evaluation",
                            "example_sha256": row_sha256,
                            "configuration_sha256": model.configuration_sha256,
                            "teacher_invocation_sha256": expected_teacher_invocation,
                            "scientific_state_sha256": state_sha256,
                        }
                    ),
                )
                output = model.evaluation_forward(row, terminal_token_id=run._terminal_token_id)
                if (
                    type(output) is not IrisScientificAblationTeacherOutputV1
                    or output.mode != "evaluation"
                    or output.example_id != row.id
                    or output.example_sha256 != row_sha256
                    or output.configuration_sha256 != model.configuration_sha256
                    or output.scientific_state_sha256 != state_sha256
                    or output.invocation_sha256 != expected_invocation
                    or type(output.teacher) is not IrisAblatedTeacherOutputV1
                    or output.teacher.mode != "evaluation"
                    or output.teacher.invocation_sha256 != expected_teacher_invocation
                    or output.teacher.prompt_sha256 != prompt_sha256
                    or output.teacher.target_sha256 != target_sha256
                    or output.teacher.teacher_graph_sha256 != run._teacher_graph_identity
                    or output.teacher.sidecar_state_sha256 != sidecar_sha256
                    or cognitive_ablation_policy_sha256_v1(output.teacher.policy)
                    != model.ablation_policy_sha256
                    or type(output.teacher.traced) is not IrisTeacherTracedOutputV1
                    or type(output.teacher.traced.teacher) is not IrisTeacherForcedOutput
                    or output.teacher.traced.teacher.supervised_token_count != expected_count
                    or output.ablation_policy_sha256 != model.ablation_policy_sha256
                    or type(output.supervised_token_count) is not int
                    or output.supervised_token_count != expected_count
                    or expected_count < 1
                ):
                    raise ValueError("Validation output provenance/count differs")
                loss = output.losses.language_model
                if (
                    type(loss) is not torch.Tensor
                    or loss.dtype not in (torch.float32, torch.float64)
                    or loss.ndim != 0
                    or loss.requires_grad
                    or loss.grad_fn is not None
                    or loss.device != optimizer._device
                    or not bool(torch.isfinite(loss))
                    or float(loss) < 0
                ):
                    raise ValueError("Invalid detached validation target NLL")
                sums.append(float(loss))
                token_count += expected_count
                del output, loss, prepared
        total = math.fsum(sums)
        if not math.isfinite(total) or token_count < 1:
            raise ValueError("Invalid validation aggregation")
        return IrisScientificAblationValidationV1(
            digest,
            len(rows),
            token_count,
            total,
            total / token_count,
            run.completed_steps,
            run.configuration_sha256,
            model.configuration_sha256,
            model.ablation_policy_sha256,
            state_sha256,
        )
    except BaseException as error:
        if _has_signal(error):
            primary_signal = error
            raise
        raise IrisScientificAblationValidationError(
            "Ablation validation could not complete"
        ) from None
    finally:
        cleanup_failed = False
        cleanup_signal: BaseException | None = None
        if locked:
            try:
                model.train(True)
                if any(module.training is not mode for module, mode in modes):
                    raise ValueError("Validation did not restore owned modes")
            except BaseException as error:
                cleanup_failed = True
                if _has_signal(error):
                    cleanup_signal = error
        if caller is not None:
            try:
                restore_runtime_state(caller)
            except BaseException as error:
                cleanup_failed = True
                if _has_signal(error) and cleanup_signal is None:
                    cleanup_signal = error
        if locked:
            try:
                run._validate_boundary()
                after = _snapshot(run)
                _training_corpus_sha256(run)
                if (before is not None and before != after) or not _same_runtime(
                    runtime_before, run._runtime
                ):
                    raise ValueError("Validation mutated owned run/optimizer state")
            except BaseException as error:
                cleanup_failed = True
                if _has_signal(error) and cleanup_signal is None:
                    cleanup_signal = error
        if caller is not None:
            try:
                if not _same_runtime(caller, capture_runtime_state(run._optimizer._device)):
                    raise ValueError("Validation did not restore caller runtime")
            except BaseException as error:
                cleanup_failed = True
                if _has_signal(error) and cleanup_signal is None:
                    cleanup_signal = error
        if locked:
            run._busy = False
        if cleanup_failed:
            run._poisoned = run._optimizer._poisoned = True
            if primary_signal is None:
                if cleanup_signal is not None:
                    raise cleanup_signal
                raise IrisScientificAblationValidationError(
                    "Ablation validation cleanup failed"
                ) from None


__all__ = [
    "IrisScientificAblationValidationError",
    "IrisScientificAblationValidationV1",
    "evaluate_scientific_ablation_validation_v1",
    "own_scientific_ablation_validation_rows_v1",
]
