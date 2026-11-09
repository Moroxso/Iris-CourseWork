"""Complete validation-only target NLL without advancing training state or RNG."""

import math
from dataclasses import dataclass
from hashlib import sha256

import torch

from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunV1
from iris.application.scientific_teacher import IrisScientificTeacherOutputV1
from iris.data.schemas import IrisExample
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state


class IrisScientificValidationError(RuntimeError):
    """Validation ownership, split, graph or non-mutation contract failed."""


@dataclass(frozen=True, slots=True)
class IrisScientificValidationV1:
    dataset_sha256: str
    example_count: int
    target_token_count: int
    target_nll_sum: float
    target_nll_per_token: float
    at_optimizer_step: int
    run_configuration_sha256: str


def own_scientific_validation_rows_v1(
    examples: tuple[IrisExample, ...],
    *,
    training_rows: tuple[IrisExample, ...],
) -> tuple[tuple[IrisExample, ...], str]:
    if type(examples) is not tuple or not 1 <= len(examples) <= 5000:
        raise IrisScientificValidationError("An explicit bounded validation set is required")
    rows = []
    identities: set[str] = set()
    train_seeds = {row.seed for row in training_rows}
    size = 0
    digest = sha256(b"iris/scientific-validation-corpus/v1\0")
    for example in examples:
        if type(example) is not IrisExample:
            raise IrisScientificValidationError("Exact validation rows required")
        row = IrisExample.model_validate(example.model_dump(mode="python"))
        if row.split != "validation" or row.id in identities or row.seed in train_seeds:
            raise IrisScientificValidationError(
                "Validation must be unique and seed-disjoint from train"
            )
        content = canonical_json_bytes(row.model_dump(mode="json"))
        size += len(content)
        if size > 64 * 1024**2:
            raise IrisScientificValidationError("Validation corpus exceeds its byte budget")
        identities.add(row.id)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        rows.append(row)
    return tuple(rows), digest.hexdigest()


def evaluate_scientific_validation_v1(
    run: IrisScientificRunV1,
    examples: tuple[IrisExample, ...],
) -> IrisScientificValidationV1:
    if type(run) is not IrisScientificRunV1 or run._busy:
        raise IrisScientificValidationError("Exact idle scientific run required")
    run._validate_boundary()
    rows, digest = own_scientific_validation_rows_v1(examples, training_rows=run._rows)
    model = run._optimizer._model
    parameters = run._optimizer._topology.parameters
    versions = tuple(parameter._version for parameter in parameters)
    caller = capture_runtime_state(run._optimizer._device)
    run._busy = True
    try:
        restore_runtime_state(run._runtime)
        model.eval()
        sums = []
        token_count = 0
        for row in rows:
            output = model.evaluation_forward(row, terminal_token_id=run._terminal_token_id)
            expected = domain_sha256(
                "iris/scientific-teacher-example/v1",
                canonical_json_bytes(row.model_dump(mode="json")),
            )
            if (
                type(output) is not IrisScientificTeacherOutputV1
                or output.example_id != row.id
                or output.example_sha256 != expected
                or output.configuration_sha256 != model.configuration_sha256
                or type(output.supervised_token_count) is not int
                or output.supervised_token_count <= 0
            ):
                raise IrisScientificValidationError("Validation row provenance/count differs")
            loss = output.losses.language_model
            if (
                type(loss) is not torch.Tensor
                or loss.dtype not in (torch.float32, torch.float64)
                or loss.ndim
                or loss.requires_grad
                or loss.grad_fn is not None
                or loss.device != run._optimizer._device
                or not bool(torch.isfinite(loss))
                or float(loss) < 0
            ):
                raise IrisScientificValidationError("Invalid detached validation target loss")
            sums.append(float(loss))
            token_count += output.supervised_token_count
            del output, loss
        total = math.fsum(sums)
        if not math.isfinite(total):
            raise IrisScientificValidationError("Nonfinite validation aggregation")
        return IrisScientificValidationV1(
            digest,
            len(rows),
            token_count,
            total,
            total / token_count,
            run.completed_steps,
            run.configuration_sha256,
        )
    except Exception:
        raise IrisScientificValidationError("Scientific validation could not complete") from None
    finally:
        try:
            model.train(True)
            restore_runtime_state(caller)
            if tuple(
                parameter._version for parameter in parameters
            ) != versions or not _same_runtime(
                capture_runtime_state(run._optimizer._device), caller
            ):
                raise IrisScientificValidationError(
                    "Validation mutated parameters or caller runtime"
                )
            run._validate_boundary()
        except BaseException:
            run._poisoned = True
            raise IrisScientificValidationError(
                "Scientific validation cleanup/ownership failed"
            ) from None
        finally:
            run._busy = False


__all__ = [
    "IrisScientificValidationError",
    "IrisScientificValidationV1",
    "evaluate_scientific_validation_v1",
    "own_scientific_validation_rows_v1",
]
