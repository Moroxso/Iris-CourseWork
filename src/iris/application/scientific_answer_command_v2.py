"""Explicit source-profile evaluation commands and strict version-dispatched reads."""

from pathlib import Path

from iris.application.errors import ApplicationError
from iris.application.scientific_answer_command import (
    IrisScientificAnswerCommandError,
    _destination,
    _recipe_inputs,
    read_scientific_test_input_v1,
)
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular
from iris.application.scientific_command import _local_path
from iris.application.scientific_evaluation import (
    IrisScientificEvaluationError,
    read_scientific_evaluation_v1,
)
from iris.application.scientific_evaluation_v2 import (
    evaluate_scientific_generation_v2,
    read_scientific_evaluation_v2,
    write_scientific_evaluation_v2,
)
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_evaluation_v1 import IrisScientificEvaluationV1
from iris.artifacts.scientific_evaluation_v2 import IrisScientificEvaluationV2


def evaluate_scientific_answers_command_v2(
    *,
    config_path: Path,
    execution_path: Path,
    test_input_path: Path,
    generation_path: Path,
    output_root: Path,
    python_policy: PythonSandboxPolicyV2 | None,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> tuple[Path, IrisScientificEvaluationV2]:
    """Score saved answers with source limits, never mutate/relabel old v1 reports."""
    try:
        test = read_scientific_test_input_v1(test_input_path)
        _destination(
            output_root,
            "scientific-evaluation-v2",
            maximum_storage_bytes,
            (
                *_recipe_inputs(config_path),
                execution_path,
                test_input_path,
                Path(test.path),
                generation_path,
            ),
        )
        result = evaluate_scientific_generation_v2(
            config_path=config_path,
            execution_path=execution_path,
            test=test,
            generation_path=generation_path,
            python_policy=python_policy,
        )
        if read_scientific_test_input_v1(test_input_path) != test:
            raise ValueError("Test descriptor changed during evaluation")
        path = write_scientific_evaluation_v2(
            result, root=output_root, maximum_storage_bytes=maximum_storage_bytes
        )
        return path, result
    except ApplicationError:
        raise
    except Exception:
        raise IrisScientificAnswerCommandError(
            "Scientific source-profile evaluation did not complete"
        ) from None


def read_scientific_evaluation_report(
    path: Path,
) -> IrisScientificEvaluationV1 | IrisScientificEvaluationV2:
    """Read declared artifact version, not trial downgrade/fallback or source audit."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        content = _bounded_read(path / "evaluation.json", 32 * 1024**2)
        if not content.endswith(b"\n"):
            raise ValueError("Canonical evaluation requires a terminal newline")
        raw = _json(content[:-1])
        kind = raw["payload"]["kind"]
        if kind == "iris-scientific-evaluation-v1":
            return read_scientific_evaluation_v1(path)
        if kind == "iris-scientific-evaluation-v2":
            return read_scientific_evaluation_v2(path)
        raise ValueError("Unknown evaluation version")
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific evaluation report version or integrity is invalid"
        ) from None


__all__ = ["evaluate_scientific_answers_command_v2", "read_scientific_evaluation_report"]
