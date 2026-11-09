"""Private scientific generation/scoring command composition, not study aliases."""

import json
from contextlib import suppress
from pathlib import Path

from iris.application.errors import ApplicationError, IrisCodeEvaluationUnavailableError
from iris.application.scientific_checkpoint import _regular, _unique
from iris.application.scientific_command import (
    ScientificParquetInputV1,
    _input_bytes,
    _local_path,
    read_scientific_command_config_v1,
)
from iris.application.scientific_evaluation import (
    evaluate_scientific_generation_v1,
    write_scientific_evaluation_v1,
)
from iris.application.scientific_generation import (
    generate_scientific_test_v1,
    write_scientific_generation_v1,
)
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.scientific_evaluation_v1 import IrisScientificEvaluationV1
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificDecodingPolicyV1,
)
from iris.utils.hashing import canonical_json_bytes


class IrisScientificAnswerCommandError(ApplicationError):
    """Content-redacted command failure; source/capture remain independently retryable."""


def read_scientific_test_input_v1(path: Path) -> ScientificParquetInputV1:
    """Read <=16 KiB strict JSON; resolve Parquet relative to THIS descriptor file."""
    try:
        path = _local_path(path)
        raw = json.loads(_input_bytes(path, 16 * 1024).decode("utf-8"), object_pairs_hook=_unique)
        spec = ScientificParquetInputV1.model_validate_json(canonical_json_bytes(raw))
        if spec.maximum_rows > 5000:
            raise ValueError("Test row budget exceeds 5000")
        return ScientificParquetInputV1.model_validate(
            {**spec.model_dump(mode="python"), "path": str(_local_path(path.parent / spec.path))}
        )
    except Exception:
        raise IrisScientificAnswerCommandError(
            "Scientific test input descriptor is invalid"
        ) from None


def _destination(root: Path, kind: str, quota: int, protected: tuple[Path, ...]) -> None:
    if type(quota) is not int or not 1 <= quota <= 16 * 1024**3:
        raise ValueError("Explicit bounded storage quota required")
    root = _local_path(root)
    for parent in (root / kind, root, *root.parents):
        with suppress(FileNotFoundError):
            _regular(parent, directory=True)
    target = (root / kind).resolve()
    for source in protected:
        source = _local_path(source).resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("Output store must not overlap its immutable inputs")


def _recipe_inputs(path: Path) -> tuple[Path, ...]:
    config = read_scientific_command_config_v1(path)
    base = _local_path(path).parent
    return (path, base / config.train.path, base / config.validation.path)


def generate_scientific_answers_command_v1(
    *,
    config_path: Path,
    execution_path: Path,
    test_input_path: Path,
    output_root: Path,
    decoding: ScientificDecodingPolicyV1,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> tuple[Path, IrisScientificGenerationV1]:
    """Generate once, then persist private answers; never execute generated code."""
    try:
        test = read_scientific_test_input_v1(test_input_path)
        decoding = ScientificDecodingPolicyV1.model_validate(decoding.model_dump(mode="python"))
        _destination(
            output_root,
            "scientific-generation-v1",
            maximum_storage_bytes,
            (*_recipe_inputs(config_path), execution_path, test_input_path, Path(test.path)),
        )
        result = generate_scientific_test_v1(
            config_path=config_path, execution_path=execution_path, test=test, decoding=decoding
        )
        if read_scientific_test_input_v1(test_input_path) != test:
            raise ValueError("Test input descriptor changed during generation")
        path = write_scientific_generation_v1(
            result, root=output_root, maximum_storage_bytes=maximum_storage_bytes
        )
        return path, result
    except ApplicationError:
        raise
    except Exception:
        raise IrisScientificAnswerCommandError(
            "Scientific generation command did not complete"
        ) from None


def evaluate_scientific_answers_command_v1(
    *,
    config_path: Path,
    execution_path: Path,
    test_input_path: Path,
    generation_path: Path,
    output_root: Path,
    python_policy: PythonSandboxPolicyV1 | None,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> tuple[Path, IrisScientificEvaluationV1]:
    """Score and publish a complete saved capture without model loading/decoding."""
    try:
        test = read_scientific_test_input_v1(test_input_path)
        _destination(
            output_root,
            "scientific-evaluation-v1",
            maximum_storage_bytes,
            (
                *_recipe_inputs(config_path),
                execution_path,
                test_input_path,
                Path(test.path),
                generation_path,
            ),
        )
        result = evaluate_scientific_generation_v1(
            config_path=config_path,
            execution_path=execution_path,
            test=test,
            generation_path=generation_path,
            python_policy=python_policy,
        )
        if read_scientific_test_input_v1(test_input_path) != test:
            raise ValueError("Test input descriptor changed during evaluation")
        path = write_scientific_evaluation_v1(
            result, root=output_root, maximum_storage_bytes=maximum_storage_bytes
        )
        return path, result
    except IrisCodeEvaluationUnavailableError:
        raise
    except ApplicationError:
        raise
    except Exception:
        raise IrisScientificAnswerCommandError(
            "Scientific evaluation command did not complete"
        ) from None


__all__ = [
    "IrisScientificAnswerCommandError",
    "evaluate_scientific_answers_command_v1",
    "generate_scientific_answers_command_v1",
    "read_scientific_test_input_v1",
]
