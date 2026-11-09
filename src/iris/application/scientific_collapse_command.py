"""Explicit source-bound full-test collapse diagnostic command delivery."""

import json
from pathlib import Path

from iris.application.errors import ApplicationError
from iris.application.scientific_answer_command import (
    IrisScientificAnswerCommandError,
    _destination,
    _recipe_inputs,
    read_scientific_test_input_v1,
)
from iris.application.scientific_checkpoint import _unique
from iris.application.scientific_collapse import (
    measure_scientific_collapse_v1,
    write_scientific_collapse_v1,
)
from iris.application.scientific_command import _input_bytes, _local_path
from iris.artifacts.scientific_collapse_v1 import (
    IrisScientificCollapseV1,
    ScientificCollapsePolicyV1,
)
from iris.utils.hashing import canonical_json_bytes


def diagnose_scientific_collapse_command_v1(
    *,
    config_path: Path,
    execution_path: Path,
    test_input_path: Path,
    generation_path: Path,
    policy_path: Path,
    output_root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> tuple[Path, IrisScientificCollapseV1]:
    """Require explicit complete-coverage policy; recheck policy/descriptor before publishing."""
    try:
        policy_path, test_input_path = _local_path(policy_path), _local_path(test_input_path)
        policy_bytes = _input_bytes(policy_path, 16 * 1024)
        descriptor_bytes = _input_bytes(test_input_path, 16 * 1024)
        raw = json.loads(policy_bytes.decode("utf-8"), object_pairs_hook=_unique)
        policy = ScientificCollapsePolicyV1.model_validate_json(canonical_json_bytes(raw))
        test = read_scientific_test_input_v1(test_input_path)
        _destination(
            output_root,
            "scientific-collapse-v1",
            maximum_storage_bytes,
            (
                *_recipe_inputs(config_path),
                execution_path,
                generation_path,
                test_input_path,
                Path(test.path),
                policy_path,
            ),
        )
        result = measure_scientific_collapse_v1(
            config_path=config_path,
            execution_path=execution_path,
            generation_path=generation_path,
            test=test,
            policy=policy,
        )
        if (
            _input_bytes(policy_path, 16 * 1024) != policy_bytes
            or _input_bytes(test_input_path, 16 * 1024) != descriptor_bytes
            or read_scientific_test_input_v1(test_input_path) != test
        ):
            raise ValueError("Collapse policy/test descriptor changed during measurement")
        path = write_scientific_collapse_v1(
            result, root=output_root, maximum_storage_bytes=maximum_storage_bytes
        )
        return path, result
    except ApplicationError:
        raise
    except Exception:
        raise IrisScientificAnswerCommandError(
            "Scientific collapse command did not complete"
        ) from None


__all__ = ["diagnose_scientific_collapse_command_v1"]
