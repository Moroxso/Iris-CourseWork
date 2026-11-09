"""Model-free source/data/capture-bound v2 scoring; no generation or host execution."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_ablation_controlled_capture import (
    IrisScientificAblationControlledCaptureProgressV1,
    read_scientific_ablation_controlled_capture_v1,
)
from iris.application.scientific_ablation_generation_source import (
    read_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_heldout import _own_rows
from iris.application.scientific_ablation_inference import _digest, _exact_tree
from iris.application.scientific_ablation_inference_io import _chain, _regular, _signature
from iris.application.scientific_ablation_validation import (
    _complete_row_fields,
    own_scientific_ablation_validation_rows_v1,
)
from iris.application.scientific_command import ScientificParquetInputV1, _rows
from iris.application.scientific_evaluation_v2 import _answer_measurements
from iris.application.synthetic_scoring_v2 import score_synthetic_answer_v2
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_controlled_capture_v1 import (
    IrisScientificAblationControlledCaptureCompleteV1,
    IrisScientificAblationControlledCapturePlanV1,
)
from iris.artifacts.scientific_ablation_controlled_evaluation_v1 import (
    IrisScientificAblationControlledEvaluationV1,
    ScientificAblationControlledEvaluationPayloadV1,
    ScientificAblationControlledScoredRowV1,
)
from iris.artifacts.scientific_evaluation_v1 import (
    TASKS,
    ScientificFamilyQualityV1,
    ScientificQualityCountsV1,
    scientific_quality_counts_v1,
)
from iris.artifacts.scientific_evaluation_v2 import ScientificPythonLimitsV2
from iris.data.schemas import IrisExample
from iris.integrations.python_sandbox import _worker
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-ablation-controlled-evaluation/v1"
_MAX_REPORT = 32 * 1024**2


class IrisScientificAblationControlledEvaluationError(RuntimeError):
    """No partial or inconsistent mode-bound evaluation is a complete score."""


def _schema_wire_bound(model: type[BaseModel], overrides: dict[str, int]) -> int:
    """Conservative canonical byte arithmetic; never creates a measurement/receipt.

    Only the closed schema forms used by the trusted hash-only score models are
    supported. Unknown/unbounded shapes fail closed and require explicit review.
    Bounds count EVERY declared field and both nullable alternatives, not merely
    the common invalid-envelope/no-execution branch of a tiny fixture.
    """
    schema = model.model_json_schema()
    if set(schema["properties"]) != set(model.model_fields):
        raise ValueError("Score serializer/schema fields differ; byte bound needs review")
    definitions = schema.get("$defs", {})

    def size(node: dict[str, Any], field: str = "") -> int:
        if field in overrides:
            return overrides[field]
        if "$ref" in node:
            reference = node["$ref"]
            if type(reference) is not str or not reference.startswith("#/$defs/"):
                raise ValueError("Only exact local score schema references supported")
            return size(definitions[reference.removeprefix("#/$defs/")])
        if "anyOf" in node:
            return max(size(member, field) for member in node["anyOf"])
        if "const" in node:
            return len(canonical_json_bytes(node["const"]))
        if "enum" in node:
            return max(len(canonical_json_bytes(value)) for value in node["enum"])
        kind = node.get("type")
        if kind == "null":
            return 4
        if kind == "boolean":
            return 5
        if kind == "number":
            return 128  # Entire spelling of any finite nonnegative binary64.
        if kind == "integer":
            maximum, minimum = node.get("maximum"), node.get("minimum", 0)
            if type(maximum) is not int or type(minimum) is not int:
                raise ValueError("Unbounded integer score field needs an explicit derived bound")
            return max(len(str(maximum)), len(str(minimum)))
        if kind == "string":
            if node.get("pattern") == r"^[0-9a-f]{64}$":
                return 66
            if node.get("pattern") == r"^sha256:[0-9a-f]{64}$":
                return 73
            maximum = node.get("maxLength")
            if type(maximum) is not int or maximum < 0:
                raise ValueError("Unbounded string score field needs an explicit derived bound")
            return 2 + 6 * maximum  # Includes worst JSON escaping, not just UTF-8.
        if kind == "object" and node.get("additionalProperties") is False:
            properties = node["properties"]
            return (
                2
                + max(0, len(properties) - 1)
                + sum(
                    len(canonical_json_bytes(name)) + 1 + size(member, name)
                    for name, member in properties.items()
                )
            )
        raise ValueError("Unknown score schema shape; complete byte preflight must be reviewed")

    return size(schema)


def _score_wire_limits(
    plan: IrisScientificAblationControlledCapturePlanV1, policy: PythonSandboxPolicyV2 | None
) -> dict[str, int]:
    maximum_output = policy.max_output_bytes if policy is not None else 8 * 1024**2
    return {
        "example_id": max(len(canonical_json_bytes(row.example_id)) for row in plan.payload.rows),
        "stdout_byte_length": len(str(maximum_output)),
        "stderr_byte_length": len(str(maximum_output)),
    }


def _report_file_bound(
    plan: IrisScientificAblationControlledCapturePlanV1,
    capture: IrisScientificAblationControlledCaptureCompleteV1,
    policy: PythonSandboxPolicyV2 | None,
) -> int:
    """Complete report bound BEFORE sandbox work; no clipping or fake score data."""
    expected = plan.payload.rows
    repair_count = sum(row.task_type == "split_spec_repair" for row in expected)
    scalar_counts = {
        "examples",
        "correct",
        "incorrect",
        "invalid_output",
        "envelope_valid",
        "structured_exact_match",
        "code_examples",
        "code_executed",
        "code_correct",
        "code_cases",
        "code_cases_passed",
        "tool_examples",
        "tool_action_correct",
        "conflict_examples",
        "conflict_flag_correct",
        "target_tokens",
    }
    if set(ScientificQualityCountsV1.model_fields) != scalar_counts | {"target_nll_sum"}:
        raise ValueError("Quality-count schema changed; complete byte bound needs review")
    # All counts <=max(5000*128, full original target count)<=2**48. Hash-only
    # evidence contains aggregate case counts, NOT per-case arrays or raw output.
    count_limits = dict.fromkeys(scalar_counts, len(str(2**48)))
    quality_bound = _schema_wire_bound(ScientificQualityCountsV1, count_limits)
    family_bound = _schema_wire_bound(ScientificFamilyQualityV1, count_limits)
    row_limits = _score_wire_limits(plan, policy)
    code_bound = _schema_wire_bound(ScientificAblationControlledScoredRowV1, row_limits)
    plain_bound = _schema_wire_bound(
        ScientificAblationControlledScoredRowV1, {**row_limits, "execution": 4}
    )
    base: dict[str, object] = dict(
        kind="iris-scientific-ablation-controlled-evaluation-v1",
        scoring_profile="iris-synthetic-01-scoring-v2",
        scientific_launch_status="not_assessed",
        nll_scope="copied_from_bound_capture_not_recomputed",
        generation_control_plan_sha256=plan.payload.input.payload.generation_control_plan_sha256,
        target_nll_semantics="selected_best_cognitive_mode_without_workspace_control",
        plan=plan.model_dump(mode="json"),
        capture=capture.model_dump(mode="json"),
        python_resource_scope="source_section15_v2" if repair_count else "not_applicable_no_code",
        python_limits=ScientificPythonLimitsV2().model_dump(mode="json") if repair_count else None,
        python_policy=policy.model_dump(mode="json") if policy is not None else None,
        rows=[],
        overall=None,
        families=[],
    )
    if set(base) != set(ScientificAblationControlledEvaluationPayloadV1.model_fields):
        raise ValueError("Complete report schema changed; byte preflight must be reviewed")
    bound = (
        len(canonical_json_file_bytes(dict(sha256="f" * 64, payload=base)))
        + repair_count * code_bound
        + (len(expected) - repair_count) * plain_bound
        + len(expected)
        - 1
        + quality_bound
        - 4
        + 4 * family_bound
        + 3
    )
    if not 1 <= bound <= _MAX_REPORT:
        raise ValueError("Complete controlled evaluation exceeds report quota BEFORE scoring")
    return bound


def _training(rows: tuple[IrisExample, ...]) -> tuple[tuple[IrisExample, ...], str]:
    if type(rows) is not tuple or not 1 <= len(rows) <= 200000:
        raise ValueError("Complete original training corpus required")
    digest = sha256(b"iris/scientific-train-corpus/v1\0")
    size = 0
    owned: list[IrisExample] = []
    identifiers: set[str] = set()
    for row in rows:
        if not _complete_row_fields(row) or row.split != "train" or row.id in identifiers:
            raise ValueError("Exact unique original training rows required")
        row = IrisExample.model_validate(row.model_dump(mode="python", warnings=False))
        content = canonical_json_bytes(row.model_dump(mode="json"))
        size += len(content)
        if size > 512 * 1024**2:
            raise ValueError("Original training logical byte limit exceeded")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        identifiers.add(row.id)
        owned.append(row)
    return tuple(owned), digest.hexdigest()


def _audit(
    capture: IrisScientificAblationControlledCaptureProgressV1,
    *,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
) -> tuple[IrisExample, ...]:
    source = capture.plan.payload.source
    s, p = source.payload, capture.plan.payload
    # Private native tensor verification requires ordinary CPU tensors, independent
    # of the caller's default device/inference mode. Keep all existing validators.
    with torch.inference_mode(False), torch.device("cpu"):
        actual = read_scientific_ablation_generation_source_v1(
            lifecycle_path=lifecycle_path,
            inference_path=inference_path,
            expected_lifecycle_sha256=s.lifecycle_sha256,
            expected_inference_sha256=s.inference_sha256,
            expected_source_checkpoint_sha256=s.best_checkpoint_sha256,
        )
    if actual.manifest != source:
        raise ValueError("Persisted source differs from independently trusted capture plan")
    train, train_sha = _training(_rows(training, inputs_root))
    valid, valid_sha = own_scientific_ablation_validation_rows_v1(
        _rows(validation, inputs_root), training_rows=train
    )
    if (
        train_sha != s.training_corpus_sha256
        or valid_sha != s.validation_corpus_sha256
        or len(train) != s.training_examples
        or len(valid) != s.validation_examples
        or test.sha256 != p.input.payload.test_parquet_sha256
    ):
        raise ValueError("Original train/validation/test bindings differ")
    rows, corpus_sha = _own_rows(
        _rows(test, inputs_root),
        excluded_seeds={row.seed for row in (*train, *valid)},
        scientific=s.purpose == "scientific",
    )
    if corpus_sha != p.input.payload.test_corpus_sha256 or len(rows) != len(p.rows):
        raise ValueError("Complete original test corpus differs")
    for row, expected in zip(rows, p.rows, strict=True):
        if (
            row.id != expected.example_id
            or row.task_type != expected.task_type
            or sha256(row.prompt.encode("utf-8")).hexdigest() != expected.prompt_sha256
            or domain_sha256(
                "iris/scientific-ablation-example/v1",
                canonical_json_bytes(row.model_dump(mode="json")),
            )
            != expected.example_sha256
        ):
            raise ValueError("Every scoring row must match the actual generation inventory")
    return rows


def _checked(result: IrisScientificAblationControlledEvaluationV1) -> bytes:
    _exact_tree(result, IrisScientificAblationControlledEvaluationV1)
    result = IrisScientificAblationControlledEvaluationV1.model_validate(
        result.model_dump(mode="python", warnings=False)
    )
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Exact mode-bound evaluation checksum differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_REPORT:
        raise ValueError("Bounded mode-bound evaluation report required")
    return content


def _source_signatures(
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    specs: tuple[ScientificParquetInputV1, ...],
) -> dict[Path, tuple[int, ...]]:
    result: dict[Path, tuple[int, ...]] = {}
    for directory, names in (
        (lifecycle_path, {"manifest.json", "current.safetensors", "best.safetensors"}),
        (inference_path, {"manifest.json", "state.safetensors"}),
    ):
        _chain(directory)
        if {path.name for path in directory.iterdir()} != names:
            raise ValueError("Exact immutable source directory required")
        result[directory] = _signature(_regular(directory, directory=True))
        for name in names:
            path = directory / name
            result[path] = _signature(_regular(path, directory=False))
    for spec in specs:
        path = inputs_root / spec.path
        _chain(path.parent)
        result[path] = _signature(_regular(path, directory=False))
    return result


@dataclass(frozen=True, slots=True)
class _EvaluationInputs:
    capture: IrisScientificAblationControlledCaptureProgressV1
    rows: tuple[IrisExample, ...]
    lifecycle_path: Path
    inference_path: Path
    inputs_root: Path
    specs: tuple[ScientificParquetInputV1, ScientificParquetInputV1, ScientificParquetInputV1]
    policy: PythonSandboxPolicyV2 | None
    worker_sha256: str | None
    maximum_capture_bytes: int
    source_signatures: dict[Path, tuple[int, ...]]
    maximum_report_file_bytes: int

    def check(self) -> None:
        if _source_signatures(
            self.lifecycle_path, self.inference_path, self.inputs_root, self.specs
        ) != self.source_signatures or (
            self.worker_sha256 is not None
            and sha256(_worker("python_worker_v2.txt")).hexdigest() != self.worker_sha256
        ):
            raise ValueError("Source/data/worker identity changed during evaluation")


def _prepare(
    *,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV2 | None,
    maximum_capture_bytes: int,
) -> _EvaluationInputs:
    _digest(expected_plan_sha256)
    _digest(expected_capture_sha256)
    owned: list[ScientificParquetInputV1] = []
    for spec in (training, validation, test):
        _exact_tree(spec, ScientificParquetInputV1)
        owned.append(ScientificParquetInputV1.model_validate(spec.model_dump(mode="python")))
    training, validation, test = owned
    specs = (training, validation, test)
    capture = read_scientific_ablation_controlled_capture_v1(
        capture_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_capture_sha256=expected_capture_sha256,
        maximum_storage_bytes=maximum_capture_bytes,
    )
    if capture.complete is None:
        raise ValueError("Complete acknowledged capture required")
    before = _source_signatures(lifecycle_path, inference_path, inputs_root, specs)
    rows = _audit(
        capture,
        lifecycle_path=lifecycle_path,
        inference_path=inference_path,
        inputs_root=inputs_root,
        training=training,
        validation=validation,
        test=test,
    )
    repair = any(row.executable for row in rows)
    if repair != (python_policy is not None):
        raise ValueError("Explicit source-v2 policy required exactly for repair corpus")
    if python_policy is not None:
        _exact_tree(python_policy, PythonSandboxPolicyV2)
        python_policy = PythonSandboxPolicyV2.model_validate(
            python_policy.model_dump(mode="python")
        )
    report_bound = _report_file_bound(capture.plan, capture.complete, python_policy)
    result = _EvaluationInputs(
        capture,
        rows,
        lifecycle_path,
        inference_path,
        inputs_root,
        specs,
        python_policy,
        sha256(_worker("python_worker_v2.txt")).hexdigest() if repair else None,
        maximum_capture_bytes,
        before,
        report_bound,
    )
    result.check()
    return result


def _score_row(inputs: _EvaluationInputs, index: int) -> ScientificAblationControlledScoredRowV1:
    inputs.check()
    row = inputs.rows[index]
    generated, ack = inputs.capture.rows[index], inputs.capture.acknowledgements[index]
    g = generated.payload
    expected = inputs.capture.plan.payload.rows[index]

    def check_row() -> None:
        if (
            domain_sha256(
                "iris/scientific-ablation-example/v1",
                canonical_json_bytes(row.model_dump(mode="json")),
            )
            != expected.example_sha256
        ):
            raise ValueError("Owned scoring example changed")

    check_row()
    policy = inputs.policy if row.executable else None
    score = score_synthetic_answer_v2(row, output_text=g.output_text, python_policy=policy)
    measurements = _answer_measurements(
        row,
        generation_example_sha256=g.example_sha256,
        output_text=g.output_text,
        score=score,
        policy=policy,
    )
    if (
        measurements.execution is not None
        and measurements.execution.worker_sha256 != inputs.worker_sha256
    ):
        raise ValueError("Receipt worker differs from originally owned evaluator")
    result = ScientificAblationControlledScoredRowV1(
        **measurements.model_dump(mode="python"),
        row_index=g.row_index,
        capture_row_sha256=generated.sha256,
        capture_ack_sha256=ack.sha256,
        teacher_invocation_sha256=g.teacher_invocation_sha256,
        teacher_target_sha256=g.teacher_target_sha256,
        target_token_count=g.target_token_count,
        target_nll_sum=g.target_nll_sum,
    )
    check_row()
    inputs.check()
    return result


def _finish(inputs: _EvaluationInputs) -> None:
    """O(corpus + capture) deep audit twice per invocation, not once per row."""
    inputs.check()
    if (
        read_scientific_ablation_controlled_capture_v1(
            inputs.capture.path,
            expected_plan_sha256=inputs.capture.plan.sha256,
            expected_capture_sha256=inputs.capture.complete.sha256
            if inputs.capture.complete
            else None,
            maximum_storage_bytes=inputs.maximum_capture_bytes,
        )
        != inputs.capture
        or _audit(
            inputs.capture,
            lifecycle_path=inputs.lifecycle_path,
            inference_path=inputs.inference_path,
            inputs_root=inputs.inputs_root,
            training=inputs.specs[0],
            validation=inputs.specs[1],
            test=inputs.specs[2],
        )
        != inputs.rows
    ):
        raise ValueError("Source/data/capture changed during scoring")
    inputs.check()


def _assemble(
    inputs: _EvaluationInputs, records: tuple[ScientificAblationControlledScoredRowV1, ...]
) -> IrisScientificAblationControlledEvaluationV1:
    if inputs.capture.complete is None:
        raise ValueError("Complete acknowledged capture required")
    return _assemble_data(
        inputs.capture.plan,
        inputs.capture.complete,
        inputs.policy,
        records,
        maximum_report_file_bytes=inputs.maximum_report_file_bytes,
    )


def _assemble_data(
    plan: IrisScientificAblationControlledCapturePlanV1,
    capture: IrisScientificAblationControlledCaptureCompleteV1,
    policy: PythonSandboxPolicyV2 | None,
    records: tuple[ScientificAblationControlledScoredRowV1, ...],
    *,
    maximum_report_file_bytes: int,
) -> IrisScientificAblationControlledEvaluationV1:
    """Exact original bounded aggregate from verified plan/measurements, no scorer."""
    if (
        type(maximum_report_file_bytes) is not int
        or not 1 <= maximum_report_file_bytes <= _MAX_REPORT
    ):
        raise ValueError("Exact bounded original report preflight required")
    repair = policy is not None
    payload = ScientificAblationControlledEvaluationPayloadV1(
        plan=plan,
        capture=capture,
        generation_control_plan_sha256=(plan.payload.input.payload.generation_control_plan_sha256),
        python_resource_scope="source_section15_v2" if repair else "not_applicable_no_code",
        python_limits=ScientificPythonLimitsV2() if repair else None,
        python_policy=policy,
        rows=records,
        overall=scientific_quality_counts_v1(records),
        families=tuple(
            ScientificFamilyQualityV1(
                task_type=task,
                counts=scientific_quality_counts_v1(
                    tuple(row for row in records if row.task_type == task)
                ),
            )
            for task in TASKS
        ),
    )
    result = IrisScientificAblationControlledEvaluationV1(
        sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    if len(_checked(result)) > maximum_report_file_bytes:
        raise ValueError("Actual complete controlled evaluation exceeds preflight byte bound")
    return result


def evaluate_scientific_ablation_controlled_capture_v1(
    *,
    capture_path: Path,
    expected_plan_sha256: str,
    expected_capture_sha256: str,
    lifecycle_path: Path,
    inference_path: Path,
    inputs_root: Path,
    training: ScientificParquetInputV1,
    validation: ScientificParquetInputV1,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV2 | None,
    maximum_capture_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationControlledEvaluationV1:
    """Score all verified answers; never instantiate a model or regenerate a row.

    Trusted plan/complete SHA must come independently from the operator/producer.
    Native state tensors are authenticated by the existing source reader, but no
    backbone/teacher/request graph is constructed. NLL is not recomputed here.
    Infrastructure failure withholds the whole report; it is not model quality.
    """
    try:
        inputs = _prepare(
            capture_path=capture_path,
            expected_plan_sha256=expected_plan_sha256,
            expected_capture_sha256=expected_capture_sha256,
            lifecycle_path=lifecycle_path,
            inference_path=inference_path,
            inputs_root=inputs_root,
            training=training,
            validation=validation,
            test=test,
            python_policy=python_policy,
            maximum_capture_bytes=maximum_capture_bytes,
        )
        records = tuple(_score_row(inputs, index) for index in range(len(inputs.rows)))
        _finish(inputs)
        return _assemble(inputs, records)
    except IrisCodeEvaluationUnavailableError:
        raise IrisCodeEvaluationUnavailableError(
            "Mode-bound Python isolation unavailable; no complete evaluation produced"
        ) from None
    except Exception:
        raise IrisScientificAblationControlledEvaluationError(
            "Mode-bound answer evaluation did not complete"
        ) from None


__all__ = [
    "IrisScientificAblationControlledEvaluationError",
    "evaluate_scientific_ablation_controlled_capture_v1",
]
