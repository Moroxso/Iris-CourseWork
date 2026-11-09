"""Source-bound evaluation of saved paired outputs; never reload a backbone."""

import os
import shutil
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import save as save_tensors

from iris.application.errors import IrisCodeEvaluationUnavailableError
from iris.application.scientific_checkpoint import _bounded_read, _json, _regular, _verify
from iris.application.scientific_command import (
    ScientificParquetInputV1,
    _local_path,
    read_scientific_command_config_v1,
)
from iris.application.scientific_evaluation import IrisScientificEvaluationError, _audit_inputs
from iris.application.scientific_evaluation_v2 import _answer_measurements
from iris.application.scientific_execution import (
    IrisScientificExecutionSnapshotV1,
    _checkpoint_for_event,
    _selected_best,
)
from iris.application.scientific_inference import IrisScientificInferenceBundleV1
from iris.application.scientific_inference import _checked as _check_inference
from iris.application.scientific_intervention import read_scientific_intervention_v1
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.synthetic_scoring_v2 import score_synthetic_answer_v2
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_evaluation_v2 import ScientificPythonLimitsV2
from iris.artifacts.scientific_intervention_evaluation_v1 import (
    IrisScientificInterventionEvaluationV1,
    ScientificInterventionEvaluationPayloadV1,
    ScientificPairedScoredRowV1,
    paired_difficulties_v1,
    paired_families_v1,
    paired_family_difficulties_v1,
    scientific_paired_quality_v1,
)
from iris.artifacts.scientific_intervention_v1 import IrisScientificInterventionV1
from iris.data.schemas import IrisExample
from iris.training.probes import probe_vocabulary_sha256, synthetic_probe_vocabulary_v1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    tensor_sha256,
)

_DOMAIN = "iris/scientific-intervention-evaluation/v1"
_FILE = "evaluation.json"
_MAX_BYTES = 64 * 1024**2


def _audit_paired_source(
    capture: IrisScientificInterventionV1,
    *,
    config_path: Path,
    execution_path: Path,
    test: ScientificParquetInputV1,
) -> tuple[tuple[IrisExample, ...], IrisScientificExecutionSnapshotV1, bytes]:
    rows, snapshot = _audit_inputs(
        capture.payload.baseline, config_path=config_path, execution_path=execution_path, test=test
    )
    config_bytes = _bounded_read(_local_path(config_path), 128 * 1024)
    config = read_scientific_command_config_v1(config_path)
    selection = None
    for event in snapshot.events:
        selection = _selected_best(selection, event)
    _, best = _bundle_parts(
        _checkpoint_for_event(
            snapshot.path, snapshot.header, snapshot.events[-1], best_event=selection
        )
    )
    if best is None:
        raise ValueError("Saved selected-best checkpoint required")
    _, tensors = _verify(best)
    exported = capture.payload.inference_manifest
    model = exported.payload.model
    runtime = model.runtime
    if (
        exported.payload.source_run_configuration_sha256
        != best.manifest.payload.configuration_sha256
        or model.backbone != config.backbone
        or model.cognitive != config.cognitive
        or model.workspace != config.workspace
        or runtime.device != config.device
    ):
        raise ValueError("Paired export does not match the source run/recipe")
    # Recover the exact export from SAVED selected-best tensors, not from the
    # output capture's declarations or a newly initialized model.
    state = {
        name.removeprefix("teacher."): tensors[f"model/{name}"]
        for name in best.manifest.payload.model_state_names
        if name.startswith("teacher.")
    }
    if tuple(state) != model.state_names:
        raise ValueError("Paired export projection differs from selected-best teacher")
    _check_inference(
        IrisScientificInferenceBundleV1(
            exported,
            save_tensors(
                dict(sorted(state.items())), metadata={"format": "iris.scientific-inference.v1"}
            ),
        )
    )
    probe_configuration = {
        "kind": "iris-workspace-probe-heads-v1",
        "vocabulary_sha256": probe_vocabulary_sha256(synthetic_probe_vocabulary_v1()),
        "state_dim": config.cognitive.state_dim,
        "pooling": "mean_all_slots",
        "init_seed": config.probe_initialization_seed,
        "initialization_dtype": runtime.dtype,
        "initialization": "cpu_local_generator_linear_uniform_v1",
    }
    probe_sha = domain_sha256(
        "iris/workspace-probe-heads/v1", canonical_json_bytes(probe_configuration)
    )
    if probe_sha != bytes(tensors["model/probes.configuration_identity"].tolist()).hex():
        raise ValueError("Probe recipe differs from the saved source")
    teacher_configuration = {
        "kind": "iris-scientific-teacher-v1",
        "arm_id": "D",
        "policy": config.objective.model_dump(mode="json"),
        "probe_configuration_sha256": probe_sha,
        "backbone_configuration_json": config.backbone.model_dump_json(),
        "cognitive_configuration_json": config.cognitive.model_dump_json(),
        "workspace_configuration_json": config.workspace.model_dump_json(),
        "hidden_size": runtime.hidden_size,
        "num_hidden_layers": runtime.num_hidden_layers,
        "vocabulary_size": runtime.vocabulary_size,
        "max_position_embeddings": runtime.max_position_embeddings,
        "device": runtime.device,
        "dtype": runtime.dtype,
    }
    if (
        domain_sha256(
            "iris/scientific-teacher-configuration/v1", canonical_json_bytes(teacher_configuration)
        )
        != bytes(tensors["model/configuration_identity"].tolist()).hex()
    ):
        raise ValueError("Graph/runtime recipe differs from the checkpoint teacher identity")
    for generated in capture.payload.rows:
        initial_rng_sha = None
        if generated.sampling_seed is not None:
            initial_rng_sha = tensor_sha256(
                torch.Generator(device="cpu").manual_seed(generated.sampling_seed).get_state(),
                domain="iris/intervened-request/sampling-rng/v1",
            )
        if generated.initial_sampling_rng_sha256 != initial_rng_sha:
            raise ValueError("Recorded paired sampling RNG differs from its input-derived seed")
    if _bounded_read(_local_path(config_path), 128 * 1024) != config_bytes:
        raise ValueError("Recipe changed during source audit")
    return rows, snapshot, config_bytes


def evaluate_scientific_intervention_v1(
    *,
    intervention_path: Path,
    execution_path: Path,
    config_path: Path,
    test: ScientificParquetInputV1,
    python_policy: PythonSandboxPolicyV2 | None,
) -> IrisScientificInterventionEvaluationV1:
    """Score both saved branches; infrastructure failure aborts the entire report.

    Hashes bind declared outputs to source files, not external authenticity. No
    answer is regenerated, no intervened teacher NLL or scientific acceptance is
    inferred. Explicit row difficulty is not a substitute for a simple-control suite.
    """
    try:
        test = ScientificParquetInputV1.model_validate(test.model_dump(mode="python"))
        capture = read_scientific_intervention_v1(intervention_path)
        rows, snapshot, recipe = _audit_paired_source(
            capture, config_path=config_path, execution_path=execution_path, test=test
        )
        has_code = any(row.executable for row in rows)
        if has_code != (python_policy is not None):
            raise ValueError("Repair rows require an explicit sandbox policy and only repair rows")
        if python_policy is not None:
            if type(python_policy) is not PythonSandboxPolicyV2:
                raise ValueError("Exact source sandbox policy required")
            python_policy = PythonSandboxPolicyV2.model_validate(
                python_policy.model_dump(mode="python")
            )
            if python_policy.wall_time_ms != 5000:
                raise ValueError("Repair timeout must be exactly five seconds")
        scored = []
        for row, control, changed in zip(
            rows, capture.payload.baseline.payload.rows, capture.payload.rows, strict=True
        ):
            policy = python_policy if row.executable else None
            measurements = tuple(
                _answer_measurements(
                    row,
                    generation_example_sha256=generated.example_sha256,
                    output_text=generated.output_text,
                    score=score_synthetic_answer_v2(
                        row, output_text=generated.output_text, python_policy=policy
                    ),
                    policy=policy,
                )
                for generated in (control, changed)
            )
            scored.append(
                ScientificPairedScoredRowV1(
                    difficulty=row.difficulty,
                    bound_plan_sha256=changed.bound_plan_sha256,
                    control=measurements[0],
                    intervened=measurements[1],
                )
            )
        records = tuple(scored)
        final_rows, final_snapshot, final_recipe = _audit_paired_source(
            capture, config_path=config_path, execution_path=execution_path, test=test
        )
        if (final_rows, final_snapshot, final_recipe) != (
            rows,
            snapshot,
            recipe,
        ) or read_scientific_intervention_v1(intervention_path) != capture:
            raise ValueError("Paired capture/source/recipe/data changed during evaluation")
        source = capture.payload.baseline.payload
        payload = ScientificInterventionEvaluationPayloadV1(
            purpose=source.purpose,
            intervention_sha256=capture.sha256,
            baseline_generation_sha256=capture.payload.baseline.sha256,
            plan_sha256=capture.payload.plan_sha256,
            request_graph_sha256=capture.payload.request_graph_sha256,
            request_sidecar_sha256=capture.payload.request_sidecar_sha256,
            execution_sha256=source.execution_sha256,
            terminal_event_sha256=source.terminal_event_sha256,
            best_checkpoint_sha256=source.best_checkpoint_sha256,
            selected_step=source.selected_step,
            test_parquet_sha256=source.test_parquet_sha256,
            test_corpus_sha256=source.test_corpus_sha256,
            python_resource_scope="source_section15_v2" if has_code else "not_applicable_no_code",
            python_limits=ScientificPythonLimitsV2() if has_code else None,
            python_policy=python_policy,
            rows=records,
            overall=scientific_paired_quality_v1(records),
            families=paired_families_v1(records),
            difficulties=paired_difficulties_v1(records),
            family_difficulties=paired_family_difficulties_v1(records),
        )
        result = IrisScientificInterventionEvaluationV1(
            sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
        _checked(result)
        return result
    except IrisCodeEvaluationUnavailableError:
        raise IrisCodeEvaluationUnavailableError(
            "Scientific paired Python evaluation infrastructure is unavailable; "
            "no complete score was produced"
        ) from None
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific paired answer evaluation did not complete"
        ) from None


def _checked(result: IrisScientificInterventionEvaluationV1) -> bytes:
    if type(result) is not IrisScientificInterventionEvaluationV1:
        raise ValueError("Exact paired scientific evaluation required")
    result = IrisScientificInterventionEvaluationV1.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Paired evaluation identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Paired evaluation exceeds its byte budget")
    return content


def read_scientific_intervention_evaluation_v1(
    path: Path,
) -> IrisScientificInterventionEvaluationV1:
    """Check saved canonical integrity and counts, not external authenticity."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("Paired report inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificInterventionEvaluationV1.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("Paired report canonical identity differs")
        return result
    except Exception:
        raise IrisScientificEvaluationError("Scientific paired report could not be read") from None


def write_scientific_intervention_evaluation_v1(
    result: IrisScientificInterventionEvaluationV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> Path:
    """Publish a whole hash-only report in a distinct bounded content-addressed store."""
    temporary: Path | None = None
    try:
        content = _checked(result)
        if type(maximum_storage_bytes) is not int or not 1 <= maximum_storage_bytes <= 16 * 1024**3:
            raise ValueError("Explicit bounded paired report storage quota required")
        root = _local_path(root) / "scientific-intervention-evaluation-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_intervention_evaluation_v1(target) != result:
                raise ValueError("Existing paired report conflicts")
            return target
        occupied = 0
        for directory in root.iterdir():
            _regular(directory, directory=True)
            for file in directory.iterdir():
                _regular(file, directory=False)
                occupied += file.stat().st_size
        if (
            occupied + len(content) > maximum_storage_bytes
            or shutil.disk_usage(root).free < len(content) + 1024**3
        ):
            raise ValueError("Paired report quota/free-space reserve exceeded")
        candidate = root / f".iris-paired-evaluation-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged paired report differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_intervention_evaluation_v1(target) != result:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific paired report could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            (temporary / _FILE).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "IrisScientificEvaluationError",
    "evaluate_scientific_intervention_v1",
    "read_scientific_intervention_evaluation_v1",
    "write_scientific_intervention_evaluation_v1",
]
