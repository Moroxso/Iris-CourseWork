"""Measure saved pairs' target NLL on their exact selected-best checkpoint.

Does not regenerate answers, execute candidate code, train or modify the source
capture. Hashes bind provenance/integrity, not external attestation of execution.
"""

import os
import shutil
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import torch

from iris.application.scientific_checkpoint import (
    _bounded_read,
    _json,
    _regular,
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_command import (
    ScientificParquetInputV1,
    _composed,
    _local_path,
    _preflight,
    read_scientific_command_config_v1,
)
from iris.application.scientific_evaluation import IrisScientificEvaluationError
from iris.application.scientific_execution import _checkpoint_for_event, _selected_best
from iris.application.scientific_inference import build_scientific_inference_v1
from iris.application.scientific_intervention import read_scientific_intervention_v1
from iris.application.scientific_intervention_evaluation import _audit_paired_source
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.scientific_teacher import (
    IrisScientificIntervenedTeacherOutputV1,
    IrisScientificTeacherOutputV1,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_intervention_nll_v1 import (
    IrisScientificInterventionNllV1,
    ScientificInterventionNllPayloadV1,
    ScientificInterventionNllRowV1,
    scientific_nll_groups_v1,
)
from iris.artifacts.scientific_intervention_v1 import intervention_plan_for_row_v1
from iris.interventions.policy import intervention_plan_sha256_v1
from iris.training.intervention import IrisIntervenedTeacherOutputV1, teacher_target_sha256_v1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-intervention-nll/v1"
_STORE = "scientific-intervention-nll-v1"
_FILE = "nll.json"
_MAX_BYTES = 16 * 1024**2


def _nll_sum(value: torch.Tensor, *, device: torch.device) -> float:
    if (
        type(value) is not torch.Tensor
        or value.ndim != 0
        or value.device != device
        or value.dtype not in (torch.float32, torch.float64)
        or value.requires_grad
        or value.grad_fn is not None
        or value.is_inference()
        or not bool(torch.isfinite(value))
        or float(value) < 0
    ):
        raise ValueError("An actual detached target NLL sum is required")
    return float(value)


def measure_scientific_intervention_nll_v1(
    *,
    intervention_path: Path,
    execution_path: Path,
    config_path: Path,
    test: ScientificParquetInputV1,
) -> IrisScientificInterventionNllV1:
    """Recompute control and intervened NLL for EVERY saved held-out pair.

    Requires exact source reconstruction and saved-control NLL reproduction.
    The source remains unchanged and its historical 'not_measured' flag remains
    truthful. This companion is not generated-answer scoring or launch approval.
    """
    try:
        if type(test) is not ScientificParquetInputV1:
            raise ValueError("Exact held-out input descriptor required")
        test = ScientificParquetInputV1.model_validate(test.model_dump(mode="python"))
        capture = read_scientific_intervention_v1(intervention_path)
        rows, snapshot, recipe = _audit_paired_source(
            capture,
            config_path=config_path,
            execution_path=execution_path,
            test=test,
        )
        config_path = _local_path(config_path)
        config = read_scientific_command_config_v1(config_path)
        selection = None
        for event in snapshot.events:
            selection = _selected_best(selection, event)
        _, best = _bundle_parts(
            _checkpoint_for_event(
                snapshot.path,
                snapshot.header,
                snapshot.events[-1],
                best_event=selection,
            )
        )
        if best is None:
            raise ValueError("Saved validation-selected best required")
        measured: list[ScientificInterventionNllRowV1] = []
        with _composed(config, base=config_path.parent) as lifecycle:
            run = lifecycle._run
            initial = build_scientific_checkpoint_v1(run)
            if (
                initial.manifest != snapshot.header.payload.initial
                or initial.state_bytes != snapshot.initial_state_bytes
                or lifecycle._configuration_json
                != snapshot.header.payload.lifecycle_configuration_json
                or lifecycle._configuration_sha256
                != snapshot.header.payload.lifecycle_configuration_sha256
            ):
                raise ValueError(
                    "Recipe did not reproduce the exact initial training/validation context"
                )
            _preflight(run._backbone, config, (), rows)
            restore_scientific_checkpoint_v1(best, run=run)
            inference = build_scientific_inference_v1(run)
            if inference.manifest != capture.payload.inference_manifest:
                raise ValueError("Restored best export differs from paired inference")
            model = run._optimizer._model
            backbone = run._backbone
            graph = domain_sha256(
                "iris/intervened-teacher-graph/v1",
                canonical_json_bytes(
                    {
                        "backbone": backbone.configuration_json,
                        "cognitive": config.cognitive.model_dump_json(),
                        "workspace": config.workspace.model_dump_json(),
                    }
                ),
            )
            sidecar = inference.manifest.payload.source_teacher_sidecar_sha256
            model.eval()
            try:
                for row, saved in zip(rows, capture.payload.baseline.payload.rows, strict=True):
                    bound = intervention_plan_for_row_v1(
                        capture.payload.plan,
                        example_id=row.id,
                        prompt_sha256=saved.prompt_sha256,
                    )
                    # Complete expected target binding independently of returned receipts.
                    prepared = prepare_teacher_forcing_input(
                        backbone=backbone,
                        prompt=row.prompt,
                        target_text=row.expected_response,
                        terminal_token_id=config.terminal_token_id,
                    )
                    target = teacher_target_sha256_v1(
                        prepared.target_input_ids, prepared.target_attention_mask
                    )
                    control = model.evaluation_forward(
                        row, terminal_token_id=config.terminal_token_id
                    )
                    changed = model.evaluation_forward_intervened(
                        row,
                        terminal_token_id=config.terminal_token_id,
                        plan=bound,
                    )
                    if (
                        type(control) is not IrisScientificTeacherOutputV1
                        or type(changed) is not IrisScientificIntervenedTeacherOutputV1
                    ):
                        raise ValueError("Exact scientific measurements required")
                    for output in (control, changed):
                        if (
                            output.example_id != row.id
                            or output.example_sha256 != saved.example_sha256
                            or output.configuration_sha256 != model.configuration_sha256
                            or type(output.supervised_token_count) is not int
                            or output.supervised_token_count != saved.target_token_count
                        ):
                            raise ValueError("Teacher row/configuration/token count differs")
                    teacher = changed.teacher
                    if (
                        type(teacher) is not IrisIntervenedTeacherOutputV1
                        or teacher.prompt_sha256
                        != domain_sha256(
                            "iris/intervened-teacher-prompt/v1", row.prompt.encode("utf-8")
                        )
                        or teacher.target_sha256 != target
                        or teacher.intervention_plan_sha256 != intervention_plan_sha256_v1(bound)
                        or teacher.teacher_graph_sha256 != graph
                        or teacher.sidecar_state_sha256 != sidecar
                        or teacher.supervised_token_count
                        != int(prepared.target_attention_mask.sum())
                        or not torch.equal(
                            changed.target_nll_sum, teacher.loss * teacher.supervised_token_count
                        )
                    ):
                        raise ValueError(
                            "Intervened teacher target/plan/graph/weights binding differs"
                        )
                    control_sum = _nll_sum(control.losses.language_model, device=backbone.device)
                    changed_sum = _nll_sum(changed.target_nll_sum, device=backbone.device)
                    if control_sum != saved.target_nll_sum:
                        raise ValueError("Saved control NLL was not exactly reproduced")
                    if not bound.interventions and changed_sum != control_sum:
                        raise ValueError("Empty-plan NLL differs from its actual control")
                    measured.append(
                        ScientificInterventionNllRowV1(
                            example_id=row.id,
                            example_sha256=saved.example_sha256,
                            prompt_sha256=saved.prompt_sha256,
                            target_sha256=target,
                            bound_plan_sha256=teacher.intervention_plan_sha256,
                            task_type=row.task_type,
                            difficulty=row.difficulty,
                            target_token_count=saved.target_token_count,
                            control_nll_sum=control_sum,
                            intervened_nll_sum=changed_sum,
                        )
                    )
                    del control, changed, teacher, prepared
            finally:
                model.train(True)
            if build_scientific_checkpoint_v1(run) != best:
                raise ValueError("NLL measurement changed the selected training state")
        final = _audit_paired_source(
            capture, config_path=config_path, execution_path=execution_path, test=test
        )
        if (
            final != (rows, snapshot, recipe)
            or read_scientific_intervention_v1(intervention_path) != capture
        ):
            raise ValueError("Capture/source/recipe/test changed during NLL measurement")
        source = capture.payload.baseline.payload
        records = tuple(measured)
        payload = ScientificInterventionNllPayloadV1(
            purpose=source.purpose,
            intervention_sha256=capture.sha256,
            baseline_generation_sha256=capture.payload.baseline.sha256,
            plan_sha256=capture.payload.plan_sha256,
            execution_sha256=source.execution_sha256,
            terminal_event_sha256=source.terminal_event_sha256,
            best_checkpoint_sha256=source.best_checkpoint_sha256,
            selected_step=source.selected_step,
            teacher_configuration_sha256=source.teacher_configuration_sha256,
            teacher_graph_sha256=graph,
            teacher_sidecar_sha256=sidecar,
            backbone_runtime_sha256=source.backbone_runtime_sha256,
            test_parquet_sha256=source.test_parquet_sha256,
            test_corpus_sha256=source.test_corpus_sha256,
            rows=records,
            groups=scientific_nll_groups_v1(records),
        )
        result = IrisScientificInterventionNllV1(
            sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
        _checked(result)
        return result
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific paired target NLL measurement did not complete"
        ) from None


def _checked(result: IrisScientificInterventionNllV1) -> bytes:
    if type(result) is not IrisScientificInterventionNllV1:
        raise ValueError("Exact paired NLL companion required")
    result = IrisScientificInterventionNllV1.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Paired NLL identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Paired NLL report exceeds its byte budget")
    return content


def read_scientific_intervention_nll_v1(path: Path) -> IrisScientificInterventionNllV1:
    """Verify canonical integrity and aggregates, not source audit or authenticity."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("NLL report inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificInterventionNllV1.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("NLL report canonical identity differs")
        return result
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific paired NLL report could not be read"
        ) from None


def write_scientific_intervention_nll_v1(
    result: IrisScientificInterventionNllV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> Path:
    """Publish a complete hash-only NLL companion in a separate bounded store."""
    temporary: Path | None = None
    try:
        content = _checked(result)
        if type(maximum_storage_bytes) is not int or not 1 <= maximum_storage_bytes <= 16 * 1024**3:
            raise ValueError("Explicit bounded NLL storage quota required")
        root = _local_path(root) / _STORE
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_intervention_nll_v1(target) != result:
                raise ValueError("Existing NLL report conflicts")
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
            raise ValueError("NLL store quota/free-space reserve exceeded")
        candidate = root / f".iris-intervention-nll-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged NLL report differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_intervention_nll_v1(target) != result:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific paired NLL report could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            (temporary / _FILE).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "measure_scientific_intervention_nll_v1",
    "read_scientific_intervention_nll_v1",
    "write_scientific_intervention_nll_v1",
]
