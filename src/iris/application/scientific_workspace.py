"""Measure a bounded workspace subset on the exact validation-selected source.

No training, answer regeneration, generated-code execution or launch approval.
Full traces are opt-in latent data and are not claimed to be anonymized.
"""

import os
import shutil
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

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
from iris.application.scientific_evaluation import IrisScientificEvaluationError, _audit_inputs
from iris.application.scientific_execution import _checkpoint_for_event, _selected_best
from iris.application.scientific_generation import read_scientific_generation_v1
from iris.application.scientific_inference import build_scientific_inference_v1
from iris.application.scientific_intervention_nll import _nll_sum
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.scientific_teacher import IrisScientificTeacherOutputV1
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.application.training_checkpoint import _write_exact_file
from iris.application.workspace_causal import measure_workspace_causal_v1
from iris.artifacts.scientific_workspace_v1 import (
    IrisScientificWorkspaceV1,
    ScientificWorkspaceMemberV1,
    ScientificWorkspacePayloadV1,
    ScientificWorkspacePolicyV1,
    ScientificWorkspaceRowV1,
    scientific_workspace_groups_v1,
    select_workspace_members_v1,
    workspace_execution_budget_v1,
)
from iris.training.intervention import teacher_target_sha256_v1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.workspace.causal import summarize_workspace_causal_v1
from iris.workspace.trace import WorkspaceTraceOptionsV1, WorkspaceTraceV1

_DOMAIN = "iris/scientific-workspace/v1"
_STORE = "scientific-workspace-v1"
_FILE = "workspace.json"
_MAX_BYTES = 64 * 1024**2


def measure_scientific_workspace_v1(
    *,
    generation_path: Path,
    execution_path: Path,
    config_path: Path,
    test: ScientificParquetInputV1,
    policy: ScientificWorkspacePolicyV1,
) -> IrisScientificWorkspaceV1:
    """Audit the ENTIRE test membership; select without looking at answers/effects.

    Reconstruct the original initialization, restore best (never latest), reproduce
    each selected row's saved NLL, and check source/state immutability afterward.
    The fixed stratified subset is descriptive, not population-weighted evidence.
    """
    try:
        if (
            type(test) is not ScientificParquetInputV1
            or type(policy) is not ScientificWorkspacePolicyV1
        ):
            raise ValueError("Exact test descriptor and diagnostic policy required")
        test = ScientificParquetInputV1.model_validate(test)
        policy = ScientificWorkspacePolicyV1.model_validate(policy)
        config_path = _local_path(config_path)
        recipe = _bounded_read(config_path, 128 * 1024)
        capture = read_scientific_generation_v1(generation_path)
        rows, snapshot = _audit_inputs(
            capture, config_path=config_path, execution_path=execution_path, test=test
        )
        config = read_scientific_command_config_v1(config_path)
        source = capture.payload
        members = tuple(
            ScientificWorkspaceMemberV1(
                example_id=row.id,
                example_sha256=saved.example_sha256,
                prompt_sha256=saved.prompt_sha256,
                teacher_prompt_sha256=domain_sha256(
                    "iris/intervened-teacher-prompt/v1", row.prompt.encode("utf-8")
                ),
                task_type=row.task_type,
                difficulty=row.difficulty,
            )
            for row, saved in zip(rows, source.rows, strict=True)
        )
        selected = select_workspace_members_v1(members, policy)
        executions, scalar_bound = workspace_execution_budget_v1(
            policy,
            examples=len(selected),
            modules=config.cognitive.num_modules,
            slots=config.workspace.num_slots,
            width=config.cognitive.state_dim,
            cycles=config.workspace.cycles,
            writers=config.workspace.winners_per_cycle,
        )
        owned = {row.id: row for row in rows}
        saved_rows = {row.example_id: row for row in source.rows}
        selection = None
        for event in snapshot.events:
            selection = _selected_best(selection, event)
        _, best = _bundle_parts(
            _checkpoint_for_event(
                snapshot.path, snapshot.header, snapshot.events[-1], best_event=selection
            )
        )
        if best is None:
            raise ValueError("A saved validation-selected best checkpoint is required")
        measured = []
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
                raise ValueError("Recipe did not reproduce exact source initialization/context")
            _preflight(run._backbone, config, (), tuple(owned[m.example_id] for m in selected))
            restore_scientific_checkpoint_v1(best, run=run)
            inference = build_scientific_inference_v1(run)
            if inference.manifest.sha256 != source.inference_sha256:
                raise ValueError("Best checkpoint export differs from saved baseline")
            backbone = run._backbone
            model = run._optimizer._model
            sidecar = inference.manifest.payload.source_teacher_sidecar_sha256
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
            model.eval()
            try:
                if model.teacher.evaluation_provenance.sidecar_state_sha256 != sidecar:
                    raise ValueError("Actual selected teacher differs from exported sidecar")
                for member in selected:
                    row, saved = owned[member.example_id], saved_rows[member.example_id]
                    options = WorkspaceTraceOptionsV1(
                        mode=policy.trace_mode,
                        example_id=row.id,
                        checkpoint_id=best.manifest.sha256,
                        seed=config.run_seed,
                        maximum_scalars=policy.maximum_trace_scalars,
                    )
                    prepared = prepare_teacher_forcing_input(
                        backbone=backbone,
                        prompt=row.prompt,
                        target_text=row.expected_response,
                        terminal_token_id=config.terminal_token_id,
                    )
                    output = model.evaluation_forward(
                        row, terminal_token_id=config.terminal_token_id, diagnostics=options
                    )
                    if (
                        type(output) is not IrisScientificTeacherOutputV1
                        or output.example_id != row.id
                        or output.example_sha256 != saved.example_sha256
                        or output.configuration_sha256 != source.teacher_configuration_sha256
                        or type(output.supervised_token_count) is not int
                        or output.supervised_token_count != saved.target_token_count
                        or output.supervised_token_count
                        != int(prepared.target_attention_mask.sum())
                    ):
                        raise ValueError(
                            "Diagnostic teacher row/configuration/target binding differs"
                        )
                    trace = output.trace.teacher.diagnostic_trace
                    control_nll = _nll_sum(output.losses.language_model, device=backbone.device)
                    if (
                        control_nll != saved.target_nll_sum
                        or type(trace) is not WorkspaceTraceV1
                        or trace.options != options
                        or (trace.num_modules, trace.num_slots, trace.state_dim, trace.cycle_count)
                        != (
                            config.cognitive.num_modules,
                            config.workspace.num_slots,
                            config.cognitive.state_dim,
                            config.workspace.cycles,
                        )
                    ):
                        raise ValueError("Saved control NLL or diagnostic graph was not reproduced")
                    causal = (
                        measure_workspace_causal_v1(
                            model.teacher,
                            prepared,
                            example_id=row.id,
                            seed=config.run_seed,
                            policy=policy.causal,
                        )
                        if policy.causal is not None
                        else None
                    )
                    measured.append(
                        ScientificWorkspaceRowV1(
                            example_id=row.id,
                            target_sha256=teacher_target_sha256_v1(
                                prepared.target_input_ids, prepared.target_attention_mask
                            ),
                            target_token_count=saved.target_token_count,
                            control_nll_sum=control_nll,
                            trace=trace,
                            causal=causal,
                        )
                    )
                    del output, prepared, trace, causal
            finally:
                model.train(True)
            if build_scientific_checkpoint_v1(run) != best:
                raise ValueError("Diagnostics changed the selected training state")
        if (
            _audit_inputs(
                capture, config_path=config_path, execution_path=execution_path, test=test
            )
            != (rows, snapshot)
            or _bounded_read(config_path, 128 * 1024) != recipe
            or read_scientific_generation_v1(generation_path) != capture
        ):
            raise ValueError("Source execution/recipe/test/capture changed during diagnostics")
        records = tuple(measured)
        captures = tuple(row.causal for row in records if row.causal is not None)
        payload = ScientificWorkspacePayloadV1(
            purpose=source.purpose,
            policy=policy,
            baseline_generation_sha256=capture.sha256,
            execution_sha256=source.execution_sha256,
            terminal_event_sha256=source.terminal_event_sha256,
            best_validation_event_sha256=source.best_validation_event_sha256,
            best_checkpoint_sha256=source.best_checkpoint_sha256,
            inference_sha256=source.inference_sha256,
            selected_step=source.selected_step,
            terminal_step=source.terminal_step,
            run_seed=config.run_seed,
            teacher_configuration_sha256=source.teacher_configuration_sha256,
            teacher_graph_sha256=graph,
            teacher_sidecar_sha256=sidecar,
            backbone_runtime_sha256=source.backbone_runtime_sha256,
            test_parquet_sha256=source.test_parquet_sha256,
            test_corpus_sha256=source.test_corpus_sha256,
            members=members,
            rows=records,
            executions=executions,
            trace_scalar_bound=scalar_bound,
            groups=scientific_workspace_groups_v1(members, records),
            causal_summary=summarize_workspace_causal_v1(
                captures, alpha=policy.alpha, numerical_tolerance=policy.numerical_tolerance
            )
            if captures
            else None,
        )
        result = IrisScientificWorkspaceV1(
            sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
            payload=payload,
        )
        _checked(result)
        return result
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific workspace measurement did not complete"
        ) from None


def _checked(result: IrisScientificWorkspaceV1) -> bytes:
    if type(result) is not IrisScientificWorkspaceV1:
        raise ValueError("Exact workspace report required")
    result = IrisScientificWorkspaceV1.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Workspace report identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Workspace report exceeds its byte budget")
    return content


def read_scientific_workspace_v1(path: Path) -> IrisScientificWorkspaceV1:
    """Verify canonical integrity/selection/aggregates, not source authenticity."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("Workspace report inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificWorkspaceV1.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("Workspace report canonical identity differs")
        return result
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific workspace report could not be read"
        ) from None


def write_scientific_workspace_v1(
    result: IrisScientificWorkspaceV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> Path:
    """Publish the complete companion into its separate quota-bound local store."""
    temporary: Path | None = None
    try:
        content = _checked(result)
        if type(maximum_storage_bytes) is not int or not 1 <= maximum_storage_bytes <= 16 * 1024**3:
            raise ValueError("Explicit bounded workspace storage quota required")
        root = _local_path(root) / _STORE
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_workspace_v1(target) != result:
                raise ValueError("Existing workspace report conflicts")
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
            raise ValueError("Workspace store quota/free-space reserve exceeded")
        candidate = root / f".iris-workspace-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged workspace report differs")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_workspace_v1(target) != result:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificEvaluationError(
            "Scientific workspace report could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            (temporary / _FILE).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "measure_scientific_workspace_v1",
    "read_scientific_workspace_v1",
    "write_scientific_workspace_v1",
]
