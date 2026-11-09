"""Actual best-checkpoint held-out decoding plus independently teacher-forced NLL.

No training, reference completions as generated answers, generated-code execution,
network, or model download. CPU/FP32 D is explicit, not an A/B/C/D study result.
"""

import os
import shutil
from collections import Counter
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import torch

from iris.application.composition import build_request_orchestrator
from iris.application.errors import ApplicationError
from iris.application.intervention_receipt import (
    IrisIntervenedRequestExecutionOutput,
    IrisIntervenedRequestExecutionReceiptV1,
)
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
    _rows,
    read_scientific_command_config_v1,
)
from iris.application.scientific_execution import (
    _checkpoint_for_event,
    _selected_best,
    read_scientific_execution_v1,
)
from iris.application.scientific_inference import (
    build_scientific_inference_v1,
    transfer_scientific_inference_to_request_v1,
)
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.scientific_teacher import IrisScientificTeacherOutputV1
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_execution_v1 import ScientificExecutionCheckpointV1
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificDecodingPolicyV1,
    ScientificGeneratedRowV1,
    ScientificGenerationPayloadV1,
    scientific_row_sampling_seed_v1,
)
from iris.artifacts.scientific_intervention_v1 import (
    IrisScientificInterventionV1,
    ScientificInterventionGeneratedRowV1,
    ScientificInterventionPayloadV1,
    intervention_plan_for_row_v1,
)
from iris.data.schemas import IrisExample
from iris.generation import TemperatureTopPSamplingConfig
from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    intervention_plan_sha256_v1,
    validate_intervention_plan_v1,
)
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    tensor_sha256,
)

_DOMAIN = "iris/scientific-generation/v1"
_MAX_BYTES = 64 * 1024**2
_FILE = "generation.json"


class IrisScientificGenerationError(ApplicationError):
    """Content-redacted failure; no partial generation is a complete test result."""


def _own_test_rows(
    rows: tuple[IrisExample, ...],
    *,
    excluded_seeds: set[int],
    scientific: bool,
) -> tuple[tuple[IrisExample, ...], str]:
    if type(rows) is not tuple or not 1 <= len(rows) <= 5000:
        raise ValueError("An explicit bounded held-out test set is required")
    digest = sha256(b"iris/scientific-test-corpus/v1\0")
    owned = []
    identities: set[str] = set()
    size = 0
    for source in rows:
        row = IrisExample.model_validate(source.model_dump(mode="python"))
        if (
            row.split != "test"
            or row.id in identities
            or row.seed in excluded_seeds
            or row.template_id != "iris-synthetic-01-v2"
        ):
            raise ValueError("Unique v2 test rows, seed-disjoint from train/validation, required")
        content = canonical_json_bytes(row.model_dump(mode="json"))
        size += len(content)
        if size > _MAX_BYTES:
            raise ValueError("Held-out corpus exceeds its logical byte budget")
        identities.add(row.id)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        owned.append(row)
    if scientific and Counter(row.task_type for row in owned) != {
        "split_spec_repair": 2000,
        "delayed_goal": 1000,
        "conflict_resolution": 1000,
        "next_tool_prediction": 1000,
    }:
        raise ValueError("Scientific test requires all 5000 rows in source family proportions")
    return tuple(owned), digest.hexdigest()


def _checked(result: IrisScientificGenerationV1) -> bytes:
    if type(result) is not IrisScientificGenerationV1:
        raise ValueError("Exact scientific generation required")
    result = IrisScientificGenerationV1.model_validate(result.model_dump(mode="python"))
    if result.sha256 != domain_sha256(
        _DOMAIN, canonical_json_bytes(result.payload.model_dump(mode="json"))
    ):
        raise ValueError("Scientific generation identity differs")
    content = canonical_json_file_bytes(result.model_dump(mode="json"))
    if len(content) > _MAX_BYTES:
        raise ValueError("Scientific generation exceeds its byte budget")
    return content


def generate_scientific_test_v1(
    *,
    config_path: Path,
    execution_path: Path,
    test: ScientificParquetInputV1,
    decoding: ScientificDecodingPolicyV1,
) -> IrisScientificGenerationV1:
    """Capture the original baseline held-out generation/NLL contract, unchanged."""
    result, _ = _generate_scientific_test_v1(
        config_path=config_path,
        execution_path=execution_path,
        test=test,
        decoding=decoding,
        plan=None,
    )
    return result


def generate_scientific_intervention_v1(
    *,
    config_path: Path,
    execution_path: Path,
    test: ScientificParquetInputV1,
    decoding: ScientificDecodingPolicyV1,
    plan: WorkspaceInterventionPlanV1,
) -> IrisScientificInterventionV1:
    """Generate every control/intervened pair from one validation-selected best.

    Both branches use the same prompt and independently initialized input-only
    decoding RNG. Operator RNG is also row/input-bound. Only the control has a
    teacher NLL measurement; no intervened teacher NLL is fabricated.
    """
    try:
        owned_plan = validate_intervention_plan_v1(plan)
    except Exception:
        raise IrisScientificGenerationError("Scientific intervention intent is invalid") from None
    _, paired = _generate_scientific_test_v1(
        config_path=config_path,
        execution_path=execution_path,
        test=test,
        decoding=decoding,
        plan=owned_plan,
    )
    assert paired is not None
    return paired


def _generate_scientific_test_v1(
    *,
    config_path: Path,
    execution_path: Path,
    test: ScientificParquetInputV1,
    decoding: ScientificDecodingPolicyV1,
    plan: WorkspaceInterventionPlanV1 | None,
) -> tuple[IrisScientificGenerationV1, IrisScientificInterventionV1 | None]:
    """Capture ALL declared held-out rows from exact validation-best trained weights.

    Test path is relative to the training recipe. Output cap is the original
    backbone max_new_tokens; EOS comes from its actual tokenizer/model. Caller
    RNG is restored by composition, and source checkpoint bytes are never written.
    """
    try:
        test = ScientificParquetInputV1.model_validate(test.model_dump(mode="python"))
        decoding = ScientificDecodingPolicyV1.model_validate(decoding.model_dump(mode="python"))
        if test.maximum_rows > 5000:
            raise ValueError("Test budget exceeds 5000 rows")
        snapshot = read_scientific_execution_v1(_local_path(execution_path))
        if not snapshot.events:
            raise ValueError("A terminal committed execution is required")
        terminal = snapshot.events[-1]
        if (
            terminal.payload.after.phase != "stopped"
            or type(terminal.payload.record) is not ScientificExecutionCheckpointV1
        ):
            raise ValueError("Test evaluation is allowed only after the terminal checkpoint")
        best_event = None
        for event in snapshot.events:
            best_event = _selected_best(best_event, event)
        _, best = _bundle_parts(
            _checkpoint_for_event(
                snapshot.path,
                snapshot.header,
                terminal,
                best_event=best_event,
            )
        )
        if best is None or best_event is None:
            raise ValueError("No saved validation-selected best exists")
        config = read_scientific_command_config_v1(config_path)
        if plan is not None:
            plan = validate_intervention_plan_v1(plan)
            plan.validate_dimensions(
                num_slots=config.workspace.num_slots,
                num_modules=config.cognitive.num_modules,
                state_dim=config.cognitive.state_dim,
                cycles=config.workspace.cycles,
                winners_per_cycle=config.workspace.winners_per_cycle,
            )
        if config.maximum_journal_bytes != snapshot.header.payload.maximum_journal_bytes:
            raise ValueError("Journal budget differs from original execution")
        base = config_path.absolute().parent
        rows, _ = _own_test_rows(
            _rows(test, base),
            excluded_seeds=set(),
            scientific=config.optimizer.purpose == "scientific",
        )
        with _composed(config, base=base) as lifecycle:
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
                    "Recipe must reproduce exact original train AND validation context"
                )
            rows, corpus_sha = _own_test_rows(
                rows,
                excluded_seeds={row.seed for row in (*run._rows, *lifecycle._validation)},
                scientific=config.optimizer.purpose == "scientific",
            )
            backbone = run._backbone
            if not 1 <= backbone.max_new_tokens <= 65536:
                raise ValueError("Generation output token budget is unsupported")
            _preflight(backbone, config, (), rows)
            for row in rows:
                count = backbone.tokenize(row.prompt)["input_ids"].shape[1]
                if (
                    count + config.workspace.num_slots + backbone.max_new_tokens
                    > backbone.max_position_embeddings
                ):
                    raise ValueError("Full prompt/workspace/generation budget does not fit context")
            restore_scientific_checkpoint_v1(best, run=run)
            inference = build_scientific_inference_v1(run)
            request = build_request_orchestrator(
                backbone=backbone,
                cognitive_config=config.cognitive,
                workspace_config=config.workspace,
                stop_token_ids=backbone.eos_token_ids,
                sampling=TemperatureTopPSamplingConfig(decoding.temperature, decoding.top_p),
            )
            transfer_scientific_inference_to_request_v1(inference, request=request)
            model = run._optimizer._model
            captured: list[ScientificGeneratedRowV1] = []
            intervened_rows: list[ScientificInterventionGeneratedRowV1] = []
            request_graph_sha = domain_sha256(
                "iris/intervened-request-graph/v1",
                canonical_json_bytes(
                    {
                        "cognitive": config.cognitive.model_dump(mode="json"),
                        "workspace": config.workspace.model_dump(mode="json"),
                    }
                ),
            )
            captured_bytes = 0
            model.eval()
            try:
                for row in rows:
                    prompt_sha = sha256(row.prompt.encode("utf-8")).hexdigest()
                    row_seed = (
                        scientific_row_sampling_seed_v1(decoding.seed, row.id, prompt_sha)
                        if decoding.strategy == "sampling"
                        else None
                    )
                    generator = (
                        None
                        if row_seed is None
                        else torch.Generator(
                            device=backbone.device,
                        ).manual_seed(row_seed)
                    )
                    # Only prompt and INPUT-derived RNG cross the generation boundary.
                    generated = request.generate(
                        row.prompt, strategy=decoding.strategy, generator=generator
                    )
                    # Labels are used ONLY in the separate teacher-forced measurement.
                    evaluated = model.evaluation_forward(
                        row, terminal_token_id=config.terminal_token_id
                    )
                    row_sha = domain_sha256(
                        "iris/scientific-teacher-example/v1",
                        canonical_json_bytes(row.model_dump(mode="json")),
                    )
                    if (
                        type(evaluated) is not IrisScientificTeacherOutputV1
                        or evaluated.example_id != row.id
                        or evaluated.example_sha256 != row_sha
                        or evaluated.configuration_sha256 != model.configuration_sha256
                        or type(evaluated.supervised_token_count) is not int
                    ):
                        raise ValueError("Teacher-held-out provenance differs")
                    loss = evaluated.losses.language_model
                    if (
                        type(loss) is not torch.Tensor
                        or loss.ndim != 0
                        or loss.requires_grad
                        or loss.grad_fn is not None
                        or loss.device != backbone.device
                        or loss.dtype not in (torch.float32, torch.float64)
                        or not bool(torch.isfinite(loss))
                        or float(loss) < 0
                    ):
                        raise ValueError("Invalid detached held-out target NLL sum")
                    if backbone.decode_tokens(generated.token_ids) != generated.text:
                        raise ValueError("Generated text differs from actual token decoding")
                    record = ScientificGeneratedRowV1(
                        example_id=row.id,
                        example_sha256=row_sha,
                        prompt_sha256=prompt_sha,
                        task_type=row.task_type,
                        sampling_seed=row_seed,
                        output_text=generated.text,
                        token_ids=tuple(generated.token_ids[0].cpu().tolist()),
                        stop_reason=generated.stop_reason,
                        matched_stop_token_id=generated.matched_stop_token_id,
                        target_token_count=evaluated.supervised_token_count,
                        target_nll_sum=float(loss),
                    )
                    captured_bytes += len(canonical_json_bytes(record.model_dump(mode="json")))
                    if captured_bytes > _MAX_BYTES - 65536:
                        raise ValueError("Private generation capture byte budget exhausted")
                    captured.append(record)
                    if plan is not None:
                        bound_plan = intervention_plan_for_row_v1(
                            plan, example_id=row.id, prompt_sha256=prompt_sha
                        )
                        paired_rng = (
                            None
                            if row_seed is None
                            else torch.Generator(device=backbone.device).manual_seed(row_seed)
                        )
                        initial_rng_sha = (
                            None
                            if paired_rng is None
                            else tensor_sha256(
                                paired_rng.get_state(),
                                domain="iris/intervened-request/sampling-rng/v1",
                            )
                        )
                        # No label, expected response, hidden fact or test enters this call.
                        changed = request.execute_intervened(
                            row.prompt,
                            plan=bound_plan,
                            strategy=decoding.strategy,
                            generator=paired_rng,
                        )
                        expected_receipt = IrisIntervenedRequestExecutionReceiptV1(
                            prompt_sha256=domain_sha256(
                                "iris/request-execution/v1/prompt", row.prompt.encode("utf-8")
                            ),
                            strategy=decoding.strategy,
                            stop_token_ids=backbone.eos_token_ids,
                            sampling=TemperatureTopPSamplingConfig(
                                decoding.temperature, decoding.top_p
                            ),
                            sampling_seed=row_seed,
                            sampling_rng_state_sha256=initial_rng_sha,
                            intervention_plan_sha256=intervention_plan_sha256_v1(bound_plan),
                            graph_sha256=request_graph_sha,
                            sidecar_state_sha256=inference.manifest.payload.model.sha256,
                            cycles=config.workspace.cycles,
                        )
                        if (
                            type(changed) is not IrisIntervenedRequestExecutionOutput
                            or type(changed.receipt) is not IrisIntervenedRequestExecutionReceiptV1
                            or changed.receipt != expected_receipt
                            or backbone.decode_tokens(changed.generation.token_ids)
                            != changed.generation.text
                        ):
                            raise ValueError("Intervened execution identity/token decoding differs")
                        paired_record = ScientificInterventionGeneratedRowV1(
                            example_id=row.id,
                            example_sha256=row_sha,
                            prompt_sha256=prompt_sha,
                            bound_plan_sha256=expected_receipt.intervention_plan_sha256,
                            sampling_seed=row_seed,
                            initial_sampling_rng_sha256=initial_rng_sha,
                            output_text=changed.generation.text,
                            token_ids=tuple(changed.generation.token_ids[0].cpu().tolist()),
                            stop_reason=changed.generation.stop_reason,
                            matched_stop_token_id=changed.generation.matched_stop_token_id,
                        )
                        captured_bytes += len(
                            canonical_json_bytes(paired_record.model_dump(mode="json"))
                        )
                        if captured_bytes > _MAX_BYTES - 65536:
                            raise ValueError("Paired capture byte budget exhausted")
                        intervened_rows.append(paired_record)
                        del changed, paired_record
                    del generated, evaluated, loss
            finally:
                model.train(True)
            if build_scientific_checkpoint_v1(run) != best:
                raise ValueError("Generation/evaluation mutated the selected training checkpoint")
            if request.sidecar_state_sha256 != inference.manifest.payload.model.sha256:
                raise ValueError("Request weights changed during generation")
            if read_scientific_execution_v1(snapshot.path) != snapshot:
                raise ValueError("Source execution changed during held-out evaluation")
            payload = ScientificGenerationPayloadV1(
                purpose=config.optimizer.purpose,
                execution_sha256=snapshot.header.sha256,
                terminal_event_sha256=terminal.sha256,
                best_validation_event_sha256=best_event.sha256,
                best_checkpoint_sha256=best.manifest.sha256,
                selected_step=best.manifest.payload.completed_steps,
                terminal_step=terminal.payload.after.step,
                inference_sha256=inference.manifest.sha256,
                teacher_configuration_sha256=model.configuration_sha256,
                backbone_runtime_sha256=backbone.checkpoint_runtime_sha256,
                test_parquet_sha256=test.sha256,
                test_corpus_sha256=corpus_sha,
                decoding=decoding,
                max_new_tokens=backbone.max_new_tokens,
                vocabulary_size=backbone.vocabulary_size,
                eos_token_ids=backbone.eos_token_ids,
                rows=tuple(captured),
            )
            result = IrisScientificGenerationV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
            _checked(result)
            paired = None
            if plan is not None:
                paired_payload = ScientificInterventionPayloadV1(
                    baseline=result,
                    inference_manifest=inference.manifest,
                    plan=plan,
                    plan_sha256=intervention_plan_sha256_v1(plan),
                    request_graph_sha256=request_graph_sha,
                    request_sidecar_sha256=inference.manifest.payload.model.sha256,
                    num_modules=config.cognitive.num_modules,
                    num_slots=config.workspace.num_slots,
                    state_dim=config.cognitive.state_dim,
                    cycles=config.workspace.cycles,
                    winners_per_cycle=config.workspace.winners_per_cycle,
                    rows=tuple(intervened_rows),
                )
                paired = IrisScientificInterventionV1(
                    sha256=domain_sha256(
                        "iris/scientific-intervention/v1",
                        canonical_json_bytes(paired_payload.model_dump(mode="json")),
                    ),
                    payload=paired_payload,
                )
                if len(canonical_json_file_bytes(paired.model_dump(mode="json"))) > _MAX_BYTES:
                    raise ValueError("Paired capture exceeds its total byte budget")
            return result, paired
    except Exception:
        raise IrisScientificGenerationError(
            "Scientific held-out generation did not complete"
        ) from None


def read_scientific_generation_v1(path: Path) -> IrisScientificGenerationV1:
    """Check bounded private capture bytes; hashes are integrity, NOT quality evidence."""
    try:
        path = _local_path(path)
        for parent in (path, *path.parents):
            _regular(parent, directory=True)
        if {entry.name for entry in path.iterdir()} != {_FILE}:
            raise ValueError("Generation capture inventory differs")
        content = _bounded_read(path / _FILE, _MAX_BYTES)
        _json(content[:-1])
        result = IrisScientificGenerationV1.model_validate_json(content)
        if path.name != result.sha256 or _checked(result) != content:
            raise ValueError("Generation identity/canonical bytes differ")
        return result
    except Exception:
        raise IrisScientificGenerationError(
            "Scientific generation capture could not be read"
        ) from None


def write_scientific_generation_v1(
    result: IrisScientificGenerationV1,
    *,
    root: Path,
    maximum_storage_bytes: int = 256 * 1024**2,
) -> Path:
    """Publish a complete private capture; no weights, labels or deletion of old runs."""
    temporary: Path | None = None
    try:
        content = _checked(result)
        if type(maximum_storage_bytes) is not int or not 1 <= maximum_storage_bytes <= 16 * 1024**3:
            raise ValueError("Explicit bounded generation store quota required")
        root = _local_path(root) / "scientific-generation-v1"
        for parent in (root, *root.parents):
            with suppress(FileNotFoundError):
                _regular(parent, directory=True)
        root.mkdir(parents=True, exist_ok=True)
        for parent in (root, *root.parents):
            _regular(parent, directory=True)
        target = root / result.sha256
        if target.exists():
            if read_scientific_generation_v1(target) != result:
                raise ValueError("Existing generation capture conflicts")
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
            raise ValueError("Generation storage quota/free-space reserve exceeded")
        candidate = root / f".iris-generation-{uuid4().hex}.tmp"
        os.mkdir(candidate, mode=0o700)
        temporary = candidate
        _write_exact_file(temporary / _FILE, content)
        if _bounded_read(temporary / _FILE, _MAX_BYTES) != content:
            raise ValueError("Staged generation bytes differ")
        try:
            rename_directory_with_windows_retry(temporary, target)
        except OSError:
            if not target.exists() or read_scientific_generation_v1(target) != result:
                raise
            if not temporary.exists():
                temporary = None
        else:
            temporary = None
        return target
    except Exception:
        raise IrisScientificGenerationError(
            "Scientific generation capture could not be published"
        ) from None
    finally:
        if temporary is not None and temporary.exists():
            _regular(temporary, directory=True)
            (temporary / _FILE).unlink(missing_ok=True)
            temporary.rmdir()


__all__ = [
    "IrisScientificGenerationError",
    "generate_scientific_intervention_v1",
    "generate_scientific_test_v1",
    "read_scientific_generation_v1",
    "write_scientific_generation_v1",
]
