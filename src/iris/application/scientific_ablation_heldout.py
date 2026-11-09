"""Full local held-out preflight and real prompt-only decoding/target NLL.

No selection/tuning/training, model download, code execution or publication.
Returned row evidence is NOT durable row-ack/resume, scoring or completed study.
All process-global runtime ownership is exclusive, not thread-safe/all-GPU.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch

from iris.ablations import cognitive_ablation_policy_sha256_v1
from iris.application.scientific_ablation_generation_context import (
    IrisScientificAblationGenerationContextV1,
    _restore_caller,
)
from iris.application.scientific_ablation_generation_context import (
    _entry as _context_entry,
)
from iris.application.scientific_ablation_generation_context import (
    _poison as _poison_context,
)
from iris.application.scientific_ablation_receipt import IrisScientificAblationRequestOutputV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherOutputV1
from iris.application.scientific_ablation_validation import (
    _complete_row_fields,
    _has_signal,
    _training_corpus_sha256,
    own_scientific_ablation_validation_rows_v1,
)
from iris.application.scientific_command import (
    ScientificParquetInputV1,
    _input_bytes,
    _local_path,
    _rows,
)
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.scientific_ablation_heldout_v1 import (
    IrisScientificAblationHeldoutInputV1,
    IrisScientificAblationHeldoutRowV1,
    ScientificAblationHeldoutInputPayloadV1,
    ScientificAblationHeldoutRowPayloadV1,
)
from iris.artifacts.scientific_generation_v1 import (
    ScientificDecodingPolicyV1,
    scientific_row_sampling_seed_v1,
)
from iris.data.schemas import IrisExample
from iris.training.ablations import IrisAblatedTeacherOutputV1
from iris.training.intervention import teacher_target_sha256_v1
from iris.training.outputs import IrisTeacherForcedOutput, IrisTeacherTracedOutputV1
from iris.training.probes import prepare_probe_targets
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state

_INPUT_DOMAIN = "iris/scientific-ablation-heldout-input/v1"
_ROW_DOMAIN = "iris/scientific-ablation-heldout-row/v1"
_REQUEST_DOMAIN = "iris/scientific-ablation-heldout-row-request/v1"
_CORE_DOMAIN = "iris/scientific-ablation-request-execution/v1"
_MAX_BYTES = 64 * 1024**2


class IrisScientificAblationHeldoutError(RuntimeError):
    """No partial row is an acknowledged or complete held-out result."""


@dataclass(frozen=True, slots=True, eq=False)
class _TestSeal:
    owner: "IrisScientificAblationHeldoutV1"
    context: IrisScientificAblationGenerationContextV1
    rows: tuple[IrisExample, ...]
    path: Path
    spec: ScientificParquetInputV1
    metadata: bytes
    fingerprints: tuple[bytes, ...]
    target_counts: tuple[int, ...]


class IrisScientificAblationHeldoutV1:
    """Private complete test ownership; no raw rows/model or transfer exposed."""

    __slots__ = ("_guard", "_state")
    _guard: _TestSeal
    _state: str

    def __init__(self) -> None:
        raise IrisScientificAblationHeldoutError("Use the full held-out preflight builder")

    @property
    def manifest(self) -> IrisScientificAblationHeldoutInputV1:
        return _entry(self)[1]

    def execute_row(self, row_index: int) -> IrisScientificAblationHeldoutRowV1:
        return _execute_row(self, row_index)


def _fail(primary: BaseException | None, cleanup: list[BaseException]) -> None:
    if primary is not None and _has_signal(primary):
        raise primary from (
            BaseExceptionGroup("Held-out cleanup failed", cleanup) if cleanup else None
        )
    signal = next((error for error in cleanup if _has_signal(error)), None)
    if signal is not None:
        raise signal from primary
    failures = ([] if primary is None else [primary]) + cleanup
    if failures:
        raise IrisScientificAblationHeldoutError("Held-out operation did not complete") from (
            failures[0] if len(failures) == 1 else BaseExceptionGroup("Held-out failures", failures)
        )


def _own_rows(
    rows: tuple[IrisExample, ...], *, excluded_seeds: set[int], scientific: bool
) -> tuple[tuple[IrisExample, ...], str]:
    if type(rows) is not tuple or not 1 <= len(rows) <= 5000:
        raise ValueError("Complete bounded held-out tuple required")
    owned: list[IrisExample] = []
    identifiers: set[str] = set()
    digest = sha256(b"iris/scientific-test-corpus/v1\0")
    logical = 0
    for row in rows:
        if not _complete_row_fields(row):
            raise ValueError("Exact complete held-out row fields required")
        row = IrisExample.model_validate(row.model_dump(mode="python", warnings=False))
        if (
            row.split != "test"
            or row.template_id != "iris-synthetic-01-v2"
            or row.seed in excluded_seeds
            or row.id in identifiers
        ):
            raise ValueError(
                "Unique source-v2 test rows disjoint from original train/validation required"
            )
        raw = canonical_json_bytes(row.model_dump(mode="json"))
        logical += len(raw)
        if logical > _MAX_BYTES:
            raise ValueError("Complete test logical byte limit exceeded")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        identifiers.add(row.id)
        owned.append(row)
    if scientific and Counter(row.task_type for row in owned) != {
        "split_spec_repair": 2000,
        "delayed_goal": 1000,
        "conflict_resolution": 1000,
        "next_tool_prediction": 1000,
    }:
        raise ValueError("Scientific test requires the entire 5000-row source distribution")
    return tuple(owned), digest.hexdigest()


def _entry(
    owner: IrisScientificAblationHeldoutV1,
) -> tuple[_TestSeal, IrisScientificAblationHeldoutInputV1]:
    if type(owner) is not IrisScientificAblationHeldoutV1:
        raise IrisScientificAblationHeldoutError("Exact held-out owner required")
    if getattr(owner, "_state", None) == "busy":
        raise IrisScientificAblationHeldoutError("Held-out owner is already busy")
    candidate = getattr(owner, "_guard", None)
    if (
        type(candidate) is _TestSeal
        and candidate.owner is owner
        and type(candidate.context) is IrisScientificAblationGenerationContextV1
        and getattr(candidate.context, "_state", None) == "busy"
    ):
        raise IrisScientificAblationHeldoutError("Native held-out context is already busy")
    try:
        seal = owner._guard
        if type(seal) is not _TestSeal or seal.owner is not owner or owner._state != "ready":
            raise ValueError("Held-out owner anchor/state changed")
        context_seal = _context_entry(seal.context)
        if context_seal.run._busy:
            raise ValueError("Native context is already busy")
        metadata = IrisScientificAblationHeldoutInputV1.model_validate_json(seal.metadata)
        if metadata.sha256 != domain_sha256(
            _INPUT_DOMAIN, canonical_json_bytes(metadata.payload.model_dump(mode="json"))
        ):
            raise ValueError("Private held-out metadata changed")
        if (
            len(seal.rows) != metadata.payload.example_count
            or len(seal.fingerprints) != len(seal.rows)
            or len(seal.target_counts) != len(seal.rows)
        ):
            raise ValueError("Private complete held-out inventory changed")
        return seal, metadata
    except BaseException as error:
        owner._state = "poisoned"
        _fail(error, [])
        raise AssertionError("Unreachable") from error


def _audit_file(seal: _TestSeal) -> None:
    if (
        sha256(_input_bytes(seal.path, seal.spec.maximum_file_bytes)).hexdigest()
        != seal.spec.sha256
    ):
        raise ValueError("Original held-out Parquet changed")


def _audit_original(context: IrisScientificAblationGenerationContextV1) -> None:
    source = context._source.manifest.payload
    if _training_corpus_sha256(context._run) != source.training_corpus_sha256:
        raise ValueError("Original training corpus changed")
    _, validation_sha = own_scientific_ablation_validation_rows_v1(
        context._validation, training_rows=context._run._rows
    )
    if validation_sha != source.validation_corpus_sha256:
        raise ValueError("Original validation corpus changed")


def build_scientific_ablation_heldout_v1(
    context: IrisScientificAblationGenerationContextV1,
    *,
    expected_reconstruction_sha256: str,
    test: ScientificParquetInputV1,
    base: Path,
    decoding: ScientificDecodingPolicyV1,
) -> IrisScientificAblationHeldoutV1:
    """Audit/own ALL actual Parquet rows and token/label budgets before ANY forward."""
    caller = None
    primary: BaseException | None = None
    cleanup: list[BaseException] = []
    result = None
    caller_dtype = torch.get_default_dtype()
    device = torch.device("cpu")
    with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
        try:
            seal = _context_entry(context)
            reconstruction = context.reconstruction
            if (
                type(expected_reconstruction_sha256) is not str
                or expected_reconstruction_sha256 != reconstruction.sha256
            ):
                raise ValueError("Independent reconstruction identity differs")
            if (
                seal.run._busy
                or type(test) is not ScientificParquetInputV1
                or type(decoding) is not ScientificDecodingPolicyV1
            ):
                raise ValueError("Exact idle context/input/decoding required")
            if (
                set(vars(test)) != set(ScientificParquetInputV1.model_fields)
                or test.__pydantic_extra__ is not None
                or set(vars(decoding)) != set(ScientificDecodingPolicyV1.model_fields)
                or decoding.__pydantic_extra__ is not None
            ):
                raise ValueError("Exact complete input/decoding fields required")
            test = ScientificParquetInputV1.model_validate_json(test.model_dump_json())
            decoding = ScientificDecodingPolicyV1.model_validate_json(decoding.model_dump_json())
            if test.maximum_rows > 5000 or test.maximum_logical_bytes > _MAX_BYTES:
                raise ValueError("Held-out input exceeds complete split budgets")
            source = context.source_manifest.payload
            sampling = seal.request._request._iris_generator.sampling
            if (decoding.temperature, decoding.top_p) != (sampling.temperature, sampling.top_p):
                raise ValueError("Held-out decoding differs from the independently owned request")
            device = seal.run._backbone.device
            caller = capture_runtime_state(device)
            torch.set_default_dtype(torch.float32)
            rows = _rows(test, _local_path(base))
            _audit_original(context)
            rows, corpus_sha = _own_rows(
                rows,
                excluded_seeds={row.seed for row in (*seal.run._rows, *seal.validation)},
                scientific=source.purpose == "scientific",
            )
            if source.purpose == "scientific" and test.maximum_rows != 5000:
                raise ValueError("Scientific test budget must bind full split")
            backbone = seal.run._backbone
            model_spec = seal.request.manifest.payload.model
            if not 1 <= backbone.max_new_tokens <= 65536:
                raise ValueError("Actual full generation cap exceeds supported limits")
            counts: list[int] = []
            with torch.device(device):
                for row in rows:
                    raw = backbone._tokenizer(
                        row.prompt,
                        return_tensors="pt",
                        truncation=False,
                        padding=False,
                        return_attention_mask=True,
                    )
                    ids, mask = raw["input_ids"], raw["attention_mask"]
                    if (
                        type(ids) is not torch.Tensor
                        or ids.ndim != 2
                        or ids.shape[0] != 1
                        or not 1 <= ids.shape[1] <= backbone.max_prompt_tokens
                        or type(mask) is not torch.Tensor
                        or mask.shape != ids.shape
                        or not bool((mask == 1).all())
                    ):
                        raise ValueError("Untruncated complete prompt does not fit")
                    actual = backbone.tokenize(row.prompt)
                    if not torch.equal(ids, actual["input_ids"].cpu()) or not torch.equal(
                        mask, actual["attention_mask"].cpu()
                    ):
                        raise ValueError("Prompt tokenization would truncate/change source")
                    prepared = prepare_teacher_forcing_input(
                        backbone=backbone,
                        prompt=row.prompt,
                        target_text=row.expected_response,
                        terminal_token_id=seal.run._terminal_token_id,
                    )
                    if (
                        ids.shape[1]
                        + model_spec.workspace.num_slots
                        + max(backbone.max_new_tokens, prepared.target_input_ids.shape[1])
                        > backbone.max_position_embeddings
                    ):
                        raise ValueError(
                            "Prompt/slots/full generation or complete target context does not fit"
                        )
                    prepare_probe_targets(
                        (row,),
                        vocabulary=seal.run._optimizer._model.probes.vocabulary,
                        purpose="analysis",
                        device=device,
                    )
                    counts.append(int(prepared.target_attention_mask.sum()))
            payload = ScientificAblationHeldoutInputPayloadV1(
                reconstruction_sha256=reconstruction.sha256,
                source_sha256=context.source_manifest.sha256,
                ablation_policy_sha256=source.ablation_policy_sha256,
                selected_checkpoint_sha256=source.best_checkpoint_sha256,
                purpose=source.purpose,
                test_parquet_sha256=test.sha256,
                test_corpus_sha256=corpus_sha,
                example_count=len(rows),
                target_token_count=sum(counts),
                decoding=decoding,
                max_new_tokens=backbone.max_new_tokens,
                vocabulary_size=backbone.vocabulary_size,
                eos_token_ids=backbone.eos_token_ids,
            )
            manifest = IrisScientificAblationHeldoutInputV1(
                sha256=domain_sha256(
                    _INPUT_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
            result = object.__new__(IrisScientificAblationHeldoutV1)
            result._state = "ready"
            result._guard = _TestSeal(
                result,
                context,
                rows,
                _local_path(base / test.path),
                test,
                canonical_json_bytes(manifest.model_dump(mode="json")),
                tuple(canonical_json_bytes(row.model_dump(mode="json")) for row in rows),
                tuple(counts),
            )
            _audit_file(result._guard)
            _entry(result)
        except BaseException as error:
            primary = error
        finally:
            if caller is not None:
                try:
                    _restore_caller(caller, caller_dtype, device)
                except BaseException as error:
                    cleanup.append(error)
            if cleanup:
                _poison_context(context)
            if result is not None and (primary is not None or cleanup):
                result._state = "poisoned"
    _fail(primary, cleanup)
    if result is None:
        raise IrisScientificAblationHeldoutError("No verified full held-out input")
    return result


def _target_nll(
    context: IrisScientificAblationGenerationContextV1,
    row: IrisExample,
    expected_count: int,
) -> tuple[float, str, str, str, str]:
    run = context._run
    model = run._optimizer._model
    prepared = prepare_teacher_forcing_input(
        backbone=run._backbone,
        prompt=row.prompt,
        target_text=row.expected_response,
        terminal_token_id=run._terminal_token_id,
    )
    target_sha = teacher_target_sha256_v1(prepared.target_input_ids, prepared.target_attention_mask)
    if int(prepared.target_attention_mask.sum()) != expected_count:
        raise ValueError("Owned complete target count changed")
    prompt_sha = domain_sha256("iris/cognitive-ablation-prompt/v1", row.prompt.encode())
    sidecar_sha = model.teacher.evaluation_provenance.sidecar_state_sha256
    teacher_invocation = domain_sha256(
        "iris/cognitive-ablation-teacher-invocation/v1",
        canonical_json_bytes(
            {
                "mode": "evaluation",
                "policy_sha256": model.ablation_policy_sha256,
                "prompt_sha256": prompt_sha,
                "target_sha256": target_sha,
                "teacher_graph_sha256": run._teacher_graph_identity,
                "sidecar_state_sha256": sidecar_sha,
            }
        ),
    )
    example_sha = domain_sha256(
        "iris/scientific-ablation-example/v1", canonical_json_bytes(row.model_dump(mode="json"))
    )
    state_sha = run._optimizer._committed_scientific_state_sha256
    invocation = domain_sha256(
        "iris/scientific-ablation-invocation/v1",
        canonical_json_bytes(
            {
                "mode": "evaluation",
                "example_sha256": example_sha,
                "configuration_sha256": model.configuration_sha256,
                "teacher_invocation_sha256": teacher_invocation,
                "scientific_state_sha256": state_sha,
            }
        ),
    )
    output = model.evaluation_forward(row, terminal_token_id=run._terminal_token_id)
    if (
        type(output) is not IrisScientificAblationTeacherOutputV1
        or output.mode != "evaluation"
        or output.example_id != row.id
        or output.example_sha256 != example_sha
        or output.configuration_sha256 != model.configuration_sha256
        or output.scientific_state_sha256 != state_sha
        or output.invocation_sha256 != invocation
        or output.ablation_policy_sha256 != model.ablation_policy_sha256
        or type(output.supervised_token_count) is not int
        or output.supervised_token_count != expected_count
        or type(output.teacher) is not IrisAblatedTeacherOutputV1
    ):
        raise ValueError("Native-best mode-bound target output provenance differs")
    teacher = output.teacher
    if (
        teacher.mode != "evaluation"
        or teacher.invocation_sha256 != teacher_invocation
        or teacher.prompt_sha256 != prompt_sha
        or teacher.target_sha256 != target_sha
        or teacher.teacher_graph_sha256 != run._teacher_graph_identity
        or teacher.sidecar_state_sha256 != sidecar_sha
        or cognitive_ablation_policy_sha256_v1(teacher.policy) != model.ablation_policy_sha256
        or type(teacher.traced) is not IrisTeacherTracedOutputV1
        or type(teacher.traced.teacher) is not IrisTeacherForcedOutput
        or teacher.traced.teacher.supervised_token_count != expected_count
    ):
        raise ValueError("Actual teacher/EOS/invocation facts differ")
    loss = output.losses.language_model
    if (
        type(loss) is not torch.Tensor
        or loss.ndim != 0
        or loss.requires_grad
        or loss.grad_fn is not None
        or loss.dtype not in (torch.float32, torch.float64)
        or loss.device != run._backbone.device
        or not bool(torch.isfinite(loss))
        or float(loss) < 0
    ):
        raise ValueError("Invalid actual detached complete target NLL sum")
    return float(loss), invocation, target_sha, state_sha, model.configuration_sha256


def _execute_row(
    owner: IrisScientificAblationHeldoutV1, row_index: int
) -> IrisScientificAblationHeldoutRowV1:
    seal, manifest = _entry(owner)
    if type(row_index) is not int or not 0 <= row_index < len(seal.rows):
        raise IrisScientificAblationHeldoutError("Exact owned row index required")
    context = seal.context
    run, request = context._run, context._request
    caller_dtype = torch.get_default_dtype()
    caller = None
    modes = tuple((module, module.training) for module in run._optimizer._model.modules())
    entered = False
    primary: BaseException | None = None
    cleanup: list[BaseException] = []
    result = None
    with torch.inference_mode(False), torch.enable_grad(), torch.device(run._backbone.device):
        try:
            _audit_file(seal)
            _audit_original(context)
            row = seal.rows[row_index]
            if (
                not _complete_row_fields(row)
                or canonical_json_bytes(row.model_dump(mode="json")) != seal.fingerprints[row_index]
            ):
                raise ValueError("Owned held-out row changed")
            caller = capture_runtime_state(run._backbone.device)
            torch.set_default_dtype(torch.float32)
            owner._state = context._state = "busy"
            run._busy = entered = True
            p = manifest.payload
            prompt_sha = sha256(row.prompt.encode()).hexdigest()
            row_seed = (
                scientific_row_sampling_seed_v1(p.decoding.seed, row.id, prompt_sha)
                if p.decoding.strategy == "sampling"
                else None
            )
            generator = (
                None
                if row_seed is None
                else torch.Generator(device=run._backbone.device).manual_seed(row_seed)
            )
            initial_rng = None if generator is None else generator.get_state().clone()
            # Labels/targets never cross this prompt-only generation boundary.
            output = request.execute(row.prompt, strategy=p.decoding.strategy, generator=generator)
            if type(output) is not IrisScientificAblationRequestOutputV1:
                raise ValueError("Exact actual mode-bound request output required")
            output.__post_init__()
            receipt = output.receipt
            receipt.__post_init__()
            generation = output.generation
            tokens = generation.token_ids
            if (
                type(tokens) is not torch.Tensor
                or tokens.dtype != torch.int64
                or tokens.ndim != 2
                or tokens.shape[0] != 1
                or not 1 <= tokens.shape[1] <= p.max_new_tokens
                or tokens.device != run._backbone.device
                or tokens.requires_grad
                or tokens.grad_fn is not None
            ):
                raise ValueError("Actual generated token tensor differs")
            # Existing cached decoders intentionally return inference tensors.
            # Own normal detached bytes under inference_mode(False), then verify
            # their exact dtype/shape/content against the actual request receipt.
            tokens = tokens.detach().clone()
            if tokens.is_inference():
                raise ValueError("Failed to own normal generated-token snapshot")
            actual_prompt = run._backbone.tokenize(row.prompt)
            source = context._guard.source_metadata
            if (
                receipt.artifact_sha256 != context._source.inference.manifest.sha256
                or receipt.source_checkpoint_sha256 != p.selected_checkpoint_sha256
                or receipt.lifecycle_sha256 != context._source.lifecycle.manifest.sha256
                or receipt.source_run_configuration_sha256 != run.configuration_sha256
                or receipt.source_model_configuration_sha256
                != run._optimizer._model.configuration_sha256
                or receipt.backbone_runtime_sha256 != run._backbone_identity
                or receipt.ablation_policy_sha256 != p.ablation_policy_sha256
                or receipt.sidecar_state_sha256
                != context._source.manifest.payload.request_sidecar_sha256
                or receipt.selection != "selected_best"
                or receipt.completed_steps != run.completed_steps
                or receipt.purpose != p.purpose
                or receipt.prompt_sha256
                != domain_sha256(_CORE_DOMAIN + "/prompt", row.prompt.encode())
                or receipt.cognitive_prompt_sha256
                != domain_sha256("iris/cognitive-ablation-prompt/v1", row.prompt.encode())
                or receipt.strategy != p.decoding.strategy
                or receipt.max_new_tokens != p.max_new_tokens
                or receipt.stop_token_ids != p.eos_token_ids
                or receipt.sampling_seed != row_seed
                or (receipt.sampling.temperature, receipt.sampling.top_p)
                != (p.decoding.temperature, p.decoding.top_p)
                or receipt.cycles != request._request._cycles
            ):
                raise ValueError("Actual request/selected-best/decoding/prompt provenance differs")
            if (
                any(
                    tensor.dtype != actual.dtype
                    or tensor.device != actual.device
                    or tensor.requires_grad
                    or tensor.grad_fn is not None
                    or tensor.is_inference()
                    for tensor, actual in (
                        (output.prompt_input_ids, actual_prompt["input_ids"]),
                        (output.prompt_attention_mask, actual_prompt["attention_mask"]),
                    )
                )
                or not torch.equal(output.prompt_input_ids, actual_prompt["input_ids"])
                or not torch.equal(output.prompt_attention_mask, actual_prompt["attention_mask"])
                or receipt.prompt_input_ids_sha256
                != tensor_sha256(output.prompt_input_ids, domain=_CORE_DOMAIN + "/ids")
                or receipt.prompt_attention_mask_sha256
                != tensor_sha256(output.prompt_attention_mask, domain=_CORE_DOMAIN + "/mask")
                or receipt.generated_token_ids_sha256
                != tensor_sha256(tokens, domain=_CORE_DOMAIN + "/generated-ids")
            ):
                raise ValueError("Actual request prompt/generated tensor provenance differs")
            if (
                initial_rng is not None
                and generator is not None
                and (
                    receipt.sampling_initial_state_sha256
                    != tensor_sha256(initial_rng, domain=_CORE_DOMAIN + "/sampling-state")
                    or receipt.sampling_final_state_sha256
                    != tensor_sha256(generator.get_state(), domain=_CORE_DOMAIN + "/sampling-state")
                )
            ):
                raise ValueError("Actual full per-row sampling state provenance differs")
            if run._backbone.decode_tokens(tokens) != generation.text:
                raise ValueError("Actual decoded text differs from generated tokens")
            restore_runtime_state(run._runtime)
            run._optimizer._model.eval()
            nll, invocation, target_sha, state_sha, configuration_sha = _target_nll(
                context, row, seal.target_counts[row_index]
            )
            receipt_json = canonical_json_bytes(asdict(receipt)).decode()
            payload = ScientificAblationHeldoutRowPayloadV1(
                input=manifest,
                row_index=row_index,
                example_id=row.id,
                example_sha256=domain_sha256(
                    "iris/scientific-ablation-example/v1", seal.fingerprints[row_index]
                ),
                prompt_sha256=prompt_sha,
                task_type=row.task_type,
                sampling_seed=row_seed,
                output_text=generation.text,
                token_ids=tuple(tokens[0].cpu().tolist()),
                stop_reason=generation.stop_reason,
                matched_stop_token_id=generation.matched_stop_token_id,
                target_token_count=seal.target_counts[row_index],
                target_nll_sum=nll,
                teacher_invocation_sha256=invocation,
                teacher_target_sha256=target_sha,
                scientific_state_sha256=state_sha,
                teacher_configuration_sha256=configuration_sha,
                request_receipt_json=receipt_json,
                request_receipt_sha256=domain_sha256(_REQUEST_DOMAIN, receipt_json.encode()),
            )
            result = IrisScientificAblationHeldoutRowV1(
                sha256=domain_sha256(
                    _ROW_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            )
            _audit_file(seal)
            if (
                canonical_json_bytes(row.model_dump(mode="json")) != seal.fingerprints[row_index]
                or source != context._guard.source_metadata
            ):
                raise ValueError("Original source/input changed during actual row execution")
            _audit_original(context)
        except BaseException as error:
            primary = error
        finally:
            if entered:
                try:
                    for module, mode in modes:
                        module.training = mode
                    if any(module.training is not mode for module, mode in modes):
                        raise ValueError("Native best evaluation modes not restored")
                except BaseException as error:
                    cleanup.append(error)
            if caller is not None:
                try:
                    _restore_caller(caller, caller_dtype, run._backbone.device)
                except BaseException as error:
                    cleanup.append(error)
            if entered:
                run._busy = False
                context._state = owner._state = "ready"
                try:
                    _context_entry(context)
                except BaseException as error:
                    cleanup.append(error)
            if primary is not None or cleanup:
                owner._state = "poisoned"
                if entered or cleanup:
                    _poison_context(context)
    _fail(primary, cleanup)
    if result is None:
        raise IrisScientificAblationHeldoutError("No verified actual held-out row")
    return result


__all__ = [
    "IrisScientificAblationHeldoutError",
    "IrisScientificAblationHeldoutV1",
    "build_scientific_ablation_heldout_v1",
]
