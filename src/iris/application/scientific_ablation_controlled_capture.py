"""Bounded private row commits and resume of real mode-bound held-out generation.

Cooperating writers hold one nonblocking persistent OS lock across computation.
Row+ack are one immutable directory publication. Uncommitted stages are preserved
and charged to quota, never reaped/adopted. No training, downloads or code execution.
Integrity is not hostile-filesystem attestation or a power-loss durability promise;
scoring must independently trust the final complete capture SHA, not just its plan.
"""

import os
import re
import shutil
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from math import fsum
from pathlib import Path
from uuid import uuid4

import torch

from iris.application._scientific_ablation_kernel_recovery import (
    _checked_interval_permission,
    _KernelIntervalPermission,
    _locked_interval_stop,
)
from iris.application.scientific_ablation_controlled_generation_context import (
    _poison as _poison_context,
)
from iris.application.scientific_ablation_controlled_generation_context import _restore_caller
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
    _audit_file,
    _audit_original,
)
from iris.application.scientific_ablation_controlled_heldout import _entry as _heldout_entry
from iris.application.scientific_ablation_controlled_receipt import (
    IrisScientificAblationControlledRequestReceiptV1,
)
from iris.application.scientific_ablation_inference import _digest
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _finish_descriptor,
    _identity,
    _path,
    _read_file,
    _regular,
    _signature,
    _write_bytes,
)
from iris.application.scientific_ablation_kernel_chunk import (
    check_kernel_chunk_prefix_v1,
    checked_kernel_chunk_v1,
)
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _json
from iris.application.target_preparation import prepare_teacher_forcing_input
from iris.artifacts.scientific_ablation_controlled_capture_v1 import (
    IrisScientificAblationControlledCaptureAckV1,
    IrisScientificAblationControlledCaptureCompleteV1,
    IrisScientificAblationControlledCapturePlanV1,
    ScientificAblationControlledCaptureAckPayloadV1,
    ScientificAblationControlledCaptureCompletePayloadV1,
    ScientificAblationControlledCaptureExpectationV1,
    ScientificAblationControlledCapturePlanPayloadV1,
)
from iris.artifacts.scientific_ablation_controlled_heldout_v1 import (
    IrisScientificAblationControlledHeldoutRowV1,
    ScientificAblationControlledHeldoutRowPayloadV1,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1,
)
from iris.artifacts.scientific_generation_v1 import scientific_row_sampling_seed_v1
from iris.generation.sampling import TemperatureTopPSamplingConfig
from iris.training.intervention import teacher_target_sha256_v1
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.hashing import (
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    tensor_sha256,
)
from iris.utils.process_file_lock import ProcessFileLock, process_file_lock
from iris.utils.seeds import capture_runtime_state

_PLAN_DOMAIN = "iris/scientific-ablation-controlled-capture-plan/v1"
_ACK_DOMAIN = "iris/scientific-ablation-controlled-capture-row-ack/v1"
_COMPLETE_DOMAIN = "iris/scientific-ablation-controlled-capture-complete/v1"
_ROW_DOMAIN = "iris/scientific-ablation-controlled-heldout-row/v1"
_REQUEST_DOMAIN = "iris/scientific-ablation-controlled-heldout-row-request/v1"
_CORE_DOMAIN = "iris/scientific-ablation-controlled-request-execution/v1"
_STORE = "scientific-ablation-controlled-capture-v1"
_LOCK = ".iris-owner.lock"
_STAGE_PATTERN = re.compile(r"\.iris-controlled-capture-[0-9a-f]{32}\.tmp")
_ROW_PATTERN = re.compile(r"row-[0-9]{5}")
_MAX_ROW = 8 * 1024**2
_MAX_PLAN = 8 * 1024**2
_MAX_STORAGE = 16 * 1024**3
_MAX_DIRS = 20000
_MAX_FILES = 60000
_ROW_FILES = {"attempt.json", "row.json", "ack.json"}
_KNOWN_FILES = {*_ROW_FILES, "plan.json", "complete.json", "failure.json"}


class IrisScientificAblationControlledCaptureError(RuntimeError):
    """Partial/unacknowledged generation is never a complete scored capture."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationControlledCaptureProgressV1:
    path: Path
    plan: IrisScientificAblationControlledCapturePlanV1
    rows: tuple[IrisScientificAblationControlledHeldoutRowV1, ...]
    acknowledgements: tuple[IrisScientificAblationControlledCaptureAckV1, ...]
    complete: IrisScientificAblationControlledCaptureCompleteV1 | None
    uncommitted_stage_count: int

    @property
    def next_row_index(self) -> int:
        return len(self.rows)


def _fail(primary: BaseException | None, secondary: list[BaseException]) -> None:
    if primary is not None and _has_signal(primary):
        raise primary from (
            BaseExceptionGroup("Capture cleanup failures", secondary) if secondary else None
        )
    signal = next((error for error in secondary if _has_signal(error)), None)
    if signal is not None:
        raise signal from primary
    failures = ([] if primary is None else [primary]) + secondary
    if failures:
        raise IrisScientificAblationControlledCaptureError(
            "Durable capture operation did not complete"
        ) from (
            failures[0] if len(failures) == 1 else BaseExceptionGroup("Capture failures", failures)
        )


def _payload_sha(domain: str, payload: object) -> str:
    return domain_sha256(domain, canonical_json_bytes(payload))


def _row_file_bound(payload: ScientificAblationControlledCapturePlanPayloadV1) -> int:
    """Upper bound of COMPLETE row JSON, including twice-quoted control intent.

    This is discarded quota arithmetic, never an execution receipt or model row.
    Variable output text is bounded by its schema (six JSON bytes per character),
    token IDs by actual cap/vocabulary/int64. Source/input JSON is serialized ONCE,
    not once per original row. No omission, clipping or shortened generation cap.
    """
    i, source = payload.input.payload, payload.source.payload
    hashes = {field.name for field in fields(IrisScientificAblationControlledRequestReceiptV1)}
    request: dict[str, object] = {name: "f" * 64 for name in hashes if name.endswith("sha256")}
    sampled = i.decoding.strategy == "sampling"
    request.update(
        ablation_policy_json=canonical_json_bytes(source.ablation.model_dump(mode="json")).decode(),
        control_plan_json=i.generation_control_plan_json,
        selection="selected_best",
        completed_steps=source.selected_step,
        purpose=i.purpose,
        strategy=i.decoding.strategy,
        stop_token_ids=i.eos_token_ids,
        sampling={"temperature": i.decoding.temperature, "top_p": i.decoding.top_p},
        max_new_tokens=i.max_new_tokens,
        cycles=payload.cycles,
        sampling_seed=2**63 - 1 if sampled else None,
        kind="iris-scientific-ablation-controlled-request-execution-v1",
        semantics="prompt-only-workspace-control-full-caller-state-transaction-v1",
    )
    if not sampled:
        request["sampling_initial_state_sha256"] = None
        request["sampling_final_state_sha256"] = None
    if set(request) != hashes:
        raise ValueError("Full request schema changed; byte preflight must be reviewed")
    row_names = set(ScientificAblationControlledHeldoutRowPayloadV1.model_fields)
    row: dict[str, object] = {name: "f" * 64 for name in row_names if name.endswith("sha256")}
    digits = len(str(min(i.vocabulary_size - 1, 2**63 - 1)))
    row.update(
        kind="iris-scientific-ablation-controlled-heldout-row-v1",
        scope="actual_row_not_durable_ack_or_complete_evaluation",
        input=payload.input.model_dump(mode="json"),
        row_index=0,
        example_id="",
        task_type="",
        sampling_seed=2**63 - 1 if sampled else None,
        output_text="",
        token_ids=(),
        stop_reason="max_new_tokens",
        matched_stop_token_id=10 ** max(4, digits) - 1,
        target_token_count=1,
        target_nll_sum=0.0,
        request_receipt_json=canonical_json_bytes(request).decode(),
    )
    if set(row) != row_names:
        raise ValueError("Full row schema changed; byte preflight must be reviewed")
    base = len(canonical_json_file_bytes({"sha256": "f" * 64, "payload": row}))
    # These are schema upper bounds, not data-dependent generation/truncation.
    text = 6 * 1048576
    tokens = i.max_new_tokens * (digits + 1) - 1
    variable = (
        max(len(canonical_json_bytes(e.example_id)) - 2 for e in payload.rows)
        + max(len(canonical_json_bytes(e.task_type)) - 2 for e in payload.rows)
        + len(str(len(payload.rows) - 1))
        - 1
        + max(len(str(e.target_token_count)) - 1 for e in payload.rows)
    )
    # Covers the complete JSON spelling of any finite nonnegative binary64 NLL.
    bound = base + text + tokens + variable + 128
    if bound > _MAX_ROW:
        raise ValueError("Full controlled row cannot fit durable byte quota BEFORE forward")
    return bound


def build_scientific_ablation_controlled_capture_plan_v1(
    heldout: IrisScientificAblationControlledHeldoutV1, *, expected_input_sha256: str
) -> IrisScientificAblationControlledCapturePlanV1:
    """Own the complete row/request/target inventory without model forward or writes."""
    caller = None
    primary: BaseException | None = None
    secondary: list[BaseException] = []
    result = None
    dtype = torch.get_default_dtype()
    with torch.device("cpu"), torch.inference_mode(False), torch.enable_grad():
        try:
            _digest(expected_input_sha256)
            seal, manifest = _heldout_entry(heldout)
            if manifest.sha256 != expected_input_sha256:
                raise ValueError("Independently trusted held-out input differs")
            context, run = seal.context, seal.context._run
            _audit_file(seal)
            _audit_original(context)
            caller = capture_runtime_state(run._backbone.device)
            torch.set_default_dtype(torch.float32)
            source, reconstruction = context.source_manifest, context.reconstruction
            rows: list[ScientificAblationControlledCaptureExpectationV1] = []
            with torch.device(run._backbone.device):
                for index, row in enumerate(seal.rows):
                    if (
                        canonical_json_bytes(row.model_dump(mode="json"))
                        != seal.fingerprints[index]
                    ):
                        raise ValueError("Held-out inventory changed")
                    prompt_sha = sha256(row.prompt.encode()).hexdigest()
                    tokenized = run._backbone.tokenize(row.prompt)
                    prepared = prepare_teacher_forcing_input(
                        backbone=run._backbone,
                        prompt=row.prompt,
                        target_text=row.expected_response,
                        terminal_token_id=run._terminal_token_id,
                    )
                    target_sha = teacher_target_sha256_v1(
                        prepared.target_input_ids, prepared.target_attention_mask
                    )
                    cognitive_prompt = domain_sha256(
                        "iris/cognitive-ablation-prompt/v1", row.prompt.encode()
                    )
                    teacher_invocation = _payload_sha(
                        "iris/cognitive-ablation-teacher-invocation/v1",
                        {
                            "mode": "evaluation",
                            "policy_sha256": source.payload.ablation_policy_sha256,
                            "prompt_sha256": cognitive_prompt,
                            "target_sha256": target_sha,
                            "teacher_graph_sha256": run._teacher_graph_identity,
                            "sidecar_state_sha256": source.payload.teacher_sidecar_sha256,
                        },
                    )
                    invocation = _payload_sha(
                        "iris/scientific-ablation-invocation/v1",
                        {
                            "mode": "evaluation",
                            "example_sha256": domain_sha256(
                                "iris/scientific-ablation-example/v1", seal.fingerprints[index]
                            ),
                            "configuration_sha256": source.payload.model_configuration_sha256,
                            "teacher_invocation_sha256": teacher_invocation,
                            "scientific_state_sha256": (
                                run._optimizer._committed_scientific_state_sha256
                            ),
                        },
                    )
                    initial = None
                    if manifest.payload.decoding.strategy == "sampling":
                        seed = scientific_row_sampling_seed_v1(
                            manifest.payload.decoding.seed, row.id, prompt_sha
                        )
                        generator = torch.Generator(device=run._backbone.device).manual_seed(seed)
                        initial = tensor_sha256(
                            generator.get_state(), domain=_CORE_DOMAIN + "/sampling-state"
                        )
                    rows.append(
                        ScientificAblationControlledCaptureExpectationV1(
                            row_index=index,
                            example_id=row.id,
                            example_sha256=domain_sha256(
                                "iris/scientific-ablation-example/v1", seal.fingerprints[index]
                            ),
                            prompt_sha256=prompt_sha,
                            request_prompt_sha256=domain_sha256(
                                _CORE_DOMAIN + "/prompt", row.prompt.encode()
                            ),
                            cognitive_prompt_sha256=cognitive_prompt,
                            prompt_input_ids_sha256=tensor_sha256(
                                tokenized["input_ids"], domain=_CORE_DOMAIN + "/ids"
                            ),
                            prompt_attention_mask_sha256=tensor_sha256(
                                tokenized["attention_mask"], domain=_CORE_DOMAIN + "/mask"
                            ),
                            target_token_count=seal.target_counts[index],
                            teacher_target_sha256=target_sha,
                            teacher_invocation_sha256=invocation,
                            sampling_initial_state_sha256=initial,
                            task_type=row.task_type,
                        )
                    )
            payload = ScientificAblationControlledCapturePlanPayloadV1(
                input=manifest,
                source=source,
                reconstruction=reconstruction,
                scientific_state_sha256=run._optimizer._committed_scientific_state_sha256,
                teacher_graph_sha256=run._teacher_graph_identity,
                cycles=context._request._request._cycles,
                maximum_row_file_bytes=_MAX_ROW,
                rows=tuple(rows),
            )
            payload = ScientificAblationControlledCapturePlanPayloadV1.model_validate(
                {
                    **payload.model_dump(mode="python"),
                    "maximum_row_file_bytes": _row_file_bound(payload),
                }
            )
            result = IrisScientificAblationControlledCapturePlanV1(
                sha256=_payload_sha(_PLAN_DOMAIN, payload.model_dump(mode="json")), payload=payload
            )
            if len(_file(result)) > _MAX_PLAN:
                raise ValueError("Full capture plan exceeds metadata budget")
            _audit_file(seal)
            _heldout_entry(heldout)
        except BaseException as error:
            primary = error
        finally:
            if caller is not None:
                try:
                    _restore_caller(caller, dtype, seal.context._run._backbone.device)
                except BaseException as error:
                    secondary.append(error)
                    _poison_context(seal.context)
    _fail(primary, secondary)
    if result is None:
        raise IrisScientificAblationControlledCaptureError("No owned capture plan")
    return result


def _file(
    value: IrisScientificAblationControlledCapturePlanV1
    | IrisScientificAblationControlledCaptureAckV1
    | IrisScientificAblationControlledCaptureCompleteV1
    | IrisScientificAblationControlledHeldoutRowV1,
) -> bytes:
    return canonical_json_file_bytes(value.model_dump(mode="json"))


def _canonical(path: Path, maximum: int) -> bytes:
    content = _read_file(path, maximum)
    if not content.endswith(b"\n"):
        raise ValueError("Capture JSON must end with exactly one LF")
    _json(content[:-1])
    return content


def _inventory(path: Path) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for member in path.iterdir():
        if len(result) >= _MAX_DIRS:
            raise ValueError("Capture inventory count exceeded")
        meta = member.lstat()
        _regular(member, directory=member.is_dir())
        result[member.name] = _signature(meta)
    return result


def _check_row(
    row: IrisScientificAblationControlledHeldoutRowV1,
    plan: IrisScientificAblationControlledCapturePlanV1,
    index: int,
) -> None:
    p, e, s = row.payload, plan.payload.rows[index], plan.payload.source.payload
    if row.sha256 != _payload_sha(_ROW_DOMAIN, p.model_dump(mode="json")) or (
        p.input != plan.payload.input
        or p.row_index != index
        or p.example_id != e.example_id
        or p.example_sha256 != e.example_sha256
        or p.prompt_sha256 != e.prompt_sha256
        or p.task_type != e.task_type
        or p.target_token_count != e.target_token_count
        or p.teacher_target_sha256 != e.teacher_target_sha256
        or p.teacher_invocation_sha256 != e.teacher_invocation_sha256
        or p.scientific_state_sha256 != plan.payload.scientific_state_sha256
        or p.teacher_configuration_sha256 != s.model_configuration_sha256
        or p.request_receipt_sha256
        != domain_sha256(_REQUEST_DOMAIN, p.request_receipt_json.encode())
    ):
        raise ValueError("Captured row/complete target/native-best provenance differs")
    if len(_file(row)) > plan.payload.maximum_row_file_bytes:
        raise ValueError("Actual complete row exceeds independently bound plan byte quota")
    raw = _json(p.request_receipt_json.encode())
    if type(raw) is not dict or set(raw) != {
        field.name for field in fields(IrisScientificAblationControlledRequestReceiptV1)
    }:
        raise ValueError("Exact request receipt fields required")
    if type(raw["stop_token_ids"]) is not list or type(raw["sampling"]) is not dict:
        raise ValueError("Exact persisted request decoding fields required")
    raw["stop_token_ids"] = tuple(raw["stop_token_ids"])
    raw["sampling"] = TemperatureTopPSamplingConfig(**raw["sampling"])
    receipt = IrisScientificAblationControlledRequestReceiptV1(**raw)
    if canonical_json_bytes(asdict(receipt)).decode() != p.request_receipt_json:
        raise ValueError("Canonical persisted request receipt differs")
    i = p.input.payload
    with torch.device("cpu"), torch.inference_mode(False):
        token_sha = tensor_sha256(
            torch.tensor([p.token_ids], dtype=torch.int64), domain=_CORE_DOMAIN + "/generated-ids"
        )
    if (
        receipt.artifact_sha256 != s.inference_sha256
        or receipt.source_checkpoint_sha256 != s.best_checkpoint_sha256
        or receipt.lifecycle_sha256 != s.lifecycle_sha256
        or receipt.source_run_configuration_sha256 != s.run_configuration_sha256
        or receipt.source_model_configuration_sha256 != s.model_configuration_sha256
        or receipt.backbone_runtime_sha256 != s.backbone_runtime_sha256
        or receipt.ablation_policy_sha256 != s.ablation_policy_sha256
        or receipt.control_plan_sha256 != i.generation_control_plan_sha256
        or receipt.control_plan_json != i.generation_control_plan_json
        or receipt.ablation_policy_json
        != canonical_json_bytes(s.ablation.model_dump(mode="json")).decode()
        or receipt.sidecar_state_sha256 != s.request_sidecar_sha256
        or receipt.selection != "selected_best"
        or receipt.completed_steps != s.selected_step
        or receipt.purpose != i.purpose
        or receipt.prompt_sha256 != e.request_prompt_sha256
        or receipt.cognitive_prompt_sha256 != e.cognitive_prompt_sha256
        or receipt.prompt_input_ids_sha256 != e.prompt_input_ids_sha256
        or receipt.prompt_attention_mask_sha256 != e.prompt_attention_mask_sha256
        or receipt.generated_token_ids_sha256 != token_sha
        or receipt.strategy != i.decoding.strategy
        or receipt.stop_token_ids != i.eos_token_ids
        or receipt.max_new_tokens != i.max_new_tokens
        or receipt.cycles != plan.payload.cycles
        or receipt.sampling_seed != p.sampling_seed
        or receipt.sampling_initial_state_sha256 != e.sampling_initial_state_sha256
        or (receipt.sampling.temperature, receipt.sampling.top_p)
        != (i.decoding.temperature, i.decoding.top_p)
    ):
        raise ValueError("Captured actual request/selection/paired RNG facts differ")


def _genesis(plan_sha: str) -> str:
    return domain_sha256(_ACK_DOMAIN + "/genesis", plan_sha.encode())


def _complete(
    plan: IrisScientificAblationControlledCapturePlanV1,
    rows: tuple[IrisScientificAblationControlledHeldoutRowV1, ...],
    head: str,
) -> IrisScientificAblationControlledCaptureCompleteV1:
    if len(rows) != len(plan.payload.rows):
        raise ValueError("Cannot complete partial held-out coverage")
    payload = ScientificAblationControlledCaptureCompletePayloadV1(
        plan_sha256=plan.sha256,
        input_sha256=plan.payload.input.sha256,
        generation_control_plan_sha256=plan.payload.input.payload.generation_control_plan_sha256,
        acknowledgement_head_sha256=head,
        example_count=len(rows),
        target_token_count=sum(row.payload.target_token_count for row in rows),
        target_nll_sum=fsum(row.payload.target_nll_sum for row in rows),
    )
    return IrisScientificAblationControlledCaptureCompleteV1(
        sha256=_payload_sha(_COMPLETE_DOMAIN, payload.model_dump(mode="json")), payload=payload
    )


def _read(
    path: Path, plan_sha: str, maximum: int
) -> IrisScientificAblationControlledCaptureProgressV1:
    _chain(path)
    original = _signature(_regular(path, directory=True))
    inventory = _inventory(path)
    content = _canonical(path / "plan.json", min(_MAX_PLAN, maximum))
    plan = IrisScientificAblationControlledCapturePlanV1.model_validate_json(content[:-1])
    if plan.sha256 != plan_sha or plan.sha256 != _payload_sha(
        _PLAN_DOMAIN, plan.payload.model_dump(mode="json")
    ):
        raise ValueError("Independent capture plan SHA differs")
    if _file(plan) != content:
        raise ValueError("Exact canonical plan fields differ")
    if plan.payload.maximum_row_file_bytes != _row_file_bound(plan.payload):
        raise ValueError("Entire controlled row byte bound differs from complete plan")
    stages = 0
    row_names = []
    occupied = len(content)
    for name in inventory:
        if name in {"plan.json", "complete"}:
            continue
        if _STAGE_PATTERN.fullmatch(name):
            stages += 1
            stage = path / name
            _chain(stage)
            for member in stage.iterdir():
                if member.name not in _KNOWN_FILES:
                    raise ValueError("Foreign uncommitted stage member")
                occupied += _regular(member, directory=False).st_size
            continue
        if not _ROW_PATTERN.fullmatch(name):
            raise ValueError("Foreign capture member")
        row_names.append(name)
    row_names.sort()
    if row_names != [f"row-{index:05d}" for index in range(len(row_names))] or len(row_names) > len(
        plan.payload.rows
    ):
        raise ValueError("Acknowledged rows are not an exact ordered prefix")
    rows: list[IrisScientificAblationControlledHeldoutRowV1] = []
    acks: list[IrisScientificAblationControlledCaptureAckV1] = []
    head = _genesis(plan.sha256)
    for index, name in enumerate(row_names):
        directory = path / name
        _chain(directory)
        before = _inventory(directory)
        if set(before) != _ROW_FILES:
            raise ValueError("Row commit must contain exactly attempt/row/ack")
        occupied += sum(item[2] for item in before.values())
        if occupied > maximum:
            raise ValueError("Capture byte budget exceeded before row allocation")
        attempt = _json(_canonical(directory / "attempt.json", 4096)[:-1])
        if attempt != _attempt(plan.sha256, index, head):
            raise ValueError("Row computation intent differs from its ack prefix")
        row_content = _canonical(directory / "row.json", _MAX_ROW)
        row = IrisScientificAblationControlledHeldoutRowV1.model_validate_json(row_content[:-1])
        _check_row(row, plan, index)
        if _file(row) != row_content:
            raise ValueError("Exact row fields differ")
        ack_content = _canonical(directory / "ack.json", 4096)
        ack = IrisScientificAblationControlledCaptureAckV1.model_validate_json(ack_content[:-1])
        expected = _ack(plan.sha256, index, head, row, row_content)
        if ack != expected or _file(ack) != ack_content:
            raise ValueError("Row ack/content/answer/ordered-chain binding differs")
        if _inventory(directory) != before:
            raise ValueError("Row changed during readback")
        rows.append(row)
        acks.append(ack)
        head = ack.sha256
    complete = None
    if "complete" in inventory:
        directory = path / "complete"
        _chain(directory)
        before = _inventory(directory)
        if set(before) != {"complete.json"}:
            raise ValueError("Complete directory inventory differs")
        occupied += before["complete.json"][2]
        complete_content = _canonical(directory / "complete.json", 4096)
        complete = IrisScientificAblationControlledCaptureCompleteV1.model_validate_json(
            complete_content[:-1]
        )
        if complete != _complete(plan, tuple(rows), head) or _file(complete) != complete_content:
            raise ValueError("Complete coverage/NLL/ack head differs")
        if _inventory(directory) != before:
            raise ValueError("Complete commit changed during readback")
    if (
        occupied > maximum
        or _signature(_regular(path, directory=True)) != original
        or _inventory(path) != inventory
    ):
        raise ValueError("Capture changed during bounded readback")
    return IrisScientificAblationControlledCaptureProgressV1(
        path, plan, tuple(rows), tuple(acks), complete, stages
    )


def read_scientific_ablation_controlled_capture_v1(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_capture_sha256: str | None = None,
    maximum_storage_bytes: int = 512 * 1024**2,
) -> IrisScientificAblationControlledCaptureProgressV1:
    """Model-free integrity view; final scoring additionally requires trusted capture SHA."""
    try:
        _digest(expected_plan_sha256)
        if expected_capture_sha256 is not None:
            _digest(expected_capture_sha256)
        _budgets(maximum_storage_bytes, 0)
        path = _path(path)
        if path.name != expected_plan_sha256:
            raise ValueError("Capture directory name differs from trusted plan")
        result = _read(path, expected_plan_sha256, maximum_storage_bytes)
        if expected_capture_sha256 is not None and (
            result.complete is None or result.complete.sha256 != expected_capture_sha256
        ):
            raise ValueError("Independently trusted complete capture differs")
        return result
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error


def _attempt(plan: str, index: int, previous: str) -> dict[str, object]:
    return {
        "kind": "iris-scientific-ablation-controlled-capture-attempt-v1",
        "status": "started_unacknowledged",
        "plan_sha256": plan,
        "row_index": index,
        "previous_ack_sha256": previous,
    }


def _ack(
    plan: str,
    index: int,
    previous: str,
    row: IrisScientificAblationControlledHeldoutRowV1,
    content: bytes,
) -> IrisScientificAblationControlledCaptureAckV1:
    payload = ScientificAblationControlledCaptureAckPayloadV1(
        plan_sha256=plan,
        row_index=index,
        previous_ack_sha256=previous,
        row_sha256=row.sha256,
        row_file_sha256=sha256(content).hexdigest(),
        row_file_bytes=len(content),
        answer_sha256=domain_sha256(_ACK_DOMAIN + "/answer", row.payload.output_text.encode()),
    )
    return IrisScientificAblationControlledCaptureAckV1(
        sha256=_payload_sha(_ACK_DOMAIN, payload.model_dump(mode="json")), payload=payload
    )


@dataclass(frozen=True, slots=True)
class _Claim:
    store: Path
    identity: tuple[int, int]
    lock: ProcessFileLock

    def check(self) -> None:
        self.lock.check()
        _chain(self.store)
        if _identity(_regular(self.store, directory=True)) != self.identity:
            raise ValueError("Capture store was replaced")


@dataclass(frozen=True, slots=True)
class _Stage:
    path: Path
    identity: tuple[int, int]
    parent_identity: tuple[int, int]

    def check(self, claim: _Claim) -> None:
        claim.check()
        _chain(self.path)
        if (
            _identity(_regular(self.path, directory=True)) != self.identity
            or _identity(_regular(self.path.parent, directory=True)) != self.parent_identity
        ):
            raise ValueError("Capture staging directory/parent was replaced")


def _stage(parent: Path, claim: _Claim) -> _Stage:
    claim.check()
    parent_id = _identity(_regular(parent, directory=True))
    candidate = parent / f".iris-controlled-capture-{uuid4().hex}.tmp"
    os.mkdir(candidate, 0o700)
    return _Stage(candidate, _identity(_regular(candidate, directory=True)), parent_id)


def _write(stage: _Stage, name: str, content: bytes, claim: _Claim) -> None:
    stage.check(claim)
    if name not in _KNOWN_FILES:
        raise ValueError("Unknown capture write member")
    path = stage.path / name
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    primary = None
    try:
        opened = _identity(os.fstat(descriptor))
        if opened != _identity(_regular(path, directory=False)):
            raise ValueError("Created capture file identity differs")
        _write_bytes(descriptor, content)
        stage.check(claim)
        if opened != _identity(_regular(path, directory=False)):
            raise ValueError("Capture file was replaced during write")
    except BaseException as error:
        primary = error
    _finish_descriptor(primary, lambda: os.close(descriptor))
    if _read_file(path, len(content)) != content:
        raise ValueError("Capture exact staged bytes differ")


def _publish(stage: _Stage, target: Path, files: dict[str, bytes], claim: _Claim) -> None:
    stage.check(claim)
    if set(_inventory(stage.path)) != set(files):
        raise ValueError("Unknown stage member; refusing publication")
    for name, content in files.items():
        if _read_file(stage.path / name, len(content)) != content:
            raise ValueError("Staged capture changed")
    try:
        rename_directory_with_windows_retry(stage.path, target)
    except OSError:
        if not os.path.lexists(target):
            raise
    _chain(target)
    if _identity(_regular(target, directory=True)) != stage.identity or os.path.lexists(stage.path):
        raise ValueError("Publication did not transfer this exact owned stage identity")
    if set(_inventory(target)) != set(files):
        raise ValueError("Concurrent/uncertain capture publication conflicts")
    for name, content in files.items():
        if _read_file(target / name, len(content)) != content:
            raise ValueError("Concurrent/uncertain capture bytes differ")
    claim.check()


def _occupied(claim: _Claim) -> int:
    """Bound exact dedicated store depth; charge all old stages, never reap them."""
    occupied, directories, files = 0, 0, 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal occupied, directories, files
        claim.check()
        _chain(directory)
        for entry in directory.iterdir():
            if entry.name == _LOCK and depth == 0:
                continue
            if entry.is_dir():
                directories += 1
                valid = (
                    _STAGE_PATTERN.fullmatch(entry.name) is not None
                    or (depth == 0 and re.fullmatch(r"[0-9a-f]{64}", entry.name) is not None)
                    or (
                        depth == 1
                        and (
                            entry.name == "complete"
                            or _ROW_PATTERN.fullmatch(entry.name) is not None
                        )
                    )
                )
                if not valid or depth >= 2 or directories > _MAX_DIRS:
                    raise ValueError("Foreign/deep/excessive capture store directory")
                visit(entry, depth + 1)
            else:
                files += 1
                if entry.name not in _KNOWN_FILES or depth == 0 or files > _MAX_FILES:
                    raise ValueError("Foreign/excessive capture store file")
                occupied += _regular(entry, directory=False).st_size
                if occupied > _MAX_STORAGE:
                    raise ValueError("Capture global byte budget exceeded")

    visit(claim.store, 0)
    return occupied


def _budgets(maximum: int, reserve: int) -> None:
    if (
        type(maximum) is not int
        or not 1 <= maximum <= _MAX_STORAGE
        or type(reserve) is not int
        or not 0 <= reserve <= _MAX_STORAGE
    ):
        raise ValueError("Explicit exact capture quota/free-space reserve required")


def _reserve(claim: _Claim, required: int, maximum: int, reserve: int) -> None:
    claim.check()
    if (
        _occupied(claim) + required > maximum
        or shutil.disk_usage(claim.store).free < required + reserve
    ):
        raise ValueError("Capture quota/free-space reserve exceeded BEFORE computation/write")


def capture_scientific_ablation_controlled_heldout_v1(
    heldout: IrisScientificAblationControlledHeldoutV1,
    *,
    root: Path,
    expected_input_sha256: str,
    expected_plan_sha256: str,
    rows_per_call: int | None = None,
    expected_chunk: ScientificAblationKernelChunkExpectationV1 | None = None,
    _recovery: _KernelIntervalPermission | None = None,
    maximum_storage_bytes: int = 512 * 1024**2,
    minimum_free_bytes: int = 1024**3,
) -> IrisScientificAblationControlledCaptureProgressV1:
    """Generate only missing prefix rows under lock; publish complete ONLY at full coverage."""
    try:
        expected_chunk = checked_kernel_chunk_v1(
            expected_chunk,
            expected_plan_sha256=expected_plan_sha256,
            rows_per_call=rows_per_call,
        )
        _checked_interval_permission(
            _recovery,
            root=root,
            namespace=_STORE,
            expected_chunk=expected_chunk,
            plan_sha256=expected_plan_sha256,
            rows_per_call=rows_per_call,
            maximum_storage_bytes=maximum_storage_bytes,
            minimum_free_bytes=minimum_free_bytes,
        )
        _budgets(maximum_storage_bytes, minimum_free_bytes)
        _digest(expected_plan_sha256)
        if rows_per_call is not None and (
            type(rows_per_call) is not int or not 1 <= rows_per_call <= 5000
        ):
            raise ValueError("Exact positive bounded rows_per_call required")
        plan = build_scientific_ablation_controlled_capture_plan_v1(
            heldout, expected_input_sha256=expected_input_sha256
        )
        if plan.sha256 != expected_plan_sha256:
            raise ValueError("Independent complete capture plan differs BEFORE writes")
        store = _path(root) / _STORE
        for parent in (store, *store.parents):
            if os.path.lexists(parent):
                _regular(parent, directory=True)
        store.mkdir(parents=True, exist_ok=True)
        _chain(store)
        identity = _identity(_regular(store, directory=True))
    except BaseException as error:
        _fail(error, [])
        raise AssertionError("Unreachable") from error
    primary: BaseException | None = None
    secondary: list[BaseException] = []
    result = None
    try:
        with process_file_lock(store / _LOCK) as lock:
            claim = _Claim(store, identity, lock)
            try:
                claim.check()
                if _occupied(claim) > maximum_storage_bytes:
                    raise ValueError("Existing capture store exceeds quota")
                target = store / plan.sha256
                if not os.path.lexists(target):
                    if _recovery is not None:
                        raise ValueError(
                            "Recovery cannot initialize a missing original kernel plan"
                        )
                    check_kernel_chunk_prefix_v1(
                        expected_chunk,
                        plan_sha256=plan.sha256,
                        acknowledgement_head_sha256=_genesis(plan.sha256),
                        acknowledged_rows=0,
                        total_rows=len(plan.payload.rows),
                        rows_per_call=rows_per_call,
                    )
                    content = _file(plan)
                    _reserve(claim, len(content), maximum_storage_bytes, minimum_free_bytes)
                    stage = _stage(store, claim)
                    _write(stage, "plan.json", content, claim)
                    _publish(stage, target, {"plan.json": content}, claim)
                progress = _read(target, plan.sha256, maximum_storage_bytes)
                if progress.plan != plan:
                    raise ValueError("Existing capture differs from actual complete owned plan")
                rows, acks = list(progress.rows), list(progress.acknowledgements)
                head = _genesis(plan.sha256) if not acks else acks[-1].sha256
                if _recovery is None:
                    check_kernel_chunk_prefix_v1(
                        expected_chunk,
                        plan_sha256=plan.sha256,
                        acknowledgement_head_sha256=head,
                        acknowledged_rows=len(rows),
                        total_rows=len(plan.payload.rows),
                        rows_per_call=rows_per_call,
                    )
                    stop = (
                        len(plan.payload.rows)
                        if rows_per_call is None
                        else min(len(plan.payload.rows), len(rows) + rows_per_call)
                    )
                else:
                    stop = _locked_interval_stop(
                        _recovery,
                        root=root,
                        namespace=_STORE,
                        expected_chunk=expected_chunk,
                        plan_sha256=plan.sha256,
                        acknowledgement_head_sha256=head,
                        acknowledged_rows=len(rows),
                        total_rows=len(plan.payload.rows),
                        rows_per_call=rows_per_call,
                        acknowledgement_prefix=tuple(ack.sha256 for ack in acks),
                        kernel_lock=lock,
                        maximum_storage_bytes=maximum_storage_bytes,
                        minimum_free_bytes=minimum_free_bytes,
                    )
                for index in range(len(rows), stop):
                    # Reserve the declared worst-case row+metadata BEFORE the expensive call.
                    _reserve(
                        claim,
                        plan.payload.maximum_row_file_bytes + 16384,
                        maximum_storage_bytes,
                        minimum_free_bytes,
                    )
                    stage = _stage(target, claim)
                    attempt = canonical_json_file_bytes(_attempt(plan.sha256, index, head))
                    _write(stage, "attempt.json", attempt, claim)
                    try:
                        row = heldout.execute_row(index)
                        row = IrisScientificAblationControlledHeldoutRowV1.model_validate_json(
                            _file(row)[:-1]
                        )
                        _check_row(row, plan, index)
                        row_content = _file(row)
                        if len(row_content) > _MAX_ROW:
                            raise ValueError("Generated row exceeds explicit durable bound")
                        ack = _ack(plan.sha256, index, head, row, row_content)
                        files = {
                            "attempt.json": attempt,
                            "row.json": row_content,
                            "ack.json": _file(ack),
                        }
                        _write(stage, "row.json", files["row.json"], claim)
                        _write(stage, "ack.json", files["ack.json"], claim)
                        _publish(stage, target / f"row-{index:05d}", files, claim)
                    except BaseException as error:
                        # Never hide the first signal or change a committed target.
                        primary = error
                        if os.path.lexists(stage.path):
                            try:
                                failure = canonical_json_file_bytes(
                                    {
                                        "kind": (
                                            "iris-scientific-ablation-controlled-capture-failure-v1"
                                        ),
                                        "status": "interrupted_unacknowledged"
                                        if _has_signal(error)
                                        else "failed_unacknowledged",
                                        "plan_sha256": plan.sha256,
                                        "row_index": index,
                                        "exception_type": type(error).__name__,
                                    }
                                )
                                _write(stage, "failure.json", failure, claim)
                            except BaseException as cleanup:
                                secondary.append(cleanup)
                        raise
                    rows.append(row)
                    acks.append(ack)
                    head = ack.sha256
                if len(rows) == len(plan.payload.rows) and progress.complete is None:
                    complete_content = _file(_complete(plan, tuple(rows), head))
                    _reserve(
                        claim, len(complete_content), maximum_storage_bytes, minimum_free_bytes
                    )
                    stage = _stage(target, claim)
                    _write(stage, "complete.json", complete_content, claim)
                    _publish(stage, target / "complete", {"complete.json": complete_content}, claim)
                result = _read(target, plan.sha256, maximum_storage_bytes)
                _heldout_entry(heldout)
                claim.check()
            except BaseException as error:
                if primary is None:
                    primary = error
                elif primary is not error:
                    secondary.append(error)
    except BaseException as error:
        if primary is None:
            primary = error
        elif primary is not error:
            secondary.append(error)
    _fail(primary, secondary)
    if result is None:
        raise IrisScientificAblationControlledCaptureError("No verified durable capture progress")
    return result


__all__ = [
    "IrisScientificAblationControlledCaptureError",
    "IrisScientificAblationControlledCaptureProgressV1",
    "build_scientific_ablation_controlled_capture_plan_v1",
    "capture_scientific_ablation_controlled_heldout_v1",
    "read_scientific_ablation_controlled_capture_v1",
]
