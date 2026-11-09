"""Offline full-split inference with complete-input preflight and exact private capture."""

from collections.abc import Generator
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args

import torch

from iris.application.arm_profiles import IrisArmExecutionProfileV4
from iris.application.arm_request_command_v4 import (
    _require_bootstrap_result,
    _require_checkpoint_identity,
    _require_exact_execution,
    require_arm_request_command_profile_v4,
)
from iris.application.arm_training_checkpoint import IrisArmTeacherCheckpointIdentity
from iris.application.bootstrap import _resolve_device
from iris.application.errors import IrisStudyGenerationError
from iris.application.request_artifact_v4 import (
    build_request_artifact_v4,
    read_request_artifact_v4,
    write_request_artifact_v4,
)
from iris.application.request_checkpoint_bootstrap import bootstrap_arm_request_runtime_v4
from iris.application.request_command import (
    _create_sampling_generator,
    _validate_request_command,
    _validate_sampling_generator,
)
from iris.application.study_benchmark import validate_study_benchmark_v1
from iris.application.study_checkpoint_scope import _scoped_generation
from iris.application.study_dataset_io import _explicit_path, _is_link_or_reparse
from iris.application.study_generation_io import (
    IrisStudyGenerationWriteResultV1,
    validate_study_generation_v1,
    write_study_generation_v1,
)
from iris.application.study_input_preflight import (
    IrisStudyExampleTokenBudgetV1,
    IrisStudyInputPreflightV1,
    preflight_study_inputs_v1,
)
from iris.application.verification_v4 import verify_request_artifact_v4
from iris.artifacts.experiment_v1 import ExperimentExampleV1, ExperimentSplitV1
from iris.artifacts.request_v1 import DecodingProvenanceV1
from iris.artifacts.request_v4 import IrisRequestArtifactEnvelopeV4, RequestArmCheckpointTransferV4
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_dataset_v1 import StudyExampleAssignmentV1
from iris.artifacts.study_generation_v1 import (
    IrisStudyGenerationV1,
    StudyGeneratedAnswerV1,
    StudyGenerationPayloadV1,
    StudyGenerationSchemaV1,
)
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256, tensor_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state


def _preflight_digest(report: IrisStudyInputPreflightV1) -> str:
    return domain_sha256(
        "iris/study-generation/v1/input-preflight", canonical_json_bytes(asdict(report))
    )


def _check_tokens(
    ids: torch.Tensor, mask: torch.Tensor, observation: IrisStudyExampleTokenBudgetV1
) -> None:
    if (
        tensor_sha256(ids, domain="iris/request-artifact/v3/prompt-input-ids")
        != observation.request_v4_prompt_input_ids_sha256
        or tensor_sha256(mask, domain="iris/request-artifact/v3/prompt-attention-mask")
        != observation.request_v4_prompt_attention_mask_sha256
    ):
        raise ValueError("runtime tokenization differs from the complete input preflight")


def _check_request(
    request: IrisRequestArtifactEnvelopeV4,
    *,
    example: ExperimentExampleV1,
    answer: StudyGeneratedAnswerV1,
    observation: IrisStudyExampleTokenBudgetV1,
    preflight: IrisStudyInputPreflightV1,
) -> None:
    payload = request.payload
    provenance = payload.request
    prompt = example.prompt.encode("utf-8", errors="strict")
    output = answer.output_text.encode("utf-8", errors="strict")
    if (
        request.payload_sha256 != answer.request_payload_sha256
        or payload.configuration.sha256 != preflight.configuration_sha256
        or payload.execution_profile_set_sha256 != preflight.execution_profile_set_sha256
        or payload.runtime.software.transformers != preflight.transformers_version
        or payload.runtime.software.tokenizers != preflight.tokenizers_version
        or provenance.decoding != answer.decoding
        or provenance.prompt.sha256 != domain_sha256("iris/request-artifact/v3/prompt", prompt)
        or provenance.prompt.byte_length != len(prompt)
        or payload.generation.text.sha256
        != domain_sha256("iris/request-artifact/v3/generated-text", output)
        or payload.generation.text.byte_length != len(output)
        or provenance.physical_prompt_tokens != observation.prompt_tokens
        or provenance.attended_prompt_tokens != observation.prompt_tokens
        or provenance.prompt_input_ids.sha256 != observation.request_v4_prompt_input_ids_sha256
        or provenance.prompt_attention_mask.sha256
        != observation.request_v4_prompt_attention_mask_sha256
    ):
        raise ValueError("request differs from exact generation/input/software identities")


@_scoped_generation
def verify_study_generation_v1(
    generation: IrisStudyGenerationV1, *, benchmark: IrisStudyBenchmarkV1, request_root: Path
) -> IrisStudyGenerationV1:
    """Recheck all selected membership, requests/checkpoints, inputs and private answers.

    Does not regenerate, score, or execute code. This is not execution attestation.
    """
    try:
        owned = validate_study_generation_v1(generation)
        study = validate_study_benchmark_v1(benchmark)
        payload, dataset = owned.payload, study.payload.dataset.payload
        if (
            not isinstance(request_root, Path)
            or not request_root.is_absolute()
            or not request_root.name
            or ".." in request_root.parts
        ):
            raise ValueError("request root must be an explicit absolute Path")
        if (
            payload.benchmark_payload_sha256 != study.payload_sha256
            or payload.dataset_payload_sha256 != study.payload.dataset.payload_sha256
            or payload.manifest_payload_sha256 != dataset.manifest.payload_sha256
            or payload.protocol_declaration_payload_sha256
            != dataset.protocol_declaration_payload_sha256
        ):
            raise ValueError("generation and benchmark identities differ")
        selected = tuple(
            (index, example, assignment)
            for index, (example, assignment) in enumerate(
                zip(dataset.manifest.payload.examples, dataset.assignments, strict=True)
            )
            if example.split == payload.split
        )
        if tuple(item.example_id for item in payload.answers) != tuple(
            example.example_id for _, example, _ in selected
        ):
            raise ValueError("generation did not preserve the complete selected split in order")
        paths = tuple(
            request_root / "requests" / "v4" / f"{item.request_payload_sha256}.json"
            for item in payload.answers
        )
        first_report = verify_request_artifact_v4(paths[0])
        first = read_request_artifact_v4(paths[0])
        if first.payload_sha256 != first_report.payload_sha256:
            raise ValueError("first request changed before preflight")
        preflight = preflight_study_inputs_v1(
            study,
            config=first.payload.configuration.value,
            terminal_token_id=payload.terminal_token_id,
        )
        if (
            not preflight.inputs_fit
            or _preflight_digest(preflight) != payload.input_preflight_sha256
            or preflight.configuration_sha256 != payload.configuration_sha256
            or preflight.execution_profile_set_sha256 != payload.execution_profile_set_sha256
        ):
            raise ValueError("generation preflight no longer matches complete input budgets")
        initial = []
        for path, answer, (index, example, assignment) in zip(
            paths, payload.answers, selected, strict=True
        ):
            report = verify_request_artifact_v4(path)
            request = read_request_artifact_v4(path)
            if (
                report.payload_sha256 != request.payload_sha256
                or request.payload.topology.arm_id != payload.arm_id
                or request.payload.checkpoint_transfer != payload.checkpoint_transfer
                or request.payload.protocol_declaration_payload_sha256
                != payload.protocol_declaration_payload_sha256
                or answer.task_family != assignment.task_family
            ):
                raise ValueError("generation arm, family or dependency identity disagrees")
            _check_request(
                request,
                example=example,
                answer=answer,
                observation=preflight.examples[index],
                preflight=preflight,
            )
            initial.append((path, report, request))
        for path, report, request in initial:
            if (
                verify_request_artifact_v4(path) != report
                or read_request_artifact_v4(path) != request
            ):
                raise ValueError("generation evidence changed during verification")
        return owned
    except Exception:
        raise IrisStudyGenerationError(
            "Iris study generation could not be independently verified"
        ) from None


def _require_new_destination(path: str | Path) -> Path:
    destination = _explicit_path(path)
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("generation destination already exists; verify it rather than regenerate")
    for parent in destination.absolute().parents:
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            continue
        if _is_link_or_reparse(metadata) or not parent.is_dir():
            raise ValueError("generation parents must be real directories")
    return destination


@dataclass(frozen=True, slots=True)
class _PreparedStudyGeneration:
    study: IrisStudyBenchmarkV1
    config: IrisBootstrapConfig
    profile: IrisArmExecutionProfileV4
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None
    split: ExperimentSplitV1
    policies: tuple[DecodingProvenanceV1, ...]
    selected: tuple[tuple[int, ExperimentExampleV1, StudyExampleAssignmentV1], ...]
    terminal_token_id: int
    max_artifact_bytes: int
    preflight: IrisStudyInputPreflightV1

    @property
    def root(self) -> Path:
        return Path(self.config.runtime.output_dir)


@dataclass(frozen=True, slots=True)
class _GeneratedStudyAnswer:
    request: IrisRequestArtifactEnvelopeV4
    answer: StudyGeneratedAnswerV1


def _prepare_study_generation(
    benchmark: IrisStudyBenchmarkV1,
    *,
    config: IrisBootstrapConfig,
    arm_id: object,
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None,
    split: ExperimentSplitV1,
    decoding: tuple[DecodingProvenanceV1, ...],
    terminal_token_id: int,
    max_artifact_bytes: int,
) -> _PreparedStudyGeneration:
    """Own the complete immutable inputs and validate every policy before model loading."""
    study = validate_study_benchmark_v1(benchmark)
    if type(config) is not IrisBootstrapConfig:
        raise TypeError("generation requires an exact configuration")
    owned_config = IrisBootstrapConfig.model_validate(config.model_dump(mode="python"))
    root = Path(owned_config.runtime.output_dir)
    if not root.is_absolute() or not root.name or ".." in root.parts:
        raise ValueError("generation requires an explicit absolute artifact root")
    if type(split) is not str or split not in get_args(ExperimentSplitV1):
        raise ValueError("unknown selected split")
    if type(max_artifact_bytes) is not int or not 1024 <= max_artifact_bytes <= 64 * 1024 * 1024:
        raise ValueError("generation requires an explicit byte budget from 1 KiB to 64 MiB")
    profile = require_arm_request_command_profile_v4(arm_id)
    _require_checkpoint_identity(profile=profile, identity=checkpoint_identity)
    dataset = study.payload.dataset.payload
    selected = tuple(
        (index, example, assignment)
        for index, (example, assignment) in enumerate(
            zip(dataset.manifest.payload.examples, dataset.assignments, strict=True)
        )
        if example.split == split
    )
    if type(decoding) is not tuple or len(decoding) != len(selected):
        raise ValueError("one explicit decoding policy is required for every selected example")
    if any(type(policy) is not DecodingProvenanceV1 for policy in decoding):
        raise TypeError("decoding policies must have exact request-artifact types")
    policies = tuple(
        DecodingProvenanceV1.model_validate(policy.model_dump(mode="python")) for policy in decoding
    )
    first_policy = policies[0]
    if any(
        policy.model_dump(exclude={"sampling_seed"})
        != first_policy.model_dump(exclude={"sampling_seed"})
        for policy in policies
    ):
        raise ValueError("batch decoding policies may differ only in explicit sampling seed")
    sampling = TemperatureTopPSamplingConfig(
        temperature=first_policy.temperature, top_p=first_policy.top_p
    )
    for policy, (_, example, _) in zip(policies, selected, strict=True):
        if policy.allow_download:
            raise ValueError("study generation cannot allow downloads")
        _validate_request_command(
            prompt=example.prompt,
            strategy=policy.strategy,
            stop_token_ids=policy.stop_token_ids,
            sampling=sampling,
            sampling_seed=policy.sampling_seed,
            allow_download=False,
        )
    if (
        sum(len(canonical_json_bytes(policy.model_dump(mode="json"))) + 200 for policy in policies)
        > max_artifact_bytes
    ):
        raise ValueError("byte budget cannot hold even the minimum answer metadata")
    preflight = preflight_study_inputs_v1(
        study, config=owned_config, terminal_token_id=terminal_token_id
    )
    if not preflight.inputs_fit:
        raise ValueError("complete benchmark input budgets must fit before model loading")
    return _PreparedStudyGeneration(
        study,
        owned_config,
        profile,
        checkpoint_identity,
        split,
        policies,
        selected,
        terminal_token_id,
        max_artifact_bytes,
        preflight,
    )


def _iter_study_generation_answers(
    prepared: _PreparedStudyGeneration, *, start_index: int, answer_bytes: int
) -> Generator[_GeneratedStudyAnswer, None, None]:
    """One runtime for the missing suffix; caller must close on publication failure."""
    if type(start_index) is not int or not 0 <= start_index < len(prepared.selected):
        raise ValueError("generation suffix index is outside its planned membership")
    if type(answer_bytes) is not int or not 0 <= answer_bytes <= prepared.max_artifact_bytes:
        raise ValueError("invalid retained answer-byte count")
    owned_config, profile = prepared.config, prepared.profile
    preflight, policies, selected = prepared.preflight, prepared.policies, prepared.selected
    checkpoint_identity = prepared.checkpoint_identity
    max_artifact_bytes = prepared.max_artifact_bytes
    first_policy = policies[0]
    sampling = TemperatureTopPSamplingConfig(
        temperature=first_policy.temperature, top_p=first_policy.top_p
    )
    device_info, _ = _resolve_device(owned_config)
    snapshot = capture_runtime_state(device_info.device)
    try:
        result = _require_bootstrap_result(
            bootstrap_arm_request_runtime_v4(
                profile.arm_id,
                owned_config,
                checkpoint_identity=checkpoint_identity,
                stop_token_ids=first_policy.stop_token_ids,
                sampling=sampling,
            ),
            profile=profile,
        )
        runtime = result.runtime
        if (
            runtime.configuration_sha256 != preflight.configuration_sha256
            or runtime.backbone.max_position_embeddings != preflight.max_position_embeddings
        ):
            raise ValueError("loaded runtime configuration/context differs from preflight")
        for policy, (index, example, assignment) in zip(
            policies[start_index:], selected[start_index:], strict=True
        ):
            observation = preflight.examples[index]
            tokenized = runtime.backbone.tokenize(example.prompt)
            _check_tokens(tokenized["input_ids"], tokenized["attention_mask"], observation)
            generator = None
            if policy.sampling_seed is not None:
                generator = _validate_sampling_generator(
                    _create_sampling_generator(
                        device=runtime.device_info.device,
                        seed=policy.sampling_seed,
                    ),
                    device=runtime.device_info.device,
                    seed=policy.sampling_seed,
                )
            execution = _require_exact_execution(
                runtime.orchestrator.execute(
                    example.prompt,
                    strategy=policy.strategy,
                    generator=generator,
                ),
                arm_id=profile.arm_id,
            )
            _check_tokens(execution.prompt_input_ids, execution.prompt_attention_mask, observation)
            output = execution.generation.text
            encoded = output.encode("utf-8", errors="strict")
            if len(encoded) > max_artifact_bytes:
                raise ValueError("answer exceeds the whole generation byte budget")
            request = build_request_artifact_v4(
                prompt=example.prompt,
                strategy=policy.strategy,
                stop_token_ids=policy.stop_token_ids,
                sampling=sampling,
                sampling_seed=policy.sampling_seed,
                allow_download=False,
                runtime=runtime,
                execution=execution,
                checkpoint_transfer=result.checkpoint_transfer,
            )
            answer = StudyGeneratedAnswerV1(
                example_id=example.example_id,
                task_family=assignment.task_family,
                request_payload_sha256=request.payload_sha256,
                decoding=policy,
                output_text=output,
                output_sha256=bytes_sha256(encoded),
                output_byte_length=len(encoded),
            )
            _check_request(
                request,
                example=example,
                answer=answer,
                observation=observation,
                preflight=preflight,
            )
            answer_bytes += len(canonical_json_bytes(answer.model_dump(mode="json")))
            if answer_bytes > max_artifact_bytes:
                raise ValueError("answer capture exceeds the generation byte budget")
            yield _GeneratedStudyAnswer(request, answer)
    finally:
        restore_runtime_state(snapshot)


def _assemble_study_generation(
    prepared: _PreparedStudyGeneration,
    answers: tuple[StudyGeneratedAnswerV1, ...],
    transfer: RequestArmCheckpointTransferV4 | None,
) -> IrisStudyGenerationV1:
    study, profile, preflight = prepared.study, prepared.profile, prepared.preflight
    dataset = study.payload.dataset.payload
    terminal_token_id, split = prepared.terminal_token_id, prepared.split
    max_artifact_bytes = prepared.max_artifact_bytes
    payload = StudyGenerationPayloadV1(
        benchmark_payload_sha256=study.payload_sha256,
        dataset_payload_sha256=study.payload.dataset.payload_sha256,
        manifest_payload_sha256=dataset.manifest.payload_sha256,
        protocol_declaration_payload_sha256=dataset.protocol_declaration_payload_sha256,
        configuration_sha256=preflight.configuration_sha256,
        execution_profile_set_sha256=preflight.execution_profile_set_sha256,
        input_preflight_sha256=_preflight_digest(preflight),
        terminal_token_id=terminal_token_id,
        arm_id=profile.arm_id,
        split=split,
        checkpoint_transfer=transfer,
        max_artifact_bytes=max_artifact_bytes,
        answers=tuple(answers),
        membership_policy="all_examples_in_selected_split_in_manifest_order",
        status="complete_generation_not_scored",
        verification_scope="bindings_not_execution_attestation",
        scientific_launch_status="not_assessed",
    )
    generation = IrisStudyGenerationV1(
        schema_identity=StudyGenerationSchemaV1(
            artifact_type="iris.study_generation",
            version=1,
            canonicalization="iris-json-v1",
            hash_algorithm="sha256",
            content_policy="private_generated_answers_and_metadata",
        ),
        payload_sha256=domain_sha256(
            "iris/study-generation/v1/payload",
            canonical_json_bytes(payload.model_dump(mode="json")),
        ),
        payload=payload,
    )
    return verify_study_generation_v1(generation, benchmark=study, request_root=prepared.root)


def generate_study_split_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    config: IrisBootstrapConfig,
    arm_id: object,
    checkpoint_identity: IrisArmTeacherCheckpointIdentity | None,
    split: ExperimentSplitV1,
    decoding: tuple[DecodingProvenanceV1, ...],
    terminal_token_id: int,
    max_artifact_bytes: int,
    path: str | Path,
) -> IrisStudyGenerationWriteResultV1:
    """Generate the complete split with one runtime and the unchanged v1 publication.

    This compatibility API still publishes raw answers only at the end. It does
    not claim durable resume; the managed run layer owns that separate contract.
    """
    try:
        destination = _require_new_destination(path)
        prepared = _prepare_study_generation(
            benchmark,
            config=config,
            arm_id=arm_id,
            checkpoint_identity=checkpoint_identity,
            split=split,
            decoding=decoding,
            terminal_token_id=terminal_token_id,
            max_artifact_bytes=max_artifact_bytes,
        )
        device_info, _ = _resolve_device(prepared.config)
        snapshot = capture_runtime_state(device_info.device)
        try:
            answers = []
            transfer = None
            with closing(
                _iter_study_generation_answers(prepared, start_index=0, answer_bytes=0)
            ) as generated:
                for item in generated:
                    write_request_artifact_v4(item.request)
                    answers.append(item.answer)
                    transfer = item.request.payload.checkpoint_transfer
            verified = _assemble_study_generation(prepared, tuple(answers), transfer)
        finally:
            restore_runtime_state(snapshot)
        return write_study_generation_v1(verified, path=destination)
    except Exception:
        raise IrisStudyGenerationError("Iris complete-split generation did not complete") from None


__all__ = ["generate_study_split_v1", "verify_study_generation_v1"]
