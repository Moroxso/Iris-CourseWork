"""Request-bound Python scoring and read-only recomputation of private observations."""

import base64
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.application.request_artifact_v4 import read_request_artifact_v4
from iris.application.study_benchmark import validate_study_benchmark_v1
from iris.application.study_code_evaluation import _score_study_python_answer_v1
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.application.verification_v4 import verify_request_artifact_v4
from iris.artifacts.code_execution_v1 import PythonSandboxPolicyV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1, StudyPythonIntegerListRuleV1
from iris.artifacts.study_code_evaluation_v1 import (
    IrisStudyCodeEvaluationV1,
    StudyCodeEvaluationPayloadV1,
    StudyCodeEvaluationSchemaV1,
    StudyCodeObservationV1,
    StudyCodeReceiptV1,
    StudyCodeStreamV1,
)
from iris.artifacts.study_evaluation_v1 import StudyEvaluationInputBindingV1
from iris.integrations.bounded_process import ProcessResult
from iris.integrations.python_sandbox import IMAGE, PythonSandboxObservationV1, _worker
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_PAYLOAD_DOMAIN = "iris/study-code-evaluation/v1/payload"
_OBSERVATION_DOMAIN = "iris/study-code-evaluation/v1/observation"
_RULE_DOMAIN = "iris/study-code-evaluation/v1/scoring-rule"
_MAX_CODE_EVALUATION_BYTES = 12 * 1024 * 1024


def _observation_digest(observation: StudyCodeObservationV1) -> str:
    return domain_sha256(
        _OBSERVATION_DOMAIN, canonical_json_bytes(observation.model_dump(mode="json"))
    )


def _validated(evaluation: IrisStudyCodeEvaluationV1) -> IrisStudyCodeEvaluationV1:
    if type(evaluation) is not IrisStudyCodeEvaluationV1:
        raise TypeError("evaluation must be exactly IrisStudyCodeEvaluationV1")
    owned = IrisStudyCodeEvaluationV1.model_validate(evaluation.model_dump(mode="python"))
    observation = owned.payload.observation
    if (
        owned.payload_sha256
        != domain_sha256(
            _PAYLOAD_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
        )
        or owned.payload.receipt.observation_sha256 != _observation_digest(observation)
        or observation.policy_sha256
        != domain_sha256(
            "iris/python-sandbox-policy/v1",
            canonical_json_bytes(observation.policy.model_dump(mode="json")),
        )
    ):
        raise ValueError("code evaluation digest bindings disagree")
    if len(canonical_json_file_bytes(owned.model_dump(mode="json"))) > _MAX_CODE_EVALUATION_BYTES:
        raise ValueError("code evaluation exceeds its safety limit")
    return owned


def validate_study_code_evaluation_v1(
    evaluation: IrisStudyCodeEvaluationV1,
) -> IrisStudyCodeEvaluationV1:
    """Check only schema, bounds and hashes; do not claim dependencies or execution truth."""
    try:
        return _validated(evaluation)
    except Exception:
        raise IrisStudyEvaluationError("Iris code evaluation integrity check failed") from None


def _stream(raw: bytes) -> StudyCodeStreamV1:
    return StudyCodeStreamV1(
        data_base64=base64.b64encode(raw).decode("ascii"),
        byte_length=len(raw),
        sha256=sha256(raw).hexdigest(),
    )


def _record(observation: PythonSandboxObservationV1) -> StudyCodeObservationV1:
    metadata = asdict(observation)
    metadata.pop("process")
    metadata["policy"] = observation.policy.model_dump(mode="python")
    return StudyCodeObservationV1.model_validate(
        {
            **metadata,
            "image_reference": IMAGE,
            "stdout": _stream(observation.process.stdout),
            "stderr": _stream(observation.process.stderr),
            "transport_returncode": observation.process.returncode,
            "transport_termination": observation.process.termination,
        }
    )


def _restore(observation: StudyCodeObservationV1) -> PythonSandboxObservationV1:
    return PythonSandboxObservationV1(
        policy=observation.policy,
        policy_sha256=observation.policy_sha256,
        worker_sha256=observation.worker_sha256,
        image_id=observation.image_id,
        runtime_version=observation.runtime_version,
        docker_version=observation.docker_version,
        kernel_release=observation.kernel_release,
        submitted_input_sha256=observation.submitted_input_sha256,
        process=ProcessResult(
            returncode=observation.transport_returncode,
            stdout=base64.b64decode(observation.stdout.data_base64, validate=True),
            stderr=base64.b64decode(observation.stderr.data_base64, validate=True),
            termination=observation.transport_termination,
        ),
        container_exit_code=observation.container_exit_code,
        oom_killed=observation.oom_killed,
    )


def _evaluate(
    benchmark: IrisStudyBenchmarkV1,
    *,
    request_path: Path,
    example_id: str,
    output_text: str,
    terminal_token_id: int,
    policy: PythonSandboxPolicyV1,
    observed: PythonSandboxObservationV1 | None,
) -> IrisStudyCodeEvaluationV1:
    if type(example_id) is not str or type(output_text) is not str:
        raise TypeError("example ID and output text must be exact strings")
    if len(output_text) > 64 * 1024 * 1024:
        raise ValueError("answer exceeds the verification safety limit")
    encoded_output = output_text.encode("utf-8", errors="strict")
    if len(encoded_output) > 64 * 1024 * 1024:
        raise ValueError("answer exceeds the verification safety limit")
    if type(policy) is not PythonSandboxPolicyV1:
        raise TypeError("code evaluation requires an explicit exact policy")
    owned_policy = PythonSandboxPolicyV1.model_validate(policy.model_dump(mode="python"))
    study = validate_study_benchmark_v1(benchmark)
    dataset = study.payload.dataset.payload
    index = next(
        i
        for i, item in enumerate(dataset.manifest.payload.examples)
        if item.example_id == example_id
    )
    example = dataset.manifest.payload.examples[index]
    rule = study.payload.scoring[index].rule
    if type(rule) is not StudyPythonIntegerListRuleV1:
        raise ValueError("selected task must have a functional Python rule")

    verified_request = verify_request_artifact_v4(request_path)
    request = read_request_artifact_v4(request_path)
    if request.payload_sha256 != verified_request.payload_sha256:
        raise ValueError("request changed after dependency verification")
    payload = request.payload
    if dataset.protocol_declaration_payload_sha256 != payload.protocol_declaration_payload_sha256:
        raise ValueError("benchmark and request protocol identities disagree")
    prompt_bytes = example.prompt.encode("utf-8", errors="strict")
    if (
        payload.request.prompt.sha256
        != domain_sha256("iris/request-artifact/v3/prompt", prompt_bytes)
        or payload.request.prompt.byte_length != len(prompt_bytes)
        or payload.generation.text.sha256
        != domain_sha256("iris/request-artifact/v3/generated-text", encoded_output)
        or payload.generation.text.byte_length != len(encoded_output)
    ):
        raise ValueError("private prompt/source does not match request provenance")
    preflight = preflight_study_inputs_v1(
        study, config=payload.configuration.value, terminal_token_id=terminal_token_id
    )
    if (
        not preflight.inputs_fit
        or preflight.configuration_sha256 != payload.configuration.sha256
        or preflight.execution_profile_set_sha256 != payload.execution_profile_set_sha256
        or preflight.transformers_version != payload.runtime.software.transformers
        or preflight.tokenizers_version != payload.runtime.software.tokenizers
    ):
        raise ValueError("request software/configuration or complete input budgets disagree")
    tokens = preflight.examples[index]
    provenance = payload.request
    if (
        tokens.example_id != example_id
        or provenance.physical_prompt_tokens != tokens.prompt_tokens
        or provenance.attended_prompt_tokens != tokens.prompt_tokens
        or provenance.prompt_input_ids.sha256 != tokens.request_v4_prompt_input_ids_sha256
        or provenance.prompt_attention_mask.sha256 != tokens.request_v4_prompt_attention_mask_sha256
    ):
        raise ValueError("request prompt tokens are not the full benchmark input")

    # Reading versioned worker DATA does not import/compile/execute candidate code.
    worker_digest = sha256(_worker()).hexdigest()
    score, native = _score_study_python_answer_v1(
        study,
        example_id=example_id,
        output_text=output_text,
        policy=owned_policy,
        observed=observed,
    )
    if native.worker_sha256 != worker_digest or sha256(_worker()).hexdigest() != worker_digest:
        raise ValueError("recorded/current versioned worker identities disagree")
    final_verification = verify_request_artifact_v4(request_path)
    if final_verification != verified_request or read_request_artifact_v4(request_path) != request:
        raise ValueError("request/checkpoint evidence changed during evaluation")
    observation = _record(native)
    receipt = StudyCodeReceiptV1.model_validate(
        {
            "benchmark_payload_sha256": study.payload_sha256,
            "dataset_payload_sha256": study.payload.dataset.payload_sha256,
            "manifest_payload_sha256": dataset.manifest.payload_sha256,
            "protocol_declaration_payload_sha256": dataset.protocol_declaration_payload_sha256,
            "example_id": example_id,
            "task_family": score.answer.task_family,
            "split": score.answer.split,
            "request_payload_sha256": request.payload_sha256,
            "configuration_sha256": payload.configuration.sha256,
            "execution_profile_set_sha256": payload.execution_profile_set_sha256,
            "arm_id": payload.topology.arm_id,
            "checkpoint_transfer": payload.checkpoint_transfer,
            "input_binding": StudyEvaluationInputBindingV1(
                preflight_sha256=domain_sha256(
                    "iris/study-evaluation/v1/input-preflight",
                    canonical_json_bytes(asdict(preflight)),
                ),
                tokenizer_semantics_sha256=preflight.tokenizer_semantics_sha256,
                transformers_version=preflight.transformers_version,
                tokenizers_version=preflight.tokenizers_version,
                terminal_token_id=preflight.terminal_token_id,
                corpus_examples=len(preflight.examples),
                prompt_tokens=tokens.prompt_tokens,
                target_tokens_including_terminal=tokens.target_tokens_including_terminal,
                prompt_token_ids_sha256=tokens.prompt_token_ids_sha256,
                target_token_ids_sha256=tokens.target_token_ids_sha256,
                request_prompt_input_ids_sha256=tokens.request_v4_prompt_input_ids_sha256,
                request_prompt_attention_mask_sha256=tokens.request_v4_prompt_attention_mask_sha256,
                scope="all_benchmark_inputs_fit_and_selected_request_prompt_matches",
            ),
            "scoring_rule_sha256": domain_sha256(
                _RULE_DOMAIN, canonical_json_bytes(rule.model_dump(mode="json"))
            ),
            "observation_sha256": _observation_digest(observation),
            "output_sha256": score.answer.output_sha256,
            "output_byte_length": score.answer.output_byte_length,
            "source_submitted": score.source_submitted,
            "case_count": score.case_count,
            "passed_case_count": score.passed_case_count,
            "outcome": score.answer.outcome,
            "reason": score.reason,
            "verification_scope": (
                "bindings_and_recomputed_recorded_observations_not_execution_attestation"
            ),
            "scientific_launch_status": "not_assessed",
        }
    )
    result = StudyCodeEvaluationPayloadV1(receipt=receipt, observation=observation)
    return _validated(
        IrisStudyCodeEvaluationV1(
            schema_identity=StudyCodeEvaluationSchemaV1(
                artifact_type="iris.study_code_evaluation",
                version=1,
                canonicalization="iris-json-v1",
                hash_algorithm="sha256",
                content_policy="private_worker_observations_and_hash_only_receipt",
            ),
            payload_sha256=domain_sha256(
                _PAYLOAD_DOMAIN, canonical_json_bytes(result.model_dump(mode="json"))
            ),
            payload=result,
        )
    )


def evaluate_study_request_python_answer_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    request_path: Path,
    example_id: str,
    output_text: str,
    terminal_token_id: int,
    policy: PythonSandboxPolicyV1,
) -> IrisStudyCodeEvaluationV1:
    """Verify request/checkpoint and full inputs, execute once in gVisor, capture privately.

    No receipt is produced for unavailable/unverified runtime or failed cleanup.
    Returned observations can contain candidate-generated private data; do not log them.
    """
    try:
        return _evaluate(
            benchmark,
            request_path=request_path,
            example_id=example_id,
            output_text=output_text,
            terminal_token_id=terminal_token_id,
            policy=policy,
            observed=None,
        )
    except IrisCodeEvaluationUnavailableError:
        raise
    except Exception:
        raise IrisStudyEvaluationError(
            "Iris request-bound Python answer could not be evaluated"
        ) from None


def verify_study_code_evaluation_v1(
    evaluation: IrisStudyCodeEvaluationV1,
    *,
    benchmark: IrisStudyBenchmarkV1,
    request_path: Path,
    output_text: str,
) -> IrisStudyCodeEvaluationV1:
    """Recheck dependencies and grade recorded data, without Docker, execution or model weights.

    This proves consistency only, NOT that recorded observations ever occurred.
    Rerunning nondeterministic Python cannot authenticate a historical result.
    """
    try:
        owned = _validated(evaluation)
        receipt = owned.payload.receipt
        expected = _evaluate(
            benchmark,
            request_path=request_path,
            example_id=receipt.example_id,
            output_text=output_text,
            terminal_token_id=receipt.input_binding.terminal_token_id,
            policy=owned.payload.observation.policy,
            observed=_restore(owned.payload.observation),
        )
        if owned != expected:
            raise ValueError("recorded evaluation does not match recomputed evidence")
        return expected
    except Exception:
        raise IrisStudyEvaluationError(
            "Iris code evaluation could not be independently verified"
        ) from None


__all__ = [
    "evaluate_study_request_python_answer_v1",
    "validate_study_code_evaluation_v1",
    "verify_study_code_evaluation_v1",
]
