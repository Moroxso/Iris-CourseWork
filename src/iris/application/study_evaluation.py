"""Bind exact-text scores to verified request-v4/checkpoint and full input identities."""

from dataclasses import asdict
from pathlib import Path

from iris.application.errors import IrisCodeEvaluationUnavailableError, IrisStudyEvaluationError
from iris.application.request_artifact_v4 import read_request_artifact_v4
from iris.application.study_benchmark import score_study_answer_v1, validate_study_benchmark_v1
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.application.verification_v4 import verify_request_artifact_v4
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1, StudyExactTextRuleV1
from iris.artifacts.study_evaluation_v1 import (
    IrisStudyEvaluationV1,
    StudyEvaluationInputBindingV1,
    StudyEvaluationPayloadV1,
    StudyEvaluationSchemaV1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_PAYLOAD_DOMAIN = "iris/study-evaluation/v1/payload"
_MAX_EVALUATION_BYTES = 1024 * 1024


def _validated(evaluation: IrisStudyEvaluationV1) -> IrisStudyEvaluationV1:
    if type(evaluation) is not IrisStudyEvaluationV1:
        raise TypeError("evaluation must be exactly IrisStudyEvaluationV1")
    owned = IrisStudyEvaluationV1.model_validate(evaluation.model_dump(mode="python"))
    if owned.payload_sha256 != domain_sha256(
        _PAYLOAD_DOMAIN, canonical_json_bytes(owned.payload.model_dump(mode="json"))
    ):
        raise ValueError("evaluation payload digest is inconsistent")
    if len(canonical_json_file_bytes(owned.model_dump(mode="json"))) > _MAX_EVALUATION_BYTES:
        raise ValueError("evaluation exceeds its safety limit")
    return owned


def validate_study_evaluation_v1(evaluation: IrisStudyEvaluationV1) -> IrisStudyEvaluationV1:
    """Check schema/hash only; use verify_study_evaluation_v1 to recheck evidence and score."""
    try:
        return _validated(evaluation)
    except Exception:
        raise IrisStudyEvaluationError("Iris study evaluation integrity check failed") from None


def evaluate_study_request_answer_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    request_path: Path,
    example_id: str,
    output_text: str,
    terminal_token_id: int,
) -> IrisStudyEvaluationV1:
    """Verify persisted evidence, complete input budgets, exact prompt/output, then score.

    Requires the private output explicitly because request artifacts store only
    its fingerprint. Does not execute any candidate code or regenerate an answer.
    A/B/C/D requests are supported; functional code scoring remains unavailable.
    """
    try:
        if type(example_id) is not str or type(output_text) is not str:
            raise TypeError("example ID and output text must be exact strings")
        encoded_output = output_text.encode("utf-8", errors="strict")
        if len(encoded_output) > 64 * 1024 * 1024:
            raise ValueError("answer exceeds the verification safety limit")
        study = validate_study_benchmark_v1(benchmark)
        dataset = study.payload.dataset.payload
        index = next(
            index
            for index, example in enumerate(dataset.manifest.payload.examples)
            if example.example_id == example_id
        )
        example = dataset.manifest.payload.examples[index]
        rule = study.payload.scoring[index].rule
        if not isinstance(rule, StudyExactTextRuleV1):
            raise IrisCodeEvaluationUnavailableError(
                "Functional Python scoring requires an explicitly isolated backend"
            )

        verified_request = verify_request_artifact_v4(request_path)
        request = read_request_artifact_v4(request_path)
        if request.payload_sha256 != verified_request.payload_sha256:
            raise ValueError("request changed after dependency verification")
        payload = request.payload
        if (
            dataset.protocol_declaration_payload_sha256
            != payload.protocol_declaration_payload_sha256
        ):
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
            raise ValueError("private prompt/output does not match request provenance")

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
        observation = preflight.examples[index]
        provenance = payload.request
        if (
            observation.example_id != example_id
            or provenance.physical_prompt_tokens != observation.prompt_tokens
            or provenance.attended_prompt_tokens != observation.prompt_tokens
            or provenance.prompt_input_ids.sha256 != observation.request_v4_prompt_input_ids_sha256
            or provenance.prompt_attention_mask.sha256
            != observation.request_v4_prompt_attention_mask_sha256
        ):
            raise ValueError("request prompt tokens are not the full benchmark input")

        score = score_study_answer_v1(study, example_id=example_id, output_text=output_text)
        final_verification = verify_request_artifact_v4(request_path)
        if (
            final_verification != verified_request
            or read_request_artifact_v4(request_path) != request
        ):
            raise ValueError("request/checkpoint evidence changed during evaluation")

        result = StudyEvaluationPayloadV1(
            benchmark_payload_sha256=study.payload_sha256,
            dataset_payload_sha256=study.payload.dataset.payload_sha256,
            manifest_payload_sha256=dataset.manifest.payload_sha256,
            protocol_declaration_payload_sha256=dataset.protocol_declaration_payload_sha256,
            example_id=example_id,
            task_family=score.task_family,
            split=score.split,
            request_payload_sha256=request.payload_sha256,
            configuration_sha256=payload.configuration.sha256,
            execution_profile_set_sha256=payload.execution_profile_set_sha256,
            arm_id=payload.topology.arm_id,
            checkpoint_transfer=payload.checkpoint_transfer,
            input_binding=StudyEvaluationInputBindingV1(
                preflight_sha256=domain_sha256(
                    "iris/study-evaluation/v1/input-preflight",
                    canonical_json_bytes(asdict(preflight)),
                ),
                tokenizer_semantics_sha256=preflight.tokenizer_semantics_sha256,
                transformers_version=preflight.transformers_version,
                tokenizers_version=preflight.tokenizers_version,
                terminal_token_id=preflight.terminal_token_id,
                corpus_examples=len(preflight.examples),
                prompt_tokens=observation.prompt_tokens,
                target_tokens_including_terminal=observation.target_tokens_including_terminal,
                prompt_token_ids_sha256=observation.prompt_token_ids_sha256,
                target_token_ids_sha256=observation.target_token_ids_sha256,
                request_prompt_input_ids_sha256=observation.request_v4_prompt_input_ids_sha256,
                request_prompt_attention_mask_sha256=observation.request_v4_prompt_attention_mask_sha256,
                scope="all_benchmark_inputs_fit_and_selected_request_prompt_matches",
            ),
            scoring_rule=rule,
            output_sha256=score.output_sha256,
            output_byte_length=score.output_byte_length,
            outcome=score.outcome,
            verification_scope="bindings_and_recomputed_text_score_not_execution_attestation",
            scientific_launch_status="not_assessed",
        )
        return _validated(
            IrisStudyEvaluationV1(
                schema_identity=StudyEvaluationSchemaV1(
                    artifact_type="iris.study_evaluation",
                    version=1,
                    canonicalization="iris-json-v1",
                    hash_algorithm="sha256",
                    content_policy="private_metadata_and_hashes_only",
                ),
                payload_sha256=domain_sha256(
                    _PAYLOAD_DOMAIN, canonical_json_bytes(result.model_dump(mode="json"))
                ),
                payload=result,
            )
        )
    except IrisCodeEvaluationUnavailableError:
        raise
    except Exception:
        raise IrisStudyEvaluationError(
            "Iris request-bound study answer could not be evaluated"
        ) from None


def verify_study_evaluation_v1(
    evaluation: IrisStudyEvaluationV1,
    *,
    benchmark: IrisStudyBenchmarkV1,
    request_path: Path,
    output_text: str,
) -> IrisStudyEvaluationV1:
    """Recheck all dependencies and independently recompute the score from private output."""
    try:
        owned = _validated(evaluation)
        expected = evaluate_study_request_answer_v1(
            benchmark,
            request_path=request_path,
            example_id=owned.payload.example_id,
            output_text=output_text,
            terminal_token_id=owned.payload.input_binding.terminal_token_id,
        )
        if owned != expected:
            raise ValueError("recorded evaluation does not match recomputed evidence")
        return expected
    except Exception:
        raise IrisStudyEvaluationError(
            "Iris study evaluation could not be independently verified"
        ) from None


__all__ = [
    "evaluate_study_request_answer_v1",
    "validate_study_evaluation_v1",
    "verify_study_evaluation_v1",
]
