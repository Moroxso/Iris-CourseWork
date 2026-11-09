"""Offline, tokenizer-only full-corpus budget checks for the four-arm study.

This diagnostic is not a persisted launch approval. Existing request commands
retain their behavior; a scientific launcher must consume/recheck these facts.
"""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Literal

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from iris.application.arm_profiles import arm_execution_profile_set_sha256_v4
from iris.application.errors import IrisStudyInputPreflightError
from iris.application.study_benchmark import validate_study_benchmark_v1
from iris.application.study_declaration import build_four_arm_protocol_declaration_v1
from iris.artifacts.experiment_v1 import ExperimentSplitV1
from iris.artifacts.study_benchmark_v1 import IrisStudyBenchmarkV1
from iris.artifacts.study_v1 import StudyTaskFamilyV1
from iris.integrations.tokenizer_semantics import (
    encode_raw_token_ids,
    tokenizer_semantics_sha256,
    tokenizer_special_ids,
)
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256

# Preserve historical private patch points while sharing exactly the same helpers.
_encode = encode_raw_token_ids
_tokenizer_fingerprint = tokenizer_semantics_sha256
_special_ids = tokenizer_special_ids

type StudyInputViolationV1 = Literal[
    "empty_prompt_encoding",
    "prompt_over_limit",
    "empty_target_encoding",
    "target_over_limit",
    "target_contains_special_tokens",
]


@dataclass(frozen=True, slots=True)
class IrisStudyExampleTokenBudgetV1:
    """Private membership and token identities, without text or token IDs."""

    example_id: str
    task_family: StudyTaskFamilyV1
    split: ExperimentSplitV1
    prompt_tokens: int
    target_tokens_including_terminal: int
    prompt_token_ids_sha256: str
    target_token_ids_sha256: str
    # Request v4 inherits v3's typed-tensor domains, not the JSON-list domain above.
    request_v4_prompt_input_ids_sha256: str
    request_v4_prompt_attention_mask_sha256: str
    violations: tuple[StudyInputViolationV1, ...]


@dataclass(frozen=True, slots=True)
class IrisStudyInputPreflightV1:
    """In-memory diagnostic, not an authenticated receipt or launch authorization."""

    benchmark_payload_sha256: str
    configuration_sha256: str
    execution_profile_set_sha256: str
    tokenizer_semantics_sha256: str
    transformers_version: str
    tokenizers_version: str
    terminal_token_id: int
    max_prompt_tokens: int
    max_new_tokens: int
    max_position_embeddings: int
    # Exact current prefix cardinalities: C's growing buffer is read into M tokens.
    arm_latent_tokens: tuple[int, int, int, int]
    configured_context_envelope_fits: bool
    examples: tuple[IrisStudyExampleTokenBudgetV1, ...]
    scientific_launch_status: Literal["not_assessed"] = "not_assessed"

    @property
    def inputs_fit(self) -> bool:
        return self.configured_context_envelope_fits and not any(
            example.violations for example in self.examples
        )


def _positive_integer(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("invalid positive model metadata")
    return value


def _own_study_config(config: IrisBootstrapConfig) -> IrisBootstrapConfig:
    if type(config) is not IrisBootstrapConfig:
        raise TypeError("config must be exactly IrisBootstrapConfig")
    owned = IrisBootstrapConfig.model_validate(config.model_dump(mode="python"))
    if not owned.backbone.local_files_only or not owned.backbone.tokenizer.use_fast:
        raise ValueError("preflight requires an offline fast tokenizer")
    constants = build_four_arm_protocol_declaration_v1().payload.architecture
    observed = (
        owned.cognitive.num_modules,
        owned.cognitive.state_dim,
        owned.workspace.num_slots,
        owned.workspace.cycles,
        owned.workspace.winners_per_cycle,
    )
    expected = (
        constants.cognitive_module_count,
        constants.cognitive_state_width,
        constants.limited_workspace_slot_count,
        constants.cognitive_cycle_count,
        constants.limited_workspace_winners_per_cycle,
    )
    if observed != expected:
        raise ValueError("study dimensions differ from the recovered protocol")
    return owned


def preflight_study_inputs_v1(
    benchmark: IrisStudyBenchmarkV1,
    *,
    config: IrisBootstrapConfig,
    terminal_token_id: int,
) -> IrisStudyInputPreflightV1:
    """Inspect every split without dropping examples, loading weights, or executing code.

    Load only the pinned tokenizer and model configuration from the local cache.
    Target length includes the explicitly selected EOS, matching teacher preparation.
    Limits are never changed. A corpus/configuration budget failure returns a
    complete diagnostic with inputs_fit=False; invalid infrastructure raises.
    """
    try:
        if type(terminal_token_id) is not int:
            raise TypeError("terminal token ID must be an exact integer")
        owned = _own_study_config(config)
        study = validate_study_benchmark_v1(benchmark)
        configuration_bytes = canonical_json_bytes(owned.model_dump(mode="json"))
        backbone = owned.backbone
        model_config: Any = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=backbone.repository,
            revision=backbone.revision,
            trust_remote_code=False,
            local_files_only=True,
        )
        vocabulary_size = _positive_integer(model_config.vocab_size)
        max_positions = _positive_integer(model_config.max_position_embeddings)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=backbone.repository,
            revision=backbone.tokenizer_revision,
            trust_remote_code=False,
            local_files_only=True,
            use_fast=True,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("loader did not return a fast tokenizer")
        tokenizer.padding_side = backbone.tokenizer.padding_side
        tokenizer.truncation_side = backbone.tokenizer.truncation_side
        special_ids = _special_ids(tokenizer, vocabulary_size)
        eos = tokenizer.eos_token_id
        eos_ids = (eos,) if type(eos) is int else tuple(eos) if type(eos) in {list, tuple} else ()
        if (
            not eos_ids
            or len(set(eos_ids)) != len(eos_ids)
            or any(type(value) is not int or value not in special_ids for value in eos_ids)
            or terminal_token_id not in eos_ids
        ):
            raise ValueError("terminal token must be a valid tokenizer EOS")
        fingerprint = _tokenizer_fingerprint(tokenizer)
        observations: list[IrisStudyExampleTokenBudgetV1] = []
        dataset = study.payload.dataset.payload
        for example, assignment in zip(
            dataset.manifest.payload.examples, dataset.assignments, strict=True
        ):
            prompt_ids = _encode(
                tokenizer, example.prompt, target=False, vocabulary_size=vocabulary_size
            )
            content_ids = _encode(
                tokenizer, example.target, target=True, vocabulary_size=vocabulary_size
            )
            target_ids = (*content_ids, terminal_token_id)
            violations: list[StudyInputViolationV1] = []
            if not prompt_ids:
                violations.append("empty_prompt_encoding")
            if len(prompt_ids) > backbone.limits.max_prompt_tokens:
                violations.append("prompt_over_limit")
            if not content_ids:
                violations.append("empty_target_encoding")
            if len(target_ids) > backbone.limits.max_new_tokens:
                violations.append("target_over_limit")
            if any(value in special_ids for value in content_ids):
                violations.append("target_contains_special_tokens")
            observations.append(
                IrisStudyExampleTokenBudgetV1(
                    example_id=example.example_id,
                    task_family=assignment.task_family,
                    split=example.split,
                    prompt_tokens=len(prompt_ids),
                    target_tokens_including_terminal=len(target_ids),
                    prompt_token_ids_sha256=domain_sha256(
                        "iris/study-input-token-ids/v1", canonical_json_bytes(prompt_ids)
                    ),
                    target_token_ids_sha256=domain_sha256(
                        "iris/study-input-token-ids/v1", canonical_json_bytes(target_ids)
                    ),
                    request_v4_prompt_input_ids_sha256=tensor_sha256(
                        torch.tensor((prompt_ids,), dtype=torch.int64, device="cpu"),
                        domain="iris/request-artifact/v3/prompt-input-ids",
                    ),
                    request_v4_prompt_attention_mask_sha256=tensor_sha256(
                        torch.ones((1, len(prompt_ids)), dtype=torch.int64, device="cpu"),
                        domain="iris/request-artifact/v3/prompt-attention-mask",
                    ),
                    violations=tuple(violations),
                )
            )
        if _tokenizer_fingerprint(tokenizer) != fingerprint:
            raise ValueError("tokenizer semantics changed during preflight")
        if (
            _positive_integer(model_config.vocab_size) != vocabulary_size
            or _positive_integer(model_config.max_position_embeddings) != max_positions
            or _special_ids(tokenizer, vocabulary_size) != special_ids
            or canonical_json_bytes(owned.model_dump(mode="json")) != configuration_bytes
        ):
            raise ValueError("configuration changed during preflight")
        latent_tokens = (
            0,
            owned.cognitive.num_modules,
            owned.cognitive.num_modules,
            owned.workspace.num_slots,
        )
        return IrisStudyInputPreflightV1(
            benchmark_payload_sha256=study.payload_sha256,
            configuration_sha256=domain_sha256("iris/configuration/v1", configuration_bytes),
            execution_profile_set_sha256=arm_execution_profile_set_sha256_v4(),
            tokenizer_semantics_sha256=fingerprint,
            transformers_version=version("transformers"),
            tokenizers_version=version("tokenizers"),
            terminal_token_id=terminal_token_id,
            max_prompt_tokens=backbone.limits.max_prompt_tokens,
            max_new_tokens=backbone.limits.max_new_tokens,
            max_position_embeddings=max_positions,
            arm_latent_tokens=latent_tokens,
            configured_context_envelope_fits=(
                backbone.limits.max_prompt_tokens
                + max(latent_tokens)
                + backbone.limits.max_new_tokens
                <= max_positions
            ),
            examples=tuple(observations),
        )
    except Exception:
        raise IrisStudyInputPreflightError(
            "Iris study input preflight could not complete"
        ) from None


__all__ = [
    "IrisStudyExampleTokenBudgetV1",
    "IrisStudyInputPreflightV1",
    "StudyInputViolationV1",
    "preflight_study_inputs_v1",
]
