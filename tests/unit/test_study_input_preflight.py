"""Real local tokenizer fixtures, without model weights or candidate execution."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from tokenizers.processors import TemplateProcessing  # type: ignore[import-untyped]
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from iris.application import study_input_preflight as service
from iris.application.errors import IrisStudyInputPreflightError
from iris.application.study_input_preflight import preflight_study_inputs_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256, tensor_sha256


def _config() -> IrisBootstrapConfig:
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        composed = compose(
            config_name="config",
            overrides=["backbone.local_files_only=true", "backbone.limits.max_new_tokens=128"],
        )
    return IrisBootstrapConfig.model_validate(OmegaConf.to_container(composed, resolve=True))


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "<eos>": 1, "<bos>": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend, eos_token="<eos>", bos_token="<bos>"
    )


@contextmanager
def _assets(tokenizer: PreTrainedTokenizerFast, *, context: int = 4096) -> Iterator[MagicMock]:
    with (
        patch.object(
            AutoConfig,
            "from_pretrained",
            return_value=SimpleNamespace(vocab_size=8, max_position_embeddings=context),
        ),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer) as loader,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model_loader,
    ):
        yield loader
    model_loader.assert_not_called()


def test_complete_ordered_corpus_and_owned_token_digests_match_runtime_encoding() -> None:
    config, tokenizer = _config(), _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    with _assets(tokenizer) as loader:
        report = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
    assert report.inputs_fit
    assert report.arm_latent_tokens == (0, 8, 8, 8)
    assert report.scientific_launch_status == "not_assessed"
    assert report.configuration_sha256 == domain_sha256(
        "iris/configuration/v1", canonical_json_bytes(config.model_dump(mode="json"))
    )
    assert report.benchmark_payload_sha256 == benchmark.payload_sha256
    loader.assert_called_once_with(
        pretrained_model_name_or_path=config.backbone.repository,
        revision=config.backbone.tokenizer_revision,
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    for example, observation in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples, report.examples, strict=True
    ):
        assert (observation.example_id, observation.split) == (example.example_id, example.split)
        # Exact current BackboneAdapter prompt kwargs, using the real fixture tokenizer.
        runtime = tokenizer(
            example.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.backbone.limits.max_prompt_tokens,
            return_attention_mask=True,
        )
        prompt_ids = runtime["input_ids"].tolist()[0]
        assert observation.request_v4_prompt_input_ids_sha256 == tensor_sha256(
            runtime["input_ids"], domain="iris/request-artifact/v3/prompt-input-ids"
        )
        assert observation.request_v4_prompt_attention_mask_sha256 == tensor_sha256(
            runtime["attention_mask"], domain="iris/request-artifact/v3/prompt-attention-mask"
        )
        target_ids = tokenizer(
            example.target, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"] + [1]
        assert observation.prompt_tokens == len(prompt_ids)
        assert observation.target_tokens_including_terminal == len(target_ids)
        assert observation.prompt_token_ids_sha256 == domain_sha256(
            "iris/study-input-token-ids/v1", canonical_json_bytes(prompt_ids)
        )
        assert observation.target_token_ids_sha256 == domain_sha256(
            "iris/study-input-token-ids/v1", canonical_json_bytes(target_ids)
        )
    with pytest.raises(FrozenInstanceError):
        cast(Any, report).max_prompt_tokens = 1


def test_overflow_is_not_truncated_dropped_or_fixed_and_every_split_is_checked() -> None:
    config, tokenizer = _config(), _tokenizer()
    config.backbone.limits.max_prompt_tokens = 1
    config.backbone.limits.max_new_tokens = 1
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    with _assets(tokenizer):
        report = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
    assert not report.inputs_fit
    assert len(report.examples) == 15
    assert {item.split for item in report.examples} == {"train", "validation", "test"}
    assert all(item.prompt_tokens > 1 for item in report.examples)
    assert all(item.target_tokens_including_terminal > 1 for item in report.examples)
    assert all(
        item.violations == ("prompt_over_limit", "target_over_limit") for item in report.examples
    )
    assert config.backbone.limits.max_prompt_tokens == report.max_prompt_tokens == 1
    assert config.backbone.limits.max_new_tokens == report.max_new_tokens == 1


def test_exact_prompt_and_target_limits_include_terminal_and_are_inclusive() -> None:
    config, tokenizer = _config(), _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    with _assets(tokenizer):
        reference = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
        config.backbone.limits.max_prompt_tokens = max(
            item.prompt_tokens for item in reference.examples
        )
        config.backbone.limits.max_new_tokens = max(
            item.target_tokens_including_terminal for item in reference.examples
        )
        exact = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
        config.backbone.limits.max_new_tokens -= 1
        overflow = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
    assert exact.inputs_fit
    assert not overflow.inputs_fit
    assert any("target_over_limit" in item.violations for item in overflow.examples)


@pytest.mark.parametrize("delta", [0, -1])
def test_full_configured_context_reserves_latent_tokens_for_all_arms(delta: int) -> None:
    config, tokenizer = _config(), _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    envelope = config.backbone.limits.max_prompt_tokens + config.backbone.limits.max_new_tokens + 8
    with _assets(tokenizer, context=envelope + delta):
        report = preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
    assert report.configured_context_envelope_fits is (delta == 0)
    assert report.inputs_fit is (delta == 0)
    assert not any(item.violations for item in report.examples)


@pytest.mark.parametrize(
    "kind", ["online", "slow_tokenizer", "dimensions", "revision", "benchmark"]
)
def test_invalid_declared_inputs_fail_before_any_asset_loading(kind: str) -> None:
    config = _config()
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    if kind == "online":
        config.backbone.local_files_only = False
    elif kind == "slow_tokenizer":
        config.backbone.tokenizer.use_fast = False
    elif kind == "dimensions":
        config.workspace.num_slots = 9
    elif kind == "revision":
        config.backbone.revision = "main"
    else:
        benchmark = benchmark.model_copy(update={"payload_sha256": "0" * 64})
    with (
        patch.object(AutoConfig, "from_pretrained") as loader,
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_inputs_v1(benchmark, config=config, terminal_token_id=1)
    loader.assert_not_called()


@pytest.mark.parametrize("terminal", [True, -1, 2, 8])
def test_terminal_must_be_an_exact_tokenizer_eos(terminal: int) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    with _assets(_tokenizer()), pytest.raises(IrisStudyInputPreflightError):
        preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=terminal)


@pytest.mark.parametrize(
    "output",
    [
        {"input_ids": [[0]], "attention_mask": [[1]]},
        {"input_ids": [True], "attention_mask": [1]},
        {"input_ids": [8], "attention_mask": [1]},
        {"input_ids": [0], "attention_mask": [0]},
        {"input_ids": [0], "attention_mask": [True]},
        {"input_ids": [0], "attention_mask": []},
        {"input_ids": [0]},
    ],
)
def test_invalid_encoding_is_infrastructure_failure_not_a_pass(output: dict[str, Any]) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    with (
        _assets(_tokenizer()),
        patch.object(PreTrainedTokenizerFast, "__call__", return_value=output),
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=1)


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([], ("empty_prompt_encoding", "empty_target_encoding")),
        ([1], ("target_contains_special_tokens",)),
    ],
)
def test_empty_encoding_and_target_special_tokens_are_explicit_corpus_failures(
    ids: list[int], expected: tuple[str, ...]
) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    output = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    with (
        _assets(_tokenizer()),
        patch.object(PreTrainedTokenizerFast, "__call__", return_value=output),
    ):
        report = preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=1)
    assert not report.inputs_fit
    assert all(item.violations == expected for item in report.examples)


def test_tokenizer_mutation_during_encoding_is_rejected() -> None:
    tokenizer = _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    original = service._encode

    def mutate(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        result = original(*args, **kwargs)
        tokenizer.add_tokens(["CHANGED"])
        return result

    with (
        _assets(tokenizer),
        patch.object(service, "_encode", side_effect=mutate),
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=1)


def test_load_failure_does_not_disclose_private_content_or_download_fallback() -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    with (
        patch.object(AutoConfig, "from_pretrained", side_effect=OSError("PRIVATE")) as loader,
        pytest.raises(IrisStudyInputPreflightError) as caught,
    ):
        preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=1)
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert loader.call_count == 1
    assert loader.call_args.kwargs["local_files_only"] is True


def test_special_token_splitting_changes_encoding_and_semantics_identity() -> None:
    tokenizer = _tokenizer()
    fingerprint = service._tokenizer_fingerprint(tokenizer)
    preserved = service._encode(tokenizer, "<eos>", target=True, vocabulary_size=8)
    tokenizer.split_special_tokens = True
    assert service._tokenizer_fingerprint(tokenizer) != fingerprint
    split = service._encode(tokenizer, "<eos>", target=True, vocabulary_size=8)
    assert preserved == (1,)
    assert split != preserved and 1 not in split


@pytest.mark.parametrize("mutation_call", [1, 30])
def test_special_token_policy_mutation_is_detected_even_after_last_encoding(
    mutation_call: int,
) -> None:
    tokenizer = _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    original = service._encode
    calls = 0

    def mutate(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == mutation_call:
            tokenizer.split_special_tokens = True
        return result

    with (
        _assets(tokenizer),
        patch.object(service, "_encode", side_effect=mutate),
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=1)
    assert calls == 30


def test_special_token_policy_rejects_nonboolean_metadata() -> None:
    tokenizer = _tokenizer()
    tokenizer.split_special_tokens = 0
    with pytest.raises(ValueError, match="exact boolean"):
        service._tokenizer_fingerprint(tokenizer)


@pytest.mark.parametrize("field", ["eos", "padding", "truncation", "vocabulary", "positions"])
def test_other_asset_metadata_changes_after_last_encoding_are_rejected(field: str) -> None:
    tokenizer = _tokenizer()
    benchmark = build_synthetic_study_pilot_v1(seed=1, problems_per_cell=1)
    metadata = SimpleNamespace(vocab_size=8, max_position_embeddings=4096)
    original = service._encode
    calls = 0

    def mutate(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 30:
            if field == "eos":
                tokenizer.eos_token = "<bos>"
            elif field == "padding":
                tokenizer.padding_side = "right"
            elif field == "truncation":
                tokenizer.truncation_side = "right"
            elif field == "vocabulary":
                metadata.vocab_size += 1
            else:
                metadata.max_position_embeddings += 1
        return result

    with (
        _assets(tokenizer),
        patch.object(AutoConfig, "from_pretrained", return_value=metadata),
        patch.object(service, "_encode", side_effect=mutate),
        pytest.raises(IrisStudyInputPreflightError),
    ):
        preflight_study_inputs_v1(benchmark, config=_config(), terminal_token_id=1)
    assert calls == 30
