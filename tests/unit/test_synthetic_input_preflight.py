"""Four-family tokenizer diagnostics using real tiny tokenizers, never model weights."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from tokenizers.processors import TemplateProcessing  # type: ignore[import-untyped]
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from iris.application import study_input_preflight as legacy
from iris.application import synthetic_input_preflight as service
from iris.application.errors import IrisSyntheticInputPreflightError
from iris.application.synthetic_input_preflight import (
    SyntheticContextBudget,
    SyntheticTokenizerSpec,
    preflight_synthetic_inputs,
)
from iris.data.audit import audit_synthetic_plan
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import (
    IrisExample,
    SyntheticDatasetPlan,
    SyntheticSplitPlan,
    SyntheticTemplate,
)
from iris.integrations.tokenizer_semantics import (
    encode_raw_token_ids,
    tokenizer_semantics_sha256,
    tokenizer_special_ids,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def _plan() -> SyntheticDatasetPlan:
    return SyntheticDatasetPlan(
        splits=(
            SyntheticSplitPlan(split="train", first_seed=0, count=15),
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
            SyntheticSplitPlan(split="test", first_seed=2000000, count=15),
        )
    )


def _spec() -> SyntheticTokenizerSpec:
    return SyntheticTokenizerSpec(
        repository="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        padding_side="left",
        truncation_side="left",
    )


def _budget(**updates: Any) -> SyntheticContextBudget:
    values: dict[str, Any] = {
        "name": "explicit_fixture",
        "max_prompt_tokens": 1024,
        "max_target_tokens_including_terminal": 512,
        "reserved_latent_tokens": 8,
    }
    return SyntheticContextBudget.model_validate(values | updates)


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "<eos>": 1, "<bos>": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend, eos_token="<eos>", bos_token="<bos>"
    )


@contextmanager
def _assets(
    tokenizer: PreTrainedTokenizerFast | None = None, **metadata: Any
) -> Iterator[tuple[MagicMock, MagicMock, SimpleNamespace]]:
    config = SimpleNamespace(
        **(
            {
                "model_type": "qwen3",
                "vocab_size": 8,
                "max_position_embeddings": 4096,
                "eos_token_id": 1,
            }
            | metadata
        )
    )
    with (
        patch.object(AutoConfig, "from_pretrained", return_value=config) as model_config,
        patch.object(
            AutoTokenizer, "from_pretrained", return_value=tokenizer or _tokenizer()
        ) as token_loader,
        patch.object(AutoModelForCausalLM, "from_pretrained") as weights,
    ):
        yield model_config, token_loader, config
    weights.assert_not_called()


def _run(**updates: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "plan": _plan(),
        "template_id": "iris-synthetic-01-v2",
        "tokenizer_spec": _spec(),
        "terminal_token_id": 1,
        "budgets": (_budget(),),
        "max_rows": 45,
        "max_logical_bytes": 1048576,
    }
    return preflight_synthetic_inputs(**(kwargs | updates))


@pytest.mark.parametrize("template", ["iris-synthetic-01-v1", "iris-synthetic-01-v2"])
def test_full_corpus_identity_and_exact_cache_only_loads(template: SyntheticTemplate) -> None:
    with _assets() as (model_config, tokenizer, _):
        report = _run(template_id=template)
    spec = _spec()
    model_config.assert_called_once_with(
        pretrained_model_name_or_path=spec.repository,
        revision=spec.model_revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    tokenizer.assert_called_once_with(
        pretrained_model_name_or_path=spec.repository,
        revision=spec.tokenizer_revision,
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    audit = audit_synthetic_plan(
        _plan(), template_id=template, max_rows=45, max_logical_bytes=1048576
    )
    for field in ("logical_sha256", "logical_bytes", "plan_sha256"):
        assert report[field] == audit[field]
    assert report["inputs_fit"] is True
    assert report["totals"]["rows"] == 45
    assert list(report["splits"]) == ["train", "validation", "test"]
    for split_name, totals in report["splits"].items():
        assert totals["rows"] == 15
        assert totals["invalid_content_rows"] == 0
        cells = report["strata"][split_name]
        assert len(cells) == 12
        assert sum(cell["rows"] for cell in cells.values()) == 15
    assert report["scientific_launch_status"] == "not_assessed"
    assert not report["model_weights_loaded"]
    assert not report["python_sources_executed"]
    assert not report["hardware_fit_verified"]
    serialized = json.dumps(report)
    assert "prompt_ids" not in serialized and "target_ids" not in serialized
    for row in generate_synthetic_split(_plan().splits[0], template_id=template):
        assert row.prompt not in serialized and row.expected_response not in serialized


def test_exact_encoding_includes_full_envelope_and_one_explicit_terminal() -> None:
    tokenizer = _tokenizer()
    with _assets(tokenizer):
        report = _run()
    digest = sha256(b"iris/synthetic-input-token-stream/v1\0")
    max_prompt = max_target = 0
    for split in _plan().splits:
        for row in generate_synthetic_split(split, template_id="iris-synthetic-01-v2"):
            prompt = tokenizer(
                row.prompt, add_special_tokens=True, padding=False, truncation=False
            )["input_ids"]
            content = tokenizer(
                row.expected_response, add_special_tokens=False, padding=False, truncation=False
            )["input_ids"]
            assert prompt[0] == 2 and 1 not in content
            target = [*content, 1]
            raw = canonical_json_bytes(row.model_dump(mode="json"))
            encoded = canonical_json_bytes(
                {
                    "id": row.id,
                    "logical_row_sha256": sha256(raw).hexdigest(),
                    "prompt_ids": prompt,
                    "target_ids": target,
                }
            )
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            max_prompt = max(max_prompt, len(prompt))
            max_target = max(max_target, len(target))
    assert report["token_stream_sha256"] == digest.hexdigest()
    assert report["totals"]["max_prompt_tokens"] == max_prompt
    assert report["totals"]["max_target_tokens_including_terminal"] == max_target


def test_overflow_keeps_every_row_and_each_budget_independent() -> None:
    small = _budget(name="tiny", max_prompt_tokens=1, max_target_tokens_including_terminal=2)
    with _assets():
        report = _run(budgets=(small, _budget()))
        same_corpus = _run(budgets=(_budget(reserved_latent_tokens=0),))
    assert report["inputs_fit"] is False
    assert report["budget_results"]["tiny"]["inputs_fit"] is False
    assert report["budget_results"]["explicit_fixture"]["inputs_fit"] is True
    assert report["totals"]["rows"] == 45
    assert report["totals"]["budgets"]["tiny"] == {
        "invalid_rows": 45,
        "violations": {"prompt_over_limit": 45, "target_over_limit": 45},
    }
    assert small.max_prompt_tokens == 1
    assert report["logical_sha256"] == same_corpus["logical_sha256"]
    assert report["token_stream_sha256"] == same_corpus["token_stream_sha256"]
    assert report["declaration_sha256"] != same_corpus["declaration_sha256"]


def test_inclusive_limits_and_full_configured_envelope() -> None:
    with _assets():
        initial = _run()
    prompt = initial["totals"]["max_prompt_tokens"]
    target = initial["totals"]["max_target_tokens_including_terminal"]
    exact = _budget(max_prompt_tokens=prompt, max_target_tokens_including_terminal=target)
    with _assets(max_position_embeddings=prompt + target + 8):
        fit = _run(budgets=(exact,))
    with _assets(max_position_embeddings=prompt + target + 7):
        overflow = _run(budgets=(exact,))
    assert fit["inputs_fit"] is True
    assert overflow["inputs_fit"] is False
    assert overflow["budget_results"][exact.name]["configured_context_envelope_fits"] is False
    assert "prompt_over_limit" not in overflow["totals"]["budgets"][exact.name]["violations"]
    with _assets(max_position_embeddings=1):
        all_overflow = _run()
    assert (
        all_overflow["totals"]["budgets"][exact.name]["violations"]["observed_context_over_limit"]
        == 45
    )


@pytest.mark.parametrize(
    "ids,violations",
    [
        ([], {"empty_prompt_encoding": 45, "empty_target_encoding": 45}),
        ([1], {"target_contains_special_tokens": 45}),
    ],
)
def test_invalid_content_is_reported_for_complete_corpus(
    ids: list[int], violations: dict[str, int]
) -> None:
    with (
        _assets(),
        patch.object(
            PreTrainedTokenizerFast,
            "__call__",
            return_value={
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            },
        ),
    ):
        report = _run()
    assert report["inputs_fit"] is False
    assert report["totals"]["rows"] == 45
    assert report["totals"]["invalid_content_rows"] == 45
    assert report["totals"]["content_violations"] == violations


@pytest.mark.parametrize(
    "updates",
    [
        {"max_rows": True},
        {"max_rows": 44},
        {"max_rows": 100001},
        {"max_logical_bytes": True},
        {"max_logical_bytes": 0},
        {"max_logical_bytes": 512 * 1024**2 + 1},
        {"template_id": "unknown"},
        {"terminal_token_id": True},
        {"terminal_token_id": -1},
        {"budgets": ()},
        {"budgets": (_budget(),) * 9},
        {"budgets": (_budget(), _budget())},
        {"budgets": [_budget()]},
        {"budgets": (_budget().model_copy(update={"reserved_latent_tokens": True}),)},
        {"tokenizer_spec": _spec().model_copy(update={"model_revision": "main"})},
        {"plan": _plan().model_copy(update={"splits": tuple(reversed(_plan().splits))})},
    ],
)
def test_bad_declarations_fail_before_loading(updates: dict[str, Any]) -> None:
    with (
        patch.object(AutoConfig, "from_pretrained") as loader,
        pytest.raises(IrisSyntheticInputPreflightError),
    ):
        _run(**updates)
    loader.assert_not_called()


def test_bad_later_v2_seed_range_fails_before_loading() -> None:
    plan = SyntheticDatasetPlan(
        splits=(
            *_plan().splits[:2],
            SyntheticSplitPlan(split="test", first_seed=2**26 - 1, count=15),
        )
    )
    with (
        patch.object(AutoConfig, "from_pretrained") as loader,
        pytest.raises(IrisSyntheticInputPreflightError),
    ):
        _run(plan=plan)
    loader.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_type": "other"},
        {"vocab_size": True},
        {"max_position_embeddings": 0},
        {"eos_token_id": 2},
        {"eos_token_id": True},
        {"eos_token_id": [1, 1]},
    ],
)
def test_bad_model_metadata_is_not_a_corpus_failure(metadata: dict[str, Any]) -> None:
    with _assets(**metadata), pytest.raises(IrisSyntheticInputPreflightError):
        _run()


@pytest.mark.parametrize(
    "output",
    [
        {"input_ids": [True], "attention_mask": [1]},
        {"input_ids": [8], "attention_mask": [1]},
        {"input_ids": [[0]], "attention_mask": [[1]]},
        {"input_ids": [0], "attention_mask": [0]},
        {"input_ids": [0], "attention_mask": []},
        {"input_ids": [0]},
    ],
)
def test_invalid_tokenizer_output_is_redacted_infrastructure_error(output: dict[str, Any]) -> None:
    with (
        _assets(),
        patch.object(PreTrainedTokenizerFast, "__call__", return_value=output),
        pytest.raises(IrisSyntheticInputPreflightError),
    ):
        _run()


@pytest.mark.parametrize("kind", ["short", "excess", "reorder", "template", "error"])
def test_late_generator_failures_never_return_a_partial_pass(kind: str) -> None:
    def rows(split: SyntheticSplitPlan, *, template_id: SyntheticTemplate) -> Iterator[IrisExample]:
        generated = list(generate_synthetic_split(split, template_id=template_id))
        if split.split == "test":
            if kind == "short":
                generated.pop()
            elif kind == "excess":
                generated.append(generated[-1])
            elif kind == "reorder":
                generated[-1], generated[-2] = generated[-2], generated[-1]
            elif kind == "template":
                generated[-1] = generated[-1].model_copy(
                    update={"template_id": "iris-synthetic-01-v1"}
                )
            else:
                yield from generated
                raise ValueError("PRIVATE generator failure")
        yield from generated

    with (
        _assets(),
        patch.object(service, "generate_synthetic_split", side_effect=rows),
        pytest.raises(IrisSyntheticInputPreflightError) as caught,
    ):
        _run()
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("field", ["vocab", "eos", "positions", "splitting", "padding"])
def test_last_encoding_metadata_mutation_is_detected(field: str) -> None:
    tokenizer = _tokenizer()
    original = encode_raw_token_ids
    calls = 0
    with _assets(tokenizer) as (_, _, config):

        def mutate(*args: Any, **kwargs: Any) -> tuple[int, ...]:
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 90:
                if field == "vocab":
                    tokenizer.add_tokens(["CHANGED"])
                elif field == "eos":
                    config.eos_token_id = 2
                elif field == "positions":
                    config.max_position_embeddings += 1
                elif field == "splitting":
                    tokenizer.split_special_tokens = True
                else:
                    tokenizer.padding_side = "right"
            return result

        with (
            patch.object(service, "encode_raw_token_ids", side_effect=mutate),
            pytest.raises(IrisSyntheticInputPreflightError),
        ):
            _run()
    assert calls == 90


def test_logical_byte_limit_is_enforced_before_encoding_first_row() -> None:
    with (
        _assets(),
        patch.object(service, "encode_raw_token_ids") as encode,
        pytest.raises(IrisSyntheticInputPreflightError),
    ):
        _run(max_logical_bytes=1)
    encode.assert_not_called()


def test_missing_cache_is_redacted_without_retry_or_download_fallback() -> None:
    with (
        patch.object(AutoConfig, "from_pretrained", side_effect=OSError("PRIVATE cache")) as loader,
        patch.object(AutoTokenizer, "from_pretrained") as tokenizer,
        pytest.raises(IrisSyntheticInputPreflightError) as caught,
    ):
        _run()
    assert loader.call_count == 1
    assert loader.call_args.kwargs["local_files_only"] is True
    tokenizer.assert_not_called()
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_shared_helpers_preserve_historical_fingerprint_payload_and_private_aliases() -> None:
    tokenizer = _tokenizer()
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("truncation", None)
    backend.pop("padding", None)
    expected = domain_sha256(
        "iris/study-tokenizer-semantics/v1",
        canonical_json_bytes(
            {
                "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "backend": backend,
                "special_tokens_map": tokenizer.special_tokens_map,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "eos_token_id": tokenizer.eos_token_id,
                "split_special_tokens": tokenizer.split_special_tokens,
                "prompt_policy": "raw_text_add_special_tokens_no_padding_no_truncation",
                "target_policy": "raw_text_no_special_tokens_no_padding_no_truncation_plus_eos",
                "transformers_version": version("transformers"),
                "tokenizers_version": version("tokenizers"),
            }
        ),
    )
    assert legacy._tokenizer_fingerprint is tokenizer_semantics_sha256
    assert legacy._encode is encode_raw_token_ids
    assert legacy._special_ids is tokenizer_special_ids
    assert tokenizer_semantics_sha256(tokenizer) == expected


def test_tokenizer_call_options_are_explicit_untruncated_unpadded_and_no_chat_template() -> None:
    with (
        _assets(),
        patch.object(
            PreTrainedTokenizerFast,
            "__call__",
            return_value={
                "input_ids": [0],
                "attention_mask": [1],
            },
        ) as encode,
        patch.object(PreTrainedTokenizerFast, "apply_chat_template") as chat,
    ):
        report = _run()
    assert report["inputs_fit"] is True
    assert encode.call_count == 90
    chat.assert_not_called()
    for index, call in enumerate(encode.call_args_list):
        assert call.kwargs == {
            "add_special_tokens": index % 2 == 0,
            "padding": False,
            "truncation": False,
            "return_attention_mask": True,
            "return_token_type_ids": False,
        }


def test_logical_byte_limit_is_inclusive_and_metadata_lists_allow_explicit_eos() -> None:
    with _assets(eos_token_id=[1, 2]):
        report = _run()
        exact = _run(max_logical_bytes=report["logical_bytes"])
        with pytest.raises(IrisSyntheticInputPreflightError):
            _run(max_logical_bytes=report["logical_bytes"] - 1)
    assert exact == report


def test_different_valid_revision_is_bound_without_substituting_default_assets() -> None:
    spec = _spec().model_copy(update={"repository": "Qwen/Qwen3-1.7B", "model_revision": "a" * 40})
    with _assets() as (model, tokenizer, _):
        report = _run(tokenizer_spec=spec)
    assert model.call_args.kwargs["revision"] == "a" * 40
    assert model.call_args.kwargs["pretrained_model_name_or_path"] == "Qwen/Qwen3-1.7B"
    assert tokenizer.call_args.kwargs["revision"] == spec.tokenizer_revision
    assert report["tokenizer_spec"] == spec.model_dump(mode="json")
