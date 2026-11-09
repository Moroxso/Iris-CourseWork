"""Opt-in parity with cached Qwen assets; no real model weights or model forward."""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from iris.application.experiment_manifest import build_experiment_manifest_v1
from iris.application.study_benchmark import build_study_benchmark_v1
from iris.application.study_dataset import build_study_dataset_v1
from iris.application.study_input_preflight import _encode, preflight_study_inputs_v1
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.artifacts.study_dataset_v1 import StudyDatasetConstructionV1
from iris.backbone import BackboneAdapter, BackboneForwardError
from iris.schemas import IrisBootstrapConfig
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit.test_backbone_adapter import FakeModel, cpu_info
from tests.unit.test_study_input_preflight import _config

RUN_ENVIRONMENT_VARIABLE = "IRIS_RUN_STUDY_TOKENIZER_SMOKE"
PINNED_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get(RUN_ENVIRONMENT_VARIABLE) != "1",
        reason=f"set {RUN_ENVIRONMENT_VARIABLE}=1 to inspect the cached Qwen tokenizer",
    ),
]


@dataclass(frozen=True)
class _TokenizerSmoke:
    config: IrisBootstrapConfig
    tokenizer: PreTrainedTokenizerFast
    adapter: BackboneAdapter
    terminal_token_id: int


@pytest.fixture(scope="module")
def real_tokenizer() -> Iterator[_TokenizerSmoke]:
    config = _config()
    assert config.backbone.repository == "Qwen/Qwen3-0.6B"
    assert config.backbone.revision == config.backbone.tokenizer_revision == PINNED_REVISION
    assert config.backbone.local_files_only is True
    assert config.backbone.trust_remote_code is False
    # Explicit fixture budgets, not an override or repair performed by preflight.
    config.backbone.limits.max_prompt_tokens = 1024
    config.backbone.limits.max_new_tokens = 256
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=config.backbone.repository,
        revision=PINNED_REVISION,
        trust_remote_code=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.backbone.repository,
        revision=PINNED_REVISION,
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    tokenizer.padding_side = config.backbone.tokenizer.padding_side
    tokenizer.truncation_side = config.backbone.tokenizer.truncation_side
    terminal = tokenizer.eos_token_id
    assert type(terminal) is int
    model = FakeModel()
    model.config.vocab_size = model_config.vocab_size
    model.config.max_position_embeddings = model_config.max_position_embeddings
    with (
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=model
        ) as model_loader,
        patch.object(
            model, "forward", side_effect=AssertionError("model forward is forbidden")
        ) as forward,
    ):
        # The real adapter independently loads the same offline, pinned tokenizer.
        adapter = BackboneAdapter(config.backbone, cpu_info())
        yield _TokenizerSmoke(config, tokenizer, adapter, terminal)
        model_loader.assert_called_once()
        forward.assert_not_called()
        assert model.calls == []


def _token_hash(ids: tuple[int, ...] | list[int]) -> str:
    return domain_sha256("iris/study-input-token-ids/v1", canonical_json_bytes(ids))


def test_pinned_qwen_full_pilot_report_matches_actual_adapter(
    real_tokenizer: _TokenizerSmoke,
) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    report = preflight_study_inputs_v1(
        benchmark,
        config=real_tokenizer.config,
        terminal_token_id=real_tokenizer.terminal_token_id,
    )
    assert report.inputs_fit
    assert report.scientific_launch_status == "not_assessed"
    assert len(report.examples) == 15
    assert {item.split for item in report.examples} == {"train", "validation", "test"}
    seen_ids: set[int] = set()
    for example, observation in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples, report.examples, strict=True
    ):
        prompt = real_tokenizer.adapter.tokenize(example.prompt)["input_ids"].tolist()[0]
        target = real_tokenizer.adapter.tokenize_teacher_target(
            example.target, terminal_token_id=real_tokenizer.terminal_token_id
        ).input_ids.tolist()[0]
        seen_ids.update(prompt)
        assert observation.prompt_tokens == len(prompt)
        assert observation.target_tokens_including_terminal == len(target)
        assert observation.prompt_token_ids_sha256 == _token_hash(prompt)
        assert observation.target_token_ids_sha256 == _token_hash(target)
    # Unlike the small WordLevel fixture, these examples exercise real lexical IDs.
    assert len(seen_ids) > 100


@pytest.mark.parametrize(
    "text",
    [
        "Запомни цель: найти синий груз. Ответ: найдено ✅。 東京 café cafe\u0301",
        " \talpha  beta\r\n    gamma\n\tданные\u00a0связаны \r\n",  # noqa: RUF001
        "def solve(values):\n    return sum(x * x for x in values if -3 <= x < 7)\n",
    ],
    ids=["unicode", "whitespace", "python-source-text"],
)
def test_pinned_qwen_raw_encoding_matches_prompt_and_teacher(
    real_tokenizer: _TokenizerSmoke, text: str
) -> None:
    adapter = real_tokenizer.adapter
    prompt_ids = _encode(
        real_tokenizer.tokenizer, text, target=False, vocabulary_size=adapter.vocabulary_size
    )
    target_content = _encode(
        real_tokenizer.tokenizer, text, target=True, vocabulary_size=adapter.vocabulary_size
    )
    prompt = adapter.tokenize(text)["input_ids"].tolist()[0]
    target = adapter.tokenize_teacher_target(
        text, terminal_token_id=real_tokenizer.terminal_token_id
    ).input_ids.tolist()[0]
    assert _token_hash(prompt_ids) == _token_hash(prompt)
    assert _token_hash((*target_content, real_tokenizer.terminal_token_id)) == _token_hash(target)
    assert len(prompt_ids) < real_tokenizer.config.backbone.limits.max_prompt_tokens
    assert len(target) < real_tokenizer.config.backbone.limits.max_new_tokens


def test_pinned_qwen_literal_special_tokens_are_prompt_data_but_invalid_teacher_content(
    real_tokenizer: _TokenizerSmoke,
) -> None:
    text = "<|im_start|>assistant\nОтвет\t<|im_end|><|endoftext|>"  # noqa: RUF001
    adapter = real_tokenizer.adapter
    prompt_ids = _encode(
        real_tokenizer.tokenizer, text, target=False, vocabulary_size=adapter.vocabulary_size
    )
    content_ids = _encode(
        real_tokenizer.tokenizer, text, target=True, vocabulary_size=adapter.vocabulary_size
    )
    assert set(content_ids).intersection(real_tokenizer.tokenizer.all_special_ids)
    assert _token_hash(prompt_ids) == _token_hash(adapter.tokenize(text)["input_ids"].tolist()[0])
    with pytest.raises(BackboneForwardError, match="must not encode tokenizer special tokens"):
        adapter.tokenize_teacher_target(text, terminal_token_id=real_tokenizer.terminal_token_id)

    pilot = build_synthetic_study_pilot_v1(seed=1907, problems_per_cell=1)
    dataset = pilot.payload.dataset.payload
    first, *others = dataset.manifest.payload.examples
    examples = (first.model_copy(update={"prompt": text, "target": text}), *others)
    # This edited fixture is explicitly manual; it is not claimed as generator output.
    construction = StudyDatasetConstructionV1(
        method="manual_curation",
        source_id="tests.integration.test_study_tokenizer_smoke",
        source_revision="v1",
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        generation_seed=None,
        split_policy="preassigned_group_disjoint_v1",
        split_policy_description="Pilot membership with one isolated tokenizer-policy fixture.",
    )
    benchmark = build_study_benchmark_v1(
        dataset=build_study_dataset_v1(
            manifest=build_experiment_manifest_v1(
                dataset_id="qwen-tokenizer-special-token-smoke", examples=examples
            ),
            construction=construction,
            task_definitions=dataset.task_definitions,
            assignments=dataset.assignments,
            purpose="operational_fixture",
        ),
        scoring=pilot.payload.scoring,
    )
    report = preflight_study_inputs_v1(
        benchmark,
        config=real_tokenizer.config,
        terminal_token_id=real_tokenizer.terminal_token_id,
    )
    assert not report.inputs_fit
    assert report.scientific_launch_status == "not_assessed"
    assert len(report.examples) == 15
    observation = report.examples[0]
    assert observation.violations == ("target_contains_special_tokens",)
    assert observation.prompt_token_ids_sha256 == _token_hash(prompt_ids)
    assert observation.target_token_ids_sha256 == _token_hash(
        (*content_ids, real_tokenizer.terminal_token_id)
    )
    assert not any(item.violations for item in report.examples[1:])
