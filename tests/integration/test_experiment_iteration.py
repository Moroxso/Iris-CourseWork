"""Offline deterministic manifest-to-teacher iteration integration tests."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from iris.application.errors import IrisExperimentIterationError
from iris.application.experiment_manifest import (
    IrisExperimentExampleIterator,
    IrisPreparedExperimentExample,
    build_experiment_manifest_v1,
    iterate_experiment_split_v1,
)
from iris.artifacts.experiment_v1 import (
    ExperimentExampleV1,
    ExperimentSplitV1,
    IrisExperimentManifestV1,
)
from iris.backbone import BackboneAdapter
from iris.training import (
    IrisAdamWConfig,
    IrisTeacherForcingModel,
    IrisTeacherOptimizer,
    IrisTokenWeightedEvaluationAccumulator,
)
from tests.integration.test_teacher_forcing_sidecar import (
    _build_offline_model,
    _TinyTokenizer,
)

pytestmark = pytest.mark.integration


def _example(example_id: str, split: str, target: str) -> ExperimentExampleV1:
    return ExperimentExampleV1.model_validate(
        {
            "example_id": example_id,
            "split": split,
            "prompt": "offline teacher prompt",
            "target": target,
        }
    )


def _manifest(*, include_test: bool = True) -> IrisExperimentManifestV1:
    examples = (
        _example("train-long", "train", "longer target"),
        _example("validation-short", "validation", "short target"),
        _example("train-short", "train", "short target"),
        *((_example("test-long", "test", "longer target"),) if include_test else ()),
    )
    return build_experiment_manifest_v1(dataset_id="offline-baseline-v1", examples=examples)


def _build() -> tuple[IrisTeacherForcingModel, _TinyTokenizer]:
    untyped_teacher, tokenizer, _ = _build_offline_model()
    return cast(IrisTeacherForcingModel, untyped_teacher), tokenizer


def test_split_iterator_is_lazy_single_pass_and_preserves_manifest_order() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, tokenizer = _build()
        iterator = iterate_experiment_split_v1(
            manifest=_manifest(),
            split="train",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )

        assert type(iterator) is IrisExperimentExampleIterator
        assert iterator.example_count == 2
        assert iterator.remaining_examples == 2
        assert tokenizer.calls == []

        first = next(iterator)
        assert iterator.remaining_examples == 1
        second = next(iterator)
        assert iterator.remaining_examples == 0
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)

    assert type(first) is IrisPreparedExperimentExample
    assert (first.example_id, second.example_id) == ("train-long", "train-short")
    assert (first.split_ordinal, second.split_ordinal) == (1, 2)
    assert first.split_example_count == second.split_example_count == 2
    assert first.dataset_id == second.dataset_id == "offline-baseline-v1"
    assert first.manifest_payload_sha256 == second.manifest_payload_sha256
    assert first.split == second.split == "train"
    assert first.teacher_input.prompt == second.teacher_input.prompt == "offline teacher prompt"
    assert torch.equal(
        first.teacher_input.target_input_ids,
        torch.tensor([[5, 6, 15]], dtype=torch.int64),
    )
    assert torch.equal(
        second.teacher_input.target_input_ids,
        torch.tensor([[5, 15]], dtype=torch.int64),
    )
    assert (
        first.teacher_input.target_input_ids.untyped_storage().data_ptr()
        != second.teacher_input.target_input_ids.untyped_storage().data_ptr()
    )
    assert [text for text, _ in tokenizer.calls] == ["longer target", "short target"]


def test_manifest_iteration_drives_real_training_then_token_weighted_validation() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        teacher, tokenizer = _build()
        owner = IrisTeacherOptimizer(
            model=teacher,
            config=IrisAdamWConfig(
                learning_rate=1e-3,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.0,
                max_gradient_norm=1.0,
            ),
        )
        manifest = _manifest()
        training_results = [
            owner.step(prepared.teacher_input)
            for prepared in iterate_experiment_split_v1(
                manifest=manifest,
                split="train",
                backbone=teacher._backbone,
                terminal_token_id=15,
            )
        ]
        assert [result.step_index for result in training_results] == [1, 2]
        assert [result.supervised_token_count for result in training_results] == [3, 2]
        assert owner.completed_steps == 2
        assert all(parameter.grad is None for parameter in teacher.parameters())

        teacher.eval()
        accumulator = IrisTokenWeightedEvaluationAccumulator()
        validation_ids: list[str] = []
        for prepared in iterate_experiment_split_v1(
            manifest=manifest,
            split="validation",
            backbone=teacher._backbone,
            terminal_token_id=15,
        ):
            validation_ids.append(prepared.example_id)
            accumulator.update(teacher.evaluation_forward(prepared.teacher_input))
        summary = accumulator.summary()

    assert validation_ids == ["validation-short"]
    assert summary.example_count == 1
    assert summary.supervised_token_count == 2
    assert summary.mean_loss >= 0.0
    assert summary.perplexity is not None
    assert [text for text, _ in tokenizer.calls] == [
        "longer target",
        "offline teacher prompt",
        "short target",
        "offline teacher prompt",
        "short target",
        "offline teacher prompt",
    ]


def test_optional_empty_test_split_is_a_valid_deterministic_empty_iterator() -> None:
    with torch.random.fork_rng(devices=[]):
        teacher, tokenizer = _build()
        iterator = iterate_experiment_split_v1(
            manifest=_manifest(include_test=False),
            split="test",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )

        assert iterator.example_count == 0
        assert iterator.remaining_examples == 0
        assert list(iterator) == []
        assert tokenizer.calls == []


def test_preparation_failure_poisons_iterator_without_skipping_to_next_example() -> None:
    manifest = build_experiment_manifest_v1(
        dataset_id="failure-v1",
        examples=(
            _example("bad-train", "train", "unsupported target"),
            _example("good-train", "train", "short target"),
            _example("validation", "validation", "short target"),
        ),
    )
    with torch.random.fork_rng(devices=[]):
        teacher, tokenizer = _build()
        iterator = iterate_experiment_split_v1(
            manifest=manifest,
            split="train",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )

        with pytest.raises(IrisExperimentIterationError, match="bad-train"):
            next(iterator)
        calls_after_failure = tuple(tokenizer.calls)
        assert iterator.remaining_examples == 2
        with pytest.raises(IrisExperimentIterationError, match="poisoned"):
            next(iterator)

    assert tuple(tokenizer.calls) == calls_after_failure
    assert [text for text, _ in tokenizer.calls] == ["unsupported target"]


def test_backbone_identity_change_between_examples_poisons_iterator() -> None:
    with torch.random.fork_rng(devices=[]):
        teacher, tokenizer = _build()
        iterator = iterate_experiment_split_v1(
            manifest=_manifest(),
            split="train",
            backbone=teacher._backbone,
            terminal_token_id=15,
        )
        first = next(iterator)
        backbone = teacher._backbone
        backbone._config.local_files_only = False

        with pytest.raises(IrisExperimentIterationError, match="train-short"):
            next(iterator)
        calls_after_failure = tuple(tokenizer.calls)
        with pytest.raises(IrisExperimentIterationError, match="poisoned"):
            next(iterator)

    assert first.example_id == "train-long"
    assert tuple(tokenizer.calls) == calls_after_failure
    assert [text for text, _ in tokenizer.calls] == ["longer target"]


@pytest.mark.parametrize(
    ("split", "terminal_token_id"),
    [("eval", 15), ("TRAIN", 15), ("train", 14), ("train", True)],
)
def test_iterator_rejects_invalid_split_or_terminal_before_tokenization(
    split: object,
    terminal_token_id: object,
) -> None:
    with torch.random.fork_rng(devices=[]):
        teacher, tokenizer = _build()

        with pytest.raises(IrisExperimentIterationError):
            iterate_experiment_split_v1(
                manifest=_manifest(),
                split=cast(ExperimentSplitV1, split),
                backbone=teacher._backbone,
                terminal_token_id=cast(int, terminal_token_id),
            )

    assert tokenizer.calls == []


def test_iterator_rejects_manifest_subclass_or_wrong_backbone() -> None:
    class ManifestSubclass(IrisExperimentManifestV1):
        pass

    manifest = _manifest()
    subclass = ManifestSubclass.model_validate(manifest.model_dump(mode="python"))
    with torch.random.fork_rng(devices=[]):
        teacher, _ = _build()
        with pytest.raises(IrisExperimentIterationError):
            iterate_experiment_split_v1(
                manifest=subclass,
                split="train",
                backbone=teacher._backbone,
                terminal_token_id=15,
            )
        with pytest.raises(IrisExperimentIterationError):
            iterate_experiment_split_v1(
                manifest=manifest,
                split="train",
                backbone=cast(BackboneAdapter, object()),
                terminal_token_id=15,
            )
