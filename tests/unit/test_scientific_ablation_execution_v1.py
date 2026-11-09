"""Strict phase DATA grammar and hostile/corrupt metadata, not native execution proof."""

from typing import Any

import pytest
from pydantic import ValidationError

from iris.artifacts.scientific_ablation_execution_v1 import (
    ScientificAblationExecutionEventPayloadV1,
    ScientificAblationExecutionTrainingMetricV1,
    ScientificAblationExecutionValidationMetricV1,
)
from iris.artifacts.scientific_execution_v1 import ScientificExecutionTrainingMetricV1


def _training() -> dict[str, Any]:
    return dict(
        step_index=1,
        example_count=1,
        supervised_token_count=3,
        learning_rate=0.001,
        gradient_l2_norm=0.5,
        gradients_clipped=False,
        loss=1.0,
        language_model_loss=1.0,
        probe_loss=0.0,
        diversity_loss=0.0,
        utilization_loss=0.0,
        norm_loss=0.0,
        configuration_sha256="a" * 64,
        window_sha256="b" * 64,
        purpose="smoke",
        ablation_policy_sha256="c" * 64,
        model_configuration_sha256="d" * 64,
        scientific_state_before_sha256="e" * 64,
        scientific_state_after_sha256="f" * 64,
        registered_parameter_count=4,
        gradient_present_parameter_count=3,
        nonzero_gradient_parameter_count_before_clip=2,
    )


def _validation() -> dict[str, Any]:
    return dict(
        dataset_sha256="a" * 64,
        example_count=2,
        target_token_count=2,
        target_nll_sum=4.0,
        target_nll_per_token=2.0,
        at_optimizer_step=1,
        run_configuration_sha256="b" * 64,
        model_configuration_sha256="c" * 64,
        ablation_policy_sha256="d" * 64,
        scientific_state_sha256="e" * 64,
    )


def test_full_native_metric_shape_is_distinct_from_baseline() -> None:
    value = _training()
    actual = ScientificAblationExecutionTrainingMetricV1.model_validate(value)
    assert actual.model_dump() == value
    with pytest.raises(ValidationError):
        ScientificExecutionTrainingMetricV1.model_validate(value)
    for key in (
        "ablation_policy_sha256",
        "scientific_state_after_sha256",
        "registered_parameter_count",
    ):
        incomplete = dict(value)
        del incomplete[key]
        with pytest.raises(ValidationError):
            ScientificAblationExecutionTrainingMetricV1.model_validate(incomplete)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_index", 0),
        ("step_index", True),
        ("example_count", 0),
        ("example_count", 4097),
        ("supervised_token_count", False),
        ("learning_rate", 0.0),
        ("gradient_l2_norm", -1.0),
        ("gradients_clipped", 1),
        ("loss", float("nan")),
        ("loss", float("inf")),
        ("probe_loss", -1.0),
        ("purpose", "approved"),
        ("configuration_sha256", "A" * 64),
        ("ablation_policy_sha256", "not-a-native-mode"),
        ("scientific_state_before_sha256", "e" * 63),
        ("registered_parameter_count", 0),
        ("registered_parameter_count", True),
        ("gradient_present_parameter_count", 5),
        ("nonzero_gradient_parameter_count_before_clip", 4),
        ("nonzero_gradient_parameter_count_before_clip", -1),
        ("native_owner", "restored-from-json"),
    ],
)
def test_training_metadata_rejects_coercion_bounds_and_laundering(
    field: str, value: object
) -> None:
    raw = _training()
    raw[field] = value
    with pytest.raises(ValidationError):
        ScientificAblationExecutionTrainingMetricV1.model_validate(raw)


def test_zero_gradient_participation_does_not_manufacture_gradients() -> None:
    value = _training()
    value.update(gradient_present_parameter_count=0, nonzero_gradient_parameter_count_before_clip=0)
    actual = ScientificAblationExecutionTrainingMetricV1.model_validate(value)
    assert (
        actual.gradient_present_parameter_count
        == actual.nonzero_gradient_parameter_count_before_clip
        == 0
    )


def test_complete_validation_ratio_and_zero_nll_are_allowed_data() -> None:
    value = _validation()
    assert (
        ScientificAblationExecutionValidationMetricV1.model_validate(value).target_nll_per_token
        == 2.0
    )
    value.update(target_nll_sum=0.0, target_nll_per_token=0.0)
    assert ScientificAblationExecutionValidationMetricV1.model_validate(value).target_nll_sum == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_token_count", 0),
        ("target_token_count", True),
        ("example_count", 5001),
        ("target_nll_sum", float("inf")),
        ("target_nll_sum", -1.0),
        ("target_nll_per_token", 1.0),
        ("target_nll_per_token", float("nan")),
        ("at_optimizer_step", False),
        ("scientific_state_sha256", "A" * 64),
        ("ablation_policy_sha256", "wrong"),
        ("native_forward", True),
    ],
)
def test_validation_metadata_rejects_incomplete_ratio_identity_and_authority(
    field: str, value: object
) -> None:
    raw = _validation()
    raw[field] = value
    with pytest.raises(ValidationError):
        ScientificAblationExecutionValidationMetricV1.model_validate(raw)


def _event() -> dict[str, Any]:
    return dict(
        kind="iris-scientific-ablation-execution-event-v1",
        execution_sha256="a" * 64,
        sequence=1,
        previous_sha256="b" * 64,
        before={},
        after=dict(phase="validation", step=1),
        examples_seen=1,
        target_tokens_seen=3,
        history_sha256="c" * 64,
        scientific_state_sha256="f" * 64,
        record=dict(kind="train", metric=_training()),
    )


def test_event_carries_distinct_state_and_mode_bound_metric_without_native_token() -> None:
    event = ScientificAblationExecutionEventPayloadV1.model_validate(_event())
    assert event.kind == "iris-scientific-ablation-execution-event-v1"
    assert event.after.phase == "validation" and event.record.kind == "train"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "iris-scientific-execution-event-v1"),
        ("sequence", 0),
        ("sequence", True),
        ("sequence", 200001),
        ("scientific_state_sha256", "bad"),
        ("native_factory", "approved"),
        ("examples_seen", -1),
        ("target_tokens_seen", False),
    ],
)
def test_event_namespace_and_exact_counters_are_not_baseline_or_authority(
    field: str, value: object
) -> None:
    raw = _event()
    raw[field] = value
    with pytest.raises(ValidationError):
        ScientificAblationExecutionEventPayloadV1.model_validate(raw)
