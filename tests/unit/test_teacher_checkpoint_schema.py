"""Strict persisted-schema tests for Iris teacher checkpoint v1."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from iris.artifacts.checkpoint_v1 import (
    IrisTeacherCheckpointEnvelopeV1,
    TeacherAdamWConfigV1,
    TeacherAdamWParameterStateV1,
    TeacherBackboneRuntimeV1,
    TeacherCheckpointModelIdentityV1,
    TeacherCheckpointOptimizerV1,
    TeacherCheckpointPayloadV1,
    TeacherCheckpointSchemaV1,
    TeacherCheckpointSoftwareV1,
    TeacherCheckpointStateFileV1,
    TeacherCheckpointTensorV1,
)
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig

SHA = "a" * 64


def _backbone() -> BackboneConfig:
    return BackboneConfig.model_validate(
        {
            "name": "tiny",
            "repository": "offline/tiny",
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "trust_remote_code": False,
            "local_files_only": True,
            "output_hidden_states": True,
            "use_cache": False,
            "dtype": "fp32",
            "tokenizer": {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
            },
            "limits": {"max_prompt_tokens": 5, "max_new_tokens": 3},
        }
    )


def _cognitive() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "tiny-cognitive",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def _workspace() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "tiny-workspace",
            "num_slots": 2,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.5,
            "empty_threshold": 1e-4,
            "arbiter": {
                "confidence_weight": 1.0,
                "novelty_weight": 1.0,
                "relevance_weight": 1.0,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def _runtime() -> TeacherBackboneRuntimeV1:
    return TeacherBackboneRuntimeV1(
        hidden_size=6,
        num_hidden_layers=4,
        vocabulary_size=16,
        max_position_embeddings=10,
        max_prompt_tokens=5,
        max_target_tokens=3,
        device="cpu",
        dtype="float32",
    )


def _model(*, state_names: tuple[str, ...] = ("weight",)) -> TeacherCheckpointModelIdentityV1:
    return TeacherCheckpointModelIdentityV1(
        backbone=_backbone(),
        cognitive=_cognitive(),
        workspace=_workspace(),
        runtime=_runtime(),
        state_names=state_names,
        sha256=SHA,
    )


def _adamw_config() -> TeacherAdamWConfigV1:
    return TeacherAdamWConfigV1(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        max_gradient_norm=1.0,
    )


def _optimizer(
    *,
    completed_steps: int = 0,
    parameter_states: tuple[TeacherAdamWParameterStateV1, ...] = (),
) -> TeacherCheckpointOptimizerV1:
    return TeacherCheckpointOptimizerV1(
        optimizer_type="torch.optim.AdamW",
        completed_steps=completed_steps,
        config=_adamw_config(),
        parameter_names=("weight",),
        parameter_states=parameter_states,
    )


def _sidecar_tensor() -> TeacherCheckpointTensorV1:
    return TeacherCheckpointTensorV1(
        key="sidecar.00000000",
        owner="sidecar",
        parameter_name=None,
        state_name="weight",
        shape=(2,),
        dtype="float32",
        byte_length=8,
        sha256=SHA,
    )


def _adamw_tensor(state_name: str) -> TeacherCheckpointTensorV1:
    return TeacherCheckpointTensorV1(
        key=f"adamw.00000000.{state_name}",
        owner="adamw",
        parameter_name="weight",
        state_name=state_name,
        shape=(2,),
        dtype="float32",
        byte_length=8,
        sha256=SHA,
    )


def _payload(
    *,
    model: TeacherCheckpointModelIdentityV1 | None = None,
    optimizer: TeacherCheckpointOptimizerV1 | None = None,
    tensors: tuple[TeacherCheckpointTensorV1, ...] | None = None,
) -> TeacherCheckpointPayloadV1:
    actual_optimizer = optimizer or _optimizer()
    actual_tensors = tensors or (_sidecar_tensor(),)
    return TeacherCheckpointPayloadV1(
        model=model or _model(),
        optimizer=actual_optimizer,
        state_file=TeacherCheckpointStateFileV1(
            filename="state.safetensors",
            size_bytes=128,
            tensor_count=len(actual_tensors),
            sha256=SHA,
        ),
        tensors=actual_tensors,
        software=TeacherCheckpointSoftwareV1(
            iris="0.1.0",
            python="3.12.0",
            torch="2.13.0",
            safetensors="0.8.0",
        ),
    )


def test_complete_zero_step_envelope_is_strict_and_frozen() -> None:
    envelope = IrisTeacherCheckpointEnvelopeV1(
        schema_identity=TeacherCheckpointSchemaV1(
            artifact_type="iris.teacher_checkpoint",
            version=1,
            canonicalization="iris-json-v1",
            tensor_encoding="safetensors-v1",
            hash_algorithm="sha256",
        ),
        payload=_payload(),
        sha256=SHA,
    )

    assert envelope.payload.optimizer.completed_steps == 0
    assert envelope.payload.state_file.tensor_count == 1
    with pytest.raises(ValidationError):
        IrisTeacherCheckpointEnvelopeV1.model_validate(
            {**envelope.model_dump(mode="python"), "unknown": 1}
        )
    with pytest.raises(ValidationError):
        envelope.sha256 = "b" * 64


@pytest.mark.parametrize("version", [True, "1", 1.0, 2])
def test_checkpoint_schema_rejects_nonexact_or_unsupported_version(
    version: object,
) -> None:
    with pytest.raises(ValidationError, match="version"):
        TeacherCheckpointSchemaV1.model_validate(
            {
                "artifact_type": "iris.teacher_checkpoint",
                "version": version,
                "canonicalization": "iris-json-v1",
                "tensor_encoding": "safetensors-v1",
                "hash_algorithm": "sha256",
            }
        )


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, 1, True])
def test_sha256_fields_reject_noncanonical_values(digest: object) -> None:
    values = _sidecar_tensor().model_dump(mode="python")
    values["sha256"] = digest
    with pytest.raises(ValidationError, match="sha256"):
        TeacherCheckpointTensorV1.model_validate(values)


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_name": "weight"},
        {"byte_length": 7},
        {"shape": (-1,)},
        {"shape": (True,)},
        {"key": "bad\0key"},
    ],
)
def test_sidecar_tensor_descriptor_rejects_invalid_layout(changes: dict[str, Any]) -> None:
    values = _sidecar_tensor().model_dump(mode="python")
    values.update(changes)
    with pytest.raises(ValidationError):
        TeacherCheckpointTensorV1.model_validate(values)


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_name": None},
        {"parameter_name": ""},
        {"parameter_name": "bad\0name"},
        {"state_name": "step"},
        {"state_name": "moment"},
    ],
)
def test_adamw_tensor_descriptor_rejects_invalid_identity(changes: dict[str, Any]) -> None:
    values = _adamw_tensor("exp_avg").model_dump(mode="python")
    values.update(changes)
    with pytest.raises(ValidationError, match="AdamW"):
        TeacherCheckpointTensorV1.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("learning_rate", "0.1"),
        ("beta1", 1.0),
        ("beta2", -0.1),
        ("epsilon", float("inf")),
        ("weight_decay", -0.1),
        ("max_gradient_norm", 0.0),
    ],
)
def test_adamw_config_rejects_invalid_values(field: str, value: object) -> None:
    values = _adamw_config().model_dump(mode="python")
    values[field] = value
    with pytest.raises(ValidationError):
        TeacherAdamWConfigV1.model_validate(values)


def test_model_identity_rejects_configuration_runtime_disagreement() -> None:
    with pytest.raises(ValidationError, match="prompt limit"):
        TeacherCheckpointModelIdentityV1(
            backbone=_backbone(),
            cognitive=_cognitive(),
            workspace=_workspace(),
            runtime=_runtime().model_copy(update={"max_prompt_tokens": 4}),
            state_names=("weight",),
            sha256=SHA,
        )
    with pytest.raises(ValidationError, match="runtime dtype"):
        TeacherCheckpointModelIdentityV1(
            backbone=_backbone(),
            cognitive=_cognitive(),
            workspace=_workspace(),
            runtime=_runtime().model_copy(update={"dtype": "float16"}),
            state_names=("weight",),
            sha256=SHA,
        )


@pytest.mark.parametrize("names", [(), ("",), ("x", "x"), ("bad\0name",)])
def test_model_state_names_are_nonempty_unique_exact_strings(names: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="state_names"):
        _model(state_names=names)


def test_optimizer_state_requires_owned_unique_names_and_consistent_steps() -> None:
    valid_state = TeacherAdamWParameterStateV1(parameter_name="weight", step=1)
    assert _optimizer(completed_steps=1, parameter_states=(valid_state,)).completed_steps == 1

    with pytest.raises(ValidationError, match="outside"):
        TeacherCheckpointOptimizerV1(
            optimizer_type="torch.optim.AdamW",
            completed_steps=1,
            config=_adamw_config(),
            parameter_names=("weight",),
            parameter_states=(TeacherAdamWParameterStateV1(parameter_name="other", step=1),),
        )
    with pytest.raises(ValidationError, match="duplicates"):
        _optimizer(completed_steps=1, parameter_states=(valid_state, valid_state))
    with pytest.raises(ValidationError, match="exceeds"):
        _optimizer(
            completed_steps=1,
            parameter_states=(TeacherAdamWParameterStateV1(parameter_name="weight", step=2),),
        )
    with pytest.raises(ValidationError, match="zero-step"):
        _optimizer(completed_steps=0, parameter_states=(valid_state,))
    with pytest.raises(ValidationError, match="stepped"):
        _optimizer(completed_steps=1, parameter_states=())


def test_payload_requires_sorted_complete_sidecar_and_adamw_tensor_tables() -> None:
    state = TeacherAdamWParameterStateV1(parameter_name="weight", step=1)
    optimizer = _optimizer(completed_steps=1, parameter_states=(state,))
    tensors = (
        _adamw_tensor("exp_avg"),
        _adamw_tensor("exp_avg_sq"),
        _sidecar_tensor(),
    )
    payload = _payload(optimizer=optimizer, tensors=tensors)
    assert payload.tensors == tensors

    with pytest.raises(ValidationError, match="sorted"):
        _payload(optimizer=optimizer, tensors=tuple(reversed(tensors)))
    with pytest.raises(ValidationError, match="AdamW tensor table"):
        _payload(
            optimizer=optimizer,
            tensors=(_adamw_tensor("exp_avg"), _sidecar_tensor()),
        )
    with pytest.raises(ValidationError, match="sidecar tensor table"):
        _payload(model=_model(state_names=("other",)), tensors=(_sidecar_tensor(),))


def test_payload_rejects_duplicate_adamw_semantic_tensor_with_a_distinct_key() -> None:
    state = TeacherAdamWParameterStateV1(parameter_name="weight", step=1)
    optimizer = _optimizer(completed_steps=1, parameter_states=(state,))
    duplicate = _adamw_tensor("exp_avg").model_copy(
        update={"key": "adamw.00000000.exp_avg_duplicate"}
    )
    tensors = tuple(
        sorted(
            (
                _adamw_tensor("exp_avg"),
                duplicate,
                _adamw_tensor("exp_avg_sq"),
                _sidecar_tensor(),
            ),
            key=lambda tensor: tensor.key,
        )
    )

    with pytest.raises(ValidationError, match="AdamW tensor table"):
        _payload(optimizer=optimizer, tensors=tensors)
