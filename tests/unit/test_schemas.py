from typing import Literal, cast

import pytest
from pydantic import ValidationError

from iris.schemas import (
    BackboneConfig,
    BackboneLimitsConfig,
    CognitiveConfig,
    HardwareConfig,
    IrisBootstrapConfig,
    LayerSamplerConfig,
    RuntimeConfig,
    SlotAllocatorConfig,
    TokenizerConfig,
    WorkspaceArbiterConfig,
    WorkspaceConfig,
)


def valid_backbone_data() -> dict[str, object]:
    return {
        "name": "qwen3_0_6b",
        "repository": "Qwen/Qwen3-0.6B",
        "revision": "main",
        "tokenizer_revision": "main",
        "trust_remote_code": False,
        "local_files_only": False,
        "output_hidden_states": True,
        "use_cache": False,
        "dtype": "fp32",
        "tokenizer": {
            "use_fast": True,
            "padding_side": "left",
            "truncation_side": "left",
        },
        "limits": {"max_prompt_tokens": 256, "max_new_tokens": 32},
    }


def valid_bootstrap_data() -> dict[str, object]:
    backbone = valid_backbone_data()
    backbone["revision"] = "a" * 40
    backbone["tokenizer_revision"] = "b" * 40
    return {
        "project": {"name": "iris", "version": "0.1.0"},
        "hardware": {
            "name": "cpu_smoke",
            "device": "cpu",
            "precision": "fp32",
            "max_memory_gb": 24.0,
            "allow_gpu": False,
        },
        "backbone": backbone,
        "cognitive": {
            "name": "iris_gw_0_1",
            "state_dim": 256,
            "num_modules": 8,
            "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
        },
        "workspace": {
            "name": "iris_gw_0_1",
            "num_slots": 8,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.85,
            "empty_threshold": 1.0e-3,
            "arbiter": {
                "confidence_weight": 0.25,
                "novelty_weight": 0.10,
                "relevance_weight": 0.25,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.10,
                "retention_age_penalty": 0.10,
            },
        },
        "runtime": {
            "seed": 42,
            "deterministic": True,
            "log_level": "INFO",
            "output_dir": "artifacts/runs",
        },
    }


def test_valid_configuration_models() -> None:
    hardware = HardwareConfig(
        name="cpu_smoke",
        device="cpu",
        precision="fp32",
        max_memory_gb=24.0,
        allow_gpu=False,
    )
    backbone = BackboneConfig.model_validate(valid_backbone_data())
    runtime = RuntimeConfig(
        seed=42,
        deterministic=True,
        log_level="INFO",
        output_dir="artifacts/runs",
    )

    assert hardware.device == "cpu"
    assert backbone.trust_remote_code is False
    assert runtime.seed == 42


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HardwareConfig.model_validate(
            {
                "name": "cpu_smoke",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": 24.0,
                "allow_gpu": False,
                "unexpected": True,
            }
        )


def test_negative_memory_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HardwareConfig.model_validate(
            {
                "name": "cpu_smoke",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": -1.0,
                "allow_gpu": False,
            }
        )


def test_trust_remote_code_is_rejected() -> None:
    data = valid_backbone_data()
    data["trust_remote_code"] = True

    with pytest.raises(ValidationError):
        BackboneConfig.model_validate(data)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_reduced_precision_on_cpu_is_rejected(precision: Literal["fp16", "bf16"]) -> None:
    with pytest.raises(ValidationError, match="only fp32 precision on CPU"):
        HardwareConfig.model_validate(
            {
                "name": "cpu_smoke",
                "device": "cpu",
                "precision": precision,
                "max_memory_gb": 24.0,
                "allow_gpu": False,
            }
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("output_hidden_states", False), ("use_cache", True)],
)
def test_required_backbone_runtime_flags_are_enforced(field: str, invalid_value: bool) -> None:
    data = valid_backbone_data()
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        BackboneConfig.model_validate(data)


@pytest.mark.parametrize("field", ["max_prompt_tokens", "max_new_tokens"])
def test_nonpositive_backbone_limits_are_rejected(field: str) -> None:
    limits = {"max_prompt_tokens": 256, "max_new_tokens": 32}
    limits[field] = 0

    with pytest.raises(ValidationError):
        BackboneLimitsConfig.model_validate(limits)


def test_unknown_tokenizer_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenizerConfig.model_validate(
            {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
                "unexpected": True,
            }
        )


def test_cognitive_config_accepts_layer_sampler_contract() -> None:
    config = CognitiveConfig.model_validate(
        {
            "name": "iris_gw_0_1",
            "state_dim": 256,
            "num_modules": 8,
            "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
        }
    )

    assert config.state_dim == 256
    assert config.num_modules == 8
    assert config.layer_sampler.layer_fractions == (0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        ((0.0, 0.5, 0.75, 1.0), "interval"),
        ((0.25, float("nan"), 0.75, 1.0), "interval"),
        ((0.25, 0.75, 0.5, 1.0), "strictly increasing"),
        ((0.25, 0.5, 0.75, 0.9), "must equal 1.0"),
    ],
)
def test_layer_sampler_config_rejects_invalid_depths(
    fractions: tuple[float, float, float, float], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        LayerSamplerConfig.model_validate({"layer_fractions": fractions})


def test_cognitive_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CognitiveConfig.model_validate(
            {
                "name": "iris_gw_0_1",
                "state_dim": 256,
                "num_modules": 8,
                "layer_sampler": {
                    "layer_fractions": [0.25, 0.5, 0.75, 1.0],
                    "workspace_slots": 8,
                },
            }
        )


def test_layer_sampler_config_requires_exactly_four_depths() -> None:
    with pytest.raises(ValidationError):
        LayerSamplerConfig.model_validate({"layer_fractions": [0.25, 0.5, 1.0]})


def test_cognitive_config_rejects_nonpositive_state_dimension() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        CognitiveConfig.model_validate(
            {
                "name": "iris_gw_0_1",
                "state_dim": 0,
                "num_modules": 8,
                "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
            }
        )


def test_cognitive_config_rejects_nonpositive_module_count() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        CognitiveConfig.model_validate(
            {
                "name": "iris_gw_0_1",
                "state_dim": 256,
                "num_modules": 0,
                "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
            }
        )


def test_workspace_config_accepts_iris_gw_contract() -> None:
    config = WorkspaceConfig.model_validate(
        {
            "name": "iris_gw_0_1",
            "num_slots": 8,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.85,
            "empty_threshold": 1.0e-3,
            "arbiter": {
                "confidence_weight": 0.25,
                "novelty_weight": 0.10,
                "relevance_weight": 0.25,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )

    assert config.num_slots == 8
    assert config.cycles == 3
    assert config.winners_per_cycle == 2
    assert config.arbiter.confidence_weight == 0.25
    assert config.allocator.empty_bonus == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_slots", 0),
        ("cycles", 0),
        ("winners_per_cycle", 0),
        ("decay_rate", 0.0),
        ("decay_rate", 1.0),
        ("decay_rate", float("nan")),
        ("decay_rate", float("inf")),
        ("decay_rate", float("-inf")),
        ("empty_threshold", 0.0),
        ("empty_threshold", float("nan")),
        ("empty_threshold", float("inf")),
        ("empty_threshold", float("-inf")),
    ],
)
def test_workspace_config_rejects_invalid_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "name": "iris_gw_0_1",
        "num_slots": 8,
        "cycles": 3,
        "winners_per_cycle": 2,
        "decay_rate": 0.85,
        "empty_threshold": 1.0e-3,
        "arbiter": {
            "confidence_weight": 0.25,
            "novelty_weight": 0.10,
            "relevance_weight": 0.25,
        },
        "allocator": {
            "empty_bonus": 1.0,
            "allocation_age_penalty": 0.1,
            "retention_age_penalty": 0.1,
        },
    }
    values[field] = value

    with pytest.raises(ValidationError):
        WorkspaceConfig.model_validate(values)


def test_workspace_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WorkspaceConfig.model_validate(
            {
                "name": "iris_gw_0_1",
                "num_slots": 8,
                "cycles": 3,
                "winners_per_cycle": 2,
                "decay_rate": 0.85,
                "empty_threshold": 1.0e-3,
                "arbiter": {
                    "confidence_weight": 0.25,
                    "novelty_weight": 0.10,
                    "relevance_weight": 0.25,
                },
                "allocator": {
                    "empty_bonus": 1.0,
                    "allocation_age_penalty": 0.1,
                    "retention_age_penalty": 0.1,
                },
                "fast_weights": True,
            }
        )


def test_workspace_config_rejects_more_winners_than_slots() -> None:
    with pytest.raises(ValidationError, match="cannot exceed num_slots"):
        WorkspaceConfig.model_validate(
            {
                "name": "iris_gw_0_1",
                "num_slots": 1,
                "cycles": 3,
                "winners_per_cycle": 2,
                "decay_rate": 0.85,
                "empty_threshold": 1.0e-3,
                "arbiter": {
                    "confidence_weight": 0.25,
                    "novelty_weight": 0.10,
                    "relevance_weight": 0.25,
                },
                "allocator": {
                    "empty_bonus": 1.0,
                    "allocation_age_penalty": 0.1,
                    "retention_age_penalty": 0.1,
                },
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_weight", -0.1),
        ("confidence_weight", True),
        ("novelty_weight", float("nan")),
        ("relevance_weight", float("inf")),
    ],
)
def test_workspace_arbiter_config_rejects_invalid_weights(field: str, value: object) -> None:
    values: dict[str, object] = {
        "confidence_weight": 0.25,
        "novelty_weight": 0.10,
        "relevance_weight": 0.25,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        WorkspaceArbiterConfig.model_validate(values)


def test_workspace_arbiter_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WorkspaceArbiterConfig.model_validate(
            {
                "confidence_weight": 0.25,
                "novelty_weight": 0.10,
                "relevance_weight": 0.25,
                "gumbel_noise": True,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("empty_bonus", -0.1),
        ("empty_bonus", True),
        ("allocation_age_penalty", float("nan")),
        ("retention_age_penalty", float("inf")),
    ],
)
def test_slot_allocator_config_rejects_invalid_weights(field: str, value: object) -> None:
    values: dict[str, object] = {
        "empty_bonus": 1.0,
        "allocation_age_penalty": 0.1,
        "retention_age_penalty": 0.1,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        SlotAllocatorConfig.model_validate(values)


def test_slot_allocator_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SlotAllocatorConfig.model_validate(
            {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
                "dynamic_routing": True,
            }
        )


@pytest.mark.parametrize("max_memory_gb", [float("nan"), float("inf"), float("-inf")])
def test_hardware_config_rejects_nonfinite_memory(max_memory_gb: float) -> None:
    values = {
        "name": "cpu_smoke",
        "device": "cpu",
        "precision": "fp32",
        "max_memory_gb": max_memory_gb,
        "allow_gpu": False,
    }

    with pytest.raises(ValidationError):
        HardwareConfig.model_validate(values)


def test_hardware_config_requires_explicit_gpu_permission_for_cuda() -> None:
    with pytest.raises(ValidationError, match="allow_gpu=true"):
        HardwareConfig.model_validate(
            {
                "name": "cuda_test",
                "device": "cuda",
                "precision": "fp16",
                "max_memory_gb": 8.0,
                "allow_gpu": False,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 2**63),
        ("deterministic", 1),
        ("log_level", "info"),
        ("output_dir", "   "),
    ],
)
def test_runtime_config_rejects_ambiguous_or_out_of_policy_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "seed": 42,
        "deterministic": True,
        "log_level": "INFO",
        "output_dir": "artifacts/runs",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(values)


def test_bootstrap_root_accepts_the_complete_cross_component_contract() -> None:
    values = valid_bootstrap_data()
    backbone = cast(dict[str, object], values["backbone"]).copy()
    backbone["revision"] = "a" * 40
    backbone["tokenizer_revision"] = "b" * 40
    values["backbone"] = backbone
    config = IrisBootstrapConfig.model_validate(values)

    assert config.hardware.precision == config.backbone.dtype == "fp32"
    assert config.workspace.winners_per_cycle <= config.cognitive.num_modules
    assert config.backbone.revision == "a" * 40
    assert config.backbone.tokenizer_revision == "b" * 40


@pytest.mark.parametrize(
    ("field", "revision"),
    [
        ("revision", "main"),
        ("revision", "a" * 39),
        ("revision", "g" * 40),
        ("tokenizer_revision", "A" * 40),
        ("tokenizer_revision", "0123456789abcdef" * 3),
    ],
)
def test_bootstrap_root_rejects_unpinned_backbone_revisions(
    field: str,
    revision: str,
) -> None:
    values = valid_bootstrap_data()
    backbone = cast(dict[str, object], values["backbone"]).copy()
    backbone[field] = revision
    values["backbone"] = backbone

    with pytest.raises(ValidationError, match=f"backbone {field} must be an immutable"):
        IrisBootstrapConfig.model_validate(values)


def test_bootstrap_root_rejects_hardware_backbone_precision_mismatch() -> None:
    values = valid_bootstrap_data()
    backbone = cast(dict[str, object], values["backbone"]).copy()
    backbone["dtype"] = "bf16"
    values["backbone"] = backbone

    with pytest.raises(ValidationError, match="precision must match"):
        IrisBootstrapConfig.model_validate(values)


def test_bootstrap_root_rejects_more_winners_than_cognitive_modules() -> None:
    values = valid_bootstrap_data()
    cognitive = cast(dict[str, object], values["cognitive"]).copy()
    cognitive["num_modules"] = 1
    values["cognitive"] = cognitive

    with pytest.raises(ValidationError, match="cannot exceed cognitive num_modules"):
        IrisBootstrapConfig.model_validate(values)


def test_bootstrap_root_rejects_unknown_top_level_fields() -> None:
    values = valid_bootstrap_data()
    values["agents"] = {"enabled": True}

    with pytest.raises(ValidationError, match="extra_forbidden"):
        IrisBootstrapConfig.model_validate(values)
