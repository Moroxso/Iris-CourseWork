"""Offline packaged-command flow over the real arm-labelled v2 runner."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import iris.application.arm_experiment_command as command
from iris.application.arm_training_checkpoint import IrisTeacherArmId
from iris.application.errors import IrisExperimentCommandError
from iris.application.experiment_log_v2 import read_experiment_log_v2
from iris.application.experiment_manifest import write_experiment_manifest_v1
from iris.artifacts.experiment_log_v2 import (
    ExperimentArmBCheckpointReferenceV2,
    ExperimentArmBModelConfigurationV2,
    ExperimentArmDCheckpointReferenceV2,
    ExperimentArmDModelConfigurationV2,
    ExperimentCheckpointEventV2,
)
from iris.schemas import CognitiveConfig, IrisBootstrapConfig
from iris.training import IrisAdamWConfig
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state
from tests.integration.test_experiment_iteration import _manifest
from tests.integration.test_teacher_forcing_sidecar import (
    _cognitive_config,
    _TinyFrozenCausalLM,
    _TinyTokenizer,
    _workspace_config,
)

pytestmark = pytest.mark.integration

_SEED = 1907
_TERMINAL_TOKEN_ID = 15
_MODEL_REVISION = "1" * 40
_TOKENIZER_REVISION = "2" * 40


def _optimizer_config() -> IrisAdamWConfig:
    return IrisAdamWConfig(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        max_gradient_norm=1.0,
    )


def _arm_cognitive_config(arm_id: IrisTeacherArmId) -> CognitiveConfig:
    values = _cognitive_config().model_dump(mode="python")
    if arm_id == "B":
        # Five prompt tokens, two module latents, and three target tokens exactly
        # fill the tiny backbone's ten-position context window.
        values["num_modules"] = 2
    return CognitiveConfig.model_validate(values)


def _offline_config(arm_id: IrisTeacherArmId) -> IrisBootstrapConfig:
    return IrisBootstrapConfig.model_validate(
        {
            "project": {"name": "iris-offline-command-flow", "version": "0.1.0"},
            "hardware": {
                "name": "offline-cpu",
                "device": "cpu",
                "precision": "fp32",
                "max_memory_gb": 1.0,
                "allow_gpu": False,
            },
            "backbone": {
                "name": "tiny-frozen",
                "repository": "offline/tiny",
                "revision": _MODEL_REVISION,
                "tokenizer_revision": _TOKENIZER_REVISION,
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
            },
            "cognitive": _arm_cognitive_config(arm_id).model_dump(mode="python"),
            "workspace": _workspace_config().model_dump(mode="python"),
            "runtime": {
                "seed": _SEED,
                "deterministic": True,
                "log_level": "INFO",
                "output_dir": "unused-offline-output",
            },
        }
    )


def _composed_offline_config(arm_id: IrisTeacherArmId) -> DictConfig:
    composed = OmegaConf.create(_offline_config(arm_id).model_dump(mode="python"))
    if not isinstance(composed, DictConfig):
        raise AssertionError("offline root configuration must be a DictConfig")
    return composed


def _load_tokenizer(*_: object, **kwargs: object) -> _TinyTokenizer:
    assert kwargs["pretrained_model_name_or_path"] == "offline/tiny"
    assert kwargs["revision"] == _TOKENIZER_REVISION
    assert kwargs["trust_remote_code"] is False
    assert kwargs["local_files_only"] is True
    return _TinyTokenizer()


def _load_model(*_: object, **kwargs: object) -> _TinyFrozenCausalLM:
    assert kwargs["pretrained_model_name_or_path"] == "offline/tiny"
    assert kwargs["revision"] == _MODEL_REVISION
    assert kwargs["trust_remote_code"] is False
    assert kwargs["local_files_only"] is True
    assert kwargs["use_safetensors"] is True
    return _TinyFrozenCausalLM()


def _assert_runtime_restored(expected: RuntimeStateSnapshot) -> None:
    observed = capture_runtime_state(torch.device("cpu"))
    assert observed.python_rng_state == expected.python_rng_state
    assert torch.equal(observed.cpu_rng_state, expected.cpu_rng_state)
    assert observed.cuda_device is expected.cuda_device is None
    assert observed.cuda_rng_state is expected.cuda_rng_state is None
    assert observed.deterministic_algorithms is expected.deterministic_algorithms
    assert observed.deterministic_warn_only is expected.deterministic_warn_only
    assert observed.cudnn_deterministic is expected.cudnn_deterministic
    assert observed.cudnn_benchmark is expected.cudnn_benchmark


def _arguments(
    *,
    arm_id: IrisTeacherArmId,
    manifest_path: Path,
    root: Path,
    maximum_new_training_steps: int | None,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "manifest_path": manifest_path,
        "root": root,
        "run_id": f"packaged-flow-{arm_id.lower()}",
        "terminal_token_id": _TERMINAL_TOKEN_ID,
        "initialization_seed": _SEED,
        "epoch_count": 1,
        "evaluate_test_after_training": True,
        "optimizer_config": _optimizer_config(),
        "maximum_new_training_steps": maximum_new_training_steps,
        "allow_download": False,
    }


@pytest.mark.parametrize(
    (
        "arm_id",
        "checkpoint_version",
        "model_type",
        "checkpoint_type",
        "arm_kind",
        "request_graph",
        "teacher_graph",
    ),
    [
        (
            "B",
            2,
            ExperimentArmBModelConfigurationV2,
            ExperimentArmBCheckpointReferenceV2,
            "llm_modules_no_workspace",
            "independent_recurrent_modules_zero_memory_read_v1",
            "independent_recurrent_modules_zero_memory_read_teacher_v1",
        ),
        (
            "D",
            1,
            ExperimentArmDModelConfigurationV2,
            ExperimentArmDCheckpointReferenceV2,
            "llm_modules_limited_competitive_workspace",
            "limited_competitive_workspace_v1",
            "limited_competitive_workspace_teacher_v1",
        ),
    ],
)
def test_packaged_command_pauses_resumes_and_reuses_real_v2_run_offline(
    tmp_path: Path,
    arm_id: IrisTeacherArmId,
    checkpoint_version: int,
    model_type: type[object],
    checkpoint_type: type[object],
    arm_kind: str,
    request_graph: str,
    teacher_graph: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_experiment_manifest_v1(_manifest(), path=manifest_path)
    root = tmp_path / "runs"
    runtime_before = capture_runtime_state(torch.device("cpu"))

    def compose_config(
        *,
        allow_download: bool,
        initialization_seed: int,
    ) -> DictConfig:
        assert allow_download is False
        assert initialization_seed == _SEED
        return _composed_offline_config(arm_id)

    try:
        with (
            patch.object(
                command,
                "_compose_experiment_config",
                side_effect=compose_config,
            ) as compose,
            patch(
                "iris.backbone.adapter.AutoTokenizer.from_pretrained",
                side_effect=_load_tokenizer,
            ) as load_tokenizer,
            patch(
                "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
                side_effect=_load_model,
            ) as load_model,
        ):
            partial = command.run_packaged_arm_experiment_v2(
                **_arguments(
                    arm_id=arm_id,
                    manifest_path=manifest_path,
                    root=root,
                    maximum_new_training_steps=1,
                )
            )
            resumed = command.run_packaged_arm_experiment_v2(
                **_arguments(
                    arm_id=arm_id,
                    manifest_path=manifest_path,
                    root=root,
                    maximum_new_training_steps=None,
                )
            )
            complete_snapshot = read_experiment_log_v2(resumed.path)
            reused = command.run_packaged_arm_experiment_v2(
                **_arguments(
                    arm_id=arm_id,
                    manifest_path=manifest_path,
                    root=root,
                    maximum_new_training_steps=None,
                )
            )

        reused_snapshot = read_experiment_log_v2(reused.path)

        assert partial.arm_id == resumed.arm_id == reused.arm_id == arm_id
        assert partial.path == resumed.path == reused.path
        assert partial.status == "step_limit_reached"
        assert partial.event_count == 1
        assert partial.optimizer_completed_steps == 1
        assert partial.replayed_training_steps == 0
        assert partial.new_training_steps == 1
        assert partial.next_epoch_index == 1
        assert partial.next_training_ordinal == 2
        assert partial.latest_checkpoint_schema_version == checkpoint_version

        assert resumed.status == "complete"
        assert resumed.event_count == 5
        assert resumed.optimizer_completed_steps == 2
        assert resumed.completed_epochs == 1
        assert resumed.replayed_training_steps == 1
        assert resumed.new_training_steps == 1
        assert resumed.next_epoch_index is None
        assert resumed.next_training_ordinal is None
        assert resumed.latest_checkpoint_schema_version == checkpoint_version

        assert reused.status == "complete"
        assert reused.event_count == resumed.event_count
        assert reused.optimizer_completed_steps == resumed.optimizer_completed_steps
        assert reused.replayed_training_steps == 0
        assert reused.new_training_steps == 0
        assert reused.run_payload_sha256 == resumed.run_payload_sha256
        assert reused.configuration_sha256 == resumed.configuration_sha256
        assert reused.latest_checkpoint_payload_sha256 == resumed.latest_checkpoint_payload_sha256
        assert reused_snapshot.run == complete_snapshot.run
        assert reused_snapshot.events == complete_snapshot.events

        run = complete_snapshot.run.payload
        assert run.arm_id == arm_id
        assert run.arm_kind == arm_kind
        assert run.request_graph == request_graph
        assert run.teacher_graph == teacher_graph
        assert type(run.configuration.model) is model_type
        owned_model = run.configuration.model
        assert owned_model.backbone == _offline_config(arm_id).backbone
        assert owned_model.cognitive == _arm_cognitive_config(arm_id)
        assert run.configuration.runtime_policy.initialization_seed == _SEED
        assert run.configuration.runtime_policy.deterministic is True

        checkpoint_record = complete_snapshot.events[2].payload.record
        assert type(checkpoint_record) is ExperimentCheckpointEventV2
        checkpoint = checkpoint_record.checkpoint
        assert type(checkpoint) is checkpoint_type
        assert checkpoint.arm_id == arm_id
        assert checkpoint.checkpoint_schema_version == checkpoint_version
        assert checkpoint.payload_sha256 == resumed.latest_checkpoint_payload_sha256
        assert (root / "teacher" / f"v{checkpoint_version}" / checkpoint.payload_sha256).is_dir()

        assert compose.call_count == 3
        assert load_tokenizer.call_count == 3
        assert load_model.call_count == 3
        _assert_runtime_restored(runtime_before)
    finally:
        # Preserve isolation if an assertion interrupts the postconditions.
        random.setstate(runtime_before.python_rng_state)
        torch.set_rng_state(runtime_before.cpu_rng_state)
        torch.use_deterministic_algorithms(
            runtime_before.deterministic_algorithms,
            warn_only=runtime_before.deterministic_warn_only,
        )
        torch.backends.cudnn.deterministic = runtime_before.cudnn_deterministic
        torch.backends.cudnn.benchmark = runtime_before.cudnn_benchmark


def test_packaged_command_rejects_cross_arm_run_before_second_backbone_load(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_experiment_manifest_v1(_manifest(), path=manifest_path)
    root = tmp_path / "runs"
    selected_arm: IrisTeacherArmId = "B"

    def compose_config(
        *,
        allow_download: bool,
        initialization_seed: int,
    ) -> DictConfig:
        assert allow_download is False
        assert initialization_seed == _SEED
        return _composed_offline_config(selected_arm)

    with (
        patch.object(command, "_compose_experiment_config", side_effect=compose_config),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=_load_tokenizer,
        ) as load_tokenizer,
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=_load_model,
        ) as load_model,
    ):
        first_arguments = _arguments(
            arm_id="B",
            manifest_path=manifest_path,
            root=root,
            maximum_new_training_steps=1,
        )
        first = command.run_packaged_arm_experiment_v2(**first_arguments)
        selected_arm = "D"
        conflicting_arguments = _arguments(
            arm_id="D",
            manifest_path=manifest_path,
            root=root,
            maximum_new_training_steps=1,
        )
        conflicting_arguments["run_id"] = first_arguments["run_id"]

        with pytest.raises(
            IrisExperimentCommandError,
            match="existing-run preflight",
        ) as captured:
            command.run_packaged_arm_experiment_v2(**conflicting_arguments)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "identity conflicts" in str(captured.value.__cause__)
    assert first.arm_id == "B"
    assert load_tokenizer.call_count == 1
    assert load_model.call_count == 1
