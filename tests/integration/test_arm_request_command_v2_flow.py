"""Offline end-to-end delivery for packaged A/B/D request artifact v2."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

import iris.application.arm_request_command as command
from iris.application.errors import IrisArmExecutionProfileError
from iris.application.request_artifact_v2 import read_request_artifact_v2
from iris.generation import TemperatureTopPSamplingConfig
from tests.unit.test_request_artifact_v2 import SECRET, _inputs, _sidecar_patch


@pytest.mark.parametrize("arm_id", ["A", "B", "D"])
def test_command_executes_once_and_installs_one_private_v2_artifact(
    tmp_path: Path,
    arm_id: str,
) -> None:
    runtime, execution, _ = _inputs(tmp_path, arm_id)
    orchestrator_type = type(runtime.orchestrator)
    policy = TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875)

    with (
        patch.object(command, "_compose_request_config", return_value=OmegaConf.create({})),
        patch.object(command, "bootstrap_arm_request_runtime_v2", return_value=runtime),
        patch.object(orchestrator_type, "execute", return_value=execution) as execute,
        _sidecar_patch(arm_id),
    ):
        first = command.generate_arm_request_v2(
            arm_id=arm_id,
            prompt=SECRET,
            strategy="greedy",
            stop_token_ids=(2, 7),
            sampling=policy,
            sampling_seed=None,
            allow_download=False,
        )
        second = command.generate_arm_request_v2(
            arm_id=arm_id,
            prompt=SECRET,
            strategy="greedy",
            stop_token_ids=(2, 7),
            sampling=policy,
            sampling_seed=None,
            allow_download=False,
        )

    assert first is execution.generation
    assert second is first
    assert execute.call_count == 2
    paths = list((tmp_path / "artifacts" / "requests" / "v2").glob("*.json"))
    assert len(paths) == 1
    content = paths[0].read_bytes()
    assert SECRET.encode() not in content
    artifact = read_request_artifact_v2(paths[0])
    assert artifact.payload.topology.arm_id == arm_id
    assert artifact.payload.request.prompt.representation == "sha256_only"
    assert paths[0].stem == artifact.payload_sha256


def test_blocked_c_creates_no_runtime_or_artifact_directory(tmp_path: Path) -> None:
    with (
        patch.object(command, "_compose_request_config") as compose,
        patch.object(command, "bootstrap_arm_request_runtime_v2") as bootstrap,
        pytest.raises(IrisArmExecutionProfileError, match="arm C request graph is blocked"),
    ):
        command.generate_arm_request_v2(
            arm_id="C",
            prompt="hello",
            strategy="greedy",
            stop_token_ids=(2, 7),
            sampling=TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875),
            sampling_seed=None,
            allow_download=False,
        )

    compose.assert_not_called()
    bootstrap.assert_not_called()
    assert list(tmp_path.iterdir()) == []
