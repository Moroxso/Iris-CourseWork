import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application.backbone import forward_backbone
from iris.application.errors import BackboneExecutionError
from iris.backbone import BackboneForwardError
from iris.backbone.outputs import BackboneForwardOutput


class FakeAdapter:
    hidden_size = 4
    vocabulary_size = 8
    trainable_parameter_count = 0
    has_parameter_gradients = False

    def __init__(self, output: BackboneForwardOutput) -> None:
        self.output = output
        self.prompts: list[str] = []

    def forward_prompt(self, text: str) -> BackboneForwardOutput:
        self.prompts.append(text)
        return self.output


def valid_output() -> BackboneForwardOutput:
    return BackboneForwardOutput(
        logits=torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
        hidden_states=tuple(torch.full((1, 3, 4), value) for value in (1.0, 2.0, 3.0)),
        attention_mask=torch.tensor([[1, 1, 1]]),
    )


def run_with_adapter(adapter: Any, artifact_path: Path) -> Any:
    config = SimpleNamespace(
        repository="Qwen/Qwen3-0.6B",
        revision="a" * 40,
    )
    device_info = SimpleNamespace(backend="cpu")
    with (
        patch("iris.application.backbone._load_backbone_config", return_value=config),
        patch("iris.application.backbone.inspect_device", return_value=device_info),
        patch("iris.application.backbone._create_adapter", return_value=adapter),
    ):
        return forward_backbone("hello", artifact_path)


def test_forward_writes_reproducible_validated_artifact(tmp_path: Path) -> None:
    output = valid_output()
    adapter = FakeAdapter(output)
    artifact_path = tmp_path / "smoke" / "backbone_forward.json"

    report = run_with_adapter(adapter, artifact_path)
    expected_hash = hashlib.sha256(
        output.hidden_states[-1].view(torch.uint8).numpy().tobytes()
    ).hexdigest()

    assert adapter.prompts == ["hello"]
    assert report.backbone == "Qwen/Qwen3-0.6B"
    assert report.device == "cpu"
    assert report.dtype == "float32"
    assert report.trainable_parameters == 0
    assert report.prompt_tokens == 3
    assert report.vocabulary_size == 8
    assert report.hidden_layers == 3
    assert report.hidden_size == 4
    assert report.logits_shape == (1, 3, 8)
    assert report.last_hidden_shape == (1, 3, 4)
    assert report.last_hidden_sha256 == expected_hash
    assert report.finite_logits is True
    assert report.finite_hidden_states is True
    assert report.forward_completed is True
    assert report.artifact_path == artifact_path

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload == {
        "backbone": "Qwen/Qwen3-0.6B",
        "device": "cpu",
        "dtype": "float32",
        "finite_hidden_states": True,
        "finite_logits": True,
        "forward_completed": True,
        "hidden_layers": 3,
        "hidden_size": 4,
        "last_hidden_sha256": expected_hash,
        "last_hidden_shape": [1, 3, 4],
        "logits_shape": [1, 3, 8],
        "prompt_tokens": 3,
        "revision": "a" * 40,
        "status": "success",
        "trainable_parameters": 0,
        "vocabulary_size": 8,
    }


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("trainable_parameter_count", 7, "7 trainable"),
        ("has_parameter_gradients", True, "contain gradients"),
    ],
)
def test_forward_rejects_mutable_model_state(
    tmp_path: Path, attribute: str, value: object, message: str
) -> None:
    adapter = FakeAdapter(valid_output())
    setattr(adapter, attribute, value)
    artifact_path = tmp_path / "backbone_forward.json"

    with pytest.raises(BackboneExecutionError, match=message):
        run_with_adapter(adapter, artifact_path)

    assert not artifact_path.exists()


@pytest.mark.parametrize("target", ["logits", "hidden"])
def test_forward_rejects_nonfinite_tensors_before_writing_artifact(
    tmp_path: Path, target: str
) -> None:
    output = valid_output()
    tensor = output.logits if target == "logits" else output.hidden_states[-1]
    tensor[0, 0, 0] = torch.nan
    artifact_path = tmp_path / "backbone_forward.json"

    with pytest.raises(BackboneExecutionError, match="non-finite"):
        run_with_adapter(FakeAdapter(output), artifact_path)

    assert not artifact_path.exists()


def test_forward_preserves_adapter_failure_as_cause(tmp_path: Path) -> None:
    original = BackboneForwardError("invalid model output")
    adapter = FakeAdapter(valid_output())
    with (
        patch.object(adapter, "forward_prompt", side_effect=original),
        pytest.raises(BackboneExecutionError, match="did not complete") as raised,
    ):
        run_with_adapter(adapter, tmp_path / "backbone_forward.json")

    assert raised.value.__cause__ is original
