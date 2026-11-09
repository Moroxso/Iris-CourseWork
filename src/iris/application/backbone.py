"""Backbone inspection application service."""

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from iris.application.errors import BackboneExecutionError
from iris.schemas import BackboneConfig
from iris.utils.devices import DeviceInfo, inspect_device


@dataclass(frozen=True, slots=True)
class BackboneReport:
    """Configuration status of the selected development backbone."""

    name: str
    configured: bool
    loaded: bool


@dataclass(frozen=True, slots=True)
class BackboneForwardReport:
    """Validated measurements from one frozen CPU backbone pass."""

    backbone: str
    revision: str
    device: str
    dtype: str
    trainable_parameters: int
    prompt_tokens: int
    vocabulary_size: int
    hidden_layers: int
    hidden_size: int
    logits_shape: tuple[int, ...]
    last_hidden_shape: tuple[int, ...]
    last_hidden_sha256: str
    finite_logits: bool
    finite_hidden_states: bool
    forward_completed: bool
    artifact_path: Path


def _load_backbone_config() -> BackboneConfig:
    config_resource = files("iris.configs").joinpath("backbone", "qwen3_0_6b.yaml")
    config = OmegaConf.create(config_resource.read_text(encoding="utf-8"))
    return BackboneConfig.model_validate(OmegaConf.to_container(config))


def inspect_backbone() -> BackboneReport:
    """Inspect packaged backbone configuration without loading model weights."""
    config = _load_backbone_config()
    return BackboneReport(name=config.name, configured=True, loaded=False)


def _create_adapter(config: BackboneConfig, device_info: DeviceInfo) -> Any:
    from iris.backbone import BackboneAdapter

    return BackboneAdapter(config, device_info)


def _tensor_sha256(tensor: Any) -> str:
    import torch

    with torch.inference_mode():
        contiguous_bytes = tensor.detach().to("cpu").contiguous().view(torch.uint8)
        return sha256(contiguous_bytes.numpy().tobytes()).hexdigest()


def _write_forward_artifact(report: BackboneForwardReport) -> None:
    payload = {
        "status": "success",
        "backbone": report.backbone,
        "revision": report.revision,
        "device": report.device,
        "dtype": report.dtype,
        "trainable_parameters": report.trainable_parameters,
        "prompt_tokens": report.prompt_tokens,
        "hidden_layers": report.hidden_layers,
        "hidden_size": report.hidden_size,
        "vocabulary_size": report.vocabulary_size,
        "logits_shape": list(report.logits_shape),
        "last_hidden_shape": list(report.last_hidden_shape),
        "last_hidden_sha256": report.last_hidden_sha256,
        "finite_logits": report.finite_logits,
        "finite_hidden_states": report.finite_hidden_states,
        "forward_completed": report.forward_completed,
    }
    report.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report.artifact_path.with_name(
        f".{report.artifact_path.name}.{uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary_path.replace(report.artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def forward_backbone(
    text: str,
    artifact_path: Path = Path("artifacts/smoke/backbone_forward.json"),
) -> BackboneForwardReport:
    """Run one real frozen CPU forward pass and persist its measurements."""
    try:
        import torch

        config = _load_backbone_config()
        device_info = inspect_device("cpu")
        adapter = _create_adapter(config, device_info)
        output = adapter.forward_prompt(text)
        trainable_parameters = adapter.trainable_parameter_count
        if trainable_parameters != 0:
            message = f"Backbone has {trainable_parameters} trainable parameters"
            raise BackboneExecutionError(message)
        if adapter.has_parameter_gradients:
            message = "Backbone parameters contain gradients after inference"
            raise BackboneExecutionError(message)

        finite_logits = bool(torch.isfinite(output.logits).all())
        finite_hidden_states = all(
            bool(torch.isfinite(state).all()) for state in output.hidden_states
        )
        if not finite_logits or not finite_hidden_states:
            message = "Backbone forward produced non-finite tensors"
            raise BackboneExecutionError(message)
        report = BackboneForwardReport(
            backbone=config.repository,
            revision=config.revision,
            device=device_info.backend,
            dtype=str(output.logits.dtype).removeprefix("torch."),
            trainable_parameters=trainable_parameters,
            prompt_tokens=int(output.attention_mask.sum().item()),
            vocabulary_size=adapter.vocabulary_size,
            hidden_layers=len(output.hidden_states),
            hidden_size=adapter.hidden_size,
            logits_shape=tuple(output.logits.shape),
            last_hidden_shape=tuple(output.hidden_states[-1].shape),
            last_hidden_sha256=_tensor_sha256(output.hidden_states[-1]),
            finite_logits=finite_logits,
            finite_hidden_states=finite_hidden_states,
            forward_completed=True,
            artifact_path=artifact_path,
        )
        _write_forward_artifact(report)
        return report
    except BackboneExecutionError:
        raise
    except Exception as error:
        message = "Backbone CPU forward did not complete"
        raise BackboneExecutionError(message) from error
