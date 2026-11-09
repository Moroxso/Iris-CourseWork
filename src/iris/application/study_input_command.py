"""Packaged, offline-only entry point for full study input budget diagnostics."""

from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from iris.application.errors import IrisStudyInputPreflightError
from iris.application.study_benchmark_io import read_study_benchmark_v1
from iris.application.study_input_preflight import (
    IrisStudyInputPreflightV1,
    preflight_study_inputs_v1,
)
from iris.schemas import IrisBootstrapConfig

type StudyBackboneProfileV1 = Literal["qwen3_0_6b", "qwen3_1_7b"]


def preflight_study_input_file_v1(
    path: Path,
    *,
    terminal_token_id: int,
    backbone_profile: StudyBackboneProfileV1,
    max_prompt_tokens: int | None = None,
    max_new_tokens: int | None = None,
) -> IrisStudyInputPreflightV1:
    """Read the private benchmark and inspect a declared packaged configuration."""
    try:
        if type(backbone_profile) is not str or backbone_profile not in (
            "qwen3_0_6b",
            "qwen3_1_7b",
        ):
            raise ValueError("unknown backbone profile")
        overrides = [f"backbone={backbone_profile}", "backbone.local_files_only=true"]
        for name, value in (
            ("max_prompt_tokens", max_prompt_tokens),
            ("max_new_tokens", max_new_tokens),
        ):
            if value is not None:
                if type(value) is not int or value <= 0:
                    raise ValueError("token budget must be an exact positive integer")
                overrides.append(f"backbone.limits.{name}={value}")
        benchmark = read_study_benchmark_v1(path)
        with initialize_config_module(version_base=None, config_module="iris.configs"):
            composed = compose(config_name="config", overrides=overrides)
        config = IrisBootstrapConfig.model_validate(
            OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
        )
        return preflight_study_inputs_v1(
            benchmark, config=config, terminal_token_id=terminal_token_id
        )
    except Exception:
        raise IrisStudyInputPreflightError("Iris study input command could not complete") from None


__all__ = ["StudyBackboneProfileV1", "preflight_study_input_file_v1"]
