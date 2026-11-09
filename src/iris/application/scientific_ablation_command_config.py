"""Distinct offline cognitive-ablation operator recipe, not execution authority.

Reading this bounded recipe never loads weights, opens its corpora, restores an
owner, advances a lifecycle or approves a scientific launch. The original baseline
command and its journal types retain their meaning and are not reinterpreted.
"""

import json
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.application.errors import ApplicationError
from iris.application.scientific_checkpoint import _unique
from iris.application.scientific_command import ScientificParquetInputV1, _input_bytes
from iris.application.scientific_optimization import IrisScientificOptimizerPolicyV1
from iris.application.scientific_teacher import IrisScientificObjectivePolicyV1
from iris.artifacts.scientific_ablation_lifecycle_v1 import IrisScientificAblationLifecyclePolicyV1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.utils.hashing import canonical_json_bytes

_RECIPE_LIMIT = 128 * 1024


class IrisScientificAblationCommandConfigError(ApplicationError):
    """Content-redacted recipe rejection; no model or input execution was requested."""


class IrisScientificAblationCommandConfigV1(BaseModel):
    """Explicit purpose, cognitive mode and original numeric/objective contracts.

    These fields describe a future owned execution, not an actual checkpoint or a
    native fresh-run factory. No callbacks or native authority are serialized.
    """

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    kind: Literal["iris-scientific-ablation-command-v1"] = "iris-scientific-ablation-command-v1"
    data_scope: Literal["candidate_not_launch_approval"]
    purpose: Literal["scientific", "smoke"]
    device: Literal["cpu"] = "cpu"
    backbone: BackboneConfig
    cognitive: CognitiveConfig
    workspace: WorkspaceConfig
    ablation: CognitiveAblationPolicyV1
    train: ScientificParquetInputV1
    validation: ScientificParquetInputV1
    initialization_seed: int = Field(ge=0, le=2**63 - 1)
    probe_initialization_seed: int = Field(ge=0, le=2**63 - 1)
    run_seed: int = Field(ge=0, le=2**63 - 1)
    deterministic: bool
    terminal_token_id: int = Field(ge=0)
    objective: IrisScientificObjectivePolicyV1
    optimizer: IrisScientificOptimizerPolicyV1
    lifecycle: IrisScientificAblationLifecyclePolicyV1
    maximum_journal_bytes: int = Field(default=128 * 1024**2, ge=1, le=1024**3)

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        backbone = self.backbone
        if (
            not backbone.local_files_only
            or not backbone.tokenizer.use_fast
            or backbone.dtype != "fp32"
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", backbone.repository
            )
            is None
            or any(
                re.fullmatch(r"[0-9a-f]{40}", revision) is None
                for revision in (backbone.revision, backbone.tokenizer_revision)
            )
        ):
            raise ValueError("CPU/FP32, fast tokenizer and offline pinned HF repository required")
        if self.workspace.winners_per_cycle > self.cognitive.num_modules:
            raise ValueError("Workspace winners exceed cognitive modules")
        if self.purpose != self.optimizer.purpose or self.purpose != self.lifecycle.purpose:
            raise ValueError("Recipe, optimizer and ablation lifecycle purposes differ")
        if self.validation.maximum_rows > 5000:
            raise ValueError("Validation row budget exceeds 5000")
        if self.purpose == "scientific" and (
            self.train.maximum_rows != 50000 or self.validation.maximum_rows != 5000
        ):
            raise ValueError("Scientific-size policy requires 50000/5000 input budgets")
        if self.purpose == "scientific" and (
            self.lifecycle.validate_every_steps != 500
            or self.lifecycle.checkpoint_every_steps != 1000
            or self.lifecycle.early_stopping_patience != 8
        ):
            raise ValueError("Scientific lifecycle requires source cadence500/1000 and patience8")
        return self


def read_scientific_ablation_command_config_v1(path: Path) -> IrisScientificAblationCommandConfigV1:
    """Read human-formatted local JSON, bounded before parsing, without coercion.

    Reuses only existing local regular-file IO and the common Parquet input shape;
    no baseline command is created. Paths inside the recipe are not accessed here.
    """
    try:
        raw = json.loads(
            _input_bytes(path, _RECIPE_LIMIT).decode("utf-8"), object_pairs_hook=_unique
        )
        return IrisScientificAblationCommandConfigV1.model_validate_json(
            canonical_json_bytes(raw), strict=True
        )
    except Exception:
        raise IrisScientificAblationCommandConfigError(
            "Ablation command configuration is invalid; check the local recipe. "
            "No model or corpus was loaded."
        ) from None


__all__ = [
    "IrisScientificAblationCommandConfigError",
    "IrisScientificAblationCommandConfigV1",
    "read_scientific_ablation_command_config_v1",
]
