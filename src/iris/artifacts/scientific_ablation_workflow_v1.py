"""Closed compound-workflow DATA intent and receipts, never native-owner authority."""

from pathlib import PureWindowsPath
from typing import Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import MAX_REPORT_BYTES
from iris.artifacts.scientific_ablation_collapse_v1 import ScientificAblationCollapsePolicyV1
from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.interventions.policy import WorkspaceInterventionPlanV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256

WORKFLOW_PLAN_DOMAIN = "iris/scientific-ablation-native-workflow-plan/v1"
WORKFLOW_DATA_DOMAIN = "iris/scientific-ablation-native-workflow-phase-data/v1"
WORKFLOW_ATTEMPT_DOMAIN = "iris/scientific-ablation-native-workflow-phase-attempt/v1"

WorkflowPhaseKind = Literal["capture", "score", "diagnostics", "workspace", "message", "publish"]
WorkflowFileName = Literal[
    "capture.json",
    "evaluation.json",
    "ordinary.json",
    "collapse.json",
    "source.json",
    "report.json",
    "runtime.json",
    "summary.json",
]


class ScientificAblationWorkflowParquetV1(_Strict):
    path: str = Field(min_length=1, max_length=4096, repr=False)
    sha256: Digest
    maximum_rows: int = Field(ge=1, le=50000)
    maximum_file_bytes: int = Field(ge=1, le=512 * 1024**2)
    maximum_logical_bytes: int = Field(ge=1, le=512 * 1024**2)

    @model_validator(mode="after")
    def local_file(self) -> Self:
        path = PureWindowsPath(self.path)
        if (
            not self.path.strip()
            or "\0" in self.path
            or "://" in self.path
            or self.path.startswith(("//", "\\\\"))
            or (path.drive and not path.root)
        ):
            raise ValueError("Explicit local original Parquet specification required")
        return self


class ScientificAblationWorkflowResourcesV1(_Strict):
    """Ceilings, not allocations. Whole storage is explicitly chosen, never inferred."""

    maximum_storage_bytes: int = Field(ge=1, le=16 * 1024**3)
    minimum_free_bytes: int = Field(ge=0, le=16 * 1024**3)
    maximum_kernel_storage_bytes: int = Field(default=512 * 1024**2, ge=1, le=16 * 1024**3)
    maximum_observation_scalars: int = Field(default=2_000_000, ge=1, le=2_000_000)
    maximum_writer_report_bytes: int = Field(default=MAX_REPORT_BYTES, ge=1, le=MAX_REPORT_BYTES)


class ScientificAblationWorkflowIntentV1(_Strict):
    kind: Literal["iris-scientific-ablation-native-workflow-intent-v1"] = (
        "iris-scientific-ablation-native-workflow-intent-v1"
    )
    source_sha256: Digest
    lifecycle_sha256: Digest
    inference_sha256: Digest
    selected_checkpoint_sha256: Digest
    lifecycle_path: str = Field(min_length=1, max_length=4096, repr=False)
    inference_path: str = Field(min_length=1, max_length=4096, repr=False)
    inputs_root: str = Field(min_length=1, max_length=4096, repr=False)
    training: ScientificAblationWorkflowParquetV1
    validation: ScientificAblationWorkflowParquetV1
    test: ScientificAblationWorkflowParquetV1
    decoding: ScientificDecodingPolicyV1
    controls: tuple[WorkspaceInterventionPlanV1, ...] = Field(max_length=64)
    collapse: ScientificAblationCollapsePolicyV1
    workspace: ScientificAblationWorkspaceCausalPolicyV1
    message: ScientificAblationMessageCausalPolicyV1
    resources: ScientificAblationWorkflowResourcesV1
    python_policy: PythonSandboxPolicyV2 | None
    scientific_acceptance: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def distinct_control_intents(self) -> Self:
        keys = [canonical_json_bytes(plan.model_dump(mode="json")) for plan in self.controls]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate controlled streams are not distinct workflow phases")
        for path in (self.lifecycle_path, self.inference_path, self.inputs_root):
            windows = PureWindowsPath(path)
            if (
                not path.strip()
                or "\0" in path
                or "://" in path
                or path.startswith(("//", "\\\\"))
                or (windows.drive and not windows.root)
            ):
                raise ValueError("Explicit local source roots required")
        return self


class IrisScientificAblationWorkflowPlanV1(_Strict):
    sha256: Digest
    payload: ScientificAblationWorkflowIntentV1

    @model_validator(mode="after")
    def matches_data_intent(self) -> Self:
        if self.sha256 != domain_sha256(
            WORKFLOW_PLAN_DOMAIN, canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Workflow DATA intent checksum differs")
        return self


class ScientificAblationWorkflowPhaseV1(_Strict):
    index: int = Field(ge=0, le=133)
    name: str = Field(
        pattern=r"^(ordinary|control-[0-9]{2})-(capture|score)$|^(diagnostics|workspace|message|publish)$"
    )
    kind: WorkflowPhaseKind
    stream_index: int | None = Field(ge=0, le=63)

    @model_validator(mode="after")
    def named_kind(self) -> Self:
        if self.kind in ("capture", "score"):
            expected = (
                "ordinary" if self.stream_index is None else f"control-{self.stream_index:02d}"
            )
            if self.name != f"{expected}-{self.kind}":
                raise ValueError("Exact original stream/phase name required")
        elif self.name != self.kind or self.stream_index is not None:
            raise ValueError("Native/publication phase is not a controlled stream")
        return self


class ScientificAblationWorkflowFileV1(_Strict):
    name: WorkflowFileName
    sha256: Digest
    bytes: int = Field(ge=1, le=MAX_REPORT_BYTES)


class ScientificAblationWorkflowPhasePayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-native-workflow-phase-data-v1"] = (
        "iris-scientific-ablation-native-workflow-phase-data-v1"
    )
    plan_sha256: Digest
    phase: ScientificAblationWorkflowPhaseV1
    predecessor_sha256: Digest | None
    origin: Literal[
        "kernel_data",
        "original_native_result_data",
        "not_applicable_no_next_cycle",
        "publication_data",
    ]
    files: tuple[ScientificAblationWorkflowFileV1, ...] = Field(min_length=1, max_length=3)
    native_execution_authority: Literal["none_data_only"] = "none_data_only"

    @model_validator(mode="after")
    def phase_files(self) -> Self:
        expected = {
            "capture": {"capture.json"},
            "score": {"evaluation.json"},
            "diagnostics": {"ordinary.json", "collapse.json"},
            "workspace": {"source.json", "report.json", "runtime.json"},
            "message": {"source.json", "report.json", "runtime.json"},
            "publish": {"summary.json"},
        }[self.phase.kind]
        if self.origin == "not_applicable_no_next_cycle":
            if self.phase.kind not in ("workspace", "message"):
                raise ValueError("Only writer next-cycle endpoints may be inapplicable")
            expected = {"summary.json"}
        elif (
            self.origin
            != {
                "capture": "kernel_data",
                "score": "kernel_data",
                "diagnostics": "original_native_result_data",
                "workspace": "original_native_result_data",
                "message": "original_native_result_data",
                "publish": "publication_data",
            }[self.phase.kind]
        ):
            raise ValueError("Phase DATA origin is inconsistent")
        if (
            len({file.name for file in self.files}) != len(self.files)
            or {file.name for file in self.files} != expected
        ):
            raise ValueError("Exact complete phase files required")
        if (self.phase.index == 0) != (self.predecessor_sha256 is None):
            raise ValueError("Original ordered phase chain required")
        return self


class IrisScientificAblationWorkflowPhaseDataV1(_Strict):
    sha256: Digest
    payload: ScientificAblationWorkflowPhasePayloadV1

    @model_validator(mode="after")
    def data_digest(self) -> Self:
        if self.sha256 != domain_sha256(
            WORKFLOW_DATA_DOMAIN, canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Workflow ready DATA checksum differs")
        return self


class ScientificAblationWorkflowAttemptPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-native-workflow-phase-attempt-v1"] = (
        "iris-scientific-ablation-native-workflow-phase-attempt-v1"
    )
    plan_sha256: Digest
    phase: ScientificAblationWorkflowPhaseV1
    predecessor_sha256: Digest | None
    recovery: Literal[
        "acknowledged_kernel_prefix_only", "never_auto_remeasure", "data_publication_only"
    ]


class IrisScientificAblationWorkflowAttemptV1(_Strict):
    sha256: Digest
    payload: ScientificAblationWorkflowAttemptPayloadV1

    @model_validator(mode="after")
    def attempt_digest(self) -> Self:
        recovery = (
            "acknowledged_kernel_prefix_only"
            if self.payload.phase.kind in ("capture", "score")
            else "data_publication_only"
            if self.payload.phase.kind == "publish"
            else "never_auto_remeasure"
        )
        if self.payload.recovery != recovery or self.sha256 != domain_sha256(
            WORKFLOW_ATTEMPT_DOMAIN, canonical_json_bytes(self.payload.model_dump(mode="json"))
        ):
            raise ValueError("Exact phase uncertainty/recovery intent required")
        return self


class ScientificAblationWorkflowCaptureDataV1(_Strict):
    protocol: Literal["ordinary", "controlled"]
    plan_sha256: Digest
    input_sha256: Digest
    capture_sha256: Digest
    examples: int = Field(ge=1, le=5000)


class ScientificAblationWorkflowScoreDataV1(_Strict):
    protocol: Literal["ordinary", "controlled"]
    evaluation_plan_sha256: Digest
    evaluation_sha256: Digest
    complete_sha256: Digest
    capture_plan_sha256: Digest
    capture_sha256: Digest
    examples: int = Field(ge=1, le=5000)


def scientific_ablation_workflow_phases_v1(
    plan: IrisScientificAblationWorkflowPlanV1,
) -> tuple[ScientificAblationWorkflowPhaseV1, ...]:
    phases = [
        ScientificAblationWorkflowPhaseV1(
            index=0, name="ordinary-capture", kind="capture", stream_index=None
        ),
        ScientificAblationWorkflowPhaseV1(
            index=1, name="ordinary-score", kind="score", stream_index=None
        ),
    ]
    for index in range(len(plan.payload.controls)):
        for kind in ("capture", "score"):
            phases.append(
                ScientificAblationWorkflowPhaseV1.model_validate(
                    dict(
                        index=len(phases),
                        name=f"control-{index:02d}-{kind}",
                        kind=kind,
                        stream_index=index,
                    )
                )
            )
    for kind in ("diagnostics", "workspace", "message", "publish"):
        phases.append(
            ScientificAblationWorkflowPhaseV1.model_validate(
                dict(
                    index=len(phases),
                    name=kind,
                    kind=kind,
                    stream_index=None,
                )
            )
        )
    return tuple(phases)
