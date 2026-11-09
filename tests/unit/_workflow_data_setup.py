"""Constructible DATA fixtures only. These never claim a native run happened."""

from pathlib import Path

from iris.application.scientific_ablation_workflow_io import (
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_collapse_v1 import ScientificAblationCollapsePolicyV1
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    ScientificAblationWorkflowParquetV1,
    ScientificAblationWorkflowResourcesV1,
)
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.interventions.policy import WorkspaceInterventionPlanV1


def intent(
    root: Path, controls: tuple[WorkspaceInterventionPlanV1, ...] = ()
) -> ScientificAblationWorkflowIntentV1:
    def parquet(name: str) -> ScientificAblationWorkflowParquetV1:
        return ScientificAblationWorkflowParquetV1(
            path=name,
            sha256="0" * 64,
            maximum_rows=4,
            maximum_file_bytes=1024**2,
            maximum_logical_bytes=1024**2,
        )

    return ScientificAblationWorkflowIntentV1(
        source_sha256="0" * 64,
        lifecycle_sha256="1" * 64,
        inference_sha256="2" * 64,
        selected_checkpoint_sha256="3" * 64,
        lifecycle_path=str(root / "original-lifecycle"),
        inference_path=str(root / "original-inference"),
        inputs_root=str(root / "original-inputs"),
        training=parquet("train.parquet"),
        validation=parquet("validation.parquet"),
        test=parquet("test.parquet"),
        decoding=ScientificDecodingPolicyV1(strategy="greedy", seed=42),
        controls=controls,
        collapse=ScientificAblationCollapsePolicyV1(),
        workspace=ScientificAblationWorkspaceCausalPolicyV1(selection_seed=42, shuffle_seed=1337),
        message=ScientificAblationMessageCausalPolicyV1(selection_seed=42, shuffle_seed=1337),
        resources=ScientificAblationWorkflowResourcesV1(
            maximum_storage_bytes=16 * 1024**2,
            minimum_free_bytes=0,
        ),
        python_policy=None,
    )


def plan(root: Path) -> IrisScientificAblationWorkflowPlanV1:
    return build_scientific_ablation_workflow_plan_v1(intent(root))
