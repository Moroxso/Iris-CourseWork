"""Actual original lifecycle/best/factory at the declared cycles and ablation.

Tiny paired geometry fixtures augment, never replace, the accepted four-family
workflow or the original scientific5000. No pretrained/candidate execution.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from iris.application.scientific_ablation_generation_source import (
    read_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_inference import build_scientific_ablation_inference_v1
from iris.application.scientific_ablation_inference_io import write_scientific_ablation_inference_v1
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    read_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
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
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt
from tests.integration.test_scientific_ablation_run import _run
from tests.integration.test_scientific_generation import _test_spec
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _rows


@dataclass(frozen=True, slots=True)
class Geometry:
    cycles: Literal[1, 5]
    routing: Literal["learned", "random", "uniform"]
    goal: Literal["enabled", "disabled"]
    broadcast: Literal["enabled", "disabled"]


GEOMETRIES = (
    Geometry(1, "learned", "enabled", "enabled"),
    Geometry(1, "uniform", "disabled", "disabled"),
    Geometry(5, "learned", "enabled", "enabled"),
    Geometry(5, "random", "disabled", "disabled"),
)


@dataclass(frozen=True, slots=True)
class GeometryCase:
    root: Path
    geometry: Geometry
    plan: IrisScientificAblationWorkflowPlanV1
    factory: Callable[[], IrisScientificAblationRunV1]


def prepare(root: Path, geometry: Geometry) -> GeometryCase:
    """Build actual cycle-specific source, never relabel an old C3checkpoint."""
    root.mkdir(parents=True)

    def factory() -> IrisScientificAblationRunV1:
        return _run(
            geometry.routing,
            goal=geometry.goal,
            broadcast=geometry.broadcast,
            cycles=geometry.cycles,
            policy=_policy(minimum_target_tokens=1, max_optimizer_steps=3),
        )

    controller = _durable(
        root / "source/native",
        geometry.routing,
        goal=geometry.goal,
        broadcast=geometry.broadcast,
        cycles=geometry.cycles,
        maximum=3,
        validate=1,
        checkpoint=3,
    )
    receipt = _receipt(controller)
    lifecycle = read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    assert lifecycle.manifest.payload.best is not None
    best_sha = lifecycle.manifest.payload.best.sha256
    projected = build_scientific_ablation_inference_v1(
        lifecycle,
        expected_lifecycle_sha256=lifecycle.manifest.sha256,
        make_fresh_run=factory,
        selection="selected_best",
    )
    inference_path = write_scientific_ablation_inference_v1(
        projected,
        root=root / "source/inference",
        expected_sha256=projected.manifest.sha256,
        expected_source_checkpoint_sha256=best_sha,
        expected_lifecycle_sha256=lifecycle.manifest.sha256,
    )
    source = read_scientific_ablation_generation_source_v1(
        lifecycle_path=receipt.path,
        inference_path=inference_path,
        expected_lifecycle_sha256=lifecycle.manifest.sha256,
        expected_inference_sha256=projected.manifest.sha256,
        expected_source_checkpoint_sha256=best_sha,
    )
    assert projected.manifest.payload.model.workspace.cycles == geometry.cycles
    assert source.manifest.payload.selected_step == 1 and source.manifest.payload.terminal_step == 3
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
    )
    rows = tuple(
        row for row in pool if row.task_type == "next_tool_prediction" and row.difficulty == 1
    )[:2]
    assert len(rows) == 2 and not any(row.executable for row in rows)
    training = _test_spec(root / "train.parquet", _rows()[:5])
    validation = _test_spec(root / "validation.parquet", _validation())
    test = _test_spec(root / "test.parquet", rows)
    controls = tuple(
        WorkspaceInterventionPlanV1(interventions=(intervention,))
        for intervention in (
            WorkspaceInterventionSpecV1(kind="zero_workspace"),
            WorkspaceInterventionSpecV1(kind="shuffle_slots", seed=1337),
        )
    )
    intent = ScientificAblationWorkflowIntentV1(
        source_sha256=source.manifest.sha256,
        lifecycle_sha256=lifecycle.manifest.sha256,
        inference_sha256=projected.manifest.sha256,
        selected_checkpoint_sha256=best_sha,
        lifecycle_path=str(receipt.path),
        inference_path=str(inference_path),
        inputs_root=str(root),
        training=ScientificAblationWorkflowParquetV1.model_validate(
            training.model_dump(mode="python")
        ),
        validation=ScientificAblationWorkflowParquetV1.model_validate(
            validation.model_dump(mode="python")
        ),
        test=ScientificAblationWorkflowParquetV1.model_validate(test.model_dump(mode="python")),
        decoding=ScientificDecodingPolicyV1(strategy="greedy", seed=42),
        controls=controls,
        collapse=ScientificAblationCollapsePolicyV1(),
        workspace=ScientificAblationWorkspaceCausalPolicyV1(
            selection_seed=42,
            shuffle_seed=1337,
            examples_per_stratum=2,
            maximum_examples=2,
        ),
        message=ScientificAblationMessageCausalPolicyV1(
            selection_seed=42,
            shuffle_seed=1337,
            examples_per_stratum=2,
            maximum_examples=2,
        ),
        resources=ScientificAblationWorkflowResourcesV1(
            maximum_storage_bytes=256 * 1024**2 if geometry.cycles == 1 else 4 * 1024**3,
            maximum_kernel_storage_bytes=64 * 1024**2 if geometry.cycles == 1 else 512 * 1024**2,
            minimum_free_bytes=0,
        ),
        python_policy=None,
    )
    return GeometryCase(root, geometry, build_scientific_ablation_workflow_plan_v1(intent), factory)
