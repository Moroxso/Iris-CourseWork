"""Actual generalized M/K/D/C/R + all24C1/C5 modes on original native seams.

Fresh random Qwen is reused only as a backbone construction primitive; the actual
sidecar is explicitly rebuilt from declared configuration, never metadata labels.
Tiny paired engineering fixtures, not scientific matrix/results or5000 coverage.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.application.scientific_ablation_generation_source import (
    read_scientific_ablation_generation_source_v1,
)
from iris.application.scientific_ablation_inference import build_scientific_ablation_inference_v1
from iris.application.scientific_ablation_inference_io import write_scientific_ablation_inference_v1
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    read_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_optimization import IrisScientificAblationOptimizerV1
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_ablation_workflow_io import (
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_teacher import (
    IrisScientificObjectivePolicyV1,
    IrisScientificTeacherV1,
)
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.scientific_ablation_causal_design_v1 import (
    ScientificAblationMessageCausalPolicyV1,
    ScientificAblationWorkspaceCausalPolicyV1,
)
from iris.artifacts.scientific_ablation_collapse_v1 import ScientificAblationCollapsePolicyV1
from iris.artifacts.scientific_ablation_lifecycle_v1 import IrisScientificAblationLifecyclePolicyV1
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
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.probes import synthetic_probe_vocabulary_v1
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_durable_lifecycle import _receipt
from tests.integration.test_scientific_generation import _test_spec
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _build, _rows

type Routing = Literal["learned", "random", "uniform"]
type Switch = Literal["enabled", "disabled"]


@dataclass(frozen=True, slots=True)
class Geometry:
    modules: int = 4
    slots: int = 4
    state_dim: int = 8
    cycles: int = 3
    winners: int = 2
    routing: Routing = "learned"
    goal: Switch = "enabled"
    broadcast: Switch = "enabled"

    def __post_init__(self) -> None:
        values = self.modules, self.slots, self.state_dim, self.cycles, self.winners
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Exact positive native geometry required")
        if self.winners > min(self.modules, self.slots):
            raise ValueError("Winner count cannot exceed actual modules or slots")
        if self.routing not in ("learned", "random", "uniform") or any(
            value not in ("enabled", "disabled") for value in (self.goal, self.broadcast)
        ):
            raise ValueError("Explicit supported native mode required")

    @property
    def key(self) -> str:
        return (
            f"m{self.modules}-k{self.slots}-d{self.state_dim}-c{self.cycles}-r{self.winners}"
            f"-{self.routing}-g{self.goal}-b{self.broadcast}"
        )


ROUTINGS: tuple[Routing, ...] = ("learned", "random", "uniform")
SWITCHES: tuple[Switch, ...] = ("enabled", "disabled")
MODE_GEOMETRIES = tuple(
    Geometry(cycles=cycles, routing=routing, goal=goal, broadcast=broadcast)
    for cycles in (1, 5)
    for routing in ROUTINGS
    for goal in SWITCHES
    for broadcast in SWITCHES
)
VARIABLE_GEOMETRIES = (
    Geometry(8, 8, 12, 3, 2),
    Geometry(4, 1, 8, 1, 1, "uniform", "disabled", "disabled"),
    Geometry(8, 4, 16, 3, 4, "random", "enabled", "disabled"),
    Geometry(16, 16, 8, 5, 4, "learned", "disabled", "enabled"),
    Geometry(4, 16, 12, 3, 1, "random", "disabled", "disabled"),
)
GEOMETRIES = VARIABLE_GEOMETRIES + MODE_GEOMETRIES


@dataclass(frozen=True, slots=True)
class GeometryCase:
    root: Path
    geometry: Geometry
    plan: IrisScientificAblationWorkflowPlanV1
    factory: Callable[[], IrisScientificAblationRunV1]


def build_run(geometry: Geometry) -> IrisScientificAblationRunV1:
    """Original real random backbone + genuinely configured sidecar and run."""
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        original, _, _ = _build(seed=1907, slots=geometry.slots, cycles=geometry.cycles)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1907)
            teacher = build_teacher_forcing_model(
                backbone=original.teacher.backbone,
                cognitive_config=CognitiveConfig.model_validate(
                    dict(
                        name="scientific-general-tiny",
                        state_dim=geometry.state_dim,
                        num_modules=geometry.modules,
                        layer_sampler=dict(layer_fractions=(0.25, 0.5, 0.75, 1.0)),
                    ),
                ),
                workspace_config=WorkspaceConfig.model_validate(
                    dict(
                        name="scientific-general-tiny",
                        num_slots=geometry.slots,
                        cycles=geometry.cycles,
                        winners_per_cycle=geometry.winners,
                        decay_rate=0.1,
                        empty_threshold=1e-4,
                        arbiter=dict(
                            confidence_weight=1.0, novelty_weight=1.0, relevance_weight=1.0
                        ),
                        allocator=dict(
                            empty_bonus=1.0, allocation_age_penalty=0.1, retention_age_penalty=0.1
                        ),
                    ),
                ),
            )
            base = IrisScientificTeacherV1(
                teacher=teacher,
                vocabulary=synthetic_probe_vocabulary_v1(),
                probe_init_seed=42,
                policy=IrisScientificObjectivePolicyV1(),
            )
        model = IrisScientificAblationTeacherV1(
            base=base,
            ablation=CognitiveAblationPolicyV1.model_validate(
                dict(
                    routing=geometry.routing,
                    goal=geometry.goal,
                    broadcast=geometry.broadcast,
                    routing_seed=42 if geometry.routing == "random" else None,
                )
            ),
        )
        return IrisScientificAblationRunV1(
            optimizer=IrisScientificAblationOptimizerV1(
                model=model,
                policy=_policy(minimum_target_tokens=1, max_optimizer_steps=3),
            ),
            examples=_rows()[:5],
            terminal_token_id=1,
            seed=901,
        )
    finally:
        restore_runtime_state(caller)


def prepare(root: Path, geometry: Geometry) -> GeometryCase:
    """Build actual cycle-specific source, never relabel an old C3checkpoint."""
    root.mkdir(parents=True)

    def factory() -> IrisScientificAblationRunV1:
        return build_run(geometry)

    controller = IrisScientificAblationLifecycleV1(
        run=factory(),
        validation=_validation(),
        policy=IrisScientificAblationLifecyclePolicyV1(
            purpose="smoke",
            validate_every_steps=1,
            checkpoint_every_steps=3,
            early_stopping_patience=8,
            minimum_improvement=1e6,
            maximum_storage_bytes=8 * 1024**3,
        ),
        checkpoint_root=root / "source/native",
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
    actual = projected.manifest.payload.model
    assert (
        actual.cognitive.num_modules,
        actual.workspace.num_slots,
        actual.cognitive.state_dim,
        actual.workspace.cycles,
        actual.workspace.winners_per_cycle,
    ) == (
        geometry.modules,
        geometry.slots,
        geometry.state_dim,
        geometry.cycles,
        geometry.winners,
    )
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
