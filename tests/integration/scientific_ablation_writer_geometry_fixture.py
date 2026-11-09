"""Actual configured native-best/capture/COMPLETE source for full writer tests.

Never edits a saved graph/checkpoint or replaces a writer output. Every geometry
and lifecycle constant is present at ORIGINAL model construction and all fresh
restore factories reproduce that exact configuration. Random offline tiny Qwen
is engineering evidence, not pretrained learning or generated-answer quality.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from iris.ablations import CognitiveAblationPolicyV1
from iris.application.scientific_ablation_capture import build_scientific_ablation_capture_plan_v1
from iris.application.scientific_ablation_generation_context import (
    IrisScientificAblationGenerationContextV1,
    build_scientific_ablation_generation_context_v1,
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
from iris.application.scientific_command import ScientificParquetInputV1
from iris.application.scientific_teacher import (
    IrisScientificObjectivePolicyV1,
    IrisScientificTeacherV1,
)
from iris.application.training_composition import build_teacher_forcing_model
from iris.artifacts.scientific_ablation_lifecycle_v1 import IrisScientificAblationLifecyclePolicyV1
from iris.data.schemas import IrisExample
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training.probes import synthetic_probe_vocabulary_v1
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_capture import _capture
from tests.integration.test_scientific_ablation_durable_lifecycle import _receipt
from tests.integration.test_scientific_ablation_evaluation_v2 import Case
from tests.integration.test_scientific_ablation_generation_source import Prepared, Routing, _read
from tests.integration.test_scientific_ablation_heldout import _plan
from tests.integration.test_scientific_generation import _test_spec
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _build, _rows


@dataclass(frozen=True)
class Geometry:
    modules: int = 4
    slots: int = 1
    width: int = 8
    cycles: int = 3
    writers: int = 1
    empty_threshold: float = 1e-4


def routing_scalar_budget(geometry: Geometry, prompt_token_cap: object) -> int:
    """Original pre-owner quota for the ENTIRE declared prompt cap, not an outcome.

    Keep the ordinary 65536 policy unchanged. Large module graphs explicitly
    reserve the canonical route guard's 3*M*L+M*D bound before training/capture;
    all fresh factories reproduce this same policy. No production cap is raised.
    """
    if type(prompt_token_cap) is not int:
        raise ValueError("Exact positive original backbone prompt cap required")
    assert isinstance(prompt_token_cap, int)
    if prompt_token_cap < 1:
        raise ValueError("Exact positive original backbone prompt cap required")
    return max(65536, 3 * geometry.modules * prompt_token_cap + geometry.modules * geometry.width)


def run(
    geometry: Geometry, routing: Routing, *, goal: str, broadcast: str
) -> IrisScientificAblationRunV1:
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1907)
            base, _, _ = _build(seed=1907, slots=geometry.slots, cycles=geometry.cycles)
            teacher = build_teacher_forcing_model(
                backbone=base.teacher._backbone,
                cognitive_config=CognitiveConfig.model_validate(
                    dict(
                        name="actual-writer-geometry",
                        state_dim=geometry.width,
                        num_modules=geometry.modules,
                        layer_sampler={"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
                    )
                ),
                workspace_config=WorkspaceConfig.model_validate(
                    dict(
                        name="actual-writer-geometry",
                        num_slots=geometry.slots,
                        cycles=geometry.cycles,
                        winners_per_cycle=geometry.writers,
                        decay_rate=0.1,
                        empty_threshold=geometry.empty_threshold,
                        arbiter=dict(
                            confidence_weight=1.0, novelty_weight=1.0, relevance_weight=1.0
                        ),
                        allocator=dict(
                            empty_bonus=1.0, allocation_age_penalty=0.1, retention_age_penalty=0.1
                        ),
                    )
                ),
            )
            scientific = IrisScientificTeacherV1(
                teacher=teacher,
                vocabulary=synthetic_probe_vocabulary_v1(),
                probe_init_seed=42,
                policy=IrisScientificObjectivePolicyV1(),
            )
            model = IrisScientificAblationTeacherV1(
                base=scientific,
                ablation=CognitiveAblationPolicyV1.model_validate(
                    dict(
                        routing=routing,
                        routing_seed=42 if routing == "random" else None,
                        goal=goal,
                        broadcast=broadcast,
                        maximum_routing_scalars=routing_scalar_budget(
                            geometry, base.teacher._backbone.max_prompt_tokens
                        ),
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


def prepared(
    root: Path, geometry: Geometry, routing: Routing, *, goal: str, broadcast: str
) -> Prepared:
    controller = IrisScientificAblationLifecycleV1(
        run=run(geometry, routing, goal=goal, broadcast=broadcast),
        validation=_validation(),
        policy=IrisScientificAblationLifecyclePolicyV1(
            purpose="smoke",
            validate_every_steps=1,
            checkpoint_every_steps=3,
            early_stopping_patience=8,
            minimum_improvement=1e6,
            maximum_storage_bytes=128 * 1024**2,
        ),
        checkpoint_root=root / "native",
    )
    receipt = _receipt(controller)
    lifecycle = read_scientific_ablation_lifecycle_checkpoint_v1(receipt.path)
    projected = build_scientific_ablation_inference_v1(
        lifecycle,
        expected_lifecycle_sha256=receipt.manifest.sha256,
        make_fresh_run=lambda: run(geometry, routing, goal=goal, broadcast=broadcast),
        selection="selected_best",
    )
    path = write_scientific_ablation_inference_v1(
        projected,
        root=root / "inference",
        expected_sha256=projected.manifest.sha256,
        expected_source_checkpoint_sha256=projected.manifest.payload.source_checkpoint_sha256,
        expected_lifecycle_sha256=receipt.manifest.sha256,
    )
    return Prepared(receipt.path, path, lifecycle, projected)


def context(
    source: Prepared, geometry: Geometry, routing: Routing, *, goal: str, broadcast: str
) -> IrisScientificAblationGenerationContextV1:
    bundle = _read(source)
    return build_scientific_ablation_generation_context_v1(
        bundle,
        expected_sha256=bundle.manifest.sha256,
        expected_lifecycle_sha256=bundle.lifecycle.manifest.sha256,
        expected_inference_sha256=bundle.inference.manifest.sha256,
        expected_source_checkpoint_sha256=bundle.manifest.payload.best_checkpoint_sha256,
        make_fresh_run=lambda: run(geometry, routing, goal=goal, broadcast=broadcast),
        validation=_validation(),
        sampling=TemperatureTopPSamplingConfig(1.0, 1.0),
    )


def case(
    root: Path,
    geometry: Geometry,
    routing: Routing,
    rows: tuple[IrisExample, ...],
    *,
    goal: str,
    broadcast: str,
) -> Case:
    root.mkdir(parents=True, exist_ok=True)
    source = prepared(root / "source", geometry, routing, goal=goal, broadcast=broadcast)
    heldout = _plan(
        context(source, geometry, routing, goal=goal, broadcast=broadcast),
        root / "test.parquet",
        rows,
    )
    plan = build_scientific_ablation_capture_plan_v1(
        heldout,
        expected_input_sha256=heldout.manifest.sha256,
    )
    captured = _capture(heldout, root / "captures", plan)
    return Case(
        root,
        source,
        captured,
        _test_spec(root / "train.parquet", _rows()[:5]),
        _test_spec(root / "validation.parquet", _validation()),
        ScientificParquetInputV1.model_validate(
            dict(
                path="test.parquet",
                sha256=plan.payload.input.payload.test_parquet_sha256,
                maximum_rows=len(rows),
                maximum_file_bytes=4 * 1024**2,
                maximum_logical_bytes=4 * 1024**2,
            )
        ),
        rows,
    )
