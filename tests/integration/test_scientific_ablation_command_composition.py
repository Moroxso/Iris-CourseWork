"""Actual offline HF-cache loading and owned tiny ablation composition, no downloads."""

import json
import random
import subprocess
import sys
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_command_composition as composition
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_command_config import (
    IrisScientificAblationCommandConfigV1,
)
from iris.application.scientific_ablation_lifecycle import (
    IrisScientificAblationLifecycleBoundaryV1,
    IrisScientificAblationLifecycleV1,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    build_scientific_ablation_lifecycle_checkpoint_v1,
    read_scientific_ablation_lifecycle_checkpoint_v1,
    restore_scientific_ablation_lifecycle_v1,
    write_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_command import _rows as read_command_rows
from iris.application.scientific_optimization import _same_runtime
from iris.backbone.adapter import BackboneAdapter
from iris.data.generator import generate_synthetic_split
from iris.data.parquet import encode_synthetic_parquet
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration._offline_composition_fixture import tiny_offline_cache
from tests.integration.test_scientific_teacher import _build, _rows
from tests.unit.test_scientific_ablation_command_config import _raw

pytestmark = pytest.mark.integration


def _input(path: Path, rows: tuple[IrisExample, ...]) -> dict[str, Any]:
    content = encode_synthetic_parquet(rows, max_logical_bytes=1024**2, max_file_bytes=1024**2)
    path.write_bytes(content)
    return dict(
        path=path.name,
        sha256=sha256(content).hexdigest(),
        maximum_rows=len(rows),
        maximum_file_bytes=1024**2,
        maximum_logical_bytes=1024**2,
    )


def _recipe(
    tmp_path: Path, *, mutate: Callable[[dict[str, Any]], None] | None = None
) -> IrisScientificAblationCommandConfigV1:
    raw = _raw()
    train = _rows()[:5]
    validation = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
            template_id="iris-synthetic-01-v2",
        )
    )[:2]
    raw["train"] = _input(tmp_path / "train.parquet", train)
    raw["validation"] = _input(tmp_path / "validation.parquet", validation)
    if mutate is not None:
        mutate(raw)
    return IrisScientificAblationCommandConfigV1.model_validate_json(json.dumps(raw), strict=True)


@pytest.fixture(scope="module")
def actual_offline_cache() -> Path:
    """Original random tiny model is saved once; original HF loaders stay unpatched afterwards."""
    return tiny_offline_cache(lambda: _build(cycles=1))


@pytest.mark.parametrize("cycles", [1, 5])
@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_actual_unpatched_hf_load_all24_modes_fresh_exact_run_no_forward(
    tmp_path: Path, actual_offline_cache: Path, cycles: int, routing: str, goal: str, broadcast: str
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["workspace"]["cycles"] = cycles
        raw["ablation"].update(routing=routing, goal=goal, broadcast=broadcast)
        if routing == "random":
            raw["ablation"]["routing_seed"] = 42

    config = _recipe(tmp_path, mutate=mutate)
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        torch.no_grad(),
        patch(
            "transformers.Qwen3ForCausalLM.forward",
            side_effect=AssertionError("Construction must not forward"),
        ) as forward,
    ):
        run = composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    assert actual_offline_cache.is_dir() and forward.call_count == 0
    assert type(run) is IrisScientificAblationRunV1 and run._controller_attachment is None
    assert type(run._optimizer._model) is IrisScientificAblationTeacherV1
    model = run._optimizer._model
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert (model.ablation.routing, model.ablation.goal, model.ablation.broadcast) == (
        routing,
        goal,
        broadcast,
    )
    assert model.teacher.proposal_head.cycles == cycles
    assert model.teacher.backbone.configuration_json == config.backbone.model_dump_json()
    assert not any(p.requires_grad for p in model.teacher.backbone._model.parameters())
    assert all(not child.training for child in model.teacher.backbone._model.modules())
    assert run.completed_steps == run.examples_seen == run.target_tokens_seen == 0
    assert sum(
        p.numel() for p in model.parameters()
    ) == composition.scientific_ablation_command_parameter_count_v1(
        config=config, backbone_hidden_size=16
    )
    initial = build_scientific_ablation_checkpoint_v1(run)
    assert initial.manifest.payload.completed_steps == 0 and initial.state_bytes
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    (tmp_path / "actual-composition-witness.json").write_text(
        json.dumps(
            dict(
                routing=routing,
                goal=goal,
                broadcast=broadcast,
                cycles=cycles,
                run_sha256=run.configuration_sha256,
                initial_sha256=initial.manifest.sha256,
                actual_parameters=sum(p.numel() for p in model.parameters()),
                native_forward_calls=forward.call_count,
                original_hf_cache=str(actual_offline_cache),
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("modules", "slots", "width", "cycles", "winners"),
    [(4, 1, 8, 1, 1), (8, 8, 12, 3, 2), (16, 16, 16, 5, 4)],
)
def test_preallocation_count_matches_actual_general_geometry(
    tmp_path: Path,
    actual_offline_cache: Path,
    modules: int,
    slots: int,
    width: int,
    cycles: int,
    winners: int,
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["cognitive"].update(num_modules=modules, state_dim=width)
        raw["workspace"].update(num_slots=slots, cycles=cycles, winners_per_cycle=winners)

    config = _recipe(tmp_path, mutate=mutate)
    run = composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    assert actual_offline_cache.is_dir()
    assert sum(p.numel() for p in run._optimizer._model.parameters()) == (
        composition.scientific_ablation_command_parameter_count_v1(
            config=config, backbone_hidden_size=16
        )
    )


def test_fresh_reconstruction_initial_bytes_ignore_loader_rng_consumption(
    tmp_path: Path, actual_offline_cache: Path
) -> None:
    config = _recipe(tmp_path)
    caller = capture_runtime_state(torch.device("cpu"))
    first = composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    left = build_scientific_ablation_checkpoint_v1(first)
    original = BackboneAdapter._load_model

    def consume(self: BackboneAdapter) -> Any:
        model = original(self)
        random.random()
        torch.rand(11)
        return model

    with patch.object(BackboneAdapter, "_load_model", new=consume):
        second = composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    right = build_scientific_ablation_checkpoint_v1(second)
    assert actual_offline_cache.is_dir() and first is not second
    assert left.manifest == right.manifest and left.state_bytes == right.state_bytes
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("cycles", [1, 5])
def test_actual_lifecycle_initial_snapshot_training_best_and_disk_restore(
    tmp_path: Path, actual_offline_cache: Path, cycles: int
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["workspace"]["cycles"] = cycles
        raw["lifecycle"].update(minimum_improvement=1e6)
        if cycles == 5:
            raw["ablation"].update(
                routing="random", routing_seed=42, goal="disabled", broadcast="disabled"
            )

    config = _recipe(tmp_path, mutate=mutate)
    caller = capture_runtime_state(torch.device("cpu"))
    with patch(
        "transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No initial forward")
    ) as forward:
        owner = composition.create_scientific_ablation_command_lifecycle_v1(
            config=config, base=tmp_path
        )
        initial = build_scientific_ablation_checkpoint_v1(owner._run)
    assert type(owner) is IrisScientificAblationLifecycleV1 and forward.call_count == 0
    assert owner.state.step == initial.manifest.payload.completed_steps == 0
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    boundaries = []
    for _ in range(7):
        if owner.state.phase == "stopped":
            break
        phase = owner.advance()
        if type(phase) is IrisScientificAblationLifecycleBoundaryV1:
            boundaries.append(phase)
    assert len(boundaries) == 1 and owner.state.step == 3 and owner.state.best_step == 1
    assert owner.state.phase == "stopped"
    bundle = build_scientific_ablation_lifecycle_checkpoint_v1(boundaries[0])
    path = write_scientific_ablation_lifecycle_checkpoint_v1(bundle, root=tmp_path / "checkpoints")
    retained = read_scientific_ablation_lifecycle_checkpoint_v1(path)
    fresh = composition.create_scientific_ablation_command_lifecycle_v1(
        config=config, base=tmp_path
    )
    restore_scientific_ablation_lifecycle_v1(retained, controller=fresh)
    assert fresh.state == owner.state and fresh.state.best_step == 1
    assert fresh.best_checkpoint is not None and owner.best_checkpoint is not None
    assert fresh.best_checkpoint.manifest == owner.best_checkpoint.manifest
    assert actual_offline_cache.is_dir()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    (tmp_path / "actual-lifecycle-witness.json").write_text(
        json.dumps(
            dict(
                cycles=cycles,
                actual_training_steps=3,
                selected_best=1,
                initial_sha256=initial.manifest.sha256,
                current_sha256=boundaries[0].current.manifest.sha256,
                lifecycle_sha256=retained.manifest.sha256,
                restored_state=fresh.state.model_dump(mode="json"),
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "missing",
        "train-heldout",
        "validation-test",
        "leakage",
        "invalid-parquet",
        "scientific-size",
    ],
)
def test_actual_corpus_failures_reject_before_loading_backbone(tmp_path: Path, fault: str) -> None:
    config = _recipe(tmp_path)
    raw = json.loads(config.model_dump_json())
    if fault == "hash":
        raw["train"]["sha256"] = "0" * 64
    elif fault == "missing":
        raw["validation"]["path"] = "absent.parquet"
    elif fault == "invalid-parquet":
        content = b"PAR1not-a-valid-parquet-filePAR1"
        (tmp_path / "invalid.parquet").write_bytes(content)
        raw["train"].update(path="invalid.parquet", sha256=sha256(content).hexdigest())
    elif fault == "scientific-size":
        raw.update(purpose="scientific")
        raw["optimizer"].update(purpose="scientific", minimum_target_tokens=16384)
        raw["lifecycle"].update(
            purpose="scientific", validate_every_steps=500, checkpoint_every_steps=1000
        )
        raw["train"]["maximum_rows"], raw["validation"]["maximum_rows"] = 50000, 5000
    else:
        rows = _rows()[:2]
        split = "test" if fault == "validation-test" else "validation"
        changed = rows[0].model_copy(update={"split": split, "id": f"{split}-{rows[0].seed}"})
        raw["train" if fault == "train-heldout" else "validation"] = _input(
            tmp_path / "changed.parquet", (changed,)
        )
    config = IrisScientificAblationCommandConfigV1.model_validate_json(json.dumps(raw), strict=True)
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(
            composition, "BackboneAdapter", side_effect=AssertionError("Too late")
        ) as loader,
        pytest.raises(composition.IrisScientificAblationCommandCompositionError),
    ):
        composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    assert loader.call_count == 0 and _same_runtime(
        caller, capture_runtime_state(torch.device("cpu"))
    )


@pytest.mark.parametrize(
    "fault", ["eos", "prompt", "target", "context", "routing", "actual-parameter-bytes"]
)
def test_full_token_and_allocation_rejection_precedes_sidecar_construction(
    tmp_path: Path, actual_offline_cache: Path, fault: str
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        if fault == "eos":
            raw["terminal_token_id"] = 0
        elif fault == "prompt":
            raw["backbone"]["limits"]["max_prompt_tokens"] = 1
        elif fault == "target":
            raw["backbone"]["limits"]["max_new_tokens"] = 1
        elif fault == "context":
            raw["backbone"]["limits"]["max_prompt_tokens"] = 2048
        elif fault == "routing":
            raw["ablation"]["maximum_routing_scalars"] = 1

    config = _recipe(tmp_path, mutate=mutate)
    # Width1lower bound fits, actual frozen width16does not. No graph is allocated.
    limit = 4 * composition.scientific_ablation_command_parameter_count_v1(
        config=config, backbone_hidden_size=1
    )
    with (
        patch.object(
            composition, "build_teacher_forcing_model", side_effect=AssertionError("Too late")
        ) as sidecar,
        pytest.raises(composition.IrisScientificAblationCommandCompositionError),
    ):
        composition.create_scientific_ablation_command_run_v1(
            config=config,
            base=tmp_path,
            maximum_trainable_parameter_bytes=limit
            if fault == "actual-parameter-bytes"
            else 1024**2,
        )
    assert actual_offline_cache.is_dir() and sidecar.call_count == 0


@pytest.mark.parametrize("limit", [0, -1, True, 512 * 1024**2 + 1, 1])
def test_invalid_or_tiny_allocation_budget_rejects_before_any_loader(
    tmp_path: Path, limit: int
) -> None:
    config = _recipe(tmp_path)
    with (
        patch.object(
            composition, "BackboneAdapter", side_effect=AssertionError("No loader")
        ) as loader,
        pytest.raises(composition.IrisScientificAblationCommandCompositionError),
    ):
        composition.create_scientific_ablation_command_run_v1(
            config=config, base=tmp_path, maximum_trainable_parameter_bytes=limit
        )
    assert loader.call_count == 0


@pytest.mark.parametrize("ambient", ["inference", "autocast"])
def test_forbidden_ambient_modes_fail_before_any_loader(tmp_path: Path, ambient: str) -> None:
    config = _recipe(tmp_path)
    boundary = (
        torch.inference_mode()
        if ambient == "inference"
        else torch.autocast("cpu", dtype=torch.bfloat16)
    )
    with (
        patch.object(
            composition, "BackboneAdapter", side_effect=AssertionError("No loader")
        ) as loader,
        boundary,
        pytest.raises(composition.IrisScientificAblationCommandCompositionError),
    ):
        composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    assert loader.call_count == 0


def test_mutated_nested_recipe_is_revalidated_before_loading(tmp_path: Path) -> None:
    config = _recipe(tmp_path)
    config.workspace.cycles = 0
    with (
        patch.object(
            composition, "BackboneAdapter", side_effect=AssertionError("No loader")
        ) as loader,
        pytest.raises(composition.IrisScientificAblationCommandCompositionError),
    ):
        composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    assert loader.call_count == 0


@pytest.mark.parametrize("signal", [KeyboardInterrupt("first-signal"), SystemExit(83)])
@pytest.mark.parametrize("lifecycle", [False, True])
def test_first_signal_survives_cleanup_fault_after_actual_caller_restore(
    tmp_path: Path, signal: BaseException, lifecycle: bool
) -> None:
    config = _recipe(tmp_path)
    caller = capture_runtime_state(torch.device("cpu"))
    original_restore = restore_runtime_state

    def fail(*args: Any, **kwargs: Any) -> Any:
        random.random()
        torch.rand(3)
        raise signal

    def cleanup(snapshot: Any) -> None:
        original_restore(snapshot)
        raise SystemExit(99)

    entry = (
        composition.create_scientific_ablation_command_lifecycle_v1
        if lifecycle
        else composition.create_scientific_ablation_command_run_v1
    )
    with (
        patch.object(composition, "_corpora", new=fail),
        patch.object(composition, "restore_runtime_state", new=cleanup),
        pytest.raises(type(signal)) as caught,
    ):
        entry(config=config, base=tmp_path)
    assert caught.value is signal and _same_runtime(
        caller, capture_runtime_state(torch.device("cpu"))
    )


def test_duplicate_decoded_training_data_counterprobe_has_no_loader(tmp_path: Path) -> None:
    """Explicit DATA corruption probe, not a claim the real Parquet decoder allows duplicates."""
    config = _recipe(tmp_path)
    original_rows = read_command_rows
    observed = []

    def duplicate(spec: Any, base: Path) -> tuple[IrisExample, ...]:
        rows = original_rows(spec, base)
        if spec.path == "train.parquet":
            observed.append(True)
            return rows[0], rows[0]
        return rows

    with (
        patch.object(composition, "_rows", new=duplicate),
        patch.object(
            composition, "BackboneAdapter", side_effect=AssertionError("No loader")
        ) as loader,
        pytest.raises(composition.IrisScientificAblationCommandCompositionError),
    ):
        composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    assert observed == [True] and loader.call_count == 0


def test_default_d512_m8_parameter_lower_bound_does_not_claim_under20m(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["cognitive"].update(num_modules=8, state_dim=512)

    config = _recipe(tmp_path, mutate=mutate)
    lower_bound = composition.scientific_ablation_command_parameter_count_v1(
        config=config, backbone_hidden_size=1
    )
    assert lower_bound == 25_260_573 and lower_bound > 20_000_000
    # This is source arithmetic, not an allocated D512model or completed scientific DoD.
    (tmp_path / "static-d512-parameter-lower-bound.json").write_text(
        json.dumps(
            dict(
                parameters=lower_bound,
                actual_backbone_width_used=False,
                actual_d512_graph_allocated=False,
                scientific_acceptance=False,
            )
        ),
        encoding="utf-8",
    )


def test_actual_fresh_process_first_import_and_native_run_from_same_recipe(
    tmp_path: Path, actual_offline_cache: Path
) -> None:
    config = _recipe(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(config.model_dump_json(), encoding="utf-8")
    stage = Path(__file__).resolve().parents[2]
    code = """import json, sys
from pathlib import Path
from unittest.mock import patch
stage, recipe_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(stage / 'src'))
assert not any(n == 'iris' or n.startswith('iris.') for n in sys.modules)
import iris.application.scientific_ablation_command_composition as composition
import iris.application.scientific_ablation_command_config as recipes
import torch
from iris.utils.seeds import capture_runtime_state
from iris.application.scientific_optimization import _same_runtime
before = capture_runtime_state(torch.device('cpu'))
config = recipes.read_scientific_ablation_command_config_v1(recipe_path)
with patch('transformers.Qwen3ForCausalLM.forward', side_effect=AssertionError('No forward')) as f:
    run = composition.create_scientific_ablation_command_run_v1(
        config=config, base=recipe_path.parent)
assert run.completed_steps == 0 and f.call_count == 0
assert _same_runtime(before, capture_runtime_state(torch.device('cpu')))
for name, module in tuple(sys.modules.items()):
    if name == 'iris' or name.startswith('iris.'):
        assert Path(module.__file__).resolve().is_relative_to(stage / 'src')
print('FRESH_ABLATION_NATIVE_RUN '+json.dumps(dict(run_sha256=run.configuration_sha256)),flush=True)
"""
    actual = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(stage), str(recipe_path)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert actual.returncode == 0, actual.stderr.decode()
    assert actual_offline_cache.is_dir()
    packets = [
        line
        for line in actual.stdout.splitlines()
        if line.startswith(b"FRESH_ABLATION_NATIVE_RUN ")
    ]
    assert len(packets) == 1
    assert (
        len(json.loads(packets[0].removeprefix(b"FRESH_ABLATION_NATIVE_RUN "))["run_sha256"]) == 64
    )


def test_parameter_preflight_includes_real_slot_allocator_projection(
    tmp_path: Path, actual_offline_cache: Path
) -> None:
    config = _recipe(tmp_path)
    run = composition.create_scientific_ablation_command_run_v1(config=config, base=tmp_path)
    model = run._optimizer._model
    assert model.teacher.slot_allocator.slot_projection.weight.numel() == 64
    assert sum(p.numel() for p in model.parameters()) == 4874
    assert (
        composition.scientific_ablation_command_parameter_count_v1(
            config=config, backbone_hidden_size=16
        )
        == 4874
    )
    assert actual_offline_cache.is_dir()
