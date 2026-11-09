"""Real recipe parsing/counterprobes, with every model/corpus operation denied."""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from iris.application import scientific_ablation_command_config as recipe
from iris.application.scientific_command import IrisScientificCommandConfigV1
from iris.artifacts.scientific_ablation_lifecycle_v1 import IrisScientificAblationLifecyclePolicyV1


def _raw(*, purpose: str = "smoke") -> dict[str, Any]:
    return {
        "kind": "iris-scientific-ablation-command-v1",
        "data_scope": "candidate_not_launch_approval",
        "purpose": purpose,
        "device": "cpu",
        "backbone": {
            "name": "recipe-only-fixture-not-pretrained",
            "repository": "offline/random-qwen3",
            "revision": "1" * 40,
            "tokenizer_revision": "2" * 40,
            "local_files_only": True,
            "dtype": "fp32",
            "tokenizer": {"use_fast": True, "padding_side": "left", "truncation_side": "left"},
            "limits": {"max_prompt_tokens": 768, "max_new_tokens": 256},
        },
        "cognitive": {
            "name": "recipe-only",
            "state_dim": 8,
            "num_modules": 4,
            "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
        },
        "workspace": {
            "name": "recipe-only",
            "num_slots": 4,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.1,
            "empty_threshold": 0.0001,
            "arbiter": {"confidence_weight": 1.0, "novelty_weight": 1.0, "relevance_weight": 1.0},
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        },
        "ablation": {"routing": "learned", "goal": "enabled", "broadcast": "enabled"},
        # These paths deliberately do not exist: parsing must never open corpora.
        "train": {
            "path": "unopened-train.parquet",
            "sha256": "a" * 64,
            "maximum_rows": 50000 if purpose == "scientific" else 5,
            "maximum_file_bytes": 1024**2,
            "maximum_logical_bytes": 1024**2,
        },
        "validation": {
            "path": "unopened-validation.parquet",
            "sha256": "b" * 64,
            "maximum_rows": 5000 if purpose == "scientific" else 2,
            "maximum_file_bytes": 1024**2,
            "maximum_logical_bytes": 1024**2,
        },
        "initialization_seed": 1337,
        "probe_initialization_seed": 42,
        "run_seed": 2026,
        "deterministic": True,
        "terminal_token_id": 1,
        "objective": {},
        "optimizer": {
            "purpose": purpose,
            "minimum_target_tokens": 16384 if purpose == "scientific" else 1,
            "max_optimizer_steps": 30000 if purpose == "scientific" else 3,
        },
        "lifecycle": {
            "kind": "iris-scientific-ablation-lifecycle-v1",
            "purpose": purpose,
            "validate_every_steps": 500 if purpose == "scientific" else 1,
            "checkpoint_every_steps": 1000 if purpose == "scientific" else 3,
        },
        "maximum_journal_bytes": 4 * 1024**2,
    }


def _file(tmp_path: Path, raw: object) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


def _set(raw: dict[str, Any], fields: tuple[str, ...], value: object) -> None:
    parent = raw
    for field in fields[:-1]:
        parent = parent[field]
    parent[fields[-1]] = value


@pytest.fixture(autouse=True)
def no_execution() -> Iterator[None]:
    with (
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No native graph")) as graph,
        patch(
            "iris.application.scientific_command._rows",
            side_effect=AssertionError("No input corpus access"),
        ) as rows,
        patch(
            "iris.application.scientific_command.BackboneAdapter",
            side_effect=AssertionError("No backbone construction"),
        ) as backbone,
    ):
        yield
        assert graph.call_count == rows.call_count == backbone.call_count == 0


@pytest.mark.parametrize("cycles", [1, 5])
@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_actual_reader_preserves_all24_c1_c5_modes_without_execution(
    tmp_path: Path, cycles: int, routing: str, goal: str, broadcast: str
) -> None:
    raw = _raw()
    raw["workspace"]["cycles"] = cycles
    raw["ablation"].update(routing=routing, goal=goal, broadcast=broadcast)
    if routing == "random":
        raw["ablation"]["routing_seed"] = 42
    path = _file(tmp_path, raw)
    before = path.read_bytes()
    result = recipe.read_scientific_ablation_command_config_v1(path)
    assert type(result) is recipe.IrisScientificAblationCommandConfigV1
    assert type(result.lifecycle) is IrisScientificAblationLifecyclePolicyV1
    assert (result.ablation.routing, result.ablation.goal, result.ablation.broadcast) == (
        routing,
        goal,
        broadcast,
    )
    assert result.workspace.cycles == cycles and result.purpose == "smoke"
    assert result.ablation.routing_seed == (42 if routing == "random" else None)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["recipe.json"]
    assert not isinstance(result, IrisScientificCommandConfigV1)


@pytest.mark.parametrize(
    ("modules", "slots", "width", "cycles", "winners"),
    [(4, 1, 8, 1, 1), (8, 8, 512, 3, 2), (16, 16, 16, 5, 4)],
)
def test_reader_preserves_actual_geometry_intent(
    tmp_path: Path, modules: int, slots: int, width: int, cycles: int, winners: int
) -> None:
    raw = _raw()
    raw["cognitive"].update(num_modules=modules, state_dim=width)
    raw["workspace"].update(num_slots=slots, cycles=cycles, winners_per_cycle=winners)
    result = recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))
    assert (
        result.cognitive.num_modules,
        result.workspace.num_slots,
        result.cognitive.state_dim,
        result.workspace.cycles,
        result.workspace.winners_per_cycle,
    ) == (modules, slots, width, cycles, winners)


@pytest.mark.parametrize(
    ("fields", "value"),
    [
        (("kind",), "iris-scientific-command-v1"),
        (("data_scope",), "scientific_launch_approved"),
        (("device",), "cuda"),
        (("backbone", "local_files_only"), False),
        (("backbone", "local_files_only"), 1),
        (("backbone", "trust_remote_code"), True),
        (("backbone", "tokenizer", "use_fast"), False),
        (("backbone", "dtype"), "bf16"),
        (("backbone", "repository"), "C:/private-model"),
        (("backbone", "repository"), "../private-model"),
        (("backbone", "repository"), "org/repo/third"),
        (("backbone", "revision"), "main"),
        (("backbone", "revision"), "A" * 40),
        (("backbone", "tokenizer_revision"), "2" * 39),
        (("initialization_seed",), -1),
        (("initialization_seed",), True),
        (("run_seed",), 2**63),
        (("probe_initialization_seed",), "42"),
        (("deterministic",), 1),
        (("terminal_token_id",), False),
        (("terminal_token_id",), -1),
        (("cognitive", "num_modules"), 1),
        (("cognitive", "state_dim"), True),
        (("workspace", "num_slots"), 1),
        (("workspace", "cycles"), 0),
        (("workspace", "winners_per_cycle"), True),
        (("validation", "maximum_rows"), 5001),
        (("validation", "path"), "https://private.invalid/data"),
        (("train", "path"), "C:relative.parquet"),
        (("train", "sha256"), "A" * 64),
        (("maximum_journal_bytes",), 0),
        (("maximum_journal_bytes",), True),
        (("optimizer", "purpose"), "scientific"),
        (("lifecycle", "purpose"), "scientific"),
        (("lifecycle", "kind"), "iris-scientific-lifecycle-v1"),
        (("ablation", "routing"), "random"),
        (("ablation", "routing"), "hard_top_token"),
        (("ablation", "routing_seed"), 42),
        (("ablation", "maximum_routing_scalars"), 0),
        (("ablation", "goal"), "dropped"),
        (("ablation", "broadcast"), True),
        (("ablation", "native_factory"), "pretend-ready"),
        (("native_owner",), "restored-from-json"),
    ],
)
def test_invalid_recipe_rejected_before_any_model_or_corpus_access(
    tmp_path: Path, fields: tuple[str, ...], value: object
) -> None:
    raw = _raw()
    _set(raw, fields, value)
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))


def test_scientific_recipe_keeps_full_source_budgets_and_cadence(tmp_path: Path) -> None:
    result = recipe.read_scientific_ablation_command_config_v1(
        _file(tmp_path, _raw(purpose="scientific"))
    )
    assert result.purpose == result.optimizer.purpose == result.lifecycle.purpose == "scientific"
    assert (result.train.maximum_rows, result.validation.maximum_rows) == (50000, 5000)
    assert result.optimizer.minimum_target_tokens == 16384
    assert (
        result.lifecycle.validate_every_steps,
        result.lifecycle.checkpoint_every_steps,
        result.lifecycle.early_stopping_patience,
    ) == (500, 1000, 8)
    assert result.data_scope == "candidate_not_launch_approval"


@pytest.mark.parametrize(
    ("fields", "value"),
    [
        (("train", "maximum_rows"), 5),
        (("validation", "maximum_rows"), 2),
        (("optimizer", "minimum_target_tokens"), 1),
        (("lifecycle", "validate_every_steps"), 1),
        (("lifecycle", "checkpoint_every_steps"), 3),
        (("lifecycle", "early_stopping_patience"), 1),
    ],
)
def test_scientific_recipe_cannot_silently_become_smoke(
    tmp_path: Path, fields: tuple[str, ...], value: object
) -> None:
    raw = _raw(purpose="scientific")
    _set(raw, fields, value)
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))


@pytest.mark.parametrize("missing", ["purpose", "ablation", "optimizer", "lifecycle"])
def test_explicit_purpose_mode_and_policies_are_required(tmp_path: Path, missing: str) -> None:
    raw = _raw()
    del raw[missing]
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))


@pytest.mark.parametrize("raw", [None, [], 1, True, "not a recipe"])
def test_nonobject_recipe_is_not_coerced(tmp_path: Path, raw: object) -> None:
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_numbers_are_rejected(tmp_path: Path, value: float) -> None:
    raw = _raw()
    raw["optimizer"]["learning_rate"] = value
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))


def test_duplicate_root_and_nested_fields_are_rejected(tmp_path: Path) -> None:
    path = _file(tmp_path, _raw())
    original = path.read_text(encoding="utf-8")
    for text in (
        original.replace("{", '{"purpose":"scientific",', 1),
        original.replace('"cycles": 3', '"cycles": 1, "cycles": 3'),
    ):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
            recipe.read_scientific_ablation_command_config_v1(path)


def test_recipe_byte_boundary_is_checked_before_json_parse(tmp_path: Path) -> None:
    path = _file(tmp_path, _raw())
    content = path.read_bytes()
    path.write_bytes(content + b" " * (128 * 1024 - len(content)))
    assert recipe.read_scientific_ablation_command_config_v1(path).purpose == "smoke"
    path.write_bytes(path.read_bytes() + b" ")
    with (
        patch.object(
            json, "loads", side_effect=AssertionError("Must reject before parse")
        ) as parser,
        pytest.raises(recipe.IrisScientificAblationCommandConfigError),
    ):
        recipe.read_scientific_ablation_command_config_v1(path)
    assert parser.call_count == 0


def test_missing_directory_and_drive_relative_paths_are_safe_failures(tmp_path: Path) -> None:
    for path in (tmp_path / "absent.json", tmp_path, Path("C:drive-relative.json")):
        with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
            recipe.read_scientific_ablation_command_config_v1(path)


def test_error_does_not_publish_recipe_content_or_paths(tmp_path: Path) -> None:
    raw = _raw()
    secret = "PRIVATE-ROW-CONTENT-DO-NOT-PRINT"
    raw["ablation"]["routing"] = secret
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError) as caught:
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))
    assert secret not in str(caught.value) and str(tmp_path) not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_legacy_baseline_and_ablation_recipe_kinds_stay_distinct(tmp_path: Path) -> None:
    raw = _raw()
    with pytest.raises(ValidationError):
        IrisScientificCommandConfigV1.model_validate_json(json.dumps(raw))
    del raw["ablation"], raw["purpose"]
    raw["kind"] = "iris-scientific-command-v1"
    raw["lifecycle"]["kind"] = "iris-scientific-lifecycle-v1"
    old = IrisScientificCommandConfigV1.model_validate_json(json.dumps(raw))
    assert old.kind == "iris-scientific-command-v1" and old.optimizer.purpose == "smoke"
    with pytest.raises(recipe.IrisScientificAblationCommandConfigError):
        recipe.read_scientific_ablation_command_config_v1(_file(tmp_path, raw))


def test_fresh_first_import_needs_no_iris_or_application_prewarming(tmp_path: Path) -> None:
    stage = Path(__file__).resolve().parents[2]
    path = _file(tmp_path, _raw())
    code = """import sys
from pathlib import Path
stage, config = map(Path, sys.argv[1:])
sys.path.insert(0, str(stage / 'src'))
assert not any(n == 'iris' or n.startswith('iris.') for n in sys.modules)
import iris.application.scientific_ablation_command_config as command
result = command.read_scientific_ablation_command_config_v1(config)
assert result.kind == 'iris-scientific-ablation-command-v1' and result.purpose == 'smoke'
for name, module in tuple(sys.modules.items()):
    if name == 'iris' or name.startswith('iris.'):
        assert Path(module.__file__).resolve().is_relative_to(stage / 'src')
print('FRESH_ABLATION_RECIPE_IMPORTED_AND_READ_NO_PREWARM', flush=True)
"""
    actual = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(stage), str(path)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert actual.returncode == 0, actual.stderr.decode()
    assert actual.stdout == b"FRESH_ABLATION_RECIPE_IMPORTED_AND_READ_NO_PREWARM\r\n" or (
        actual.stdout == b"FRESH_ABLATION_RECIPE_IMPORTED_AND_READ_NO_PREWARM\n"
    )
