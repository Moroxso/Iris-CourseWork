"""Actual CPU scientific CLI composition; only HF loaders use tiny local fixtures."""

import json
import os
import random
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application import scientific_command as command
from iris.application.scientific_execution import read_scientific_execution_v1
from iris.application.scientific_optimization import _same_runtime
from iris.cli import app
from iris.data.generator import generate_synthetic_split
from iris.data.parquet import encode_synthetic_parquet
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def _input(path: Path, rows: tuple[IrisExample, ...]) -> dict[str, Any]:
    content = encode_synthetic_parquet(rows, max_logical_bytes=1024**2, max_file_bytes=1024**2)
    path.write_bytes(content)
    return {
        "path": path.name,
        "sha256": sha256(content).hexdigest(),
        "maximum_rows": len(rows),
        "maximum_file_bytes": 1024**2,
        "maximum_logical_bytes": 1024**2,
    }


def _recipe(tmp_path: Path) -> Path:
    validation = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
            template_id="iris-synthetic-01-v2",
        )
    )[:2]
    raw = {
        "kind": "iris-scientific-command-v1",
        "data_scope": "candidate_not_launch_approval",
        "device": "cpu",
        "backbone": {
            "name": "tiny-command-fixture-not-pretrained",
            "repository": "offline/random-qwen3",
            # Test-only immutable-shape identifiers, NOT real repository commits.
            "revision": "1" * 40,
            "tokenizer_revision": "2" * 40,
            "local_files_only": True,
            "dtype": "fp32",
            "tokenizer": {"use_fast": True, "padding_side": "left", "truncation_side": "left"},
            "limits": {"max_prompt_tokens": 768, "max_new_tokens": 256},
        },
        "cognitive": {
            "name": "tiny-command",
            "state_dim": 8,
            "num_modules": 4,
            "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
        },
        "workspace": {
            "name": "tiny-command",
            "num_slots": 2,
            "cycles": 1,
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
        "train": _input(tmp_path / "train.parquet", _rows()[:5]),
        "validation": _input(tmp_path / "validation.parquet", validation),
        "initialization_seed": 1337,
        "probe_initialization_seed": 42,
        "run_seed": 2026,
        "deterministic": True,
        "terminal_token_id": 1,
        "objective": {},
        "optimizer": {"purpose": "smoke", "minimum_target_tokens": 1, "max_optimizer_steps": 3},
        "lifecycle": {"purpose": "smoke", "validate_every_steps": 1, "checkpoint_every_steps": 2},
        "maximum_journal_bytes": 4 * 1024**2,
    }
    path = tmp_path / "command.json"
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


def _edit(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


@pytest.fixture
def loaders() -> Any:
    _, tokenizer, model = _build(cycles=1, slots=2)

    def load_model(*args: Any, **kwargs: Any) -> Any:
        # Deliberate unrelated consumption: initialization must not depend on this.
        random.random()
        torch.rand(11)
        return deepcopy(model)

    with (
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=lambda *a, **k: deepcopy(tokenizer),
        ) as tok,
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", side_effect=load_model
        ) as lm,
    ):
        yield tok, lm


def _files(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_create_preflights_without_forward_and_preserves_caller(
    tmp_path: Path, loaders: Any
) -> None:
    config = _recipe(tmp_path)
    before = capture_runtime_state(torch.device("cpu"))
    with patch("transformers.Qwen3ForCausalLM.forward", side_effect=AssertionError("No forward")):
        report = command.create_scientific_command_v1(
            config_path=config, output_root=tmp_path / "runs"
        )
    assert report.sequence == report.state.step == 0
    assert report.state.phase == "train" and report.purpose == "smoke"
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    tok, lm = loaders
    assert tok.call_args.kwargs["local_files_only"] is True
    assert tok.call_args.kwargs["revision"] == "2" * 40
    assert lm.call_args.kwargs["local_files_only"] is True
    assert lm.call_args.kwargs["trust_remote_code"] is False
    assert lm.call_args.kwargs["use_safetensors"] is True
    assert lm.call_args.kwargs["revision"] == "1" * 40
    assert (
        command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
        == report
    )


@pytest.mark.parametrize("cut", [1, 3, 5, 7])
def test_fresh_composition_resume_matches_uninterrupted_bytes(
    tmp_path: Path,
    loaders: Any,
    cut: int,
) -> None:
    config = _recipe(tmp_path)
    left = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "left")
    right = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "right")
    before = capture_runtime_state(torch.device("cpu"))
    complete = command.resume_scientific_command_v1(
        config_path=config, execution_path=left.path, phases=1000
    )
    partial = command.resume_scientific_command_v1(
        config_path=config, execution_path=right.path, phases=cut
    )
    assert partial.sequence == cut
    resumed = command.resume_scientific_command_v1(
        config_path=config, execution_path=right.path, phases=1000
    )
    assert complete.state == resumed.state
    assert resumed.state.phase == "stopped" and resumed.state.step == 3
    assert resumed.sequence == 8
    assert _files(left.path) == _files(right.path)
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    assert (
        command.resume_scientific_command_v1(
            config_path=config, execution_path=right.path, phases=1
        )
        == resumed
    )


def test_cli_create_resume_inspect_are_real_redacted_services(tmp_path: Path, loaders: Any) -> None:
    config = _recipe(tmp_path)
    runner = CliRunner()
    created = runner.invoke(
        app,
        ["scientific", "create", "--config", str(config), "--output-root", str(tmp_path / "runs")],
    )
    assert created.exit_code == 0, created.output
    execution = next((tmp_path / "runs" / "scientific-executions-v1").iterdir())
    resumed = runner.invoke(
        app,
        [
            "scientific",
            "resume",
            "--config",
            str(config),
            "--execution",
            str(execution),
            "--phases",
            "1",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert read_scientific_execution_v1(execution).events[-1].payload.after.phase == "validation"
    before = _files(execution)
    with patch.object(command, "BackboneAdapter", side_effect=AssertionError("No loader")):
        inspected = runner.invoke(app, ["scientific", "inspect", "--execution", str(execution)])
    assert inspected.exit_code == 0, inspected.output
    assert _files(execution) == before
    for result in (created, resumed, inspected):
        assert "not scientific acceptance" in result.output
        for row in _rows()[:5]:
            assert row.prompt not in result.output and row.expected_response not in result.output


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["backbone"].update(local_files_only=False),
        lambda c: c["backbone"].update(trust_remote_code=True),
        lambda c: c["backbone"].update(revision="main"),
        lambda c: c["backbone"].update(tokenizer_revision="main"),
        lambda c: c["backbone"].update(repository="C:/local/model"),
        lambda c: c["backbone"].update(dtype="bf16"),
        lambda c: c["backbone"]["tokenizer"].update(use_fast=False),
        lambda c: c.update(device="cuda"),
        lambda c: c.update(initialization_seed=True),
        lambda c: c.update(run_seed="42"),
        lambda c: c.update(deterministic=1),
        lambda c: c.update(data_scope="scientific_approved"),
        lambda c: c.update(unknown_option=True),
        lambda c: c["validation"].update(maximum_rows=5001),
        lambda c: c["train"].update(path="https://example.com/data.parquet"),
        lambda c: c["train"].update(path="   "),
        lambda c: c["lifecycle"].update(purpose="scientific"),
    ],
)
def test_invalid_recipe_rejected_before_load(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    config = _recipe(tmp_path)
    _edit(config, mutation)
    with patch.object(command, "BackboneAdapter") as loader:
        with pytest.raises(command.IrisScientificCommandError, match="configuration is invalid"):
            command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
        loader.assert_not_called()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "content",
    ['{"device":"cpu","device":"cuda"}', '{"x":NaN}', "[]", "x" * 131073],
    ids=["duplicate-fields", "nan", "array", "over-budget"],
)
def test_json_parser_rejects_duplicates_constants_shape_and_size(
    tmp_path: Path, content: str
) -> None:
    config = tmp_path / "bad.json"
    config.write_text(content, encoding="utf-8")
    with pytest.raises(command.IrisScientificCommandError):
        command.read_scientific_command_config_v1(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["train"].update(sha256="0" * 64),
        lambda c: c["train"].update(maximum_file_bytes=1),
        lambda c: c["train"].update(maximum_logical_bytes=1),
        lambda c: c["train"].update(maximum_rows=1),
        lambda c: c.update(train=c["validation"]),
        lambda c: c.update(validation=c["train"]),
    ],
)
def test_input_failures_precede_model_load(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    config = _recipe(tmp_path)
    _edit(config, mutation)
    with patch.object(command, "BackboneAdapter") as loader:
        with pytest.raises(command.IrisScientificCommandError, match="creation failed"):
            command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
        loader.assert_not_called()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["backbone"]["limits"].update(max_prompt_tokens=8),
        lambda c: c["backbone"]["limits"].update(max_new_tokens=2),
        lambda c: c["backbone"]["limits"].update(max_prompt_tokens=2040),
        lambda c: c.update(terminal_token_id=0),
    ],
)
def test_complete_context_and_eos_fail_before_forward_or_publication(
    tmp_path: Path,
    loaders: Any,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    config = _recipe(tmp_path)
    _edit(config, mutation)
    before = capture_runtime_state(torch.device("cpu"))
    with patch("transformers.Qwen3ForCausalLM.forward") as forward:
        with pytest.raises(command.IrisScientificCommandError, match="creation failed"):
            command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
        forward.assert_not_called()
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("field", ["initialization_seed", "probe_initialization_seed", "run_seed"])
def test_changed_recipe_initial_state_rejected_before_restore(
    tmp_path: Path,
    loaders: Any,
    field: str,
) -> None:
    config = _recipe(tmp_path)
    report = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
    before = _files(report.path)
    _edit(config, lambda c: c.update({field: c[field] + 1}))
    with patch.object(command, "resume_scientific_execution_v1") as restore:
        with pytest.raises(command.IrisScientificCommandError, match="resume failed"):
            command.resume_scientific_command_v1(
                config_path=config, execution_path=report.path, phases=1
            )
        restore.assert_not_called()
    assert _files(report.path) == before


@pytest.mark.parametrize("exception", [RuntimeError("PRIVATE_LOADER_DETAIL"), KeyboardInterrupt()])
def test_failed_or_cancelled_loader_restores_runtime(
    tmp_path: Path, exception: BaseException
) -> None:
    config = _recipe(tmp_path)
    before = capture_runtime_state(torch.device("cpu"))

    def fail(*args: Any, **kwargs: Any) -> None:
        random.random()
        torch.rand(7)
        raise exception

    with patch.object(command, "BackboneAdapter", side_effect=fail):
        expected = (
            KeyboardInterrupt
            if isinstance(exception, KeyboardInterrupt)
            else command.IrisScientificCommandError
        )
        with pytest.raises(expected) as caught:
            command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
    assert "PRIVATE_LOADER_DETAIL" not in str(caught.value)
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("phases", [None, "0", "1001"])
def test_cli_requires_bounded_explicit_phase_budget(phases: str | None) -> None:
    arguments = ["scientific", "resume", "--config", "missing.json", "--execution", "missing"]
    if phases is not None:
        arguments += ["--phases", phases]
    with patch.object(command, "resume_scientific_command_v1") as resume:
        result = CliRunner().invoke(app, arguments)
        resume.assert_not_called()
    assert result.exit_code == 2


def test_corrupt_journal_rejected_before_model_load(tmp_path: Path, loaders: Any) -> None:
    config = _recipe(tmp_path)
    report = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
    (report.path / "run.json").write_bytes(b"{}\n")
    with patch.object(command, "BackboneAdapter") as loader:
        with pytest.raises(command.IrisScientificCommandError):
            command.resume_scientific_command_v1(
                config_path=config, execution_path=report.path, phases=1
            )
        loader.assert_not_called()


def test_exact_command_resume_in_a_separate_process(tmp_path: Path, loaders: Any) -> None:
    config = _recipe(tmp_path)
    left = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "left")
    right = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "right")
    command.resume_scientific_command_v1(config_path=config, execution_path=left.path, phases=1000)
    command.resume_scientific_command_v1(config_path=config, execution_path=right.path, phases=3)
    script = """
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from iris.application import scientific_command as command
from tests.integration.test_scientific_teacher import _build
assert Path(command.__file__).resolve().is_relative_to(Path(sys.argv[3]))
_, tokenizer, model = _build(cycles=1, slots=2)
with (
    patch('iris.backbone.adapter.AutoTokenizer.from_pretrained',
          side_effect=lambda *a, **k: deepcopy(tokenizer)),
    patch('iris.backbone.adapter.AutoModelForCausalLM.from_pretrained',
          side_effect=lambda *a, **k: deepcopy(model)),
):
    report = command.resume_scientific_command_v1(
        config_path=Path(sys.argv[1]), execution_path=Path(sys.argv[2]), phases=1000,
    )
assert report.state.phase == 'stopped' and report.state.step == 3
print(report.execution_sha256)
"""
    project = Path(__file__).resolve().parents[2]
    package_root = Path(command.__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script, str(config), str(right.path), str(package_root)],
        cwd=project,
        env=os.environ
        | {
            "PYTHONPATH": os.pathsep.join((str(package_root), str(project))),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == right.execution_sha256
    assert _files(left.path) == _files(right.path)


def test_deterministic_false_initialization_does_not_inherit_benchmark_flag(
    tmp_path: Path,
    loaders: Any,
) -> None:
    config = _recipe(tmp_path)
    _edit(config, lambda c: c.update(deterministic=False))
    original = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        left = command.create_scientific_command_v1(
            config_path=config, output_root=tmp_path / "left"
        )
        assert torch.backends.cudnn.benchmark is True
        torch.backends.cudnn.benchmark = False
        right = command.create_scientific_command_v1(
            config_path=config, output_root=tmp_path / "right"
        )
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.benchmark = original
    assert _files(left.path) == _files(right.path)


@pytest.mark.parametrize(
    "path", ["//server/data.parquet", "\\\\server\\data.parquet", "C:relative.parquet"]
)
def test_network_and_drive_relative_dataset_paths_rejected(tmp_path: Path, path: str) -> None:
    config = _recipe(tmp_path)
    _edit(config, lambda c: c["train"].update(path=path))
    with pytest.raises(command.IrisScientificCommandError):
        command.read_scientific_command_config_v1(config)


def test_scientific_counts_are_not_silently_replaced_with_smoke(tmp_path: Path) -> None:
    config = _recipe(tmp_path)

    def change(raw: dict[str, Any]) -> None:
        raw["train"]["maximum_rows"] = 50000
        raw["validation"]["maximum_rows"] = 5000
        raw["optimizer"].update(purpose="scientific", minimum_target_tokens=16384)
        raw["lifecycle"]["purpose"] = "scientific"

    _edit(config, change)
    with patch.object(command, "BackboneAdapter") as loader:
        with pytest.raises(command.IrisScientificCommandError):
            command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
        loader.assert_not_called()


def test_reordered_pinned_data_rejected_before_restore(tmp_path: Path, loaders: Any) -> None:
    config = _recipe(tmp_path)
    report = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
    before = _files(report.path)
    updated = _input(tmp_path / "reordered.parquet", tuple(reversed(_rows()[:5])))
    _edit(config, lambda c: c.update(train=updated))
    with patch.object(command, "resume_scientific_execution_v1") as restore:
        with pytest.raises(command.IrisScientificCommandError):
            command.resume_scientific_command_v1(
                config_path=config, execution_path=report.path, phases=1
            )
        restore.assert_not_called()
    assert _files(report.path) == before


def test_identical_data_can_be_relocated(tmp_path: Path, loaders: Any) -> None:
    config = _recipe(tmp_path)
    report = command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in ("train.parquet", "validation.parquet", "command.json"):
        (relocated / name).write_bytes((tmp_path / name).read_bytes())
    result = command.resume_scientific_command_v1(
        config_path=relocated / "command.json", execution_path=report.path, phases=1
    )
    assert result.sequence == 1 and result.state.step == 1


def test_shadowing_local_repository_is_rejected_before_load(tmp_path: Path) -> None:
    config = _recipe(tmp_path)
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        return str(path).replace("\\", "/") == "offline/random-qwen3" or original_exists(path)

    with patch.object(Path, "exists", exists), patch.object(command, "BackboneAdapter") as loader:
        with pytest.raises(command.IrisScientificCommandError):
            command.create_scientific_command_v1(config_path=config, output_root=tmp_path / "runs")
        loader.assert_not_called()
