"""Source-style argument contracts, including every section18 operation."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iris import application
from iris.application import intervention_command as command
from iris.cli import app


@pytest.mark.parametrize(
    ("kind", "parameters", "expected"),
    [
        ("zero_workspace", (), {}),
        ("zero_slot", ("slot_id=0",), {"slot": 0}),
        ("freeze_slot", ("slot_id=1",), {"slot": 1}),
        ("swap_slots", ("slot_id=0", "other_slot_id=1"), {"slot": 0, "other_slot": 1}),
        ("shuffle_slots", (), {"seed": 42}),
        ("gaussian_noise", ("sigma=0.25",), {"sigma": 0.25, "seed": 42}),
        (
            "replace_slot",
            ("slot_id=1", "values=[0.25,-0.5]", "source_module=0", "confidence=0.8", "ttl=2"),
            {"slot": 1, "values": (0.25, -0.5), "source_module": 0, "confidence": 0.8, "ttl": 2},
        ),
        ("block_module_write", ("module_id=3",), {"module": 3}),
        ("block_module_read", ("module_id=2",), {"module": 2}),
        ("force_module_winner", ("module_id=1",), {"module": 1}),
        ("scale_module_bid", ("module_id=1", "factor=0.5"), {"module": 1, "factor": 0.5}),
        ("reduce_workspace_capacity", ("capacity=1",), {"capacity": 1}),
    ],
)
def test_all_source_operations(
    kind: str, parameters: tuple[str, ...], expected: dict[str, Any]
) -> None:
    values = command._arguments(
        (
            "checkpoint=checkpoints/igw-best",
            f"intervention={kind}",
            *(f"intervention.{item}" for item in parameters),
        )
    )
    plan = command._plan(values, seed=42)
    assert plan.interventions[0].model_dump(exclude_none=True) == {"kind": kind, **expected}
    plan.validate_dimensions(num_slots=2, num_modules=4, state_dim=2, cycles=3, winners_per_cycle=2)


def test_explicit_seed_and_cycle_schedule() -> None:
    values = command._arguments(
        (
            "checkpoint=x",
            "intervention=shuffle_slots",
            "intervention.seed=1337",
            "intervention.cycles=[1,3]",
        )
    )
    spec = command._plan(values, seed=42).interventions[0]
    assert spec.seed == 1337 and spec.cycles == (1, 3)


def test_explicit_nll_modes_require_a_saved_capture_or_report() -> None:
    assert (
        command._arguments(("mode=measure-nll", "checkpoint=x", "capture=y"))["mode"]
        == "measure-nll"
    )
    assert command._arguments(("mode=inspect-nll", "report=y"))["mode"] == "inspect-nll"
    for values in (
        ("mode=measure-nll", "checkpoint=x"),
        ("mode=measure-nll", "capture=y"),
        ("mode=measure-nll", "checkpoint=x", "capture=y", "intervention=zero_workspace"),
        ("mode=inspect-nll", "report=y", "context=x"),
    ):
        with pytest.raises(ValueError):
            command._arguments(values)


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("checkpoint=x",),
        ("checkpoint=x", "checkpoint=y", "intervention=zero_workspace"),
        ("checkpoint=x", "intervention=zero_workspace", "PRIVATE_MARKER=yes"),
        ("checkpoint=", "intervention=zero_workspace"),
        ("PRIVATE_MARKER",),
        ("mode=inspect", "report=x", "checkpoint=PRIVATE_MARKER"),
        ("mode=evaluate", "checkpoint=x", "capture=x", "intervention=PRIVATE_MARKER"),
        ("mode=PRIVATE_MARKER",),
        ("checkpoint=x\0", "intervention=zero_workspace"),
        ("checkpoint=x", "intervention=zero_workspace", "intervention.module=0"),
        ("checkpoint=x" + "a" * 65536, "intervention=zero_workspace"),
    ],
)
def test_invalid_arguments_fail_without_io_and_are_redacted(arguments: tuple[str, ...]) -> None:
    with (
        patch.object(command, "_read_context") as read,
        pytest.raises(command.IrisInterventionCommandError) as caught,
    ):
        command.run_intervention_command_v1(arguments)
    read.assert_not_called()
    assert "PRIVATE_MARKER" not in str(caught.value)


@pytest.mark.parametrize(
    "arguments",
    [
        ("intervention=zero_workspace", "intervention.module_id=null"),
        ("intervention=zero_workspace", "intervention.sigma=1"),
        ("intervention=gaussian_noise", "intervention.sigma=NaN"),
        ("intervention=gaussian_noise", "intervention.sigma=-1"),
        ("intervention=block_module_write", "intervention.module_id=true"),
        ("intervention=block_module_write", "intervention.module_id=1.5"),
        ("intervention=block_module_write",),
        ("intervention=zero_slot", "intervention.slot_id=-1"),
        ("intervention=zero_workspace", "intervention.cycles=[1,1]"),
        ("intervention=zero_workspace", "intervention.cycles=[]"),
        ("intervention=zero_workspace", "intervention.cycles=[0]"),
        ("intervention=shuffle_slots", "intervention.seed=9223372036854775808"),
        ("intervention=PRIVATE_MARKER",),
    ],
)
def test_wrong_or_inapplicable_parameters_are_not_ignored(arguments: tuple[str, ...]) -> None:
    values = command._arguments(("checkpoint=x", *arguments))
    with pytest.raises(ValueError):
        command._plan(values, seed=42)


@pytest.mark.parametrize(
    "path", ["", " ", "x ", "https://PRIVATE_MARKER", "\\\\server\\share", "C:foo", "x\0"]
)
def test_nonlocal_or_ambiguous_paths_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        command._path(Path.cwd(), path)


def test_help_does_not_load_context_or_models_and_public_api_is_exported() -> None:
    with patch.object(command, "run_intervention_command_v1") as execute:
        result = CliRunner().invoke(app, ["intervene", "--help"])
    assert result.exit_code == 0, result.output
    assert "key=value" in result.output and "downloads." in result.output
    execute.assert_not_called()
    for name in command.__all__:
        assert getattr(application, name) is getattr(command, name)
