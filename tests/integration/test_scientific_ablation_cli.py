"""Actual registered CLI through offline native phases, persisted best and export."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM
from typer.testing import CliRunner

from iris.application import scientific_ablation_operator as operator
from iris.artifacts.scientific_ablation_execution_v1 import ScientificAblationExecutionCheckpointV1
from iris.cli import app
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_execution import _config, _tree
from tests.integration.test_scientific_ablation_execution import cache_ready as cache_ready

pytestmark = pytest.mark.integration
runner = CliRunner()


def _call(*arguments: str) -> dict[str, Any]:
    result = runner.invoke(app, ["scientific-ablation", *arguments])
    assert result.exit_code == 0, result.output
    value: dict[str, Any] = json.loads(result.stdout)
    return value


def _ids(root: Path) -> dict[str, tuple[int, int]]:
    return {
        name: (meta.st_dev, meta.st_ino) for name in _tree(root) for meta in [(root / name).lstat()]
    }


@pytest.mark.parametrize("cycles", [1, 5])
def test_actual_cli_phase_intents_late_resume_best_and_both_exports(
    tmp_path: Path,
    cache_ready: Path,
    cycles: int,
) -> None:
    config = _config(tmp_path, cycles=cycles)
    recipe = tmp_path / "recipe.json"
    recipe.write_bytes(canonical_json_file_bytes(config.model_dump(mode="json")))
    created = _call(
        "new",
        "--recipe",
        str(recipe),
        "--base",
        str(tmp_path),
        "--root",
        str(tmp_path / "operators"),
    )
    path = Path(created["operator_path"])
    assert created["acknowledged_phases"] == created["step"] == 0
    saved_requests = []
    actual_results = []
    for index in range(8):
        before = _call("inspect", "--operator", str(path))
        phase = before["phase"]
        saved = tmp_path / "intents" / f"phase-{index + 1}.json"
        intent = _call(
            "phase-request",
            "--operator",
            str(path),
            "--expected-previous-sha256",
            before["head_sha256"],
            "--phase",
            phase,
            "--out",
            str(saved),
        )
        assert json.loads(saved.read_bytes()) == intent
        command = "validate" if phase == "validation" else phase
        response = _call(
            command, "--operator", str(path), "--request", str(saved), "--base", str(tmp_path)
        )
        assert response["payload"]["request"] == intent
        saved_requests.append(saved)
        actual_results.append(response)
        with patch(
            "iris.application.scientific_ablation_operator.create_scientific_ablation_command_lifecycle_v1",
            side_effect=AssertionError("No native retry"),
        ) as factory:
            repeat = _call(
                "resume", "--operator", str(path), "--request", str(saved), "--base", str(tmp_path)
            )
        assert factory.call_count == 0 and repeat == response
    final = _call("inspect", "--operator", str(path))
    assert final["step"] == 3 and final["best_step"] == 1 and final["phase"] == "stopped"
    before_tree, before_ids = _tree(path), _ids(path)
    with patch(
        "iris.application.scientific_ablation_operator.create_scientific_ablation_command_lifecycle_v1",
        side_effect=AssertionError("Late DATA retry"),
    ) as factory:
        for saved, response in zip(saved_requests, actual_results, strict=True):
            assert (
                _call(
                    "resume",
                    "--operator",
                    str(path),
                    "--request",
                    str(saved),
                    "--base",
                    str(tmp_path),
                )
                == response
            )
        assert (
            _call(
                "new",
                "--recipe",
                str(recipe),
                "--base",
                str(tmp_path),
                "--root",
                str(tmp_path / "operators"),
            )["operator_sha256"]
            == created["operator_sha256"]
        )
    assert factory.call_count == 0 and _tree(path) == before_tree and _ids(path) == before_ids
    best = _call("best", "--operator", str(path))
    assert (
        best["payload"]["validated_best_step"] == 1
        and best["payload"]["validated_best_is_persisted"] is True
    )
    snapshot = operator.inspect_scientific_ablation_operator_v1(path)
    event = snapshot.execution.events[-1]
    record = event.payload.record
    assert type(record) is ScientificAblationExecutionCheckpointV1
    exported = []
    for selection in ("current", "selected_best"):
        saved = tmp_path / "intents" / f"export-{selection}.json"
        intent = _call(
            "export-request",
            "--operator",
            str(path),
            "--checkpoint-event-sha256",
            event.sha256,
            "--lifecycle-sha256",
            record.lifecycle_checkpoint_sha256,
            "--selection",
            selection,
            "--out",
            str(saved),
            "--maximum-storage-bytes",
            str(4 * 1024**2),
            "--minimum-free-bytes",
            "0",
        )
        assert intent == json.loads(saved.read_bytes())
        with patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("Export never forwards")
        ) as forward:
            response = _call(
                "export",
                "--operator",
                str(path),
                "--request",
                str(saved),
                "--base",
                str(tmp_path),
                "--root",
                str(tmp_path / "exports"),
            )
        assert forward.call_count == 0
        assert response["result"]["payload"]["selected_completed_steps"] == (
            3 if selection == "current" else 1
        )
        export_tree, export_ids = _tree(tmp_path / "exports"), _ids(tmp_path / "exports")
        with (
            patch(
                "iris.application.scientific_ablation_operator_export.create_scientific_ablation_command_run_v1",
                side_effect=AssertionError("Export DATA retry"),
            ) as factory,
            patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        ):
            assert (
                _call(
                    "export",
                    "--operator",
                    str(path),
                    "--request",
                    str(saved),
                    "--base",
                    str(tmp_path),
                    "--root",
                    str(tmp_path / "exports"),
                )
                == response
            )
        assert factory.call_count == graph.call_count == 0
        with (
            patch(
                "iris.application.scientific_ablation_operator_export.create_scientific_ablation_command_run_v1",
                side_effect=AssertionError("Inspection must never construct"),
            ) as inspect_factory,
            patch(
                "torch.nn.Module.__init__", side_effect=AssertionError("No inspection graph")
            ) as inspect_graph,
        ):
            inspected = _call(
                "inspect-export",
                "--operator",
                str(path),
                "--request",
                str(saved),
                "--root",
                str(tmp_path / "exports"),
            )
        assert inspect_factory.call_count == inspect_graph.call_count == 0
        assert inspected["read_only"] is True and inspected["result"] == response["result"]
        assert (
            _tree(tmp_path / "exports") == export_tree and _ids(tmp_path / "exports") == export_ids
        )
        exported.append(response)
    assert _tree(path) == before_tree and _ids(path) == before_ids and cache_ready.is_dir()
    (tmp_path / "actual-ablation-cli-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-scientific-ablation-cli-native-witness-v1",
                operator_path=str(path),
                cycles=cycles,
                phases=8,
                requests=[json.loads(p.read_bytes()) for p in saved_requests],
                results=actual_results,
                exports=exported,
                all_native_phase_retries_data_only=True,
                all_export_retries_data_only=True,
                all_source_and_export_bytes_and_native_ids_stable=True,
                installed_or_scientific_acceptance=False,
            )
        )
    )
