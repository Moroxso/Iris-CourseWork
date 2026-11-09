"""Real current/best delivery cadence, receipt integrity and independent retries."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from iris.application import scientific_command as command
from iris.application import scientific_publication as publication
from iris.application.scientific_checkpoint import build_scientific_checkpoint_v1
from iris.application.scientific_delivery import write_scientific_delivery_v1
from iris.application.scientific_execution import (
    create_scientific_execution_v1,
    read_scientific_execution_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunV1
from iris.artifacts.scientific_lifecycle_v1 import IrisScientificLifecyclePolicyV1
from iris.cli import app
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_checkpoint import _run, _stochastic
from tests.integration.test_scientific_command import _edit, _recipe
from tests.integration.test_scientific_command import loaders as loaders
from tests.integration.test_scientific_lifecycle import _lifecycle, _short_lifecycle

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    lifecycle = _lifecycle()
    owner = create_scientific_execution_v1(
        lifecycle, root=tmp_path_factory.mktemp("publication-source")
    )
    with _stochastic(lifecycle._run):
        while lifecycle.state.phase != "stopped":
            owner.advance()
    return owner.path


def _publish(
    completed: Path, root: Path, **kwargs: Any
) -> publication.IrisScientificPublicationReportV1:
    return publication.publish_scientific_execution_deliveries_v1(
        execution_path=completed,
        context=_run(),
        output_root=root,
        **kwargs,
    )


def test_all_committed_current_best_pairs_are_exact_idempotent_and_nonmutating(
    tmp_path: Path,
    completed: Path,
) -> None:
    context = _run()
    before = build_scientific_checkpoint_v1(context)
    ambient = capture_runtime_state(torch.device("cpu"))
    source = read_scientific_execution_v1(completed)
    with patch.object(context, "step", side_effect=AssertionError("No training")):
        report = publication.publish_scientific_execution_deliveries_v1(
            execution_path=completed,
            context=context,
            output_root=tmp_path,
        )
    assert report.complete and report.checkpoint_count == len(report.receipts) == 2
    assert build_scientific_checkpoint_v1(context) == before
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    assert read_scientific_execution_v1(completed) == source
    best_deliveries = []
    for path in report.receipts:
        payload = json.loads(path.read_bytes())["payload"]
        assert payload["current"]["checkpoint_sha256"] != payload["best"]["checkpoint_sha256"]
        best_deliveries.append(payload["best"]["delivery_sha256"])
        best_path = tmp_path / "scientific-delivery-v1" / payload["best"]["delivery_sha256"]
        assert (
            json.loads((best_path / "metrics.json").read_bytes())["validation"]["at_optimizer_step"]
            == 1
        )
    assert len(set(best_deliveries)) == 1
    assert len(list((tmp_path / "scientific-delivery-v1").iterdir())) == 3
    with patch.object(
        publication, "write_scientific_delivery_v1", side_effect=AssertionError("Already published")
    ):
        assert (
            _publish(
                completed,
                tmp_path,
                maximum_delivery_storage_bytes=1,
                maximum_receipt_storage_bytes=1,
            )
            == report
        )
    assert (
        publication.inspect_scientific_publications_v1(
            execution_path=completed, output_root=tmp_path
        )
        == report
    )


def test_no_checkpoint_has_no_delivery_side_effect(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    source = create_scientific_execution_v1(lifecycle, root=tmp_path / "source")
    root = tmp_path / "output"
    report = _publish(source.path, root)
    assert report.complete and report.checkpoint_count == 0
    assert not root.exists()


@pytest.mark.parametrize("shared", [False, True])
def test_no_best_and_shared_best_current_are_explicit(tmp_path: Path, shared: bool) -> None:
    lifecycle = _short_lifecycle()
    if shared:
        # Recompose its fresh owner under a matching validation/checkpoint cadence.
        lifecycle = type(lifecycle)(
            run=lifecycle._run,
            validation=lifecycle._validation,
            policy=IrisScientificLifecyclePolicyV1(
                purpose="smoke", validate_every_steps=1, checkpoint_every_steps=1
            ),
        )
    source = create_scientific_execution_v1(lifecycle, root=tmp_path / "source")
    while lifecycle.state.checkpoint_count == 0:
        source.advance()
    report = publication.publish_scientific_execution_deliveries_v1(
        execution_path=source.path,
        context=lifecycle._run,
        output_root=tmp_path / "output",
    )
    payload = json.loads(report.receipts[0].read_bytes())["payload"]
    if shared:
        assert payload["current"] == payload["best"]
    else:
        assert payload["best"] is None and payload["current"]["validation_event_sha256"] is None
    assert len(list((tmp_path / "output/scientific-delivery-v1").iterdir())) == 1


def test_partial_pair_publication_retry_does_not_advance_source(
    tmp_path: Path, completed: Path
) -> None:
    original = write_scientific_delivery_v1
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture full disk")
        return original(*args, **kwargs)

    before = read_scientific_execution_v1(completed)
    with (
        patch.object(publication, "write_scientific_delivery_v1", side_effect=fail_second),
        pytest.raises(publication.IrisScientificPublicationError),
    ):
        _publish(completed, tmp_path)
    partial = publication.inspect_scientific_publications_v1(
        execution_path=completed, output_root=tmp_path
    )
    assert not partial.complete and not partial.receipts
    assert len(list((tmp_path / "scientific-delivery-v1").iterdir())) == 1
    assert _publish(completed, tmp_path).complete
    assert read_scientific_execution_v1(completed) == before


@pytest.mark.parametrize("cancel", [False, True])
def test_receipt_lost_acknowledgement_is_retryable(
    tmp_path: Path, completed: Path, cancel: bool
) -> None:
    original = os.link

    def lost(source: Path, target: Path) -> None:
        original(source, target)
        if cancel:
            raise KeyboardInterrupt
        raise OSError("lost success acknowledgement")

    with patch.object(os, "link", side_effect=lost):
        if cancel:
            with pytest.raises(KeyboardInterrupt):
                _publish(completed, tmp_path)
        else:
            assert _publish(completed, tmp_path).complete
    assert _publish(completed, tmp_path).complete


@pytest.mark.parametrize("corruption", ["event", "metric", "swap", "weights", "extra"])
def test_rehashed_receipts_and_delivered_bytes_are_audited(
    tmp_path: Path,
    completed: Path,
    corruption: str,
) -> None:
    report = _publish(completed, tmp_path)
    path = report.receipts[0]
    raw = json.loads(path.read_bytes())
    payload = raw["payload"]
    if corruption == "event":
        payload["event_sha256"] = "0" * 64
    elif corruption == "metric":
        payload["current"]["validation_event_sha256"] = payload["best"]["validation_event_sha256"]
    elif corruption == "swap":
        payload["current"], payload["best"] = payload["best"], payload["current"]
    elif corruption == "weights":
        (
            tmp_path
            / "scientific-delivery-v1"
            / payload["best"]["delivery_sha256"]
            / "iris_model.safetensors"
        ).write_bytes(b"invalid")
    else:
        (path.parent / "00000001.json").write_bytes(path.read_bytes())
    raw["sha256"] = domain_sha256(publication._DOMAIN, canonical_json_bytes(payload))
    path.write_bytes(canonical_json_file_bytes(raw))
    with pytest.raises(publication.IrisScientificPublicationError):
        publication.inspect_scientific_publications_v1(
            execution_path=completed, output_root=tmp_path
        )
    with pytest.raises(publication.IrisScientificPublicationError):
        _publish(completed, tmp_path)


@pytest.mark.parametrize("kind", ["receipts", "deliveries", "inside-source", "context"])
def test_budget_location_and_context_fences(tmp_path: Path, completed: Path, kind: str) -> None:
    root = completed / "forbidden" if kind == "inside-source" else tmp_path / "output"
    context = _run(reverse=kind == "context")
    before = read_scientific_execution_v1(completed)
    with pytest.raises(publication.IrisScientificPublicationError):
        publication.publish_scientific_execution_deliveries_v1(
            execution_path=completed,
            context=context,
            output_root=root,
            maximum_delivery_storage_bytes=1 if kind == "deliveries" else 1024**3,
            maximum_receipt_storage_bytes=1 if kind == "receipts" else 16 * 1024**2,
        )
    assert read_scientific_execution_v1(completed) == before
    assert not list(root.rglob("*.safetensors"))


def test_failed_exclusive_stage_creation_never_cleans_collision(
    tmp_path: Path, completed: Path
) -> None:
    original = os.mkdir
    collided: list[Path] = []

    def collide(path: Any, *args: Any, **kwargs: Any) -> None:
        candidate = Path(path)
        original(path, *args, **kwargs)
        if candidate.name == ".iris-publication-" + "a" * 32 + ".tmp":
            collided.append(candidate)
            (candidate / "publication.json").write_bytes(b"pre-existing fixture")
            raise FileExistsError("not owned")

    with (
        patch.object(publication, "uuid4", return_value=SimpleNamespace(hex="a" * 32)),
        patch.object(os, "mkdir", side_effect=collide),
        pytest.raises(publication.IrisScientificPublicationError),
    ):
        _publish(completed, tmp_path)
    assert len(collided) == 1
    assert (collided[0] / "publication.json").read_bytes() == b"pre-existing fixture"


def test_cli_cadence_retry_catches_up_before_next_training(tmp_path: Path, loaders: Any) -> None:
    config = _recipe(tmp_path)
    created = command.create_scientific_command_v1(
        config_path=config, output_root=tmp_path / "source"
    )
    root = tmp_path / "deliveries"
    original = publication.publish_scientific_execution_deliveries_v1
    calls = 0

    def lost(*args: Any, **kwargs: Any) -> publication.IrisScientificPublicationReportV1:
        nonlocal calls
        calls += 1
        report = original(*args, **kwargs)
        if calls == 2:
            raise OSError("lost completed first cadence acknowledgement")
        return report

    with (
        patch.object(command, "publish_scientific_execution_deliveries_v1", side_effect=lost),
        pytest.raises(command.IrisScientificCommandError),
    ):
        command.resume_scientific_command_v1(
            config_path=config, execution_path=created.path, phases=1000, delivery_root=root
        )
    partial = read_scientific_execution_v1(created.path)
    assert len(partial.events) == 5 and partial.events[-1].payload.after.step == 2
    updates: list[int] = []
    original_step = IrisScientificRunV1.step

    def step(run: IrisScientificRunV1) -> Any:
        updates.append(run.completed_steps)
        return original_step(run)

    with patch.object(IrisScientificRunV1, "step", step):
        result = CliRunner().invoke(
            app,
            [
                "scientific",
                "resume",
                "--config",
                str(config),
                "--execution",
                str(created.path),
                "--phases",
                "1000",
                "--delivery-root",
                str(root),
            ],
        )
    assert result.exit_code == 0, result.output
    assert updates == [2]
    report = publication.inspect_scientific_publications_v1(
        execution_path=created.path, output_root=root
    )
    assert report.complete and len(report.receipts) == 2
    with patch.object(command, "BackboneAdapter", side_effect=AssertionError("No model")):
        inspected = CliRunner().invoke(
            app,
            [
                "scientific",
                "inspect-deliveries",
                "--execution",
                str(created.path),
                "--delivery-root",
                str(root),
            ],
        )
    assert inspected.exit_code == 0, inspected.output
    assert "not scientific acceptance" in inspected.output


def test_failed_receipt_link_keeps_deliveries_retryable(tmp_path: Path, completed: Path) -> None:
    with (
        patch.object(os, "link", side_effect=OSError("fixture unavailable hard links")),
        pytest.raises(publication.IrisScientificPublicationError),
    ):
        _publish(completed, tmp_path)
    partial = publication.inspect_scientific_publications_v1(
        execution_path=completed,
        output_root=tmp_path,
    )
    assert not partial.complete and not partial.receipts
    assert len(list((tmp_path / "scientific-delivery-v1").iterdir())) == 2
    assert _publish(completed, tmp_path).complete


def test_scientific_purpose_cannot_disable_checkpoint_delivery(tmp_path: Path) -> None:
    config = _recipe(tmp_path)

    def scientific(raw: dict[str, Any]) -> None:
        raw["train"]["maximum_rows"] = 50000
        raw["validation"]["maximum_rows"] = 5000
        raw["optimizer"].update(purpose="scientific", minimum_target_tokens=16384)
        raw["lifecycle"]["purpose"] = "scientific"

    _edit(config, scientific)
    with patch.object(command, "BackboneAdapter") as load:
        with pytest.raises(command.IrisScientificCommandError, match="delivery root"):
            command.resume_scientific_command_v1(
                config_path=config,
                execution_path=tmp_path / "not-read",
                phases=1,
            )
        load.assert_not_called()
