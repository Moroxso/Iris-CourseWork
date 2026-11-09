"""Explicit expected phase/prefix controller tests, never native execution proof."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as module
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowUncertainError,
    ScientificAblationWorkflowStoreV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from tests.unit._workflow_data_setup import plan


def unused_factory() -> NoReturn:
    raise AssertionError("Labelled controller tests never construct a native graph")


def job_and_phase(
    tmp_path: Path,
) -> tuple[module.IrisScientificAblationWorkflowV1, ScientificAblationWorkflowPhaseV1]:
    intent = plan(tmp_path)
    return module.create_scientific_ablation_workflow_v1(
        root=tmp_path / "output", plan=intent, make_fresh_run=unused_factory
    ), scientific_ablation_workflow_phases_v1(intent)[0]


def probe(
    job: object,
    phase: object,
    anchors: object,
    rows: object = None,
    method: str = "advance_expected",
) -> object:
    """Intentionally ill-typed runtime admission probes; no invented native owner."""
    return getattr(job, method)(
        expected_phase=phase, expected_completed_phase_sha256=anchors, rows_per_call=rows
    )


@pytest.mark.parametrize("index", range(1, 6))
def test_other_due_phase_kinds_are_rejected_before_claim_or_original_work(
    tmp_path: Path, index: int
) -> None:
    job, _ = job_and_phase(tmp_path)
    phase = scientific_ablation_workflow_phases_v1(plan(tmp_path))[index]
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        patch.object(steps, "execute") as execute,
        pytest.raises(module.IrisScientificAblationWorkflowError),
    ):
        job.advance_expected(expected_phase=phase, expected_completed_phase_sha256=())
    assert claim.call_count == execute.call_count == 0
    assert not (tmp_path / "output").exists() and job._state == "idle"


def test_matching_name_kind_but_wrong_index_is_not_the_expected_phase(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    wrong = ScientificAblationWorkflowPhaseV1.model_validate(
        phase.model_dump(mode="python") | {"index": 1}
    )
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(module.IrisScientificAblationWorkflowError),
    ):
        job.advance_expected(expected_phase=wrong, expected_completed_phase_sha256=())
    assert claim.call_count == 0 and not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "anchors",
    (
        None,
        [],
        ["0" * 64],
        (True,),
        (b"0" * 64,),
        (1,),
        ("bad",),
        tuple("0" * 64 for _ in range(135)),
    ),
)
def test_prefix_must_be_exact_bounded_immutable_external_digests(
    tmp_path: Path, anchors: object
) -> None:
    job, phase = job_and_phase(tmp_path)
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises((TypeError, ValueError)),
    ):
        probe(job, phase, anchors)
    assert claim.call_count == 0 and not (tmp_path / "output").exists() and job._state == "idle"


def test_different_wellformed_independent_prefix_is_rejected_before_claim(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(module.IrisScientificAblationWorkflowError),
    ):
        job.advance_expected(expected_phase=phase, expected_completed_phase_sha256=("0" * 64,))
    assert claim.call_count == 0 and not (tmp_path / "output").exists()


@pytest.mark.parametrize("rows", (True, False, 0, -1, 5001, 1.0, "1"))
def test_wrong_explicit_chunk_is_still_rejected_before_claim(tmp_path: Path, rows: object) -> None:
    job, phase = job_and_phase(tmp_path)
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(ValueError),
    ):
        probe(job, phase, (), rows)
    assert claim.call_count == 0 and not (tmp_path / "output").exists() and job._state == "idle"


@pytest.mark.parametrize("kind", ("hidden", "private", "unchecked", "subclass", "object"))
def test_unchecked_or_hidden_phase_objects_do_not_get_laundered_by_model_dump(
    tmp_path: Path, kind: str
) -> None:
    job, phase = job_and_phase(tmp_path)
    value: object = phase
    if kind == "hidden":
        phase.__dict__["unexpected"] = 1
    elif kind == "private":
        object.__setattr__(phase, "__pydantic_private__", {"hidden": True})
    elif kind == "unchecked":
        value = ScientificAblationWorkflowPhaseV1.model_construct(
            index=True, name="ordinary-capture", kind="capture", stream_index=None
        )
    elif kind == "subclass":

        class Other(ScientificAblationWorkflowPhaseV1):
            pass

        value = Other.model_validate(phase.model_dump(mode="python"))
    else:
        value = object()
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises((TypeError, ValueError)),
    ):
        probe(job, value, ())
    assert claim.call_count == 0 and not (tmp_path / "output").exists()


def test_unsealed_and_busy_owners_are_rejected_before_claim(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    fake = object.__new__(module.IrisScientificAblationWorkflowV1)
    with patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim:
        with pytest.raises(module.IrisScientificAblationWorkflowError):
            fake.advance_expected(expected_phase=phase, expected_completed_phase_sha256=())
        job._state = "busy"
        with pytest.raises(module.IrisScientificAblationWorkflowError):
            job.advance_expected(expected_phase=phase, expected_completed_phase_sha256=())
    assert claim.call_count == 0 and not (tmp_path / "output").exists()


def test_manufactured_complete_prefix_cannot_silently_turn_dispatch_into_noop(
    tmp_path: Path,
) -> None:
    job, phase = job_and_phase(tmp_path)
    anchors = tuple(f"{index:064x}" for index in range(6))
    job._seal = replace(job._seal, anchors=anchors)
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim") as claim,
        pytest.raises(module.IrisScientificAblationWorkflowError),
    ):
        job.advance_expected(expected_phase=phase, expected_completed_phase_sha256=anchors)
    assert claim.call_count == 0 and not (tmp_path / "output").exists()


def test_expected_phase_is_snapshotted_and_checked_again_inside_original_claim(
    tmp_path: Path,
) -> None:
    job, phase = job_and_phase(tmp_path)
    original_claim = ScientificAblationWorkflowStoreV1.claim
    seen: list[dict[str, object]] = []

    @contextmanager
    def claim(store: ScientificAblationWorkflowStoreV1) -> Iterator[object]:
        with original_claim(store) as owned:
            phase.__dict__["name"] = "workspace"
            yield owned

    def controller_partial(*args: object, **kwargs: object) -> None:
        actual = args[1]
        assert type(actual) is ScientificAblationWorkflowPhaseV1
        seen.append(actual.model_dump(mode="json"))
        return None

    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim", new=claim),
        patch.object(steps, "execute", side_effect=controller_partial),
        patch.object(module, "_check_expected", wraps=module._check_expected) as guard,
    ):
        progress = job.advance_expected(
            expected_phase=phase, expected_completed_phase_sha256=(), rows_per_call=1
        )
    assert guard.call_count == 3
    assert seen == [dict(index=0, name="ordinary-capture", kind="capture", stream_index=None)]
    assert progress.completed_phase_sha256 == () and progress.next_phase == "ordinary-capture"
    assert phase.name == "workspace" and job._state == "idle"


def test_expected_call_preserves_first_signal_and_original_owner_cleanup(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    signal = KeyboardInterrupt("labelled controller signal, not a native result")
    with patch.object(steps, "execute", side_effect=signal):
        try:
            job.advance_expected(expected_phase=phase, expected_completed_phase_sha256=())
        except BaseException as error:
            assert error is signal
        else:
            pytest.fail("First original signal must propagate")
    assert job._state == "idle"


def test_unacknowledged_existing_phase_stays_uncertain_not_another_phase(tmp_path: Path) -> None:
    job, phase = job_and_phase(tmp_path)
    with job._seal.store.claim():
        pass
    folder = job._seal.path / "phase-000"
    folder.mkdir()
    marker = folder / "preserved-unacknowledged-data.json"
    marker.write_bytes(b'{"not_native_authority":true}\n')
    before = marker.read_bytes(), marker.stat().st_ino
    with (
        patch.object(steps, "execute") as execute,
        pytest.raises(IrisScientificAblationWorkflowUncertainError),
    ):
        job.advance_expected(expected_phase=phase, expected_completed_phase_sha256=())
    assert execute.call_count == 0
    assert (marker.read_bytes(), marker.stat().st_ino) == before
