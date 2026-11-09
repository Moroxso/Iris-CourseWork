"""Synthetic metadata guards only: no native/model/source/execution evidence."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_writer_reader_test",
    Path(__file__).resolve().parents[2]
    / "scripts/verify_scientific_ablation_native_writer_effects.py",
)
assert SPEC is not None and SPEC.loader is not None
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def frame(role: str = "zero") -> dict[str, Any]:
    return dict(
        row_index=0,
        role=role,
        write_cycle=None if role == "empty" else 1,
        writer=0 if role in ("zero", "block", "donor") else None,
        slot=0 if role in ("zero", "block", "donor") else None,
        donor_row_index=1 if role == "donor" else None,
        donor_pair_sha256="a" * 64 if role == "donor" else None,
        output={},  # OPAQUE placeholder; _frame never claims native receipt acceptance.
    )


def effect(message: bool = False) -> dict[str, Any]:
    return dict(
        kind=f"iris-scientific-ablation-{'message' if message else 'workspace'}-writer-effect-v1",
        row_index=0,
        write_cycle=1,
        source_module=0,
        target_module=1,
        slot=0,
        survives_to_next_read=True,
        overwrote_existing_record=False,
        genuine_distance=0.0,
        negative_control_distance=0.0,
        empty_invocation_sha256="a" * 64,
        intervention_invocation_sha256="b" * 64,
        negative_control_invocation_sha256="c" * 64,
        recipient_message_sha256="d" * 64 if message else None,
        donor_message_sha256="e" * 64 if message else None,
        donor_pair_sha256="f" * 64 if message else None,
        donor_row_index=1 if message else None,
    )


def trace(full: bool = False) -> dict[str, Any]:
    return dict(
        kind="iris-workspace-diagnostic-trace-v1",
        semantics="post-intervention-read-and-post-write-v1",
        options=dict(
            mode="full" if full else "metadata",
            example_id="synthetic",
            checkpoint_id="synthetic",
            seed=0,
            maximum_scalars=65536,
        ),
        num_modules=4,
        num_slots=4,
        state_dim=8,
        cycle_count=3,
        intervention_plan_sha256="a" * 64,
        cycles=[
            dict(
                cycle=i,
                **{
                    n: [0.0] * 4
                    for n in (
                        "module_bids",
                        "module_confidence",
                        "module_state_norms",
                        "slot_ttl",
                        "slot_norms",
                    )
                },
                **{n: [0] * 4 for n in ("slot_ages", "slot_write_counts")},
                **{n: [-1] * 4 for n in ("slot_sources", "read_slot_sources")},
                **{n: [False] * 4 for n in ("slot_occupancy", "read_slot_occupancy")},
                **{
                    n: []
                    for n in (
                        "selected_modules",
                        "selected_slots",
                        "write_modules",
                        "overwritten_slots",
                    )
                },
                workspace_cosine_matrix=[[0.0] * 4 for _ in range(4)],
                read_attention=[[0.0] * 4 for _ in range(4)],
                full={
                    n: [[0.0] * 8 for _ in range(4)]
                    for n in (
                        "read_workspace_slots",
                        "workspace_slots",
                        "module_states",
                        "workspace_reads",
                    )
                }
                if full
                else None,
            )
            for i in (1, 2, 3)
        ],
    )


@pytest.mark.parametrize("role", ("empty", "shuffle", "zero", "block", "donor"))
def test_exact_metadata_frame_roles_without_claiming_native_output(role: str) -> None:
    reader._frame(frame(role))


@pytest.mark.parametrize("field", ("row_index", "write_cycle", "writer", "slot", "donor_row_index"))
@pytest.mark.parametrize("wrong", (True, 1.0, "1"))
def test_frame_integer_fields_reject_equal_foreign_families(field: str, wrong: Any) -> None:
    value = frame("donor")
    value[field] = wrong
    with pytest.raises(ValueError, match="integer domain"):
        reader._frame(value)


@pytest.mark.parametrize(
    ("role", "field"),
    (
        ("empty", "writer"),
        ("empty", "slot"),
        ("empty", "write_cycle"),
        ("shuffle", "writer"),
        ("shuffle", "slot"),
        ("shuffle", "donor_row_index"),
        ("zero", "donor_row_index"),
        ("block", "donor_row_index"),
    ),
)
def test_inapplicable_frame_fields_are_null(role: str, field: str) -> None:
    value = frame(role)
    value[field] = 0
    with pytest.raises(ValueError, match="Inapplicable"):
        reader._frame(value)


@pytest.mark.parametrize("message", (False, True))
def test_exact_effect_metadata_distinct_protocols(message: bool) -> None:
    reader._effect(effect(message), message)


@pytest.mark.parametrize(
    "field", ("row_index", "write_cycle", "source_module", "target_module", "slot")
)
@pytest.mark.parametrize("wrong", (True, 1.0, "1"))
def test_effect_integer_fields_reject_equal_foreign_families(field: str, wrong: Any) -> None:
    value = effect()
    value[field] = wrong
    with pytest.raises(ValueError, match="integer domain"):
        reader._effect(value, False)


@pytest.mark.parametrize("field", ("genuine_distance", "negative_control_distance"))
@pytest.mark.parametrize("wrong", (True, 0, float("nan"), float("inf"), -0.01, 1.01))
def test_distances_require_exact_finite_unit_float(field: str, wrong: Any) -> None:
    value = effect()
    value[field] = wrong
    with pytest.raises(ValueError, match="effect distance"):
        reader._effect(value, False)


@pytest.mark.parametrize(
    "field",
    ("recipient_message_sha256", "donor_message_sha256", "donor_pair_sha256", "donor_row_index"),
)
def test_workspace_metadata_cannot_claim_a_message_donor(field: str) -> None:
    value = effect()
    value[field] = 1 if field == "donor_row_index" else "a" * 64
    with pytest.raises(ValueError, match="no donor"):
        reader._effect(value, False)


@pytest.mark.parametrize("family", ("frame", "effect", "trace"))
def test_hidden_metadata_is_not_ignored(family: str) -> None:
    value = frame() if family == "frame" else effect() if family == "effect" else trace()
    value["foreign"] = "hidden"
    with pytest.raises(ValueError, match="Closed"):
        if family == "frame":
            reader._frame(value)
        elif family == "effect":
            reader._effect(value, False)
        else:
            reader._trace(value)


@pytest.mark.parametrize("full", (False, True))
def test_synthetic_trace_shape_only_not_native_state(full: bool) -> None:
    reader._trace(trace(full))


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("module_bids", 0),
        ("slot_ages", False),
        ("slot_sources", -1.0),
        ("slot_occupancy", 0),
        ("read_slot_sources", False),
        ("read_slot_occupancy", 0),
    ),
)
def test_trace_vectors_reject_equal_wrong_scalar_families(field: str, wrong: Any) -> None:
    value = trace()
    value["cycles"][0][field][0] = wrong
    with pytest.raises(ValueError, match="vector families"):
        reader._trace(value)


@pytest.mark.parametrize(
    "field", ("module_states", "workspace_slots", "workspace_reads", "read_workspace_slots")
)
def test_all_full_matrices_reject_boolean_as_float(field: str) -> None:
    value = trace(True)
    value["cycles"][0]["full"][field][0][0] = False
    with pytest.raises(ValueError, match="vector families"):
        reader._trace(value)


@pytest.mark.parametrize("wrong", (True, 0.0, "0"))
def test_original_trace_seed_is_exact_integer(wrong: Any) -> None:
    value = trace()
    value["options"]["seed"] = wrong
    with pytest.raises(ValueError, match="integer domain"):
        reader._trace(value)


def test_plan_comparison_cannot_equate_boolean_and_original_zero_slot() -> None:
    assert not reader.statistics.exact({"slot": False}, {"slot": 0})


@pytest.mark.parametrize("message", (False, True))
def test_foreign_effect_protocol_is_rejected(message: bool) -> None:
    with pytest.raises(ValueError, match="Distinct effect domain"):
        reader._effect(effect(not message), message)
