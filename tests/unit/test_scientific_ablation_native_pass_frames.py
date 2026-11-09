"""Private metadata-family guards ONLY; no native measurements/source attestation."""

from dataclasses import replace
from typing import Any

import pytest

from iris.application.scientific_ablation_causal_receipts import _ObservedPassSeal
from iris.application.scientific_ablation_writer_effects import _WriterPass


def frame(role: Any = "donor", **changes: Any) -> _WriterPass:
    # An opaque exact-type placeholder is sufficient for the frame's metadata
    # guard. It is NEVER audited or claimed as a genuine native retention seal.
    opaque = object.__new__(_ObservedPassSeal)
    return _WriterPass(
        **(
            dict(
                row_index=0,
                role=role,
                write_cycle=None if role == "empty" else 1,
                writer=None if role in ("empty", "shuffle") else 2,
                slot=None if role in ("empty", "shuffle") else 1,
                donor_row_index=1 if role == "donor" else None,
                donor_pair_sha256="a" * 64 if role == "donor" else None,
                seal=opaque,
            )
            | changes
        )
    )


@pytest.mark.parametrize("role", ("empty", "shuffle", "zero", "block", "donor"))
def test_exact_private_frame_metadata_for_every_scheduled_role(role: Any) -> None:
    value = frame(role)
    value.__post_init__()


@pytest.mark.parametrize("name", ("row_index", "write_cycle", "writer", "slot", "donor_row_index"))
@pytest.mark.parametrize("bad", (True, 1.0, "1"))
def test_frame_denies_wrong_primitive_families_even_copy_or_bypass(name: str, bad: Any) -> None:
    with pytest.raises(ValueError):
        frame(**{name: bad})
    value = frame()
    object.__setattr__(value, name, bad)
    with pytest.raises(ValueError):
        value.__post_init__()


@pytest.mark.parametrize(
    "changes",
    (
        {"write_cycle": 0},
        {"write_cycle": 64},
        {"row_index": 5000},
        {"writer": 64},
        {"slot": -1},
        {"donor_row_index": 5000},
        {"donor_pair_sha256": "A" * 64},
        {"donor_pair_sha256": None},
        {"role": "noise"},
    ),
)
def test_complete_frame_applicability_and_bounded_domains(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(frame(), **changes)


@pytest.mark.parametrize("role", ("empty", "shuffle", "zero", "block"))
def test_non_donor_frames_cannot_hide_donor_or_extra_indices(role: Any) -> None:
    with pytest.raises(ValueError):
        frame(role, donor_pair_sha256="a" * 64)
    with pytest.raises(ValueError):
        frame(role, donor_row_index=1)
    if role in ("empty", "shuffle"):
        with pytest.raises(ValueError):
            frame(role, writer=0)
        with pytest.raises(ValueError):
            frame(role, slot=0)
