"""Explicit ONE-source corrective epoch, never an old-reader/hash bypass.

All619other inherited protected files and48original public/preflight inputs must
remain exact. The old/new usage implementation hashes and byte-exact archived
original are mandatory independently fixed expectations, not self-derived values.
Old1808/134/pilot verdicts remain historical; current code needs fresh regression.
"""

from typing import Any

import verify_scientific_ablation_writer_report_reader as boundary

OLD = dict(
    path="src/iris/training/usage.py",
    bytes=5146,
    sha256="41e85e5e05c68315454cf1ef32b03879b6ca764411b7563d4af9b949ff79bc7f",
)
NEW = dict(
    path="src/iris/training/usage.py",
    bytes=5681,
    sha256="1e55c7d2c749c973f32afdf3f01b28a1d239b637d7a1b4a358a0d3dd03fccef5",
)
ARCHIVE = "artifacts/operational/r015-single-slot-usage-original-source-20261014.py"
BYTE = "artifacts/operational/r015-independent-byte-readback-20261014-run01.json"
BYTE_SHA = "1245dbb23d5ada5ba12d4e1ff21b56e312c5605c668b317fe117647c5f390843"


def scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    before = boundary.load(boundary.OLD_BEFORE, boundary.OLD_BEFORE_SHA)
    accepted = boundary.load(boundary.OLD_RAW, boundary.OLD_RAW_SHA)
    byte = boundary.load(BYTE, BYTE_SHA)
    original = [
        *before["inherited_bindings"],
        *before["inherited_auditors"],
        *before["new_bindings"],
        before["collector"],
        *before["original_native_files"],
        *accepted["reader_pins"],
        *byte["current_bindings"],
    ]
    if len(original) != 620 or accepted["passed"] is not True or byte["passed"] is not True:
        raise ValueError("Entire actual inherited620source inventory required")
    current = []
    changed = 0
    for pin in original:
        actual = boundary.binding(pin["path"])
        if pin["path"] == OLD["path"]:
            if pin != OLD or actual != NEW:
                raise ValueError("Only independently fixed usage correction allowed")
            changed += 1
        elif actual != pin:
            raise ValueError("Unrelated inherited source changed: " + pin["path"])
        current.append(actual)
    if changed != 1:
        raise ValueError("Exactly ONE old→new production source change required")
    archived = boundary.binding(ARCHIVE)
    if archived["bytes"] != OLD["bytes"] or archived["sha256"] != OLD["sha256"]:
        raise ValueError("Byte-exact original usage source archive required")
    inputs = [
        pin
        for entry in accepted["public_reports"]
        for pin in (entry["report"], entry["separately_retained_preflight"])
    ]
    if len(inputs) != 48 or len({pin["path"] for pin in inputs}) != 48:
        raise ValueError("ALL24original public and separate preflight pairs")
    if any(boundary.binding(pin["path"]) != pin for pin in inputs):
        raise ValueError("Original retained public/preflight input changed")
    delta = dict(
        kind="iris-single-slot-usage-intentional-code-delta-v1",
        old=OLD,
        new=NEW,
        archived_original=archived,
        unchanged_other_inherited_files=619,
        old_verdicts_not_promoted_to_current=True,
    )
    return current, inputs, delta
