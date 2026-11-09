"""Read-only full observation/frame component runtime and fresh JUnit readback.

Exercises pinned production DTOs on all accepted original native projections.
This observer IMPORTS Iris/Torch for validation and explicit forward traps; it is
NOT the future model-import-free independent complete-report semantic reader.
No native measurement, source-chain replay, generated execution or file writes.
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "artifacts/operational/r015-writer-observations-before-20261013-run01.json"
JUNIT = "artifacts/operational/r015-writer-observations-unit-20261013-run01.xml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def binding(path: str) -> dict[str, Any]:
    raw = (ROOT / path).read_bytes()
    return dict(path=path, bytes=len(raw), sha256=sha256(raw).hexdigest())


def load(path: str) -> dict[str, Any]:
    raw = (ROOT / path).read_bytes()
    value = json.loads(raw)
    require(
        type(value) is dict and raw == canonical(value) + b"\n",
        "Canonical receipt/witness required",
    )
    assert isinstance(value, dict)
    return value


def audit() -> dict[str, Any]:
    own_path = str(Path(__file__).relative_to(ROOT)).replace("\\", "/")
    own_pin, before_pin, xml_pin = binding(own_path), binding(BEFORE), binding(JUNIT)
    require(
        before_pin["sha256"] == "c4d48794d77ff03f7f52b238a0cbb5e4e0e3bb36bc8fc55e3fb9942b195c0351",
        "Exact original fresh observations snapshot required",
    )
    before = load(BEFORE)
    require(
        before["exact_cases"] == 361 and before["new_cases"] == 123, "Full scoped runtime inventory"
    )
    receipt_pins = [
        before[name]
        for name in (
            "tensor_before",
            "tensor_readback",
            "native_before",
            "native_readback",
        )
    ]
    require(
        all(binding(pin["path"]) == pin for pin in receipt_pins),
        "Original prerequisite receipt bytes",
    )
    tensor_before, tensor_readback = (
        load(before["tensor_before"]["path"]),
        load(before["tensor_readback"]["path"]),
    )
    native_before, native_readback = (
        load(before["native_before"]["path"]),
        load(before["native_readback"]["path"]),
    )
    frozen = (
        native_before["bindings"]
        + native_before["auditors"]
        + tensor_before["new_bindings"]
        + [tensor_readback["auditor"]]
        + before["new_bindings"]
    )
    require(len(frozen) == 585, "566+17 inherited and two new source/test pins")
    require(
        all(binding(pin["path"]) == pin for pin in frozen), "All source/test/auditor bytes before"
    )
    require(
        native_readback["passed"] is True
        and native_readback["exact_cases"] == 1371
        and native_readback["native_writer_observed_calls"] == 468,
        "Actual accepted original native runtime prerequisite",
    )
    cases = list(ET.parse(ROOT / JUNIT).getroot().iter("testcase"))
    require(len(cases) == 361 and all(not list(case) for case in cases), "All current cases passed")
    ids = [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]
    require(
        ids == before["nodeids"] and len(set(ids)) == 361, "Exact ordered current JUnit identities"
    )
    require(
        [node for node in ids if not node.startswith(before["modules"][0] + "::")]
        == tensor_before["nodeids"],
        "All exact238 predecessor cases retained",
    )
    witnesses = before["original_native_files"]
    require(
        witnesses == native_readback["native_writer_source_files"] and len(witnesses) == 24,
        "All original native witnesses",
    )
    require(
        all(binding(pin["path"]) == pin for pin in witnesses), "Original native file bytes before"
    )
    # Imports occur ONLY after all frozen source/receipt bindings have passed.
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from iris.artifacts.scientific_ablation_writer_observations_v1 import (
        ScientificAblationWriterPassFrameV1,
        writer_observed_wire_v1,
        writer_pass_wire_v1,
    )
    from iris.training.model import IrisTeacherForcingModel

    files: list[dict[str, Any]] = []
    modes: set[bytes] = set()
    roles: Counter[str] = Counter()
    proposals = donors = states = 0
    with (
        patch.object(
            torch.nn.Module, "_call_impl", side_effect=AssertionError("No neural forward allowed")
        ) as neural,
        patch.object(
            IrisTeacherForcingModel,
            "evaluation_forward_ablation_observed",
            side_effect=AssertionError("No observed teacher forward allowed"),
        ) as observed,
        patch.object(
            IrisTeacherForcingModel,
            "evaluation_forward_ablation_controlled",
            side_effect=AssertionError("No controlled teacher forward allowed"),
        ) as controlled,
        patch.object(
            IrisTeacherForcingModel,
            "evaluation_forward",
            side_effect=AssertionError("No teacher forward allowed"),
        ) as ordinary,
    ):
        for pin in witnesses:
            raw = load(pin["path"])
            data = raw["projected_report"]["payload"]
            groups = data["controlled_passes"]
            frames = groups["empty"] + [
                frame for group in groups["trials"].values() for frame in group
            ]
            for frame in frames:
                owned = ScientificAblationWriterPassFrameV1.model_validate(frame)
                require(
                    canonical(writer_pass_wire_v1(owned)) == canonical(frame),
                    "Complete original frame/output bytes",
                )
                require(
                    canonical(writer_observed_wire_v1(owned.output)) == canonical(frame["output"]),
                    "Complete original observed output bytes",
                )
                roles[owned.role] += 1
                proposals += len(owned.output.controlled.messages)
                donors += int(
                    getattr(owned.output.controlled.plan.control, "donor", None) is not None
                )
                states += len(owned.output.controlled.states) + 2 * len(owned.output.observations)
            modes.add(canonical(raw["mode"]))
            files.append(dict(witness=pin, observed_passes=len(frames)))
        neural.assert_not_called()
        observed.assert_not_called()
        controlled.assert_not_called()
        ordinary.assert_not_called()
    require(
        len(modes) == 12 and len(files) == 24 and sum(roles.values()) == 468,
        "All modes/protocols/passes",
    )
    require(
        dict(roles) == dict(empty=60, block=96, donor=96, shuffle=72, zero=144),
        "Complete original role inventory",
    )
    require(
        states == 4212 and proposals == 1440 and donors == 96,
        "All phases/proposals/embedded donors retained",
    )
    require(
        all(binding(pin["path"]) == pin for pin in frozen), "Source/test/auditor bytes stable after"
    )
    require(
        all(binding(pin["path"]) == pin for pin in witnesses), "Original witness bytes stable after"
    )
    require(
        all(binding(pin["path"]) == pin for pin in receipt_pins), "Original receipts stable after"
    )
    require(
        binding(own_path) == own_pin
        and binding(BEFORE) == before_pin
        and binding(JUNIT) == xml_pin,
        "Observer/before/JUnit stable",
    )
    return dict(
        kind="iris-operational-writer-observations-component-readback-v1",
        passed=361,
        new_cases=123,
        inherited_cases=238,
        before=before_pin,
        junit=xml_pin,
        auditor=own_pin,
        new_bindings=before["new_bindings"],
        stable_inherited_source_test_pins=566,
        stable_inherited_auditor_pins=17,
        original_native_files=files,
        original_modes=12,
        original_observed_passes=468,
        original_all7_states=4212,
        original_proposals=1440,
        original_embedded_donor_snapshots=96,
        original_role_inventory=dict(roles),
        additional_neural_teacher_forwards=0,
        project_validation_imports=True,
        torch_imported_for_forward_guards=True,
        fresh_native_measurements=False,
        independent_complete_report_semantics=False,
        full_public_report_complete=False,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


if __name__ == "__main__":
    print(canonical(audit()).decode("utf-8"))
