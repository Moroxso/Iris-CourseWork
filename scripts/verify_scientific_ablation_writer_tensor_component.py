"""Read-only component/runtime readback against accepted native writer witnesses.

This loads the pinned NEW codec without importing the Iris package or Torch.
It independently reconciles the fresh scoped JUnit IDs/pins, then exercises the
codec on every original state. It is NOT an independent full-report semantic
auditor, a model execution, or a replay of the accepted physical source chains.
Prints one canonical receipt to stdout; never writes files or changes witnesses.
"""

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "artifacts/operational/r015-writer-tensor-before-20261013-run01.json"
JUNIT = "artifacts/operational/r015-writer-tensor-unit-20261013-run01.xml"
PREDECESSOR = "artifacts/operational/r015-native-writer-readback-20261013-run01.json"
CODEC = "src/iris/artifacts/scientific_ablation_writer_tensors_v1.py"


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
    require(type(value) is dict and raw == canonical(value) + b"\n", "Canonical receipt required")
    assert isinstance(value, dict)
    return value


def forbidden_imports() -> None:
    require(
        not any(
            name.split(".", 1)[0] in {"iris", "torch", "tests", "pytest"} for name in sys.modules
        ),
        "No Iris/Torch/tests/pytest imports allowed in this observer",
    )


def codec_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_iris_writer_tensor_codec_readback", ROOT / CODEC
    )
    require(spec is not None, "Trusted codec module spec required")
    assert spec is not None
    require(spec.loader is not None, "Trusted codec module loader required")
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit() -> dict[str, Any]:
    forbidden_imports()
    own_path = str(Path(__file__).relative_to(ROOT)).replace("\\", "/")
    auditor_pin = binding(own_path)
    before_pin, xml_pin, predecessor_pin = binding(BEFORE), binding(JUNIT), binding(PREDECESSOR)
    require(
        before_pin["sha256"] == "ff645f90ca4caf04721b1419b38c4c45fa2ae21d9731b57a25b08f7276eab266",
        "Original component before snapshot required",
    )
    require(
        predecessor_pin["sha256"]
        == "db2470b637990d439341f10ba7014b2dfd2527ff18df9c39ec8aac46df4bfe52",
        "Accepted full native writer readback required",
    )
    before, predecessor = load(BEFORE), load(PREDECESSOR)
    require(before["exact_cases"] == 238 and before["new_cases"] == 123, "Exact new unit scope")
    inherited_pin = before["inherited_before"]
    require(binding(inherited_pin["path"]) == inherited_pin, "Original inherited before bytes")
    inherited = load(inherited_pin["path"])
    require(
        len(inherited["bindings"]) == 564 and len(inherited["auditors"]) == 16,
        "Inherited pin counts",
    )
    frozen = inherited["bindings"] + inherited["auditors"] + before["new_bindings"]
    require(
        all(binding(pin["path"]) == pin for pin in frozen), "All source/test/auditor pins before"
    )
    suite = ET.parse(ROOT / JUNIT).getroot()
    cases = list(suite.iter("testcase"))
    require(len(cases) == 238, "Fresh JUnit count")
    for group in suite.iter("testsuite"):
        require(
            all(
                int(group.attrib.get(name, "0")) == 0 for name in ("failures", "errors", "skipped")
            ),
            "No failed/errored/skipped cases",
        )
    ids: list[str] = []
    for case in cases:
        require(not list(case), "Passing testcase only, no skipped/error/system-output masking")
        ids.append(case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"])
    require(
        len(set(ids)) == 238 and ids == before["nodeids"], "Exact ordered fresh case identities"
    )
    inherited_ids = [node for node in ids if not node.startswith(before["modules"][0] + "::")]
    require(
        len(inherited_ids) == 115 and set(inherited_ids) <= set(inherited["nodeids"]),
        "Exact unchanged predecessor unit coverage",
    )
    require(
        predecessor["passed"] is True
        and predecessor["exact_cases"] == 1371
        and predecessor["native_writer_observed_calls"] == 468,
        "Actual accepted full native runtime prerequisite",
    )
    witnesses = predecessor["native_writer_source_files"]
    require(len(witnesses) == 24, "All original native writer files")
    require(all(binding(pin["path"]) == pin for pin in witnesses), "Original native bytes before")
    codec = codec_module()
    files: list[dict[str, Any]] = []
    total_states = total_passes = 0
    modes: set[bytes] = set()
    protocols: dict[str, int] = {}
    for pin in witnesses:
        raw = load(pin["path"])
        data = raw["projected_report"]["payload"]
        geometry = raw["source_design"]["payload"]["geometry"]
        modules = geometry["num_modules"]
        protocol = "message" if pin["path"].endswith("native-writer-message.json") else "workspace"
        groups = data["controlled_passes"]
        frames = groups["empty"] + [frame for group in groups["trials"].values() for frame in group]
        states_in_file = 0
        for frame in frames:
            output = frame["output"]
            states = [(state, None) for state in output["controlled"]["states"]]
            for observation in output["observations"]:
                states.extend(
                    [
                        (observation["before_write"], observation["before_write_sha256"]),
                        (observation["native_written"], observation["native_written_sha256"]),
                    ]
                )
            for state, native_hash in states:
                owned = codec.ScientificAblationWriterStateV1.model_validate(state)
                projected = codec.writer_state_wire_v1(owned, num_modules=modules)
                require(canonical(projected) == canonical(state), "Exact original ALL7 wire bytes")
                if native_hash is not None:
                    require(
                        codec.writer_state_sha256_v1(owned, num_modules=modules) == native_hash,
                        "Original native observation state digest",
                    )
                states_in_file += 1
        modes.add(canonical(raw["mode"]))
        protocols[protocol] = protocols.get(protocol, 0) + 1
        total_states += states_in_file
        total_passes += len(frames)
        files.append(dict(witness=pin, observed_passes=len(frames), states=states_in_file))
    require(total_states == 4212 and total_passes == 468, "All genuine stored state/pass inventory")
    require(
        len(modes) == 12 and protocols == {"message": 12, "workspace": 12},
        "All modes and both protocols",
    )
    require(all(binding(pin["path"]) == pin for pin in witnesses), "Original native bytes after")
    require(
        all(binding(pin["path"]) == pin for pin in frozen), "All source/test/auditor pins after"
    )
    require(binding(BEFORE) == before_pin and binding(JUNIT) == xml_pin, "Before/XML stable")
    require(
        binding(PREDECESSOR) == predecessor_pin and binding(own_path) == auditor_pin,
        "Receipts/auditor stable",
    )
    forbidden_imports()
    return dict(
        kind="iris-operational-writer-tensor-component-readback-v1",
        before=before_pin,
        junit=xml_pin,
        predecessor=predecessor_pin,
        auditor=auditor_pin,
        passed=238,
        new_cases=123,
        inherited_cases=115,
        stable_inherited_source_test_pins=564,
        stable_inherited_auditor_pins=16,
        new_bindings=before["new_bindings"],
        original_native_files=files,
        original_modes=12,
        original_observed_passes=468,
        original_all7_states=4212,
        original_tensor_components=29484,
        additional_model_forwards=0,
        iris_torch_test_pytest_imports=False,
        independent_complete_report_semantics=False,
        full_public_report_complete=False,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


if __name__ == "__main__":
    print(canonical(audit()).decode("utf-8"))
