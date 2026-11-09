"""Read-only honest full inherited writer-report gate collection/snapshot stdout.

No test execution/publication/writes. New source bytes do NOT inherit old verdicts.
This preserves exact prior1371 IDs plus BOTH123-case representation additions and
all current report tests, with current inherited source/auditor/witness bindings.
"""

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PREREQUISITES = {
    "native_before": (
        "artifacts/operational/r015-native-writer-before-20261013-run04.json",
        "55cae0c18f94455fc9a35b9e0f8688686ffa7b99690af44cdaa12c823774d4aa",
    ),
    "native_readback": (
        "artifacts/operational/r015-native-writer-readback-20261013-run01.json",
        "db2470b637990d439341f10ba7014b2dfd2527ff18df9c39ec8aac46df4bfe52",
    ),
    "tensor_before": (
        "artifacts/operational/r015-writer-tensor-before-20261013-run01.json",
        "ff645f90ca4caf04721b1419b38c4c45fa2ae21d9731b57a25b08f7276eab266",
    ),
    "tensor_readback": (
        "artifacts/operational/r015-writer-tensor-readback-20261013-run01.json",
        "858547e4665109066b84fa920044afe372044daed6dc5a97921c71afab503908",
    ),
    "observations_before": (
        "artifacts/operational/r015-writer-observations-before-20261013-run01.json",
        "c4d48794d77ff03f7f52b238a0cbb5e4e0e3bb36bc8fc55e3fb9942b195c0351",
    ),
    "observations_readback": (
        "artifacts/operational/r015-writer-observations-readback-20261014-run01.json",
        "862ed398216478f1e4b959add9b3a2acd957520694b84735ea080cc5fa38d950",
    ),
}
NEW_FILES = (
    "src/iris/artifacts/scientific_ablation_writer_reports_v1.py",
    "src/iris/application/scientific_ablation_writer_reports.py",
    "tests/unit/test_scientific_ablation_writer_reports.py",
    "tests/integration/test_scientific_ablation_writer_reports.py",
)


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


def snapshot() -> dict[str, Any]:
    prerequisites: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in PREREQUISITES.items():
        raw = (ROOT / path).read_bytes()
        require(sha256(raw).hexdigest() == expected, "Original prerequisite bytes: " + name)
        value = json.loads(raw)
        require(
            type(value) is dict and raw == canonical(value) + b"\n",
            "Canonical prerequisite receipt",
        )
        prerequisites[name] = binding(path)
        receipts[name] = value
    native, tensors, observations = (
        receipts[name] for name in ("native_before", "tensor_before", "observations_before")
    )
    inherited_bindings = native["bindings"] + tensors["new_bindings"] + observations["new_bindings"]
    inherited_auditors = native["auditors"] + [
        receipts["tensor_readback"]["auditor"],
        receipts["observations_readback"]["auditor"],
    ]
    require(
        len(inherited_bindings) == 568 and len(inherited_auditors) == 18, "Full inherited pin scope"
    )
    witnesses = receipts["native_readback"]["native_writer_source_files"]
    require(len(witnesses) == 24, "All original native files required")
    pins = inherited_bindings + inherited_auditors + witnesses + list(prerequisites.values())
    require(
        all(binding(pin["path"]) == pin for pin in pins),
        "Stable original source/auditor/witness pins before",
    )
    additions = [tensors["modules"][0], observations["modules"][0], *NEW_FILES[2:]]
    modules = [*native["modules"], *additions]
    require(
        len(modules) == len(set(modules)) == 37, "Every original/current module, no dedup shrink"
    )
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *modules,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    require(
        process.returncode == 0,
        "Actual full pytest collection failed: " + process.stderr.decode("utf-8", errors="replace"),
    )
    nodeids = [
        line.strip()
        for line in process.stdout.decode("utf-8").splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    require(len(nodeids) == len(set(nodeids)), "Exact unique complete collected identities")
    inherited_ids = (
        native["nodeids"]
        + [node for node in tensors["nodeids"] if node.startswith(additions[0] + "::")]
        + [node for node in observations["nodeids"] if node.startswith(additions[1] + "::")]
    )
    require(
        len(inherited_ids) == 1617 and nodeids[:1617] == inherited_ids,
        "Exact whole1371+123+123 inherited IDs in original order",
    )
    unit_ids = [node for node in nodeids if node.startswith(NEW_FILES[2] + "::")]
    integration_ids = [node for node in nodeids if node.startswith(NEW_FILES[3] + "::")]
    require(
        len(unit_ids) == 166 and len(integration_ids) == 25 and len(nodeids) == 1808,
        "All current report cases and complete original scope",
    )
    new_bindings = [binding(path) for path in NEW_FILES]
    own = binding(str(Path(__file__).relative_to(ROOT)).replace("\\", "/"))
    require(
        all(binding(pin["path"]) == pin for pin in pins + new_bindings + [own]),
        "All frozen bytes stable after collection",
    )
    return dict(
        kind="iris-operational-complete-writer-report-before-v1",
        collection_exit_code=process.returncode,
        exact_cases=len(nodeids),
        inherited_cases=len(inherited_ids),
        new_cases=len(unit_ids) + len(integration_ids),
        modules=modules,
        nodeids=nodeids,
        prerequisites=prerequisites,
        inherited_bindings=inherited_bindings,
        inherited_auditors=inherited_auditors,
        original_native_files=witnesses,
        new_bindings=new_bindings,
        collector=own,
        source_test_pins=572,
        inherited_auditor_pins=18,
        full_gate_runtime_verdict="not_run",
        public_service_scoped_acceptance=False,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


if __name__ == "__main__":
    sys.stdout.buffer.write(canonical(snapshot()) + b"\n")
