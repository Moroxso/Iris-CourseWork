"""Functional suite admission/scoring tests; no model execution or network."""

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def suite() -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_suite_test", ROOT / "scripts/coursework_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_cases_and_strict_scoring() -> None:
    module = suite()
    cases = module.load_cases(module.CASES)
    assert [c["id"] for c in cases] == ["arithmetic", "fact-retrieval", "state-tracking"]
    assert module.exact_answer_passes(" 8.\n", cases[0])
    assert not module.exact_answer_passes("8, потому что 12 - 4 = 8", cases[0])
    assert not module.exact_answer_passes("18", cases[0])
    assert not module.exact_answer_passes("выключена", cases[2])


@pytest.mark.parametrize("ids", [["../escape"], ["valid", "valid"], []])
def test_invalid_cases_refuse(tmp_path: Path, ids: list[str]) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            dict(
                kind="iris-coursework-cases-v1",
                cases=[dict(id=i, prompt="hello", accepted_answers=["yes"]) for i in ids],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        suite().load_cases(path)


def write_result(
    path: Path, prompt: str, text: str, source_hash: str, system_prompt: str = ""
) -> None:
    path.mkdir()
    result = dict(
        kind="iris-coursework-two-pass-demo-v1",
        completed=True,
        backbone_pretrained=True,
        backbone_frozen=True,
        sidecar_trained=False,
        scientific_acceptance=False,
        prefix_layout="chat-boundary",
        bridge_initialization_scale=0.025,
        prompt=prompt,
        system_prompt=system_prompt,
        text=text,
        source_script_sha256=source_hash,
        stop_reason="stop_token",
        frozen_qwen_baseline=dict(text=text),
    )
    cycle = dict(
        cycle=1,
        module_bids=[0.1] * 8,
        module_confidence=[0.5] * 8,
        selected_modules=[0, 1],
        selected_slots=[0, 1],
        slot_sources=[-1] * 8,
        slot_occupancy=[False] * 8,
        slot_norms=[0.0] * 8,
        slot_ages=[0] * 8,
        slot_ttl=[0.0] * 8,
    )
    trace = dict(
        kind="iris-workspace-diagnostic-trace-v1",
        cycle_count=3,
        num_modules=8,
        num_slots=8,
        state_dim=256,
        cycles=[cycle] * 3,
    )
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (path / "workspace-trace.json").write_text(json.dumps(trace), encoding="utf-8")


@pytest.mark.parametrize(
    "field,value",
    [
        ("backbone_frozen", False),
        ("sidecar_trained", True),
        ("prompt", "different prompt"),
        ("source_script_sha256", "different source"),
        ("system_prompt", "unreported instruction"),
    ],
)
def test_provenance_mismatch_is_not_runtime_pass(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    module = suite()
    case = module.load_cases(module.CASES)[0]
    path = tmp_path / "run"
    write_result(path, case["prompt"], "8", "expected")
    result_path = path / "result.json"
    result = json.loads(result_path.read_text("utf-8"))
    result[field] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")
    row = module.inspect_case(path, case, "expected")
    assert row["runtime_contract_passed"] is False


@pytest.mark.parametrize("response_style", ["original", "concise"])
@pytest.mark.parametrize("scenario", ["mixed", "all-correct", "viewer-failure"])
def test_suite_retains_success_wrong_answer_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_style: str,
    scenario: str,
) -> None:
    module = suite()
    snapshot = tmp_path / "c1899de289a04d12100db370d81485cdf75e47ca"
    snapshot.mkdir()
    identity = module.file_identity

    def fixture_identity(path: Path) -> dict[str, Any]:
        if path.parent == snapshot:
            # Simulated inventory only. Never claim these are actual model weights.
            return dict(bytes=2 * 1024**3, sha256="fixture")
        return identity(path)

    monkeypatch.setattr(module, "file_identity", fixture_identity)
    if scenario == "viewer-failure":
        monkeypatch.syspath_prepend(str(ROOT / "scripts"))
        viewer = importlib.import_module("coursework_trace_viewer")

        def fail_render(*args: Any) -> str:
            raise ValueError("Injected viewer failure")

        monkeypatch.setattr(viewer, "render_report", fail_render)
    calls = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        assert options["timeout"] == 30
        assert command[:3] == [sys.executable, "-I", "-B"]
        calls.append(command)
        path = Path(command[command.index("--out") + 1])
        prompt = command[command.index("--prompt") + 1]
        if len(calls) == 3 and scenario == "mixed":
            raise subprocess.TimeoutExpired(command, 30)
        if scenario == "mixed":
            text = "8" if len(calls) == 1 else "wrong answer"
        else:
            text = module.load_cases(module.CASES)[len(calls) - 1]["accepted_answers"][0]
        instruction = (
            command[command.index("--system-prompt") + 1] if "--system-prompt" in command else ""
        )
        assert bool(instruction) == (response_style == "concise")
        write_result(
            path, prompt, text, identity(ROOT / "scripts/coursework_demo.py")["sha256"], instruction
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "suite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suite",
            "--snapshot",
            str(snapshot),
            "--out",
            str(output),
            "--timeout-per-case",
            "30",
            "--response-style",
            response_style,
        ],
    )
    assert module.main() == (0 if scenario == "all-correct" else 1)
    assert len(calls) == 3
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert report["all_cases_attempted"] is True
    assert report["response_style"] == response_style
    assert report["runtime_passed"] is (scenario == "all-correct")
    assert report["all_exact_answers_passed"] is (scenario != "mixed")
    assert report["cases"][0]["iris_exact_answer_passed"] is True
    assert report["cases"][1]["iris_exact_answer_passed"] is (scenario != "mixed")
    if scenario == "mixed":
        assert report["cases"][2]["error"]["type"] == "TimeoutExpired"
    elif scenario == "viewer-failure":
        assert all(row["viewer_created"] is False for row in report["cases"])
        assert all(row["error"]["type"] == "ValueError" for row in report["cases"])
    else:
        assert all(row["viewer_created"] is True for row in report["cases"])
    assert (output / "fact-retrieval/viewer.html").is_file()
    assert len(list(output.glob("*.observation.json"))) == 3
    before = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="Fresh absolute"):
        module.main()
    assert len(calls) == 3
    assert (output / "report.json").read_bytes() == before
