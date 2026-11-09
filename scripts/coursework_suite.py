"""Run all fixed coursework examples offline, retaining every result and failure."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs/examples/coursework-cases-v1.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > 65536:
        raise ValueError("Bounded cases file required")
    data = json.loads(path.read_text("utf-8"))
    cases = data.get("cases")
    if data.get("kind") != "iris-coursework-cases-v1" or not isinstance(cases, list):
        raise ValueError("Versioned cases required")
    if not 1 <= len(cases) <= 8:
        raise ValueError("One to eight cases required")
    ids = set()
    for case in cases:
        identifier = case.get("id")
        prompt = case.get("prompt")
        answers = case.get("accepted_answers")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[a-z][a-z0-9-]{0,39}", identifier) is None
            or identifier in ids
        ):
            raise ValueError("Unique safe case ID required")
        ids.add(identifier)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 8192:
            raise ValueError("Bounded nonempty prompt required")
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(answer, str) and answer.strip() for answer in answers)
        ):
            raise ValueError("Explicit accepted answers required")
    return cases


def normalize_answer(value: str) -> str:
    """Only whitespace, case, and one final sentence period are ignored."""
    return " ".join(value.casefold().split()).removesuffix(".")


def exact_answer_passes(text: str, case: dict[str, Any]) -> bool:
    return normalize_answer(text) in [normalize_answer(a) for a in case["accepted_answers"]]


def file_identity(path: Path) -> dict[str, Any]:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def inspect_case(
    path: Path, case: dict[str, Any], source_hash: str, system_prompt: str = ""
) -> dict[str, Any]:
    result = json.loads((path / "result.json").read_text("utf-8"))
    trace = json.loads((path / "workspace-trace.json").read_text("utf-8"))
    valid = (
        result.get("kind") == "iris-coursework-two-pass-demo-v1"
        and result.get("completed") is True
        and result.get("backbone_pretrained") is True
        and result.get("backbone_frozen") is True
        and result.get("sidecar_trained") is False
        and result.get("scientific_acceptance") is False
        and result.get("prefix_layout") == "chat-boundary"
        and result.get("bridge_initialization_scale") == 0.025
        and result.get("prompt") == case["prompt"]
        and result.get("system_prompt", "") == system_prompt
        and result.get("source_script_sha256") == source_hash
        and trace.get("kind") == "iris-workspace-diagnostic-trace-v1"
        and (trace.get("num_modules"), trace.get("num_slots"), trace.get("state_dim"))
        == (8, 8, 256)
        and trace.get("cycle_count") == 3
        and len(trace.get("cycles", [])) == 3
        and not (path / "error.json").exists()
    )
    text = result.get("text")
    baseline = result.get("frozen_qwen_baseline")
    if not isinstance(text, str) or not isinstance(baseline, dict):
        raise ValueError("Actual Iris and separate baseline text required")
    baseline_text = baseline.get("text")
    if not isinstance(baseline_text, str):
        raise ValueError("Actual baseline text required")
    return dict(
        runtime_contract_passed=valid,
        iris_text=text,
        iris_stop_reason=result.get("stop_reason"),
        iris_exact_answer_passed=exact_answer_passes(text, case),
        iris_finished=result.get("stop_reason") == "stop_token",
        baseline_text=baseline_text,
        baseline_exact_answer_passed=exact_answer_passes(baseline_text, case),
        quality_improvement_proven=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-per-case", type=int, default=300)
    parser.add_argument("--response-style", choices=("original", "concise"), default="original")
    args = parser.parse_args()
    if not 30 <= args.timeout_per_case <= 600:
        raise ValueError("Per-case timeout must be 30..600 seconds")
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute output required")
    cases = load_cases(CASES)
    sys.path.insert(0, str(ROOT / "scripts"))
    from coursework_demo import CONCISE_SYSTEM_PROMPT, REVISION, save_new
    from coursework_trace_viewer import render_report

    system_prompt = CONCISE_SYSTEM_PROMPT if args.response_style == "concise" else ""

    snapshot: Path = args.snapshot
    if not snapshot.is_absolute() or snapshot.name != REVISION:
        raise ValueError("Pinned absolute snapshot required")
    names = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    inventory = {name: file_identity(snapshot / name) for name in names}
    if inventory["model.safetensors"]["bytes"] < 1024**3:
        raise ValueError("Actual pretrained weights required, not a tiny fixture")
    source_paths = [
        Path(__file__),
        ROOT / "scripts/coursework_demo.py",
        ROOT / "scripts/coursework_prefix.py",
        ROOT / "scripts/coursework_trace_viewer.py",
        CASES,
    ]
    source_paths += sorted((ROOT / "src/iris").rglob("*.py"))
    source_paths += sorted((ROOT / "src/iris/configs").rglob("*.yaml"))
    sources = {str(path.relative_to(ROOT)): file_identity(path) for path in source_paths}
    demo_hash = file_identity(ROOT / "scripts/coursework_demo.py")["sha256"]
    args.out.mkdir(parents=True)
    save_new(
        args.out / "inputs.json",
        dict(
            kind="iris-coursework-suite-inputs-v1",
            cases=cases,
            response_style=args.response_style,
            system_prompt=system_prompt,
            cases_file=file_identity(CASES),
            snapshot=str(snapshot),
            snapshot_files=inventory,
            revision=REVISION,
            source_files=sources,
            seed=42,
            max_new_tokens=64,
            cpu_threads=4,
            prefix_layout="chat-boundary",
            bridge_initialization_scale=0.025,
            timeout_per_case=args.timeout_per_case,
            scientific_acceptance=False,
        ),
    )
    observations = []
    began = time.perf_counter()
    for case in cases:
        path = args.out / case["id"]
        command = [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "scripts/coursework_demo.py"),
            "--snapshot",
            str(snapshot),
            "--out",
            str(path),
            "--prompt",
            case["prompt"],
            "--max-new-tokens",
            "64",
            "--prefix-layout",
            "chat-boundary",
            "--bridge-init-scale",
            "0.025",
            "--diagnose-bridge",
        ]
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        row: dict[str, Any] = dict(
            id=case["id"],
            command=command,
            process_exit_code=None,
            runtime_contract_passed=False,
            iris_exact_answer_passed=False,
            iris_finished=False,
            viewer_created=False,
        )
        with (
            (args.out / (case["id"] + ".stdout.txt")).open("xb") as stdout,
            (args.out / (case["id"] + ".stderr.txt")).open("xb") as stderr,
        ):
            try:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=args.timeout_per_case,
                    check=False,
                )
                row["process_exit_code"] = process.returncode
                row.update(inspect_case(path, case, demo_hash, system_prompt))
                if process.returncode == 0 and row["runtime_contract_passed"]:
                    result = json.loads((path / "result.json").read_text("utf-8"))
                    trace = json.loads((path / "workspace-trace.json").read_text("utf-8"))
                    with (path / "viewer.html").open("x", encoding="utf-8") as page:
                        page.write(render_report(result, trace))
                    row["viewer_created"] = True
            except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                row["error"] = dict(type=type(error).__name__, message=str(error))
        observations.append(row)
        save_new(args.out / (case["id"] + ".observation.json"), row)
        print(
            json.dumps(
                dict(
                    id=case["id"],
                    exit_code=row["process_exit_code"],
                    exact_answer_passed=row["iris_exact_answer_passed"],
                )
            ),
            flush=True,
        )
    runtime_passed = all(
        r["process_exit_code"] == 0 and r["runtime_contract_passed"] and r["viewer_created"]
        for r in observations
    )
    answers_passed = all(r["iris_exact_answer_passed"] and r["iris_finished"] for r in observations)
    files = {
        str(p.relative_to(args.out)): file_identity(p)
        for p in sorted(args.out.rglob("*"))
        if p.is_file()
    }
    report = dict(
        kind="iris-coursework-suite-report-v1",
        all_cases_attempted=True,
        response_style=args.response_style,
        runtime_passed=runtime_passed,
        all_exact_answers_passed=answers_passed,
        cases=observations,
        elapsed_seconds=time.perf_counter() - began,
        retained_files=files,
        scientific_acceptance=False,
        quality_improvement_proven=False,
    )
    save_new(args.out / "report.json", report)
    return 0 if runtime_passed and answers_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
