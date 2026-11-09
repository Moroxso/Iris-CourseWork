"""Transient owned pytest supervisor with durable exclusive launch/log/exit evidence.

Only fixed trusted project test scopes, no arbitrary command/candidate code,
downloads, installations, source edits, cleanup or automatic restart. Start hidden
when using Windows Start-Process. Observe real OS PID/creation/command while live;
a launch file alone never means running and a missing exit file never means PASS.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = "tests/integration/test_scientific_ablation_writer_geometry_reports.py"
NONWINNING = "tests/integration/test_scientific_ablation_writer_nonwinning_report.py"
WIDE = GEOMETRY + "::test_actual_whole_wide_both_reports_12096_effects_4032_directed_pairs"
SCOPES = {
    "nonwinning-pilot": ([NONWINNING], 1),
    "wide-pilot": ([WIDE], 1),
    "original-geometry-full": ([GEOMETRY], 37),
    "combined-full": ([GEOMETRY, NONWINNING], 38),
}


def scope_files(selected: list[str]) -> list[str]:
    """Pin source files even for the fixed explicit wide node selector."""
    return list(dict.fromkeys(node.split("::", 1)[0] for node in selected))


def scope_config(mode: str) -> tuple[list[str], int]:
    if mode == "affected-full":
        # Lazy pure-stdlib import keeps supervisor unit checks independent of
        # model/project/test imports and excludes arbitrary before-file commands.
        from scientific_ablation_writer_affected_scope import EXPECTED_CASES, modules

        return modules()[0], EXPECTED_CASES
    return SCOPES[mode]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(
            json.dumps(
                value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def output_path(value: str, parent: Path) -> Path:
    path = (ROOT / value).resolve()
    if path.parent != parent.resolve() or not path.name.startswith("r015-writer-"):
        raise ValueError("Exact NEW direct-child project output required")
    if path.exists() or path.is_symlink():
        raise ValueError("Output occupied; no overwrite/delete/auto-resume")
    return path


def pins_match(pins: list[dict[str, Any]]) -> bool:
    return all(
        (ROOT / pin["path"]).stat().st_size == pin["bytes"]
        and digest(ROOT / pin["path"]) == pin["sha256"]
        for pin in pins
    )


def passing_ids(junit: Path) -> list[str]:
    root = ET.fromstring(junit.read_bytes())
    if any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("JUnit contains failure/error/skip")
    for suite in root.iter("testsuite"):
        if any(int(suite.attrib.get(key, "0")) != 0 for key in ("failures", "errors", "skipped")):
            raise ValueError("JUnit suite counters are not all zero")
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("Actual nonempty JUnit required")
    return [
        case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--before-sha", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.before_sha) is None:
        raise ValueError("Independent original before SHA required")
    before_path = (ROOT / args.before).resolve()
    if before_path.parent != (ROOT / "artifacts/operational").resolve():
        raise ValueError("Before must be an exact retained operational file")
    if digest(before_path) != args.before_sha:
        raise ValueError("Original before changed")
    before = json.loads(before_path.read_bytes())
    # Only main (not module import) loads the pure inherited audit. This checks
    # ALL620protected and48original input pins immediately before the child.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import scientific_ablation_writer_usage_epoch as inherited

    protected, originals, delta = inherited.scope()
    if before["protected_count"] != len(protected) or before["original_inputs"] != originals:
        raise ValueError("Entire inherited inventory changed")
    if before["intentional_source_delta"] != delta:
        raise ValueError("Explicit independently fixed production correction required")
    expected_commitment = sha256(
        json.dumps(
            protected, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if before["protected_sha256"] != expected_commitment:
        raise ValueError("Inherited inventory commitment changed")
    mode = before["gate_scope"]
    modules, count = scope_config(mode)
    if before["modules"] != modules or before["expected_cases"] != count:
        raise ValueError("Exact full declared collection, no scope clipping")
    pins = [
        *before["new_bindings"],
        *before["extra_producer_bindings"],
        before["geometry_v2_reader_pin"],
        before["inherited_byte_receipt"],
        *before.get("affected_module_bindings", []),
    ]
    if not pins_match(pins):
        raise ValueError("Frozen producer/source bindings changed")
    base = output_path(args.base, ROOT / "tmp")
    junit = output_path(args.junit, ROOT / "artifacts/operational")
    prefix = output_path(args.prefix, ROOT / "artifacts/operational")
    paths = {
        name: Path(str(prefix) + "." + name)
        for name in ("stdout.log", "stderr.log", "launch.json", "exit.json")
    }
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise ValueError("Exclusive owned supervisor outputs already exist")
    command = [
        sys.executable,
        "-I",
        "-B",
        "-u",
        "-m",
        "pytest",
        "-q",
        "-x",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(base),
        "--junitxml",
        str(junit),
        *modules,
    ]
    environment = dict(os.environ)
    environment.update(
        dict(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
        )
    )
    started = datetime.now(UTC).isoformat()
    own_sha = digest(Path(__file__).resolve())
    with paths["stdout.log"].open("xb") as stdout, paths["stderr.log"].open("xb") as stderr:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            write_once(
                paths["launch.json"],
                dict(
                    kind="iris-r015-owned-writer-gate-launch-v1",
                    runner_pid=os.getpid(),
                    child_launch_pid=child.pid,
                    started_utc=started,
                    command=command,
                    cwd=str(ROOT),
                    before_path=str(before_path),
                    before_sha256=args.before_sha,
                    runner_sha256=own_sha,
                    gate_scope=mode,
                    expected_cases=count,
                    base=str(base),
                    junit=str(junit),
                    note="Popen PID may be a venv launcher; observe OS descendants while live",
                ),
            )
            exit_code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            raise
    ids: list[str] = []
    actual_junit_cases = 0
    junit_error: str | None = None
    if junit.exists():
        try:
            actual_junit_cases = len(list(ET.fromstring(junit.read_bytes()).iter("testcase")))
            ids = passing_ids(junit)
        except (ET.ParseError, KeyError, ValueError) as error:
            junit_error = f"{type(error).__name__}: {error}"
    stable = (
        pins_match(pins)
        and digest(before_path) == args.before_sha
        and inherited.scope() == (protected, originals, delta)
    )
    exact = ids == before["nodeids"] and len(ids) == len(set(ids)) == count
    write_once(
        paths["exit.json"],
        dict(
            kind="iris-r015-owned-writer-gate-exit-v1",
            runner_pid=os.getpid(),
            child_launch_pid=child.pid,
            finished_utc=datetime.now(UTC).isoformat(),
            actual_child_exit_code=exit_code,
            launch_sha256=digest(paths["launch.json"]),
            stdout_sha256=digest(paths["stdout.log"]),
            stderr_sha256=digest(paths["stderr.log"]),
            before_sha256_after=digest(before_path),
            runner_sha256_after=digest(Path(__file__).resolve()),
            frozen_bindings_stable=stable,
            actual_junit_cases=actual_junit_cases,
            validated_passing_id_count=len(ids),
            exact_ordered_junit_ids=exact,
            junit_parse_error=junit_error,
            junit_sha256=digest(junit) if junit.exists() else None,
            actual_runtime_passed=exit_code == 0 and stable and exact,
            independent_raw_readback_not_run=True,
            full_r015_complete=False,
            full_iris_gw_01_complete=False,
            scientific_acceptance="not_assessed",
        ),
    )
    return exit_code if exit_code != 0 else (0 if stable and exact else 1)


if __name__ == "__main__":
    raise SystemExit(main())
