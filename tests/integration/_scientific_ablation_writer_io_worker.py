"""Fixed trusted IO/crash/lock worker; never candidate-generated Python.

Only canonical disk data and independently retained job anchors cross processes.
Abrupt exits deliberately bypass Python cleanup in THIS child only. The bounded
lock handshake exits itself; no guessed PID or launcher-only termination is used.
"""

from __future__ import annotations

import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def _bindings() -> int:
    count = 0
    for name, value in tuple(sys.modules.items()):
        if name == "iris" or name.startswith("iris."):
            filename = getattr(value, "__file__", None)
            if filename is not None:
                assert Path(filename).resolve().is_relative_to(ROOT), name
                count += 1
    assert not any(name == "tests" or name.startswith("tests.") for name in sys.modules)
    return count


def _event(phase: str, event: str, directory: Path, options: dict[str, Any]) -> bytes:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            metadata = path.lstat()
            observed: dict[str, object] = dict(
                bytes=metadata.st_size,
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
            if path.name == ".iris-owner.lock" and event in ("lock-acquired", "lock-deadline"):
                # Windows byte-range locks also deny reads through a second handle.
                # The active ProcessFileLock checks its own descriptor/identity.
                observed["content_not_read_while_locked"] = True
            else:
                data = path.read_bytes()
                assert len(data) == metadata.st_size
                observed["sha256"] = sha256(data).hexdigest()
            files[path.name] = observed
    return (
        json.dumps(
            dict(
                kind="iris-writer-io-child-event-v1",
                phase=phase,
                event=event,
                pid=os.getpid(),
                parent_pid=os.getppid(),
                directory=str(directory),
                files=files,
                report_sha256=options["expected_report_sha256"],
                source_sha256=options["expected_source_sha256"],
                loaded_project_bindings=_bindings(),
                all_project_bindings_staged=True,
                model_module_calls_constructors_and_loaders_denied=True,
                candidate_execution_denied=True,
                test_package_modules_imported=False,
                scientific_acceptance="not_assessed",
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _emit(content: bytes) -> None:
    sys.stdout.buffer.write(content)
    sys.stdout.buffer.flush()


def _marker(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    job_path = Path(sys.argv[1]).resolve()
    data = job_path.read_bytes()
    assert 1 <= len(data) <= 64 * 1024
    job: dict[str, Any] = json.loads(data)
    assert set(job) == {"phase", "input_root", "store_root", "marker", "release", "bindings"}
    phase = job["phase"]
    assert phase in ("read", "source", "report", "commit", "lock")
    paths = {
        name: Path(job[name]).resolve()
        for name in ("input_root", "store_root", "marker", "release")
    }
    assert all(path.is_relative_to(job_path.parent) for path in paths.values())
    options: dict[str, Any] = job["bindings"]

    # Warm third-party definitions before the import/IO constructor/call traps.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("No worker model")),
        patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No worker forward")),
        patch.object(Qwen3ForCausalLM, "__init__", side_effect=AssertionError("No worker Qwen")),
        patch.object(
            AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("No worker weights")
        ),
        patch.object(
            AutoTokenizer, "from_pretrained", side_effect=AssertionError("No worker tokenizer")
        ),
    ):
        from iris.application import scientific_ablation_writer_io as module
        from iris.application.scientific_ablation_inference_io import (
            _chain,
            _identity,
            _read_file,
            _regular,
        )
        from iris.artifacts.scientific_ablation_writer_reports_v1 import (
            ScientificAblationWriterCPURuntimeV1,
            read_scientific_ablation_writer_report_v1,
        )
        from iris.artifacts.scientific_ablation_writer_source_v1 import (
            read_scientific_ablation_writer_source_v1,
            scientific_ablation_writer_source_preflight_v1,
        )
        from iris.utils.directory_publication import rename_directory_with_windows_retry
        from iris.utils.process_file_lock import process_file_lock

        options["runtime"] = ScientificAblationWriterCPURuntimeV1(**options["runtime"])
        store = paths["store_root"] / module._store(options["expected_protocol"])
        target = store / options["expected_report_sha256"]
        with patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No worker candidate execution"),
        ):
            if phase == "read":
                result = module.read_scientific_ablation_writer_bundle_v1(target, **options)
                assert result.report.sha256 == options["expected_report_sha256"]
                assert result.source.sha256 == options["expected_source_sha256"]
                _emit(_event(phase, "read-complete", target, options))
                return 0

            source = read_scientific_ablation_writer_source_v1(
                _read_file(paths["input_root"] / "source.json", options["maximum_source_bytes"]),
                expected_sha256=options["expected_source_sha256"],
                maximum_source_bytes=options["maximum_source_bytes"],
            )
            pf = scientific_ablation_writer_source_preflight_v1(
                source, expected_sha256=options["expected_source_sha256"]
            )
            report = read_scientific_ablation_writer_report_v1(
                _read_file(paths["input_root"] / "report.json", options["maximum_report_bytes"]),
                preflight=pf,
                runtime=options["runtime"],
            )
            assert report.sha256 == options["expected_report_sha256"]
            assert source.payload.protocol == options["expected_protocol"]

            if phase == "lock":
                store.mkdir(parents=True, exist_ok=True)
                _chain(store)
                with process_file_lock(store / ".iris-owner.lock") as acquired:
                    claim = module._Claim(
                        store, _identity(_regular(store, directory=True)), acquired
                    )
                    claim.check()
                    event = _event(phase, "lock-acquired", store, options)
                    _marker(paths["marker"], event)
                    _emit(event)
                    deadline = time.monotonic() + 90
                    while not paths["release"].exists():
                        if time.monotonic() >= deadline:
                            _emit(_event(phase, "lock-deadline", store, options))
                            os._exit(81)
                        acquired.check()
                        time.sleep(0.05)
                    claim.check()
                _emit(_event(phase, "lock-released", store, options))
                return 0

            original_write = module._write

            def write(
                stage: module._Stage, name: str, content: bytes, claim: module._Claim
            ) -> None:
                original_write(stage, name, content, claim)
                if (phase, name) in (("source", "source.json"), ("report", "report.json")):
                    _emit(_event(phase, "abrupt-after-file", stage.path, options))
                    os._exit(73 if phase == "source" else 74)

            def rename(before: Path, after: Path) -> None:
                rename_directory_with_windows_retry(before, after)
                if phase == "commit":
                    _emit(_event(phase, "abrupt-after-commit", after, options))
                    os._exit(75)

            with (
                patch.object(module, "_write", new=write),
                patch.object(module, "rename_directory_with_windows_retry", new=rename),
            ):
                module.write_scientific_ablation_writer_bundle_v1(
                    report,
                    source=source,
                    root=paths["store_root"],
                    minimum_free_bytes=0,
                    **options,
                )
            raise AssertionError("Declared abrupt child exit point was not reached")


if __name__ == "__main__":
    raise SystemExit(main())
