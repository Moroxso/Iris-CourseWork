"""Complete cold file/CLI delivery from pinned retained DATA, not new native work."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
from typer.testing import CliRunner

from iris.application import scientific_ablation_writer_command as module
from iris.application import scientific_ablation_writer_io as io
from iris.application.scientific_ablation_inference_io import _read_file
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    read_scientific_ablation_writer_report_v1,
)
from iris.cli import app

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/scientific_ablation_writer_command"
WORKER = Path(__file__).with_name("_scientific_ablation_writer_command_worker.py")
PINS = {
    "workspace": {
        "source.json": "736043226a731ab3624ab68cd02a0f3bef55940710304352f55140d65a9dd701",
        "report.json": "53e48403675243b10a1ba9b1271f322ee1e5597318ce0b47bd188ab7dacf3e88",
        "bindings.json": "e274a2e71e63af8df3061a979c635288fa92f6ddbc006e82a8e5e4704e7360ae",
    },
    "message": {
        "source.json": "435b5f3577fb646ff95932b1fec00f2ad27a97a56664a48aa1ec057f20b8370c",
        "report.json": "5105539e17de1b47d32747643ac5f716a417714838ef9491b77cf9cad1bcea7b",
        "bindings.json": "7e25b89f8437303ac2056c3b68a68cec0eb11340d8bde2859fe59cfa357d0852",
    },
}


@dataclass(frozen=True, slots=True)
class Job:
    protocol: str
    root: Path
    bindings: dict[str, Any]
    source_content: bytes
    report_content: bytes

    def arguments(self) -> dict[str, Any]:
        return self.bindings | {
            "runtime": ScientificAblationWriterCPURuntimeV1(**self.bindings["runtime"])
        }

    def cli(self) -> list[str]:
        return [
            "--protocol",
            self.protocol,
            "--expected-source-sha256",
            self.bindings["expected_source_sha256"],
            "--expected-report-sha256",
            self.bindings["expected_report_sha256"],
            "--expected-torch-version",
            self.bindings["runtime"]["torch_version"],
            "--expected-torch-git-version",
            self.bindings["runtime"]["torch_git_version"],
            "--maximum-source-bytes",
            str(self.bindings["maximum_source_bytes"]),
            "--maximum-report-bytes",
            str(self.bindings["maximum_report_bytes"]),
        ]


@pytest.fixture(params=("workspace", "message"))
def job(request: pytest.FixtureRequest, tmp_path: Path) -> Job:
    protocol = str(request.param)
    directory = FIXTURES / protocol
    original = {}
    for name, expected in PINS[protocol].items():
        data = (directory / name).read_bytes()
        assert sha256(data).hexdigest() == expected
        original[name] = data
    bindings = json.loads(original["bindings.json"])
    assert bindings["expected_protocol"] == protocol
    root = tmp_path / "inputs"
    root.mkdir()
    for name in ("source.json", "report.json"):
        shutil.copyfile(directory / name, root / name)
    return Job(protocol, root, bindings, original["source.json"], original["report.json"])


@pytest.fixture(autouse=True)
def no_execution() -> Iterator[None]:
    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("No command model")),
        patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No command forward")),
        patch.object(Qwen3ForCausalLM, "__init__", side_effect=AssertionError("No command Qwen")),
        patch.object(
            AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("No weights")
        ),
        patch.object(AutoTokenizer, "from_pretrained", side_effect=AssertionError("No tokenizer")),
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate Python"),
        ),
        patch(
            "iris.application.scientific_ablation_writer_effects._measure_writer_effects",
            side_effect=AssertionError("No native writer stage"),
        ),
        patch(
            "iris.application.scientific_ablation_causal_controls._source_transaction",
            side_effect=AssertionError("No native source transaction"),
        ),
    ):
        yield


def publish(job: Job, root: Path, **changes: Any) -> module.ScientificAblationWriterCommandResultV1:
    return module.publish_scientific_ablation_writer_command_v1(
        **(
            dict(
                source_path=job.root / "source.json",
                report_path=job.root / "report.json",
                output_root=root,
                minimum_free_bytes=0,
                **job.arguments(),
            )
            | changes
        )
    )


def assert_complete(job: Job, result: module.ScientificAblationWriterCommandResultV1) -> None:
    original = json.loads(job.report_content)["payload"]
    assert result.protocol == job.protocol
    assert result.source_sha256 == job.bindings["expected_source_sha256"]
    assert result.report_sha256 == job.bindings["expected_report_sha256"]
    assert result.effects == len(original["effects"])
    assert result.pairs == len(original["statistics"]["pairs"])
    assert result.stored_teacher_executions == (24 if job.protocol == "workspace" else 20)
    assert result.scope == "stored-writer-data-not-fresh-native-execution"
    assert result.scientific_acceptance == "not_assessed"
    assert (result.path / "source.json").read_bytes() == job.source_content
    assert (result.path / "report.json").read_bytes() == job.report_content
    assert {p.name for p in result.path.iterdir()} == {"source.json", "report.json"}


def test_entire_file_publication_inspection_and_retry_do_not_repeat_work(
    job: Job, tmp_path: Path
) -> None:
    before = torch.get_rng_state().clone()
    result = publish(job, tmp_path / "published")
    assert_complete(job, result)
    identities = {p.name: p.stat().st_ino for p in result.path.iterdir()}
    with patch.object(
        io, "_write", side_effect=AssertionError("Exact command retry must not write")
    ):
        assert publish(job, tmp_path / "published") == result
    assert identities == {p.name: p.stat().st_ino for p in result.path.iterdir()}
    assert (
        module.inspect_scientific_ablation_writer_command_v1(result.path, **job.arguments())
        == result
    )
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "change",
    ["source", "report", "protocol", "runtime", "source-cap", "report-cap", "storage", "reserve"],
)
def test_wrong_independent_anchor_or_budget_creates_no_store(
    job: Job, tmp_path: Path, change: str
) -> None:
    changes: dict[str, Any]
    if change in ("source", "report"):
        changes = {f"expected_{change}_sha256": "f" * 64}
    elif change == "protocol":
        changes = {"expected_protocol": "message" if job.protocol == "workspace" else "workspace"}
    elif change == "runtime":
        changes = {"runtime": ScientificAblationWriterCPURuntimeV1("untrusted", "f" * 40)}
    elif change in ("source-cap", "report-cap"):
        name = "maximum_" + change.split("-")[0] + "_bytes"
        changes = {name: job.bindings[name] - 1}
    elif change == "storage":
        changes = {"maximum_storage_bytes": 1}
    else:
        changes = {"minimum_free_bytes": -1}
    root = tmp_path / "must-not-exist"
    with pytest.raises(module.IrisScientificAblationWriterCommandError):
        publish(job, root, **changes)
    assert not root.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_source_sha256", "bad"),
        ("expected_report_sha256", "bad"),
        ("expected_protocol", "unknown"),
        ("maximum_source_bytes", True),
        ("maximum_source_bytes", 0),
        ("maximum_report_bytes", 512 * 1024**2 + 1),
        ("maximum_storage_bytes", False),
        ("minimum_free_bytes", -1),
    ],
)
def test_malformed_expectations_denied_before_any_input_read(
    job: Job, tmp_path: Path, field: str, value: object
) -> None:
    with patch.object(module, "_read_file", side_effect=AssertionError("No input open")) as read:
        with pytest.raises(module.IrisScientificAblationWriterCommandError):
            publish(job, tmp_path / "must-not-exist", **{field: value})
        read.assert_not_called()


def test_wrong_original_source_is_denied_before_report_read(job: Job, tmp_path: Path) -> None:
    with (
        patch.object(module, "_read_file", wraps=_read_file) as read,
        pytest.raises(module.IrisScientificAblationWriterCommandError),
    ):
        publish(job, tmp_path / "must-not-exist", expected_source_sha256="f" * 64)
    assert len(read.call_args_list) == 1
    assert read.call_args_list[0].args[0] == job.root / "source.json"


@pytest.mark.parametrize("name", ["source.json", "report.json"])
def test_either_input_changed_after_parsing_denies_before_publication(
    job: Job, tmp_path: Path, name: str
) -> None:
    original = read_scientific_ablation_writer_report_v1

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        path = job.root / name
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    with (
        patch.object(module, "read_scientific_ablation_writer_report_v1", new=changed),
        pytest.raises(module.IrisScientificAblationWriterCommandError),
    ):
        publish(job, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("kind", ["error", "interrupt", "system-exit"])
def test_failure_after_real_commit_never_becomes_success_and_retry_never_writes(
    job: Job, tmp_path: Path, kind: str
) -> None:
    failure = {
        "error": RuntimeError("lost ack"),
        "interrupt": KeyboardInterrupt(),
        "system-exit": SystemExit(0),
    }[kind]
    expected = (
        module.IrisScientificAblationWriterCommandError
        if isinstance(failure, Exception)
        else type(failure)
    )
    original = io.write_scientific_ablation_writer_bundle_v1
    committed: list[Path] = []

    def completed(*args: Any, **kwargs: Any) -> Path:
        path = original(*args, **kwargs)
        committed.append(path)
        raise failure

    with (
        patch.object(io, "write_scientific_ablation_writer_bundle_v1", new=completed),
        pytest.raises(expected),
    ):
        publish(job, tmp_path / "published")
    assert len(committed) == 1
    assert (committed[0] / "source.json").read_bytes() == job.source_content
    assert (committed[0] / "report.json").read_bytes() == job.report_content
    identities = {p.name: p.stat().st_ino for p in committed[0].iterdir()}
    with patch.object(io, "_write", side_effect=AssertionError("No lost-ack retry writes")):
        result = publish(job, tmp_path / "published")
    assert_complete(job, result)
    assert identities == {p.name: p.stat().st_ino for p in result.path.iterdir()}


def cli_publish(job: Job, root: Path) -> list[str]:
    return [
        "scientific",
        "publish-writer",
        "--source",
        str(job.root / "source.json"),
        "--report",
        str(job.root / "report.json"),
        "--output-root",
        str(root),
        *job.cli(),
    ]


def test_real_cli_publishes_and_inspects_only_redacted_stored_metadata(
    job: Job, tmp_path: Path
) -> None:
    root = tmp_path / "published[bold]literal"
    result = CliRunner().invoke(app, cli_publish(job, root))
    assert result.exit_code == 0, result.output
    assert "stored-writer-data-not-fresh-native-execution" in result.stdout
    assert "not_assessed" in result.stdout
    compact = "".join(char for char in result.stdout if not char.isspace() and char not in "│┃")
    assert "[bold]literal" in compact
    assert job.bindings["expected_report_sha256"] in compact
    assert job.bindings["expected_source_sha256"] in compact
    target = (
        root
        / f"scientific-ablation-{job.protocol}-writer-v1"
        / job.bindings["expected_report_sha256"]
    )
    result = CliRunner().invoke(app, ["scientific", "inspect-writer", str(target), *job.cli()])
    assert result.exit_code == 0, result.output
    assert "Stored teacher executions" in result.stdout
    assert "Traceback" not in result.output
    assert (target / "source.json").read_bytes() == job.source_content
    assert (target / "report.json").read_bytes() == job.report_content


@pytest.mark.parametrize(
    "missing",
    [
        "--protocol",
        "--expected-source-sha256",
        "--expected-report-sha256",
        "--expected-torch-version",
        "--expected-torch-git-version",
        "--maximum-source-bytes",
    ],
)
def test_cli_requires_independent_expectations_never_infers_from_candidate(
    job: Job, tmp_path: Path, missing: str
) -> None:
    root = tmp_path / "must-not-exist"
    arguments = cli_publish(job, root)
    index = arguments.index(missing)
    del arguments[index : index + 2]
    with patch.object(
        module, "_read_file", side_effect=AssertionError("No CLI candidate read")
    ) as read:
        result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    read.assert_not_called()
    assert not root.exists()


def test_actual_cold_cli_publish_and_inspect_processes_have_no_native_or_model_attempts(
    job: Job, tmp_path: Path
) -> None:
    root = tmp_path / "published"
    target = (
        root
        / f"scientific-ablation-{job.protocol}-writer-v1"
        / job.bindings["expected_report_sha256"]
    )
    for phase in ("publish", "inspect"):
        job_path = tmp_path / (phase + "-job.json")
        packet = dict(
            phase=phase,
            input_root=str(job.root),
            output_root=str(root),
            artifact_path=str(target),
            bindings=job.bindings,
        )
        job_path.write_bytes(
            (json.dumps(packet, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        command = [sys.executable, "-I", "-B", str(WORKER), str(job_path)]
        environment = dict(os.environ)
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
        )
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as child:
            stdout, stderr = child.communicate()
            code, launch_pid = child.returncode, child.pid
        evidence = tmp_path / ("cold-" + phase)
        evidence.mkdir()
        (evidence / "worker.stdout.log").write_bytes(stdout)
        (evidence / "worker.stderr.log").write_bytes(stderr)
        (evidence / "worker-outer.json").write_bytes(
            (
                json.dumps(
                    dict(
                        actual_child_returncode=code,
                        observer_pid=os.getpid(),
                        launch_pid=launch_pid,
                        command_argv=command,
                        worker_sha256=sha256(WORKER.read_bytes()).hexdigest(),
                        job_path=str(job_path),
                        job_sha256=sha256(job_path.read_bytes()).hexdigest(),
                        stdout_sha256=sha256(stdout).hexdigest(),
                        stderr_sha256=sha256(stderr).hexdigest(),
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
        assert code == 0, (stdout + stderr).decode(errors="replace")
        assert stderr == b""
        assert b"stored-writer-data-not-fresh-native-execution" in stdout
        assert b"not_assessed" in stdout
        events = [
            line.removeprefix(b"IRIS_WRITER_COMMAND_EVENT ")
            for line in stdout.splitlines()
            if line.startswith(b"IRIS_WRITER_COMMAND_EVENT ")
        ]
        assert len(events) == 1
        event = json.loads(events[0])
        assert event["phase"] == phase and event["actual_cli_exit_code"] == 0
        assert event["all_project_bindings_staged"] is True
        assert event["no_model_native_candidate_attempts"] is True
        assert event["caller_torch_state_unchanged"] is True
        assert event["test_package_modules_imported"] is False
        assert event["project_bindings"] > 0
        assert (event["pid"] == launch_pid and event["parent_pid"] == os.getpid()) or (
            event["pid"] != launch_pid and event["parent_pid"] == launch_pid
        )
    assert (target / "source.json").read_bytes() == job.source_content
    assert (target / "report.json").read_bytes() == job.report_content
