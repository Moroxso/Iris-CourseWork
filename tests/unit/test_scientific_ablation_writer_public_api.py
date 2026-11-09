"""Exact writer exports preserve legacy signatures and import without native work."""

import inspect
import json
import os
import subprocess
import sys
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest

import iris.application as application
import iris.artifacts as artifacts
from iris.artifacts.scientific_workspace_v1 import IrisScientificWorkspaceV1

ROOT = Path(__file__).resolve().parents[2]
APPLICATION = {
    "scientific_ablation_writer_command": (
        "IrisScientificAblationWriterCommandError",
        "ScientificAblationWriterCommandResultV1",
        "inspect_scientific_ablation_writer_command_v1",
        "publish_scientific_ablation_writer_command_v1",
    ),
    "scientific_ablation_writer_io": (
        "IrisScientificAblationWriterStorageError",
        "ScientificAblationWriterBundleV1",
        "read_scientific_ablation_writer_bundle_v1",
        "write_scientific_ablation_writer_bundle_v1",
    ),
    "scientific_ablation_writer_reports": (
        "ScientificAblationMessageWriterCompletedV1",
        "ScientificAblationWorkspaceWriterCompletedV1",
        "WriterCompleted",
        "measure_scientific_ablation_message_writer_completed_v1",
        "measure_scientific_ablation_message_writer_report_v1",
        "measure_scientific_ablation_workspace_writer_completed_v1",
        "measure_scientific_ablation_workspace_writer_report_v1",
    ),
}
ARTIFACTS = {
    "scientific_ablation_writer_reports_v1": (
        "IrisScientificAblationMessageWriterReportV1",
        "IrisScientificAblationWorkspaceWriterReportV1",
        "ScientificAblationMessageWriterReportPayloadV1",
        "ScientificAblationWorkspaceWriterReportPayloadV1",
        "ScientificAblationWriterCPURuntimeV1",
        "ScientificAblationWriterSourceMetadataV1",
        "WriterReport",
        "build_scientific_ablation_writer_report_v1",
        "read_scientific_ablation_writer_report_v1",
        "scientific_ablation_writer_report_bytes_v1",
    ),
    "scientific_ablation_writer_source_v1": (
        "IrisScientificAblationWriterSourceV1",
        "ScientificAblationWriterSourcePayloadV1",
        "WriterSourceProtocol",
        "build_scientific_ablation_writer_source_v1",
        "read_scientific_ablation_writer_source_v1",
        "scientific_ablation_writer_source_bytes_v1",
        "scientific_ablation_writer_source_preflight_v1",
    ),
}


@pytest.mark.parametrize(
    "namespace,module,name",
    [
        (namespace, module, name)
        for namespace, modules in (("application", APPLICATION), ("artifacts", ARTIFACTS))
        for module, names in modules.items()
        for name in names
    ],
)
def test_canonical_export_is_exact_declaring_object_once(
    namespace: str, module: str, name: str
) -> None:
    public = import_module(f"iris.{namespace}")
    declared = import_module(f"iris.{namespace}.{module}")
    assert public.__all__.count(name) == 1
    assert getattr(public, name) is getattr(declared, name)


def test_no_private_owner_or_filesystem_capability_exports() -> None:
    for public in (application, artifacts):
        assert len(public.__all__) == len(set(public.__all__))
        assert not any(name.startswith("_") for name in public.__all__)
        assert not {"_Claim", "_Stage", "_Source", "_SourceWire"} & set(public.__all__)


def test_writer_domains_are_not_legacy_or_cross_protocol_aliases() -> None:
    assert id(artifacts.IrisScientificAblationMessageWriterReportV1) != id(
        artifacts.IrisScientificAblationWorkspaceWriterReportV1
    )
    assert id(artifacts.IrisScientificAblationWorkspaceWriterReportV1) != id(
        IrisScientificWorkspaceV1
    )
    assert id(application.ScientificAblationWriterBundleV1) != id(
        application.IrisScientificAblationInferenceBundleV1
    )
    assert id(application.ScientificAblationMessageWriterCompletedV1) != id(
        application.ScientificAblationWorkspaceWriterCompletedV1
    )
    assert id(application.read_scientific_ablation_writer_bundle_v1) != id(
        application.read_scientific_ablation_inference_v1
    )


def test_legacy_report_only_signatures_remain_literal() -> None:
    for protocol in ("workspace", "message"):
        function = getattr(application, f"measure_scientific_ablation_{protocol}_writer_report_v1")
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == ("heldout", "input", "policy", "limits")
        assert signature.parameters["heldout"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(
            signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            and signature.parameters[name].default is inspect.Parameter.empty
            for name in ("input", "policy", "limits")
        )
        expected = f"IrisScientificAblation{protocol.title()}WriterReportV1"
        assert signature.return_annotation is getattr(artifacts, expected)


@pytest.mark.parametrize("first", ["application", "artifacts"])
def test_actual_cold_namespace_import_has_no_model_native_or_candidate_work(
    first: str, tmp_path: Path
) -> None:
    worker = ROOT / "tests/integration/_scientific_ablation_writer_public_worker.py"
    before = sha256(worker.read_bytes()).hexdigest()
    command = [sys.executable, "-I", "-B", str(worker), first]
    with subprocess.Popen(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as child:
        stdout, stderr = child.communicate()
        code, launch_pid = child.returncode, child.pid
    (tmp_path / "worker.stdout.jsonl").write_bytes(stdout)
    (tmp_path / "worker.stderr.log").write_bytes(stderr)
    (tmp_path / "worker-outer.json").write_text(
        json.dumps(
            dict(
                actual_child_returncode=code,
                launch_pid=launch_pid,
                observer_pid=os.getpid(),
                command_argv=command,
                worker_sha256=before,
                stdout_sha256=sha256(stdout).hexdigest(),
                stderr_sha256=sha256(stderr).hexdigest(),
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert code == 0, (stdout + stderr).decode(errors="replace")
    assert stderr == b""
    assert before == sha256(worker.read_bytes()).hexdigest()
    event = json.loads(stdout)
    assert event["first_namespace"] == first
    assert event["all_project_bindings_staged"] is True
    assert event["no_model_native_candidate_work"] is True
    assert event["caller_torch_state_unchanged"] is True
    assert event["test_package_modules_imported"] is False
    assert event["project_bindings"] > 0
    assert (event["pid"] == launch_pid and event["parent_pid"] == os.getpid()) or (
        event["pid"] != launch_pid and event["parent_pid"] == launch_pid
    )
