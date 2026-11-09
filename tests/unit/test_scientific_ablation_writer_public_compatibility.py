"""Entire previous canonical API inventory/object bindings, not runtime parity."""

import ast
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/scientific_ablation_writer_public_legacy"
PINS = {
    "application": "32e71b13abcc074b9d322818f25be1d684cd070ecc04e16a53b7ef81f2d118b7",
    "artifacts": "26e7512a19b6a1fdfc6f39342c3e9ec529b942e9509f562dddfd27701d31ee45",
}
ADDITIONS = {
    "application": {
        "IrisScientificAblationWriterCommandError",
        "ScientificAblationWriterCommandResultV1",
        "inspect_scientific_ablation_writer_command_v1",
        "publish_scientific_ablation_writer_command_v1",
        "IrisScientificAblationWriterStorageError",
        "ScientificAblationWriterBundleV1",
        "ScientificAblationMessageWriterCompletedV1",
        "ScientificAblationWorkspaceWriterCompletedV1",
        "WriterCompleted",
        "read_scientific_ablation_writer_bundle_v1",
        "write_scientific_ablation_writer_bundle_v1",
        "measure_scientific_ablation_message_writer_completed_v1",
        "measure_scientific_ablation_message_writer_report_v1",
        "measure_scientific_ablation_workspace_writer_completed_v1",
        "measure_scientific_ablation_workspace_writer_report_v1",
    },
    "artifacts": {
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
        "IrisScientificAblationWriterSourceV1",
        "ScientificAblationWriterSourcePayloadV1",
        "WriterSourceProtocol",
        "build_scientific_ablation_writer_source_v1",
        "read_scientific_ablation_writer_source_v1",
        "scientific_ablation_writer_source_bytes_v1",
        "scientific_ablation_writer_source_preflight_v1",
    },
}


@pytest.mark.parametrize("namespace", ["application", "artifacts"])
def test_all_existing_export_names_order_and_declaring_objects_survive(namespace: str) -> None:
    data = (FIXTURES / f"{namespace}.py.txt").read_bytes()
    assert sha256(data).hexdigest() == PINS[namespace]
    tree = ast.parse(data.decode("utf-8"))
    declarations: dict[str, tuple[str, str]] = {}
    inventories = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module is not None
            for alias in node.names:
                name = alias.asname or alias.name
                assert name not in declarations
                declarations[name] = (node.module, alias.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            value: object = ast.literal_eval(node.value)
            assert isinstance(value, list) and all(isinstance(name, str) for name in value)
            inventories.append(value)
    assert len(inventories) == 1
    legacy = inventories[0]
    assert legacy and len(legacy) == len(set(legacy))
    public = import_module(f"iris.{namespace}")
    assert not set(legacy) & ADDITIONS[namespace]
    assert set(public.__all__) == set(legacy) | ADDITIONS[namespace]
    assert [name for name in public.__all__ if name not in ADDITIONS[namespace]] == legacy
    for name in legacy:
        module, original = declarations[name]
        assert public.__all__.count(name) == 1
        assert getattr(public, name) is getattr(import_module(module), original)
