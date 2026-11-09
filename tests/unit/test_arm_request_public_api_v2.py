"""Public package surface for arm-labelled request provenance v2."""

import iris.application as application
import iris.application.arm_request_command as command_module
import iris.application.request_artifact_v2 as artifact_application_module
import iris.application.verification_v2 as verification_module
import iris.artifacts as artifacts
import iris.artifacts.request_v2 as artifact_schema_module


def _assert_exports(
    exported: list[str],
    namespace: object,
    expected: dict[str, object],
) -> None:
    assert type(exported) is list
    assert len(exported) == len(set(exported))
    for name, value in expected.items():
        assert name in exported
        assert getattr(namespace, name) is value


def test_application_exports_complete_arm_request_vertical_slice() -> None:
    expected = {
        name: getattr(module, name)
        for module in (
            command_module,
            artifact_application_module,
            verification_module,
        )
        for name in module.__all__
        if "experiment" not in name.lower()
    }
    _assert_exports(application.__all__, application, expected)


def test_artifacts_export_every_persisted_request_v2_schema() -> None:
    expected = {
        name: getattr(artifact_schema_module, name) for name in artifact_schema_module.__all__
    }
    _assert_exports(artifacts.__all__, artifacts, expected)
