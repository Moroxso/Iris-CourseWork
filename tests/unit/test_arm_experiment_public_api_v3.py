"""Public package surface for the profile-v4 B/C/D experiment slice."""

import iris.application as application
import iris.application.arm_experiment_command_v3 as command_module
import iris.application.arm_training_checkpoint as checkpoint_module
import iris.application.experiment_execution_v3 as execution_module
import iris.application.experiment_log_v3 as log_module
import iris.application.experiment_runner_v3 as runner_module
import iris.application.verification_v3 as verification_module
import iris.artifacts as artifacts
import iris.artifacts.experiment_log_v3 as artifact_module


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


def test_application_exports_complete_profile_v4_experiment_vertical_slice() -> None:
    expected = {
        name: getattr(module, name)
        for module in (
            command_module,
            checkpoint_module,
            execution_module,
            log_module,
            runner_module,
            verification_module,
        )
        for name in module.__all__
    }
    _assert_exports(application.__all__, application, expected)


def test_artifacts_export_every_persisted_experiment_v3_schema() -> None:
    expected = {name: getattr(artifact_module, name) for name in artifact_module.__all__}
    _assert_exports(artifacts.__all__, artifacts, expected)

