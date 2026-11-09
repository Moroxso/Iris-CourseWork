"""Public application surface for trained request-sidecar transfer."""

import iris.application as application
import iris.application.arm_request_checkpoint as transfer_module


def test_application_exports_complete_checkpoint_transfer_surface() -> None:
    assert type(application.__all__) is list
    assert len(application.__all__) == len(set(application.__all__))
    for name in transfer_module.__all__:
        assert name in application.__all__
        assert getattr(application, name) is getattr(transfer_module, name)


def test_checkpoint_transfer_error_is_public() -> None:
    assert "IrisRequestCheckpointError" in application.__all__
    assert application.IrisRequestCheckpointError.__module__ == "iris.application.errors"
