"""Canonical namespaces bind the exact new APIs, not aliases of baseline domains."""

from importlib import import_module

import pytest

import iris.application as application
import iris.artifacts as artifacts

_APPLICATION = {
    "scientific_ablation_inference": (
        "IrisScientificAblationInferenceBundleV1",
        "IrisScientificAblationInferenceError",
        "build_scientific_ablation_inference_v1",
        "verify_scientific_ablation_inference_v1",
    ),
    "scientific_ablation_inference_io": (
        "read_scientific_ablation_inference_v1",
        "write_scientific_ablation_inference_v1",
    ),
    "scientific_ablation_receipt": (
        "IrisScientificAblationRequestOutputV1",
        "IrisScientificAblationRequestReceiptV1",
    ),
    "scientific_ablation_request": (
        "IrisScientificAblationRequestV1",
        "build_scientific_ablation_request_v1",
    ),
}
_ARTIFACTS = (
    "IrisScientificAblationInferenceManifestV1",
    "ScientificAblationInferenceGenerationPolicyV1",
    "ScientificAblationInferencePayloadV1",
    "ScientificAblationInferenceSelectionV1",
)


@pytest.mark.parametrize(
    "module,name", [(module, name) for module, names in _APPLICATION.items() for name in names]
)
def test_application_exports_exact_public_object_once(module: str, name: str) -> None:
    assert application.__all__.count(name) == 1
    assert getattr(application, name) is getattr(import_module(f"iris.application.{module}"), name)


@pytest.mark.parametrize("name", _ARTIFACTS)
def test_artifacts_exports_exact_schema_once(name: str) -> None:
    assert artifacts.__all__.count(name) == 1
    assert getattr(artifacts, name) is getattr(
        import_module("iris.artifacts.scientific_ablation_inference_v1"), name
    )


def test_no_private_owner_capability_or_legacy_type_laundering_export() -> None:
    assert not any(name.startswith("_") for name in application.__all__)
    assert id(application.IrisScientificAblationInferenceBundleV1) != id(
        application.IrisScientificInferenceBundleV1
    )
    assert id(artifacts.IrisScientificAblationInferenceManifestV1) != id(
        artifacts.IrisScientificInferenceManifestV1
    )
    assert id(application.build_scientific_ablation_request_v1) != id(
        application.build_request_orchestrator
    )
