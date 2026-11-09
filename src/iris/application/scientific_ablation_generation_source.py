"""Bind a persisted terminal lifecycle to its actual native-best inference weights.

No graph constructors, generation, scoring or filesystem writes. Three independent
trusted anchors are mandatory before state reads/decoding. This proves a terminal
source binding, not reproduction of original train/validation inputs, held-out data
eligibility, an acknowledged row capture, or permission to launch an experiment.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from iris.application.scientific_ablation_checkpoint import _RunConfiguration, _verify
from iris.application.scientific_ablation_inference import (
    IrisScientificAblationInferenceBundleV1,
    _bounded_metadata,
    _checked,
    _checked_manifest,
    _digest,
    _exact_tree,
)
from iris.application.scientific_ablation_inference_io import (
    _chain,
    _read_file,
    read_scientific_ablation_inference_v1,
)
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    _bundle_parts,
    _Configuration,
    read_scientific_ablation_lifecycle_checkpoint_v1,
)
from iris.application.scientific_checkpoint import _json
from iris.artifacts.scientific_ablation_generation_source_v1 import (
    IrisScientificAblationGenerationSourceManifestV1,
    ScientificAblationGenerationSourcePayloadV1,
)
from iris.artifacts.scientific_ablation_inference_v1 import (
    IrisScientificAblationInferenceManifestV1,
)
from iris.artifacts.scientific_ablation_lifecycle_checkpoint_v1 import (
    IrisScientificAblationLifecycleCheckpointManifestV1,
)
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256

_DOMAIN = "iris/scientific-ablation-generation-source/v1/payload"
_MAX_SOURCE_METADATA = 64 * 1024
_TEACHER_PREFIX = "model/base.teacher."


class IrisScientificAblationGenerationSourceError(RuntimeError):
    """Source is not a trusted persisted terminal-best/inference pair."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationGenerationSourceBundleV1:
    """Private verified inputs; native training state is never put in the descriptor."""

    manifest: IrisScientificAblationGenerationSourceManifestV1
    lifecycle: IrisScientificAblationLifecycleBundleV1 = field(repr=False)
    inference: IrisScientificAblationInferenceBundleV1 = field(repr=False)


def _metadata(
    lifecycle: IrisScientificAblationLifecycleCheckpointManifestV1,
    inference: IrisScientificAblationInferenceManifestV1,
    *,
    expected_lifecycle_sha256: str,
    expected_inference_sha256: str,
    expected_source_checkpoint_sha256: str,
) -> tuple[
    IrisScientificAblationLifecycleCheckpointManifestV1,
    IrisScientificAblationInferenceManifestV1,
    _Configuration,
    _RunConfiguration,
]:
    for digest in (
        expected_lifecycle_sha256,
        expected_inference_sha256,
        expected_source_checkpoint_sha256,
    ):
        _digest(digest)
    _exact_tree(lifecycle, IrisScientificAblationLifecycleCheckpointManifestV1)
    _bounded_metadata(lifecycle, maximum=16 * 1024**2)
    lifecycle = IrisScientificAblationLifecycleCheckpointManifestV1.model_validate(
        lifecycle.model_dump(mode="python", warnings=False)
    )
    native = lifecycle.payload
    if (
        lifecycle.sha256 != expected_lifecycle_sha256
        or domain_sha256(
            "iris/scientific-ablation-lifecycle-checkpoint/v1",
            canonical_json_bytes(native.model_dump(mode="json")),
        )
        != expected_lifecycle_sha256
        or native.state.phase != "stopped"
        or native.state.stop_reason is None
        or native.state.validation_count < 1
        or native.state.last_validation_step != native.state.step
        or native.best is None
        or native.best.sha256 != expected_source_checkpoint_sha256
    ):
        raise ValueError("Trusted terminal committed validation-best required before decode")
    inference = _checked_manifest(
        inference,
        expected_sha256=expected_inference_sha256,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_lifecycle_sha256=expected_lifecycle_sha256,
    )
    exported = inference.payload
    selected = exported.selection
    if (
        selected.kind != "selected_best"
        or selected.state != native.state
        or selected.lifecycle_configuration_json != native.configuration_json
        or selected.lifecycle_configuration_sha256 != native.configuration_sha256
        or selected.current_checkpoint_sha256 != native.current.sha256
        or selected.current_completed_steps != native.current.payload.completed_steps
        or selected.best_checkpoint_sha256 != native.best.sha256
        or exported.completed_steps != native.best.payload.completed_steps
        or exported.source_run_configuration_json != native.best.payload.configuration_json
        or exported.source_run_configuration_sha256 != native.best.payload.configuration_sha256
    ):
        raise ValueError("Inference is not the exact terminal lifecycle's selected best")
    configuration = _Configuration.model_validate(_json(native.configuration_json.encode()))
    run = _RunConfiguration.model_validate(_json(native.best.payload.configuration_json.encode()))
    # The inference verifier already checks canonical nested configurations,
    # hashes, cadence, purpose, model/policy, pinned revisions and runtime geometry.
    return lifecycle, inference, configuration, run


def _descriptor(
    lifecycle: IrisScientificAblationLifecycleCheckpointManifestV1,
    inference: IrisScientificAblationInferenceManifestV1,
    configuration: _Configuration,
    run: _RunConfiguration,
) -> IrisScientificAblationGenerationSourceManifestV1:
    native, exported = lifecycle.payload, inference.payload
    best, state = native.best, native.state
    if best is None or state.stop_reason is None or state.best_nll_per_token is None:
        raise ValueError("No terminal selected-best record")
    payload = ScientificAblationGenerationSourcePayloadV1(
        purpose=exported.purpose,
        ablation=exported.ablation,
        ablation_policy_sha256=exported.ablation_policy_sha256,
        lifecycle_sha256=lifecycle.sha256,
        lifecycle_configuration_sha256=native.configuration_sha256,
        current_checkpoint_sha256=native.current.sha256,
        best_checkpoint_sha256=best.sha256,
        selected_step=best.payload.completed_steps,
        terminal_step=state.step,
        stop_reason=state.stop_reason,
        best_validation_nll_per_token=state.best_nll_per_token,
        run_configuration_sha256=best.payload.configuration_sha256,
        model_configuration_sha256=run.model_configuration_sha256,
        initial_scientific_state_sha256=run.initial_scientific_state_sha256,
        training_corpus_sha256=run.dataset_sha256,
        training_examples=run.dataset_examples,
        validation_corpus_sha256=configuration.validation_sha256,
        validation_examples=configuration.validation_examples,
        validation_target_tokens=configuration.validation_target_tokens,
        backbone_runtime_sha256=run.backbone_runtime_sha256,
        inference_sha256=inference.sha256,
        teacher_sidecar_sha256=exported.source_teacher_sidecar_sha256,
        request_sidecar_sha256=exported.model.sha256,
    )
    return IrisScientificAblationGenerationSourceManifestV1(
        sha256=domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )


def _actual_projection(
    lifecycle: IrisScientificAblationLifecycleBundleV1,
    inference: IrisScientificAblationInferenceBundleV1,
    *,
    expected_lifecycle_sha256: str,
    expected_inference_sha256: str,
    expected_source_checkpoint_sha256: str,
) -> None:
    _, _, best = _bundle_parts(lifecycle)
    if best is None:
        raise ValueError("No actual selected native checkpoint")
    selected, native_tensors = _verify(best)
    exported, request_tensors = _checked(
        inference,
        expected_sha256=expected_inference_sha256,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_lifecycle_sha256=expected_lifecycle_sha256,
    )
    names = tuple(
        name.removeprefix("base.teacher.")
        for name in selected.payload.parameter_names
        if name.startswith("base.teacher.")
    )
    if set(names) != set(exported.payload.model.state_names) or any(
        not torch.equal(native_tensors[_TEACHER_PREFIX + name], request_tensors[name])
        for name in names
    ):
        raise ValueError("Actual inference weights are not the selected native teacher")


def _disk_metadata(path: Path, expected: str, maximum: int) -> bytes:
    _digest(expected)
    if not isinstance(path, Path) or path.name != expected:
        raise ValueError("Explicit digest-named persisted source directory required")
    _chain(path)
    content = _read_file(path / "manifest.json", maximum)
    if not content.endswith(b"\n"):
        raise ValueError("Persisted source manifest requires canonical LF")
    _json(content[:-1])
    return content


def read_scientific_ablation_generation_source_v1(
    *,
    lifecycle_path: Path,
    inference_path: Path,
    expected_lifecycle_sha256: str,
    expected_inference_sha256: str,
    expected_source_checkpoint_sha256: str,
) -> IrisScientificAblationGenerationSourceBundleV1:
    """Read persisted terminal source without constructing a teacher/request.

    Trust anchors must come from a trusted operator/producer, never these candidates.
    Both manifests authenticate and match BEFORE either state's read/decode.
    Subsequent native/IO verifiers authenticate complete bytes; every inference
    parameter must equal the actual native-best teacher slice, not a metadata tag.
    """
    try:
        for digest in (
            expected_lifecycle_sha256,
            expected_inference_sha256,
            expected_source_checkpoint_sha256,
        ):
            _digest(digest)
        native_raw = _disk_metadata(lifecycle_path, expected_lifecycle_sha256, 16 * 1024**2)
        exported_raw = _disk_metadata(inference_path, expected_inference_sha256, 4 * 1024**2)
        metadata = _metadata(
            IrisScientificAblationLifecycleCheckpointManifestV1.model_validate_json(native_raw),
            IrisScientificAblationInferenceManifestV1.model_validate_json(exported_raw),
            expected_lifecycle_sha256=expected_lifecycle_sha256,
            expected_inference_sha256=expected_inference_sha256,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        )
        if any(
            canonical_json_file_bytes(manifest.model_dump(mode="json")) != raw
            for manifest, raw in zip(metadata[:2], (native_raw, exported_raw), strict=True)
        ):
            raise ValueError("Noncanonical source manifest bytes")
        lifecycle = read_scientific_ablation_lifecycle_checkpoint_v1(lifecycle_path)
        inference = read_scientific_ablation_inference_v1(
            inference_path,
            expected_sha256=expected_inference_sha256,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
            expected_lifecycle_sha256=expected_lifecycle_sha256,
        )
        if lifecycle.manifest != metadata[0] or inference.manifest != metadata[1]:
            raise ValueError("Source manifests changed between preflight and complete read")
        _actual_projection(
            lifecycle,
            inference,
            expected_lifecycle_sha256=expected_lifecycle_sha256,
            expected_inference_sha256=expected_inference_sha256,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        )
        return IrisScientificAblationGenerationSourceBundleV1(
            _descriptor(*metadata), lifecycle, inference
        )
    except Exception as error:
        raise IrisScientificAblationGenerationSourceError("Terminal source read failed") from error


def verify_scientific_ablation_generation_source_v1(
    bundle: IrisScientificAblationGenerationSourceBundleV1,
    *,
    expected_sha256: str,
    expected_lifecycle_sha256: str,
    expected_inference_sha256: str,
    expected_source_checkpoint_sha256: str,
) -> IrisScientificAblationGenerationSourceManifestV1:
    """Revalidate private source bytes and return defensive exact redacted metadata."""
    try:
        _digest(expected_sha256)
        if (
            type(bundle) is not IrisScientificAblationGenerationSourceBundleV1
            or type(bundle.lifecycle) is not IrisScientificAblationLifecycleBundleV1
            or type(bundle.inference) is not IrisScientificAblationInferenceBundleV1
        ):
            raise ValueError("Exact private mode-bound source bundle required")
        _exact_tree(bundle.manifest, IrisScientificAblationGenerationSourceManifestV1)
        _bounded_metadata(bundle.manifest, maximum=_MAX_SOURCE_METADATA)
        manifest = IrisScientificAblationGenerationSourceManifestV1.model_validate(
            bundle.manifest.model_dump(mode="python", warnings=False)
        )
        if manifest.sha256 != expected_sha256 or manifest.sha256 != domain_sha256(
            _DOMAIN, canonical_json_bytes(manifest.payload.model_dump(mode="json"))
        ):
            raise ValueError("Trusted source descriptor differs before state decode")
        metadata = _metadata(
            bundle.lifecycle.manifest,
            bundle.inference.manifest,
            expected_lifecycle_sha256=expected_lifecycle_sha256,
            expected_inference_sha256=expected_inference_sha256,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        )
        if manifest != _descriptor(*metadata):
            raise ValueError("Descriptor differs from actual native/inference source")
        _actual_projection(
            bundle.lifecycle,
            bundle.inference,
            expected_lifecycle_sha256=expected_lifecycle_sha256,
            expected_inference_sha256=expected_inference_sha256,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        )
        return manifest
    except Exception as error:
        raise IrisScientificAblationGenerationSourceError(
            "Terminal source verification failed"
        ) from error


__all__ = [
    "IrisScientificAblationGenerationSourceBundleV1",
    "IrisScientificAblationGenerationSourceError",
    "read_scientific_ablation_generation_source_v1",
    "verify_scientific_ablation_generation_source_v1",
]
