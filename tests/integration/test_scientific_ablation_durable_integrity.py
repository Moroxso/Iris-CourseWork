"""Rehashed durable selected-best candidates reject before native owner writes."""

from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load as load_tensors

from iris.application import scientific_ablation_checkpoint as native
from iris.application import scientific_ablation_lifecycle_checkpoint as io
from iris.application.scientific_ablation_lifecycle import IrisScientificAblationLifecycleV1
from iris.application.scientific_ablation_validation import (
    _snapshot,
    own_scientific_ablation_validation_rows_v1,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.training_checkpoint import _write_exact_file
from iris.artifacts.scientific_ablation_checkpoint_v1 import (
    IrisScientificAblationCheckpointManifestV1,
)
from iris.artifacts.scientific_ablation_lifecycle_checkpoint_v1 import (
    IrisScientificAblationLifecycleCheckpointManifestV1,
    ScientificAblationLifecycleCheckpointPayloadV1,
)
from iris.artifacts.scientific_ablation_lifecycle_v1 import (
    IrisScientificAblationLifecyclePolicyV1,
    ScientificAblationLifecycleStateV1,
)
from iris.artifacts.scientific_checkpoint_v1 import ScientificRuntimeV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256, named_module_state_sha256
from iris.utils.process_file_lock import process_file_lock
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_checkpoint import _repack
from tests.integration.test_scientific_ablation_durable_lifecycle import _durable, _receipt
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def durable_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> io.IrisScientificAblationLifecycleBundleV1:
    controller = _durable(tmp_path_factory.mktemp("durable-integrity-source"))
    bundle = io.read_scientific_ablation_lifecycle_checkpoint_v1(_receipt(controller).path)
    assert bundle.manifest.payload.current.payload.completed_steps == 3
    assert bundle.manifest.payload.best is not None
    assert bundle.manifest.payload.best.payload.completed_steps == 1
    assert bundle.best_state_bytes is not None
    return bundle


def _components(
    bundle: io.IrisScientificAblationLifecycleBundleV1,
) -> tuple[
    native.IrisScientificAblationCheckpointBundleV1,
    native.IrisScientificAblationCheckpointBundleV1,
]:
    assert bundle.manifest.payload.best is not None and bundle.best_state_bytes is not None
    return (
        native.IrisScientificAblationCheckpointBundleV1(
            bundle.manifest.payload.current, bundle.current_state_bytes
        ),
        native.IrisScientificAblationCheckpointBundleV1(
            bundle.manifest.payload.best, bundle.best_state_bytes
        ),
    )


def _native_candidate(
    bundle: native.IrisScientificAblationCheckpointBundleV1,
    *,
    updates: dict[str, Any] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
) -> native.IrisScientificAblationCheckpointBundleV1:
    repacked = _repack(bundle, updates=updates, tensors=tensors)
    # The shared helper rehashes bytes, descriptors and payload, but uses model_copy.
    # Require a genuinely valid schema here, not an unchecked nested dictionary.
    manifest = IrisScientificAblationCheckpointManifestV1.model_validate(
        repacked.manifest.model_dump(mode="python")
    )
    return native.IrisScientificAblationCheckpointBundleV1(manifest, repacked.state_bytes)


def _envelope(
    bundle: io.IrisScientificAblationLifecycleBundleV1,
    *,
    current: native.IrisScientificAblationCheckpointBundleV1 | None = None,
    best: native.IrisScientificAblationCheckpointBundleV1 | None = None,
    state: ScientificAblationLifecycleStateV1 | None = None,
    configuration: io._Configuration | None = None,
) -> io.IrisScientificAblationLifecycleBundleV1:
    original_current, original_best = _components(bundle)
    current = original_current if current is None else current
    best = original_best if best is None else best
    configuration = (
        io._Configuration.model_validate_json(bundle.manifest.payload.configuration_json)
        if configuration is None
        else configuration
    )
    encoded = canonical_json_bytes(configuration.model_dump(mode="json"))
    payload = ScientificAblationLifecycleCheckpointPayloadV1(
        configuration_json=encoded.decode(),
        configuration_sha256=domain_sha256(io._CONFIG_DOMAIN, encoded),
        state=bundle.manifest.payload.state if state is None else state,
        current=current.manifest,
        best=best.manifest,
    )
    manifest = IrisScientificAblationLifecycleCheckpointManifestV1(
        sha256=domain_sha256(io._DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))),
        payload=payload,
    )
    return io.IrisScientificAblationLifecycleBundleV1(
        manifest, current.state_bytes, best.state_bytes
    )


def _assert_restore_rejected(
    candidate: io.IrisScientificAblationLifecycleBundleV1,
    original: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
) -> None:
    fresh: IrisScientificAblationLifecycleV1 = _durable(tmp_path / "fresh")
    before = native.build_scientific_ablation_checkpoint_v1(fresh._run)
    state, seal, snapshot = fresh.state, fresh._control_seal, _snapshot(fresh._run)
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(
            io,
            "_restore_scientific_ablation_checkpoint_v1",
            wraps=native._restore_scientific_ablation_checkpoint_v1,
        ) as restore,
        patch.object(fresh, "_commit", wraps=fresh._commit) as commit,
        pytest.raises(io.IrisScientificAblationLifecycleCheckpointError),
    ):
        io.restore_scientific_ablation_lifecycle_v1(candidate, controller=fresh)
    restore.assert_not_called()
    commit.assert_not_called()
    assert native.build_scientific_ablation_checkpoint_v1(fresh._run) == before
    assert _snapshot(fresh._run) == snapshot and fresh._run_snapshot == snapshot
    assert fresh.state == state and fresh._control_seal == seal
    assert fresh.best_checkpoint is None and not fresh.publication_pending
    assert not fresh.poisoned and not fresh._run._optimizer.poisoned
    assert not fresh._busy and not fresh._run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))

    # Rejection must not consume or poison the fresh owner; reuse this exact pair.
    io.restore_scientific_ablation_lifecycle_v1(original, controller=fresh)
    current, best = _components(original)
    assert fresh.state == original.manifest.payload.state and fresh.best_checkpoint == best
    assert native.build_scientific_ablation_checkpoint_v1(fresh._run) == current
    assert not fresh.poisoned and not fresh.publication_pending
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def _assert_all_preflights_reject(
    candidate: io.IrisScientificAblationLifecycleBundleV1,
    original: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
) -> None:
    caller = capture_runtime_state(torch.device("cpu"))
    with pytest.raises((ValueError, RuntimeError)):
        io._bundle_parts(candidate)
    root = tmp_path / "rejected-publication"
    with (
        patch.object(io, "_write_exact_file", wraps=_write_exact_file) as write,
        patch.object(io, "process_file_lock", wraps=process_file_lock) as lock,
        pytest.raises(io.IrisScientificAblationLifecycleCheckpointError),
    ):
        io.write_scientific_ablation_lifecycle_checkpoint_v1(candidate, root=root)
    write.assert_not_called()
    lock.assert_not_called()
    assert not root.exists()
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    _assert_restore_rejected(candidate, original, tmp_path)


@pytest.mark.parametrize("kind", ("immutable_buffer", "runtime_device"))
def test_canonical_separate_best_requires_current_configuration_buffers_and_device(
    durable_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    _, best = _components(durable_bundle)
    payload = best.manifest.payload
    tensors = load_tensors(best.state_bytes)
    updates: dict[str, Any] = {}
    if kind == "immutable_buffer":
        name = next(
            name for name in payload.model_state_names if name not in payload.parameter_names
        )
        tensors[f"model/{name}"].flatten()[0] += 1
        updates["scientific_state_sha256"] = named_module_state_sha256(
            component_names=("scientific_ablation_teacher",),
            state={
                f"scientific_ablation_teacher.{name}": tensors[f"model/{name}"]
                for name in payload.model_state_names
            },
        )
    else:
        updates["runtime"] = ScientificRuntimeV1.model_validate(
            payload.runtime.model_dump(mode="python") | {"device": "cuda:0"}
        )
        # No CUDA allocation: the canonical CUDA inventory must exist so this is
        # not rejected merely for a missing tensor before device compatibility.
        tensors["runtime/cuda"] = tensors["runtime/cpu"].clone()
    modified = _native_candidate(best, updates=updates, tensors=tensors)
    native._verify(modified)  # Native canonical/hash/model/runtime checks genuinely pass.
    candidate = _envelope(durable_bundle, best=modified)
    _assert_all_preflights_reject(candidate, durable_bundle, tmp_path)


@pytest.mark.parametrize("selected", ("current", "best"))
def test_rehashed_invalid_cpu_rng_reaches_runtime_validation_before_any_owner_write(
    durable_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    selected: Literal["current", "best"],
) -> None:
    current, best = _components(durable_bundle)
    component = current if selected == "current" else best
    tensors = load_tensors(component.state_bytes)
    tensors["runtime/cpu"] = torch.zeros(1, dtype=torch.uint8)
    modified = _native_candidate(component, tensors=tensors)
    with (
        patch.object(native, "_runtime", wraps=native._runtime) as runtime,
        pytest.raises(RuntimeError),
    ):
        native._verify(modified)
    runtime.assert_called_once()  # All preceding schema/hash/encoding checks passed.
    candidate = _envelope(
        durable_bundle,
        current=modified if selected == "current" else None,
        best=modified if selected == "best" else None,
    )
    _assert_all_preflights_reject(candidate, durable_bundle, tmp_path)


@pytest.mark.parametrize("kind", ("examples_seen", "target_tokens_seen", "parameter_steps"))
def test_standalone_valid_selected_best_cannot_be_ahead_of_current_progress(
    durable_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    current, best = _components(durable_bundle)
    if kind == "parameter_steps":
        index = next(
            index for index, count in enumerate(best.manifest.payload.parameter_steps) if count
        )
        counts = list(current.manifest.payload.parameter_steps)
        counts[index] = 0
        assert max(counts) == current.manifest.payload.completed_steps
        tensors = load_tensors(current.state_bytes)
        for suffix in ("exp_avg", "exp_avg_sq"):
            del tensors[f"adamw/{index}/{suffix}"]
        current = _native_candidate(
            current, updates={"parameter_steps": tuple(counts)}, tensors=tensors
        )
        native._verify(current)
    else:
        best = _native_candidate(best, updates={kind: getattr(current.manifest.payload, kind) + 1})
        native._verify(best)
    candidate = _envelope(durable_bundle, current=current, best=best)
    _assert_all_preflights_reject(candidate, durable_bundle, tmp_path)


@pytest.mark.parametrize(
    "kind",
    (
        "checkpoint_count",
        "validation_count",
        "last_validation_step",
        "nonimproving_validations",
        "stop_reason",
    ),
)
def test_rehashed_schema_valid_lifecycle_counters_must_follow_declared_cadence(
    durable_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    changes: dict[str, Any]
    if kind == "checkpoint_count":
        changes = {"checkpoint_count": 2}
    elif kind == "validation_count":
        changes = {"validation_count": 2, "nonimproving_validations": 1}
    elif kind == "last_validation_step":
        changes = {"last_validation_step": 2}
    elif kind == "nonimproving_validations":
        changes = {"nonimproving_validations": 1}
    else:
        changes = {"phase": "stopped", "stop_reason": "step_budget"}
    state = ScientificAblationLifecycleStateV1.model_validate(
        durable_bundle.manifest.payload.state.model_dump(mode="python") | changes
    )
    candidate = _envelope(durable_bundle, state=state)
    _assert_all_preflights_reject(candidate, durable_bundle, tmp_path)


@pytest.mark.parametrize(
    "kind",
    (
        "ablation_policy_sha256",
        "purpose",
        "validate_every_steps",
        "checkpoint_every_steps",
        "early_stopping_patience",
    ),
)
def test_rehashed_configuration_requires_native_mode_purpose_and_matching_cadence(
    durable_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    original = io._Configuration.model_validate_json(
        durable_bundle.manifest.payload.configuration_json
    )
    changes: dict[str, Any]
    if kind == "ablation_policy_sha256":
        changes = {"ablation_policy_sha256": "0" * 64}
    else:
        values: dict[str, str | int] = {
            "purpose": "scientific",
            "validate_every_steps": 2,
            "checkpoint_every_steps": 2,
            "early_stopping_patience": 1,
        }
        policy = IrisScientificAblationLifecyclePolicyV1.model_validate(
            original.policy.model_dump(mode="python") | {kind: values[kind]}
        )
        changes = {"policy": policy}
    configuration = io._Configuration.model_validate(original.model_dump(mode="python") | changes)
    candidate = _envelope(durable_bundle, configuration=configuration)
    _assert_all_preflights_reject(candidate, durable_bundle, tmp_path)


@pytest.mark.parametrize("kind", ("validation_sha256", "minimum_improvement"))
def test_canonical_candidate_with_different_controller_configuration_rejects_before_native_restore(
    durable_bundle: io.IrisScientificAblationLifecycleBundleV1,
    tmp_path: Path,
    kind: str,
) -> None:
    original = io._Configuration.model_validate_json(
        durable_bundle.manifest.payload.configuration_json
    )
    changes: dict[str, Any]
    if kind == "validation_sha256":
        _, digest = own_scientific_ablation_validation_rows_v1(
            tuple(reversed(_validation())), training_rows=_rows()[:5]
        )
        assert digest != original.validation_sha256
        changes = {"validation_sha256": digest}
    else:
        policy = IrisScientificAblationLifecyclePolicyV1.model_validate(
            original.policy.model_dump(mode="python")
            | {"minimum_improvement": original.policy.minimum_improvement + 1.0}
        )
        changes = {"policy": policy}
    configuration = io._Configuration.model_validate(original.model_dump(mode="python") | changes)
    candidate = _envelope(durable_bundle, configuration=configuration)
    io._bundle_parts(candidate)  # Valid durable content; only this recipient configuration differs.
    _assert_restore_rejected(candidate, durable_bundle, tmp_path)
