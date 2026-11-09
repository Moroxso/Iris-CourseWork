"""Coherent mode-bound inference projection, without exposing a mutable request.

A trusted lifecycle digest is required BEFORE native decode or construction.
Selected best is restored only into an explicitly supplied fresh standalone
factory result. A run seed is not inferred to be the initial graph's seed.
Offline verification requires external expected artifact/source digests.

Publication and privately owned request execution are separate R01.4 work;
this module never transfers these weights through the legacy request API.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

import torch
from pydantic import BaseModel, Field
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.application.scientific_ablation_checkpoint import (
    _RunConfiguration,
    _verify,
    restore_scientific_ablation_checkpoint_v1,
)
from iris.application.scientific_ablation_lifecycle import _validate_cadence
from iris.application.scientific_ablation_lifecycle_checkpoint import (
    IrisScientificAblationLifecycleBundleV1,
    _bundle_parts,
    _Configuration,
)
from iris.application.scientific_ablation_run import IrisScientificAblationRunV1
from iris.application.scientific_ablation_validation import _has_signal
from iris.application.scientific_checkpoint import _json
from iris.application.scientific_teacher import IrisScientificObjectivePolicyV1
from iris.artifacts.checkpoint_v1 import TeacherBackboneRuntimeV1, TeacherCheckpointModelIdentityV1
from iris.artifacts.scientific_ablation_inference_v1 import (
    IrisScientificAblationInferenceManifestV1,
    ScientificAblationInferenceGenerationPolicyV1,
    ScientificAblationInferencePayloadV1,
    ScientificAblationInferenceSelectionV1,
)
from iris.artifacts.scientific_ablation_lifecycle_checkpoint_v1 import (
    IrisScientificAblationLifecycleCheckpointManifestV1,
)
from iris.artifacts.scientific_checkpoint_v1 import Digest, ScientificTensorV1, _Strict
from iris.artifacts.scientific_inference_v1 import REQUEST_COMPONENTS_V1
from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
from iris.utils.hashing import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    named_module_state_sha256,
)
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state

_DOMAIN = "iris/scientific-ablation-inference/v1/payload"
_META = {"format": "iris.scientific-ablation-inference.v1"}
_MAX_MANIFEST = 4 * 1024**2
_MAX_ARTIFACT = 200_000_000 - 1
_TEACHER_COMPONENTS = (*REQUEST_COMPONENTS_V1[:-1], "sequence_assembler")
_IDENTITY_BUFFERS = {
    "configuration_identity",
    "base.configuration_identity",
    "base.probes.configuration_identity",
}


class IrisScientificAblationInferenceError(RuntimeError):
    """An exact mode-bound inference candidate/source/transaction was rejected."""


@dataclass(frozen=True, slots=True)
class IrisScientificAblationInferenceBundleV1:
    manifest: IrisScientificAblationInferenceManifestV1
    state_bytes: bytes


class _BaseModelConfiguration(_Strict):
    kind: Literal["iris-scientific-teacher-v1"]
    arm_id: Literal["D"]
    policy: IrisScientificObjectivePolicyV1
    probe_configuration_sha256: Digest
    backbone_configuration_json: str = Field(min_length=1, max_length=_MAX_MANIFEST)
    cognitive_configuration_json: str = Field(min_length=1, max_length=_MAX_MANIFEST)
    workspace_configuration_json: str = Field(min_length=1, max_length=_MAX_MANIFEST)
    hidden_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    vocabulary_size: int = Field(gt=0)
    max_position_embeddings: int = Field(gt=0)
    device: str = Field(pattern=r"^(cpu|cuda:[0-9]+)$")
    dtype: Literal["float32"]


class _ModelConfiguration(_Strict):
    kind: Literal["iris-scientific-ablation-teacher-v1"]
    arm_id: Literal["D"]
    protocol_role: Literal["post_main_ablation"]
    base_configuration: _BaseModelConfiguration
    base_configuration_sha256: Digest
    ablation: CognitiveAblationPolicyV1
    ablation_policy_sha256: Digest


def _digest(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Explicit external lower-case SHA256 binding required")


def _exact_tree(value: object, annotation: object, *, depth: int = 0) -> None:
    """Guard unchecked nested objects BEFORE model_dump can erase hidden fields.

    Pydantic still validates field ranges/canonical data after this structural
    gate. This is bounded local DTO validation, not an introspection sandbox.
    """
    if depth > 32:
        raise ValueError("Inference metadata depth exceeded")
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        _exact_tree(value, args[0], depth=depth + 1)
    elif origin in (Union, UnionType):
        matches = 0
        for item in args:
            try:
                _exact_tree(value, item, depth=depth + 1)
            except ValueError:
                continue
            matches += 1
        if matches != 1:
            raise ValueError("Exact metadata union variant required")
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if (
            type(value) is not annotation
            or set(vars(value)) != set(annotation.model_fields)
            or value.__pydantic_extra__ is not None
        ):
            raise ValueError("Exact complete inference metadata instance required")
        for name, field in annotation.model_fields.items():
            _exact_tree(getattr(value, name), field.annotation, depth=depth + 1)
    elif origin is tuple:
        if type(value) is not tuple or len(value) > 16384:
            raise ValueError("Bounded exact metadata tuple required")
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                _exact_tree(item, args[0], depth=depth + 1)
        else:
            if len(args) != len(value):
                raise ValueError("Exact fixed metadata tuple required")
            for item, expected in zip(value, args, strict=True):
                _exact_tree(item, expected, depth=depth + 1)
    elif origin is dict:
        if type(value) is not dict or len(value) > 16384:
            raise ValueError("Bounded exact metadata dictionary required")
        for key, item in value.items():
            _exact_tree(key, args[0], depth=depth + 1)
            _exact_tree(item, args[1], depth=depth + 1)
    elif origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            raise ValueError("Exact literal metadata value required")
    elif annotation in (str, int, float, bool, type(None)):
        if type(value) is not annotation or (type(value) is str and len(value) > _MAX_MANIFEST):
            raise ValueError("Exact bounded metadata scalar required")
    else:
        raise ValueError("Unsupported inference metadata annotation")


def _bounded_metadata(value: BaseModel, *, maximum: int) -> None:
    """Count canonical JSON bytes before constructing a complete model_dump tree.

    The exact structural gate must run first. Only one bounded scalar is encoded
    at a time, so unchecked shared strings cannot inflate a whole DTO dump before
    the total byte limit is checked. Include the persisted manifest's LF byte.
    """
    pending: list[object] = [value]
    used = 1
    while pending:
        item = pending.pop()
        if isinstance(item, BaseModel):
            record = vars(item)
            used += 2 + max(0, len(record) - 1) + len(record)
            pending.extend(record.keys())
            pending.extend(record.values())
        elif type(item) is dict:
            used += 2 + max(0, len(item) - 1) + len(item)
            pending.extend(item.keys())
            pending.extend(item.values())
        elif type(item) is tuple:
            used += 2 + max(0, len(item) - 1)
            pending.extend(item)
        else:
            used += len(canonical_json_bytes(item))
        if used > maximum:
            raise ValueError("Inference metadata byte budget exceeded before model_dump")


def _coherent_source(
    bundle: IrisScientificAblationLifecycleBundleV1, *, expected_lifecycle_sha256: str
) -> IrisScientificAblationLifecycleBundleV1:
    _digest(expected_lifecycle_sha256)
    if type(bundle) is not IrisScientificAblationLifecycleBundleV1:
        raise ValueError("Exact coherent ablation lifecycle source required")
    _exact_tree(bundle.manifest, IrisScientificAblationLifecycleCheckpointManifestV1)
    _bounded_metadata(bundle.manifest, maximum=16 * 1024**2)
    manifest = IrisScientificAblationLifecycleCheckpointManifestV1.model_validate(
        bundle.manifest.model_dump(mode="python", warnings=False)
    )
    content = canonical_json_bytes(manifest.payload.model_dump(mode="json"))
    if (
        manifest.sha256 != expected_lifecycle_sha256
        or domain_sha256("iris/scientific-ablation-lifecycle-checkpoint/v1", content)
        != expected_lifecycle_sha256
        or len(canonical_json_file_bytes(manifest.model_dump(mode="json"))) > 16 * 1024**2
        or type(bundle.current_state_bytes) is not bytes
        or not 1 <= len(bundle.current_state_bytes) <= 512 * 1024**2
    ):
        raise ValueError("Trusted source lifecycle digest/budget differs before decode")
    payload = manifest.payload
    if (
        len(bundle.current_state_bytes) != payload.current.payload.state_size_bytes
        or bytes_sha256(bundle.current_state_bytes) != payload.current.payload.state_sha256
    ):
        raise ValueError("Current source bytes differ before native decode")
    if payload.best is None or payload.best.sha256 == payload.current.sha256:
        if bundle.best_state_bytes is not None:
            raise ValueError("Unexpected separate best state encoding")
    elif (
        type(bundle.best_state_bytes) is not bytes
        or not 1 <= len(bundle.best_state_bytes) <= 512 * 1024**2
        or len(bundle.best_state_bytes) != payload.best.payload.state_size_bytes
        or bytes_sha256(bundle.best_state_bytes) != payload.best.payload.state_sha256
    ):
        raise ValueError("Selected-best source bytes differ before native decode")
    # This detached exact manifest is the ONE private selection snapshot. Its
    # full native RNG records are never copied into the public inference payload.
    return IrisScientificAblationLifecycleBundleV1(
        manifest, bundle.current_state_bytes, bundle.best_state_bytes
    )


def _configuration_bindings(payload: ScientificAblationInferencePayloadV1) -> None:
    run = _RunConfiguration.model_validate(_json(payload.source_run_configuration_json.encode()))
    model = _ModelConfiguration.model_validate(
        _json(payload.source_model_configuration_json.encode())
    )
    lifecycle = _Configuration.model_validate(
        _json(payload.selection.lifecycle_configuration_json.encode())
    )
    for text, record, digest, domain in (
        (
            payload.source_run_configuration_json,
            run,
            payload.source_run_configuration_sha256,
            "iris/scientific-ablation-run-configuration/v1",
        ),
        (
            payload.source_model_configuration_json,
            model,
            payload.source_model_configuration_sha256,
            "iris/scientific-ablation-teacher-configuration/v1",
        ),
        (
            payload.selection.lifecycle_configuration_json,
            lifecycle,
            payload.selection.lifecycle_configuration_sha256,
            "iris/scientific-ablation-lifecycle-configuration/v1",
        ),
    ):
        if (
            canonical_json_bytes(record.model_dump(mode="json")).decode() != text
            or domain_sha256(domain, text.encode()) != digest
        ):
            raise ValueError("Canonical source configuration binding differs")
    base, identity = model.base_configuration, payload.model
    for config_text, config_type, expected_config in (
        (base.backbone_configuration_json, BackboneConfig, identity.backbone),
        (base.cognitive_configuration_json, CognitiveConfig, identity.cognitive),
        (base.workspace_configuration_json, WorkspaceConfig, identity.workspace),
    ):
        configuration = config_type.model_validate_json(config_text)
        if configuration != expected_config or configuration.model_dump_json() != config_text:
            # Native backbone/cognitive/workspace snapshots use Pydantic field
            # order, unlike the canonically sorted outer configuration envelope.
            raise ValueError("Exact nested teacher/request configuration snapshot differs")
    if (
        model.ablation != run.ablation
        or model.ablation != payload.ablation
        or model.ablation_policy_sha256 != run.ablation_policy_sha256
        or run.ablation_policy_sha256 != payload.ablation_policy_sha256
        or model.base_configuration_sha256
        != domain_sha256(
            "iris/scientific-teacher-configuration/v1",
            canonical_json_bytes(base.model_dump(mode="json")),
        )
        or run.model_configuration_sha256 != payload.source_model_configuration_sha256
        or lifecycle.run_sha256 != payload.source_run_configuration_sha256
        or lifecycle.ablation_policy_sha256 != payload.ablation_policy_sha256
        or run.optimizer_policy.purpose != lifecycle.policy.purpose
        or payload.purpose != run.optimizer_policy.purpose
        or run.teacher_policy != base.policy
        or run.backbone_runtime_sha256 != payload.backbone_runtime_sha256
        or run.terminal_token_id != payload.generation.terminal_token_id
        or (
            base.hidden_size,
            base.num_hidden_layers,
            base.vocabulary_size,
            base.max_position_embeddings,
            base.device,
            base.dtype,
        )
        != (
            identity.runtime.hidden_size,
            identity.runtime.num_hidden_layers,
            identity.runtime.vocabulary_size,
            identity.runtime.max_position_embeddings,
            identity.runtime.device,
            identity.runtime.dtype,
        )
        or payload.completed_steps > run.optimizer_policy.max_optimizer_steps
        or (
            payload.purpose == "scientific"
            and (run.dataset_examples != 50000 or lifecycle.validation_examples != 5000)
        )
    ):
        raise ValueError("Inference source/purpose/policy/revisions/geometry binding differs")
    _validate_cadence(
        payload.selection.state, lifecycle.policy, run.optimizer_policy.max_optimizer_steps
    )


def _checked_manifest(
    candidate: IrisScientificAblationInferenceManifestV1,
    *,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
) -> IrisScientificAblationInferenceManifestV1:
    """Authenticate bounded metadata before allocating/reading tensor payloads."""
    for digest in (expected_sha256, expected_source_checkpoint_sha256, expected_lifecycle_sha256):
        _digest(digest)
    _exact_tree(candidate, IrisScientificAblationInferenceManifestV1)
    _bounded_metadata(candidate, maximum=_MAX_MANIFEST)
    manifest = IrisScientificAblationInferenceManifestV1.model_validate(
        candidate.model_dump(mode="python", warnings=False)
    )
    payload = manifest.payload
    content = canonical_json_file_bytes(manifest.model_dump(mode="json"))
    if (
        manifest.sha256 != expected_sha256
        or manifest.sha256
        != domain_sha256(_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json")))
        or payload.source_checkpoint_sha256 != expected_source_checkpoint_sha256
        or payload.selection.lifecycle_sha256 != expected_lifecycle_sha256
        or len(content) > _MAX_MANIFEST
    ):
        raise ValueError("External inference/source binding or canonical byte budget differs")
    _configuration_bindings(payload)  # ALL metadata checks before tensor decode.
    if len(content) + payload.state_size_bytes > _MAX_ARTIFACT:
        raise ValueError("Inference total artifact byte budget differs")
    return manifest


def _checked(
    bundle: IrisScientificAblationInferenceBundleV1,
    *,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
) -> tuple[IrisScientificAblationInferenceManifestV1, dict[str, torch.Tensor]]:
    for digest in (expected_sha256, expected_source_checkpoint_sha256, expected_lifecycle_sha256):
        _digest(digest)
    if (
        type(bundle) is not IrisScientificAblationInferenceBundleV1
        or type(bundle.state_bytes) is not bytes
        or not 1 <= len(bundle.state_bytes) <= _MAX_ARTIFACT
    ):
        raise ValueError("Exact mode-bound inference bundle required")
    manifest = _checked_manifest(
        bundle.manifest,
        expected_sha256=expected_sha256,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        expected_lifecycle_sha256=expected_lifecycle_sha256,
    )
    payload = manifest.payload
    if payload.state_size_bytes != len(bundle.state_bytes) or payload.state_sha256 != bytes_sha256(
        bundle.state_bytes
    ):
        raise ValueError("Inference state byte size/hash differs")
    tensors = load_tensors(bundle.state_bytes)
    if set(tensors) != set(payload.model.state_names):
        raise ValueError("Inference parameter inventory differs")
    for entry in payload.tensors:
        value = tensors[entry.key]
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.is_inference()
            or tuple(value.shape) != entry.shape
            or value.numel() * value.element_size() != entry.byte_length
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Inference tensor shape/dtype/finite state differs")
    if (
        save_tensors(dict(sorted(tensors.items())), metadata=_META) != bundle.state_bytes
        or named_module_state_sha256(component_names=REQUEST_COMPONENTS_V1, state=tensors)
        != payload.model.sha256
        or named_module_state_sha256(component_names=_TEACHER_COMPONENTS, state=tensors)
        != payload.source_teacher_sidecar_sha256
    ):
        raise ValueError("Inference canonical parameter/topology identity differs")
    return manifest, tensors


def verify_scientific_ablation_inference_v1(
    bundle: IrisScientificAblationInferenceBundleV1,
    *,
    expected_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_lifecycle_sha256: str,
) -> IrisScientificAblationInferenceManifestV1:
    """Defensive verified metadata; expected bindings MUST come from trusted handoff.

    Reading these expected values from this candidate or an unsigned sibling
    manifest defeats source authentication. No training owner/graph is needed.
    """
    try:
        manifest, _ = _checked(
            bundle,
            expected_sha256=expected_sha256,
            expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
            expected_lifecycle_sha256=expected_lifecycle_sha256,
        )
        return manifest
    except Exception as error:
        raise IrisScientificAblationInferenceError(
            "Mode-bound inference verification failed"
        ) from error


def build_scientific_ablation_inference_v1(
    bundle: IrisScientificAblationLifecycleBundleV1,
    *,
    expected_lifecycle_sha256: str,
    make_fresh_run: Callable[[], IrisScientificAblationRunV1],
    selection: Literal["current", "selected_best"] = "selected_best",
) -> IrisScientificAblationInferenceBundleV1:
    """Project verified current/best using a genuinely compatible fresh standalone.

    The explicit factory preserves the source's initial graph/configuration
    identity, including distinct model/run seeds. Caller runtime is snapshotted
    BEFORE invoking it. Never accepts/restores over an attached/nonfresh run.
    Factory execution is trusted caller code, not construction from file metadata.
    """
    caller: RuntimeStateSnapshot | None = None
    fresh: IrisScientificAblationRunV1 | None = None
    primary: BaseException | None = None
    result: IrisScientificAblationInferenceBundleV1 | None = None
    try:
        if (
            type(selection) is not str
            or selection not in ("current", "selected_best")
            or not callable(make_fresh_run)
        ):
            raise ValueError("Explicit selection and fresh standalone factory required")
        if torch.is_inference_mode_enabled():
            raise ValueError("Inference projection requires a non-inference construction boundary")
        private = _coherent_source(bundle, expected_lifecycle_sha256=expected_lifecycle_sha256)
        manifest, current, best = _bundle_parts(private)
        chosen = current if selection == "current" else best
        if chosen is None:
            raise ValueError("Selected-best export requires an actual validated best")
        source, tensors = _verify(chosen)
        caller = capture_runtime_state(torch.device(source.payload.runtime.device))
        candidate = make_fresh_run()
        if type(candidate) is not IrisScientificAblationRunV1 or candidate._busy:
            raise ValueError("Exact idle fresh standalone factory result required")
        candidate._authorize_operation("restore", None)
        candidate._validate_boundary()
        if candidate.completed_steps or candidate.examples_seen or candidate.target_tokens_seen:
            raise ValueError("Fresh factory cannot return a progressed source run")
        fresh = candidate
        restore_scientific_ablation_checkpoint_v1(chosen, run=fresh)
        model = fresh._optimizer._model
        teacher = model.teacher
        full_state = model.state_dict()
        if (
            tuple(full_state) != source.payload.model_state_names
            or tuple(name for name, _ in model.named_parameters()) != source.payload.parameter_names
            or set(dict(model.named_buffers())) != _IDENTITY_BUFFERS
            or tuple(teacher._modules) != _TEACHER_COMPONENTS
            or tuple(teacher.state_dict()) != tuple(name for name, _ in teacher.named_parameters())
            or any(
                not torch.equal(value.detach().cpu(), tensors[f"model/{name}"])
                for name, value in full_state.items()
            )
        ):
            raise ValueError("Full restored graph/model/buffer inventory differs before projection")
        provenance = teacher.checkpoint_provenance
        names = tuple(teacher.state_dict())
        state = {name: tensors[f"model/base.teacher.{name}"].detach().clone() for name in names}
        raw = save_tensors(dict(sorted(state.items())), metadata=_META)
        run_configuration = _RunConfiguration.model_validate(
            _json(source.payload.configuration_json.encode())
        )
        runtime = TeacherBackboneRuntimeV1(
            hidden_size=provenance.hidden_size,
            num_hidden_layers=provenance.num_hidden_layers,
            vocabulary_size=provenance.vocabulary_size,
            max_position_embeddings=provenance.max_position_embeddings,
            max_prompt_tokens=provenance.max_prompt_tokens,
            max_target_tokens=provenance.max_target_tokens,
            device=provenance.device,
            dtype="float32",
        )
        payload = ScientificAblationInferencePayloadV1(
            purpose=run_configuration.optimizer_policy.purpose,
            ablation=model.ablation,
            ablation_policy_sha256=model.ablation_policy_sha256,
            selection=ScientificAblationInferenceSelectionV1(
                kind=selection,
                lifecycle_sha256=manifest.sha256,
                lifecycle_configuration_json=manifest.payload.configuration_json,
                lifecycle_configuration_sha256=manifest.payload.configuration_sha256,
                current_checkpoint_sha256=current.manifest.sha256,
                current_completed_steps=current.manifest.payload.completed_steps,
                best_checkpoint_sha256=None if best is None else best.manifest.sha256,
                state=manifest.payload.state,
            ),
            source_checkpoint_sha256=source.sha256,
            source_run_configuration_json=source.payload.configuration_json,
            source_run_configuration_sha256=source.payload.configuration_sha256,
            source_model_configuration_json=model.configuration_json,
            source_model_configuration_sha256=model.configuration_sha256,
            source_teacher_sidecar_sha256=provenance.sidecar_state_sha256,
            backbone_runtime_sha256=run_configuration.backbone_runtime_sha256,
            completed_steps=source.payload.completed_steps,
            generation=ScientificAblationInferenceGenerationPolicyV1(
                terminal_token_id=run_configuration.terminal_token_id
            ),
            model=TeacherCheckpointModelIdentityV1(
                sha256=named_module_state_sha256(
                    component_names=REQUEST_COMPONENTS_V1, state=state
                ),
                backbone=BackboneConfig.model_validate_json(provenance.backbone_configuration_json),
                cognitive=CognitiveConfig.model_validate_json(
                    provenance.cognitive_configuration_json
                ),
                workspace=WorkspaceConfig.model_validate_json(
                    provenance.workspace_configuration_json
                ),
                runtime=runtime,
                state_names=names,
            ),
            tensors=tuple(
                ScientificTensorV1(
                    key=name,
                    shape=tuple(value.shape),
                    dtype="float32",
                    byte_length=value.numel() * value.element_size(),
                )
                for name, value in sorted(state.items())
            ),
            state_sha256=bytes_sha256(raw),
            state_size_bytes=len(raw),
        )
        result = IrisScientificAblationInferenceBundleV1(
            IrisScientificAblationInferenceManifestV1(
                sha256=domain_sha256(
                    _DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
                ),
                payload=payload,
            ),
            raw,
        )
        _checked(
            result,
            expected_sha256=result.manifest.sha256,
            expected_source_checkpoint_sha256=source.sha256,
            expected_lifecycle_sha256=manifest.sha256,
        )
        fresh._validate_boundary()
    except BaseException as error:
        primary = error
    finally:
        if caller is not None:
            try:
                restore_runtime_state(caller)
            except BaseException as cleanup:
                if fresh is not None:
                    fresh._poisoned = fresh._optimizer._poisoned = True
                if primary is not None and _has_signal(primary):
                    raise primary from cleanup
                if _has_signal(cleanup):
                    raise
                cause = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup(
                        "Export and caller restoration failed", [primary, cleanup]
                    )
                )
                raise IrisScientificAblationInferenceError(
                    "Caller runtime restoration failed; export withheld"
                ) from cause
    if primary is not None:
        if _has_signal(primary):
            raise primary
        raise IrisScientificAblationInferenceError(
            "Mode-bound inference projection failed"
        ) from primary
    if result is None:
        raise IrisScientificAblationInferenceError("No verified inference projection produced")
    return result


__all__ = [
    "IrisScientificAblationInferenceBundleV1",
    "IrisScientificAblationInferenceError",
    "build_scientific_ablation_inference_v1",
    "verify_scientific_ablation_inference_v1",
]
