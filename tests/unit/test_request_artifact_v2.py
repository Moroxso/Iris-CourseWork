"""Builder and durable-integrity tests for arm-labelled request artifacts v2."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import PropertyMock, patch

import pytest
import torch

import iris.application.request_artifact as request_artifact
import iris.application.request_artifact_v2 as request_artifact_v2
from iris.application.arm_profiles import (
    IrisArmExecutionProfileV3,
    get_arm_execution_profile_v3,
)
from iris.application.base_request import BaseLlmRequestOrchestrator
from iris.application.bootstrap import IrisArmRequestRuntimeV2
from iris.application.errors import IrisRequestArtifactError
from iris.application.no_workspace_request import CognitiveNoWorkspaceRequestOrchestrator
from iris.application.request import (
    IrisRequestExecutionOutput,
    IrisRequestExecutionReceiptV1,
    IrisRequestOrchestrator,
)
from iris.application.request_artifact_v2 import (
    build_request_artifact_v2,
    read_request_artifact_v2,
    write_request_artifact_v2,
)
from iris.artifacts.request_v2 import IrisRequestArtifactEnvelopeV2
from iris.backbone import BackboneAdapter
from iris.generation import IrisGeneratorOutput, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace import WorkspaceState
from tests.unit.test_request_artifact_schema import valid_bootstrap_config

CPU = torch.device("cpu")
STOP_TOKEN_IDS = (2, 7)
SIDECAR_SHA256 = "c" * 64
SECRET = "IRIS_V2_SECRET_d877_Привет\r\n"
PROFILE_SHA256: Literal["5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"] = (
    "5099bc5a69ec10ddcf73bda5f2f85a1ccd74c09f2f0d2458b3f6c0eda982ab73"
)


def _config(output_dir: Path) -> IrisBootstrapConfig:
    raw = valid_bootstrap_config().model_dump(mode="python")
    cast(dict[str, object], raw["runtime"])["output_dir"] = str(output_dir)
    return IrisBootstrapConfig.model_validate(raw)


def _workspace_state(cycle: int) -> WorkspaceState:
    slots = torch.zeros((1, 8, 256), dtype=torch.float32)
    age = torch.zeros((1, 8), dtype=torch.int64)
    occupancy = torch.zeros((1, 8), dtype=torch.bool)
    source_module = torch.full((1, 8), -1, dtype=torch.int64)
    confidence = torch.zeros((1, 8), dtype=torch.float32)
    ttl = torch.zeros((1, 8), dtype=torch.float32)
    write_count = torch.zeros((1, 8), dtype=torch.int64)
    slots[0, 0, 0] = float(cycle)
    occupancy[0, 0] = True
    source_module[0, 0] = cycle - 1
    confidence[0, 0] = 0.75
    ttl[0, 0] = 3.0
    write_count[0, 0] = cycle
    return WorkspaceState(
        slots=slots,
        age=age,
        occupancy=occupancy,
        source_module=source_module,
        confidence=confidence,
        ttl=ttl,
        write_count=write_count,
    )


def _sidecar_patch(
    arm_id: str,
    *,
    current_sha256: str = SIDECAR_SHA256,
) -> AbstractContextManager[object]:
    if arm_id == "B":
        return patch.object(
            CognitiveNoWorkspaceRequestOrchestrator,
            "sidecar_state_sha256",
            new_callable=PropertyMock,
            return_value=current_sha256,
        )
    if arm_id == "D":
        return patch.object(
            IrisRequestOrchestrator,
            "sidecar_state_sha256",
            new_callable=PropertyMock,
            return_value=current_sha256,
        )
    return nullcontext()


def _inputs(
    tmp_path: Path,
    arm_id: str,
    *,
    trace: tuple[WorkspaceState, ...] | None = None,
) -> tuple[IrisArmRequestRuntimeV2, IrisRequestExecutionOutput, dict[str, object]]:
    config = _config(tmp_path / "artifacts")
    configuration_json = canonical_json_bytes(config.model_dump(mode="json")).decode()
    backbone = cast(BackboneAdapter, object())
    orchestrator: (
        BaseLlmRequestOrchestrator
        | CognitiveNoWorkspaceRequestOrchestrator
        | IrisRequestOrchestrator
    )
    if arm_id == "A":
        orchestrator = object.__new__(BaseLlmRequestOrchestrator)
    elif arm_id == "B":
        orchestrator = object.__new__(CognitiveNoWorkspaceRequestOrchestrator)
    else:
        orchestrator = object.__new__(IrisRequestOrchestrator)
    orchestrator._backbone = backbone
    sidecar = None if arm_id == "A" else SIDECAR_SHA256
    runtime = IrisArmRequestRuntimeV2(
        profile=get_arm_execution_profile_v3(arm_id),
        execution_profile_set_sha256=PROFILE_SHA256,
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=DeviceInfo(
            device=CPU,
            backend="cpu",
            name="Request Artifact V2 Test CPU",
            total_memory_bytes=None,
            supports_fp16=False,
            supports_bf16=False,
        ),
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        configuration_json=configuration_json,
        configuration_sha256=domain_sha256(
            "iris/configuration/v1",
            configuration_json.encode(),
        ),
        sidecar_state_sha256=sidecar,
    )
    if trace is None:
        trace = (
            (_workspace_state(1), _workspace_state(2), _workspace_state(3)) if arm_id == "D" else ()
        )
    execution = IrisRequestExecutionOutput(
        generation=IrisGeneratorOutput(
            text=SECRET,
            token_ids=torch.tensor([[5, 2]], dtype=torch.int64),
            stop_reason="stop_token",
            matched_stop_token_id=2,
            workspace_trace=trace,
        ),
        prompt_input_ids=torch.tensor([[11, 12]], dtype=torch.int64),
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
        receipt=IrisRequestExecutionReceiptV1(
            prompt_sha256=domain_sha256(
                "iris/request-execution/v1/prompt",
                SECRET.encode(),
            ),
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875),
            sampling_seed=None,
        ),
    )
    kwargs: dict[str, object] = {
        "prompt": SECRET,
        "strategy": "greedy",
        "stop_token_ids": STOP_TOKEN_IDS,
        "sampling": TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875),
        "sampling_seed": None,
        "allow_download": False,
        "runtime": runtime,
        "execution": execution,
    }
    return runtime, execution, kwargs


def _build(
    tmp_path: Path,
    arm_id: str,
    *,
    trace: tuple[WorkspaceState, ...] | None = None,
    current_sidecar_sha256: str = SIDECAR_SHA256,
) -> IrisRequestArtifactEnvelopeV2:
    _, _, kwargs = _inputs(tmp_path, arm_id, trace=trace)
    with _sidecar_patch(arm_id, current_sha256=current_sidecar_sha256):
        return build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arm_id", "arm_kind", "request_graph", "cycles"),
    [
        ("A", "base_llm", "frozen_backbone_prompt_only_v1", 0),
        (
            "B",
            "llm_modules_no_workspace",
            "independent_recurrent_modules_zero_memory_read_v1",
            0,
        ),
        (
            "D",
            "llm_modules_limited_competitive_workspace",
            "limited_competitive_workspace_v1",
            3,
        ),
    ],
)
def test_builder_records_distinct_exact_arm_topologies(
    tmp_path: Path,
    arm_id: str,
    arm_kind: str,
    request_graph: str,
    cycles: int,
) -> None:
    artifact = _build(tmp_path, arm_id)

    assert artifact.payload.topology.arm_id == arm_id
    assert artifact.payload.topology.arm_kind == arm_kind
    assert artifact.payload.topology.request_graph == request_graph
    assert artifact.payload.workspace_trace.cycle_count == cycles
    assert artifact.payload.execution_profile_set_sha256 == PROFILE_SHA256
    assert artifact.payload.protocol_declaration_payload_sha256.startswith("b7b25609")
    if arm_id == "A":
        assert artifact.payload.runtime.sidecar_state_sha256 is None
        assert artifact.payload.topology.latent_token_count == 0
    elif arm_id == "B":
        assert artifact.payload.topology.zero_memory_read_cycles == 3
        assert artifact.payload.topology.latent_token_count == 8
    else:
        assert artifact.payload.topology.latent_token_count == 8

    raw = canonical_json_bytes(artifact.model_dump(mode="json", by_alias=True))
    assert SECRET.encode() not in raw
    assert repr(SECRET).encode() not in raw


@pytest.mark.parametrize("arm_id", ["A", "B", "D"])
def test_builder_is_deterministic_for_each_arm(tmp_path: Path, arm_id: str) -> None:
    assert _build(tmp_path, arm_id) == _build(tmp_path, arm_id)


def test_a_and_b_share_only_the_empty_trace_hash_not_the_payload_identity(tmp_path: Path) -> None:
    arm_a = _build(tmp_path, "A")
    arm_b = _build(tmp_path, "B")

    assert arm_a.payload.workspace_trace.sha256 == arm_b.payload.workspace_trace.sha256
    assert arm_a.payload.topology.arm_id == "A"
    assert arm_b.payload.topology.arm_id == "B"
    assert arm_a.payload_sha256 != arm_b.payload_sha256


def test_arm_d_cycle_change_updates_nested_trace_and_payload_hashes(tmp_path: Path) -> None:
    original_trace = (_workspace_state(1), _workspace_state(2), _workspace_state(3))
    changed_trace = (_workspace_state(1), _workspace_state(2), _workspace_state(3))
    changed_trace[1].slots[0, 0, 1] = 0.25

    original = _build(tmp_path, "D", trace=original_trace)
    changed = _build(tmp_path, "D", trace=changed_trace)

    assert original.payload.workspace_trace.cycles[0] == changed.payload.workspace_trace.cycles[0]
    assert (
        original.payload.workspace_trace.cycles[1].slots.sha256
        != changed.payload.workspace_trace.cycles[1].slots.sha256
    )
    assert original.payload.workspace_trace.sha256 != changed.payload.workspace_trace.sha256
    assert original.payload_sha256 != changed.payload_sha256


@pytest.mark.parametrize("arm_id", ["A", "B"])
def test_empty_trace_arms_reject_workspace_state(
    tmp_path: Path,
    arm_id: str,
) -> None:
    with pytest.raises(IrisRequestArtifactError, match="could not be built"):
        _build(tmp_path, arm_id, trace=(_workspace_state(1),))


def test_arm_d_rejects_empty_trace(tmp_path: Path) -> None:
    with pytest.raises(IrisRequestArtifactError, match="could not be built"):
        _build(tmp_path, "D", trace=())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("confidence", 2.0, "confidence must lie in"),
        ("confidence", -0.25, "confidence must lie in"),
        ("ttl", -1.0, "TTL cannot be negative"),
    ],
)
def test_arm_d_rejects_invalid_occupied_workspace_metadata(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    trace = tuple(_workspace_state(cycle) for cycle in range(1, 4))
    replacement = getattr(trace[0], field).clone()
    replacement[0, 0] = value
    invalid_first = replace(trace[0], **{field: replacement})

    with pytest.raises(IrisRequestArtifactError, match="could not be built") as raised:
        _build(tmp_path, "D", trace=(invalid_first, *trace[1:]))
    assert message in str(raised.value.__cause__)


@pytest.mark.parametrize("arm_id", ["B", "D"])
def test_builder_rejects_sidecar_mutation_after_execution(tmp_path: Path, arm_id: str) -> None:
    with pytest.raises(IrisRequestArtifactError, match="could not be built"):
        _build(tmp_path, arm_id, current_sidecar_sha256="d" * 64)


def test_builder_rejects_cross_arm_profile_and_orchestrator(tmp_path: Path) -> None:
    runtime, execution, kwargs = _inputs(tmp_path, "B")
    kwargs["runtime"] = replace(runtime, profile=get_arm_execution_profile_v3("D"))

    with (
        _sidecar_patch("B"),
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]
    assert execution.generation.workspace_trace == ()


@pytest.mark.parametrize(
    ("runtime_arm", "profile_arm", "orchestrator_type"),
    [
        ("B", "D", CognitiveNoWorkspaceRequestOrchestrator),
        ("D", "B", IrisRequestOrchestrator),
    ],
)
def test_builder_rejects_wrong_arm_class_before_inspecting_graph_properties(
    tmp_path: Path,
    runtime_arm: Literal["B", "D"],
    profile_arm: Literal["B", "D"],
    orchestrator_type: type[object],
) -> None:
    runtime, _, kwargs = _inputs(tmp_path, runtime_arm)
    kwargs["runtime"] = replace(
        runtime,
        profile=get_arm_execution_profile_v3(profile_arm),
    )

    with (
        patch.object(orchestrator_type, "backbone", new_callable=PropertyMock) as backbone,
        patch.object(
            orchestrator_type,
            "sidecar_state_sha256",
            new_callable=PropertyMock,
        ) as sidecar,
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]

    backbone.assert_not_called()
    sidecar.assert_not_called()


def test_builder_rejects_profile_impostor_before_reading_arm_or_graph(tmp_path: Path) -> None:
    class ProfileImpostor:
        @property
        def arm_id(self) -> str:
            raise AssertionError("profile impostor was inspected")

    runtime, _, kwargs = _inputs(tmp_path, "A")
    kwargs["runtime"] = replace(
        runtime,
        profile=cast(IrisArmExecutionProfileV3, ProfileImpostor()),
    )
    with (
        patch.object(
            BaseLlmRequestOrchestrator,
            "backbone",
            new_callable=PropertyMock,
        ) as backbone,
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]

    backbone.assert_not_called()


def test_builder_rejects_sampling_subclass_before_runtime_inspection(tmp_path: Path) -> None:
    class SamplingSubclass(TemperatureTopPSamplingConfig):
        pass

    _, _, kwargs = _inputs(tmp_path, "A")
    kwargs["sampling"] = SamplingSubclass(temperature=0.75, top_p=0.875)
    with (
        patch.object(
            BaseLlmRequestOrchestrator,
            "backbone",
            new_callable=PropertyMock,
        ) as backbone,
        pytest.raises(IrisRequestArtifactError, match="could not be built"),
    ):
        build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]

    backbone.assert_not_called()


def test_builder_requires_execution_receipt_v2(tmp_path: Path) -> None:
    _, execution, kwargs = _inputs(tmp_path, "A")
    kwargs["execution"] = replace(execution, receipt=None)

    with pytest.raises(IrisRequestArtifactError, match="could not be built") as raised:
        build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]

    assert "requires an exact execution receipt" in str(raised.value.__cause__)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("prompt", "prompt differs"),
        ("strategy", "strategy differs"),
        ("stop_tokens", "stop-token policy differs"),
        ("sampling", "sampling policy differs"),
        ("sampling_seed", "sampling seed differs"),
    ],
)
def test_builder_rejects_delivery_policy_that_differs_from_execution_receipt(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    _, _, kwargs = _inputs(tmp_path, "A")
    if case == "prompt":
        kwargs["prompt"] = "different prompt"
    elif case == "strategy":
        kwargs["strategy"] = "sampling"
    elif case == "stop_tokens":
        kwargs["stop_token_ids"] = (2,)
    elif case == "sampling":
        kwargs["sampling"] = TemperatureTopPSamplingConfig(temperature=0.5, top_p=0.9)
    else:
        kwargs["sampling_seed"] = 17

    with pytest.raises(IrisRequestArtifactError, match="could not be built") as raised:
        build_request_artifact_v2(**kwargs)  # type: ignore[arg-type]

    assert message in str(raised.value.__cause__)


@pytest.mark.parametrize("arm_id", ["A", "B", "D"])
def test_writer_reader_round_trip_is_content_addressed_and_reusable(
    tmp_path: Path,
    arm_id: str,
) -> None:
    artifact = _build(tmp_path, arm_id)

    first = write_request_artifact_v2(artifact)
    second = write_request_artifact_v2(artifact)

    assert first.path == second.path
    assert first.path.parent.name == "v2"
    assert first.path.parent.parent.name == "requests"
    assert first.path.stem == artifact.payload_sha256
    assert first.reused_existing is False
    assert second.reused_existing is True
    assert read_request_artifact_v2(first.path) == artifact
    content = first.path.read_bytes()
    assert content.endswith(b"\n")
    assert SECRET.encode() not in content


def test_reader_rejects_tampered_payload_and_noncanonical_filename_content(
    tmp_path: Path,
) -> None:
    artifact = _build(tmp_path, "A")
    result = write_request_artifact_v2(artifact)
    content = result.path.read_bytes()
    result.path.write_bytes(content.replace(b'"generated_tokens":2', b'"generated_tokens":1'))

    with pytest.raises(IrisRequestArtifactError, match="could not be read"):
        read_request_artifact_v2(result.path)


@pytest.mark.parametrize(
    "transform",
    [
        lambda content: b"\xef\xbb\xbf" + content,
        lambda content: content + b"\n",
        lambda content: content.replace(b'"schema":{', b'"schema":{"version":2,', 1),
    ],
)
def test_reader_rejects_bom_extra_lf_and_duplicate_keys(
    tmp_path: Path,
    transform: Callable[[bytes], bytes],
) -> None:
    artifact = _build(tmp_path, "A")
    result = write_request_artifact_v2(artifact)
    result.path.write_bytes(transform(result.path.read_bytes()))

    with pytest.raises(IrisRequestArtifactError, match="could not be read"):
        read_request_artifact_v2(result.path)


def test_writer_never_clobbers_conflicting_content_at_the_digest_path(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "B")
    path = (
        Path(artifact.payload.configuration.value.runtime.output_dir)
        / "requests"
        / "v2"
        / f"{artifact.payload_sha256}.json"
    )
    path.parent.mkdir(parents=True)
    conflict = b"private conflicting bytes"
    path.write_bytes(conflict)

    with pytest.raises(IrisRequestArtifactError, match="conflicts"):
        write_request_artifact_v2(artifact)

    assert path.read_bytes() == conflict


def test_writer_rejects_symlink_target_even_when_bytes_match(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "A")
    target = (
        Path(artifact.payload.configuration.value.runtime.output_dir)
        / "requests"
        / "v2"
        / f"{artifact.payload_sha256}.json"
    )
    target.parent.mkdir(parents=True)
    _, content = request_artifact_v2._validate_integrity(artifact)
    external = tmp_path / "external-identical.json"
    external.write_bytes(content)
    try:
        target.symlink_to(external)
    except OSError:
        pytest.skip("file symlinks are unavailable in this environment")

    with pytest.raises(IrisRequestArtifactError, match="conflicts"):
        write_request_artifact_v2(artifact)

    assert target.is_symlink()
    assert external.read_bytes() == content


def test_writer_fail_closed_branch_rejects_simulated_linked_target(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "A")
    target = (
        Path(artifact.payload.configuration.value.runtime.output_dir)
        / "requests"
        / "v2"
        / f"{artifact.payload_sha256}.json"
    )
    target.parent.mkdir(parents=True)
    _, content = request_artifact_v2._validate_integrity(artifact)
    target.write_bytes(content)

    with (
        patch.object(
            request_artifact,
            "_path_is_link_or_reparse",
            side_effect=lambda candidate: candidate == target,
        ),
        pytest.raises(IrisRequestArtifactError, match="conflicts"),
    ):
        write_request_artifact_v2(artifact)

    assert target.read_bytes() == content


def test_writer_rejects_symlink_in_managed_directory_hierarchy(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "B")
    output_root = Path(artifact.payload.configuration.value.runtime.output_dir)
    output_root.mkdir(parents=True)
    external_requests = tmp_path / "external-requests"
    external_requests.mkdir()
    requests_link = output_root / "requests"
    try:
        requests_link.symlink_to(external_requests, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")

    with pytest.raises(IrisRequestArtifactError, match="real directory"):
        write_request_artifact_v2(artifact)

    assert requests_link.is_symlink()
    assert tuple(external_requests.iterdir()) == ()


def test_writer_fail_closed_branch_rejects_simulated_reparse_parent(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "B")
    output_root = Path(artifact.payload.configuration.value.runtime.output_dir)
    requests_directory = output_root / "requests"

    with (
        patch.object(
            request_artifact,
            "_path_is_link_or_reparse",
            side_effect=lambda candidate: candidate == requests_directory,
        ),
        pytest.raises(IrisRequestArtifactError, match="real directory"),
    ):
        write_request_artifact_v2(artifact)

    assert not (requests_directory / "v2").exists()
    assert list(output_root.rglob("*.json")) == []


def test_writer_rejects_an_in_memory_payload_digest_mutation(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "A")
    tampered = artifact.model_copy(update={"payload_sha256": "0" * 64})

    with pytest.raises(IrisRequestArtifactError, match="could not be persisted"):
        write_request_artifact_v2(tampered)


def test_writer_uses_owned_validated_snapshot_for_path_and_result(tmp_path: Path) -> None:
    artifact = _build(tmp_path, "A")
    original_root = Path(artifact.payload.configuration.value.runtime.output_dir)
    redirected_root = tmp_path / "redirected-after-validation"
    validate = request_artifact_v2._validate_integrity

    def validate_then_mutate(
        candidate: IrisRequestArtifactEnvelopeV2,
    ) -> tuple[IrisRequestArtifactEnvelopeV2, bytes]:
        owned, content = validate(candidate)
        candidate.payload.configuration.value.runtime.output_dir = str(redirected_root)
        return owned, content

    with patch.object(
        request_artifact_v2,
        "_validate_integrity",
        side_effect=validate_then_mutate,
    ):
        result = write_request_artifact_v2(artifact)

    assert result.path.parent.parent.parent == original_root
    assert result.payload_sha256 == result.path.stem
    assert not redirected_root.exists()
