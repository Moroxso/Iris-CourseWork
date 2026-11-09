"""Offline contract tests for the packaged one-request application service."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

import iris.application.request_command as request_command
from iris.application.bootstrap import IrisRequestRuntime
from iris.application.errors import (
    IrisBootstrapError,
    IrisRequestArtifactError,
    IrisRequestCommandError,
    IrisRequestError,
)
from iris.application.request import IrisRequestExecutionOutput, IrisRequestOrchestrator
from iris.backbone import BackboneAdapter
from iris.generation import (
    DecodingStrategy,
    IrisGeneratorOutput,
    TemperatureTopPSamplingConfig,
)
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import domain_sha256

CPU = torch.device("cpu")
CPU_INFO = DeviceInfo(
    device=CPU,
    backend="cpu",
    name="Test CPU",
    total_memory_bytes=None,
    supports_fp16=False,
    supports_bf16=False,
)
STOP_TOKEN_IDS = (7, 2)
RUNTIME_SEED = 41
SAMPLING_SEED = 9_973
MAX_SEED = 2**63 - 1
RUNTIME_CONFIGURATION_JSON = '{"request_command_test":true}'
RUNTIME_CONFIGURATION_SHA256 = domain_sha256(
    "iris/configuration/v1", RUNTIME_CONFIGURATION_JSON.encode()
)
SIDECAR_STATE_SHA256 = "c" * 64


class StrictStringSubclass(str):
    """A value equal to a string but not accepted by an exact-type boundary."""


class StrictIntegerSubclass(int):
    """A value equal to an integer but not accepted by an exact-type boundary."""


class SamplingPolicySubclass(TemperatureTopPSamplingConfig):
    """A structurally valid policy that must not cross the exact DTO boundary."""


class GeneratorOutputSubclass(IrisGeneratorOutput):
    """A structurally valid result that must not cross the exact DTO boundary."""


class EosForbiddenBackbone:
    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        raise AssertionError("the command must not infer tokenizer EOS IDs")


@dataclass(frozen=True, slots=True)
class GenerateCall:
    prompt: str
    strategy: DecodingStrategy
    generator: torch.Generator | None


class SpyOrchestrator:
    def __init__(
        self,
        result: object,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[GenerateCall] = []

    @property
    def sidecar_state_sha256(self) -> str:
        return SIDECAR_STATE_SHA256

    def execute(
        self,
        prompt: str,
        *,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
    ) -> IrisRequestExecutionOutput:
        self.calls.append(
            GenerateCall(
                prompt=prompt,
                strategy=strategy,
                generator=generator,
            )
        )
        if self.failure is not None:
            raise self.failure
        return IrisRequestExecutionOutput(
            generation=cast(IrisGeneratorOutput, self.result),
            prompt_input_ids=torch.tensor([[11, 12]], dtype=torch.int64),
            prompt_attention_mask=torch.ones((1, 2), dtype=torch.int64),
        )


@pytest.fixture(autouse=True)
def stub_request_artifact_pipeline() -> Iterator[None]:
    artifact = object()
    with (
        patch.object(request_command, "build_request_artifact_v1", return_value=artifact),
        patch.object(request_command, "write_request_artifact_v1", return_value=object()),
    ):
        yield


def generation_output(text: str = "generated response") -> IrisGeneratorOutput:
    return IrisGeneratorOutput(
        text=text,
        token_ids=torch.tensor([[5, 7]], dtype=torch.int64),
        stop_reason="stop_token",
        matched_stop_token_id=7,
        workspace_trace=(),
    )


def sampling_policy() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.875)


def runtime_with(
    orchestrator: SpyOrchestrator,
    *,
    device_info: DeviceInfo = CPU_INFO,
    runtime_seed: int = RUNTIME_SEED,
) -> IrisRequestRuntime:
    return IrisRequestRuntime(
        backbone=cast(BackboneAdapter, EosForbiddenBackbone()),
        orchestrator=cast(IrisRequestOrchestrator, orchestrator),
        device_info=device_info,
        dtype=torch.float32,
        seed=runtime_seed,
        deterministic=True,
        configuration_json=RUNTIME_CONFIGURATION_JSON,
        configuration_sha256=RUNTIME_CONFIGURATION_SHA256,
        sidecar_state_sha256=SIDECAR_STATE_SHA256,
    )


def inert_config() -> DictConfig:
    config = OmegaConf.create({"request_command_test": True})
    assert isinstance(config, DictConfig)
    return config


def assert_preflight_rejected(
    *,
    message: str,
    prompt: str = "hello",
    strategy: DecodingStrategy = "greedy",
    stop_token_ids: tuple[int, ...] = STOP_TOKEN_IDS,
    sampling: TemperatureTopPSamplingConfig | None = None,
    sampling_seed: int | None = None,
    allow_download: bool = False,
) -> None:
    policy = sampling_policy() if sampling is None else sampling
    with (
        patch.object(request_command, "_compose_request_config") as compose,
        patch.object(request_command, "bootstrap_request_runtime") as bootstrap,
        patch.object(request_command, "_create_sampling_generator") as create_generator,
        pytest.raises(IrisRequestCommandError, match=message),
    ):
        request_command.generate_request(
            prompt=prompt,
            strategy=strategy,
            stop_token_ids=stop_token_ids,
            sampling=policy,
            sampling_seed=sampling_seed,
            allow_download=allow_download,
        )

    compose.assert_not_called()
    bootstrap.assert_not_called()
    create_generator.assert_not_called()


@pytest.mark.parametrize(
    "invalid_prompt",
    [
        "",
        " \t\r\n",
        cast(str, None),
        cast(str, 17),
        cast(str, StrictStringSubclass("hello")),
        "bad\ud800prompt",
    ],
)
def test_prompt_preflight_is_strict_and_runs_before_composition(
    invalid_prompt: str,
) -> None:
    assert_preflight_rejected(
        message="prompt",
        prompt=invalid_prompt,
    )


@pytest.mark.parametrize(
    "invalid_strategy",
    [
        cast(DecodingStrategy, ""),
        cast(DecodingStrategy, "beam"),
        cast(DecodingStrategy, True),
        cast(DecodingStrategy, StrictStringSubclass("greedy")),
    ],
)
def test_strategy_preflight_accepts_only_the_two_exact_literals(
    invalid_strategy: DecodingStrategy,
) -> None:
    assert_preflight_rejected(
        message="strategy",
        strategy=invalid_strategy,
    )


@pytest.mark.parametrize(
    "invalid_stop_ids",
    [
        (),
        cast(tuple[int, ...], []),
        cast(tuple[int, ...], None),
        (True,),
        (7, -1),
        cast(tuple[int, ...], (7, 2.0)),
        (7, StrictIntegerSubclass(2)),
        (7, 2, 7),
    ],
)
def test_stop_policy_is_nonempty_exact_ordered_unique_nonnegative_integers(
    invalid_stop_ids: tuple[int, ...],
) -> None:
    assert_preflight_rejected(
        message="stop_token|unique",
        stop_token_ids=invalid_stop_ids,
    )


def test_sampling_policy_requires_the_exact_frozen_dto_type() -> None:
    assert_preflight_rejected(
        message="sampling",
        sampling=cast(
            TemperatureTopPSamplingConfig,
            SamplingPolicySubclass(temperature=0.75, top_p=0.875),
        ),
    )
    assert_preflight_rejected(
        message="sampling",
        sampling=cast(TemperatureTopPSamplingConfig, object()),
    )


@pytest.mark.parametrize(
    ("field", "corrupt_value"),
    [
        ("temperature", True),
        ("temperature", 1),
        ("temperature", 0.0),
        ("temperature", -0.5),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("top_p", True),
        ("top_p", 1),
        ("top_p", 0.0),
        ("top_p", 1.0001),
        ("top_p", float("nan")),
        ("top_p", float("-inf")),
    ],
)
def test_corrupted_frozen_sampling_policy_is_revalidated_before_composition(
    field: str,
    corrupt_value: object,
) -> None:
    policy = sampling_policy()
    object.__setattr__(policy, field, corrupt_value)

    assert_preflight_rejected(
        message="sampling policy",
        sampling=policy,
    )


@pytest.mark.parametrize(
    "invalid_seed",
    [0, 1, -1, MAX_SEED, True, 1.0, "7"],
)
def test_greedy_forbids_every_non_none_sampling_seed(invalid_seed: object) -> None:
    assert_preflight_rejected(
        message="greedy.*sampling_seed",
        sampling_seed=cast(int, invalid_seed),
    )


@pytest.mark.parametrize(
    "invalid_seed",
    [None, True, -1, MAX_SEED + 1, 1.0, "7", StrictIntegerSubclass(7)],
)
def test_sampling_requires_a_strict_bounded_integer_seed(invalid_seed: object) -> None:
    assert_preflight_rejected(
        message="sampling.*seed",
        strategy="sampling",
        sampling_seed=cast(int | None, invalid_seed),
    )


@pytest.mark.parametrize("invalid_allow_download", [None, 0, 1, "false", object()])
def test_download_permission_requires_an_explicit_exact_boolean(
    invalid_allow_download: object,
) -> None:
    assert_preflight_rejected(
        message="allow_download",
        allow_download=cast(bool, invalid_allow_download),
    )


@pytest.mark.parametrize(
    ("allow_download", "expected_local_files_only"),
    [(False, True), (True, False)],
)
def test_packaged_composition_enforces_download_gate_without_process_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_download: bool,
    expected_local_files_only: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    cwd_before = Path.cwd()
    assert list(tmp_path.iterdir()) == []
    assert not GlobalHydra.instance().is_initialized()

    config = request_command._compose_request_config(allow_download=allow_download)

    assert isinstance(config, DictConfig)
    assert config.backbone.local_files_only is expected_local_files_only
    assert config.backbone.trust_remote_code is False
    assert len(config.backbone.revision) == 40
    assert len(config.backbone.tokenizer_revision) == 40
    assert config.project.name == "iris"
    assert config.cognitive.state_dim == 256
    assert config.workspace.cycles == 3
    assert Path.cwd() == cwd_before
    assert list(tmp_path.iterdir()) == []
    assert not GlobalHydra.instance().is_initialized()


@pytest.mark.parametrize(
    ("allow_download", "override"),
    [(False, "backbone.local_files_only=true"), (True, "backbone.local_files_only=false")],
)
def test_composition_uses_only_the_canonical_packaged_hydra_entrypoint(
    allow_download: bool,
    override: str,
) -> None:
    config = inert_config()
    entered = 0
    exited = 0

    @contextmanager
    def initialize_spy(*, version_base: None, config_module: str) -> Any:
        nonlocal entered, exited
        assert version_base is None
        assert config_module == "iris.configs"
        entered += 1
        try:
            yield
        finally:
            exited += 1

    with (
        patch.object(
            request_command,
            "initialize_config_module",
            side_effect=initialize_spy,
        ) as initialize,
        patch.object(request_command, "compose", return_value=config) as compose,
    ):
        actual = request_command._compose_request_config(allow_download=allow_download)

    assert actual is config
    initialize.assert_called_once_with(version_base=None, config_module="iris.configs")
    compose.assert_called_once_with(
        config_name="config",
        overrides=[override],
    )
    assert (entered, exited) == (1, 1)


def test_non_dict_packaged_composition_result_is_rejected_after_context_exit() -> None:
    events: list[str] = []

    @contextmanager
    def initialize_spy(*, version_base: None, config_module: str) -> Any:
        assert version_base is None
        assert config_module == "iris.configs"
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    with (
        patch.object(request_command, "initialize_config_module", side_effect=initialize_spy),
        patch.object(request_command, "compose", return_value=object()) as compose,
        pytest.raises(TypeError, match="DictConfig"),
    ):
        request_command._compose_request_config(allow_download=False)

    compose.assert_called_once()
    assert events == ["enter", "exit"]


def test_exact_config_stop_order_and_sampling_identity_reach_one_bootstrap() -> None:
    config = inert_config()
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator)
    stop_ids = (11, 3, 19)
    policy = sampling_policy()
    prompt = "  Preserve my whitespace.\n"

    with (
        patch.object(
            request_command,
            "_compose_request_config",
            return_value=config,
        ) as compose,
        patch.object(
            request_command,
            "bootstrap_request_runtime",
            return_value=runtime,
        ) as bootstrap,
    ):
        actual = request_command.generate_request(
            prompt=prompt,
            strategy="greedy",
            stop_token_ids=stop_ids,
            sampling=policy,
            sampling_seed=None,
            allow_download=False,
        )

    assert actual is output
    compose.assert_called_once_with(allow_download=False)
    bootstrap.assert_called_once()
    assert bootstrap.call_args.args == (config,)
    assert set(bootstrap.call_args.kwargs) == {"stop_token_ids", "sampling"}
    assert bootstrap.call_args.kwargs["stop_token_ids"] is stop_ids
    assert bootstrap.call_args.kwargs["sampling"] is policy
    assert orchestrator.calls == [
        GenerateCall(
            prompt=prompt,
            strategy="greedy",
            generator=None,
        )
    ]


def test_success_builds_and_writes_one_artifact_after_one_execution() -> None:
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator)
    policy = sampling_policy()
    artifact = object()

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        patch.object(
            request_command,
            "build_request_artifact_v1",
            return_value=artifact,
        ) as build_artifact,
        patch.object(request_command, "write_request_artifact_v1") as write_artifact,
    ):
        actual = request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=policy,
            sampling_seed=None,
            allow_download=False,
        )

    assert actual is output
    assert len(orchestrator.calls) == 1
    build_artifact.assert_called_once()
    artifact_kwargs = build_artifact.call_args.kwargs
    assert artifact_kwargs["prompt"] == "hello"
    assert artifact_kwargs["strategy"] == "greedy"
    assert artifact_kwargs["stop_token_ids"] is STOP_TOKEN_IDS
    assert artifact_kwargs["sampling"] is policy
    assert artifact_kwargs["sampling_seed"] is None
    assert artifact_kwargs["allow_download"] is False
    assert artifact_kwargs["runtime"] is runtime
    execution = artifact_kwargs["execution"]
    assert type(execution) is IrisRequestExecutionOutput
    assert execution.generation is output
    write_artifact.assert_called_once_with(artifact)


def test_artifact_write_failure_is_preserved_after_one_execution_without_retry() -> None:
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator)
    artifact = object()
    failure = IrisRequestArtifactError("disk is unavailable")

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        patch.object(request_command, "build_request_artifact_v1", return_value=artifact) as build,
        patch.object(
            request_command,
            "write_request_artifact_v1",
            side_effect=failure,
        ) as write,
        pytest.raises(IrisRequestArtifactError, match="disk is unavailable") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert captured.value is failure
    assert len(orchestrator.calls) == 1
    build.assert_called_once()
    write.assert_called_once_with(artifact)


def test_greedy_never_creates_or_passes_a_random_generator() -> None:
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        patch.object(request_command, "_create_sampling_generator") as create_generator,
    ):
        actual = request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert actual is output
    create_generator.assert_not_called()
    assert len(orchestrator.calls) == 1
    assert orchestrator.calls[0].generator is None


@pytest.mark.parametrize("sampling_seed", [0, MAX_SEED])
def test_sampling_accepts_both_seed_boundaries_and_uses_exact_state(
    sampling_seed: int,
) -> None:
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator, runtime_seed=RUNTIME_SEED)
    global_state_before = torch.get_rng_state().clone()

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
    ):
        actual = request_command.generate_request(
            prompt="sample",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=sampling_seed,
            allow_download=False,
        )

    assert actual is output
    assert len(orchestrator.calls) == 1
    generator = orchestrator.calls[0].generator
    assert isinstance(generator, torch.Generator)
    assert generator is not torch.default_generator
    assert generator.device == CPU
    expected = torch.Generator(device=CPU).manual_seed(sampling_seed)
    assert torch.equal(generator.get_state(), expected.get_state())
    assert torch.equal(torch.get_rng_state(), global_state_before)


def test_sampling_creates_a_fresh_generator_per_request_from_only_request_seed() -> None:
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator, runtime_seed=RUNTIME_SEED)
    config = inert_config()
    policy = sampling_policy()
    global_state_before = torch.get_rng_state().clone()

    with (
        patch.object(request_command, "_compose_request_config", return_value=config) as compose,
        patch.object(
            request_command,
            "bootstrap_request_runtime",
            return_value=runtime,
        ) as bootstrap,
    ):
        first = request_command.generate_request(
            prompt="sample one",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=policy,
            sampling_seed=SAMPLING_SEED,
            allow_download=False,
        )
        second = request_command.generate_request(
            prompt="sample two",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=policy,
            sampling_seed=SAMPLING_SEED,
            allow_download=False,
        )

    assert first is output
    assert second is output
    assert compose.call_count == 2
    assert bootstrap.call_count == 2
    first_generator = orchestrator.calls[0].generator
    second_generator = orchestrator.calls[1].generator
    assert isinstance(first_generator, torch.Generator)
    assert isinstance(second_generator, torch.Generator)
    assert first_generator is not second_generator
    expected = torch.Generator(device=CPU).manual_seed(SAMPLING_SEED)
    runtime_seeded = torch.Generator(device=CPU).manual_seed(RUNTIME_SEED)
    assert torch.equal(first_generator.get_state(), expected.get_state())
    assert torch.equal(second_generator.get_state(), expected.get_state())
    assert not torch.equal(first_generator.get_state(), runtime_seeded.get_state())
    assert torch.equal(torch.get_rng_state(), global_state_before)


def test_sampling_generator_uses_exact_concrete_runtime_device_and_identity() -> None:
    concrete_device = torch.device("cuda", 3)
    cuda_info = DeviceInfo(
        device=concrete_device,
        backend="cuda",
        name="Test CUDA 3",
        total_memory_bytes=8 * 1024**3,
        supports_fp16=True,
        supports_bf16=True,
    )
    output = generation_output()
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator, device_info=cuda_info)
    owned_generator = torch.Generator(device=CPU).manual_seed(SAMPLING_SEED)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        patch.object(
            request_command,
            "_create_sampling_generator",
            return_value=owned_generator,
        ) as create_generator,
        patch.object(
            request_command,
            "_validate_sampling_generator",
            return_value=owned_generator,
        ) as validate_generator,
    ):
        actual = request_command.generate_request(
            prompt="sample",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=SAMPLING_SEED,
            allow_download=False,
        )

    assert actual is output
    create_generator.assert_called_once()
    assert create_generator.call_args.args == ()
    assert create_generator.call_args.kwargs["device"] is concrete_device
    assert create_generator.call_args.kwargs["seed"] == SAMPLING_SEED
    assert set(create_generator.call_args.kwargs) == {"device", "seed"}
    validate_generator.assert_called_once_with(
        owned_generator,
        device=concrete_device,
        seed=SAMPLING_SEED,
    )
    assert orchestrator.calls[0].generator is owned_generator


@pytest.mark.parametrize(
    ("invalid_generator", "message"),
    [
        (object(), "did not return torch.Generator"),
        (torch.default_generator, "global default generator"),
        (torch.Generator(device=CPU).manual_seed(SAMPLING_SEED + 1), "explicit sampling seed"),
    ],
)
def test_sampling_generator_validation_rejects_invalid_ownership_or_state(
    invalid_generator: object,
    message: str,
) -> None:
    orchestrator = SpyOrchestrator(generation_output())
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        patch.object(
            request_command,
            "_create_sampling_generator",
            return_value=invalid_generator,
        ),
        pytest.raises(IrisRequestCommandError, match="generator") as captured,
    ):
        request_command.generate_request(
            prompt="sample",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=SAMPLING_SEED,
            allow_download=False,
        )

    assert orchestrator.calls == []
    assert captured.value.__cause__ is not None
    assert message in str(captured.value.__cause__)


def test_sampling_generator_validation_rejects_wrong_runtime_device() -> None:
    generator = torch.Generator(device=CPU).manual_seed(SAMPLING_SEED)

    with pytest.raises(ValueError, match="wrong runtime device"):
        request_command._validate_sampling_generator(
            generator,
            device=torch.device("cuda", 3),
            seed=SAMPLING_SEED,
        )


def test_exact_empty_text_output_is_returned_without_copying() -> None:
    output = generation_output(text="")
    orchestrator = SpyOrchestrator(output)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
    ):
        actual = request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert actual is output
    assert actual.text == ""


@pytest.mark.parametrize("malformed", [object(), None, "text"])
def test_non_output_result_is_rejected_after_exactly_one_generate(
    malformed: object,
) -> None:
    orchestrator = SpyOrchestrator(malformed)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        pytest.raises(IrisRequestCommandError, match="invalid text output") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert len(orchestrator.calls) == 1
    assert captured.value.__cause__ is None


def test_output_subclass_is_rejected_at_the_exact_result_boundary() -> None:
    base = generation_output()
    malformed = GeneratorOutputSubclass(
        text=base.text,
        token_ids=base.token_ids,
        stop_reason=base.stop_reason,
        matched_stop_token_id=base.matched_stop_token_id,
        workspace_trace=base.workspace_trace,
    )
    orchestrator = SpyOrchestrator(malformed)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        pytest.raises(IrisRequestCommandError, match="invalid text output"),
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert len(orchestrator.calls) == 1


def test_corrupted_frozen_output_text_is_rejected_before_return() -> None:
    malformed = generation_output()
    object.__setattr__(malformed, "text", 17)
    orchestrator = SpyOrchestrator(malformed)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        pytest.raises(IrisRequestCommandError, match="invalid text output"),
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert len(orchestrator.calls) == 1


def test_hydra_failure_is_wrapped_with_exact_cause_and_not_retried() -> None:
    failure = RuntimeError("packaged composition exploded")
    with (
        patch.object(
            request_command,
            "_compose_request_config",
            side_effect=failure,
        ) as compose,
        patch.object(request_command, "bootstrap_request_runtime") as bootstrap,
        pytest.raises(IrisRequestCommandError, match="did not compose") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    compose.assert_called_once_with(allow_download=False)
    bootstrap.assert_not_called()
    assert captured.value.__cause__ is failure


def test_typed_bootstrap_failure_is_preserved_and_not_retried() -> None:
    failure = IrisBootstrapError("offline weights unavailable")
    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(
            request_command,
            "bootstrap_request_runtime",
            side_effect=failure,
        ) as bootstrap,
        patch.object(request_command, "_create_sampling_generator") as create_generator,
        pytest.raises(IrisBootstrapError, match="offline weights unavailable") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    bootstrap.assert_called_once()
    create_generator.assert_not_called()
    assert captured.value is failure


def test_unexpected_bootstrap_failure_is_wrapped_with_exact_cause_and_not_retried() -> None:
    failure = RuntimeError("unexpected bootstrap bug")
    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(
            request_command,
            "bootstrap_request_runtime",
            side_effect=failure,
        ) as bootstrap,
        patch.object(request_command, "_create_sampling_generator") as create_generator,
        pytest.raises(IrisRequestCommandError, match="bootstrap") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    bootstrap.assert_called_once()
    create_generator.assert_not_called()
    assert captured.value.__cause__ is failure


def test_sampling_generator_failure_is_chained_without_generate_or_retry() -> None:
    failure = RuntimeError("generator allocation failed")
    orchestrator = SpyOrchestrator(generation_output())
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        patch.object(
            request_command,
            "_create_sampling_generator",
            side_effect=failure,
        ) as create_generator,
        pytest.raises(IrisRequestCommandError, match="generator") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="sampling",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=SAMPLING_SEED,
            allow_download=False,
        )

    create_generator.assert_called_once()
    assert orchestrator.calls == []
    assert captured.value.__cause__ is failure


def test_typed_orchestrator_failure_is_preserved_and_not_retried() -> None:
    failure = IrisRequestError("expected request failure")
    orchestrator = SpyOrchestrator(generation_output(), failure=failure)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        pytest.raises(IrisRequestError, match="expected request failure") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert len(orchestrator.calls) == 1
    assert captured.value is failure


def test_unexpected_orchestrator_failure_is_wrapped_with_exact_cause_and_not_retried() -> None:
    failure = RuntimeError("unexpected request bug")
    orchestrator = SpyOrchestrator(generation_output(), failure=failure)
    runtime = runtime_with(orchestrator)

    with (
        patch.object(request_command, "_compose_request_config", return_value=inert_config()),
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        pytest.raises(IrisRequestCommandError, match="did not complete") as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert len(orchestrator.calls) == 1
    assert captured.value.__cause__ is failure


def test_failed_request_creates_no_artifacts_and_leaves_hydra_and_cwd_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cwd_before = Path.cwd()
    failure = RuntimeError("generation failed")
    orchestrator = SpyOrchestrator(generation_output(), failure=failure)
    runtime = runtime_with(orchestrator)
    assert not GlobalHydra.instance().is_initialized()

    with (
        patch.object(request_command, "bootstrap_request_runtime", return_value=runtime),
        pytest.raises(IrisRequestCommandError) as captured,
    ):
        request_command.generate_request(
            prompt="hello",
            strategy="greedy",
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_policy(),
            sampling_seed=None,
            allow_download=False,
        )

    assert captured.value.__cause__ is failure
    assert Path.cwd() == cwd_before
    assert list(tmp_path.iterdir()) == []
    assert not GlobalHydra.instance().is_initialized()
