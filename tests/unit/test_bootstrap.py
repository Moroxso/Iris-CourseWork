"""Transactional contract tests for Iris request-runtime bootstrap."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

import iris.application as application
import iris.application.bootstrap as bootstrap
from iris.application import (
    IrisBootstrapError,
    IrisRequestOrchestrator,
    IrisRequestRuntime,
    bootstrap_request_runtime,
)
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig, RuntimeConfig
from iris.utils.devices import DeviceInfo
from iris.utils.hashing import domain_sha256
from iris.utils.seeds import (
    RuntimeStateSnapshot,
    apply_runtime_policy,
    capture_runtime_state,
    restore_runtime_state,
)

CPU = torch.device("cpu")
CUDA_2 = torch.device("cuda", 2)
STOP_TOKEN_IDS = (2, 7)
VOCABULARY_SIZE = 17
MAX_PROMPT_TOKENS = 256
MAX_NEW_TOKENS = 32
SIDECAR_STATE_SHA256 = "c" * 64

CPU_INFO = DeviceInfo(
    device=CPU,
    backend="cpu",
    name="Test CPU",
    total_memory_bytes=None,
    supports_fp16=False,
    supports_bf16=False,
)
CUDA_INFO = DeviceInfo(
    device=CUDA_2,
    backend="cuda",
    name="Test CUDA 2",
    total_memory_bytes=8 * 1024**3,
    supports_fp16=True,
    supports_bf16=True,
)


class OfflineBootstrapBackbone(BackboneAdapter):
    """Already-loaded adapter double exposing bootstrap's public boundary."""

    def __init__(
        self,
        *,
        device: torch.device = CPU,
        dtype: torch.dtype = torch.float32,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        vocabulary_size: int = VOCABULARY_SIZE,
        trainable_parameter_count: int = 0,
        has_parameter_gradients: bool = False,
    ) -> None:
        self._reported_device = device
        self._reported_dtype = dtype
        self._reported_max_prompt_tokens = max_prompt_tokens
        self._reported_max_new_tokens = max_new_tokens
        self._reported_vocabulary_size = vocabulary_size
        self._reported_trainable_parameter_count = trainable_parameter_count
        self._reported_has_parameter_gradients = has_parameter_gradients
        self.eos_reads = 0

    @property
    def device(self) -> torch.device:
        return self._reported_device

    @property
    def dtype(self) -> torch.dtype:
        return self._reported_dtype

    @property
    def max_prompt_tokens(self) -> int:
        return self._reported_max_prompt_tokens

    @property
    def max_new_tokens(self) -> int:
        return self._reported_max_new_tokens

    @property
    def vocabulary_size(self) -> int:
        return self._reported_vocabulary_size

    @property
    def trainable_parameter_count(self) -> int:
        return self._reported_trainable_parameter_count

    @property
    def has_parameter_gradients(self) -> bool:
        return self._reported_has_parameter_gradients

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        self.eos_reads += 1
        raise AssertionError("bootstrap must not infer an implicit EOS stop policy")


@pytest.fixture(autouse=True)
def preserve_process_runtime_state() -> Iterator[None]:
    """Keep bootstrap tests isolated despite deliberate global-state commits."""
    snapshot = capture_runtime_state(CPU)
    try:
        yield
    finally:
        restore_runtime_state(snapshot)


def valid_bootstrap_data(
    *,
    device: str = "cpu",
    precision: str = "fp32",
    seed: int = 1729,
    deterministic: bool = True,
) -> dict[str, object]:
    return {
        "project": {"name": "iris", "version": "0.1.0"},
        "hardware": {
            "name": f"{device}_test",
            "device": device,
            "precision": precision,
            "max_memory_gb": 8.0,
            "allow_gpu": device == "cuda",
        },
        "backbone": {
            "name": "qwen3_0_6b",
            "repository": "Qwen/Qwen3-0.6B",
            "revision": "a" * 40,
            "tokenizer_revision": "a" * 40,
            "trust_remote_code": False,
            "local_files_only": True,
            "output_hidden_states": True,
            "use_cache": False,
            "dtype": precision,
            "tokenizer": {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
            },
            "limits": {
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        },
        "cognitive": {
            "name": "tiny_bootstrap_cognitive",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": [0.25, 0.5, 0.75, 1.0]},
        },
        "workspace": {
            "name": "tiny_bootstrap_workspace",
            "num_slots": 2,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.5,
            "empty_threshold": 0.01,
            "arbiter": {
                "confidence_weight": 0.25,
                "novelty_weight": 0.1,
                "relevance_weight": 0.25,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        },
        "runtime": {
            "seed": seed,
            "deterministic": deterministic,
            "log_level": "INFO",
            "output_dir": "artifacts/runs",
        },
    }


def validated_config(**kwargs: Any) -> IrisBootstrapConfig:
    return IrisBootstrapConfig.model_validate(valid_bootstrap_data(**kwargs))


def sampling_config() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)


class OfflineBootstrapOrchestrator(IrisRequestOrchestrator):
    @property
    def sidecar_state_sha256(self) -> str:
        return SIDECAR_STATE_SHA256


def orchestrator_double() -> IrisRequestOrchestrator:
    return object.__new__(OfflineBootstrapOrchestrator)


def assert_current_state_matches(snapshot: RuntimeStateSnapshot) -> None:
    assert random.getstate() == snapshot.python_rng_state
    assert torch.equal(torch.get_rng_state(), snapshot.cpu_rng_state)
    assert torch.are_deterministic_algorithms_enabled() is snapshot.deterministic_algorithms
    assert torch.is_deterministic_algorithms_warn_only_enabled() is snapshot.deterministic_warn_only
    assert torch.backends.cudnn.deterministic is snapshot.cudnn_deterministic
    assert torch.backends.cudnn.benchmark is snapshot.cudnn_benchmark


def test_real_hydra_composition_is_owned_and_bootstrapped_without_mutation() -> None:
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        config = compose(config_name="config")
    assert isinstance(config, DictConfig)
    assert len(config.backbone.revision) == 40
    assert config.backbone.revision == config.backbone.revision.lower()
    assert len(config.backbone.tokenizer_revision) == 40
    assert config.backbone.tokenizer_revision == config.backbone.tokenizer_revision.lower()
    config_snapshot = OmegaConf.to_yaml(config, resolve=False, sort_keys=False)
    backbone = OfflineBootstrapBackbone()
    orchestrator = orchestrator_double()
    sampling = sampling_config()

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO) as inspect,
        patch.object(bootstrap, "_create_backbone", return_value=backbone) as load,
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            return_value=orchestrator,
        ) as build,
    ):
        runtime = bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling,
        )

    assert runtime.backbone is backbone
    assert runtime.orchestrator is orchestrator
    assert runtime.sidecar_state_sha256 == SIDECAR_STATE_SHA256
    assert OmegaConf.to_yaml(config, resolve=False, sort_keys=False) == config_snapshot
    inspect.assert_called_once_with("cpu")
    load.assert_called_once()
    build.assert_called_once()


def test_validated_dto_is_deeply_owned_and_exact_policies_are_forwarded_once() -> None:
    config = validated_config()
    config_snapshot = config.model_dump_json()
    backbone = OfflineBootstrapBackbone()
    orchestrator = orchestrator_double()
    sampling = sampling_config()
    observed: dict[str, object] = {}

    def load_backbone(config_value: object, device_info: object) -> BackboneAdapter:
        observed["backbone_config"] = config_value
        observed["device_info"] = device_info
        return backbone

    def build_graph(**kwargs: object) -> IrisRequestOrchestrator:
        observed.update(kwargs)
        return orchestrator

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO) as inspect,
        patch.object(bootstrap, "_create_backbone", side_effect=load_backbone) as load,
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            side_effect=build_graph,
        ) as build,
    ):
        runtime = bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling,
        )

    assert runtime == IrisRequestRuntime(
        backbone=backbone,
        orchestrator=orchestrator,
        device_info=CPU_INFO,
        dtype=torch.float32,
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        configuration_json=bootstrap._canonical_config_json(config),
        configuration_sha256=domain_sha256(
            "iris/configuration/v1", bootstrap._canonical_config_json(config).encode()
        ),
        sidecar_state_sha256=SIDECAR_STATE_SHA256,
    )
    assert observed["backbone_config"] is not config.backbone
    owned_backbone = cast(Any, observed["backbone_config"])
    assert owned_backbone.tokenizer is not config.backbone.tokenizer
    assert owned_backbone.limits is not config.backbone.limits
    assert observed["cognitive_config"] is not config.cognitive
    owned_cognitive = cast(Any, observed["cognitive_config"])
    assert owned_cognitive.layer_sampler is not config.cognitive.layer_sampler
    assert observed["workspace_config"] is not config.workspace
    owned_workspace = cast(Any, observed["workspace_config"])
    assert owned_workspace.arbiter is not config.workspace.arbiter
    assert owned_workspace.allocator is not config.workspace.allocator
    assert observed["backbone"] is backbone
    assert observed["device_info"] is CPU_INFO
    assert observed["stop_token_ids"] is STOP_TOKEN_IDS
    assert observed["sampling"] is sampling
    assert config.model_dump_json() == config_snapshot
    assert backbone.eos_reads == 0
    inspect.assert_called_once_with("cpu")
    load.assert_called_once()
    build.assert_called_once()


@pytest.mark.parametrize(
    ("config_factory", "stop_token_ids", "sampling_factory", "stage"),
    [
        (lambda: object(), STOP_TOKEN_IDS, sampling_config, "configuration validation"),
        (
            lambda: OmegaConf.create({**valid_bootstrap_data(), "unexpected": True}),
            STOP_TOKEN_IDS,
            sampling_config,
            "configuration validation",
        ),
        (
            lambda: OmegaConf.create(
                {
                    **valid_bootstrap_data(),
                    "runtime": {
                        **cast(dict[str, object], valid_bootstrap_data()["runtime"]),
                        "seed": "???",
                    },
                }
            ),
            STOP_TOKEN_IDS,
            sampling_config,
            "configuration validation",
        ),
        (validated_config, [2, 7], sampling_config, "generation-policy validation"),
        (validated_config, (2, 2), sampling_config, "generation-policy validation"),
        (validated_config, (True,), sampling_config, "generation-policy validation"),
        (validated_config, STOP_TOKEN_IDS, object, "generation-policy validation"),
    ],
)
def test_invalid_preflight_never_inspects_device_or_touches_runtime(
    config_factory: Any,
    stop_token_ids: object,
    sampling_factory: Any,
    stage: str,
) -> None:
    config = config_factory()
    sampling = sampling_factory()

    with (
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "capture_runtime_state") as capture,
        patch.object(bootstrap, "apply_runtime_policy") as apply,
        patch.object(bootstrap, "_create_backbone") as load,
        patch.object(bootstrap, "build_request_orchestrator") as build,
        pytest.raises(IrisBootstrapError, match=stage) as raised,
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=cast(tuple[int, ...], stop_token_ids),
            sampling=cast(TemperatureTopPSamplingConfig, sampling),
        )

    assert raised.value.__cause__ is not None
    inspect.assert_not_called()
    capture.assert_not_called()
    apply.assert_not_called()
    load.assert_not_called()
    build.assert_not_called()


def test_nested_model_construct_cannot_bypass_owned_root_revalidation() -> None:
    valid = validated_config()
    constructed_runtime = RuntimeConfig.model_construct(
        seed=True,
        deterministic=valid.runtime.deterministic,
        log_level=valid.runtime.log_level,
        output_dir=valid.runtime.output_dir,
    )
    constructed_root = IrisBootstrapConfig.model_construct(
        project=valid.project,
        hardware=valid.hardware,
        backbone=valid.backbone,
        cognitive=valid.cognitive,
        workspace=valid.workspace,
        runtime=constructed_runtime,
    )

    with (
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "capture_runtime_state") as capture,
        pytest.raises(IrisBootstrapError, match="configuration validation") as raised,
    ):
        bootstrap_request_runtime(
            constructed_root,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert raised.value.__cause__ is not None
    inspect.assert_not_called()
    capture.assert_not_called()


def test_deterministic_cuda_requires_environment_before_device_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = validated_config(device="cuda", precision="fp16")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with (
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "capture_runtime_state") as capture,
        pytest.raises(IrisBootstrapError, match="CUDA deterministic preflight") as raised,
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    inspect.assert_not_called()
    capture.assert_not_called()


def test_deterministic_cuda_rejects_initialized_runtime_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = validated_config(device="cuda", precision="fp16")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    with (
        patch.object(torch.cuda, "is_initialized", return_value=True) as initialized,
        patch.object(bootstrap, "inspect_device") as inspect,
        patch.object(bootstrap, "capture_runtime_state") as capture,
        pytest.raises(IrisBootstrapError, match="CUDA deterministic preflight") as raised,
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert "before CUDA initialization" in str(raised.value.__cause__)
    initialized.assert_called_once_with()
    inspect.assert_not_called()
    capture.assert_not_called()


def test_success_commits_seeded_rng_and_leaves_constructor_draws_advanced() -> None:
    seed = 31337
    config = validated_config(seed=seed)
    backbone = OfflineBootstrapBackbone()
    orchestrator = orchestrator_double()
    observed_python: list[float] = []
    observed_torch: list[torch.Tensor] = []

    def load_backbone(*_args: object) -> BackboneAdapter:
        observed_python.append(random.random())
        observed_torch.append(torch.rand(3))
        return backbone

    def build_graph(**_kwargs: object) -> IrisRequestOrchestrator:
        observed_python.append(random.random())
        observed_torch.append(torch.rand(2))
        return orchestrator

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", side_effect=load_backbone),
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            side_effect=build_graph,
        ),
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    expected_python = random.Random(seed)
    assert observed_python == [expected_python.random(), expected_python.random()]
    assert random.random() == expected_python.random()

    expected_torch = torch.Generator(device="cpu")
    expected_torch.manual_seed(seed)
    assert torch.equal(observed_torch[0], torch.rand(3, generator=expected_torch))
    assert torch.equal(observed_torch[1], torch.rand(2, generator=expected_torch))
    assert torch.equal(torch.rand(4), torch.rand(4, generator=expected_torch))
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_failure_restores_python_cpu_rng_determinism_and_cudnn_exactly() -> None:
    python_rng = random.Random(991)
    random.setstate(python_rng.getstate())
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(992)
    torch.set_rng_state(cpu_rng.get_state())
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    before = capture_runtime_state(CPU)
    failure = RuntimeError("composition exploded")

    def load_backbone(*_args: object) -> BackboneAdapter:
        _ = random.random()
        _ = torch.rand(4)
        return OfflineBootstrapBackbone()

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", side_effect=load_backbone),
        patch.object(bootstrap, "build_request_orchestrator", side_effect=failure),
        pytest.raises(IrisBootstrapError, match="request-graph composition") as raised,
    ):
        bootstrap_request_runtime(
            validated_config(seed=1234),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert raised.value.__cause__ is failure
    assert_current_state_matches(before)


def test_cpu_bootstrap_never_reads_or_seeds_cuda_or_uses_global_manual_seed() -> None:
    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(
            bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            return_value=orchestrator_double(),
        ),
        patch.object(torch.cuda, "get_rng_state") as get_cuda_rng,
        patch.object(torch.cuda, "set_rng_state") as set_cuda_rng,
        patch.object(torch, "manual_seed") as global_manual_seed,
    ):
        bootstrap_request_runtime(
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    get_cuda_rng.assert_not_called()
    set_cuda_rng.assert_not_called()
    global_manual_seed.assert_not_called()


def test_concrete_cuda_device_and_explicit_environment_flow_through_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = validated_config(device="cuda", precision="fp16")
    backbone = OfflineBootstrapBackbone(device=CUDA_2, dtype=torch.float16)
    orchestrator = orchestrator_double()
    snapshot = capture_runtime_state(CPU)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with (
        patch.object(bootstrap, "inspect_device", return_value=CUDA_INFO) as inspect,
        patch.object(bootstrap, "capture_runtime_state", return_value=snapshot) as capture,
        patch.object(bootstrap, "apply_runtime_policy") as apply,
        patch.object(bootstrap, "_create_backbone", return_value=backbone) as load,
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            return_value=orchestrator,
        ) as build,
    ):
        runtime = bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert runtime.device_info is CUDA_INFO
    assert runtime.dtype is torch.float16
    inspect.assert_called_once_with("cuda")
    capture.assert_called_once_with(CUDA_2)
    apply.assert_called_once_with(
        seed=config.runtime.seed,
        deterministic=True,
        device=CUDA_2,
    )
    load.assert_called_once_with(cast(Any, load.call_args.args[0]), CUDA_INFO)
    assert load.call_args.args[0] is not config.backbone
    assert build.call_args.kwargs["backbone"] is backbone


def test_cuda_seed_policy_targets_only_cpu_and_selected_cuda_generator() -> None:
    created_devices: list[torch.device] = []
    seeded: list[tuple[torch.device, int]] = []

    class FakeGenerator:
        def __init__(self, *, device: torch.device | str) -> None:
            self.device = torch.device(device)
            created_devices.append(self.device)

        def manual_seed(self, seed: int) -> FakeGenerator:
            seeded.append((self.device, seed))
            return self

        def get_state(self) -> torch.Tensor:
            return torch.tensor([1, 2, 3], dtype=torch.uint8)

    with (
        patch("iris.utils.seeds.torch.Generator", side_effect=FakeGenerator),
        patch("iris.utils.seeds.torch.set_rng_state") as set_cpu_rng,
        patch("iris.utils.seeds.torch.cuda.set_rng_state") as set_cuda_rng,
        patch("iris.utils.seeds.torch.manual_seed") as global_manual_seed,
        patch("iris.utils.seeds.torch.cuda.manual_seed") as cuda_manual_seed,
        patch("iris.utils.seeds.torch.cuda.manual_seed_all") as cuda_manual_seed_all,
    ):
        apply_runtime_policy(seed=41, deterministic=False, device=CUDA_2)

    assert created_devices == [CPU, CUDA_2]
    assert seeded == [(CPU, 41), (CUDA_2, 41)]
    set_cpu_rng.assert_called_once()
    set_cuda_rng.assert_called_once_with(
        set_cuda_rng.call_args.args[0],
        device=CUDA_2,
    )
    global_manual_seed.assert_not_called()
    cuda_manual_seed.assert_not_called()
    cuda_manual_seed_all.assert_not_called()


def test_cuda_state_capture_reads_only_the_selected_concrete_device() -> None:
    cuda_state = torch.tensor([9, 8, 7], dtype=torch.uint8)
    with patch.object(torch.cuda, "get_rng_state", return_value=cuda_state) as get_rng:
        snapshot = capture_runtime_state(CUDA_2)

    get_rng.assert_called_once_with(CUDA_2)
    assert snapshot.cuda_device == CUDA_2
    assert snapshot.cuda_rng_state is not cuda_state
    assert torch.equal(cast(torch.Tensor, snapshot.cuda_rng_state), cuda_state)


@pytest.mark.parametrize(
    ("target", "stage"),
    [
        ("inspect_device", "device resolution"),
        ("capture_runtime_state", "runtime-state capture"),
        ("apply_runtime_policy", "runtime-policy application"),
        ("_create_backbone", "backbone loading"),
        ("build_request_orchestrator", "request-graph composition"),
    ],
)
def test_stage_failures_are_chained_and_never_retried(target: str, stage: str) -> None:
    failure = RuntimeError(f"{target} exploded")
    with ExitStack() as stack:
        stage_mocks: dict[str, MagicMock] = {
            "inspect_device": stack.enter_context(
                patch.object(bootstrap, "inspect_device", return_value=CPU_INFO)
            ),
            "capture_runtime_state": stack.enter_context(
                patch.object(
                    bootstrap,
                    "capture_runtime_state",
                    return_value=capture_runtime_state(CPU),
                )
            ),
            "apply_runtime_policy": stack.enter_context(
                patch.object(bootstrap, "apply_runtime_policy")
            ),
            "_create_backbone": stack.enter_context(
                patch.object(
                    bootstrap,
                    "_create_backbone",
                    return_value=OfflineBootstrapBackbone(),
                )
            ),
            "build_request_orchestrator": stack.enter_context(
                patch.object(
                    bootstrap,
                    "build_request_orchestrator",
                    return_value=orchestrator_double(),
                )
            ),
        }
        failing = stage_mocks[target]
        failing.side_effect = failure
        with pytest.raises(IrisBootstrapError, match=stage) as raised:
            bootstrap_request_runtime(
                validated_config(),
                stop_token_ids=STOP_TOKEN_IDS,
                sampling=sampling_config(),
            )

    assert raised.value.__cause__ is failure
    failing.assert_called_once()


def test_rollback_failure_raises_uncertain_state_with_both_failures() -> None:
    primary = RuntimeError("composition exploded")
    rollback = OSError("rollback exploded")

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(
            bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(bootstrap, "build_request_orchestrator", side_effect=primary),
        patch.object(bootstrap, "restore_runtime_state", side_effect=rollback),
        pytest.raises(IrisBootstrapError, match="state is uncertain") as raised,
    ):
        bootstrap_request_runtime(
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    group = raised.value.__cause__
    assert isinstance(group, ExceptionGroup)
    assert len(group.exceptions) == 2
    staged_failure, rollback_failure = group.exceptions
    assert isinstance(staged_failure, IrisBootstrapError)
    assert staged_failure.__cause__ is primary
    assert rollback_failure is rollback


def test_owned_config_mutation_is_rejected_without_mutating_caller_dto() -> None:
    config = validated_config()
    caller_snapshot = config.model_dump_json()
    failure_backbone = OfflineBootstrapBackbone()

    def mutate_owned_config(**kwargs: object) -> IrisRequestOrchestrator:
        owned_cognitive = cast(Any, kwargs["cognitive_config"])
        owned_cognitive.name = "mutated-by-composer"
        return orchestrator_double()

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", return_value=failure_backbone),
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            side_effect=mutate_owned_config,
        ),
        pytest.raises(IrisBootstrapError, match="final runtime validation") as raised,
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert "Owned validated configuration changed" in str(raised.value.__cause__)
    assert config.model_dump_json() == caller_snapshot


def test_external_config_mutation_during_load_is_detected_and_rolled_back() -> None:
    config = validated_config()
    before = capture_runtime_state(CPU)

    def mutate_caller(*_args: object) -> BackboneAdapter:
        config.project.name = "externally-mutated"
        return OfflineBootstrapBackbone()

    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", side_effect=mutate_caller),
        patch.object(
            bootstrap,
            "build_request_orchestrator",
            return_value=orchestrator_double(),
        ),
        pytest.raises(IrisBootstrapError, match="final runtime validation") as raised,
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert "Caller configuration changed" in str(raised.value.__cause__)
    assert_current_state_matches(before)


@pytest.mark.parametrize(
    ("backbone", "message"),
    [
        (OfflineBootstrapBackbone(vocabulary_size=7), "out-of-vocabulary"),
        (OfflineBootstrapBackbone(trainable_parameter_count=1), "trainable"),
        (OfflineBootstrapBackbone(has_parameter_gradients=True), "gradients"),
        (OfflineBootstrapBackbone(dtype=torch.float64), "dtype"),
    ],
)
def test_invalid_loaded_backbone_fails_before_composition(
    backbone: BackboneAdapter,
    message: str,
) -> None:
    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(bootstrap, "_create_backbone", return_value=backbone),
        patch.object(bootstrap, "build_request_orchestrator") as build,
        pytest.raises(IrisBootstrapError, match="backbone validation") as raised,
    ):
        bootstrap_request_runtime(
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert message in str(raised.value.__cause__)
    build.assert_not_called()


def test_wrong_composition_result_is_rejected_and_runtime_is_rolled_back() -> None:
    before = capture_runtime_state(CPU)
    with (
        patch.object(bootstrap, "inspect_device", return_value=CPU_INFO),
        patch.object(
            bootstrap,
            "_create_backbone",
            return_value=OfflineBootstrapBackbone(),
        ),
        patch.object(bootstrap, "build_request_orchestrator", return_value=object()),
        pytest.raises(IrisBootstrapError, match="did not return"),
    ):
        bootstrap_request_runtime(
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert_current_state_matches(before)


def test_runtime_and_error_are_canonical_public_frozen_exports() -> None:
    assert application.IrisBootstrapError is IrisBootstrapError
    assert application.IrisRequestRuntime is IrisRequestRuntime
    assert application.bootstrap_request_runtime is bootstrap_request_runtime
    assert "IrisBootstrapError" in application.__all__
    assert "IrisRequestRuntime" in application.__all__
    assert "bootstrap_request_runtime" in application.__all__

    runtime = IrisRequestRuntime(
        backbone=OfflineBootstrapBackbone(),
        orchestrator=orchestrator_double(),
        device_info=CPU_INFO,
        dtype=torch.float32,
        seed=42,
        deterministic=True,
        configuration_json=bootstrap._canonical_config_json(validated_config()),
        configuration_sha256=domain_sha256(
            "iris/configuration/v1",
            bootstrap._canonical_config_json(validated_config()).encode(),
        ),
        sidecar_state_sha256=SIDECAR_STATE_SHA256,
    )
    field_name = "seed"
    with pytest.raises(FrozenInstanceError):
        setattr(runtime, field_name, 43)


def test_restore_runtime_state_attempts_every_owned_restore_before_grouping() -> None:
    snapshot = capture_runtime_state(CPU)
    python_failure = RuntimeError("python restore exploded")
    deterministic_failure = RuntimeError("determinism restore exploded")

    with (
        patch("iris.utils.seeds.random.setstate", side_effect=python_failure) as restore_python,
        patch("iris.utils.seeds.torch.set_rng_state") as restore_cpu,
        patch(
            "iris.utils.seeds.torch.use_deterministic_algorithms",
            side_effect=deterministic_failure,
        ) as restore_determinism,
        pytest.raises(ExceptionGroup) as raised,
    ):
        restore_runtime_state(snapshot)

    assert raised.value.exceptions == (python_failure, deterministic_failure)
    restore_python.assert_called_once()
    restore_cpu.assert_called_once()
    restore_determinism.assert_called_once()
    assert torch.backends.cudnn.deterministic is snapshot.cudnn_deterministic
    assert torch.backends.cudnn.benchmark is snapshot.cudnn_benchmark


def test_concrete_cuda_env_accepts_both_documented_workspace_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = validated_config(device="cuda", precision="fp16")
    for workspace_value in (":4096:8", ":16:8"):
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace_value)
        bootstrap._validate_cuda_determinism(config)


def test_cuda_device_resolution_rejects_an_indexless_result_before_rng_capture() -> None:
    config = validated_config(
        device="cuda",
        precision="fp16",
        deterministic=False,
    )
    indexless = DeviceInfo(
        device=torch.device("cuda"),
        backend="cuda",
        name="Indexless CUDA",
        total_memory_bytes=1024,
        supports_fp16=True,
        supports_bf16=False,
    )
    with (
        patch.object(bootstrap, "inspect_device", return_value=indexless),
        patch.object(bootstrap, "capture_runtime_state") as capture,
        pytest.raises(IrisBootstrapError, match="device resolution") as raised,
    ):
        bootstrap_request_runtime(
            config,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert "concrete nonnegative device index" in str(raised.value.__cause__)
    capture.assert_not_called()


@pytest.mark.parametrize(
    ("device_info", "message"),
    [
        (
            DeviceInfo(
                device=cast(Any, "cpu"),
                backend="cpu",
                name="String CPU",
                total_memory_bytes=None,
                supports_fp16=False,
                supports_bf16=False,
            ),
            "must contain a torch.device",
        ),
        (
            DeviceInfo(
                device=torch.device("cpu", 1),
                backend="cpu",
                name="Indexed CPU",
                total_memory_bytes=None,
                supports_fp16=False,
                supports_bf16=False,
            ),
            "canonical indexless CPU",
        ),
    ],
)
def test_device_resolution_rejects_malformed_cpu_metadata_before_rng_capture(
    device_info: DeviceInfo,
    message: str,
) -> None:
    with (
        patch.object(bootstrap, "inspect_device", return_value=device_info),
        patch.object(bootstrap, "capture_runtime_state") as capture,
        pytest.raises(IrisBootstrapError, match="device resolution") as raised,
    ):
        bootstrap_request_runtime(
            validated_config(),
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling_config(),
        )

    assert message in str(raised.value.__cause__)
    capture.assert_not_called()
