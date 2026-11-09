"""Offline contracts for the packaged managed-experiment command service."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

import iris.application.bootstrap as bootstrap_service
import iris.application.experiment_command as experiment_command
from iris.application.errors import IrisExperimentCommandError
from iris.application.experiment_manifest import (
    build_experiment_manifest_v1,
    read_experiment_manifest_v1,
    write_experiment_manifest_v1,
)
from iris.application.experiment_runner import IrisExperimentExecutionResult
from iris.artifacts.experiment_log_v1 import ExperimentEpochExecutionPolicyV1
from iris.artifacts.experiment_v1 import ExperimentExampleV1, IrisExperimentManifestV1
from iris.backbone import BackboneAdapter
from iris.schemas import IrisBootstrapConfig
from iris.training import IrisAdamWConfig, IrisTeacherForcingModel, IrisTeacherOptimizer
from iris.utils.devices import DeviceInfo
from iris.utils.seeds import RuntimeStateSnapshot, capture_runtime_state, restore_runtime_state

CPU = torch.device("cpu")
CPU_INFO = DeviceInfo(
    device=CPU,
    backend="cpu",
    name="Test CPU",
    total_memory_bytes=None,
    supports_fp16=False,
    supports_bf16=False,
)
SEED = 41
TERMINAL_TOKEN_ID = 151_645
OPTIMIZER_CONFIG = IrisAdamWConfig(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    max_gradient_norm=1.0,
)
BACKBONE = cast(BackboneAdapter, object())
MODEL = cast(IrisTeacherForcingModel, object())
OPTIMIZER = cast(IrisTeacherOptimizer, object())
RUNTIME_SNAPSHOT = cast(RuntimeStateSnapshot, object())


class AdamWConfigSubclass(IrisAdamWConfig):
    """A structurally valid value rejected by the exact ownership boundary."""


def _example(example_id: str, split: str) -> ExperimentExampleV1:
    return ExperimentExampleV1.model_validate(
        {
            "example_id": example_id,
            "split": split,
            "prompt": f"prompt-{example_id}",
            "target": f"target-{example_id}",
        }
    )


def _manifest(
    *, include_test: bool = True, dataset_id: str = "command-dataset"
) -> IrisExperimentManifestV1:
    examples = [
        _example("train-1", "train"),
        _example("validation-1", "validation"),
    ]
    if include_test:
        examples.append(_example("test-1", "test"))
    return build_experiment_manifest_v1(dataset_id=dataset_id, examples=tuple(examples))


def _write_manifest(
    tmp_path: Path,
    *,
    include_test: bool = True,
    dataset_id: str = "command-dataset",
) -> tuple[Path, IrisExperimentManifestV1]:
    manifest = _manifest(include_test=include_test, dataset_id=dataset_id)
    path = tmp_path / f"{dataset_id}.json"
    write_experiment_manifest_v1(manifest, path=path)
    return path, manifest


def _arguments(
    tmp_path: Path,
    *,
    manifest_path: Path | None = None,
    **overrides: object,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "manifest_path": manifest_path or (tmp_path / "manifest.json"),
        "root": tmp_path / "runs",
        "run_id": "managed-command-run",
        "terminal_token_id": TERMINAL_TOKEN_ID,
        "initialization_seed": SEED,
        "epoch_count": 2,
        "evaluate_test_after_training": True,
        "optimizer_config": OPTIMIZER_CONFIG,
        "maximum_new_training_steps": None,
        "allow_download": False,
    }
    values.update(overrides)
    return values


def _owned_config(tmp_path: Path, *, allow_download: bool = False) -> IrisBootstrapConfig:
    command = experiment_command._own_command(
        **_arguments(
            tmp_path,
            allow_download=allow_download,
        )
    )
    composed = experiment_command._compose_experiment_config(
        allow_download=allow_download,
        initialization_seed=SEED,
    )
    return experiment_command._own_config(composed, command=command)


def _result(
    root: Path,
    *,
    status: str = "complete",
    maximum_new_training_steps: int | None = None,
) -> IrisExperimentExecutionResult:
    bounded = status == "step_limit_reached"
    return IrisExperimentExecutionResult(
        path=root / "experiments" / "v1" / "managed-command-run",
        status=cast(Any, status),
        run_payload_sha256="a" * 64,
        configuration_sha256="b" * 64,
        event_count=7,
        optimizer_completed_steps=3,
        completed_epochs=1,
        next_epoch_index=2 if bounded else None,
        next_training_ordinal=1 if bounded else None,
        latest_checkpoint_payload_sha256="c" * 64,
        replayed_training_steps=1,
        new_training_steps=maximum_new_training_steps or 2,
    )


def _assert_runtime_state(snapshot: RuntimeStateSnapshot) -> None:
    assert random.getstate() == snapshot.python_rng_state
    assert torch.equal(torch.get_rng_state(), snapshot.cpu_rng_state)
    assert torch.are_deterministic_algorithms_enabled() is snapshot.deterministic_algorithms
    assert torch.is_deterministic_algorithms_warn_only_enabled() is snapshot.deterministic_warn_only
    assert torch.backends.cudnn.deterministic is snapshot.cudnn_deterministic
    assert torch.backends.cudnn.benchmark is snapshot.cudnn_benchmark


@contextmanager
def _mocked_runtime(
    *,
    config: IrisBootstrapConfig,
    manifests: tuple[IrisExperimentManifestV1, ...],
    result: IrisExperimentExecutionResult | None = None,
    execution_failure: BaseException | None = None,
    restore_failure: BaseException | None = None,
) -> Iterator[dict[str, MagicMock]]:
    if result is None:
        result = _result(Path("runs"))

    def run_execution(**_: object) -> IrisExperimentExecutionResult:
        if execution_failure is not None:
            raise execution_failure
        assert result is not None
        return result

    with (
        patch.object(
            experiment_command,
            "read_experiment_manifest_v1",
            side_effect=manifests,
        ) as read_manifest,
        patch.object(
            experiment_command,
            "_compose_experiment_config",
            return_value=object(),
        ) as compose_config,
        patch.object(experiment_command, "_own_config", return_value=config) as own_config,
        patch.object(experiment_command, "_preflight_existing_run") as existing_preflight,
        patch.object(bootstrap_service, "_validate_cuda_determinism"),
        patch.object(
            bootstrap_service,
            "_resolve_device",
            return_value=(CPU_INFO, torch.float32),
        ),
        patch.object(experiment_command, "_validate_ambient_training_context"),
        patch.object(
            experiment_command,
            "capture_runtime_state",
            return_value=RUNTIME_SNAPSHOT,
        ) as capture_runtime,
        patch.object(experiment_command, "apply_runtime_policy") as apply_runtime,
        patch.object(experiment_command, "_create_backbone", return_value=BACKBONE) as create,
        patch.object(experiment_command, "_validate_runtime_backbone"),
        patch.object(experiment_command, "_validate_teacher_target_policy"),
        patch.object(
            experiment_command,
            "build_teacher_forcing_model",
            return_value=MODEL,
        ) as build_model,
        patch.object(
            experiment_command,
            "IrisTeacherOptimizer",
            return_value=OPTIMIZER,
        ) as build_optimizer,
        patch.object(experiment_command, "_validate_pristine_training_owner"),
        patch.object(
            experiment_command,
            "execute_experiment_v1",
            side_effect=run_execution,
        ) as execute,
        patch.object(experiment_command, "_validate_live_training_owner") as validate_final,
        patch.object(
            experiment_command,
            "restore_runtime_state",
            side_effect=restore_failure,
        ) as restore_runtime,
    ):
        yield {
            "read_manifest": read_manifest,
            "compose_config": compose_config,
            "own_config": own_config,
            "existing_preflight": existing_preflight,
            "capture_runtime": capture_runtime,
            "apply_runtime": apply_runtime,
            "create": create,
            "build_model": build_model,
            "build_optimizer": build_optimizer,
            "execute": execute,
            "validate_final": validate_final,
            "restore_runtime": restore_runtime,
        }


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("manifest_path", "manifest.json"),
        ("manifest_path", Path("manifest.txt")),
        ("root", "."),
        ("run_id", "../escape"),
        ("terminal_token_id", True),
        ("terminal_token_id", -1),
        ("initialization_seed", True),
        ("initialization_seed", -1),
        ("initialization_seed", 2**63),
        ("epoch_count", True),
        ("epoch_count", 0),
        ("epoch_count", 1_000_001),
        ("evaluate_test_after_training", 1),
        ("optimizer_config", object()),
        (
            "optimizer_config",
            AdamWConfigSubclass(
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.01,
                max_gradient_norm=1.0,
            ),
        ),
        ("maximum_new_training_steps", True),
        ("maximum_new_training_steps", 0),
        ("allow_download", 0),
    ],
)
def test_strict_input_preflight_runs_before_manifest_or_configuration(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    arguments = _arguments(tmp_path)
    arguments[field] = invalid
    with (
        patch.object(experiment_command, "read_experiment_manifest_v1") as read_manifest,
        patch.object(experiment_command, "_compose_experiment_config") as compose_config,
        pytest.raises(IrisExperimentCommandError, match="input validation") as captured,
    ):
        experiment_command.run_packaged_experiment_v1(**arguments)

    read_manifest.assert_not_called()
    compose_config.assert_not_called()
    assert captured.value.__cause__ is not None


def test_existing_root_must_be_a_real_directory_before_manifest_read(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("private", encoding="utf-8")
    with (
        patch.object(experiment_command, "read_experiment_manifest_v1") as read_manifest,
        pytest.raises(IrisExperimentCommandError, match="input validation"),
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path, root=root))
    read_manifest.assert_not_called()


def test_manifest_failure_precedes_configuration_and_runtime(tmp_path: Path) -> None:
    failure = RuntimeError("PRIVATE_MANIFEST_CONTENT")
    with (
        patch.object(
            experiment_command,
            "read_experiment_manifest_v1",
            side_effect=failure,
        ) as read_manifest,
        patch.object(experiment_command, "_compose_experiment_config") as compose_config,
        patch.object(experiment_command, "capture_runtime_state") as capture_runtime,
        pytest.raises(IrisExperimentCommandError, match="manifest verification") as captured,
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))

    read_manifest.assert_called_once()
    compose_config.assert_not_called()
    capture_runtime.assert_not_called()
    assert "PRIVATE_MANIFEST_CONTENT" not in str(captured.value)
    assert captured.value.__cause__ is failure


def test_test_split_policy_is_rejected_before_configuration(tmp_path: Path) -> None:
    manifest = _manifest(include_test=False)
    with (
        patch.object(
            experiment_command,
            "read_experiment_manifest_v1",
            return_value=manifest,
        ),
        patch.object(experiment_command, "_compose_experiment_config") as compose_config,
        pytest.raises(IrisExperimentCommandError, match="schedule validation"),
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))
    compose_config.assert_not_called()


def test_schedule_event_bound_is_rejected_before_configuration(tmp_path: Path) -> None:
    manifest = _manifest(include_test=False)
    with (
        patch.object(
            experiment_command,
            "read_experiment_manifest_v1",
            return_value=manifest,
        ),
        patch.object(experiment_command, "_compose_experiment_config") as compose_config,
        pytest.raises(IrisExperimentCommandError, match="schedule validation"),
    ):
        experiment_command.run_packaged_experiment_v1(
            **_arguments(
                tmp_path,
                epoch_count=1_000_000,
                evaluate_test_after_training=False,
            )
        )
    compose_config.assert_not_called()


@pytest.mark.parametrize(
    ("allow_download", "expected_local_files_only"),
    [(False, True), (True, False)],
)
def test_packaged_configuration_is_offline_first_and_owns_the_explicit_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_download: bool,
    expected_local_files_only: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    cwd_before = Path.cwd()
    assert not GlobalHydra.instance().is_initialized()

    composed = experiment_command._compose_experiment_config(
        allow_download=allow_download,
        initialization_seed=SEED,
    )

    assert composed.backbone.local_files_only is expected_local_files_only
    assert composed.runtime.seed == SEED
    assert composed.runtime.deterministic is True
    assert composed.backbone.trust_remote_code is False
    assert Path.cwd() == cwd_before
    assert list(tmp_path.iterdir()) == []
    assert not GlobalHydra.instance().is_initialized()


def test_composition_uses_only_packaged_config_and_two_owned_overrides() -> None:
    composed = object()

    @contextmanager
    def initialize_spy(*, version_base: None, config_module: str) -> Iterator[None]:
        assert version_base is None
        assert config_module == "iris.configs"
        yield

    with (
        patch.object(
            experiment_command,
            "initialize_config_module",
            side_effect=initialize_spy,
        ) as initialize,
        patch.object(experiment_command, "compose", return_value=composed) as compose,
        pytest.raises(TypeError, match="DictConfig"),
    ):
        experiment_command._compose_experiment_config(
            allow_download=False,
            initialization_seed=SEED,
        )

    initialize.assert_called_once_with(version_base=None, config_module="iris.configs")
    compose.assert_called_once_with(
        config_name="config",
        overrides=["backbone.local_files_only=true", f"runtime.seed={SEED}"],
    )


def test_existing_run_conflict_fails_before_device_or_model(tmp_path: Path) -> None:
    config = _owned_config(tmp_path)
    manifest = _manifest()
    with (
        patch.object(
            experiment_command,
            "read_experiment_manifest_v1",
            return_value=manifest,
        ),
        patch.object(experiment_command, "_compose_experiment_config", return_value=object()),
        patch.object(experiment_command, "_own_config", return_value=config),
        patch.object(
            experiment_command,
            "_preflight_existing_run",
            side_effect=ValueError("identity conflict"),
        ) as preflight,
        patch.object(bootstrap_service, "_resolve_device") as resolve_device,
        patch.object(experiment_command, "_create_backbone") as create,
        pytest.raises(IrisExperimentCommandError, match="existing-run preflight"),
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))

    preflight.assert_called_once()
    resolve_device.assert_not_called()
    create.assert_not_called()


def test_existing_run_preflight_compares_every_identity_bearing_input(tmp_path: Path) -> None:
    config = _owned_config(tmp_path)
    manifest = _manifest()
    command = experiment_command._own_command(**_arguments(tmp_path))
    policy = experiment_command._execution_policy(command)
    path = command.root / "experiments" / "v1" / command.run_id
    path.mkdir(parents=True)
    optimizer = SimpleNamespace(
        optimizer_type="torch.optim.AdamW",
        config=SimpleNamespace(
            learning_rate=OPTIMIZER_CONFIG.learning_rate,
            beta1=OPTIMIZER_CONFIG.beta1,
            beta2=OPTIMIZER_CONFIG.beta2,
            epsilon=OPTIMIZER_CONFIG.epsilon,
            weight_decay=OPTIMIZER_CONFIG.weight_decay,
            max_gradient_norm=OPTIMIZER_CONFIG.max_gradient_norm,
        ),
    )
    training = SimpleNamespace(
        target_preparation=SimpleNamespace(terminal_token_id=TERMINAL_TOKEN_ID),
        runtime_policy=SimpleNamespace(initialization_seed=SEED, deterministic=True),
        execution_policy=policy,
        model=SimpleNamespace(
            backbone=config.backbone,
            cognitive=config.cognitive,
            workspace=config.workspace,
        ),
        optimizer=optimizer,
    )
    payload = SimpleNamespace(
        run_id=command.run_id,
        manifest=SimpleNamespace(
            dataset_id=manifest.payload.dataset_id,
            payload_sha256=manifest.payload_sha256,
            split_counts=manifest.payload.split_counts,
        ),
        configuration=training,
    )
    snapshot = SimpleNamespace(run=SimpleNamespace(payload=payload))
    with patch.object(experiment_command, "read_experiment_log_v1", return_value=snapshot):
        experiment_command._preflight_existing_run(
            command=command,
            manifest=manifest,
            config=config,
            policy=policy,
        )

    training.runtime_policy.initialization_seed = SEED + 1
    with (
        patch.object(experiment_command, "read_experiment_log_v1", return_value=snapshot),
        pytest.raises(ValueError, match="identity conflicts"),
    ):
        experiment_command._preflight_existing_run(
            command=command,
            manifest=manifest,
            config=config,
            policy=policy,
        )


def test_ambient_training_context_rejects_no_grad_inference_and_autocast() -> None:
    with torch.no_grad(), pytest.raises(ValueError, match="autograd"):
        experiment_command._validate_ambient_training_context(CPU_INFO)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        experiment_command._validate_ambient_training_context(CPU_INFO)
    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="automatic mixed precision"),
    ):
        experiment_command._validate_ambient_training_context(CPU_INFO)


@pytest.mark.parametrize(
    ("eos_token_ids", "max_new_tokens", "message"),
    [
        ((2,), 32, "declared tokenizer EOS"),
        ((TERMINAL_TOKEN_ID,), 1, "content plus its terminal"),
    ],
)
def test_teacher_target_policy_fails_before_teacher_composition(
    eos_token_ids: tuple[int, ...],
    max_new_tokens: int,
    message: str,
) -> None:
    backbone = SimpleNamespace(
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
    )
    with pytest.raises(ValueError, match=message):
        experiment_command._validate_teacher_target_policy(
            cast(BackboneAdapter, backbone),
            terminal_token_id=TERMINAL_TOKEN_ID,
        )


def test_success_forwards_one_fixed_policy_and_restores_global_runtime(tmp_path: Path) -> None:
    manifest_path, manifest = _write_manifest(tmp_path)
    root = tmp_path / "runs"
    result = _result(root)
    random.seed(912_771, version=2)
    torch.manual_seed(821_663)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    before = capture_runtime_state(CPU)

    def execute_side_effect(**kwargs: object) -> IrisExperimentExecutionResult:
        random.random()
        torch.rand(3)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.initial_seed() == SEED
        return result

    try:
        with (
            patch.object(
                experiment_command,
                "read_experiment_manifest_v1",
                wraps=read_experiment_manifest_v1,
            ) as read_manifest,
            patch.object(experiment_command, "_create_backbone", return_value=BACKBONE) as create,
            patch.object(experiment_command, "_validate_runtime_backbone"),
            patch.object(experiment_command, "_validate_teacher_target_policy"),
            patch.object(
                experiment_command,
                "build_teacher_forcing_model",
                return_value=MODEL,
            ) as build_model,
            patch.object(
                experiment_command,
                "IrisTeacherOptimizer",
                return_value=OPTIMIZER,
            ) as build_optimizer,
            patch.object(experiment_command, "_validate_pristine_training_owner"),
            patch.object(
                experiment_command,
                "execute_experiment_v1",
                side_effect=execute_side_effect,
            ) as execute,
            patch.object(experiment_command, "_validate_live_training_owner") as validate_final,
        ):
            actual = experiment_command.run_packaged_experiment_v1(
                **_arguments(
                    tmp_path,
                    manifest_path=manifest_path,
                    root=root,
                )
            )

        assert actual is result
        assert read_manifest.call_count == 3
        create.assert_called_once()
        build_model.assert_called_once()
        build_optimizer.assert_called_once()
        owned_optimizer_config = build_optimizer.call_args.kwargs["config"]
        assert owned_optimizer_config == OPTIMIZER_CONFIG
        assert owned_optimizer_config is not OPTIMIZER_CONFIG
        execute.assert_called_once()
        arguments = execute.call_args.kwargs
        assert arguments["run_id"] == "managed-command-run"
        assert arguments["manifest"] == manifest
        assert arguments["terminal_token_id"] == TERMINAL_TOKEN_ID
        assert arguments["initialization_seed"] == SEED
        assert arguments["deterministic"] is True
        assert arguments["model"] is MODEL
        assert arguments["optimizer"] is OPTIMIZER
        assert arguments["root"] == root
        assert arguments["maximum_new_training_steps"] is None
        policy = arguments["execution_policy"]
        assert type(policy) is ExperimentEpochExecutionPolicyV1
        assert policy == ExperimentEpochExecutionPolicyV1(
            mode="deterministic_epochs",
            epoch_count=2,
            training_order="manifest_order_each_epoch",
            checkpoint_policy="after_each_complete_epoch",
            validation_policy="after_each_complete_epoch",
            evaluate_test_after_training=True,
            resume_policy="latest_checkpoint_plus_verified_replay",
        )
        validate_final.assert_called_once()
        _assert_runtime_state(before)
    finally:
        restore_runtime_state(before)


def test_runner_failure_is_wrapped_without_content_and_restores_runtime(tmp_path: Path) -> None:
    manifest_path, _ = _write_manifest(tmp_path)
    random.seed(122, version=2)
    torch.manual_seed(123)
    before = capture_runtime_state(CPU)
    failure = RuntimeError("PRIVATE_TARGET_AND_EXAMPLE_ID")
    try:
        with (
            patch.object(experiment_command, "_create_backbone", return_value=BACKBONE),
            patch.object(experiment_command, "_validate_runtime_backbone"),
            patch.object(experiment_command, "_validate_teacher_target_policy"),
            patch.object(
                experiment_command,
                "build_teacher_forcing_model",
                return_value=MODEL,
            ),
            patch.object(
                experiment_command,
                "IrisTeacherOptimizer",
                return_value=OPTIMIZER,
            ),
            patch.object(experiment_command, "_validate_pristine_training_owner"),
            patch.object(
                experiment_command,
                "execute_experiment_v1",
                side_effect=failure,
            ) as execute,
            pytest.raises(IrisExperimentCommandError, match="managed execution") as captured,
        ):
            experiment_command.run_packaged_experiment_v1(
                **_arguments(tmp_path, manifest_path=manifest_path)
            )

        execute.assert_called_once()
        assert "PRIVATE_TARGET_AND_EXAMPLE_ID" not in str(captured.value)
        assert captured.value.__cause__ is failure
        _assert_runtime_state(before)
    finally:
        restore_runtime_state(before)


def test_manifest_change_before_execution_restores_runtime_without_runner(tmp_path: Path) -> None:
    config = _owned_config(tmp_path)
    original = _manifest()
    changed = _manifest(dataset_id="changed-dataset")
    with (
        _mocked_runtime(
            config=config,
            manifests=(original, changed),
            result=_result(tmp_path / "runs"),
        ) as spies,
        pytest.raises(
            IrisExperimentCommandError,
            match="pre-execution manifest stability check",
        ),
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))

    spies["execute"].assert_not_called()
    spies["validate_final"].assert_not_called()
    spies["restore_runtime"].assert_called_once_with(RUNTIME_SNAPSHOT)


def test_manifest_change_after_execution_fails_closed_and_restores_runtime(tmp_path: Path) -> None:
    config = _owned_config(tmp_path)
    original = _manifest()
    changed = _manifest(dataset_id="changed-dataset")
    with (
        _mocked_runtime(
            config=config,
            manifests=(original, original, changed),
            result=_result(tmp_path / "runs"),
        ) as spies,
        pytest.raises(
            IrisExperimentCommandError,
            match="post-execution manifest stability check",
        ),
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))

    spies["execute"].assert_called_once()
    spies["validate_final"].assert_not_called()
    spies["restore_runtime"].assert_called_once_with(RUNTIME_SNAPSHOT)


def test_primary_and_restore_failures_report_uncertain_process_state(tmp_path: Path) -> None:
    config = _owned_config(tmp_path)
    manifest = _manifest()
    primary = RuntimeError("runner failed")
    rollback = RuntimeError("rollback failed")
    with (
        _mocked_runtime(
            config=config,
            manifests=(manifest, manifest),
            execution_failure=primary,
            restore_failure=rollback,
        ),
        pytest.raises(IrisExperimentCommandError, match="state is uncertain") as captured,
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))

    failures = captured.value.__cause__
    assert isinstance(failures, BaseExceptionGroup)
    assert len(failures.exceptions) == 2
    assert isinstance(failures.exceptions[0], IrisExperimentCommandError)
    assert failures.exceptions[0].__cause__ is primary
    assert failures.exceptions[1] is rollback


def test_successful_execution_with_restore_failure_reports_uncertain_state(tmp_path: Path) -> None:
    config = _owned_config(tmp_path)
    manifest = _manifest()
    rollback = RuntimeError("rollback failed")
    with (
        _mocked_runtime(
            config=config,
            manifests=(manifest, manifest, manifest),
            result=_result(tmp_path / "runs"),
            restore_failure=rollback,
        ),
        pytest.raises(
            IrisExperimentCommandError, match=r"completed.*state is uncertain"
        ) as captured,
    ):
        experiment_command.run_packaged_experiment_v1(**_arguments(tmp_path))

    assert captured.value.__cause__ is rollback


@pytest.mark.parametrize(
    ("mutation", "maximum_new_training_steps", "message"),
    [
        ({"status": "step_limit_reached"}, None, "unbounded managed execution"),
        ({"path": Path("wrong")}, None, "wrong run path"),
        ({"run_payload_sha256": "not-a-hash"}, None, "SHA-256"),
        ({"completed_epochs": 3}, None, "outside requested bounds"),
        ({"status": "unknown"}, 1, "invalid status"),
    ],
)
def test_execution_result_boundary_rejects_malformed_or_impossible_reports(
    tmp_path: Path,
    mutation: dict[str, object],
    maximum_new_training_steps: int | None,
    message: str,
) -> None:
    command = experiment_command._own_command(
        **_arguments(
            tmp_path,
            maximum_new_training_steps=maximum_new_training_steps,
        )
    )
    result = _result(
        command.root,
        maximum_new_training_steps=maximum_new_training_steps,
    )
    for field, value in mutation.items():
        object.__setattr__(result, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        experiment_command._validate_execution_result(
            result,
            command=command,
            manifest=_manifest(),
        )
