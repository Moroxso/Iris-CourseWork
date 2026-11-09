"""Dataset-cursor and RNG owner for scientific optimizer windows.

This declared cyclic, fixed-order training stream never silently drops an epoch
tail. Validation/checkpoint cadence and AMP are separate remaining run policies.
"""

import platform
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version

import torch

from iris import __version__
from iris.application.scientific_ablation_optimization import (
    IrisScientificAblationOptimizationOutputV1,
    IrisScientificAblationOptimizerV1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizationOutputV1,
    IrisScientificOptimizerV1,
)
from iris.data.schemas import IrisExample
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state


class IrisScientificRunError(RuntimeError):
    """Scientific stream ownership, cursor or runtime is incompatible."""


def scientific_software_v1() -> dict[str, str]:
    return {
        "iris": __version__,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
        "safetensors": version("safetensors"),
        "platform": platform.platform(),
        "torch_threads": str(torch.get_num_threads()),
        "torch_interop_threads": str(torch.get_num_interop_threads()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": str(torch.backends.mkldnn.enabled),
    }


type _ScientificRunOptimizer = IrisScientificOptimizerV1 | IrisScientificAblationOptimizerV1
type _ScientificRunOutput = (
    IrisScientificOptimizationOutputV1 | IrisScientificAblationOptimizationOutputV1
)


class _ScientificRunCore[OptimizerT: _ScientificRunOptimizer, OutputT: _ScientificRunOutput](ABC):
    """Private fixed-order cursor/runtime kernel; public variants fix all domains."""

    def __init__(
        self,
        *,
        optimizer: OptimizerT,
        examples: tuple[IrisExample, ...],
        terminal_token_id: int,
        seed: int,
        configuration_kind: str,
        configuration_domain: str,
        history_domain: str,
        configuration_extra: dict[str, object],
    ) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise IrisScientificRunError("An explicit bounded scientific run seed is required")
        optimizer._validate_contract()
        if optimizer._busy or optimizer.completed_steps or optimizer._optimizer.state:
            raise IrisScientificRunError("Run attachment requires a fresh idle optimizer")
        if type(examples) is not tuple or not 1 <= len(examples) <= 200000:
            raise IrisScientificRunError("An explicit bounded train corpus is required")
        rows = []
        row_fingerprints = []
        identities: set[str] = set()
        total_bytes = 0
        digest = sha256(b"iris/scientific-train-corpus/v1\0")
        for example in examples:
            if type(example) is not IrisExample:
                raise IrisScientificRunError("Exact scientific rows required")
            row = IrisExample.model_validate(example.model_dump(mode="python"))
            if row.split != "train" or row.id in identities:
                raise IrisScientificRunError("Train corpus must have unique train row identities")
            identities.add(row.id)
            encoded = canonical_json_bytes(row.model_dump(mode="json"))
            total_bytes += len(encoded)
            if total_bytes > 512 * 1024**2:
                raise IrisScientificRunError("Train corpus exceeds the logical byte budget")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            row_fingerprints.append(sha256(encoded).hexdigest())
            rows.append(row)
        self._optimizer = optimizer
        self._rows = tuple(rows)
        self._row_fingerprints = tuple(row_fingerprints)
        self._backbone = optimizer._model.teacher.backbone
        if (
            type(terminal_token_id) is not int
            or terminal_token_id not in self._backbone.eos_token_ids
        ):
            raise IrisScientificRunError("Explicit validated terminal token required")
        self._terminal_token_id = terminal_token_id
        self._backbone_identity = self._backbone.checkpoint_runtime_sha256
        self._backbone_guard = self._backbone.checkpoint_revision_token
        self._software = scientific_software_v1()
        self._history_domain = history_domain
        self._configuration_json = canonical_json_bytes(
            {
                "kind": configuration_kind,
                "ordering": "cyclic_fixed_corpus_order_v1",
                "seed": seed,
                "dataset_sha256": digest.hexdigest(),
                "dataset_examples": len(rows),
                "dataset_logical_bytes": total_bytes,
                "terminal_token_id": terminal_token_id,
                "optimizer_sha256": optimizer.configuration_sha256,
                "optimizer_policy": optimizer._policy.model_dump(mode="json"),
                "teacher_policy": optimizer._model.policy.model_dump(mode="json"),
                "backbone_runtime_sha256": self._backbone_identity,
                "software": self._software,
                **configuration_extra,
            }
        ).decode("utf-8")
        self._configuration_sha256 = domain_sha256(
            configuration_domain, self._configuration_json.encode("utf-8")
        )
        self._examples_seen = 0
        self._target_tokens_seen = 0
        self._completed_steps = 0
        self._history_sha256 = domain_sha256(
            history_domain, self._configuration_sha256.encode("ascii")
        )
        runtime = capture_runtime_state(optimizer._device)
        cpu_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        cuda_rng = None
        if optimizer._device.type == "cuda":
            cuda_rng = torch.Generator(device=optimizer._device).manual_seed(seed).get_state()
        self._runtime = replace(
            runtime,
            python_rng_state=random.Random(seed).getstate(),
            cpu_rng_state=cpu_rng.clone(),
            cuda_rng_state=cuda_rng,
        )
        self._poisoned = False
        self._busy = False
        self._initialize_owner()
        self._validate_boundary()

    def _initialize_owner(self) -> None:
        return None

    def _commit_owner(self) -> None:
        return None

    @abstractmethod
    def _execute_window(
        self, window: tuple[IrisExample, ...], target_counts: tuple[int, ...]
    ) -> OutputT:
        """Execute through the variant's exact typed optimizer."""

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def examples_seen(self) -> int:
        return self._examples_seen

    @property
    def target_tokens_seen(self) -> int:
        return self._target_tokens_seen

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def poisoned(self) -> bool:
        return self._poisoned or self._optimizer.poisoned

    def _validate_boundary(self) -> None:
        optimizer = self._optimizer
        if self.poisoned or optimizer._busy or optimizer.completed_steps != self._completed_steps:
            raise IrisScientificRunError("Scientific run is not at an owned clean step boundary")
        optimizer._validate_contract()
        if (
            any(p.grad is not None for p in optimizer._topology.parameters)
            or tuple(p._version for p in optimizer._topology.parameters) != optimizer._versions
            or self._backbone.checkpoint_revision_token != self._backbone_guard
            or scientific_software_v1() != self._software
        ):
            raise IrisScientificRunError(
                "Scientific run parameters/runtime changed outside ownership"
            )

    def _authorize_operation(self, operation: str, capability: object | None) -> None:
        """Variants may require explicit delegation without changing historical owners."""
        if capability is not None:
            raise IrisScientificRunError("Unexpected scientific run delegation")

    def step(self) -> OutputT:
        return self._step_authorized(None)

    def _step_authorized(self, capability: object | None) -> OutputT:
        # Gate in the shared kernel too: unbound core.step must not bypass a variant.
        self._authorize_operation("train", capability)
        if self._busy:
            raise IrisScientificRunError("Reentrant scientific run operations are forbidden")
        self._validate_boundary()
        if self._completed_steps >= self._optimizer._policy.max_optimizer_steps:
            raise IrisScientificRunError("Scientific run step budget is exhausted")
        caller = capture_runtime_state(self._optimizer._device)
        self._busy = True
        committed = False
        try:
            restore_runtime_state(self._runtime)
            policy = self._optimizer._policy
            window = []
            target_counts = []
            target_tokens = 0
            for offset in range(min(policy.maximum_examples, len(self._rows))):
                row = self._rows[(self._examples_seen + offset) % len(self._rows)]
                target = self._backbone.tokenize_teacher_target(
                    row.expected_response, terminal_token_id=self._terminal_token_id
                )
                count = int(target.attention_mask.sum())
                target_tokens += count
                target_counts.append(count)
                window.append(row)
                if target_tokens >= policy.minimum_target_tokens:
                    break
            if target_tokens < policy.minimum_target_tokens:
                raise IrisScientificRunError(
                    "Corpus/window cannot meet its token minimum without duplicates"
                )
            if target_tokens > policy.maximum_target_tokens:
                raise IrisScientificRunError("Selected window exceeds its target-token maximum")
            self._validate_boundary()
            result = self._execute_window(tuple(window), tuple(target_counts))
            committed = True
            if result.supervised_token_count != target_tokens or result.example_count != len(
                window
            ):
                raise IrisScientificRunError(
                    "Selected token window differs from the executed window"
                )
            self._examples_seen += result.example_count
            self._target_tokens_seen += result.supervised_token_count
            self._completed_steps = result.step_index
            self._history_sha256 = domain_sha256(
                self._history_domain,
                canonical_json_bytes(
                    {
                        "previous": self._history_sha256,
                        "result": asdict(result),
                        "examples_seen": self._examples_seen,
                        "target_tokens_seen": self._target_tokens_seen,
                    }
                ),
            )
            self._runtime = capture_runtime_state(self._optimizer._device)
            self._commit_owner()
            self._validate_boundary()
            return result
        except BaseException as error:
            if (
                committed
                or self._optimizer.completed_steps != self._completed_steps
                or self._optimizer.poisoned
            ):
                self._poisoned = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise IrisScientificRunError("Scientific run step could not complete") from None
        finally:
            try:
                restore_runtime_state(caller)
            except BaseException:
                self._poisoned = True
                raise IrisScientificRunError(
                    "Scientific caller runtime restoration failed"
                ) from None
            finally:
                self._busy = False


class IrisScientificRunV1(
    _ScientificRunCore[IrisScientificOptimizerV1, IrisScientificOptimizationOutputV1]
):
    """Historical exact scientific-D corpus/cursor/runtime owner.

    Public types, canonical configuration/history and checkpoint-facing fields
    remain historical. A distinct ablation optimizer cannot attach through this API.
    """

    def __init__(
        self,
        *,
        optimizer: IrisScientificOptimizerV1,
        examples: tuple[IrisExample, ...],
        terminal_token_id: int,
        seed: int,
    ) -> None:
        if type(optimizer) is not IrisScientificOptimizerV1:
            raise IrisScientificRunError("Exact scientific optimizer required")
        super().__init__(
            optimizer=optimizer,
            examples=examples,
            terminal_token_id=terminal_token_id,
            seed=seed,
            configuration_kind="iris-scientific-run-v1",
            configuration_domain="iris/scientific-run-configuration/v1",
            history_domain="iris/scientific-run-history/v1",
            configuration_extra={},
        )

    def _execute_window(
        self, window: tuple[IrisExample, ...], target_counts: tuple[int, ...]
    ) -> IrisScientificOptimizationOutputV1:
        return self._optimizer.step(window, terminal_token_id=self._terminal_token_id)

    def step(self) -> IrisScientificOptimizationOutputV1:
        return super().step()


__all__ = ["IrisScientificRunError", "IrisScientificRunV1"]
