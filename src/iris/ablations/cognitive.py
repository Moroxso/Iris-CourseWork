"""Call-owned differentiable prompt-routing, goal and broadcast alternatives."""

import re
from collections.abc import Callable
from typing import TypeVar

import torch

from iris.ablations.policy import (
    CognitiveAblationPolicyV1,
    cognitive_ablation_policy_sha256_v1,
    validate_cognitive_ablation_policy_v1,
)
from iris.cognitive.module_input_router import ModuleInputRouter
from iris.cognitive.outputs import GoalEncoderOutput, ModuleInputRouterOutput
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.workspace.outputs import WorkspaceReaderOutput

_T = TypeVar("_T")


class CognitiveAblationSessionV1:
    """Single ordered microbatch-one session; no model hooks/parameter mutation.

    Random soft attention uses independent standard-normal logits over attended
    positions, keyed only by explicit seed and prompt hash. CPU private generator;
    no global RNG, target dependence, row-order or padding-length dependence.
    Disabled broadcast removes module reads, NOT the final workspace-to-LM bridge.
    """

    def __init__(
        self,
        policy: CognitiveAblationPolicyV1,
        *,
        prompt_sha256: str,
        num_modules: int,
        num_slots: int,
        state_dim: int,
        cycles: int,
    ) -> None:
        self._policy = validate_cognitive_ablation_policy_v1(policy)
        if (
            not isinstance(prompt_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256) is None
        ):
            raise ValueError("Explicit prompt-only digest required")
        if any(type(v) is not int or v < 1 for v in (num_modules, num_slots, state_dim, cycles)):
            raise ValueError("Exact positive ablation geometry required")
        self.policy_sha256 = cognitive_ablation_policy_sha256_v1(self._policy)
        self.prompt_sha256 = prompt_sha256
        self.modules, self.slots, self.width, self.cycles = (
            num_modules,
            num_slots,
            state_dim,
            cycles,
        )
        self._phase = 0
        self._failed = self._finished = False
        self._length: int | None = None

    def _validate_start(
        self,
        *,
        prompt_sha256: str,
        num_modules: int,
        num_slots: int,
        state_dim: int,
        cycles: int,
    ) -> None:
        """Fresh exact session binding before request effects; NOT authorization."""
        if type(self) is not CognitiveAblationSessionV1 or set(vars(self)) != {
            "_policy",
            "policy_sha256",
            "prompt_sha256",
            "modules",
            "slots",
            "width",
            "cycles",
            "_phase",
            "_failed",
            "_finished",
            "_length",
        }:
            raise ValueError("Exact complete cognitive session required")
        if (
            type(self._policy) is not CognitiveAblationPolicyV1
            or set(vars(self._policy)) != set(CognitiveAblationPolicyV1.model_fields)
            or self._policy.__pydantic_extra__ is not None
            or type(self.prompt_sha256) is not str
            or self.prompt_sha256 != prompt_sha256
            or type(self.policy_sha256) is not str
            or self.policy_sha256 != cognitive_ablation_policy_sha256_v1(self._policy)
            or any(
                type(value) is not int
                for value in (self.modules, self.slots, self.width, self.cycles)
            )
            or (self.modules, self.slots, self.width, self.cycles)
            != (num_modules, num_slots, state_dim, cycles)
            or type(self._phase) is not int
            or self._phase != 0
            or self._failed is not False
            or self._finished is not False
            or self._length is not None
        ):
            raise ValueError("Cognitive session prompt/policy/geometry/freshness binding differs")

    def _step(self, expected: int, action: Callable[[], _T]) -> _T:
        if self._failed or self._finished or self._phase != expected:
            self._failed = True
            raise ValueError("Cognitive ablation phases are closed and strictly ordered")
        try:
            result = action()
            self._phase += 1
            return result
        except BaseException:
            self._failed = True
            raise

    @staticmethod
    def _floating(value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != shape
            or not value.is_floating_point()
            or value.layout != torch.strided
            or value.device.type not in {"cpu", "cuda"}
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Invalid cognitive ablation tensor boundary")

    def after_goal(self, goal: GoalEncoderOutput) -> GoalEncoderOutput:
        def apply() -> GoalEncoderOutput:
            if type(goal) is not GoalEncoderOutput or goal.attention_weights.ndim != 2:
                raise ValueError("Exact goal output required")
            self._floating(goal.goal_state, (1, self.width))
            self._length = goal.attention_weights.shape[1]
            self._floating(goal.attention_weights, (1, self._length))
            if self._length < 1 or goal.attention_weights.device != goal.goal_state.device:
                raise ValueError("Goal attention geometry/device differs")
            if self._policy.goal == "enabled":
                return goal
            return GoalEncoderOutput(
                goal_state=torch.zeros_like(goal.goal_state),
                attention_weights=torch.zeros_like(goal.attention_weights),
            )

        return self._step(0, apply)

    def route(
        self, router: ModuleInputRouter, context: torch.Tensor, attention_mask: torch.Tensor
    ) -> ModuleInputRouterOutput:
        def apply() -> ModuleInputRouterOutput:
            if type(router) is not ModuleInputRouter:
                raise ValueError("Exact canonical module-input router required")
            batch, length, dtype, device = router._validate_context(context)
            if (batch, length, router.num_modules, router.state_dim) != (
                1,
                self._length,
                self.modules,
                self.width,
            ):
                raise ValueError("Ablation routing geometry differs from the source graph")
            if (
                3 * self.modules * length + self.modules * self.width
                > self._policy.maximum_routing_scalars
            ):
                raise ValueError("Ablation routing scalar budget exceeded before allocation")
            mask = router._validate_attention_mask(
                attention_mask, batch_size=batch, sequence_length=length, device=device
            )
            router._validate_parameter_boundary(dtype=dtype, device=device)
            if self._policy.routing == "learned":
                output = router(context, attention_mask)
                if type(output) is not ModuleInputRouterOutput:
                    raise ValueError("Canonical routing output type differs")
                return output
            if context.is_inference() and not torch.is_inference_mode_enabled():
                owned_context = context.detach().clone()
            else:
                owned_context = context
            accumulation = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
            with torch.autocast(device_type=device.type, enabled=False):
                safe = owned_context.masked_fill(~mask.unsqueeze(-1), 0).to(dtype=accumulation)
                if self._policy.routing == "uniform":
                    weights = (mask.to(dtype=accumulation) / mask.sum(-1, keepdim=True)).unsqueeze(
                        1
                    )
                    weights = weights.expand(1, self.modules, length).clone()
                else:
                    key = domain_sha256(
                        "iris/cognitive-ablation-random-routing/v1",
                        canonical_json_bytes(
                            {"seed": self._policy.routing_seed, "prompt_sha256": self.prompt_sha256}
                        ),
                    )
                    generator = torch.Generator(device="cpu").manual_seed(int(key[:16], 16) % 2**63)
                    attended = mask[0].nonzero().flatten()
                    logits = torch.randn(
                        (1, self.modules, attended.numel()),
                        generator=generator,
                        dtype=torch.float64,
                        device="cpu",
                    )
                    selected = torch.softmax(logits.to(device=device, dtype=accumulation), dim=-1)
                    weights = torch.zeros(
                        (1, self.modules, length), dtype=accumulation, device=device
                    )
                    weights = weights.index_copy(-1, attended, selected)
                inputs = torch.einsum("bmt,btd->bmd", weights, safe).to(dtype=dtype)
            router._validate_output(
                inputs,
                weights,
                mask,
                batch_size=1,
                sequence_length=length,
                dtype=dtype,
                device=device,
            )
            return ModuleInputRouterOutput(module_inputs=inputs, attention_weights=weights)

        return self._step(1, apply)

    def after_read(self, reads: WorkspaceReaderOutput, cycle: int) -> WorkspaceReaderOutput:
        def apply() -> WorkspaceReaderOutput:
            if type(cycle) is not int or not 1 <= cycle <= self.cycles:
                raise ValueError("Exact in-domain read cycle required")
            if type(reads) is not WorkspaceReaderOutput:
                raise ValueError("Exact workspace reader output required")
            self._floating(reads.workspace_reads, (1, self.modules, self.width))
            self._floating(reads.attention_weights, (1, self.modules, self.slots))
            if reads.workspace_reads.device != reads.attention_weights.device:
                raise ValueError("Read output devices differ")
            if self._policy.broadcast == "enabled":
                return reads
            return WorkspaceReaderOutput(
                workspace_reads=torch.zeros_like(reads.workspace_reads),
                attention_weights=torch.zeros_like(reads.attention_weights),
            )

        if type(cycle) is not int:
            self._failed = True
            raise ValueError("Exact read cycle required")
        return self._step(cycle + 1, apply)

    def finish(self) -> None:
        def apply() -> None:
            self._finished = True

        self._step(self.cycles + 2, apply)
