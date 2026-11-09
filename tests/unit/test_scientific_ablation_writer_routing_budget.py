"""Original fixed wide quota; production defaults/early denial remain unchanged."""

from unittest.mock import patch

import pytest
import torch

from iris.ablations import CognitiveAblationPolicyV1, CognitiveAblationSessionV1
from iris.cognitive import ModuleInputRouter
from iris.cognitive.outputs import GoalEncoderOutput
from tests.integration.scientific_ablation_writer_geometry_fixture import (
    Geometry,
    routing_scalar_budget,
)


@pytest.mark.parametrize(
    "geometry",
    (Geometry(), Geometry(slots=4, cycles=5, empty_threshold=1.0), Geometry(empty_threshold=4.0)),
)
def test_all_existing_small_original_policies_keep_default_quota(geometry: Geometry) -> None:
    assert (
        routing_scalar_budget(geometry, 768)
        == CognitiveAblationPolicyV1().maximum_routing_scalars
        == 65536
    )
    assert (
        CognitiveAblationPolicyV1(maximum_routing_scalars=routing_scalar_budget(geometry, 768))
        == CognitiveAblationPolicyV1()
    )


def test_wide_reservation_covers_entire_original_cap_not_observed_first_failure() -> None:
    geometry = Geometry(modules=64, width=1, cycles=2)
    quota = routing_scalar_budget(geometry, 768)
    assert quota == 147520
    assert all(3 * 64 * length + 64 <= quota for length in range(1, 769))
    assert quota < 3 * 64 * 769 + 64
    assert 3 * 64 * 349 + 64 == 67072 > 65536


@pytest.mark.parametrize("cap", (0, -1, True, 1.0))
def test_invalid_original_prompt_cap_cannot_be_coerced(cap: object) -> None:
    with pytest.raises(ValueError):
        routing_scalar_budget(Geometry(modules=64, width=1), cap)


def session(length: int, quota: int) -> CognitiveAblationSessionV1:
    value = CognitiveAblationSessionV1(
        CognitiveAblationPolicyV1(maximum_routing_scalars=quota),
        prompt_sha256="1" * 64,
        num_modules=64,
        num_slots=1,
        state_dim=1,
        cycles=2,
    )
    value.after_goal(
        GoalEncoderOutput(
            goal_state=torch.ones(1, 1, dtype=torch.float32),
            attention_weights=torch.ones(1, length, dtype=torch.float32) / length,
        )
    )
    return value


@pytest.mark.parametrize("length,quota", ((342, 65536), (349, 65536), (769, 147520)))
def test_real_native_guard_still_denies_before_router_allocation(length: int, quota: int) -> None:
    router = ModuleInputRouter(state_dim=1, num_modules=64)
    value = session(length, quota)
    with patch.object(
        router, "forward", side_effect=AssertionError("No over-budget routing")
    ) as call:
        with pytest.raises(ValueError, match="scalar budget exceeded before allocation"):
            value.route(
                router,
                torch.ones(1, length, 1, dtype=torch.float32),
                torch.ones(1, length, dtype=torch.int64),
            )
        call.assert_not_called()


@pytest.mark.parametrize("length,quota", ((341, 65536), (768, 147520)))
def test_real_native_route_accepts_exact_original_boundary(length: int, quota: int) -> None:
    router = ModuleInputRouter(state_dim=1, num_modules=64)
    value = session(length, quota)
    before = torch.get_rng_state().clone()
    with patch.object(router, "forward", wraps=router.forward) as call:
        output = value.route(
            router,
            torch.ones(1, length, 1, dtype=torch.float32),
            torch.ones(1, length, dtype=torch.int64),
        )
        assert call.call_count == 1
    assert output.module_inputs.shape == (1, 64, 1)
    assert output.attention_weights.shape == (1, 64, length)
    assert torch.equal(torch.get_rng_state(), before)
