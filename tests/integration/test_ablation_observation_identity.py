"""Exact identity primitives must reject before overloaded comparison executes.

New regression module: intentionally NOT in the already-running frozen711 scope.
The fixture uses project-authored offline tiny evaluation, not candidate Python.
"""

from dataclasses import replace
from typing import Any

import pytest

from iris.ablations.policy import CognitiveAblationPolicyV1
from iris.training.ablation_observation import IrisAblationObservedTeacherOutputV1
from tests.integration.test_cognitive_ablation_controls import _control, _options
from tests.integration.test_intervened_teacher import prepared
from tests.integration.test_scientific_teacher import _build

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def observed() -> IrisAblationObservedTeacherOutputV1:
    model, _, _ = _build()
    model.eval()
    return model.teacher.evaluation_forward_ablation_observed(
        prepared(model),
        plan=_control(CognitiveAblationPolicyV1()),
        diagnostics=_options(),
        maximum_observation_scalars=336,
    )


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "invocation_sha256",
        "controlled_kind",
        "controlled_plan_sha256",
        "controlled_ablation_policy_sha256",
        "controlled_control_plan_sha256",
        "controlled_prompt_sha256",
        "controlled_target_sha256",
        "controlled_teacher_graph_sha256",
        "controlled_sidecar_state_sha256",
        "controlled_invocation_sha256",
        "before_write_sha256",
        "native_written_sha256",
    ),
)
@pytest.mark.parametrize("payload_type", ("object", "str-subclass"))
def test_exact_identity_rejects_foreign_primitives_without_comparing(
    observed: IrisAblationObservedTeacherOutputV1,
    field: str,
    payload_type: str,
) -> None:
    calls: list[str] = []

    class EqualityObject:
        def __ne__(self, other: object) -> bool:
            calls.append("object comparison")
            return False

    class EqualityString(str):
        def __ne__(self, other: object) -> bool:
            calls.append("str-subclass comparison")
            return bool(super().__ne__(other))

    original = (
        getattr(observed.controlled, field.removeprefix("controlled_"))
        if field.startswith("controlled_")
        else getattr(observed.observations[0], field)
        if field.endswith("written_sha256") or field == "before_write_sha256"
        else getattr(observed, field)
    )
    payload: Any = EqualityObject() if payload_type == "object" else EqualityString(original)
    # Controlled V1 is a trusted child here; bypass its older constructor and
    # require the new observed-root guard to reject before delegating validation.
    if field.startswith("controlled_"):
        child = replace(observed.controlled)
        object.__setattr__(child, field.removeprefix("controlled_"), payload)
    with pytest.raises(ValueError):
        if field.startswith("controlled_"):
            replace(observed, controlled=child)
        elif field in ("before_write_sha256", "native_written_sha256"):
            phase = replace(observed.observations[0], **{field: payload})
            replace(observed, observations=(phase, *observed.observations[1:]))
        else:
            replace(observed, **{field: payload})
    assert not calls, "Untrusted comparison must not run before strict type rejection"
