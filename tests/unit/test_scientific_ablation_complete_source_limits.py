"""Complete transaction quotas are exact data, not foreign executable objects."""

from typing import Any

import pytest

from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
    _complete_limits,
)


def test_complete_limits_are_owned_without_source_metadata_or_callbacks() -> None:
    limits = ScientificAblationCausalCompleteLimitsV1()
    owned = _complete_limits(limits)
    assert owned == limits and owned is not limits
    assert not hasattr(owned, "__dict__")
    with pytest.raises(AttributeError):
        object.__setattr__(limits, "callback", lambda: None)


@pytest.mark.parametrize("field", ("maximum_observation_scalars", "maximum_report_bytes"))
@pytest.mark.parametrize("value", (True, False, 0, -1, 1.0, "1", None, 2**63))
def test_constructed_bad_complete_limits_reject_exact_types(field: str, value: Any) -> None:
    limits = ScientificAblationCausalCompleteLimitsV1()
    object.__setattr__(limits, field, value)
    with pytest.raises(ValueError):
        _complete_limits(limits)


def test_foreign_complete_limits_are_denied_before_their_attributes() -> None:
    class Foreign(ScientificAblationCausalCompleteLimitsV1):
        def __getattribute__(self, name: str) -> Any:
            pytest.fail("Foreign quotas must not execute any attributes")

    with pytest.raises(ValueError):
        _complete_limits(Foreign())
