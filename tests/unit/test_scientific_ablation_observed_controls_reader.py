"""Independent observed-stage budget guard; metadata only, no model forwards."""

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    build_scientific_ablation_causal_observed_budget_v1,
)
from tests.unit.test_scientific_ablation_causal_design import Protocol, build

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "observed_controls_reader_test",
    ROOT / "scripts/scientific_ablation_observed_budget_guard.py",
)
assert SPEC is not None and SPEC.loader is not None
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def _budget(protocol: Protocol = "workspace") -> dict[str, Any]:
    return build_scientific_ablation_causal_observed_budget_v1(
        build(protocol), maximum_observation_scalars=168
    ).model_dump(mode="json")


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    value["sha256"] = reader.domain(reader.budgets.DOMAIN, value["payload"])
    return value


@pytest.mark.parametrize("protocol", ("workspace", "message"))
def test_original_declared_budget_metadata_guard_accepts_without_relabeling(
    protocol: Protocol,
) -> None:
    value = _budget(protocol)
    before = copy.deepcopy(value)
    reader.checked_budget(value)
    assert value == before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", "foreign-budget-kind"),
        ("scope", "full-scientific-acceptance"),
        ("scientific_acceptance", "assessed"),
    ),
)
def test_fully_rehashed_foreign_kind_scope_assessment_is_not_budget_evidence(
    field: str, value: str
) -> None:
    raw = _budget()
    raw["payload"][field] = value
    with pytest.raises(ValueError):
        reader.checked_budget(_rehash(raw))


@pytest.mark.parametrize("where", ("root", "payload"))
def test_hidden_root_or_payload_metadata_is_rejected_after_rehash(where: str) -> None:
    raw = _budget()
    (raw if where == "root" else raw["payload"])["foreign"] = "hidden"
    with pytest.raises(ValueError):
        reader.checked_budget(_rehash(raw))


@pytest.mark.parametrize(
    "field",
    (
        "planned_ordinary_teacher_executions",
        "planned_controlled_teacher_executions",
        "planned_total_teacher_executions",
        "planned_observed_teacher_executions",
        "observation_scalar_bound_per_pass",
        "total_observation_scalar_bound",
        "original_trace_proposal_scalar_bound",
        "total_trace_proposal_observation_scalar_bound",
    ),
)
def test_each_fully_rehashed_numeric_count_is_independently_rederived(field: str) -> None:
    raw = _budget()
    raw["payload"][field] += 1
    with pytest.raises(ValueError):
        reader.checked_budget(_rehash(raw))


def test_boolean_observation_cap_is_not_an_exact_integer_budget() -> None:
    raw = _budget()
    raw["payload"]["maximum_observation_scalars"] = True
    with pytest.raises(ValueError):
        reader.checked_budget(_rehash(raw))


def test_budget_metadata_byte_limit_includes_the_canonical_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _budget()
    monkeypatch.setattr(reader, "canonical", lambda value: b"x" * reader.MAX_BYTES)
    with pytest.raises(ValueError, match="bounded"):
        reader.checked_budget(raw)
