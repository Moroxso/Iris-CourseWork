"""Strict actual score-store discovery must retain its known owner lock."""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

# The standalone reader is checked explicitly in the strict gate; a dynamic
# script import avoids assigning that same source a second static module name.
reader = import_module("scripts.verify_scientific_ablation_observation")

PLAN = "a" * 64


@pytest.mark.parametrize(
    "layout",
    (
        "valid",
        "missing-lock",
        "missing-plan",
        "extra-file",
        "extra-directory",
        "plan-is-file",
        "lock-is-directory",
        "foreign-plan",
        "uppercase-expectation",
        "object-expectation",
        "subclass-expectation",
    ),
)
def test_score_store_has_only_the_exact_plan_and_regular_lock(tmp_path: Path, layout: str) -> None:
    store = tmp_path / "scores/scientific-ablation-evaluation-ledger-v2"
    store.mkdir(parents=True)
    if layout != "missing-plan":
        plan = store / ("b" * 64 if layout == "foreign-plan" else PLAN)
        if layout == "plan-is-file":
            plan.write_bytes(b"")
        else:
            plan.mkdir()
    if layout != "missing-lock":
        lock = store / ".iris-owner.lock"
        if layout == "lock-is-directory":
            lock.mkdir()
        else:
            lock.write_bytes(b"")
    if layout == "extra-file":
        (store / "extra").write_bytes(b"")
    if layout == "extra-directory":
        (store / "extra").mkdir()

    class ForeignString(str):
        pass

    expected: Any = (
        object()
        if layout == "object-expectation"
        else ForeignString(PLAN)
        if layout == "subclass-expectation"
        else PLAN.upper()
        if layout == "uppercase-expectation"
        else PLAN
    )
    if layout == "valid":
        assert reader.score_chain_directory(tmp_path, expected) == store / PLAN
        assert (store / ".iris-owner.lock").is_file()
    else:
        with pytest.raises(ValueError):
            reader.score_chain_directory(tmp_path, expected)
