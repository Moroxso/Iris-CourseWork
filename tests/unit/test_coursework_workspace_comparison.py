"""The paired coursework comparison reports regressions as well as wins."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest


def comparison(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("coursework_workspace_comparison")


def test_cases_have_two_distinct_expected_answers_per_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = comparison(monkeypatch).cases()
    assert len(rows) == len({r["id"] for r in rows}) == 8
    for family in {r["family"] for r in rows}:
        examples = [r for r in rows if r["family"] == family]
        assert len(examples) == len({r["expected"] for r in examples}) == 2


def test_paired_score_keeps_wins_losses_and_changed_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = comparison(monkeypatch)
    rows = [
        dict(
            correct=module.score(dict(qwen="no", iris="YES!", empty_workspace="no"), "yes"),
            token_ids=dict(iris=[[1]], empty_workspace=[[2]]),
        ),
        dict(
            correct=module.score(dict(qwen="yes", iris="no", empty_workspace="no"), "yes"),
            token_ids=dict(iris=[[2]], empty_workspace=[[2]]),
        ),
    ]
    result = module.summarize(rows)
    assert result["exact"] == dict(qwen=1, iris=1, empty_workspace=0)
    assert result["paired"]["qwen"] == dict(iris_only=1, control_only=1, both=0, neither=0)
    assert result["changed_after_clearing"] == 1
