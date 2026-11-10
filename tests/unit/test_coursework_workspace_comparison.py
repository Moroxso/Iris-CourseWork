"""The paired coursework comparison reports regressions as well as wins."""

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest


def comparison(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("coursework_workspace_comparison")


def test_cases_are_balanced_and_long_prompts_fit_token_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = comparison(monkeypatch)

    def count(text: str) -> int:
        return len(text.encode("utf-8")) // 5 + 16

    rows = module.prepare("full", count, len)
    smoke = module.prepare("smoke", count, len)
    assert len(rows) == len({r["id"] for r in rows}) == 96
    assert len(smoke) == 12 and all(row in rows for row in smoke)
    for lang in ("ru", "en", "zh"):
        examples = [r for r in rows if r["language"] == lang]
        assert len(examples) == 32
        assert {r["family"] for r in examples} == set(module.FAMILIES)
        for r in examples:
            if r["target_band"]:
                low, high = r["target_band"]
                assert low <= r["prompt_tokens"] <= high
            if r["family"] == "batch-extract":
                assert len(r["expected"]) == len(set(r["expected"])) == 24
                assert r["reference_tokens"] > 32


def test_multilingual_scoring_separates_format_and_partial_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grade = comparison(monkeypatch).grade
    assert grade("「灯塔。」", "灯塔")["exact"]
    full_width = "".join(chr(ord(c) + 0xFEE0) for c in "BEACON")
    assert grade(full_width + "。", "beacon")["exact"]
    assert not grade("The answer is beacon", "beacon")["exact"]
    assert grade('["S1", "S3", "S2"]', ["S1", "S2", "S3"]) == dict(
        exact=False, matched=1, total=3, format_valid=True
    )
    assert grade('{"room":"R4", "extra":"ignored"}', {"room": "R4"}) == dict(
        exact=False, matched=1, total=1, format_valid=False
    )
    assert not grade('{"room":"R5", "room":"R4"}', {"room": "R4"})["format_valid"]
    assert not grade('before ["S1"]', ["S1"])["format_valid"]


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


def test_math_code_suite_and_all_trusted_references(monkeypatch: pytest.MonkeyPatch) -> None:
    module = comparison(monkeypatch)
    full = module.prepare("full", len, len, "math-code")
    smoke = module.prepare("smoke", len, len, "math-code")
    assert len(full) == len({row["id"] for row in full}) == 32
    assert len(smoke) == 8 and all(row in full for row in smoke)
    assert {row["grading"] for row in full} == {"function", "number"}
    for row in full:
        result = module.grade_case(row["reference"], row)
        assert result["exact"] and result["matched"] == result["total"]
        if row["grading"] == "function":
            assert result["total"] >= 5


def test_number_grading_preserves_sign_and_exactness(monkeypatch: pytest.MonkeyPatch) -> None:
    grade = comparison(monkeypatch).grade_case
    task = dict(grading="number", expected="-1/2")
    assert grade("-2/4", task)["exact"]
    assert grade("-0,5", task)["exact"]
    assert not grade("0.5", task)["exact"]
    assert not grade("-0.4999", task)["exact"]
    assert grade("-1/0", task)["status"] == "not-a-number"
    assert grade("x = -1/2", task)["status"] == "not-a-number"
    label = "Ответ:"
    explained = grade(f"Пояснение.\n\n{label} $$\n\\boxed{{-\\frac{{1}}{{2}}}}\n$$", task)
    assert not explained["exact"] and explained["numeric_value_correct"] is True
    assert grade(f"Пояснение.\n{label} $-0.5$.", task)["numeric_value_correct"] is True
    assert grade(f"Пояснение.\n{label} $0.5$.", task)["numeric_value_correct"] is False
    assert grade("Сначала получаем -0.5.", task)["numeric_value_correct"] is None


def test_integer_function_grading_is_bounded_and_reports_partial_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grade = comparison(monkeypatch).grade_case
    task = dict(
        grading="function",
        expected=dict(
            name="f",
            parameters=["x"],
            checks=[dict(args=[-2], value=2), dict(args=[1], value=1)],
        ),
    )
    assert grade("def f(x):\n    return abs(x)", task)["exact"]
    source = "def f(x: int) -> int:\n    if x < 0:\n        return -x\n    return x"
    fence = "\x60" * 3
    assert grade(f"{fence}python\n{source}\n{fence}", task)["exact"]
    assert grade("def f(x): return x", task)["matched"] == 1
    assert grade("def f(x): return True", task)["matched"] == 0
    assert grade("def f(x): return 1 // (x + 2)", task)["outcomes"][0] == "evaluation-error"
    assert grade("def f(x): return x ** 99", task)["status"] == "unsupported"
    assert grade("def f(x): return x.real", task)["status"] == "unsupported"
    assert grade("def f(x):\n    import math\n    return abs(x)", task)["status"] == "unsupported"
    assert grade("def f(x) return x", task)["status"] == "syntax"


def test_report_breaks_down_scores_and_keeps_text_inert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = comparison(monkeypatch)
    answer = "<script>example</script>"
    length_band = "257\u2013640"
    grades = {arm: module.grade(answer, ["K01", "K02"]) for arm in module.ARMS}
    row = dict(
        id="zh-batch-extract-4",
        language="zh",
        family="batch-extract",
        length_band=length_band,
        prompt="提取编号",
        prompt_tokens=418,
        expected=["K01", "K02"],
        grades=grades,
        correct={arm: False for arm in module.ARMS},
        answers=dict.fromkeys(module.ARMS, answer),
        token_ids={arm: [[1]] for arm in module.ARMS},
        stop_reasons=dict.fromkeys(module.ARMS, "max_new_tokens"),
    )
    module.write_report(tmp_path, [row], "smoke")
    report = json.loads((tmp_path / "report.json").read_text("utf-8"))
    assert report["groups"]["language"]["zh"]["hit_token_limit"]["iris"] == 1
    assert report["groups"]["family"]["batch-extract"]["count"] == 1
    assert report["groups"]["length_band"][length_band]["exact"]["qwen"] == 0
    html = (tmp_path / "index.html").read_text("utf-8")
    assert answer not in html and "&lt;script&gt;example&lt;/script&gt;" in html
