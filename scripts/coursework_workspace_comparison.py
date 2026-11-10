# ruff: noqa: E402, E501, RUF001
"""Paired coursework checks: Qwen, empty workspace, and full Iris."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from coursework_code_grade import grade_function
from coursework_cognitive_tests import normalize
from coursework_math_code_cases import FAMILIES as MATH_CODE_FAMILIES
from coursework_math_code_cases import cases as math_code_cases
from coursework_math_code_cases import grade_number
from coursework_multilingual_cases import FAMILIES, LANGUAGES, OUTPUT_BUDGET, cases
from coursework_trained_demo import GeneratePrompt
from coursework_trained_demo import main as trained_main
from coursework_visuals import VISUAL_STYLE

ARMS = {
    "qwen": "Qwen без вставки",
    "empty_workspace": "Iris: пустые слоты",
    "iris": "Iris: рабочее пространство",
}
BUDGET = OUTPUT_BUDGET
Expected = str | list[str] | dict[str, str]


def word(text: str) -> str:
    return normalize(unicodedata.normalize("NFKC", text))


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return result


def grade(answer: str, expected: Expected) -> dict[str, Any]:
    if isinstance(expected, str):
        exact = word(answer) == word(expected)
        return dict(exact=exact, matched=int(exact), total=1, format_valid=None)
    try:
        actual = json.loads(answer, object_pairs_hook=unique_keys)
    except (ValueError, RecursionError):
        return dict(exact=False, matched=0, total=len(expected), format_valid=False)
    matched = 0
    valid = False
    if isinstance(expected, list) and isinstance(actual, list):
        valid = len(actual) == len(expected) and all(isinstance(x, str) for x in actual)
        matched = sum(isinstance(a, str) and a == e for a, e in zip(actual, expected, strict=False))
    elif isinstance(expected, dict) and isinstance(actual, dict):
        valid = actual.keys() == expected.keys() and all(
            isinstance(x, str) for x in actual.values()
        )
        matched = sum(actual.get(k) == e for k, e in expected.items())
    return dict(
        exact=valid and matched == len(expected),
        matched=matched,
        total=len(expected),
        format_valid=valid,
    )


def score(answers: dict[str, str], expected: Expected) -> dict[str, bool]:
    if set(answers) != set(ARMS):
        raise ValueError("All three comparison arms are required")
    return {arm: bool(grade(answer, expected)["exact"]) for arm, answer in answers.items()}


def grade_case(answer: str, row: dict[str, Any]) -> dict[str, Any]:
    kind = row.get("grading")
    if kind == "number":
        return grade_number(answer, row["expected"])
    if kind == "function":
        return grade_function(answer, row["expected"])
    return grade(answer, row["expected"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = {}
    for control in ("qwen", "empty_workspace"):
        paired[control] = dict(
            iris_only=sum(row["correct"]["iris"] and not row["correct"][control] for row in rows),
            control_only=sum(
                not row["correct"]["iris"] and row["correct"][control] for row in rows
            ),
            both=sum(row["correct"]["iris"] and row["correct"][control] for row in rows),
            neither=sum(not row["correct"]["iris"] and not row["correct"][control] for row in rows),
        )
    return dict(
        count=len(rows),
        exact={arm: sum(row["correct"][arm] for row in rows) for arm in ARMS},
        paired=paired,
        changed_after_clearing=sum(
            row["token_ids"]["iris"] != row["token_ids"]["empty_workspace"] for row in rows
        ),
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(rows)
    result.update(
        mean_component_accuracy={
            arm: sum(row["grades"][arm]["matched"] / row["grades"][arm]["total"] for row in rows)
            / len(rows)
            for arm in ARMS
        },
        hit_token_limit={
            arm: sum(row["stop_reasons"][arm] == "max_new_tokens" for row in rows) for arm in ARMS
        },
    )
    numeric = [row for row in rows if row.get("grading") == "number"]
    if numeric:
        result["numeric_value"] = {
            arm: dict(
                correct=sum(
                    row["grades"][arm].get("numeric_value_correct") is True for row in numeric
                ),
                unrecognized=sum(
                    row["grades"][arm].get("numeric_value_correct") is None for row in numeric
                ),
                total=len(numeric),
            )
            for arm in ARMS
        }
    return result


def breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        value: aggregate([row for row in rows if row[key] == value])
        for value in dict.fromkeys(row[key] for row in rows)
    }


def write_report(
    out: Path,
    rows: list[dict[str, Any]],
    profile: str,
    suite: str = "multilingual",
    *,
    rescored: bool = False,
) -> None:
    summary = aggregate(rows)
    groups = {k: breakdown(rows, k) for k in ("language", "family", "length_band")}
    (out / ("report-numeric.json" if rescored else "report.json")).write_text(
        json.dumps(
            dict(
                kind="iris-coursework-workspace-comparison-v2",
                profile=profile,
                suite=suite,
                rescored=rescored,
                summary=summary,
                groups=groups,
                rows=rows,
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    cards = "".join(
        f'<section class="visual-panel"><h2>{escape(ARMS[arm])}</h2>'
        f'<meter min="0" max="{summary["count"]}" value="{summary["exact"][arm]}"></meter>'
        f"<p>{summary['exact'][arm]} из {summary['count']} точных ответов</p></section>"
        for arm in ARMS
    )
    pairs = "".join(
        f"<li>Против «{escape(ARMS[arm])}»: только полный Iris — {item['iris_only']}, "
        f"только контроль — {item['control_only']}, оба — {item['both']}, "
        f"ни один — {item['neither']}.</li>"
        for arm, item in summary["paired"].items()
    )
    numeric = ""
    if summary.get("numeric_value"):
        numeric = (
            "<h2>Математика: численно верный итог</h2><ul>"
            + "".join(
                f"<li>{escape(ARMS[arm])}: {item['correct']} из {item['total']}; "
                f"итог не распознан в {item['unrecognized']}.</li>"
                for arm, item in summary["numeric_value"].items()
            )
            + "</ul><p>Дополнительный критерий принимает явно отмеченный итог, даже если модель нарушила формат.</p>"
        )
        if rescored:
            numeric += (
                "<p>Критерий добавлен после просмотра первых ответов; генерация не повторялась.</p>"
            )
    tables = "".join(
        f"<h2>{title}</h2><table><thead><tr><th>Группа</th><th>Заданий</th>"
        + "".join(f"<th>{escape(name)}</th>" for name in ARMS.values())
        + "</tr></thead><tbody>"
        + "".join(
            f"<tr><th>{escape(labels.get(key, key))}</th><td>{group['count']}</td>"
            + "".join(
                f"<td>{group['exact'][arm]} · {group['mean_component_accuracy'][arm]:.0%}"
                f" · {group['hit_token_limit'][arm]}</td>"
                for arm in ARMS
            )
            + "</tr>"
            for key, group in groups[axis].items()
        )
        + "</tbody></table>"
        for axis, title, labels in (
            ("language", "По языкам", LANGUAGES),
            ("family", "По типам задач", FAMILIES | MATH_CODE_FAMILIES),
            ("length_band", "По длине входа, токены", {}),
        )
    )
    details = "".join(
        f'<details class="visual-panel"><summary>{escape(row["id"])} · вход {row["prompt_tokens"]} токенов'
        f"</summary><p>Эталон: {escape(row.get('reference') or json.dumps(row['expected'], ensure_ascii=False))}</p>"
        f"<pre>{escape(row['prompt'])}</pre><ul>"
        + "".join(
            f"<li>{escape(ARMS[arm])}: {escape(row['answers'][arm])} "
            f" ({row['grades'][arm]['matched']}/{row['grades'][arm]['total']} проверок;"
            + (
                f" {escape(row['grades'][arm]['status'])};"
                if "status" in row["grades"][arm]
                else ""
            )
            + f" {len(row['token_ids'][arm][0])} токенов ответа)</li>"
            for arm in ARMS
        )
        + f'</ul><a href="{escape(row["id"])}/viewer.html">Граф полного Iris</a></details>'
        for row in rows
    )
    (out / ("index-numeric.html" if rescored else "index.html")).write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Сравнение Qwen и Iris</title><style>"
        + VISUAL_STYLE
        + "body{font:16px system-ui;max-width:1000px;margin:auto;padding:20px}"
        "pre{white-space:pre-wrap}meter{width:100%;height:22px}td,th{text-align:left;padding:7px}</style>"
        "<h1>Рабочее пространство: три режима</h1>"
        "<p>Один текст, одна замороженная Qwen, один способ жадного выбора токена "
        f"и один предел ответа ({BUDGET}). В пустом контроле сохранены мост и "
        "позиции вставки, но мост получает пустые слоты. Только полный Iris "
        'передаёт ему результат трёх циклов.</p><div class="metric-grid">'
        + cards
        + "</div><h2>Парные исходы</h2><ul>"
        + pairs
        + "</ul>"
        + numeric
        + f"<p>Очистка слотов изменила последовательность ответа в "
        f"{summary['changed_after_clearing']} запросах. Изменение не равно улучшению. "
        "Синтетические запросы — пилотная проверка, а не оценка общего качества.</p>"
        "<p>В таблицах: точных ответов · средняя доля совпавших элементов · остановок по лимиту. "
        + (
            "Математика: равенство чисел и дробей. Python: доля пройденных входов; "
            "неподдерживаемый синтаксис не исполняется.</p>"
            if suite == "math-code"
            else "Для массивов важен порядок, для объектов — точный набор ключей.</p>"
        )
        + tables
        + details
        + "</html>",
        encoding="utf-8",
    )


def prepare(
    profile: str,
    prompt_count: Callable[[str], int],
    answer_count: Callable[[str], int],
    suite: str = "multilingual",
) -> list[dict[str, Any]]:
    if suite not in {"multilingual", "math-code"}:
        raise ValueError("Unknown suite")
    rows = (math_code_cases if suite == "math-code" else cases)(profile, prompt_count)
    for row in rows:
        expected = row["expected"]
        reference = row.get("reference") or (
            expected
            if isinstance(expected, str)
            else json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        )
        tokens = answer_count(reference)
        if not 1 <= tokens < BUDGET or (row["family"] == "batch-extract" and tokens <= 32):
            raise ValueError(f"Reference answer outside intended token budget: {row['id']}")
        row["reference_tokens"] = tokens
    return rows


def rescore_math_run(out: Path) -> None:
    targets = (out / "report-numeric.json", out / "index-numeric.html")
    if any(path.exists() for path in targets):
        raise ValueError("Rescoring must not replace earlier reports")
    report = json.loads((out / "report.json").read_text("utf-8"))
    if (
        report.get("kind") != "iris-coursework-workspace-comparison-v2"
        or report.get("suite") != "math-code"
    ):
        raise ValueError("A completed math-code comparison is required")
    rows = report["rows"]
    if report["profile"] not in {"full", "smoke"} or len(rows) != (
        8 if report["profile"] == "smoke" else 32
    ):
        raise ValueError("Incomplete math-code report")
    for row in rows:
        for arm in ARMS:
            fresh = grade_case(row["answers"][arm], row)
            if fresh["exact"] != row["correct"][arm]:
                raise ValueError("Strict score changed during supplemental rescoring")
            if row["grading"] == "number":
                row["grades"][arm] = fresh
    write_report(out, rows, report["profile"], "math-code", rescored=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "out"):
        parser.add_argument("--" + name, type=Path)
    for name in ("checkpoint", "manifest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--suite", choices=("multilingual", "math-code"), default="multilingual")
    parser.add_argument(
        "--rescore-run", type=Path, help="Add numeric scoring to an existing math-code report"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Verify all lengths without loading weights"
    )
    args = parser.parse_args()
    if args.rescore_run is not None:
        if any((args.snapshot, args.out, args.checkpoint, args.manifest, args.prepare_only)):
            parser.error("--rescore-run cannot be combined with generation options")
        rescore_math_run(args.rescore_run)
        return 0
    if args.snapshot is None or args.out is None:
        parser.error("--snapshot and --out are required for a new run")
    if not args.prepare_only and (args.checkpoint is None or args.manifest is None):
        parser.error("--checkpoint and --manifest are required for generation")
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute comparison output required")
    from coursework_demo import REVISION, render_prompt
    from transformers import AutoTokenizer

    if not args.snapshot.is_absolute() or args.snapshot.name != REVISION:
        raise ValueError("Pinned absolute snapshot required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.snapshot), local_files_only=True, trust_remote_code=False
    )

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    suite = prepare(
        args.profile,
        lambda prompt: count(render_prompt(tokenizer, prompt, raw=False)),
        count,
        args.suite,
    )
    args.out.mkdir(parents=True)
    (args.out / "suite.json").write_text(
        json.dumps(
            dict(
                kind=(
                    "iris-coursework-math-code-suite-v1"
                    if args.suite == "math-code"
                    else "iris-coursework-multilingual-suite-v1"
                ),
                profile=args.profile,
                output_budget=BUDGET,
                revision=REVISION,
                cases=suite,
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(suite)} cases; maximum input {max(row['prompt_tokens'] for row in suite)} tokens.",
        flush=True,
    )
    if args.prepare_only:
        return 0
    receipt = args.out / "kit-verification.json"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "scripts/coursework_offline_kit.py"),
            "verify",
            "--snapshot",
            str(args.snapshot),
            "--checkpoint",
            str(args.checkpoint),
            "--manifest",
            str(args.manifest),
            "--receipt",
            str(receipt),
        ],
        check=True,
    )
    rows: list[dict[str, Any]] = []

    def run_cases(generate: GeneratePrompt, append: Callable[[str], Any]) -> None:
        for case in suite:
            result = generate(case["prompt"], args.out / case["id"])
            comparison = result["comparison"]
            settings = comparison["settings"]
            if (
                settings["max_new_tokens"] != BUDGET
                or settings["prompt_tokens"] != case["prompt_tokens"]
                or comparison["qwen"]["positions"] != settings["prompt_tokens"]
                or comparison["empty_workspace"]["positions"] != settings["iris_positions"]
                or settings["iris_positions"] - settings["prompt_tokens"] != 8
                or not settings["filled_slots"]
                or settings["empty_latents_sha256"] == settings["iris_latents_sha256"]
            ):
                raise ValueError("Control positions or latent intervention were not verified")
            arms = dict(
                qwen=comparison["qwen"], empty_workspace=comparison["empty_workspace"], iris=result
            )
            answers = {arm: item["text"] for arm, item in arms.items()}
            grades = {arm: grade_case(answer, case) for arm, answer in answers.items()}
            row = dict(
                **case,
                settings=settings,
                answers=answers,
                correct={arm: value["exact"] for arm, value in grades.items()},
                grades=grades,
                token_ids={arm: item["token_ids"] for arm, item in arms.items()},
                stop_reasons={arm: item["stop_reason"] for arm, item in arms.items()},
            )
            append(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(row)
            print(f"{len(rows)}/{len(suite)}", case["id"], row["correct"], flush=True)

    def serve(generate: GeneratePrompt) -> None:
        with (args.out / "observations.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            run_cases(generate, stream.write)
        write_report(args.out, rows, args.profile, args.suite)

    return trained_main(
        [
            "--snapshot",
            str(args.snapshot),
            "--checkpoint",
            str(args.checkpoint),
            "--out",
            str(args.out / "verification"),
        ],
        serve=serve,
        quiet=True,
        comparison_budget=BUDGET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
