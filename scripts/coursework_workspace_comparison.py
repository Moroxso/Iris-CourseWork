# ruff: noqa: E402, E501, RUF001
"""Paired coursework checks: Qwen, empty workspace, and full Iris."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from coursework_cognitive_tests import normalize
from coursework_trained_demo import GeneratePrompt
from coursework_trained_demo import main as trained_main
from coursework_visuals import VISUAL_STYLE

ARMS = {
    "qwen": "Qwen без вставки",
    "empty_workspace": "Iris: пустые слоты",
    "iris": "Iris: рабочее пространство",
}
BUDGET = 24


def cases() -> list[dict[str, str]]:
    """Counterfactual pairs change a relevant fact without changing the question."""
    rows = []
    for i, (word, color, tree, fruit) in enumerate(
        (("маяк", "синий", "липа", "груша"), ("туман", "белый", "клён", "слива")), 1
    ):
        log = [f"Запись {n}: плановая проверка оборудования завершена." for n in range(1, 16)]
        log[7] = f"Запись 8: действующее кодовое слово для архива — «{word}»."
        for family, prompt, expected in (
            (
                "search",
                "\n".join(log) + "\nКакое кодовое слово для архива? Ответь одним словом.",
                word,
            ),
            (
                "update",
                "Старый пропуск: зелёный. "
                f"Запись обновлена: действует только {color}. Какой пропуск действует? Ответь одним словом.",
                color,
            ),
            (
                "combine",
                "Ключ лежит в ящике. "
                f"Ящик стоит под деревом {tree}. Под каким деревом ключ? Назови дерево одним словом.",
                tree,
            ),
            (
                "select",
                f"Слева на столе — зелёная {fruit}. Справа — красное яблоко. "
                "Назови зелёный фрукт слева одним словом.",
                fruit,
            ),
        ):
            rows.append(dict(id=f"{family}-{i}", family=family, prompt=prompt, expected=expected))
    return sorted(rows, key=lambda row: row["id"])


def score(answers: dict[str, str], expected: str) -> dict[str, bool]:
    if set(answers) != set(ARMS):
        raise ValueError("All three comparison arms are required")
    return {arm: normalize(answer) == normalize(expected) for arm, answer in answers.items()}


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


def write_report(out: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize(rows)
    (out / "report.json").write_text(
        json.dumps(
            dict(kind="iris-coursework-workspace-comparison-v1", summary=summary, rows=rows),
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
    details = "".join(
        f'<details class="visual-panel"><summary>{escape(row["id"])} · ожидалось '
        f"«{escape(row['expected'])}»</summary><pre>{escape(row['prompt'])}</pre><ul>"
        + "".join(
            f"<li>{escape(ARMS[arm])}: {escape(row['answers'][arm])} "
            f"({'совпало' if row['correct'][arm] else 'не совпало'})</li>"
            for arm in ARMS
        )
        + f'</ul><a href="{escape(row["id"])}/viewer.html">Граф полного Iris</a></details>'
        for row in rows
    )
    (out / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Сравнение Qwen и Iris</title><style>"
        + VISUAL_STYLE
        + "body{font:16px system-ui;max-width:1000px;margin:auto;padding:20px}"
        "pre{white-space:pre-wrap}meter{width:100%;height:22px}</style>"
        "<h1>Рабочее пространство: три режима</h1>"
        "<p>Один текст, одна замороженная Qwen, один способ жадного выбора токена "
        f"и один предел ответа ({BUDGET}). В пустом контроле сохранены мост и "
        "позиции вставки, но мост получает пустые слоты. Только полный Iris "
        'передаёт ему результат трёх циклов.</p><div class="metric-grid">'
        + cards
        + "</div><h2>Парные исходы</h2><ul>"
        + pairs
        + "</ul>"
        f"<p>Очистка слотов изменила последовательность ответа в "
        f"{summary['changed_after_clearing']} запросах. Изменение не равно улучшению. "
        "Восемь синтетических запросов — пилотная проверка, а не оценка общего качества.</p>"
        + details
        + "</html>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "checkpoint", "manifest", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute comparison output required")
    args.out.mkdir(parents=True)
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

    def serve(generate: GeneratePrompt) -> None:
        for case in cases():
            result = generate(case["prompt"], args.out / case["id"])
            comparison = result["comparison"]
            settings = comparison["settings"]
            if (
                settings["max_new_tokens"] != BUDGET
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
            rows.append(
                dict(
                    **case,
                    settings=settings,
                    answers=answers,
                    correct=score(answers, case["expected"]),
                    token_ids={arm: item["token_ids"] for arm, item in arms.items()},
                    stop_reasons={arm: item["stop_reason"] for arm, item in arms.items()},
                )
            )
            print(case["id"], rows[-1]["correct"], flush=True)
        write_report(args.out, rows)

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
