# ruff: noqa: E501, RUF001
"""Render a read-only training storyboard from a completed local bridge run."""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from math import isfinite
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from coursework_visuals import VISUAL_STYLE

EXAMPLES = {
    "train-color": "Цвет",
    "train-password": "Пароль",
    "train-addition": "Сложение",
}


def read_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 512 * 1024:
        raise ValueError("Training record exceeds expected size")
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training record must be an object")
    return value


def render(run: Path) -> str:
    result = read_object(run / "result.json")
    history = result["gradient_history"]
    if (
        result["kind"] != "iris-coursework-bridge-tutorial-result-v1"
        or result["completed"] is not True
        or result["trained_component"] != "copy-of-latent-bridge-only"
        or result["backbone_frozen"] is not True
        or result["cognitive_components_trained"] is not False
        or not isinstance(history, list)
        or not 1 <= len(history) <= 12
        or len(history) != result["optimizer_steps_performed"]
    ):
        raise ValueError("Completed local bridge training with a bounded history required")
    points = []
    for i, row in enumerate(history, result["restored_step"] + 1):
        name = f"step-{i:03d}"
        receipt = read_object(run / f"{name}.json")
        nll = row["nll"]
        elapsed = receipt["seconds"]
        if (
            row["step"] != i
            or row["id"] not in EXAMPLES
            or row["split"] != "train"
            or receipt["stage"] != name
            or not isfinite(nll)
            or nll < 0
            or not isfinite(elapsed)
            or elapsed < 0
            or (points and elapsed < points[-1][3])
        ):
            raise ValueError("Invalid or out-of-order training measurements")
        points.append((i, EXAMPLES[row["id"]], nll, elapsed))
    if points[-1][0] != result["completed_steps"]:
        raise ValueError("Training history ends before the declared last step")
    chart_rows = "".join(
        f'<li data-step="{i}"><span>Шаг {i} · {escape(label)}</span>'
        f'<meter min="0" max="{max(1, *(p[2] for p in points))}" value="{nll}"></meter>'
        f"<span>{nll:.4g}</span></li>"
        for i, label, nll, _ in points
    )
    table_rows = "".join(
        f"<tr><td>{i}</td><td>{escape(label)}</td><td>{nll:.8g}</td><td>{elapsed:.2f}</td></tr>"
        for i, label, nll, elapsed in points
    )
    held = [
        result[name]["heldout"]
        for name in ("baseline_summary", "untrained_summary", "trained_summary")
    ]
    outcomes = "".join(
        f"<li>{label}: {int(row['exact'])} из {int(row['total'])}</li>"
        for label, row in zip(
            ("Qwen без моста", "Iris до обучения", "Iris после обучения"), held, strict=True
        )
    )
    return (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Как обучался векторный мост</title><style>"
        + VISUAL_STYLE
        + "body{font:16px system-ui;max-width:900px;margin:auto;padding:20px}"
        "li[data-step]{display:grid;grid-template-columns:10em 1fr 5em;gap:9px;"
        "align-items:center;padding:7px}li[data-step][aria-current=true]{"
        "outline:2px solid var(--iris-accent);border-radius:8px}"
        "meter,input[type=range]{width:100%}th,td{text-align:left;padding:7px}</style>"
        "<h1>Как обучался векторный мост</h1>"
        "<p>Воспроизведение сохранённого прогона. Новое обучение не запускается.</p>"
        '<section class="visual-panel"><h2>Что обновлялось</h2>'
        "<p>Запрос → готовое состояние рабочего пространства → обучаемый мост → Qwen → ошибка ответа. "
        "Правильный ответ нужен для подсчёта ошибки при обучении. "
        "Обновлялся только мост; веса Qwen и модулей не изменялись.</p></section>"
        '<section class="visual-panel"><h2><abbr title="Отрицательное логарифмическое правдоподобие. '
        'Чем ниже значение на том же примере, тем выше вероятность правильного ответа.">'
        "Ошибка на шагах обучения</abbr></h2><p>Сравнивайте один и тот же пример. "
        "Разные примеры не образуют единую кривую качества.</p>"
        f'<label for="step">Выделить сохранённый шаг <output id="current"></output></label>'
        f'<input id="step" type="range" min="{points[0][0]}" max="{points[-1][0]}" '
        f'value="{points[0][0]}" step="1"><ul>'
        + chart_rows
        + '</ul><details class="raw-data"><summary>Точные значения</summary><table><thead>'
        "<tr><th>Шаг</th><th>Пример</th><th>Ошибка</th><th>Время от начала, с</th></tr>"
        "</thead><tbody>" + table_rows + "</tbody></table></details></section>"
        '<section class="visual-panel"><h2>Отложенные примеры</h2><ul>'
        + outcomes
        + "</ul><p>Эти примеры не использовались для обновления. Малый набор не доказывает "
        "общего улучшения качества.</p></section><script>"
        'const step=document.getElementById("step");function show(){'
        'document.getElementById("current").textContent=step.value;'
        'document.querySelectorAll("[data-step]").forEach(x=>'
        'x.setAttribute("aria-current",String(x.dataset.step===step.value)));}'
        'step.addEventListener("input",show);show();</script></html>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    page = render(args.run)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(page)


if __name__ == "__main__":
    main()
