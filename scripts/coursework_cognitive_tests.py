# ruff: noqa: E501, RUF001
"""Run small, reproducible coursework demonstrations on the local trained bridge."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import coursework_trained_demo as backend  # noqa: E402


def make_cases() -> list[dict[str, str]]:
    cases = []
    for size in (12, 48):
        lines = [f"Запись {i:03d}: проверка оборудования завершена." for i in range(1, size + 1)]
        lines[size // 2] = f"Запись {size // 2 + 1:03d}: пароль для архива — кедр."
        document = "\n".join(lines)
        cases.append(
            dict(
                id=f"password-{size}",
                name=f"Поиск пароля: {size} записей",
                description="Найти конкретный факт среди отвлекающих записей.",
                document=document,
                expected="кедр",
                prompt=f"Документ:\n{document}\nВопрос: какой пароль указан для архива? Ответь одним словом.",
            )
        )
    cases.extend(
        [
            dict(
                id="integration",
                name="Связывание двух фактов",
                description="Объединить два факта в один вывод.",
                expected="липа",
                document="",
                prompt="Ключ лежит в синей коробке. Синяя коробка стоит под деревом липа. Под каким деревом ключ? Ответь одним словом.",
            ),
            dict(
                id="update",
                name="Обновление информации",
                description="Удержать последнее значение при противоречии со старым.",
                expected="синий",
                document="",
                prompt="Сначала кодовое слово было красный. Затем его заменили на синий. Старое слово больше не действует. Какое слово действует сейчас? Ответь одним словом.",
            ),
            dict(
                id="rule",
                name="Выбор по правилу",
                description="Применить правило к двум независимым признакам.",
                expected="груша",
                document="",
                prompt="Выбери зелёный фрукт, который находится слева. Слева находится зелёная груша. Справа находится красное яблоко. Ответь одним словом.",
            ),
        ]
    )
    return cases


def normalize(text: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", text.strip().casefold())


def write_report(out: Path, results: list[dict], status: str) -> None:
    report = {
        "kind": "iris-coursework-cognitive-demo-v1",
        "status": status,
        "method": "Независимые запросы; точное совпадение после удаления внешней пунктуации. Это не сравнение с системой без рабочего пространства.",
        "results": results,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    esc = html.escape
    cards = []
    for result in results:
        cards.append(
            "<article><h2>"
            + esc(result["name"])
            + "</h2><p>"
            + esc(result["description"])
            + "</p><p><b>Результат:</b> "
            + ("совпадает" if result["passed"] else "не совпадает")
            + " · ожидалось: "
            + esc(result["expected"])
            + " · получено: "
            + esc(result["answer"])
            + "</p><p>Запрос: "
            + str(result["prompt_tokens"])
            + " токенов после шаблона; "
            + str(result["prompt_bytes"])
            + " байт UTF-8. Время: "
            + str(round(result["seconds"], 1))
            + " с. Остановка: "
            + esc(result["stop_reason"])
            + '.</p><p><a href="'
            + result["id"]
            + '/viewer.html">Открыть фактический граф запроса</a></p></article>'
        )
    page = (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Iris · демонстрационные тесты</title><style>body{font:16px system-ui;max-width:900px;margin:auto;padding:24px;background:#101827;color:#e6edf6}article{background:#1b293b;border:1px solid #44546a;border-radius:12px;padding:12px 18px;margin:14px 0}a{color:#90c8ff}p{line-height:1.5}</style><h1>Демонстрационные тесты Iris</h1><p>Статус: """
        + esc(status)
        + """. Независимые запросы, простое точное сравнение; тесты не доказывают влияние рабочего пространства. Модули не обучены; обучен только мост. Документы синтетические.</p>"""
        + "".join(cards)
        + "</html>"
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute output directory required")
    args.out.mkdir(parents=True)
    check = subprocess.run(
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
            str(args.out / "kit-verification.json"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check.returncode:
        raise RuntimeError("Kit verification failed: " + check.stderr[-2000:])
    from coursework_demo import render_prompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.snapshot), local_files_only=True, trust_remote_code=False
    )
    results: list[dict] = []
    write_report(args.out, results, "выполняется")

    def serve(generate: backend.GeneratePrompt) -> None:
        try:
            for case in make_cases():
                print("Тест: " + case["name"], flush=True)
                if case["document"]:
                    (args.out / (case["id"] + ".txt")).write_text(
                        case["document"], encoding="utf-8"
                    )
                rendered = render_prompt(tokenizer, case["prompt"], raw=False)
                start = time.perf_counter()
                result = generate(case["prompt"], args.out / case["id"])
                row = {k: case[k] for k in ("id", "name", "description", "expected")}
                row.update(
                    answer=result["text"],
                    passed=normalize(result["text"]) == normalize(case["expected"]),
                    seconds=time.perf_counter() - start,
                    stop_reason=result["stop_reason"],
                    prompt_bytes=len(case["prompt"].encode("utf-8")),
                    document_bytes=len(case["document"].encode("utf-8")),
                    prompt_tokens=len(tokenizer.encode(rendered, add_special_tokens=False)),
                )
                results.append(row)
                write_report(args.out, results, "выполняется")
                print(json.dumps(row, ensure_ascii=False), flush=True)
        except BaseException:
            write_report(args.out, results, "прервано")
            raise
        write_report(args.out, results, "завершено")

    return backend.main(
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
