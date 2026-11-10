# ruff: noqa: E501, RUF001
"""Deterministic arithmetic and safe integer-function generation tasks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from fractions import Fraction
from math import comb
from typing import Any

FAMILIES = {
    "math-arithmetic": "Целочисленные выражения",
    "math-fraction": "Дроби",
    "math-equation": "Линейные уравнения",
    "math-counting": "Комбинаторика",
    "code-clamp": "Python: ограничение диапазона",
    "code-rounding": "Python: округление вверх",
    "code-interval": "Python: пересечение интервалов",
    "code-multiples": "Python: подсчёт кратных",
}
NUMBER = re.compile(r"[+-]?(?:[0-9]{1,20}(?:\.[0-9]{1,20})?|[0-9]{1,20}/[0-9]{1,20})")
FINAL_MARKER = re.compile(r"(?:^|\n|\*\*)\s*(?:ответ|итог|answer)\s*:\s*", re.I)


def as_number(text: str) -> Fraction | None:
    if len(text) > 128:
        return None
    number = unicodedata.normalize("NFKC", text).replace("−", "-").replace(",", ".").strip()
    try:
        return Fraction(number) if NUMBER.fullmatch(number) else None
    except ZeroDivisionError:
        return None


def marked_result(answer: str) -> Fraction | None:
    if len(answer) > 4096:
        return None
    markers = list(FINAL_MARKER.finditer(answer))
    if markers:
        final = answer[markers[-1].end() :]
    else:
        box = re.search(r"\\boxed\s*\{([^\n]+)\}\s*(?:\$\$?|\\\]|[.。])?\s*$", answer)
        if not box:
            return None
        final = box[1]
    final = final.strip().rstrip(".。").strip().strip("*").strip().strip("$").strip()
    for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]")):
        if final.startswith(opening) and final.endswith(closing):
            final = final[len(opening) : -len(closing)].strip()
    if box := re.fullmatch(r"\\boxed\s*\{(.+)\}", final, re.S):
        final = box[1].strip()
    final = re.sub(r"^x\s*=\s*", "", final, flags=re.I)
    if ratio := re.fullmatch(r"([+-]?)\\(?:d|t)?frac\s*\{([+-]?[0-9]+)\}\s*\{([0-9]+)\}", final):
        final = ratio[1] + ratio[2] + "/" + ratio[3]
    return as_number(final)


def grade_number(answer: str, expected: str) -> dict[str, Any]:
    actual = as_number(answer)
    target = Fraction(expected)
    exact = actual is not None and actual == target
    recognized = actual if actual is not None else marked_result(answer)
    return dict(
        exact=exact,
        matched=int(exact),
        total=1,
        format_valid=actual is not None,
        status="evaluated" if actual is not None else "not-a-number",
        numeric_value_correct=None if recognized is None else recognized == target,
    )


def math_tasks(v: int) -> list[tuple[str, str, str]]:
    a, b, c, d = ((17, 9, 6, 81), (21, 5, 4, 110), (13, 7, 9, 191), (31, 8, 5, 59))[v]
    x = (-7, 11, -4, 9)[v]
    p, q, r, s = ((3, 8, 5, 12), (7, 15, 1, 6), (2, 7, 5, 9), (4, 9, 3, 14))[v]
    n, k = ((8, 3), (10, 4), (9, 2), (11, 3))[v]
    rhs = (a - c) * x + b
    return [
        ("math-arithmetic", f"Вычисли ({a} - {b}) * {c} - {d}.", str((a - b) * c - d)),
        ("math-fraction", f"Вычисли {p}/{q} - {r}/{s}.", str(Fraction(p, q) - Fraction(r, s))),
        ("math-equation", f"Найди x: {a}*x + {b} = {c}*x + ({rhs}).", str(x)),
        (
            "math-counting",
            f"Сколькими способами можно выбрать {k} разных предметов из {n}, если порядок не важен?",
            str(comb(n, k)),
        ),
    ]


def code_task(v: int) -> list[tuple[str, str, dict[str, Any], str]]:
    low, high = ((-3, 5), (2, 9), (-11, -2), (0, 0))[v]
    width = (6, 7, 4, 11)[v]
    start, stop = ((-3, 7), (5, 12), (-9, -2), (0, 4))[v]
    step = (3, 5, 4, 7)[v]

    def item(
        family: str,
        name: str,
        parameters: list[str],
        description: str,
        inputs: list[tuple[int, ...]],
        oracle: Callable[..., int],
        reference: str,
    ) -> tuple[str, str, dict[str, Any], str]:
        prompt = (
            f"Напиши одну функцию Python 3: def {name}({', '.join(parameters)}). {description} "
            "Верни только код. Используй целые числа, арифметику + - * // %, сравнения, "
            "присваивания, if/else, return, min, max или abs. Без циклов, импорта и других функций."
        )
        return (
            family,
            prompt,
            dict(
                name=name,
                parameters=parameters,
                checks=[dict(args=list(values), value=oracle(*values)) for values in inputs],
            ),
            reference,
        )

    return [
        item(
            "code-clamp",
            "clip",
            ["x"],
            f"Верни x, ограниченный включительно диапазоном [{low}, {high}]. x — целое.",
            [
                (x,)
                for x in sorted(
                    {low - 100, low - 1, low, (low + high) // 2, high, high + 1, high + 100}
                )
            ],
            lambda x: min(high, max(low, x)),
            f"def clip(x):\n    return min({high}, max({low}, x))",
        ),
        item(
            "code-rounding",
            "box_count",
            ["n"],
            f"Верни минимальное число коробок, чтобы упаковать n предметов по {width} в коробку. n — неотрицательное целое.",
            [(n,) for n in (0, 1, width - 1, width, width + 1, width * 4, 199)],
            lambda n: -(-n // width),
            f"def box_count(n):\n    return (n + {width - 1}) // {width}",
        ),
        item(
            "code-interval",
            "overlap",
            ["a", "b"],
            f"Верни длину пересечения полуоткрытых интервалов [a,b) и [{start},{stop}). a и b — целые, a <= b.",
            [
                (start - 5, start),
                (stop, stop + 5),
                (start - 3, start + 1),
                (start, stop),
                (stop - 1, stop + 3),
                (start + 1, start + 1),
                (start - 10, stop + 10),
            ],
            lambda a, b: len(set(range(a, b)) & set(range(start, stop))),
            f"def overlap(a, b):\n    return max(0, min(b, {stop}) - max(a, {start}))",
        ),
        item(
            "code-multiples",
            "count_hits",
            ["lo", "hi"],
            f"Посчитай целые числа, кратные {step}, на отрезке [lo, hi] с обеими границами. lo <= hi; числа могут быть отрицательными. Ноль кратен {step}.",
            [
                (-13, -1),
                (-9, 9),
                (0, 0),
                (1, step - 1),
                (step, step),
                (step + 1, step * 4),
                (-step, -step),
            ],
            lambda lo, hi: sum(n % step == 0 for n in range(lo, hi + 1)),
            f"def count_hits(lo, hi):\n    return hi // {step} - (lo - 1) // {step}",
        ),
    ]


def cases(profile: str, count: Callable[[str], int]) -> list[dict[str, Any]]:
    if profile not in {"smoke", "full"}:
        raise ValueError("Unknown math-code profile")
    rows: list[dict[str, Any]] = []
    for v in (0,) if profile == "smoke" else range(4):
        for family, prompt, expected in math_tasks(v):
            rows.append(
                dict(
                    family=family,
                    prompt=prompt + " Ответь только числом или точной дробью a/b.",
                    expected=expected,
                    reference=expected,
                    grading="number",
                    variant=v + 1,
                )
            )
        for family, prompt, expected_code, reference in code_task(v):
            rows.append(
                dict(
                    family=family,
                    prompt=prompt,
                    expected=expected_code,
                    reference=reference,
                    grading="function",
                    variant=v + 1,
                )
            )
    for row in rows:
        tokens = count(row["prompt"])
        if not 1 <= tokens <= 1024 or len(row["prompt"].encode("utf-8")) > 32768:
            raise ValueError("Math/code task exceeds model input limits")
        row.update(
            id=f"{row['family']}-{row['variant']}",
            language="ru",
            prompt_tokens=tokens,
            length_band="1–256" if tokens <= 256 else "257–1024",
        )
    return rows
