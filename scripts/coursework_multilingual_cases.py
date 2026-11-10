# ruff: noqa: E501, RUF001
"""Matched synthetic tasks; token bands count the actual rendered Qwen input."""

from collections.abc import Callable
from typing import Any

LANGUAGES = {"ru": "Русский", "en": "English", "zh": "简体中文"}
FAMILIES = {
    "search": "Поиск факта",
    "update": "Обновление факта",
    "combine": "Связывание фактов",
    "rule": "Фильтрация по правилу",
    "long-retrieval": "Поиск в длинном журнале",
    "ordered-plan": "Восстановление порядка",
    "multi-field": "Извлечение полей",
    "batch-extract": "Длинный структурированный ответ",
}
SMOKE_FAMILIES = {"combine", "long-retrieval", "multi-field", "batch-extract"}
INPUT_LIMIT = 1024
BYTE_LIMIT = 32768
OUTPUT_BUDGET = 192

TEXT: dict[str, dict[str, Any]] = {
    "ru": {
        "words": ("маяк", "туман", "камень", "север"),
        "colors": ("синий", "белый", "красный", "чёрный"),
        "trees": ("липа", "клён", "дуб", "берёза"),
        "record": "Запись {n}: в секции {room} проверили освещение и закрыли журнал.",
        "fact": "Запись {n}: действующее кодовое слово архива — «{word}».",
        "search": "Назови действующее кодовое слово архива. Ответь только словом.",
        "update": "Вчера пропуск был зелёный. Сегодня его заменили: действует только {color}. Старый недействителен. Какой цвет действует? Ответь только словом.",
        "combine": "Ключ лежит в коробке. Коробка стоит под деревом {tree}. Под каким деревом ключ? Ответь только названием дерева в именительном падеже.",
        "rule": "Подходит только предмет, который синий, целый и лежит слева. {good}: синий, целый, слева. A-00: синий, сломан, слева. B-00: синий, целый, справа. C-00: красный, целый, слева. Верни только код подходящего предмета.",
        "plan": "Этап {a} идёт сразу перед {b}. Этап {c} идёт после {b}, но перед {d}. Других этапов нет. Верни только JSON-массив кодов от первого до последнего.",
        "fields": "Первичная заявка: код {old}, секция Z9, ответственный Kai. Исправление для той же заявки: код {ticket}, секция {room}, ответственный {owner}. Верни только JSON-объект с ключами ticket, room, owner и итоговыми значениями.",
        "batch": "Из журнала ниже извлеки коды всех строк со статусом ДА, в порядке строк. Верни только JSON-массив строк, не добавляй объяснение.",
        "row": "Строка {n}: {code}; статус {status}.",
        "yes": "ДА",
        "no": "НЕТ",
    },
    "en": {
        "words": ("beacon", "fog", "stone", "north"),
        "colors": ("blue", "white", "red", "black"),
        "trees": ("linden", "maple", "oak", "birch"),
        "record": "Entry {n}: staff checked the lights in section {room} and closed the log.",
        "fact": "Entry {n}: the active archive code word is '{word}'.",
        "search": "What is the active archive code word? Return only the word.",
        "update": "Yesterday the pass was green. It was replaced today: only {color} is valid. The old pass is invalid. Which color is valid? Return only the word.",
        "combine": "The key is inside a box. The box is under a {tree} tree. Which kind of tree is the key under? Return only the tree name.",
        "rule": "A qualifying item must be blue, intact, and on the left. {good}: blue, intact, left. A-00: blue, broken, left. B-00: blue, intact, right. C-00: red, intact, left. Return only the qualifying item's code.",
        "plan": "Step {a} comes immediately before {b}. Step {c} comes after {b} but before {d}. There are no other steps. Return only a JSON array of codes from first to last.",
        "fields": "Original request: code {old}, section Z9, owner Kai. Correction for the same request: code {ticket}, section {room}, owner {owner}. Return only a JSON object with keys ticket, room, owner and the final values.",
        "batch": "Extract the codes of every row with status YES, in row order. Return only a JSON array of strings, without explanation.",
        "row": "Row {n}: {code}; status {status}.",
        "yes": "YES",
        "no": "NO",
    },
    "zh": {
        "words": ("灯塔", "雾", "石头", "北方"),
        "colors": ("蓝色", "白色", "红色", "黑色"),
        "trees": ("椴树", "枫树", "橡树", "桦树"),
        "record": "第{n}条：工作人员检查了{room}区的照明，并关闭了登记簿。",
        "fact": "第{n}条：档案当前有效的口令是「{word}」。",
        "search": "档案当前有效的口令是什么？只输出口令。",
        "update": "昨天的通行证是绿色的。今天更换了通行证：只有{color}有效，旧证已经作废。哪种颜色有效？只输出颜色。",
        "combine": "钥匙在盒子里。盒子放在一棵{tree}下面。钥匙在哪种树下？只输出树名。",
        "rule": "合格物品必须是蓝色、完好且放在左边。{good}：蓝色、完好、左边。A-00：蓝色、损坏、左边。B-00：蓝色、完好、右边。C-00：红色、完好、左边。只输出合格物品的编号。",
        "plan": "步骤{a}紧接在{b}之前。步骤{c}在{b}之后且在{d}之前。没有其他步骤。只输出按先后顺序排列的编号 JSON 数组。",
        "fields": "原申请：编号{old}，区域Z9，负责人Kai。同一申请更正为：编号{ticket}，区域{room}，负责人{owner}。只输出一个 JSON 对象，键为 ticket、room、owner，值为更正后的内容。",
        "batch": "从下面的登记记录中提取所有状态为「是」的编号，保持原有顺序。只输出 JSON 字符串数组，不要解释。",
        "row": "第{n}行：{code}；状态{status}。",
        "yes": "是",
        "no": "否",
    },
}


def journal(lang: str, variant: int, size: int, fraction: float) -> str:
    t = TEXT[lang]
    at = min(size, 1 + int((size - 1) * fraction))
    return "\n".join(
        [
            t["fact"].format(n=n, word=t["words"][variant])
            if n == at
            else t["record"].format(n=n, room=f"{chr(65 + (n + variant) % 9)}{n % 10}")
            for n in range(1, size + 1)
        ]
        + [t["search"]]
    )


def fit_long(lang: str, variant: int, count: Callable[[str], int], low: int, high: int) -> str:
    fraction = (0.12, 0.40, 0.64, 0.86)[variant]
    left, right = 1, 256
    while left <= right:
        size = (left + right) // 2
        prompt = journal(lang, variant, size, fraction)
        tokens = count(prompt)
        if low <= tokens <= high:
            return prompt
        if tokens < low:
            left = size + 1
        else:
            right = size - 1
    raise ValueError(f"No prompt inside token band: {lang}, {variant}, {low}–{high}")


def cases(profile: str, count: Callable[[str], int]) -> list[dict[str, Any]]:
    if profile not in {"smoke", "full"}:
        raise ValueError("Unknown comparison profile")
    rows = []
    for lang in LANGUAGES:
        t = TEXT[lang]
        for family in FAMILIES:
            if profile == "smoke" and family not in SMOKE_FAMILIES:
                continue
            for v in (3,) if profile == "smoke" else range(4):
                band = None
                if family == "search":
                    prompt, expected = journal(lang, v, 9, 0.5), t["words"][v]
                elif family == "long-retrieval":
                    band = (460, 550) if v < 2 else (820, 960)
                    prompt = fit_long(lang, v, count, *band)
                    expected = t["words"][v]
                elif family == "update":
                    expected = t["colors"][v]
                    prompt = t["update"].format(color=expected)
                elif family == "combine":
                    expected = t["trees"][v]
                    prompt = t["combine"].format(tree=expected)
                elif family == "rule":
                    expected = f"M-{14 + v}"
                    prompt = t["rule"].format(good=expected)
                elif family == "ordered-plan":
                    expected = [f"S{(v + j * 3) % 13 + 1}" for j in range(4)]
                    prompt = t["plan"].format(**dict(zip("abcd", expected, strict=True)))
                elif family == "multi-field":
                    expected = dict(
                        ticket=f"T{20 + v}",
                        room=f"R{4 + v}",
                        owner=("Ada", "Mina", "Lin", "Mika")[v],
                    )
                    prompt = t["fields"].format(old=f"X{v}", **expected)
                else:
                    log = []
                    expected = []
                    for i in range(1, 33):
                        code = f"K{(i * 13 + v * 7) % 97:02d}"
                        selected = (i + v) % 4 != 0
                        log.append(
                            t["row"].format(n=i, code=code, status=t["yes" if selected else "no"])
                        )
                        if selected:
                            expected.append(code)
                    prompt = t["batch"] + "\n" + "\n".join(log)
                tokens = count(prompt)
                if not 1 <= tokens <= INPUT_LIMIT or len(prompt.encode("utf-8")) > BYTE_LIMIT:
                    raise ValueError(f"Prompt exceeds limits: {lang}/{family}/{v + 1}")
                rows.append(
                    dict(
                        id=f"{lang}-{family}-{v + 1}",
                        language=lang,
                        family=family,
                        variant=v + 1,
                        prompt=prompt,
                        expected=expected,
                        prompt_tokens=tokens,
                        target_band=band,
                        length_band="1–256"
                        if tokens <= 256
                        else "257–640"
                        if tokens <= 640
                        else "641–1024",
                    )
                )
    return rows
