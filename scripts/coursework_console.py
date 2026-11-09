"""Verified offline interactive console, one model load, independent requests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import coursework_trained_demo as backend  # noqa: E402
from coursework_live_viewer import LiveState, live_view  # noqa: E402

HELP = (
    "Один запрос на строку. /help — справка, /exit — выход. Ctrl+C или EOF — выход.\n"
    "Запросы независимы: история не добавляется к prompt, workspace создаётся заново.\n"
    "Не более 128 токенов после chat template. Скрытого усечения нет.\n"  # noqa: RUF001
    "Qwen заморожена, обучен только bridge. Правильность ответа не гарантируется."
)


def visible(text: str) -> str:
    """Never interpret model-provided terminal escape/control characters."""
    return "".join(
        c if (c in "\n\t" or (c.isprintable() and c != "\x7f")) else ascii(c)[1:-1] for c in text
    )


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def console_loop(
    generate: backend.GeneratePrompt,
    out: Path,
    *,
    read: Callable[[str], str] = input,
    stream: TextIO | None = None,
    max_requests: int = 32,
    progress: backend.ProgressSink | None = None,
) -> dict[str, Any]:
    if type(max_requests) is not int or not 1 <= max_requests <= 100:
        raise ValueError("Session limit must be an integer between 1 and 100")
    stream = stream if stream is not None else sys.stdout
    events: list[dict[str, Any]] = []
    count = 0
    reason = "limit"
    print(HELP, file=stream, flush=True)
    try:
        while count < max_requests:
            try:
                prompt = read("Вы> ")
            except (EOFError, KeyboardInterrupt):
                reason = "input-closed"
                print("\nСеанс завершён.", file=stream, flush=True)  # noqa: RUF001
                break
            if not prompt.strip():
                continue
            if prompt.strip() == "/exit":
                reason = "exit-command"
                break
            if prompt.strip() == "/help":
                print(HELP, file=stream, flush=True)
                continue
            if prompt.startswith("/"):
                print("Неизвестная команда. /help — справка.", file=stream, flush=True)
                continue
            request_out = out / f"request-{count + 1:03d}"
            print("Вычисляю ответ…", file=stream, flush=True)
            if progress is not None:
                progress("request", dict(request=count + 1))
            try:
                result = generate(prompt, request_out)
            except backend.PromptValidationError as error:
                events.append(dict(status="rejected", message=str(error)))
                if progress is not None:
                    progress("rejected", dict(message=str(error)))
                print("Запрос отклонён: " + visible(str(error)), file=stream, flush=True)
                continue
            except BaseException as error:
                reason = "generation-error"
                events.append(
                    dict(
                        status="error",
                        type=type(error).__name__,
                        message=str(error),
                        out=str(request_out),
                    )
                )
                if progress is not None:
                    progress("failed", dict(message="Сеанс остановлен; подробности в консоли."))
                raise
            if (
                result.get("completed") is not True
                or result.get("optimizer_updates_performed") != 0
                or result.get("original_request_unchanged") is not True
                or not isinstance(result.get("text"), str)
            ):
                reason = "invalid-result"
                events.append(dict(status="invalid-result", out=str(request_out)))
                if progress is not None:
                    progress("failed", dict(message="Результат не прошёл проверку."))
                raise ValueError("Incomplete or changed-runtime generation result")
            if progress is not None:
                progress("completed", dict(text=result["text"], stop_reason=result["stop_reason"]))
            count += 1
            events.append(dict(status="completed", out=str(request_out), text=result["text"]))
            print("Iris> " + visible(result["text"]), file=stream, flush=True)
            print(
                f"Остановка: {visible(str(result['stop_reason']))}. Результаты: {request_out}",
                file=stream,
                flush=True,
            )
        else:
            print("Достигнут лимит запросов сеанса.", file=stream, flush=True)
    finally:
        receipt = dict(
            kind="iris-coursework-console-session-v1",
            completed_requests=count,
            stop_reason=reason,
            conversation_memory=False,
            optimizer_updates_performed=0,
            events=events,
        )
        write_json_new(out / "session.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--live", action="store_true", help="Enable a read-only local browser view")
    args = parser.parse_args(argv)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    out = args.out or ROOT / "artifacts/coursework" / (
        "console-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    )
    if not out.is_absolute() or ".." in out.parts or out.exists():
        raise ValueError("Fresh absolute session output required")
    if not 1 <= args.max_requests <= 100:
        raise ValueError("max-requests must be between 1 and 100")
    out.mkdir(parents=True)
    print("Проверяю локальный комплект…", flush=True)
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
            str(out / "kit-verification.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check.returncode:
        print("Проверка не прошла. Модель не загружалась. Сеанс: " + str(out), flush=True)
        print(visible(check.stderr[-3000:]), file=sys.stderr, flush=True)
        return 1
    print("Загружаю Qwen3-0.6B и проверяю bridge один раз. Подождите…", flush=True)
    live = LiveState() if args.live else None

    def serve(generate: backend.GeneratePrompt) -> None:
        print("Готово. Каталог сеанса: " + str(out), flush=True)
        if live is None:
            console_loop(generate, out, max_requests=args.max_requests)
        else:
            with live_view(live) as url:
                print("Откройте живую таблицу в браузере: " + url, flush=True)
                console_loop(generate, out, max_requests=args.max_requests, progress=live.publish)

    return backend.main(
        [
            "--snapshot",
            str(args.snapshot),
            "--checkpoint",
            str(args.checkpoint),
            "--out",
            str(out / "verification"),
        ],
        serve=serve,
        quiet=True,
        **({"progress": live.publish} if live is not None else {}),
    )


if __name__ == "__main__":
    raise SystemExit(main())
