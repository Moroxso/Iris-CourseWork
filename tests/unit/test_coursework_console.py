"""Interactive lifecycle tests without loading weights or launching processes."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAIN_INPUTS = [
    "--snapshot",
    "test-snapshot",
    "--checkpoint",
    "test-checkpoint",
    "--manifest",
    "test-manifest",
]


def console() -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_console_test", ROOT / "scripts/coursework_console.py"
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def answer(text: str) -> dict[str, Any]:
    return dict(
        completed=True,
        text=text,
        stop_reason="stop_token",
        optimizer_updates_performed=0,
        original_request_unchanged=True,
    )


def reader(lines: list[str]) -> Any:
    items = iter(lines)

    def read(_: str) -> str:
        try:
            return next(items)
        except StopIteration:
            raise EOFError from None

    return read


def test_two_independent_prompts_use_same_generator_and_distinct_outputs(tmp_path: Path) -> None:
    module = console()
    calls: list[tuple[str, Path]] = []

    def generate(prompt: str, out: Path) -> dict[str, Any]:
        calls.append((prompt, out))
        return answer("reply: " + prompt)

    stream = io.StringIO()
    receipt = module.console_loop(
        generate, tmp_path, read=reader(["first", "second", "/exit"]), stream=stream
    )
    assert calls == [("first", tmp_path / "request-001"), ("second", tmp_path / "request-002")]
    assert "Iris> reply: first" in stream.getvalue()
    assert "Iris> reply: second" in stream.getvalue()
    assert receipt["completed_requests"] == 2
    assert receipt["conversation_memory"] is False
    assert json.loads((tmp_path / "session.json").read_text("utf-8")) == receipt


def test_commands_and_blank_lines_do_not_generate(tmp_path: Path) -> None:
    def never(*_: Any) -> Any:
        raise AssertionError("Command must not reach model")

    stream = io.StringIO()
    receipt = console().console_loop(
        never, tmp_path, read=reader(["", " ", "/help", "/unknown", "/exit"]), stream=stream
    )
    assert receipt["completed_requests"] == 0
    assert receipt["stop_reason"] == "exit-command"
    assert "/help" in stream.getvalue()


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_input_interruption_is_clean(tmp_path: Path, interruption: type[BaseException]) -> None:
    def read(_: str) -> str:
        raise interruption

    result = console().console_loop(
        lambda *_: answer("unused"), tmp_path, read=read, stream=io.StringIO()
    )
    assert result["stop_reason"] == "input-closed"
    assert result["completed_requests"] == 0


def test_rejected_prompt_can_be_corrected_without_advancing_output(tmp_path: Path) -> None:
    module = console()
    calls: list[Path] = []

    def generate(prompt: str, out: Path) -> dict[str, Any]:
        calls.append(out)
        if prompt == "too long":
            raise module.backend.PromptValidationError("128 token limit")
        return answer("ok")

    receipt = module.console_loop(
        generate, tmp_path, read=reader(["too long", "valid", "/exit"]), stream=io.StringIO()
    )
    assert calls == [tmp_path / "request-001"] * 2
    assert [x["status"] for x in receipt["events"]] == ["rejected", "completed"]


@pytest.mark.parametrize("failure", [ValueError, MemoryError, KeyboardInterrupt])
def test_runtime_failure_ends_session_and_retains_receipt(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    def generate(*_: Any) -> Any:
        raise failure("generation failed")

    with pytest.raises(failure):
        console().console_loop(
            generate, tmp_path, read=reader(["prompt", "must not run"]), stream=io.StringIO()
        )
    receipt = json.loads((tmp_path / "session.json").read_text("utf-8"))
    assert receipt["completed_requests"] == 0
    assert receipt["stop_reason"] == "generation-error"
    assert receipt["events"][0]["type"] == failure.__name__


def test_untrusted_model_terminal_controls_are_escaped(tmp_path: Path) -> None:
    module = console()
    raw = "hello\x1b[2J\rbye\x07"
    stream = io.StringIO()
    receipt = module.console_loop(
        lambda *_: answer(raw), tmp_path, read=reader(["prompt", "/exit"]), stream=stream
    )
    assert "\x1b" not in stream.getvalue()
    assert "\r" not in stream.getvalue()
    assert "\\x1b" in stream.getvalue()
    assert receipt["events"][0]["text"] == raw


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", False),
        ("optimizer_updates_performed", 1),
        ("original_request_unchanged", False),
        ("text", None),
    ],
)
def test_invalid_result_never_counts_as_complete(tmp_path: Path, field: str, value: Any) -> None:
    result = answer("ok")
    result[field] = value
    with pytest.raises(ValueError, match="Incomplete"):
        console().console_loop(
            lambda *_: result, tmp_path, read=reader(["prompt"]), stream=io.StringIO()
        )
    receipt = json.loads((tmp_path / "session.json").read_text("utf-8"))
    assert receipt["completed_requests"] == 0
    assert receipt["stop_reason"] == "invalid-result"


def test_limit_ends_without_an_extra_read(tmp_path: Path) -> None:
    result = console().console_loop(
        lambda *_: answer("ok"),
        tmp_path,
        read=reader(["prompt"]),
        stream=io.StringIO(),
        max_requests=1,
    )
    assert result["completed_requests"] == 1
    assert result["stop_reason"] == "limit"


@pytest.mark.parametrize("value", [0, 101, True])
def test_bad_session_limit_is_rejected_before_writes(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError, match="Session limit"):
        console().console_loop(lambda *_: answer("unused"), tmp_path, max_requests=value)
    assert not (tmp_path / "session.json").exists()


def test_existing_session_receipt_never_overwritten(tmp_path: Path) -> None:
    receipt = tmp_path / "session.json"
    receipt.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        console().console_loop(
            lambda *_: answer("unused"), tmp_path, read=reader(["/exit"]), stream=io.StringIO()
        )
    assert receipt.read_text("utf-8") == "original"


def test_failed_kit_never_loads_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = console()
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(
        module.subprocess, "run", lambda *_, **__: CompletedProcess([], 1, "", "drift")
    )

    def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("Model must not load after failed verification")

    monkeypatch.setattr(module.backend, "main", never)
    assert module.main([*MAIN_INPUTS, "--out", str(tmp_path / "session")]) == 1


def test_main_loads_backend_once_and_uses_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = console()
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(module.subprocess, "run", lambda *_, **__: CompletedProcess([], 0, "", ""))
    calls: list[str] = []

    def backend_main(argv: list[str], *, serve: Any, quiet: bool) -> int:
        assert quiet and "--out" in argv and "--prompt" not in argv
        calls.append("load")
        serve(lambda prompt, _: answer(prompt))
        return 0

    monkeypatch.setattr(module.backend, "main", backend_main)
    monkeypatch.setattr(
        module,
        "console_loop",
        lambda generate, *_args, **_kw: [generate("a", tmp_path), generate("b", tmp_path)],
    )
    assert module.main([*MAIN_INPUTS, "--out", str(tmp_path / "session")]) == 0
    assert calls == ["load"]
