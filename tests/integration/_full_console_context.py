"""Explicit console-harness binding, NOT execution attestation or a fake client.

Ordinary collection imports no operational files. A real owned process runner
must be installed by the integration harness; missing setup fails, never skips.
Concrete installed-console execution/guards remain a separate required gate.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Protocol


class ConsoleClient(Protocol):
    base: Path
    child_totals: dict[str, int]
    jobs: list[dict[str, Any]]

    def run(
        self,
        argv: list[str],
        *,
        effect: str,
        expected: int = 0,
        fixture_manifest_sha: str | None = None,
    ) -> dict[str, Any]: ...


_CLIENT: ContextVar[ConsoleClient | None] = ContextVar("iris_actual_console_client", default=None)
_FIXTURE: ContextVar[bool] = ContextVar("iris_console_random_fixture_active", default=False)


def client() -> ConsoleClient:
    current = _CLIENT.get()
    if current is None:
        raise RuntimeError(
            "A real owned installed-console runner is required. Bind it with "
            "use_console_client in the integration harness; do not replace it with CliRunner "
            "or fabricated child receipts. Collection alone is not console acceptance."
        )
    return current


def fixture_is_active() -> bool:
    return _FIXTURE.get()


@contextmanager
def use_console_client(current: ConsoleClient) -> Iterator[None]:
    if _CLIENT.get() is not None or _FIXTURE.get():
        raise RuntimeError("Console context cannot replace an active owned client")
    token = _CLIENT.set(current)
    try:
        yield
    finally:
        _CLIENT.reset(token)


@contextmanager
def fixture_scope() -> Iterator[None]:
    if _FIXTURE.get() or (client().base / "tiny-hf-cache").exists():
        raise RuntimeError("A fresh random-only console fixture scope is required")
    token = _FIXTURE.set(True)
    try:
        yield
    finally:
        _FIXTURE.reset(token)
