"""LABELLED context/wire vectors, never a fabricated actual console witness."""

from pathlib import Path
from typing import Any

import pytest

from tests.integration._full_console_context import (
    client,
    fixture_is_active,
    fixture_scope,
    use_console_client,
)


class LabelledClient:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.child_totals: dict[str, int] = {}
        self.jobs: list[dict[str, Any]] = []

    def run(
        self,
        argv: list[str],
        *,
        effect: str,
        expected: int = 0,
        fixture_manifest_sha: str | None = None,
    ) -> dict[str, Any]:
        raise AssertionError("No genuine console client in a labelled context unit test")


def test_unconfigured_console_is_a_clear_failure_not_skip_or_mock() -> None:
    with pytest.raises(RuntimeError, match="real owned installed-console"):
        client()
    with pytest.raises(RuntimeError), fixture_scope():
        raise AssertionError("Must fail before entering")


def test_binding_and_random_fixture_scope_restore_after_signal(tmp_path: Path) -> None:
    labelled = LabelledClient(tmp_path)
    signal = KeyboardInterrupt()
    with (
        pytest.raises(KeyboardInterrupt) as caught,
        use_console_client(labelled),
        fixture_scope(),
    ):
        assert client() is labelled and fixture_is_active()
        raise signal
    assert caught.value is signal and not fixture_is_active()
    with pytest.raises(RuntimeError):
        client()


def test_nested_bindings_and_existing_fixture_are_refused(tmp_path: Path) -> None:
    labelled = LabelledClient(tmp_path)
    with use_console_client(labelled):
        with pytest.raises(RuntimeError), use_console_client(labelled):
            raise AssertionError("No nested client")
        with fixture_scope(), pytest.raises(RuntimeError), fixture_scope():
            raise AssertionError("No nested fixture")
        (tmp_path / "tiny-hf-cache").mkdir()
        with pytest.raises(RuntimeError), fixture_scope():
            raise AssertionError("Existing fixture is not fresh")


def test_new_context_does_not_inherit_owned_client(tmp_path: Path) -> None:
    from contextvars import Context

    labelled = LabelledClient(tmp_path)
    with use_console_client(labelled), pytest.raises(RuntimeError):
        Context().run(client)


@pytest.mark.parametrize(
    "wire",
    (
        b"{}\n",
        b"{}",
        b"{}\r\n\r\n",
        b" []\r\n",
        b"null\r\n",
        b'{"x":1, "x":2}\r\n',
        b'{"x":NaN}\r\n',
        b"\xff\r\n",
    ),
)
def test_wrong_console_framing_duplicates_and_nonfinite_fail(wire: bytes) -> None:
    from tests.integration._console_json_wire import windows_json_object

    with pytest.raises(ValueError):
        windows_json_object(wire)


def test_exact_windows_json_is_not_normalized() -> None:
    from tests.integration._console_json_wire import windows_json_object

    wire = '{"x":"Ирис"}\r\n'.encode()
    assert windows_json_object(wire) == {"x": "Ирис"}
