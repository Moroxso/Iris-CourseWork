"""Real pipe tests use fixed trusted test programs, never model/reference code."""

from __future__ import annotations

import asyncio
import sys
import traceback
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iris.integrations import bounded_process as process

_LIMITS = process.ProcessLimits(
    max_input_bytes=8 * 1024 * 1024,
    max_output_bytes=65536,
    startup_timeout_ms=5000,
    execution_timeout_ms=5000,
    cleanup_timeout_ms=3000,
)
_ECHO = (
    "import sys\ndata = sys.stdin.buffer.read()\n"
    "while data: data = data[sys.stdout.buffer.write(data):]"
)
_IDLE = "import time; time.sleep(60)"
_PRIVATE = "PRIVATE_IRIS_TRANSPORT_TEST_MARKER"


def _argv(source: str) -> tuple[str, ...]:
    # This helper is confined to tests and supplied with fixed trusted snippets.
    return sys.executable, "-I", "-S", "-B", "-u", "-c", source


def _run(source: str, data: bytes = b"", **kwargs: Any) -> process.ProcessResult:
    return asyncio.run(
        process.run_bounded_process(
            _argv(source), input_bytes=data, limits=kwargs.pop("limits", _LIMITS), **kwargs
        )
    )


@pytest.mark.parametrize("data", [b"", b"a\r\n\0\xff\xfe", "Привет\n☃".encode()])
def test_binary_input_output_is_exact_and_not_decoded(data: bytes) -> None:
    observed = _run(_ECHO, data)
    assert observed == process.ProcessResult(0, data, b"", "exited")


def test_binary_echo_larger_than_native_pipe_buffers_is_complete() -> None:
    data = bytes(range(256)) * 8192
    observed = _run(_ECHO, data, limits=replace(_LIMITS, max_output_bytes=len(data)))
    assert observed == process.ProcessResult(0, data, b"", "exited")


def test_nonzero_exit_preserves_both_streams_without_raising() -> None:
    observed = _run(
        "import sys; sys.stdout.buffer.write(b'out'); "
        "sys.stderr.buffer.write(b'err\\xff'); sys.exit(17)"
    )
    assert observed == process.ProcessResult(17, b"out", b"err\xff", "exited")


def test_output_exactly_at_the_combined_limit_is_not_overflow() -> None:
    observed = _run(
        "import sys; sys.stdout.buffer.write(b'x' * 32768); sys.stderr.buffer.write(b'y' * 32768)"
    )
    assert observed.termination == "exited"
    assert observed.returncode == 0
    assert observed.stdout == b"x" * 32768
    assert observed.stderr == b"y" * 32768


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_one_extra_byte_exceeds_the_single_combined_budget(stream: str) -> None:
    observed = _run(f"import sys; sys.{stream}.buffer.write(b'x' * 65537)")
    assert observed.termination == "output_limit"
    assert len(observed.stdout) + len(observed.stderr) == 65536


def test_infinite_output_is_cut_off_and_does_not_fill_host_memory() -> None:
    observed = _run("import os\nwhile True: os.write(1, b'x' * 8192)")
    assert observed.termination == "output_limit"
    assert observed.stdout == b"x" * 65536
    assert observed.stderr == b""


def test_both_streams_are_drained_while_large_input_is_written() -> None:
    observed = _run(
        "import sys\n"
        "while chunk := sys.stdin.buffer.read(1024):\n"
        "    sys.stdout.buffer.write(chunk)\n"
        "    sys.stderr.buffer.write(chunk)\n",
        b"z" * 512 * 1024,
        limits=replace(_LIMITS, max_output_bytes=1024 * 1024),
    )
    assert observed.termination == "exited"
    assert observed.returncode == 0
    assert observed.stdout == observed.stderr == b"z" * 512 * 1024


@pytest.mark.parametrize(
    "input_bytes", [b"", b"x" * (8 * 1024 * 1024)], ids=["idle", "backpressure"]
)
def test_timeout_handles_idle_and_stdin_backpressure(input_bytes: bytes) -> None:
    observed = _run(
        _IDLE,
        input_bytes,
        limits=replace(_LIMITS, execution_timeout_ms=250),
    )
    assert observed.termination == "timeout"
    assert observed.returncode != 0
    assert observed.stdout == observed.stderr == b""


def test_child_may_exit_without_consuming_all_input() -> None:
    observed = _run("raise SystemExit(3)", b"x" * (8 * 1024 * 1024))
    assert observed.termination == "exited"
    assert observed.returncode == 3


def test_input_is_released_only_after_successful_gate() -> None:
    stages: list[str] = []
    real_feed = process._Protocol.feed

    async def gate() -> None:
        assert stages == []
        await asyncio.sleep(0.05)
        stages.append("verified")

    async def feed(protocol: process._Protocol, data: bytes) -> None:
        assert stages == ["verified"]
        stages.append("input")
        await real_feed(protocol, data)

    async def stop() -> None:
        assert stages == ["verified", "input"]
        stages.append("remote cleanup")

    with patch.object(process._Protocol, "feed", feed):
        observed = _run(_ECHO, b"private", before_input=gate, before_stop=stop)
    assert observed.stdout == b"private"
    assert stages == ["verified", "input", "remote cleanup"]


@pytest.mark.parametrize("failure", ["exception", "timeout", "early_exit", "output_limit"])
def test_failed_gate_never_releases_input_and_still_cleans_up(failure: str) -> None:
    cleaned: list[bool] = []
    tracked: list[process._Protocol] = []
    real_protocol = process._Protocol

    def factory(limit: int) -> process._Protocol:
        instance = real_protocol(limit)
        tracked.append(instance)
        return instance

    async def gate() -> None:
        if failure == "exception":
            raise RuntimeError(_PRIVATE)
        if failure == "timeout":
            await asyncio.sleep(60)
        else:
            await asyncio.shield(tracked[0].done)

    async def stop() -> None:
        cleaned.append(True)

    source = _IDLE
    if failure == "early_exit":
        source = "pass"
    elif failure == "output_limit":
        source = "import sys; sys.stdout.buffer.write(b'x' * 65537)"
    with (
        patch.object(real_protocol, "feed", new_callable=AsyncMock) as feed,
        patch.object(process, "_Protocol", factory),
    ):
        with pytest.raises(process.BoundedProcessError) as caught:
            _run(
                source,
                _PRIVATE.encode(),
                limits=replace(_LIMITS, startup_timeout_ms=150 if failure == "timeout" else 5000),
                before_input=gate,
                before_stop=stop,
            )
        feed.assert_not_called()
    assert cleaned == [True]
    assert _PRIVATE not in "".join(traceback.format_exception(caught.value))


def test_remote_cleanup_precedes_local_client_termination() -> None:
    tracked: list[process._Protocol] = []
    real_protocol = process._Protocol

    def factory(limit: int) -> process._Protocol:
        instance = real_protocol(limit)
        tracked.append(instance)
        return instance

    async def stop() -> None:
        assert tracked[0].transport is not None
        assert tracked[0].transport.get_returncode() is None
        assert not tracked[0].closed.done()

    with patch.object(process, "_Protocol", factory):
        observed = _run(
            _IDLE,
            before_stop=stop,
            limits=replace(_LIMITS, execution_timeout_ms=200),
        )
    assert observed.termination == "timeout"
    assert tracked[0].closed.done()
    assert tracked[0].transport is not None
    assert tracked[0].transport.get_returncode() is not None


@pytest.mark.parametrize("failure", ["exception", "timeout"])
def test_failed_remote_cleanup_rejects_result_and_closes_local_transport(failure: str) -> None:
    closed: list[bool] = []
    real_close = process._close

    async def stop() -> None:
        if failure == "exception":
            raise RuntimeError(_PRIVATE)
        await asyncio.sleep(60)

    async def close(protocol: process._Protocol, timeout_ms: int) -> None:
        await real_close(protocol, timeout_ms)
        assert protocol.closed.done()
        closed.append(True)

    with (
        patch.object(process, "_close", close),
        pytest.raises(process.BoundedProcessError) as caught,
    ):
        _run(
            _ECHO,
            before_stop=stop,
            limits=replace(_LIMITS, cleanup_timeout_ms=200),
        )
    assert closed == [True]
    assert _PRIVATE not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("phase", ["gate", "exchange"])
def test_cancellation_is_preserved_after_remote_and_local_cleanup(phase: str) -> None:
    events: list[str] = []

    async def scenario() -> None:
        ready = asyncio.Event()

        async def gate() -> None:
            ready.set()
            if phase == "gate":
                await asyncio.sleep(60)

        async def stop() -> None:
            events.append("remote cleanup")

        task = asyncio.create_task(
            process.run_bounded_process(
                _argv(_IDLE),
                input_bytes=b"private",
                limits=_LIMITS,
                before_input=gate,
                before_stop=stop,
            )
        )
        await ready.wait()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario(), debug=True)
    assert events == ["remote cleanup"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_input_bytes", 0),
        ("max_input_bytes", True),
        ("max_output_bytes", 8 * 1024 * 1024 + 1),
        ("startup_timeout_ms", 60001),
        ("execution_timeout_ms", 600001),
        ("cleanup_timeout_ms", -1),
        ("execution_timeout_ms", 1.0),
    ],
)
def test_invalid_budgets_fail_before_spawning_or_hooks(field: str, value: Any) -> None:
    async def scenario() -> None:
        with patch.object(asyncio.get_running_loop(), "subprocess_exec") as spawn:
            hook = AsyncMock()
            with pytest.raises(process.BoundedProcessError):
                await process.run_bounded_process(
                    _argv(_ECHO),
                    input_bytes=b"",
                    limits=replace(_LIMITS, **{field: value}),
                    before_input=hook,
                    before_stop=hook,
                )
            spawn.assert_not_called()
            hook.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("argv", "data"),
    [
        ([], b""),
        ((), b""),
        (("",), b""),
        (("x\0y",), b""),
        (("x",), bytearray()),
        (("x",), "text"),
        (("x",), b"x" * (8 * 1024 * 1024 + 1)),
        (("x", "x" * (128 * 1024)), b""),
        (("x", "\ud800"), b""),
    ],
    ids=[
        "list",
        "empty",
        "empty-exe",
        "nul",
        "mutable",
        "text",
        "large-input",
        "large-argv",
        "utf8",
    ],
)
def test_invalid_argv_or_input_is_redacted_before_spawning(argv: Any, data: Any) -> None:
    async def scenario() -> None:
        with patch.object(asyncio.get_running_loop(), "subprocess_exec") as spawn:
            with pytest.raises(process.BoundedProcessError):
                await process.run_bounded_process(argv, input_bytes=data, limits=_LIMITS)
            spawn.assert_not_called()

    asyncio.run(scenario())


def test_spawn_failure_runs_owner_cleanup_and_redacts_native_error() -> None:
    async def scenario() -> None:
        hook = AsyncMock()
        with (
            patch.object(
                asyncio.get_running_loop(), "subprocess_exec", side_effect=OSError(_PRIVATE)
            ),
            pytest.raises(process.BoundedProcessError) as caught,
        ):
            await process.run_bounded_process(
                (_PRIVATE,), input_bytes=b"", limits=_LIMITS, before_stop=hook
            )
        hook.assert_awaited_once_with()
        assert _PRIVATE not in "".join(traceback.format_exception(caught.value))

    asyncio.run(scenario())


def test_local_cleanup_failure_is_not_an_accepted_observation() -> None:
    real_close = process._close

    async def close(protocol: process._Protocol, timeout_ms: int) -> None:
        await real_close(protocol, timeout_ms)
        raise RuntimeError(_PRIVATE)

    with (
        patch.object(process, "_close", close),
        pytest.raises(process.BoundedProcessError) as caught,
    ):
        _run(_ECHO)
    assert _PRIVATE not in "".join(traceback.format_exception(caught.value))


def test_repeated_cancellation_does_not_abandon_cleanup() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        completed = asyncio.Event()

        async def stop() -> None:
            entered.set()
            await asyncio.sleep(0.1)
            completed.set()

        task = asyncio.create_task(
            process.run_bounded_process(
                _argv(_ECHO), input_bytes=b"", limits=_LIMITS, before_stop=stop
            )
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario(), debug=True)


def test_process_exit_does_not_discard_output_arriving_before_eof() -> None:
    async def scenario() -> None:
        protocol = process._Protocol(100)
        protocol.process_exited()
        assert not protocol.done.done()
        protocol.pipe_data_received(1, b"last stdout")
        protocol.pipe_connection_lost(1, None)
        assert not protocol.done.done()
        protocol.pipe_data_received(2, b"last stderr")
        protocol.pipe_connection_lost(2, None)
        assert protocol.done.result() == "exited"
        assert protocol.stdout == b"last stdout"
        assert protocol.stderr == b"last stderr"

    asyncio.run(scenario())


@pytest.mark.parametrize("fd", [0, 1, 2])
def test_pipe_io_failure_is_an_infrastructure_error(fd: int) -> None:
    real_feed = process._Protocol.feed

    async def feed(protocol: process._Protocol, data: bytes) -> None:
        protocol.pipe_connection_lost(fd, OSError(_PRIVATE))
        await real_feed(protocol, data)

    with (
        patch.object(process._Protocol, "feed", feed),
        pytest.raises(process.BoundedProcessError) as caught,
    ):
        _run(_IDLE)
    assert _PRIVATE not in "".join(traceback.format_exception(caught.value))


def test_backpressure_keeps_the_write_buffer_bounded() -> None:
    sizes: list[int] = []
    real_pause = process._Protocol.pause_writing

    def pause(protocol: process._Protocol) -> None:
        assert protocol.transport is not None
        pipe = protocol.transport.get_pipe_transport(0)
        assert isinstance(pipe, asyncio.WriteTransport)
        sizes.append(pipe.get_write_buffer_size())
        real_pause(protocol)

    with patch.object(process._Protocol, "pause_writing", pause):
        observed = _run(
            _IDLE,
            b"x" * (8 * 1024 * 1024),
            limits=replace(_LIMITS, execution_timeout_ms=250),
        )
    assert observed.termination == "timeout"
    assert sizes and max(sizes) <= 3 * process._CHUNK


def test_overflow_discards_later_bytes_without_growing_retained_payload() -> None:
    async def scenario() -> None:
        protocol = process._Protocol(3)
        protocol.pipe_data_received(1, b"abcd")
        assert protocol.done.result() == "output_limit"
        for _ in range(16):
            protocol.pipe_data_received(1, b"x" * 8192)
            protocol.pipe_data_received(2, b"x" * 8192)
        assert protocol.stdout == b"abc" and protocol.stderr == b""

    asyncio.run(scenario())
