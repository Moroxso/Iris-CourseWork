"""Bounded binary I/O for a trusted controller command, NOT code isolation.

Only trusted executable/argument tuples belong here. Candidate source must never
be a host Python program or shell argument. A future sandbox adapter must own
its container and supply an independently verified input gate and remote cleanup.
Killing a Docker/WSL client alone does not establish container termination.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from typing import Literal, cast

_CHUNK = 8192
_MAX_BYTES = 8 * 1024 * 1024
_Hook = Callable[[], Awaitable[None]]
_Signal = Literal["exited", "output_limit", "io_error"]


class BoundedProcessError(RuntimeError):
    """A controller, its transport, or its cleanup failed; never a model score."""


@dataclass(frozen=True, slots=True)
class ProcessLimits:
    """Explicit engineering budgets; these do not define a scientific policy.

    Output is one combined stdout/stderr byte allowance. The execution clock
    begins only after the input gate succeeds. Startup includes that gate;
    cleanup has separate allowances for the owner hook and local pipe closure.
    """

    max_input_bytes: int
    max_output_bytes: int
    startup_timeout_ms: int
    execution_timeout_ms: int
    cleanup_timeout_ms: int


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Private, bounded raw observations, not execution/scoring attestation.

    Nonzero exit is an observation, not a transport exception. On timeout or
    overflow, returncode describes the local client after cleanup, not its job.
    """

    returncode: int
    stdout: bytes
    stderr: bytes
    termination: Literal["exited", "timeout", "output_limit"]


def _validate(argv: tuple[str, ...], input_bytes: bytes, limits: ProcessLimits) -> ProcessLimits:
    if (
        type(argv) is not tuple
        or not 1 <= len(argv) <= 128
        or any(type(arg) is not str or "\0" in arg for arg in argv)
        or not argv[0]
        or sum(len(arg.encode("utf-8")) for arg in argv) > 128 * 1024
        or type(input_bytes) is not bytes
        or type(limits) is not ProcessLimits
    ):
        raise ValueError("Invalid process arguments")
    owned = ProcessLimits(**{field.name: getattr(limits, field.name) for field in fields(limits)})
    for field in fields(owned):
        value = getattr(owned, field.name)
        ceiling = _MAX_BYTES if field.name.endswith("bytes") else 60_000
        if field.name == "execution_timeout_ms":
            ceiling = 600_000
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError("Invalid process budget")
    if len(input_bytes) > owned.max_input_bytes:
        raise ValueError("Input exceeds the explicit budget")
    return owned


class _Protocol(asyncio.SubprocessProtocol):
    def __init__(self, max_output_bytes: int) -> None:
        loop = asyncio.get_running_loop()
        self.transport: asyncio.SubprocessTransport | None = None
        self.done: asyncio.Future[_Signal] = loop.create_future()
        self.closed: asyncio.Future[None] = loop.create_future()
        self.writable = asyncio.Event()
        self.writable.set()
        self.input_closed = False
        self.output_closed: set[int] = set()
        self.exited = False
        self.max_output_bytes = max_output_bytes
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.io_failed = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.SubprocessTransport, transport)
        pipe = cast(asyncio.WriteTransport, self.transport.get_pipe_transport(0))
        pipe.set_write_buffer_limits(high=2 * _CHUNK, low=_CHUNK)

    def _signal(self, signal: _Signal) -> None:
        if not self.done.done():
            self.done.set_result(signal)

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if self.done.done() or fd not in (1, 2):
            return
        remaining = self.max_output_bytes - len(self.stdout) - len(self.stderr)
        target = self.stdout if fd == 1 else self.stderr
        target.extend(data[:remaining])
        if len(data) > remaining:
            self._signal("output_limit")
            # Continue draining, discarding every subsequent chunk above. A
            # paused attach pipe can block Docker's job-exit/kill completion.
            # Retention stops here; owner cleanup bounds the remaining draining.

    def pause_writing(self) -> None:
        self.writable.clear()

    def resume_writing(self) -> None:
        self.writable.set()

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if fd == 0:
            # A program may exit or close stdin without consuming its input.
            self.input_closed = True
            self.writable.set()
            if exc is not None and not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.io_failed = True
                self._signal("io_error")
        elif fd in (1, 2):
            self.output_closed.add(fd)
            if exc is not None:
                self.io_failed = True
                self._signal("io_error")
        self._finish_if_drained()

    def process_exited(self) -> None:
        self.exited = True
        self._finish_if_drained()

    def _finish_if_drained(self) -> None:
        # process_exited can precede the final output callbacks. Do not lose them.
        if self.exited and self.output_closed == {1, 2}:
            self._signal("exited")

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            self.io_failed = True
            self._signal("io_error")
        if not self.closed.done():
            self.closed.set_result(None)

    async def feed(self, data: bytes) -> None:
        if self.transport is None:
            raise RuntimeError("Controller transport is absent")
        pipe = cast(asyncio.WriteTransport, self.transport.get_pipe_transport(0))
        for offset in range(0, len(data), _CHUNK):
            await self.writable.wait()
            if self.input_closed or self.done.done():
                return
            pipe.write(data[offset : offset + _CHUNK])
            # Permit output/timeout processing even when writes need no drain.
            await asyncio.sleep(0)
        if not self.input_closed:
            pipe.close()  # Flush this bounded buffer, then deliver EOF.


async def _exchange(
    protocol: _Protocol, input_bytes: bytes, timeout_ms: int
) -> Literal["exited", "timeout", "output_limit"]:
    feeder = asyncio.create_task(protocol.feed(input_bytes))
    try:
        async with asyncio.timeout(timeout_ms / 1000):
            await asyncio.wait((feeder, protocol.done), return_when=asyncio.FIRST_COMPLETED)
            if feeder.done():
                feeder.result()
            signal = await asyncio.shield(protocol.done)
            if signal == "io_error":
                raise BoundedProcessError("Controller I/O failed")
            return signal
    except TimeoutError:
        return "timeout"
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)


async def _close(protocol: _Protocol, timeout_ms: int) -> None:
    transport = protocol.transport
    if transport is None:
        return
    # Explicitly close inherited pipes too: waiting for client exit is not EOF
    # when a child retains the handles. No blocked reader/writer threads exist.
    for fd in (0, 1, 2):
        pipe = transport.get_pipe_transport(fd)
        if pipe is not None:
            if fd == 0:
                cast(asyncio.WriteTransport, pipe).abort()
            else:
                pipe.close()
    transport.close()  # Documented to kill the local subprocess if still running.
    async with asyncio.timeout(timeout_ms / 1000):
        await asyncio.shield(protocol.closed)
    if transport.get_returncode() is None:
        raise BoundedProcessError("Controller termination was not confirmed")


async def _finalize(protocol: _Protocol, hook: _Hook | None, timeout_ms: int) -> None:
    async def cleanup() -> None:
        try:
            if hook is not None:
                async with asyncio.timeout(timeout_ms / 1000):
                    await hook()
        finally:
            await _close(protocol, timeout_ms)

    # Caller cancellation (including a second cancel during cleanup) must not
    # abandon the owner's remote cleanup. Keep its own explicit timeout active.
    task = asyncio.create_task(cleanup())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()  # Cleanup failure takes precedence over an accepted result.
    if cancelled:
        raise asyncio.CancelledError


async def run_bounded_process(
    argv: tuple[str, ...],
    *,
    input_bytes: bytes,
    limits: ProcessLimits,
    before_input: _Hook | None = None,
    before_stop: _Hook | None = None,
) -> ProcessResult:
    """Observe one trusted argv without a shell or unbounded output collection.

    Hooks are trusted, cooperative async controller code, never candidate code.
    before_input must verify the live sandbox before any input byte is released.
    before_stop runs on success, failure, timeout and cancellation, BEFORE the
    local client is closed; it must stop/remove and verify the owned remote job.
    Hook/transport/cleanup failure raises a redacted infrastructure error. A
    cleanup failure cannot turn a partial observation into an accepted result.

    Uses public asyncio transports (Windows requires a Proactor event loop).
    Budgets require a responsive OS/event loop and cooperative hooks; they are
    not hard real-time guarantees or protection against host/controller failure.
    The caller owns crash recovery and descendant/container lifecycle. This
    helper does not install runtimes, pull images, execute source, or score code.
    """
    try:
        owned = _validate(argv, input_bytes, limits)
        if any(hook is not None and not callable(hook) for hook in (before_input, before_stop)):
            raise TypeError("Invalid controller hook")
        protocol = _Protocol(owned.max_output_bytes)
        try:
            async with asyncio.timeout(owned.startup_timeout_ms / 1000):
                await asyncio.get_running_loop().subprocess_exec(
                    lambda: protocol,
                    *argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                )
                if before_input is not None:
                    await before_input()
                    if protocol.done.done():
                        raise BoundedProcessError("Controller ended before input was authorized")
            termination = await _exchange(protocol, input_bytes, owned.execution_timeout_ms)
        finally:
            await _finalize(protocol, before_stop, owned.cleanup_timeout_ms)
        if protocol.io_failed or protocol.transport is None:
            raise BoundedProcessError("Controller I/O failed")
        code = protocol.transport.get_returncode()
        if code is None:
            raise BoundedProcessError("Controller termination was not confirmed")
        return ProcessResult(code, bytes(protocol.stdout), bytes(protocol.stderr), termination)
    except Exception:
        # In particular, do not propagate argv, private bytes, native stderr,
        # or a trusted hook's potentially data-bearing exception text.
        raise BoundedProcessError(
            "Bounded controller execution could not complete safely"
        ) from None


__all__ = ["BoundedProcessError", "ProcessLimits", "ProcessResult", "run_bounded_process"]
