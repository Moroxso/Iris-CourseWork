"""Authored trace validation vectors; no models, commands or candidate execution."""

import pytest

from tests.integration._sandbox_startup_trace import assert_safe_sandbox_startup_trace


def _trace(failures: int = 0) -> list[dict[str, object]]:
    phases = ["test"] + ["info"] * (failures + 1) + ["dpkg", "runsc", "image", "uname"]
    phases += [
        "create",
        "container",
        "container",
        "readlink",
        "cat",
        "cat",
        "cat",
        "cat",
        "cat",
        "container",
        "container",
        "ps",
        "container",
        "rm",
        "ps",
    ]
    result: list[dict[str, object]] = [{"phase": phase, "passed": True} for phase in phases]
    for i in range(1, failures + 1):
        result[i] = {"phase": "info", "passed": False, "type": "PythonSandboxError"}
    return result


@pytest.mark.parametrize("failures", (0, 1, 2))
def test_exact_zero_one_two_initial_info_failures_are_allowed_without_erasure(
    failures: int,
) -> None:
    trace = _trace(failures)
    before = [dict(event) for event in trace]
    assert_safe_sandbox_startup_trace(trace)
    assert trace == before


@pytest.mark.parametrize(
    "phase",
    (
        "test",
        "dpkg",
        "runsc",
        "image",
        "uname",
        "create",
        "readlink",
        "cat",
        "container",
        "ps",
        "rm",
    ),
)
def test_no_other_preflight_execution_or_cleanup_failure_is_accepted(phase: str) -> None:
    trace = _trace(2)
    index = next(i for i, event in enumerate(trace) if event["phase"] == phase)
    trace[index] = {"phase": phase, "passed": False, "type": "PythonSandboxError"}
    with pytest.raises(AssertionError):
        assert_safe_sandbox_startup_trace(trace)


@pytest.mark.parametrize(
    "change",
    (
        "empty",
        "tuple",
        "not_object",
        "bool_status",
        "wrong_phase_type",
        "extra_key",
        "wrong_error",
        "missing_error",
        "successful_error",
        "too_many_info",
        "extra_success",
        "no_success",
        "missing_create",
        "duplicate_create",
        "missing_rm",
        "duplicate_rm",
        "early_rm",
        "wrong_prefix",
        "info_after_create",
        "info_after_rm",
        "too_long",
    ),
)
def test_partial_reordered_replayed_or_relaxed_traces_fail_closed(change: str) -> None:
    trace = _trace(1)
    candidate: object = trace
    if change == "empty":
        candidate = []
    elif change == "tuple":
        candidate = tuple(trace)
    elif change == "not_object":
        candidate = [*trace, "not an observation"]
    elif change == "bool_status":
        trace[0]["passed"] = 1
    elif change == "wrong_phase_type":
        trace[0]["phase"] = None
    elif change == "extra_key":
        trace[0]["opaque"] = True
    elif change == "wrong_error":
        trace[1]["type"] = "OtherError"
    elif change == "missing_error":
        del trace[1]["type"]
    elif change == "successful_error":
        trace[0]["type"] = "PythonSandboxError"
    elif change == "too_many_info":
        candidate = _trace(3)
    elif change == "extra_success":
        trace[1] = {"phase": "info", "passed": True}
    elif change == "no_success":
        trace[2] = {"phase": "info", "passed": False, "type": "PythonSandboxError"}
    elif change == "missing_create":
        trace[:] = [event for event in trace if event["phase"] != "create"]
    elif change == "duplicate_create":
        trace.append({"phase": "create", "passed": True})
    elif change == "missing_rm":
        trace[:] = [event for event in trace if event["phase"] != "rm"]
    elif change == "duplicate_rm":
        trace.append({"phase": "rm", "passed": True})
    elif change == "early_rm":
        trace[:] = [event for event in trace if event["phase"] != "rm"]
        trace.insert(0, {"phase": "rm", "passed": True})
    elif change == "wrong_prefix":
        trace[0], trace[1] = trace[1], trace[0]
    elif change == "info_after_create":
        trace.insert(
            next(i for i, e in enumerate(trace) if e["phase"] == "rm"),
            {"phase": "info", "passed": True},
        )
    elif change == "info_after_rm":
        trace.append({"phase": "info", "passed": True})
    elif change == "too_long":
        trace.extend([{"phase": "ps", "passed": True}] * 128)
    else:
        raise AssertionError("Uncovered authored validation vector")
    with pytest.raises(AssertionError):
        assert_safe_sandbox_startup_trace(candidate)
