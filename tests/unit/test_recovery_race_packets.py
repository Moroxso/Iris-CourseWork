"""LABELLED packet/storage fences; NO worker, Iris, model or native race proof."""

import copy
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.integration import _recovery_race_packets as wire


def specimen(tmp_path: Path, scenario: str = "partial") -> tuple[dict[str, Any], dict[str, Any]]:
    binding = dict(
        job_sha256="a" * 64, command_sha256="b" * 64, nonce="c" * 32, phase=0, scenario=scenario
    )
    value = dict(
        kind="iris-recovery-race-ready-v1",
        binding=binding,
        pid=123,
        parent_pid=456,
        started_sha256="d" * 64,
        source_sha256="e" * 64,
        worker_sha256="f" * 64,
        rows=1 if scenario == "partial" else 4,
        locks={
            key: dict(
                path=str(tmp_path / key / ".iris-owner.lock"),
                signature=[1, 2 if key == "workflow" else 3, 1, 1, 4, 5],
                content_sha256="6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
            )
            for key in ("workflow", "kernel")
        },
    )
    return binding, value


@pytest.mark.parametrize("scenario", ("partial", "aggregate"))
def test_complete_packet_publication_readback_and_no_overwrite(
    tmp_path: Path, scenario: str
) -> None:
    binding, value = specimen(tmp_path, scenario)
    digest = wire.publish(tmp_path, "ready", value, binding)
    assert wire.read(tmp_path, "ready", binding) == (value, digest)
    assert {p.name for p in tmp_path.iterdir()} == {"ready"}
    before = (tmp_path / "ready/packet.json").read_bytes()
    with pytest.raises(FileExistsError):
        wire.publish(tmp_path, "ready", value, binding)
    assert (tmp_path / "ready/packet.json").read_bytes() == before
    release = dict(
        kind="iris-recovery-race-release-v1",
        binding=binding,
        ready_sha256=digest,
        contender_process_sha256="1" * 64,
    )
    released = wire.publish(tmp_path, "release", release, binding)
    assert wire.read(tmp_path, "release", binding) == (release, released)


@pytest.mark.parametrize("field", ("job_sha256", "command_sha256", "nonce", "phase", "scenario"))
def test_every_job_binding_field_is_bound(tmp_path: Path, field: str) -> None:
    binding, value = specimen(tmp_path)
    foreign = binding.copy()
    foreign[field] = (
        1
        if field == "phase"
        else "aggregate"
        if field == "scenario"
        else "0" * (32 if field == "nonce" else 64)
    )
    with pytest.raises(ValueError):
        wire.checked(value, foreign, "ready")


@pytest.mark.parametrize(
    "fault",
    (
        "not_dict",
        "extra",
        "missing",
        "upper_digest",
        "short_digest",
        "bool_phase",
        "negative_phase",
        "large_phase",
        "foreign_scenario",
        "nonstring_scenario",
        "nonstring_nonce",
    ),
)
def test_closed_binding_refusals(tmp_path: Path, fault: str) -> None:
    binding, _ = specimen(tmp_path)
    value: Any = binding.copy()
    if fault == "not_dict":
        value = []
    elif fault == "extra":
        value["owner"] = "no authority"
    elif fault == "missing":
        del value["nonce"]
    elif fault == "upper_digest":
        value["job_sha256"] = "A" * 64
    elif fault == "short_digest":
        value["command_sha256"] = "b" * 63
    elif fault == "bool_phase":
        value["phase"] = True
    elif fault == "negative_phase":
        value["phase"] = -1
    elif fault == "large_phase":
        value["phase"] = 4
    elif fault == "foreign_scenario":
        value["scenario"] = "warm"
    elif fault == "nonstring_scenario":
        value["scenario"] = False
    else:
        value["nonce"] = 1
    with pytest.raises(ValueError):
        wire.binding(value)


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "missing",
        "kind",
        "bool_pid",
        "zero_pid",
        "same_pid",
        "bool_rows",
        "wrong_rows",
        "missing_lock",
        "same_locks",
        "relative_lock",
        "dotdot_lock",
        "hardlinked_lock",
        "wide_lock",
        "bool_stamp",
        "negative_stamp",
        "short_stamp",
        "wrong_lock_bytes",
        "upper_digest",
        "too_large",
    ),
)
def test_ready_refuses_constructible_wrong_packet(tmp_path: Path, fault: str) -> None:
    binding, value = specimen(tmp_path)
    value = copy.deepcopy(value)
    lock = value["locks"]["kernel"]
    if fault == "extra":
        value["native_owner"] = {}
    elif fault == "missing":
        del value["rows"]
    elif fault == "kind":
        value["kind"] = "other"
    elif fault == "bool_pid":
        value["pid"] = True
    elif fault == "zero_pid":
        value["parent_pid"] = 0
    elif fault == "same_pid":
        value["pid"] = value["parent_pid"]
    elif fault == "bool_rows":
        value["rows"] = True
    elif fault == "wrong_rows":
        value["rows"] = 4
    elif fault == "missing_lock":
        del value["locks"]["workflow"]
    elif fault == "same_locks":
        lock["path"] = value["locks"]["workflow"]["path"]
    elif fault == "relative_lock":
        lock["path"] = "relative/lock"
    elif fault == "dotdot_lock":
        lock["path"] = str(tmp_path / "a/../lock")
    elif fault == "hardlinked_lock":
        lock["signature"][2] = 2
    elif fault == "wide_lock":
        lock["signature"][3] = 2
    elif fault == "bool_stamp":
        lock["signature"][0] = True
    elif fault == "negative_stamp":
        lock["signature"][0] = -1
    elif fault == "short_stamp":
        lock["signature"].pop()
    elif fault == "wrong_lock_bytes":
        lock["content_sha256"] = "0" * 64
    elif fault == "upper_digest":
        value["started_sha256"] = "D" * 64
    else:
        lock["path"] = str(tmp_path / ("a" * wire.MAX_PACKET))
    with pytest.raises(ValueError):
        wire.checked(value, binding, "ready")


@pytest.mark.parametrize(
    "fault", ("extra", "missing", "kind", "upper_digest", "short_digest", "nonstring_digest")
)
def test_release_refusals(tmp_path: Path, fault: str) -> None:
    binding, _ = specimen(tmp_path)
    value: dict[str, Any] = dict(
        kind="iris-recovery-race-release-v1",
        binding=binding,
        ready_sha256="a" * 64,
        contender_process_sha256="b" * 64,
    )
    if fault == "extra":
        value["stage"] = {}
    elif fault == "missing":
        del value["ready_sha256"]
    elif fault == "kind":
        value["kind"] = "other"
    elif fault == "upper_digest":
        value["ready_sha256"] = "A" * 64
    elif fault == "short_digest":
        value["contender_process_sha256"] = "b" * 63
    else:
        value["ready_sha256"] = False
    with pytest.raises(ValueError):
        wire.checked(value, binding, "release")


@pytest.mark.parametrize(
    "fault", ("whitespace", "duplicate", "NaN", "oversized", "empty", "extra_member", "hardlink")
)
def test_read_refuses_untrusted_bytes_and_layout(tmp_path: Path, fault: str) -> None:
    binding, value = specimen(tmp_path)
    directory = tmp_path / "ready"
    directory.mkdir()
    raw = wire.canonical(value)
    if fault == "whitespace":
        raw += b" "
    elif fault == "duplicate":
        raw = raw[:-2] + b',"rows":1}\n'
    elif fault == "NaN":
        raw = json.dumps(value).replace('"rows": 1', '"rows": NaN').encode()
    elif fault == "oversized":
        raw = b"x" * (wire.MAX_PACKET + 1)
    elif fault == "empty":
        raw = b""
    path = directory / "packet.json"
    path.write_bytes(raw)
    if fault == "extra_member":
        (directory / "other").write_bytes(b"x")
    if fault == "hardlink":
        os.link(path, tmp_path / "linked")
    with pytest.raises(ValueError):
        wire.read(tmp_path, "ready", binding)


def test_wait_requires_actual_liveness_even_if_complete_marker_exists(tmp_path: Path) -> None:
    binding, value = specimen(tmp_path)
    wire.publish(tmp_path, "ready", value, binding)
    with (
        patch.object(wire, "read", side_effect=AssertionError("No marker-based restart")),
        pytest.raises(RuntimeError),
    ):
        wire.wait(tmp_path, "ready", binding, alive=lambda: False)


def test_observation_timeout_never_restarts_or_publishes(tmp_path: Path) -> None:
    binding, _ = specimen(tmp_path)
    with (
        patch.object(time, "monotonic", side_effect=[0, 601]),
        patch.object(time, "sleep") as sleep,
        pytest.raises(TimeoutError),
    ):
        wire.wait(tmp_path, "ready", binding, alive=lambda: True)
    sleep.assert_not_called()
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value",
    (
        ("timeout", True),
        ("timeout", 0),
        ("timeout", 601),
        ("timeout", float("nan")),
        ("interval", False),
        ("interval", 0),
        ("interval", 1),
        ("interval", float("inf")),
    ),
)
def test_wait_is_bounded_before_any_poll(tmp_path: Path, field: str, value: Any) -> None:
    binding, _ = specimen(tmp_path)
    with pytest.raises(ValueError):
        wire.wait(tmp_path, "ready", binding, alive=lambda: True, **{field: value})


def test_wait_retains_same_wait_after_missing_packet(tmp_path: Path) -> None:
    binding, value = specimen(tmp_path)
    with (
        patch.object(wire, "read", side_effect=[FileNotFoundError(), (value, "a" * 64)]),
        patch.object(time, "sleep") as sleep,
    ):
        assert wire.wait(tmp_path, "ready", binding, alive=lambda: True) == (value, "a" * 64)
    sleep.assert_called_once_with(0.1)


def test_windows_signature_uses_exact_same_birthtime_not_unlike_ctime() -> None:
    from types import SimpleNamespace

    named: Any = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_nlink=1,
        st_size=3,
        st_mtime_ns=1000000100,
        st_ctime_ns=1000000000,
        st_birthtime_ns=1000000000,
    )
    opened: Any = copy.copy(named)
    opened.st_ctime_ns = opened.st_mtime_ns
    with patch.object(os, "name", "nt"):
        assert (
            wire.signature(named) == wire.signature(opened) == [1, 2, 1, 3, 1000000100, 1000000000]
        )


@pytest.mark.parametrize(
    "field", ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns", "st_birthtime_ns")
)
def test_every_exact_windows_metadata_component_remains_bound(field: str) -> None:
    from types import SimpleNamespace

    named: Any = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_nlink=1,
        st_size=3,
        st_mtime_ns=1000000100,
        st_ctime_ns=1000000000,
        st_birthtime_ns=1000000000,
    )
    changed: Any = copy.copy(named)
    setattr(changed, field, getattr(changed, field) + 1)
    with patch.object(os, "name", "nt"):
        assert wire.signature(named) != wire.signature(changed)


def test_missing_exact_windows_creation_stamp_is_not_a_tolerance_waiver() -> None:
    from types import SimpleNamespace

    info: Any = SimpleNamespace(st_ctime_ns=1)
    with patch.object(os, "name", "nt"), pytest.raises(OSError):
        wire.signature(info)


def test_posix_exact_change_time_is_retained() -> None:
    from types import SimpleNamespace

    info: Any = SimpleNamespace(
        st_dev=1, st_ino=2, st_nlink=1, st_size=3, st_mtime_ns=4, st_ctime_ns=5, st_birthtime_ns=6
    )
    with patch.object(os, "name", "posix"):
        assert wire.signature(info) == [1, 2, 1, 3, 4, 5]


def test_same_api_named_ctime_change_is_still_bound_on_windows() -> None:
    from types import SimpleNamespace

    named: Any = SimpleNamespace(
        st_dev=1, st_ino=2, st_nlink=1, st_size=3, st_mtime_ns=4, st_ctime_ns=5, st_birthtime_ns=6
    )
    changed: Any = copy.copy(named)
    changed.st_ctime_ns += 1
    with patch.object(os, "name", "nt"):
        assert wire.signature(named) == wire.signature(changed)
        assert wire._named_signature(named) != wire._named_signature(changed)
