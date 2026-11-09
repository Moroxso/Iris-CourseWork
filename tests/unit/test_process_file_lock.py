"""Real native nonblocking locks, process death recovery and persistent identity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from iris.utils.process_file_lock import process_file_lock


def test_lock_is_exclusive_and_persistent_without_unlink(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    with process_file_lock(path) as claim:
        claim.check()
        assert not os.get_inheritable(claim.descriptor)
        with pytest.raises(OSError), process_file_lock(path):
            pytest.fail("another handle acquired the active claim")
    assert path.read_bytes() == b"1"
    identity = path.stat().st_ino
    with process_file_lock(path):
        pass
    assert path.stat().st_ino == identity


@pytest.mark.parametrize("content", [b"wrong", b"x", b"11"])
def test_invalid_existing_lock_is_never_overwritten(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / ".lock"
    path.write_bytes(content)
    with pytest.raises(OSError), process_file_lock(path):
        pytest.fail("invalid ownership file accepted")
    assert path.read_bytes() == content


def test_empty_interrupted_initialization_is_recoverable(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    path.touch()
    with process_file_lock(path) as claim:
        claim.check()
    assert path.read_bytes() == b"1"


def test_base_exception_releases_lock_without_removing_it(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    with pytest.raises(KeyboardInterrupt), process_file_lock(path):
        raise KeyboardInterrupt
    with process_file_lock(path) as claim:
        claim.check()


def test_hardlink_alias_is_not_accepted_as_lock_ownership(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    path.write_bytes(b"1")
    os.link(path, tmp_path / "alias")
    with pytest.raises(OSError), process_file_lock(path):
        pytest.fail("hard-linked lock accepted")
    assert path.read_bytes() == b"1"


def test_os_releases_claim_after_abrupt_death_of_exact_child(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    source = (
        "import os, sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from iris.utils.process_file_lock import process_file_lock\n"
        "with process_file_lock(Path(sys.argv[1])):\n"
        " print(os.getpid(), flush=True)\n sys.stdin.read(1)\n"
    )
    child = subprocess.Popen(
        [
            str(getattr(sys, "_base_executable", sys.executable)),
            "-I",
            "-c",
            source,
            str(path),
            str(Path(__file__).resolve().parents[2] / "src"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout is not None
        # Windows venv launchers can spawn a second interpreter. Test actual
        # ownership death, not just the death of a redirector observing its child.
        assert int(child.stdout.readline().strip()) == child.pid
        with pytest.raises(OSError), process_file_lock(path):
            pytest.fail("second process acquired another process's claim")
        child.kill()
        child.wait(timeout=10)
        with process_file_lock(path) as claim:
            claim.check()
        assert path.read_bytes() == b"1"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
