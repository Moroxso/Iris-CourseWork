"""Abrupt native process death bypasses Python finally blocks; committed answers survive."""

import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from iris.application.study_generation_run import verify_study_generation_run_v1
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_generation_run import _inference, _run

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("boundary", ["partial", "staged", "linked"])
def test_native_death_releases_owner_and_recovers_answer_prefix(
    tmp_path: Path, boundary: str
) -> None:
    fixture, directory = _fixture(tmp_path), tmp_path / "run"
    source = """
import json, os, sys
from pathlib import Path
from unittest.mock import patch
sys.path[:0] = [str(Path(sys.argv[3]) / 'src'), sys.argv[3], sys.argv[4]]
from tests.unit.test_study_generation import _fixture
from tests.unit.test_study_generation_run import _inference, _run
from iris.application import study_generation_run as service
from iris.application import study_generation_run_io as storage
directory = Path(sys.argv[1]) / 'run'
boundary = sys.argv[2]
fixture = _fixture(Path(sys.argv[1]))
original_stage, original_link = storage._stage, os.link
def die():
 print(os.getpid(), flush=True)
 os._exit(77)
def stage(pending, content, **kwargs):
 target = pending.name == '.answer-pending.json' and json.loads(content)['payload']['index'] == 2
 if boundary == 'partial' and target:
  pending.write_bytes(content[:23])
  die()
 original_stage(pending, content, **kwargs)
 if boundary == 'staged' and target:
  die()
def link(source, destination, **kwargs):
 original_link(source, destination, **kwargs)
 if boundary == 'linked' and Path(destination).name == '000002.json':
  die()
with (
 _inference(fixture), patch.object(service, '_stage', side_effect=stage),
 patch.object(os, 'link', side_effect=link),
):
 _run(fixture, directory)
raise AssertionError('required abrupt-death boundary was not reached')
"""
    child = subprocess.Popen(
        [
            str(getattr(sys, "_base_executable", sys.executable)),
            "-I",
            "-c",
            source,
            str(tmp_path),
            boundary,
            str(Path(__file__).resolve().parents[2]),
            sysconfig.get_path("purelib"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        stdout, stderr = child.communicate(timeout=45)
        assert child.returncode == 77, stderr.decode(errors="replace")[-1200:]
        assert int(stdout.strip()) == child.pid
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
    lock_inode = (directory / ".lock").stat().st_ino
    assert not (directory / "generation.json").exists()
    assert len(tuple((directory / "answers").iterdir())) == (3 if boundary == "linked" else 2)
    with _inference(fixture) as seen:
        result = _run(fixture, directory, resume=True)
        missing = 3 if boundary == "partial" else 2
        assert seen == list(range(5 - missing, 5))
        assert (result.generated_now, result.reused_answers) == (missing, 5 - missing)
        assert verify_study_generation_run_v1(directory, benchmark=fixture[1]) == fixture[0]
    assert (directory / ".lock").stat().st_ino == lock_inode
    assert (directory / ".lock").read_bytes() == b"1"
