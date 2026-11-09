"""Owned context lifetime, full-byte revalidation and fail-closed invalidation."""

import asyncio
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from iris.application import study_checkpoint_scope as scope
from iris.application.arm_request_checkpoint import IrisArmRequestCheckpointProjectionV1
from iris.application.errors import IrisStudyEvaluationError, IrisStudyGenerationError
from tests.unit.test_request_artifact_v4 import build_artifact_v4
from tests.unit.test_verification_v4 import _projection


def _fixture(tmp_path: Path) -> tuple[Path, IrisArmRequestCheckpointProjectionV1, Mock]:
    projection = _projection(build_artifact_v4(tmp_path, "B"))
    state = b"unit checkpoint bytes; semantic loader is substituted"
    identity = replace(projection.checkpoint_identity, state_file_sha256=sha256(state).hexdigest())
    projection = replace(projection, checkpoint_identity=identity)
    root = tmp_path / "artifacts"
    directory = identity.expected_path(root=root)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_bytes(b'{"unit_semantic_loader":"substituted"}\n')
    (directory / "state.safetensors").write_bytes(state)
    return root, projection, Mock(return_value=projection)


def _get(
    current: scope._CheckpointScope,
    fixture: tuple[Path, IrisArmRequestCheckpointProjectionV1, Mock],
) -> IrisArmRequestCheckpointProjectionV1:
    root, projection, load = fixture
    return current.projection(projection.checkpoint_identity, root, load)


def test_context_reuses_only_deeply_owned_projection_and_discards_it_on_exit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with scope._checkpoint_verification_context():
        current = scope._current_checkpoint_scope()
        assert current is not None
        first = _get(current, fixture)
        expected = first.backbone.limits.max_prompt_tokens
        first.backbone.limits.max_prompt_tokens += 1
        assert _get(current, fixture).backbone.limits.max_prompt_tokens == expected
        with scope._checkpoint_verification_context():
            assert scope._current_checkpoint_scope() is current
            assert _get(current, fixture) == fixture[1]
        retained_context = copy_context()
        fixture[2].assert_called_once_with()
    assert current.closed and not current.entries
    assert scope._current_checkpoint_scope() is None
    assert retained_context.run(scope._current_checkpoint_scope) is None
    with scope._checkpoint_verification_context():
        fresh = scope._current_checkpoint_scope()
        assert fresh is not None and fresh is not current
        _get(fresh, fixture)
    assert fixture[2].call_count == 2


def test_identical_checkpoint_in_a_second_root_requires_its_own_verification(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    root, projection, load = fixture
    directory = projection.checkpoint_identity.expected_path(root=root)
    second_root = tmp_path / "second-artifacts"
    second = projection.checkpoint_identity.expected_path(root=second_root)
    second.mkdir(parents=True)
    for name in scope._FILES:
        (second / name).write_bytes((directory / name).read_bytes())
    with scope._checkpoint_verification_context():
        current = scope._current_checkpoint_scope()
        assert current is not None
        assert _get(current, fixture) == projection
        assert _get(current, (second_root, projection, load)) == projection
        assert len(current.entries) == 2
        assert load.call_count == 2


def test_directory_scan_stops_at_first_extra_entry(tmp_path: Path) -> None:
    root, projection, _ = _fixture(tmp_path)
    directory = projection.checkpoint_identity.expected_path(root=root)

    @contextmanager
    def scan(path: Path) -> Iterator[Iterator[SimpleNamespace]]:
        assert path == directory

        def entries() -> Iterator[SimpleNamespace]:
            for name in (*scope._FILES, "extra"):
                yield SimpleNamespace(name=name)
            raise AssertionError("must not enumerate beyond the first extra entry")

        yield entries()

    with patch.object(os, "scandir", scan), pytest.raises(ValueError, match="extra"):
        scope._directory_chain(directory)


@pytest.mark.parametrize(
    "mutation", ["state", "manifest", "extra", "missing", "replacement", "directory"]
)
def test_reuse_rejects_changed_files_and_poisoned_scope_cannot_be_revived(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    root, projection, load = fixture
    directory = projection.checkpoint_identity.expected_path(root=root)
    current = scope._CheckpointScope()
    _get(current, fixture)
    target = directory / ("manifest.json" if mutation == "manifest" else "state.safetensors")
    original = target.read_bytes()
    metadata = target.stat()
    if mutation in {"state", "manifest"}:
        target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        os.utime(target, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    elif mutation == "extra":
        (directory / "foreign.txt").write_bytes(b"preserve")
    elif mutation == "missing":
        target.rename(directory / "preserved-state")
    elif mutation == "replacement":
        other = directory / "replacement"
        other.write_bytes(original)
        os.utime(other, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        os.replace(other, target)
    else:
        directory.rename(directory.with_name("preserved-directory"))
        directory.mkdir()
        (directory / "manifest.json").write_bytes(b"{}\n")
        target.write_bytes(original)
    with pytest.raises(ValueError):
        _get(current, fixture)
    assert current.invalid and not current.entries
    with pytest.raises(ValueError, match="invalidated"):
        _get(current, fixture)
    load.assert_called_once_with()


@pytest.mark.parametrize("phase", ["load", "final"])
def test_mutation_during_first_load_or_at_scope_exit_is_not_reported_as_success(
    tmp_path: Path, phase: str
) -> None:
    fixture = _fixture(tmp_path)
    root, projection, load = fixture
    target = projection.checkpoint_identity.expected_path(root=root) / "manifest.json"

    def mutate() -> IrisArmRequestCheckpointProjectionV1:
        target.write_bytes(b"changed private checkpoint metadata")
        return projection

    if phase == "load":
        load.side_effect = mutate
    with pytest.raises(ValueError), scope._checkpoint_verification_context():
        current = scope._current_checkpoint_scope()
        assert current is not None
        _get(current, fixture)
        if phase == "final":
            mutate()
    assert scope._current_checkpoint_scope() is None
    assert current is not None
    assert current.closed and not current.entries


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_failed_semantic_load_never_seeds_a_cache(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    fixture = _fixture(tmp_path)
    fixture[2].side_effect = failure("PRIVATE_LOAD_FAILURE")
    current = scope._CheckpointScope()
    with pytest.raises(failure):
        _get(current, fixture)
    assert current.invalid and not current.entries


def test_projection_identity_must_match_requested_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture[2].return_value = replace(
        fixture[1],
        checkpoint_identity=replace(fixture[1].checkpoint_identity, completed_steps=8),
        completed_steps=8,
    )
    with pytest.raises(ValueError, match="identity"):
        _get(scope._CheckpointScope(), fixture)


def test_context_is_not_shared_with_copied_thread_or_async_task(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with scope._checkpoint_verification_context():
        current = scope._current_checkpoint_scope()
        assert current is not None
        _get(current, fixture)
        copied = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(copied.run, scope._current_checkpoint_scope).result() is None

        async def foreign() -> None:
            assert scope._current_checkpoint_scope() is None
            with scope._checkpoint_verification_context():
                assert scope._current_checkpoint_scope() is not current

        asyncio.run(foreign())
        assert scope._current_checkpoint_scope() is current
        with patch.object(scope, "_owner", return_value=(current.owner[0] + 1, *current.owner[1:])):
            assert scope._current_checkpoint_scope() is None
            with pytest.raises(ValueError, match="foreign"):
                _get(current, fixture)


def test_caught_nested_failure_invalidates_shared_scope() -> None:
    with pytest.raises(ValueError, match="invalidated"), scope._checkpoint_verification_context():
        current = scope._current_checkpoint_scope()
        assert current is not None
        with pytest.raises(RuntimeError), scope._checkpoint_verification_context():
            raise RuntimeError("nested action failed")
        assert current.invalid
    assert scope._current_checkpoint_scope() is None


@pytest.mark.parametrize("kind", ["evaluation", "generation"])
def test_final_boundary_failure_uses_redacted_domain_error(tmp_path: Path, kind: str) -> None:
    fixture = _fixture(tmp_path)

    def action() -> int:
        current = scope._current_checkpoint_scope()
        assert current is not None
        _get(current, fixture)
        (
            fixture[1].checkpoint_identity.expected_path(root=fixture[0]) / "manifest.json"
        ).write_bytes(b"PRIVATE_CHANGED_BYTES")
        return 42

    wrapped = (
        scope._scoped_evaluation(action)
        if kind == "evaluation"
        else scope._scoped_generation(action)
    )
    error_type = IrisStudyEvaluationError if kind == "evaluation" else IrisStudyGenerationError
    with pytest.raises(error_type) as caught:
        wrapped()
    assert "PRIVATE" not in str(caught.value) and caught.value.__suppress_context__
    assert scope._current_checkpoint_scope() is None


@pytest.mark.parametrize("kind", ["empty", "directory", "reparse"])
def test_fingerprint_refuses_non_regular_or_reparse_inputs(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "input"
    if kind == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"" if kind == "empty" else b"private bytes")
    with (
        patch.object(scope, "_is_link_or_reparse", return_value=kind == "reparse"),
        pytest.raises(ValueError),
    ):
        scope._fingerprint(target)


@pytest.mark.parametrize("mutation", ["none", "truncate", "grow", "change"])
def test_fingerprint_reads_bounded_chunks_and_rejects_mid_read_changes(
    tmp_path: Path, mutation: str
) -> None:
    target = tmp_path / "input"
    content = b"abcdefghi"
    target.write_bytes(content)
    real_fdopen = os.fdopen
    requested: list[int] = []

    def tracked(fd: int, mode: str) -> Any:
        stream = real_fdopen(fd, mode)

        class Reader:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *args: object) -> None:
                stream.close()

            def fileno(self) -> int:
                return stream.fileno()

            def read(self, size: int) -> bytes:
                return read(size)

        def read(size: int) -> bytes:
            requested.append(size)
            result = stream.read(size)
            assert isinstance(result, bytes)
            if len(requested) == 1 and mutation != "none":
                target.write_bytes(
                    content[:-1]
                    if mutation == "truncate"
                    else content + b"x"
                    if mutation == "grow"
                    else b"changed!!"
                )
            return result

        return Reader()

    with patch.object(scope, "_CHUNK_BYTES", 3), patch.object(os, "fdopen", side_effect=tracked):
        if mutation == "none":
            result = scope._fingerprint(target)
            assert result.sha256 == sha256(content).hexdigest()
            assert requested == [3, 3, 3, 1]
        else:
            with pytest.raises(ValueError):
                scope._fingerprint(target)
    assert all(0 < size <= 3 for size in requested)
