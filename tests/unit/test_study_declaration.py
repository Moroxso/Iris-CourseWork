"""Canonical I/O contracts for the Iris four-arm protocol declaration."""

from pathlib import Path
from typing import Any, cast

import pytest

from iris.application import (
    IrisFourArmProtocolError,
    build_four_arm_protocol_declaration_v1,
    read_four_arm_protocol_declaration_v1,
    validate_four_arm_protocol_declaration_v1,
    verify_four_arm_protocol_declaration_v1,
    write_four_arm_protocol_declaration_v1,
)
from iris.utils.hashing import canonical_json_file_bytes

_CHECKED_IN_DECLARATION = (
    Path(__file__).parents[2] / "docs" / "protocols" / "iris-gw-0.1-four-arm.v1.json"
)


def test_checked_in_declaration_matches_the_canonical_builder() -> None:
    declaration = build_four_arm_protocol_declaration_v1()

    assert read_four_arm_protocol_declaration_v1(_CHECKED_IN_DECLARATION) == declaration
    assert declaration.payload_sha256 == (
        "b7b25609f8ba611a9f8037108e6404fe41a89be01b16242ab0708a3ef475d747"
    )


def test_builder_is_deterministic_and_returns_deeply_equal_owned_values() -> None:
    first = build_four_arm_protocol_declaration_v1()
    second = build_four_arm_protocol_declaration_v1()

    assert first == second
    assert first is not second
    assert first.payload is not second.payload
    assert first.payload_sha256 == second.payload_sha256


def test_validator_returns_an_owned_declaration() -> None:
    declaration = build_four_arm_protocol_declaration_v1()

    owned = validate_four_arm_protocol_declaration_v1(declaration)

    assert owned == declaration
    assert owned is not declaration
    assert owned.payload is not declaration.payload


def test_validator_rejects_tampered_payload_hash() -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    tampered = declaration.model_copy(update={"payload_sha256": "0" * 64})

    with pytest.raises(IrisFourArmProtocolError, match="could not be verified"):
        validate_four_arm_protocol_declaration_v1(tampered)


def test_writer_and_reader_round_trip_canonical_bytes(tmp_path: Path) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    path = tmp_path / "protocol.json"

    result = write_four_arm_protocol_declaration_v1(declaration, path=path)
    restored = read_four_arm_protocol_declaration_v1(path)

    assert restored == declaration
    assert result.path == path
    assert result.payload_sha256 == declaration.payload_sha256
    assert result.size_bytes == path.stat().st_size
    assert not result.reused_existing
    assert path.read_bytes() == canonical_json_file_bytes(declaration.model_dump(mode="json"))


def test_read_only_verifier_reports_exact_scope_without_mutation(tmp_path: Path) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    path = tmp_path / "protocol.json"
    write_four_arm_protocol_declaration_v1(declaration, path=path)
    before = path.read_bytes()

    report = verify_four_arm_protocol_declaration_v1(path)

    assert path.read_bytes() == before
    assert report.artifact_type == "iris.four_arm_protocol_declaration"
    assert report.payload_sha256 == declaration.payload_sha256
    assert report.protocol_id == "iris-gw-0.1-four-arm"
    assert report.arm_count == 4
    assert report.task_family_count == 5
    assert report.unresolved_item_count == 10
    assert report.execution_status == ("blocked_until_explicit_protocol_and_arm_implementations")


def test_writer_reuses_only_identical_existing_bytes(tmp_path: Path) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    path = tmp_path / "protocol.json"

    first = write_four_arm_protocol_declaration_v1(declaration, path=path)
    second = write_four_arm_protocol_declaration_v1(declaration, path=path)

    assert not first.reused_existing
    assert second.reused_existing
    assert first.size_bytes == second.size_bytes


def test_writer_rejects_conflicting_existing_bytes(tmp_path: Path) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    path = tmp_path / "protocol.json"
    path.write_bytes(b"{}\n")

    with pytest.raises(IrisFourArmProtocolError, match="conflicts"):
        write_four_arm_protocol_declaration_v1(declaration, path=path)


@pytest.mark.parametrize("name", ["protocol", "protocol.txt", ""])
def test_io_requires_an_explicit_json_path(tmp_path: Path, name: str) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    path = tmp_path / name if name else Path()

    with pytest.raises(IrisFourArmProtocolError):
        write_four_arm_protocol_declaration_v1(declaration, path=path)
    with pytest.raises(IrisFourArmProtocolError):
        read_four_arm_protocol_declaration_v1(path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: b"\xef\xbb\xbf" + value,
        lambda value: value + b"\n",
        lambda value: value.rstrip(b"\n"),
        lambda value: value.replace(b'"version":1', b'"version": 1', 1),
    ],
)
def test_reader_rejects_noncanonical_encodings(tmp_path: Path, mutator: object) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    path = tmp_path / "protocol.json"
    content = canonical_json_file_bytes(declaration.model_dump(mode="json"))
    path.write_bytes(mutator(content))  # type: ignore[operator]

    with pytest.raises(IrisFourArmProtocolError, match="could not be read"):
        read_four_arm_protocol_declaration_v1(path)


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_bytes(b'{"payload":{},"payload":{}}\n')

    with pytest.raises(IrisFourArmProtocolError, match="could not be read"):
        read_four_arm_protocol_declaration_v1(path)


def test_reader_rejects_a_changed_payload_with_stale_hash(tmp_path: Path) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    raw = declaration.model_dump(mode="json")
    raw["payload"]["execution_status"] = "blocked_until_explicit_protocol_and_arm_implementations"
    raw["payload"]["architecture"]["cognitive_state_width"] = 8
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json_file_bytes(raw))

    with pytest.raises(IrisFourArmProtocolError, match="could not be read"):
        read_four_arm_protocol_declaration_v1(path)


def test_writer_rejects_a_non_declaration_value(tmp_path: Path) -> None:
    with pytest.raises(IrisFourArmProtocolError, match="could not be persisted"):
        write_four_arm_protocol_declaration_v1(
            cast(Any, object()),
            path=tmp_path / "protocol.json",
        )


def test_reader_rejects_symlink_when_supported(tmp_path: Path) -> None:
    declaration = build_four_arm_protocol_declaration_v1()
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    write_four_arm_protocol_declaration_v1(declaration, path=target)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")

    with pytest.raises(IrisFourArmProtocolError, match="could not be read"):
        read_four_arm_protocol_declaration_v1(link)
