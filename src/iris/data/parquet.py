"""Bounded Parquet codec for private source-reconciled synthetic rows.

Encoding is logical-row reproducible, not a cross-version byte-identical build
promise. Physical file hashes belong to the eventual dataset release manifest.
"""

from hashlib import sha256
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from iris.data.schemas import IrisExample
from iris.utils.hashing import canonical_json_bytes

_SOURCE_SHA256 = "c424fa92a37547bf5af9ebc9faf822f9453b616fa98f4e4da63477b8691fcf7e"


def _schema() -> Any:
    fact = pa.struct(
        [
            pa.field("kind", pa.string(), nullable=False),
            pa.field("string_value", pa.string()),
            pa.field("integer_value", pa.int64()),
            pa.field("boolean_value", pa.bool_()),
        ]
    )
    case = pa.struct(
        [
            pa.field("inputs", pa.list_(pa.int64()), nullable=False),
            pa.field("expected", pa.int64(), nullable=False),
        ]
    )
    kinds = {
        "id": pa.string(),
        "task_type": pa.string(),
        "difficulty": pa.int64(),
        "seed": pa.int64(),
        "prompt": pa.string(),
        "expected_response": pa.string(),
        "hidden_facts": pa.map_(pa.string(), fact),
        "required_modules": pa.list_(pa.string()),
        "distractor_count": pa.int64(),
        "executable": pa.bool_(),
        "test_code": pa.string(),
        "timeout_seconds": pa.int64(),
        "split": pa.string(),
        "template_id": pa.string(),
        "problem_sha256": pa.string(),
        "code_cases": pa.list_(case),
    }
    return pa.schema(
        [pa.field(name, kind, nullable=name == "test_code") for name, kind in kinds.items()]
    )


def _budget(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside the explicit codec budget")


def _physical(row: IrisExample) -> dict[str, Any]:
    raw: dict[str, Any] = row.model_dump(mode="json")
    facts = []
    for key, value in sorted(row.hidden_facts.items()):
        kind = {str: "string", int: "integer", bool: "boolean"}[type(value)]
        facts.append(
            (
                key,
                {
                    "kind": kind,
                    "string_value": value if kind == "string" else None,
                    "integer_value": value if kind == "integer" else None,
                    "boolean_value": value if kind == "boolean" else None,
                },
            )
        )
    raw["hidden_facts"] = facts
    return raw


def _logical(raw: dict[str, Any]) -> IrisExample:
    facts = {}
    for key, tagged in raw["hidden_facts"]:
        if key in facts:
            raise ValueError("duplicate Parquet fact key")
        kind = tagged["kind"]
        if kind not in ("string", "integer", "boolean"):
            raise ValueError("unknown Parquet fact kind")
        selected = f"{kind}_value"
        if tagged[selected] is None or any(
            tagged[f"{other}_value"] is not None
            for other in ("string", "integer", "boolean")
            if other != kind
        ):
            raise ValueError("invalid tagged Parquet fact")
        facts[key] = tagged[selected]
    raw["hidden_facts"] = facts
    # JSON mode deliberately restores tuple fields from Arrow's list representation.
    return IrisExample.model_validate_json(canonical_json_bytes(raw))


def _rows_digest(rows: tuple[IrisExample, ...], max_logical_bytes: int) -> str:
    digest = sha256(b"iris/synthetic-01/logical-rows/v1\0")
    used = 0
    ids, problems, seeds = set(), set(), set()
    for row in rows:
        content = canonical_json_bytes(row.model_dump(mode="json"))
        used += len(content)
        if used > max_logical_bytes:
            raise ValueError("synthetic logical data exceeds budget")
        if row.id in ids or row.problem_sha256 in problems or row.seed in seeds:
            raise ValueError("duplicate synthetic row, seed or problem")
        ids.add(row.id)
        problems.add(row.problem_sha256)
        seeds.add(row.seed)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def encode_synthetic_parquet(
    rows: tuple[IrisExample, ...], *, max_logical_bytes: int, max_file_bytes: int
) -> bytes:
    """Encode validated private rows; no filesystem writes or model execution."""
    _budget("max_logical_bytes", max_logical_bytes, 512 * 1024**2)
    _budget("max_file_bytes", max_file_bytes, 512 * 1024**2)
    if type(rows) is not tuple or not 1 <= len(rows) <= 100000:
        raise ValueError("codec requires 1 to 100000 explicit rows")
    owned = tuple(IrisExample.model_validate(row.model_dump(mode="python")) for row in rows)
    identity = _rows_digest(owned, max_logical_bytes)
    metadata = {
        b"iris.format": b"iris-synthetic-01-parquet-v1",
        b"iris.source_sha256": _SOURCE_SHA256.encode(),
        b"iris.logical_sha256": identity.encode(),
        b"iris.scope": b"candidate_not_launch_approval",
    }
    table = pa.Table.from_pylist(
        [_physical(row) for row in owned], schema=_schema().with_metadata(metadata)
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        version="2.6",
        compression="zstd",
        compression_level=3,
        row_group_size=256,
        use_dictionary=True,
        write_page_checksum=True,
    )
    content: bytes = sink.getvalue().to_pybytes()
    if len(content) > max_file_bytes:
        raise ValueError("synthetic Parquet file exceeds budget")
    return content


def decode_synthetic_parquet(
    content: bytes, *, max_rows: int, max_logical_bytes: int, max_file_bytes: int
) -> tuple[IrisExample, ...]:
    """Validate schema, typed rows and logical checksum, not source authenticity."""
    _budget("max_rows", max_rows, 100000)
    _budget("max_logical_bytes", max_logical_bytes, 512 * 1024**2)
    _budget("max_file_bytes", max_file_bytes, 512 * 1024**2)
    if type(content) is not bytes or not 1 <= len(content) <= max_file_bytes:
        raise ValueError("invalid synthetic Parquet bytes")
    source = pa.BufferReader(content)
    parquet = pq.ParquetFile(
        source,
        thrift_string_size_limit=1048576,
        thrift_container_size_limit=1048576,
        page_checksum_verification=True,
    )
    if not parquet.schema_arrow.equals(_schema(), check_metadata=False):
        raise ValueError("unexpected synthetic Parquet schema")
    metadata = parquet.schema_arrow.metadata or {}
    if (
        set(metadata)
        != {b"iris.format", b"iris.source_sha256", b"iris.logical_sha256", b"iris.scope"}
        or metadata[b"iris.format"] != b"iris-synthetic-01-parquet-v1"
        or metadata[b"iris.source_sha256"] != _SOURCE_SHA256.encode()
        or metadata[b"iris.scope"] != b"candidate_not_launch_approval"
    ):
        raise ValueError("unexpected synthetic Parquet metadata")
    count = parquet.metadata.num_rows
    if not 1 <= count <= max_rows or parquet.metadata.num_row_groups > max_rows:
        raise ValueError("synthetic Parquet row budget exceeded")
    # Bound declared decompressed column data before allocating Arrow batches.
    if (
        sum(
            parquet.metadata.row_group(i).total_byte_size
            for i in range(parquet.metadata.num_row_groups)
        )
        > max_logical_bytes
    ):
        raise ValueError("synthetic Parquet decompressed-data budget exceeded")
    rows: list[IrisExample] = []
    used = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for raw in batch.to_pylist():
            row = _logical(raw)
            used += len(canonical_json_bytes(row.model_dump(mode="json")))
            if used > max_logical_bytes:
                raise ValueError("synthetic logical data exceeds budget")
            rows.append(row)
    result = tuple(rows)
    if (
        len(result) != count
        or _rows_digest(result, max_logical_bytes).encode() != metadata[b"iris.logical_sha256"]
    ):
        raise ValueError("synthetic Parquet logical identity mismatch")
    return result
