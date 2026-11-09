"""Exact Windows console JSON framing, not canonical on-disk JSON-file framing.

No trimming, newline normalization, diagnostic filtering or runtime acceptance.
The caller retains and hashes the original subprocess bytes separately.
"""

import json
from typing import Any


def windows_json_object(stdout: bytes) -> dict[str, Any]:
    """Require one canonical UTF8 JSON object followed by exactly one CRLF."""
    if type(stdout) is not bytes:
        raise ValueError("Original console stdout must be bytes")
    try:
        value: object = json.loads(stdout.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ValueError("Original stdout is not UTF8 JSON") from error
    if type(value) is not dict:
        raise ValueError("One JSON object required, not a scalar or array")
    expected = (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\r\n"
    )
    if stdout != expected:
        raise ValueError("Exact canonical JSON and one Windows CRLF required")
    return value
