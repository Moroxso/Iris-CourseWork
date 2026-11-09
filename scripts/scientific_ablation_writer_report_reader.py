"""Bounded canonical byte API for the independent COMPLETE writer validator.

No model/project/Pydantic imports or execution. Separate ORIGINAL authenticated
preflight and CPU stream context are mandatory; candidate metadata cannot serve
as its own provenance. The accepted raw validator remains unchanged.
"""

import json
import math
from typing import Any

import scientific_ablation_writer_report_raw as raw
import verify_scientific_ablation_complete_source_preflight as sources


def _expectation(expected: object, runtime: tuple[str, str]) -> tuple[dict[str, Any], int]:
    raw.runtime_guard(runtime)
    owned = raw.own_json(expected)
    raw.require(type(owned) is dict, "Exact separately trusted original preflight")
    assert isinstance(owned, dict)
    reservation = sources.preflight(owned)
    bound, cap = reservation["bound"], owned["maximum_report_bytes"]
    raw.require(
        type(bound) is int and type(cap) is int and 1 <= bound <= cap <= 512 * 1024**2,
        "Original whole byte reservation, no clipping or fallback quota",
    )
    assert isinstance(bound, int)
    return owned, bound


def _nesting(content: bytes) -> None:
    """Bound structural nesting BEFORE parser allocation; strings/escapes ignored."""
    depth = 0
    quoted = escaped = False
    for byte in content:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            raw.require(depth <= 40, "Complete JSON depth bound exceeded before parsing")
        elif byte in (93, 125):
            depth -= 1
            raw.require(depth >= 0, "Invalid complete JSON structural framing")
    raw.require(
        depth == 0 and not quoted and not escaped, "Complete JSON closing/string framing required"
    )


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON field forbidden at every depth")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError("Nonfinite JSON constants forbidden")


def _float(value: str) -> float:
    result = float(value)
    raw.require(math.isfinite(result), "Overflowing/nonfinite JSON float forbidden")
    return result


def read_report(content: object, *, expected: object, runtime: tuple[str, str]) -> dict[str, Any]:
    """Read ONLY exact canonical whole report bytes against separate original source.

    This is metadata/source consistency, NOT proof of native execution/cleanup or
    preregistration/task independence. The caller owns physical trust anchors.
    """
    # Identity precedes len/decode/bytes/equality on any candidate byte subclass.
    raw.require(type(content) is bytes, "Exact report bytes required before callbacks")
    assert isinstance(content, bytes)
    original, bound = _expectation(expected, runtime)
    raw.require(1 <= len(content) <= bound, "Whole report byte length exceeds original preflight")
    _nesting(content)
    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_float,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Complete exact UTF8 JSON required") from error
    raw.require(type(decoded) is dict, "Distinct complete report root required")
    assert isinstance(decoded, dict)
    # Parser results are builtin JSON families; no candidate serializers can run.
    raw.require(
        raw.canonical(decoded) + b"\n" == content, "Exact canonical whole report framing required"
    )
    raw.validate(decoded, original, runtime=runtime)
    return decoded


def format_report(report: object, *, expected: object, runtime: tuple[str, str]) -> bytes:
    """Own ENTIRE exact tree BEFORE callbacks, semantics, canonical final bytes."""
    owned = raw.own_json(report)
    raw.require(type(owned) is dict, "Exact independent complete report dictionary")
    assert isinstance(owned, dict)
    original, bound = _expectation(expected, runtime)
    raw.validate(owned, original, runtime=runtime)
    content = raw.canonical(owned) + b"\n"
    raw.require(len(content) <= bound, "Complete final canonical bytes exceed original reservation")
    return content
