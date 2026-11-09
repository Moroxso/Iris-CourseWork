"""Independent byte-boundary tests on pinned whole public reports, no new forwards."""

import json
import math
import struct
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import scientific_ablation_writer_report_raw as raw
import scientific_ablation_writer_report_reader as reader

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "artifacts/operational/r015-writer-report-readback-20261014-run01.json"
RECEIPT_SHA = "f3062411b304e72479221777c210b7279f2aac2bffec61af1e2b4de004f994db"
type Samples = dict[tuple[str, int], tuple[bytes, dict[str, Any]]]


@pytest.fixture(scope="module")
def samples() -> Samples:
    content = (ROOT / RECEIPT).read_bytes()
    assert sha256(content).hexdigest() == RECEIPT_SHA
    receipt = json.loads(content)
    assert receipt["passed"] is True and receipt["actual_runtime_cases"] == 1808
    result: Samples = {}
    for entry in receipt["public_reports"]:
        report, expected = entry["report"], entry["separately_retained_preflight"]
        body = (ROOT / report["path"]).read_bytes()
        original = (ROOT / expected["path"]).read_bytes()
        for body_bytes, pin in ((body, report), (original, expected)):
            assert len(body_bytes) == pin["bytes"]
            assert sha256(body_bytes).hexdigest() == pin["sha256"]
        mode = int(Path(report["path"]).parent.name.rsplit("_", 1)[1])
        result[entry["protocol"], mode] = body, json.loads(original)
    assert len(result) == 24
    return result


def _root(report: dict[str, Any], protocol: str) -> bytes:
    report["sha256"] = raw.domain(
        "iris/scientific-ablation-" + protocol + "-writer-report/v1", report["payload"]
    )
    return raw.canonical(report) + b"\n"


def _rehash_output(output: dict[str, Any]) -> None:
    """Rebind ALL plan/control/native-phase/observed invocation hashes, not just root."""
    ctl = output["controlled"]
    plan = ctl["plan"]
    ctl["plan_sha256"] = raw.domain("iris/cognitive-ablation-control-plan/v1", plan)
    ctl["ablation_policy_sha256"] = raw.domain(
        "iris/cognitive-ablation-policy/v1", plan["ablation"]
    )
    ctl["control_plan_sha256"] = raw.domain(
        "iris/workspace-message-plan/v1"
        if plan["control"]["kind"] == "iris-workspace-message-plan-v1"
        else "iris/workspace-intervention-plan/v1",
        plan["control"],
    )
    ctl["trace"]["intervention_plan_sha256"] = ctl["plan_sha256"]
    ctl["invocation_sha256"] = raw.domain(
        "iris/cognitive-ablation-controlled-teacher-invocation/v1",
        {
            name: ctl[name]
            for name in (
                "plan_sha256",
                "prompt_sha256",
                "target_sha256",
                "teacher_graph_sha256",
                "sidecar_state_sha256",
                "supervised_token_count",
            )
        },
    )
    for observation in output["observations"]:
        for phase in ("before_write", "native_written"):
            observation[phase + "_sha256"] = raw.state(observation[phase], m=4, k=4, d=8)
    output["invocation_sha256"] = raw.domain(
        "iris/ablation-workspace-observed-teacher-invocation/v1",
        dict(
            controlled_invocation_sha256=ctl["invocation_sha256"],
            maximum_observation_scalars=output["maximum_observation_scalars"],
            actual_observation_scalars=output["actual_observation_scalars"],
            observations=[
                {
                    name: observation[name]
                    for name in (
                        "cycle",
                        "before_write_sha256",
                        "native_written_sha256",
                    )
                }
                for observation in output["observations"]
            ],
        ),
    )


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("mode", range(12))
def test_all24_whole_public_bytes_against_separate_pinned_original_preflights(
    samples: Samples, protocol: str, mode: int
) -> None:
    content, original = samples[protocol, mode]
    original_before = raw.canonical(original)
    report = reader.read_report(content, expected=original, runtime=raw.CPU_STREAM)
    assert reader.format_report(report, expected=original, runtime=raw.CPU_STREAM) == content
    assert raw.canonical(original) == original_before
    # Returned trees are owned, never aliases into the separately trusted input.
    report["payload"]["source_metadata"]["run_seed"] += 1
    assert raw.canonical(original) == original_before


@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"{}",
        b"[]\n",
        b"null\n",
        b"true\n",
        b"1\n",
        b'"x"\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":-Infinity}\n',
        b'{"x":1e1000}\n',
        b'{"x":-1e1000}\n',
        b'{"x":0,"x":1}\n',
        b'{"x":{"y":0,"y":1}}\n',
        b'{"x":[{"y":0,"y":1}]}\n',
        b'{"x":0,"\\u0078":1}\n',
        b"\xef\xbb\xbf{}\n",
        b'{"x":"\xff"}\n',
        b"{}\n{}\n",
        b'{"x":}\n',
        b'{"x":"\\"}\n',
        b'{"x":]\n',
    ),
)
def test_malformed_duplicate_nonfinite_and_nonroot_json_denied(
    samples: Samples, body: bytes
) -> None:
    _, original = samples["workspace", 0]
    with pytest.raises(ValueError):
        reader.read_report(body, expected=original, runtime=raw.CPU_STREAM)


@pytest.mark.parametrize("change", ("space", "crlf", "no-lf", "two-lf", "indent", "order"))
def test_legitimate_report_noncanonical_framing_denied(samples: Samples, change: str) -> None:
    content, original = samples["workspace", 0]
    report = json.loads(content)
    changed = {
        "space": b" " + content,
        "crlf": content[:-1] + b"\r\n",
        "no-lf": content[:-1],
        "two-lf": content + b"\n",
        "indent": json.dumps(report, indent=1).encode() + b"\n",
        "order": json.dumps(dict(reversed(list(report.items()))), separators=(",", ":")).encode()
        + b"\n",
    }[change]
    with pytest.raises(ValueError):
        reader.read_report(changed, expected=original, runtime=raw.CPU_STREAM)


@pytest.mark.parametrize("change", ("oversize", "depth", "unclosed"))
def test_bound_and_depth_denied_before_json_parser(samples: Samples, change: str) -> None:
    _, original = samples["workspace", 0]
    body = (
        b" " * (original["report_byte_preflight_bound"] + 1)
        if change == "oversize"
        else b"[" * 41 + b"0" + b"]" * 41
        if change == "depth"
        else b'{"x":"unterminated'
    )
    original_loads = json.loads

    def only_original(encoded: str, **options: Any) -> Any:
        # The authenticated layout has fixed encoded source JSON of its own.
        # Deny candidate parsing, not that prior legitimate trust-anchor check.
        if "object_pairs_hook" in options:
            raise AssertionError("Candidate parser ran before guard")
        return original_loads(encoded, **options)

    with (
        patch.object(json, "loads", side_effect=only_original),
        pytest.raises(ValueError),
    ):
        reader.read_report(body, expected=original, runtime=raw.CPU_STREAM)


def test_json_brackets_and_escaped_quotes_inside_strings_are_not_structural_depth(
    samples: Samples,
) -> None:
    _, original = samples["workspace", 0]
    body = raw.canonical({"x": "[" * 100 + '\\"}' * 100}) + b"\n"
    with patch.object(json, "loads", wraps=json.loads) as parse:
        with pytest.raises(ValueError):
            reader.read_report(body, expected=original, runtime=raw.CPU_STREAM)
        assert sum("object_pairs_hook" in call.kwargs for call in parse.call_args_list) == 1


class Foreign:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("Foreign candidate attribute callback")


class ForeignBytes(bytes):
    def __len__(self) -> int:
        raise AssertionError("Foreign candidate byte length callback")


class ForeignDict(dict[str, Any]):
    def items(self) -> Any:
        raise AssertionError("Foreign candidate dictionary callback")


class ForeignList(list[Any]):
    def __iter__(self) -> Any:
        raise AssertionError("Foreign candidate list callback")


class ForeignFloat(float):
    def __float__(self) -> float:
        raise AssertionError("Foreign candidate numeric coercion")


class ForeignKey(str):
    armed = False

    def __hash__(self) -> int:
        if self.armed:
            raise AssertionError("Foreign key hashed before owning key inventory")
        return str.__hash__(self)


@pytest.mark.parametrize("kind", ("foreign", "byte-subclass", "bytearray", "str"))
def test_nonexact_bytes_denied_before_candidate_callbacks(samples: Samples, kind: str) -> None:
    _, original = samples["workspace", 0]
    value: object = {
        "foreign": Foreign(),
        "byte-subclass": ForeignBytes(b"{}\n"),
        "bytearray": bytearray(b"{}\n"),
        "str": "{}\n",
    }[kind]
    with pytest.raises(ValueError):
        reader.read_report(value, expected=original, runtime=raw.CPU_STREAM)


@pytest.mark.parametrize("change", ("root", "dictionary", "array", "float", "leaf", "key"))
@pytest.mark.parametrize("location", ("report", "original"))
def test_foreign_subclasses_and_keys_denied_before_callbacks(
    samples: Samples, change: str, location: str
) -> None:
    content, original = samples["workspace", 0]
    report = json.loads(content)
    expected = deepcopy(original)
    target = report if location == "report" else expected
    if change == "root":
        target = ForeignDict(target)
    elif change == "dictionary":
        target["foreign"] = ForeignDict()
    elif change == "array":
        target["foreign"] = ForeignList()
    elif change == "float":
        target["foreign"] = ForeignFloat(0.0)
    elif change == "leaf":
        target["foreign"] = Foreign()
    else:
        key = ForeignKey("foreign")
        target[key] = 0
        key.armed = True
    with pytest.raises(ValueError):
        reader.format_report(
            target if location == "report" else report,
            expected=target if location == "original" else expected,
            runtime=raw.CPU_STREAM,
        )


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("field", ("hidden-root", "hidden-payload", "bool-count", "float-count"))
def test_rehashed_closed_inventory_and_primitive_family_denials(
    samples: Samples, protocol: str, field: str
) -> None:
    content, original = samples[protocol, 0]
    report = json.loads(content)
    if field == "hidden-root":
        report["private_rng"] = []
    elif field == "hidden-payload":
        report["payload"]["private_rng"] = []
    else:
        report["payload"]["actual_teacher_executions"] = (
            True if field == "bool-count" else float(report["payload"]["actual_teacher_executions"])
        )
    with pytest.raises(ValueError):
        reader.read_report(_root(report, protocol), expected=original, runtime=raw.CPU_STREAM)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("change", ("prompt", "trace-seed", "quota", "control-context"))
def test_locally_consistent_all_nested_rehashes_do_not_replace_original_source(
    samples: Samples, protocol: str, change: str
) -> None:
    content, original = samples[protocol, 0]
    report = json.loads(content)
    role = "shuffle" if protocol == "workspace" else "donor"
    output = report["payload"]["controlled_passes"]["trials"][role][0]["output"]
    ctl = output["controlled"]
    previous = output["invocation_sha256"]
    if change == "prompt":
        ctl["prompt_sha256"] = "e" * 64
        for proposal in ctl["messages"]:
            proposal["prompt_sha256"] = ctl["prompt_sha256"]
    elif change == "trace-seed":
        ctl["trace"]["options"]["seed"] += 1
    elif change == "quota":
        output["maximum_observation_scalars"] += 1
    elif protocol == "workspace":
        ctl["plan"]["control"]["interventions"][0]["seed"] += 1
    else:
        ctl["plan"]["control"]["donor"]["prompt_sha256"] = "e" * 64
    _rehash_output(output)
    # Local output integrity PASSES first, so rejection is not an obsolete nested SHA.
    raw.output(output, m=4, k=4, d=8, c=3)
    for effect in report["payload"]["effects"]:
        for name in (
            "empty_invocation_sha256",
            "intervention_invocation_sha256",
            "negative_control_invocation_sha256",
        ):
            if effect[name] == previous:
                effect[name] = output["invocation_sha256"]
    with pytest.raises(ValueError):
        reader.read_report(_root(report, protocol), expected=original, runtime=raw.CPU_STREAM)


def test_whole_wrong_shuffle_order_with_valid_tensor_trace_and_nested_hashes_denied(
    samples: Samples,
) -> None:
    content, original = samples["workspace", 0]
    report = json.loads(content)
    frame = report["payload"]["controlled_passes"]["trials"]["shuffle"][0]
    output = frame["output"]
    cycle = frame["write_cycle"] - 1
    state = output["controlled"]["states"][cycle]
    order = list(reversed(range(4)))
    for tensor in state.values():
        tensor["values"][0] = [tensor["values"][0][index] for index in order]
        tensor["tensor_sha256"] = raw.fingerprint(
            b"iris/tensor/v1",
            raw.canonical({"dtype": tensor["dtype"], "shape": tensor["shape"]}),
            raw.dense(tensor["values"], tensor["shape"], tensor["dtype"]),
        )
    raw.state(state, m=4, k=4, d=8)
    trace = output["controlled"]["trace"]["cycles"][cycle]
    trace["full"]["workspace_slots"] = deepcopy(state["slots"]["values"][0])
    for name in (
        "slot_ages",
        "slot_ttl",
        "slot_norms",
        "slot_sources",
        "slot_occupancy",
        "slot_write_counts",
    ):
        trace[name] = [trace[name][index] for index in order]
    matrix = trace["workspace_cosine_matrix"]
    trace["workspace_cosine_matrix"] = [[matrix[i][j] for j in order] for i in order]
    _rehash_output(output)
    raw.output(output, m=4, k=4, d=8, c=3)
    with pytest.raises(ValueError, match="EXACT input-only whole CPU SHUFFLE"):
        reader.read_report(_root(report, "workspace"), expected=original, runtime=raw.CPU_STREAM)


@pytest.mark.parametrize(
    "runtime",
    (
        ("2.13.0+cuda", raw.CPU_STREAM[1]),
        (raw.CPU_STREAM[0], "e" * 40),
        list(raw.CPU_STREAM),
        (True, raw.CPU_STREAM[1]),
        (),
    ),
)
def test_separately_trusted_cpu_stream_drift_or_family_denied(
    samples: Samples, runtime: Any
) -> None:
    content, original = samples["workspace", 0]
    with pytest.raises(ValueError):
        reader.read_report(content, expected=original, runtime=runtime)


@pytest.mark.parametrize(
    "dtype,fmt,width",
    (
        ("float16", "e", 2),
        ("bfloat16", "f", 2),
        ("float32", "f", 4),
        ("float64", "d", 8),
    ),
)
def test_all_four_dtype_signed_zero_extrema_subnormals_and_exact_roundtrip(
    dtype: str, fmt: str, width: int
) -> None:
    max_bits, tiny_bits = {
        "float16": (0x7BFF, 1),
        "bfloat16": (0x7F7F, 1),
        "float32": (0x7F7FFFFF, 1),
        "float64": (0x7FEFFFFFFFFFFFFF, 1),
    }[dtype]
    for bits in (0, 1 << (8 * width - 1), max_bits, tiny_bits):
        body = bits.to_bytes(width, "little")
        value = struct.unpack("<" + fmt, b"\0\0" + body if dtype == "bfloat16" else body)[0]
        assert raw.scalar_bytes(value, dtype) == body
        if bits == 1 << (8 * width - 1):
            assert math.copysign(1.0, value) == -1.0


@pytest.mark.parametrize("dtype", raw.FLOATS)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf"), True, 1))
def test_all_four_dtype_nonfinite_and_wrong_families_denied(dtype: str, value: object) -> None:
    with pytest.raises(ValueError):
        raw.scalar_bytes(value, dtype)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_no_rounding_fabricated_decimal_or_overflow_into_declared_dtype(dtype: str) -> None:
    for value in (0.1, 1e300):
        with pytest.raises(ValueError):
            raw.scalar_bytes(value, dtype)
    assert raw.scalar_bytes(0.1, "float64") == struct.pack("<d", 0.1)


@pytest.mark.parametrize("dtype,fmt", (("float16", "e"), ("bfloat16", "f")))
def test_every_finite_16bit_pattern_in_independent_codec(dtype: str, fmt: str) -> None:
    finite = 0
    for bits in range(65536):
        body = bits.to_bytes(2, "little")
        value = struct.unpack("<" + fmt, b"\0\0" + body if dtype == "bfloat16" else body)[0]
        if math.isfinite(value):
            assert raw.scalar_bytes(value, dtype) == body
            finite += 1
    assert finite == (63488 if dtype == "float16" else 65280)


@pytest.mark.parametrize(
    "seed,slots", ((True, 4), (1.0, 4), (-1, 4), (2**63, 4), (0, True), (0, 4.0), (0, 0), (0, 65))
)
def test_exact_cpu_order_never_coerces_seed_or_geometry(seed: Any, slots: Any) -> None:
    with pytest.raises(ValueError):
        raw.cpu_order(seed, slots, raw.CPU_STREAM)


def test_independent_cpu_order_all64_slot_counts_128_seeds_matches_pinned_torch() -> None:
    # Separate trusted primitive parity process; this intentionally imports Torch.
    # It is NOT the import-free reader observer and executes no model.
    code = r"""
import sys
from pathlib import Path
from hashlib import sha256
sys.path.insert(0,str(Path.cwd()/'scripts'))
import scientific_ablation_writer_report_raw as raw
import torch
assert (str(torch.__version__),torch.version.git_version)==raw.CPU_STREAM
before=torch.random.get_rng_state().clone()
seeds=[0,1,42,1337,2026,2**32-1,2**32,2**32+1,2**63-1]
seeds += [int.from_bytes(sha256(f'iris-independent-cpu/{i}'.encode()).digest()[:8],'big')
          %2**63 for i in range(119)]
assert len(seeds)==128 and len(set(seeds))==128
identities=0
for slots in range(1,65):
    for seed in seeds:
        expected=torch.randperm(slots,device='cpu',
            generator=torch.Generator(device='cpu').manual_seed(seed)).tolist()
        assert raw.cpu_order(seed,slots,raw.CPU_STREAM)==expected
        identities += expected==list(range(slots))
assert torch.equal(before,torch.random.get_rng_state())
assert identities>=128
print('8192 pinned CPU permutations; caller RNG unchanged; zero model forwards')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip() == (
        "8192 pinned CPU permutations; caller RNG unchanged; zero model forwards"
    )


def test_independent_all_finite_16bit_patterns_match_pinned_torch_bytes_and_values() -> None:
    # This separate trusted primitive process is not an import-free observer or AMP run.
    code = r"""
import math, struct, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import scientific_ablation_writer_report_raw as raw
import torch
assert (str(torch.__version__),torch.version.git_version)==raw.CPU_STREAM
before=torch.random.get_rng_state().clone()
bits=torch.arange(65536,dtype=torch.int32,device='cpu').to(torch.int16)
for dtype,fmt,count in (('float16','e',63488),('bfloat16','f',65280)):
    native=bits.view(getattr(torch,dtype))
    finite=torch.isfinite(native)
    values=native[finite].tolist()
    patterns=bits[finite].view(torch.uint16).tolist()
    assert len(values)==len(patterns)==count
    for value,pattern in zip(values,patterns,strict=True):
        body=pattern.to_bytes(2,'little')
        decoded=struct.unpack('<'+fmt,b'\0\0'+body if dtype=='bfloat16' else body)[0]
        assert decoded==value and math.copysign(1.0,decoded)==math.copysign(1.0,value)
        assert raw.scalar_bytes(value,dtype)==body
for dtype,fmt,maxbits,width in (
    ('float32','f',0x7f7fffff,4),('float64','d',0x7fefffffffffffff,8)):
    values=[struct.unpack('<'+fmt,bits.to_bytes(width,'little'))[0]
            for bits in (0,1,1<<(width*8-1),maxbits)]
    native=torch.tensor(values,dtype=getattr(torch,dtype),device='cpu')
    assert native.tolist()==values
    assert native.view(torch.uint8).tolist()==list(b''.join(raw.scalar_bytes(v,dtype)
                                                          for v in values))
assert torch.equal(before,torch.random.get_rng_state())
print('63488 float16 + 65280 bfloat16 patterns; four dtype native edges; zero model forwards')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip() == (
        "63488 float16 + 65280 bfloat16 patterns; four dtype native edges; zero model forwards"
    )


def test_fresh_reader_import_and_all24_byte_checks_without_project_or_model_imports() -> None:
    code = r"""
import importlib.abc, json, sys
from pathlib import Path
from hashlib import sha256
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'iris','torch','pydantic','tests','pytest'}:
            raise AssertionError('Forbidden reader import: '+fullname)
root=Path.cwd()
sys.meta_path.insert(0,Deny())
sys.path.insert(0,str(root/'scripts'))
import scientific_ablation_writer_report_reader as reader
import scientific_ablation_writer_report_raw as raw
receipt_bytes=(root/'artifacts/operational/r015-writer-report-readback-20261014-run01.json').read_bytes()
expected_sha='f3062411b304e72479221777c210b7279f2aac2bffec61af1e2b4de004f994db'
assert sha256(receipt_bytes).hexdigest()==expected_sha
receipt=json.loads(receipt_bytes)
for entry in receipt['public_reports']:
    body=(root/entry['report']['path']).read_bytes()
    expected_bytes=(root/entry['separately_retained_preflight']['path']).read_bytes()
    pins=((body,entry['report']),(expected_bytes,entry['separately_retained_preflight']))
    for content,pin in pins:
        assert len(content)==pin['bytes'] and sha256(content).hexdigest()==pin['sha256']
    expected=json.loads(expected_bytes)
    value=reader.read_report(body,expected=expected,runtime=raw.CPU_STREAM)
    assert reader.format_report(value,expected=expected,runtime=raw.CPU_STREAM)==body
assert len(receipt['public_reports'])==24
forbidden={'iris','torch','pydantic','tests','pytest'}
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
print('24 whole read/format checks; no forbidden imports; zero new model forwards')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    assert result.stdout.strip() == (
        "24 whole read/format checks; no forbidden imports; zero new model forwards"
    )
