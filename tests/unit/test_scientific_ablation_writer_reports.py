"""Whole retained native reports vs separate source expectations; no new forwards."""

import json
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.artifacts import scientific_ablation_writer_reports_v1 as module
from iris.artifacts.scientific_ablation_causal_report_layout_v1 import (
    ScientificAblationCausalReportBytePreflightV1,
    build_scientific_ablation_causal_report_byte_preflight_v1,
)
from iris.artifacts.scientific_ablation_writer_statistics_v1 import (
    ScientificAblationWorkspaceWriterEffectV1,
)
from iris.artifacts.scientific_ablation_writer_tensors_v1 import (
    ScientificAblationWriterStateV1,
    writer_state_wire_v1,
)
from iris.interventions.messages import message_content_sha256_v1
from iris.utils.hashing import (
    canonical_json_bytes,
    canonical_json_file_bytes,
    domain_sha256,
    tensor_sha256,
)
from tests.unit.test_scientific_ablation_writer_observations import _rehash

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = module.ScientificAblationWriterCPURuntimeV1(
    torch_version=module.CPU_VERSION, torch_git_version=module.CPU_REVISION
)
type Samples = dict[tuple[str, int], dict[str, Any]]


@pytest.fixture(scope="module")
def native_reports() -> Samples:
    receipt = (
        ROOT / "artifacts/operational/r015-native-writer-readback-20261013-run01.json"
    ).read_bytes()
    assert (
        sha256(receipt).hexdigest()
        == "db2470b637990d439341f10ba7014b2dfd2527ff18df9c39ec8aac46df4bfe52"
    )
    saved = json.loads(receipt)
    assert saved["passed"] is True and saved["exact_cases"] == 1371
    result: Samples = {}
    for pin in saved["native_writer_source_files"]:
        path = ROOT / pin["path"]
        raw = path.read_bytes()
        assert len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["sha256"]
        match = re.fullmatch(r"test_all12_both_full_original_([0-9]+)", path.parent.name)
        assert match is not None
        protocol = "message" if path.name.endswith("message.json") else "workspace"
        result[protocol, int(match.group(1))] = json.loads(raw)["projected_report"]["payload"]
    assert len(result) == 24
    return result


def payload(raw: dict[str, Any], protocol: str) -> module.WriterPayload:
    if protocol == "message":
        return module.ScientificAblationMessageWriterReportPayloadV1.model_validate(raw)
    return module.ScientificAblationWorkspaceWriterReportPayloadV1.model_validate(raw)


def original_preflight(
    value: module.WriterPayload,
) -> ScientificAblationCausalReportBytePreflightV1:
    source = value.source_metadata
    return build_scientific_ablation_causal_report_byte_preflight_v1(
        source.observed_budget,
        evaluation_plan=source.evaluation_plan,
        evaluation=source.evaluation,
        evaluation_complete=source.evaluation_complete,
        run_seed=source.run_seed,
        ordinary_teacher_graph_sha256=source.ordinary_teacher_graph_sha256,
        controlled_teacher_graph_sha256=source.controlled_teacher_graph_sha256,
        sidecar_sha256=source.sidecar_sha256,
        maximum_report_bytes=value.byte_preflight.maximum_report_bytes,
    )


def production_root(raw: dict[str, Any], protocol: str) -> module.WriterReport:
    envelope = dict(
        sha256=domain_sha256(module._domain(protocol == "message"), canonical_json_bytes(raw)),
        payload=raw,
    )
    if protocol == "message":
        return module.IrisScientificAblationMessageWriterReportV1.model_validate(envelope)
    return module.IrisScientificAblationWorkspaceWriterReportV1.model_validate(envelope)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("mode", range(12))
def test_all24_retained_complete_native_payloads_new_domains_and_canonical_roundtrip(
    native_reports: Samples, protocol: str, mode: int
) -> None:
    original = native_reports[protocol, mode]
    value = payload(original, protocol)
    pf = original_preflight(value)
    # This fixture's separately retained source metadata is the expectation.
    # This does not create fresh physical source authentication/native measurement.
    with patch.object(
        torch.nn.Module, "_call_impl", side_effect=AssertionError("Unexpected model call")
    ) as forward:
        report = module.build_scientific_ablation_writer_report_v1(
            value, preflight=pf, runtime=RUNTIME
        )
        content = module.scientific_ablation_writer_report_bytes_v1(
            report, preflight=pf, runtime=RUNTIME
        )
        reread = module.read_scientific_ablation_writer_report_v1(
            content, preflight=pf, runtime=RUNTIME
        )
        assert not forward.called
    assert canonical_json_bytes(report.payload.model_dump(mode="json")) == canonical_json_bytes(
        original
    )
    assert canonical_json_file_bytes(reread.model_dump(mode="json")) == content
    assert report.sha256 == domain_sha256(
        module._domain(protocol == "message"), canonical_json_bytes(original)
    )
    assert len(content) <= pf.report_byte_preflight_bound <= pf.maximum_report_bytes


@pytest.mark.parametrize("slots", range(1, 65))
def test_exact_cpu_permutation_all_declared_slots_128_seeds_private_rng(slots: int) -> None:
    assert (
        str(torch.__version__) == module.CPU_VERSION
        and torch.version.git_version == module.CPU_REVISION
    )
    before = torch.random.get_rng_state().clone()
    for index in range(128):
        seed = (
            int.from_bytes(sha256(f"iris-writer-cpu-parity/{index}".encode()).digest()[:8], "big")
            % 2**63
        )
        actual = tuple(
            torch.randperm(
                slots, generator=torch.Generator(device="cpu").manual_seed(seed), device="cpu"
            ).tolist()
        )
        assert module.writer_cpu_permutation_v1(seed, slots, runtime=RUNTIME) == actual
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize("field", ("run_seed", "controlled_teacher_graph_sha256", "sidecar_sha256"))
def test_coherently_rehashed_foreign_source_cannot_supply_its_own_expectation(
    native_reports: Samples, protocol: str, field: str
) -> None:
    original = payload(native_reports[protocol, 0], protocol)
    pf = original_preflight(original)
    broken = deepcopy(native_reports[protocol, 0])
    broken["source_metadata"][field] = pf.run_seed + 1 if field == "run_seed" else "e" * 64
    report = production_root(broken, protocol)
    with pytest.raises(ValueError, match="original source/preflight"):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize(
    "change",
    (
        "ordinary-order",
        "missing-pass",
        "duplicate-pass",
        "reordered-role",
        "effect-distance",
        "effect-flag",
        "effect-anchor",
        "effect-order",
        "byte-binding",
        "teacher-count",
        "scalar-count",
        "post-count",
        "score-ack",
    ),
)
def test_rehashed_whole_report_schedule_original_endpoint_and_counter_denials(
    native_reports: Samples, protocol: str, change: str
) -> None:
    original = payload(native_reports[protocol, 0], protocol)
    pf = original_preflight(original)
    broken = deepcopy(native_reports[protocol, 0])
    trials = broken["controlled_passes"]["trials"]
    first_role = "block" if protocol == "message" else "shuffle"
    if change == "ordinary-order":
        broken["ordinary_rows"].reverse()
    elif change == "missing-pass":
        trials[first_role].pop()
    elif change == "duplicate-pass":
        trials[first_role][1] = deepcopy(trials[first_role][0])
    elif change == "reordered-role":
        trials[first_role].reverse()
    elif change == "effect-distance":
        broken["effects"][0]["genuine_distance"] = 0.123456
    elif change == "effect-flag":
        broken["effects"][0]["survives_to_next_read"] = not broken["effects"][0][
            "survives_to_next_read"
        ]
    elif change == "effect-anchor":
        broken["effects"][0]["intervention_invocation_sha256"] = "e" * 64
    elif change == "effect-order":
        broken["effects"].reverse()
    elif change == "byte-binding":
        broken["byte_preflight"]["layout_sha256"] = "e" * 64
    elif change == "teacher-count":
        broken["actual_teacher_executions"] -= 1
    elif change == "scalar-count":
        broken["actual_trace_proposal_observation_scalar_bound"] -= 1
    elif change == "post-count":
        broken["actual_post_intervention_scalar_bound"] -= 1
    else:
        broken["ordinary_rows"][0]["evaluation_ack_sha256"] = "e" * 64
    report = production_root(broken, protocol)
    with pytest.raises(ValueError):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize(
    "change",
    (
        "leading-space",
        "no-lf",
        "extra-lf",
        "duplicate-key",
        "nonfinite",
        "too-deep",
        "oversize",
        "invalid-utf8",
        "integer-float",
    ),
)
def test_bounded_reader_rejects_noncanonical_or_nonowned_json(
    native_reports: Samples, protocol: str, change: str
) -> None:
    value = payload(native_reports[protocol, 0], protocol)
    pf = original_preflight(value)
    report = production_root(native_reports[protocol, 0], protocol)
    content = canonical_json_file_bytes(report.model_dump(mode="json"))
    if change == "leading-space":
        content = b" " + content
    elif change == "no-lf":
        content = content[:-1]
    elif change == "extra-lf":
        content += b"\n"
    elif change == "duplicate-key":
        content = b'{"sha256":"' + report.sha256.encode() + b'",' + content[1:]
    elif change == "nonfinite":
        content = b'{"sha256":NaN,"payload":{}}\n'
    elif change == "too-deep":
        content = b"[" * 41 + b"0" + b"]" * 41
    elif change == "oversize":
        content = b" " * (pf.report_byte_preflight_bound + 1)
    elif change == "invalid-utf8":
        content = b"\xff\n"
    else:
        raw = deepcopy(native_reports[protocol, 0])
        raw["ordinary_rows"][0]["ordinary_mean_target_nll"] = 1
        content = canonical_json_file_bytes(
            dict(
                sha256=domain_sha256(
                    module._domain(protocol == "message"), canonical_json_bytes(raw)
                ),
                payload=raw,
            )
        )
    with pytest.raises(ValueError):
        module.read_scientific_ablation_writer_report_v1(content, preflight=pf, runtime=RUNTIME)


def test_wide_effect_ownership_keeps_12096_items_without_generic_8192_limit(
    native_reports: Samples,
) -> None:
    value = payload(native_reports["workspace", 0], "workspace")
    wide = value.model_copy(update={"effects": (value.effects[0],) * 12096})
    raw = module._report_raw(wide, module.ScientificAblationWorkspaceWriterReportPayloadV1)
    assert isinstance(raw["effects"], tuple) and len(raw["effects"]) == 12096
    # Ownership is NOT semantic acceptance of repeated effects. Separate original
    # source schedule/count denies this synthetic specimen at the full formatter.
    with pytest.raises(ValueError, match="inventory"):
        module.build_scientific_ablation_writer_report_v1(
            wide, preflight=original_preflight(value), runtime=RUNTIME
        )


def test_foreign_effect_class_serializer_denied_before_callbacks(native_reports: Samples) -> None:
    calls: list[str] = []

    class ForeignEffect(ScientificAblationWorkspaceWriterEffectV1):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append("serializer")
            raise AssertionError("Foreign serializer called")

    value = payload(native_reports["workspace", 0], "workspace")
    bad = ForeignEffect.model_construct(**vars(value.effects[0]))
    reflected = value.model_copy(update={"effects": (bad, *value.effects[1:])})
    with pytest.raises(ValueError):
        module._report_raw(reflected, module.ScientificAblationWorkspaceWriterReportPayloadV1)
    assert not calls


@pytest.mark.parametrize(
    "change",
    (
        "input-seed",
        "recipient-prompt",
        "competition",
        "prefix",
        "trace-seed",
        "checkpoint",
        "observation-quota",
    ),
)
def test_locally_consistent_rehashed_pass_is_not_original_schedule_or_context(
    native_reports: Samples, change: str
) -> None:
    value = payload(native_reports["workspace", 0], "workspace")
    pf = original_preflight(value)
    broken = deepcopy(native_reports["workspace", 0])
    frames = broken["controlled_passes"]["trials"]["shuffle"]
    frame = frames[1] if change == "prefix" else frames[0]
    control = frame["output"]["controlled"]
    if change == "input-seed":
        control["plan"]["control"]["interventions"][0]["seed"] += 1
    elif change == "recipient-prompt":
        control["prompt_sha256"] = "e" * 64
    elif change in ("competition", "prefix"):
        index = 0 if change == "prefix" else frame["write_cycle"] - 1
        control["trace"]["cycles"][index]["module_bids"][0] += 0.125
    elif change == "trace-seed":
        control["trace"]["options"]["seed"] += 1
    elif change == "checkpoint":
        control["trace"]["options"]["checkpoint_id"] = "e" * 64
    else:
        frame["output"]["maximum_observation_scalars"] += 1
    _rehash(frame["output"])
    report = production_root(broken, "workspace")
    with pytest.raises(ValueError):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)


@pytest.mark.parametrize(
    "seed,slots", ((True, 4), (1.0, 4), (-1, 4), (2**63, 4), (0, True), (0, 4.0), (0, 0), (0, 65))
)
def test_cpu_permutation_never_coerces_seed_or_geometry(seed: Any, slots: Any) -> None:
    with pytest.raises(ValueError):
        module.writer_cpu_permutation_v1(seed, slots, runtime=RUNTIME)


def test_complete_report_reflected_hidden_or_raw_metadata_denied(native_reports: Samples) -> None:
    value = payload(native_reports["workspace", 0], "workspace")
    pf = original_preflight(value)
    report = production_root(native_reports["workspace", 0], "workspace")
    object.__setattr__(report.payload, "unreserved", "private-state")
    with pytest.raises(ValueError, match="field inventory"):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)
    reflected = production_root(native_reports["workspace", 0], "workspace")
    object.__setattr__(reflected, "payload", reflected.payload.model_dump(mode="json"))
    with pytest.raises(ValueError, match="declared model family"):
        module.scientific_ablation_writer_report_bytes_v1(reflected, preflight=pf, runtime=RUNTIME)


@pytest.mark.parametrize("protocol", ("workspace", "message"))
@pytest.mark.parametrize(
    "field", ("mean_genuine_distance", "mean_negative_control_distance", "mean_excess")
)
def test_rehashed_locally_valid_statistics_not_recomputed_original_means(
    native_reports: Samples, protocol: str, field: str
) -> None:
    value = payload(native_reports[protocol, 0], protocol)
    pf = original_preflight(value)
    broken = deepcopy(native_reports[protocol, 0])
    pair = next(pair for pair in broken["statistics"]["pairs"] if pair[field] is not None)
    pair[field] = 0.123456
    report = production_root(broken, protocol)
    with pytest.raises(ValueError, match="grouped sign/Holm"):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)


@pytest.mark.parametrize("change", ("pair", "content", "proposal-source"))
def test_local_donor_rehash_cannot_substitute_original_disjoint_pair_proposal(
    native_reports: Samples, change: str
) -> None:
    value = payload(native_reports["message", 0], "message")
    pf = original_preflight(value)
    broken = deepcopy(native_reports["message", 0])
    frame = broken["controlled_passes"]["trials"]["donor"][0]
    if change == "pair":
        frame["donor_pair_sha256"] = "e" * 64
    else:
        donor = frame["output"]["controlled"]["plan"]["control"]["donor"]
        if change == "content":
            donor["values"][0] += 0.125
            donor["content_sha256"] = message_content_sha256_v1(tuple(donor["values"]))
        else:
            donor["prompt_sha256"] = "e" * 64
        _rehash(frame["output"])
    report = production_root(broken, "message")
    with pytest.raises(ValueError, match=r"original.*(donor|paired)|Original.*donor"):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)


def test_wrong_whole_shuffle_order_keeps_multiset_but_fails_exact_cpu_stream(
    native_reports: Samples,
) -> None:
    value = payload(native_reports["workspace", 0], "workspace")
    pf = original_preflight(value)
    broken = deepcopy(native_reports["workspace", 0])
    frame = broken["controlled_passes"]["trials"]["shuffle"][0]
    cycle = frame["write_cycle"] - 1
    state = frame["output"]["controlled"]["states"][cycle]
    order = list(reversed(range(len(state["slots"]["values"][0]))))
    for tensor in state.values():
        tensor["values"][0] = [tensor["values"][0][index] for index in order]
        tensor["tensor_sha256"] = tensor_sha256(
            torch.tensor(tensor["values"], dtype=getattr(torch, tensor["dtype"]))
        )
    assert (
        writer_state_wire_v1(ScientificAblationWriterStateV1.model_validate(state), num_modules=4)
        == state
    )
    trace = frame["output"]["controlled"]["trace"]["cycles"][cycle]
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
    report = production_root(broken, "workspace")
    with pytest.raises(ValueError, match="exact original input-only CPU permutation"):
        module.scientific_ablation_writer_report_bytes_v1(report, preflight=pf, runtime=RUNTIME)
