"""Independent synthetic statistic arithmetic plus current native preflight readback.

Audit imports no Iris/Torch/tests/pytest and makes no model passes. Synthetic
writer effects remain false as native measurements; full execution stays OPEN.
"""

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from re import fullmatch
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_controls as controls
import verify_scientific_ablation_causal_report_layout as layout
import verify_scientific_ablation_complete_source_preflight as sources
import verify_scientific_ablation_observation as native
import verify_scientific_ablation_observed_controls as stages
from verify_scientific_ablation_causal_design import MODES, read_design
from verify_scientific_ablation_evaluation_io import binding, canonical, manifest, require
from verify_scientific_ablation_evaluation_ledger import pin

BASELINE = "artifacts/operational/r015-complete-source-before-20261013-run01.json"
BASELINE_SHA = "76430a30b310a31dafb4ff091cd6278df206e1b1ce9136d32cdbfc4750addd6e"
VERDICT = "artifacts/operational/r015-complete-source-readback-20261013-run01.json"
VERDICT_SHA = "dbddb9a7ab6c91170ddc6e638fd81edb4e886c1877dddbec4965578bfbc51bc3"
CHANGED = {
    "src/iris/artifacts/scientific_ablation_causal_report_layout_v1.py",
    "tests/unit/test_scientific_ablation_causal_report_source_types.py",
}
NEW = (
    "src/iris/artifacts/scientific_ablation_writer_statistics_v1.py",
    "tests/unit/test_scientific_ablation_writer_statistics.py",
)
SELF = "scripts/verify_scientific_ablation_writer_statistics.py"


def scope() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for p, sha in ((BASELINE, BASELINE_SHA), (VERDICT, VERDICT_SHA)):
        require(binding(Path(p))["sha256"] == sha, "Immutable accepted1105producer evidence drift")
    old = controls.load(Path(BASELINE))
    bindings, changed = [], []
    for previous in old["bindings"]:
        current = pin(Path.cwd(), previous["path"])
        if previous["path"] in CHANGED:
            require(current != previous, "Both explicit callback guard edits required")
            changed.append(dict(previous=previous, current=current))
        else:
            require(current == previous, "Unrelated accepted source/test changed")
        bindings.append(current)
    auditors = []
    for previous in old["auditors"]:
        current = pin(Path.cwd(), previous["path"])
        require(current == previous, "Accepted observer changed")
        auditors.append(current)
    require(len(changed) == 2, "Only layout guard/source-type negative edits to existing files")
    return (
        [*bindings, *(pin(Path.cwd(), p) for p in NEW)],
        [*auditors, pin(Path.cwd(), SELF)],
        changed,
    )


def snapshot() -> dict[str, Any]:
    frozen = scope()
    old, previous_layout = controls.load(Path(BASELINE)), controls.load(Path(sources.BASELINE))
    inherited = set(previous_layout["nodeids"]) | {
        s for s in old["nodeids"] if any(s.startswith(p + "::") for p in sources.NEW)
    }
    modules = list(dict.fromkeys([*previous_layout["modules"], *sources.NEW, NEW[1]]))
    collected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *modules,
        ],
        capture_output=True,
        check=True,
        timeout=180,
    )
    ids = [s.strip() for s in collected.stdout.decode().splitlines() if "::test_" in s]
    added = set(ids) - inherited
    require(
        len(ids) == len(set(ids)) and inherited <= set(ids) and len(inherited) == 409,
        "ALL direct metadata consumers and all12 native COMPLETE source cases remain",
    )
    require(
        all(s.startswith(NEW[1] + "::") or s.startswith(sources.NEW[0] + "::") for s in added)
        and scope() == frozen,
        "Only new statistics/callback cases and stable scope",
    )
    return dict(
        kind="iris-r015-writer-statistics-before-v1",
        bindings=frozen[0],
        auditors=frozen[1],
        changed=frozen[2],
        modules=modules,
        nodeids=ids,
        expected_cases=len(ids),
        inherited_cases=len(inherited),
        new_cases=len(added),
        actual_collection_exit_code=collected.returncode,
        compatibility=(
            "unchanged legacy/kernel native paths preserved by accepted1105pins; "
            "helper is not called there"
        ),
        native_writer_effects=False,
        full_r015_complete=False,
        scientific_acceptance="not_assessed",
    )


def coin_tail(positive: int, negative: int) -> float:
    # Independent integer Pascal recurrence, NOT production math.comb calls.
    coefficients = [1]
    for _ in range(positive + negative):
        coefficients = [a + b for a, b in zip([0, *coefficients], [*coefficients, 0], strict=True)]
    return float(Fraction(sum(coefficients[positive:]), 2 ** (positive + negative)))


def exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            exact(actual[k], v) for k, v in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            exact(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def specimen(path: Path) -> dict[str, Any]:
    raw = controls.load(path)
    require(
        set(raw)
        == {
            "kind",
            "native_measurements",
            "teacher_executions",
            "scientific_acceptance",
            "design",
            "effects",
            "statistics",
        }
        and raw["kind"] == "iris-tests-synthetic-native-ablation-writer-statistics-v1"
        and raw["native_measurements"] is False
        and type(raw["teacher_executions"]) is int
        and raw["teacher_executions"] == 0
        and raw["scientific_acceptance"] == "not_assessed",
        "Honest SYNTHETIC arithmetic witness, not a measured causal report",
    )
    design_path = path.with_name("design.json")
    protocol, mode, _, _ = read_design(design_path)
    require(controls.load(design_path) == raw["design"], "Exact independently checked input design")
    message = protocol == "paired-message"
    short = "message" if message else "workspace"
    p, stats, effects = raw["design"]["payload"], raw["statistics"], raw["effects"]
    g = p["geometry"]
    require(len(effects) == p["budget"]["maximum_effect_rows"], "No whole effect-inventory shrink")
    donors = {
        row: (other, pair["sha256"])
        for pair in p["donor_pairs"]
        for row, other in (
            (pair["first_row_index"], pair["second_row_index"]),
            (pair["second_row_index"], pair["first_row_index"]),
        )
    }
    units = (
        [list((pair["first_row_index"], pair["second_row_index"])) for pair in p["donor_pairs"]]
        if message
        else [[i] for i in p["selected_row_indices"]]
    )
    indexed: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    trials: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for e in effects:
        names = {
            "kind",
            "row_index",
            "write_cycle",
            "source_module",
            "target_module",
            "slot",
            "survives_to_next_read",
            "overwrote_existing_record",
            "genuine_distance",
            "negative_control_distance",
            "empty_invocation_sha256",
            "intervention_invocation_sha256",
            "negative_control_invocation_sha256",
            "recipient_message_sha256",
            "donor_message_sha256",
            "donor_pair_sha256",
            "donor_row_index",
        }
        require(
            type(e) is dict
            and set(e) == names
            and all(
                type(e[k]) is int
                for k in ("row_index", "write_cycle", "source_module", "target_module", "slot")
            )
            and all(
                type(e[k]) is bool for k in ("survives_to_next_read", "overwrote_existing_record")
            )
            and all(
                type(e[k]) is str and fullmatch(r"[0-9a-f]{64}", e[k]) is not None
                for k in (
                    "empty_invocation_sha256",
                    "intervention_invocation_sha256",
                    "negative_control_invocation_sha256",
                )
            ),
            "Exact closed effect primitive families/digests",
        )
        require(
            e["kind"] == f"iris-scientific-ablation-{short}-writer-effect-v1"
            and e["row_index"] in p["selected_row_indices"]
            and 1 <= e["write_cycle"] < g["cycle_count"]
            and 0 <= e["source_module"] < g["num_modules"]
            and 0 <= e["target_module"] < g["num_modules"]
            and e["source_module"] != e["target_module"]
            and 0 <= e["slot"] < g["num_slots"]
            and all(
                type(e[f]) is float and math.isfinite(e[f]) and 0 <= e[f] <= 1
                for f in ("genuine_distance", "negative_control_distance")
            ),
            "Closed original effect domain",
        )
        if message:
            require(
                type(e["donor_row_index"]) is int
                and all(
                    type(e[k]) is str and fullmatch(r"[0-9a-f]{64}", e[k]) is not None
                    for k in (
                        "donor_pair_sha256",
                        "recipient_message_sha256",
                        "donor_message_sha256",
                    )
                )
                and (e["donor_row_index"], e["donor_pair_sha256"]) == donors[e["row_index"]],
                "Original disjoint pair",
            )
        else:
            require(
                all(
                    e[f] is None
                    for f in (
                        "donor_row_index",
                        "donor_pair_sha256",
                        "recipient_message_sha256",
                        "donor_message_sha256",
                    )
                ),
                "No message data in record effects",
            )
        indexed[e["source_module"], e["target_module"], e["row_index"]].append(e)
        trials[e["row_index"], e["write_cycle"], e["source_module"]].append(e)
    for row in p["selected_row_indices"]:
        for cycle in range(1, g["cycle_count"]):
            groups = [v for (i, c, _), v in trials.items() if (i, c) == (row, cycle)]
            require(
                len(groups) == g["writers_per_cycle"]
                and len({v[0]["slot"] for v in groups}) == len(groups),
                "Every actual declared writer/slot",
            )
            for v in groups:
                require(
                    len(v) == g["num_modules"] - 1
                    and {e["target_module"] for e in v}
                    == set(range(g["num_modules"])) - {v[0]["source_module"]},
                    "Every other receiver exactly once",
                )
    records = []
    for s in range(g["num_modules"]):
        for t in range(g["num_modules"]):
            if s == t:
                continue
            a, b = [], []
            recipients = count = overwritten = expired = 0
            for unit in units:
                ua, ub = [], []
                for i in unit:
                    rows = indexed.get((s, t, i), [])
                    if not rows:
                        continue
                    recipients += 1
                    count += len(rows)
                    overwritten += sum(e["overwrote_existing_record"] for e in rows)
                    expired += sum(not e["survives_to_next_read"] for e in rows)
                    ua.append(math.fsum(e["genuine_distance"] for e in rows) / len(rows))
                    ub.append(math.fsum(e["negative_control_distance"] for e in rows) / len(rows))
                if ua:
                    a.append(math.fsum(ua) / len(ua))
                    b.append(math.fsum(ub) / len(ub))
            delta = [x - y for x, y in zip(a, b, strict=True)]
            pos = sum(x > p["policy"]["numerical_tolerance"] for x in delta)
            neg = sum(x < -p["policy"]["numerical_tolerance"] for x in delta)
            records.append(
                dict(
                    source_module=s,
                    target_module=t,
                    independent_units=len(a),
                    recipient_examples=recipients,
                    write_trials=count,
                    overwritten_trials=overwritten,
                    expired_trials=expired,
                    mean_genuine_distance=math.fsum(a) / len(a) if a else None,
                    mean_negative_control_distance=math.fsum(b) / len(b) if b else None,
                    mean_excess=math.fsum(delta) / len(delta) if delta else None,
                    positive_units=pos,
                    negative_units=neg,
                    tied_units=len(a) - pos - neg,
                    sign_test_p=coin_tail(pos, neg),
                )
            )
    order = sorted(range(len(records)), key=lambda i: records[i]["sign_test_p"])
    scaled = [records[i]["sign_test_p"] * (len(records) - rank) for rank, i in enumerate(order)]
    for rank, index in enumerate(order):
        adjusted = min(1.0, max(scaled[: rank + 1]))
        records[index].update(
            holm_adjusted_p=adjusted, significant_positive_effect=adjusted <= p["policy"]["alpha"]
        )
    require(
        exact(
            stats,
            dict(
                kind=f"iris-scientific-ablation-{short}-writer-statistics-v1",
                inferential_unit="disjoint-donor-pair" if message else "example-cycle-mean",
                independence="input-uniqueness-does-not-prove-task-independence",
                alpha=p["policy"]["alpha"],
                numerical_tolerance=p["policy"]["numerical_tolerance"],
                pairs=records,
            ),
        ),
        "Independent ALL-pair clustered means/counts/ties/exact sign/Holm reconciliation",
    )
    return dict(
        protocol=protocol,
        mode=mode,
        effects=len(effects),
        directed_pairs=len(records),
        native_measurements=False,
    )


def audit(before_path: Path, junit: Path, base: Path) -> dict[str, Any]:
    frozen, before_pin, junit_pin = scope(), binding(before_path), binding(junit)
    before = controls.load(before_path)
    require(
        frozen == (before["bindings"], before["auditors"], before["changed"]),
        "Exact current freeze",
    )
    ids = stages._passing_ids(junit)
    require(
        len(ids) == len(set(ids)) == before["expected_cases"]
        and sorted(ids) == sorted(before["nodeids"]),
        "Exact current affected runtime IDs",
    )
    specimens = sorted(base.rglob("statistics-specimen.json"))
    preflights = sorted(base.glob("test_all12_complete_original*/preflight-*.json"))
    encodings = sorted(base.glob("test_whole_encoding_specimen*/encoding-specimen.json"))
    require(
        (len(specimens), len(preflights), len(encodings)) == (25, 24, 2),
        "All current synthetic/wide/native/layout specimens",
    )
    fixtures = sorted(
        {p.parent for p in preflights}
        | {p for p in base.glob("report-layout-original-source*") if not p.is_symlink()}
    )
    paths = sorted(
        {
            p
            for w in (*specimens, *preflights, *encodings, *fixtures)
            for p in (w if w.is_dir() else w.parent).rglob("*")
            if p.is_file()
        }
    )
    pins = [binding(p) for p in paths]
    results = [specimen(p) for p in specimens]
    require(
        {(r["protocol"], tuple(r["mode"])) for r in results}
        == {(p, m) for p in ("workspace-record", "paired-message") for m in MODES}
        and any(r["effects"] == 12096 and r["directed_pairs"] == 4032 for r in results),
        "Synthetic all12 both/wide complete coverage",
    )
    chains = []
    for fixture in fixtures:
        plans = [
            p
            for p in (fixture / "scores/scientific-ablation-evaluation-ledger-v2").iterdir()
            if p.is_dir()
        ]
        require(len(plans) == 1, "ONE original COMPLETE chain per physical fixture")
        chains.append(sources.physical_chain(fixture, plans[0].name))
    for p in preflights:
        result = sources.preflight(controls.load(p))
        stages.stage(p.with_name(f"stage-{result['protocol']}.json"))
        ledger = native.score_chain_directory(p.parent, result["evaluation_plan_sha256"])
        meta = controls.load(p)["source_metadata"]
        require(
            meta["evaluation"]
            == manifest(
                ledger / "complete/evaluation.json", "iris/scientific-ablation-evaluation/v2"
            ),
            "Actual original scores, not synthetic source attestations",
        )
    for p in encodings:
        layout.specimen(p)
    require(
        scope() == frozen
        and binding(before_path) == before_pin
        and binding(junit) == junit_pin
        and [binding(p) for p in paths] == pins,
        "All producer/observer/positive bytes stable before/end",
    )
    return dict(
        kind="iris-r015-writer-statistics-readback-v1",
        passed=True,
        before=before_pin,
        junit=junit_pin,
        exact_cases=len(ids),
        inherited_cases=before["inherited_cases"],
        new_cases=before["new_cases"],
        changed=frozen[2],
        source_test_pins=len(frozen[0]),
        auditor_pins=len(frozen[1]),
        synthetic_statistics=results,
        actual_native_complete_preflights=24,
        original_physical_chains=chains,
        stable_positive_file_pins=pins,
        reader_forwards=0,
        native_measured_writer_effects=0,
        full_r015_complete=False,
        full_iris_gw_01_complete=False,
        scientific_acceptance="not_assessed",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--before", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        result = snapshot()
    else:
        require(
            args.before is not None and args.junit is not None and args.base is not None,
            "Explicit inputs",
        )
        result = audit(args.before, args.junit, args.base)
    require(
        not any(
            n == p or n.startswith(p + ".")
            for n in sys.modules
            for p in ("iris", "torch", "tests", "pytest")
        ),
        "No project/model imports in reader",
    )
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(
        json.dumps(
            dict(
                mode=args.mode,
                receipt=binding(args.output),
                cases=result.get("expected_cases", result.get("exact_cases")),
                passed=result.get("passed"),
                forwards=0,
            )
        )
    )


if __name__ == "__main__":
    main()
