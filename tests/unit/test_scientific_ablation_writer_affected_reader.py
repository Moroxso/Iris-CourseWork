"""Pure readback matrix guards; these dictionaries are NOT native report evidence."""

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import scientific_ablation_writer_affected_raw as affected
from verify_scientific_ablation_causal_design import MODES


def matrix(*, nonwinning: bool) -> list[dict[str, Any]]:
    results = []
    for scenario in ("occupied", "expiry", "nonsurvival", "wide"):
        modes = sorted(MODES) if scenario != "wide" else [("learned", "enabled", "enabled")]
        if scenario == "occupied" and nonwinning:
            modes.append(("random", "disabled", "enabled"))
        for index, mode in enumerate(modes):
            for protocol in ("message", "workspace"):
                wide = scenario == "wide"
                results.append(
                    dict(
                        report={
                            "path": f"tmp/fixture-{scenario}-{index}/public-writer-{protocol}.json"
                        },
                        protocol=protocol,
                        original_mode=mode,
                        geometry=dict(
                            modules=64 if wide else 4,
                            slots=4 if scenario == "expiry" else 1,
                            width=1 if wide else 8,
                            cycles=5 if scenario == "expiry" else 2 if wide else 3,
                            writers=1,
                            empty_threshold=1.0
                            if scenario == "expiry"
                            else 4.0
                            if scenario == "nonsurvival"
                            else 1e-4,
                        ),
                        metrics=dict(
                            effects=12096 if wide else 72,
                            pairs=4032 if wide else 12,
                            actual_teacher_executions=768 if wide else 72,
                            overwrite_effects=1 if scenario == "occupied" else 0,
                            occupied_block_frames=int(
                                scenario == "occupied" and protocol == "message"
                            ),
                            actual_expired_records=int(scenario == "expiry"),
                            nonsurviving_effects=int(scenario == "nonsurvival"),
                        ),
                    )
                )
    return results


@pytest.mark.parametrize("nonwinning", (False, True))
def test_declared_complete_matrix_guard_metadata(nonwinning: bool) -> None:
    affected.geometry_matrix(matrix(nonwinning=nonwinning), includes_nonwinning=nonwinning)


@pytest.mark.parametrize(
    "corruption",
    (
        "missing",
        "extra",
        "duplicate-protocol",
        "mismatched-mode",
        "missing-mode",
        "mismatched-geometry",
        "unexpected-geometry",
        "no-overwrite",
        "no-block",
        "no-expiry",
        "no-nonsurvival",
        "clipped-wide-effects",
        "clipped-wide-pairs",
        "wrong-wide-calls",
        "missing-nonwinning-fixture",
        "extra-nonwinning-fixture",
    ),
)
def test_incomplete_or_count_only_matrix_cannot_pass(corruption: str) -> None:
    results = matrix(nonwinning=True)
    if corruption == "missing":
        results.pop()
    elif corruption == "extra":
        results.append(copy.deepcopy(results[-1]))
    elif corruption == "duplicate-protocol":
        results[0]["protocol"] = results[1]["protocol"]
    elif corruption == "mismatched-mode":
        results[0]["original_mode"] = ("foreign", "enabled", "enabled")
    elif corruption == "missing-mode":
        results[0]["original_mode"] = results[2]["original_mode"]
        results[1]["original_mode"] = results[2]["original_mode"]
    elif corruption == "mismatched-geometry":
        results[0]["geometry"]["slots"] = 2
    elif corruption == "unexpected-geometry":
        results[0]["geometry"]["width"] = 4096
        results[1]["geometry"]["width"] = 4096
    elif corruption == "no-overwrite":
        results[0]["metrics"]["overwrite_effects"] = 0
    elif corruption == "no-block":
        results[0]["metrics"]["occupied_block_frames"] = 0
    elif corruption in ("no-expiry", "no-nonsurvival"):
        scenario, name = (
            ("expiry", "actual_expired_records")
            if corruption == "no-expiry"
            else ("nonsurvival", "nonsurviving_effects")
        )
        for result in results:
            if f"fixture-{scenario}-0/" in result["report"]["path"]:
                result["metrics"][name] = 0
    elif corruption.startswith("clipped-wide") or corruption == "wrong-wide-calls":
        name = {
            "clipped-wide-effects": "effects",
            "clipped-wide-pairs": "pairs",
            "wrong-wide-calls": "actual_teacher_executions",
        }[corruption]
        results[-1]["metrics"][name] -= 1
    elif corruption == "missing-nonwinning-fixture":
        results = matrix(nonwinning=False)
    else:
        extra = copy.deepcopy(results[:2])
        for result in extra:
            result["report"]["path"] = result["report"]["path"].replace(
                "occupied-0/", "occupied-99/"
            )
        results.extend(extra)
    with pytest.raises(ValueError):
        affected.geometry_matrix(results, includes_nonwinning=True)
