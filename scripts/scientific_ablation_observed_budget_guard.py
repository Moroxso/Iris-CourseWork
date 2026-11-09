"""Closed raw observed-budget metadata guard for independent source readback.

No Iris/Torch imports or execution. The caller MUST also validate the complete
input design/membership/budget (read_design) and original physical source chain.
This guard checks the distinct outer schema and observed counts, not attestation.
"""

import sys
from pathlib import Path
from re import fullmatch
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_scientific_ablation_causal_observed_budget as budgets
from verify_scientific_ablation_evaluation_io import canonical, domain, require

MAX_BYTES = 4 * 1024**2


def checked_budget(raw: dict[str, Any]) -> None:
    require(
        type(raw) is dict
        and set(raw) == {"sha256", "payload"}
        and type(raw["sha256"]) is str
        and fullmatch(r"[0-9a-f]{64}", raw["sha256"]) is not None
        and type(raw["payload"]) is dict
        and len(canonical(raw)) + 1 <= MAX_BYTES,
        "Exact closed bounded raw observed-budget root required",
    )
    p = raw["payload"]
    require(
        set(p)
        == {
            "kind",
            "scope",
            "scientific_acceptance",
            "design",
            "maximum_observation_scalars",
            *budgets.COUNTS,
        }
        and type(p["kind"]) is str
        and p["kind"] == "iris-scientific-ablation-causal-observed-budget-v1"
        and type(p["scope"]) is str
        and p["scope"]
        == "all-controlled-passes-observed-numeric-preflight-not-measurements-or-byte-guarantee"
        and type(p["scientific_acceptance"]) is str
        and p["scientific_acceptance"] == "not_assessed"
        and raw["sha256"] == domain(budgets.DOMAIN, p),
        "Closed honest observed-budget kind/scope/assessment/domain required",
    )
    design = p["design"]
    require(
        type(design) is dict
        and set(design) == {"payload", "sha256"}
        and type(design["payload"]) is dict
        and design["sha256"]
        == domain("iris/scientific-ablation-causal-design/v1", design["payload"]),
        "Original complete input-design checksum required; caller must rederive it",
    )
    d = design["payload"]
    require(
        all(type(d.get(name)) is dict for name in ("geometry", "budget", "policy")),
        "Original declared design geometry/budget/policy required",
    )
    g, b, policy = d["geometry"], d["budget"], d["policy"]
    require(
        all(
            type(g.get(name)) is int and g[name] > 0
            for name in ("cycle_count", "num_slots", "state_dim")
        )
        and all(
            type(b.get(name)) is int and b[name] > 0
            for name in (
                "ordinary_executions",
                "controlled_executions",
                "total_executions",
                "total_numeric_scalars",
            )
        )
        and type(policy.get("maximum_total_numeric_scalars")) is int
        and 1 <= policy["maximum_total_numeric_scalars"] <= 64_000_000,
        "Exact positive original numeric geometry and quotas required",
    )
    observation = 2 * g["cycle_count"] * g["num_slots"] * (g["state_dim"] + 6)
    total_observation = observation * b["controlled_executions"]
    expected = dict(
        planned_ordinary_teacher_executions=b["ordinary_executions"],
        planned_controlled_teacher_executions=b["controlled_executions"],
        planned_total_teacher_executions=b["total_executions"],
        planned_observed_teacher_executions=b["controlled_executions"],
        observation_scalar_bound_per_pass=observation,
        total_observation_scalar_bound=total_observation,
        original_trace_proposal_scalar_bound=b["total_numeric_scalars"],
        total_trace_proposal_observation_scalar_bound=b["total_numeric_scalars"]
        + total_observation,
    )
    require(
        all(type(p[name]) is int and p[name] == count for name, count in expected.items())
        and type(p["maximum_observation_scalars"]) is int
        and observation <= p["maximum_observation_scalars"] <= 2_000_000
        and expected["total_trace_proposal_observation_scalar_bound"]
        <= policy["maximum_total_numeric_scalars"],
        "Exact rederived observed counts and declared caps required",
    )
