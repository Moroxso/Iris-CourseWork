"""Exact whole affected module union after the one-slot production correction.

Not all repository tests/scientific acceptance. Includes every inherited1808
case, independent byte reader, all geometry/nonwinning cases, loss/usage and
direct/transitive five-term optimizer/run/checkpoint/collapse compatibility.
No module chosen by passing outcome, no exclusions to fit time or memory.
"""

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "artifacts/operational/r015-writer-report-before-20261014-run01.json"
SHA = "880409b2dc28cb9b239ac503351f01c3c7bda7fec7a3124df090c6f9bff70144"
EXPECTED_CASES = 3137  # ACTUAL complete collectione43a75, not a sum of prior gates.
ADDITIONAL = (
    "tests/unit/test_scientific_ablation_writer_independent_reader.py",
    "tests/integration/test_scientific_ablation_writer_geometry_reports.py",
    "tests/integration/test_scientific_ablation_writer_nonwinning_report.py",
    "tests/unit/test_scientific_ablation_writer_gate_supervisor.py",
    "tests/unit/test_scientific_ablation_writer_affected_reader.py",
    "tests/unit/test_scientific_ablation_writer_routing_budget.py",
    "tests/unit/test_workspace_usage.py",
    "tests/unit/test_workspace_usage_single_slot.py",
    "tests/unit/test_losses.py",
    "tests/unit/test_scientific_intervention_nll.py",
    "tests/unit/test_scientific_collapse.py",
    "tests/unit/test_scientific_ablation_collapse.py",
    "tests/unit/test_scientific_ablation_lifecycle_state.py",
    "tests/integration/test_scientific_teacher.py",
    "tests/integration/test_scientific_optimization.py",
    "tests/integration/test_scientific_run_scale.py",
    "tests/integration/test_scientific_checkpoint.py",
    "tests/integration/test_scientific_checkpoint_delivery.py",
    "tests/integration/test_scientific_lifecycle.py",
    "tests/integration/test_scientific_collapse.py",
    "tests/integration/test_scientific_workspace.py",
    "tests/integration/test_scientific_message_causal.py",
    "tests/integration/test_scientific_intervention_nll.py",
    "tests/integration/test_scientific_ablation_teacher.py",
    "tests/integration/test_scientific_ablation_optimization.py",
    "tests/integration/test_scientific_ablation_run.py",
    "tests/integration/test_scientific_ablation_checkpoint.py",
    "tests/integration/test_scientific_ablation_lifecycle.py",
    "tests/integration/test_scientific_ablation_durable_lifecycle.py",
    "tests/integration/test_scientific_ablation_generation_source.py",
    "tests/integration/test_scientific_ablation_generation_context.py",
    "tests/integration/test_scientific_ablation_heldout.py",
    "tests/integration/test_scientific_ablation_collapse.py",
    "tests/integration/test_scientific_ablation_collapse_scale.py",
)


def modules() -> tuple[list[str], list[str]]:
    content = (ROOT / BEFORE).read_bytes()
    if sha256(content).hexdigest() != SHA:
        raise ValueError("Immutable inherited1808scope changed")
    original = json.loads(content)
    ids = original["nodeids"]
    if len(ids) != len(set(ids)) or len(ids) != 1808:
        raise ValueError("Inherited whole1808ordered IDs required")
    result = list(dict.fromkeys([*original["modules"], *ADDITIONAL]))
    if any(not (ROOT / module).is_file() for module in result):
        raise ValueError("Whole affected module missing; no scope clipping")
    return result, ids


if __name__ == "__main__":
    selected, inherited = modules()
    print(json.dumps(dict(modules=selected, inherited_cases=len(inherited)), sort_keys=True))
