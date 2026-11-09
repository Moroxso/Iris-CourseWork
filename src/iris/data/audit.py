"""Bounded construction audit, not model evaluation or a scientific release gate.

Rows are streamed from explicit development seed ranges. Python source/test_code
is never executed. Metrics diagnose references, case labels, duplicates and simple
shortcuts; absence of these findings does not prove no leakage or task validity.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from typing import Any

from iris.data.generator import generate_synthetic_split
from iris.data.schemas import (
    IrisExample,
    SyntheticDatasetPlan,
    SyntheticTemplate,
    parse_structured_response,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate audit JSON member")
        result[key] = value
    return result


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object)


def _repair(row: IrisExample, counts: Counter[str]) -> str:
    facts = row.hidden_facts
    lower, upper, divisor, residue = (
        facts[key] for key in ("lower", "upper", "divisor", "residue")
    )
    if any(type(value) is not int for value in (lower, upper, divisor, residue)):
        raise ValueError("Invalid repair integers")
    assert isinstance(lower, int) and isinstance(upper, int)
    assert isinstance(divisor, int) and isinstance(residue, int)
    valid_bounds = (
        0 < lower < upper <= 300000
        if row.template_id == "iris-synthetic-01-v2"
        else -10000 <= lower < upper <= 10000
    )
    if not valid_bounds or not 2 <= divisor <= 100 or not 0 <= residue < divisor:
        raise ValueError("Unsupported repair audit domain")
    if any(type(facts[key]) is not bool for key in ("include_lower", "include_upper")):
        raise ValueError("Invalid repair boundaries")
    operation = facts["operation"]
    if operation not in ("sum", "count", "squared_sum"):
        raise ValueError("Unsupported repair operation")
    weights = {}
    # Canonical finite coefficient map for an additive function of integer lists.
    # Zero-weight points are omitted, so semantically irrelevant acceptance of 0
    # cannot create a falsely distinct sum/squared_sum function identity.
    first = lower + int(not facts["include_lower"])
    first += (residue - first) % divisor
    points = range(first, upper + int(bool(facts["include_upper"])), divisor)
    if len(points) > 10000:
        raise ValueError("Repair audit coefficient map exceeds its bound")
    for value in points:
        coefficient = 1 if operation == "count" else value if operation == "sum" else value**2
        if coefficient:
            weights[value] = coefficient
    for case in row.code_cases:
        if sum(weights.get(value, 0) for value in case.inputs) != case.expected:
            counts["repair_case_label_mismatches"] += 1
    if not weights:
        counts["repair_constant_zero_functions"] += 1
    counts["repair_cases"] += len(row.code_cases)
    return domain_sha256(
        "iris/synthetic-repair-coefficients/v1", canonical_json_bytes(sorted(weights.items()))
    )


def _delayed(row: IrisExample, counts: Counter[str]) -> None:
    facts = row.hidden_facts
    records = _json(str(facts["records"]))
    if type(records) is not list or not records:
        raise ValueError("Invalid job records")
    for item in records:
        if (
            type(item) is not list
            or len(item) != 3
            or type(item[0]) is not str
            or type(item[1]) is not str
            or type(item[2]) is not int
        ):
            raise ValueError("Invalid job record")
    if facts["operation"] not in ("maximum", "minimum"):
        raise ValueError("Invalid job operation")
    high = facts["operation"] == "maximum"
    qualified = [item for item in records if item[1] == facts["desired_color"]]
    selected = min(qualified, key=lambda item: (-item[2] if high else item[2], item[0]))
    output = parse_structured_response(row.expected_response)
    if output.content != selected[0] or output.next_action != "finish":
        counts["reference_mismatches"] += 1
    if len(qualified) == 1:
        counts["delayed_single_qualifying_job"] += 1
    reversed_goal = min(qualified, key=lambda item: (item[2] if high else -item[2], item[0]))
    if selected[0] == reversed_goal[0]:
        counts["delayed_priority_direction_irrelevant"] += 1
    without_color = min(records, key=lambda item: (-item[2] if high else item[2], item[0]))
    if selected[0] == without_color[0]:
        counts["delayed_color_agnostic_success"] += 1


def _conflict(row: IrisExample, counts: Counter[str]) -> None:
    facts = row.hidden_facts
    priorities, reports = _json(str(facts["priorities"])), _json(str(facts["reports"]))
    if type(priorities) is not dict or not priorities or type(reports) is not list or not reports:
        raise ValueError("Invalid conflict records")
    if any(type(key) is not str or type(value) is not int for key, value in priorities.items()):
        raise ValueError("Invalid source priorities")
    if len(set(priorities.values())) != len(priorities):
        raise ValueError("Tied source priorities need an explicit rule")
    for item in reports:
        if (
            type(item) is not list
            or len(item) != 3
            or type(item[0]) is not str
            or item[0] not in priorities
            or type(item[1]) is not int
            or type(item[2]) is not str
        ):
            raise ValueError("Invalid source report")
    best_source = max(priorities, key=lambda source: priorities[source])
    candidates = [item for item in reports if item[0] == best_source]
    chosen = max(candidates, key=lambda item: item[1])
    status_counts = Counter(item[2] for item in reports)
    expected = {
        "conflict": len(status_counts) > 1,
        "source": best_source,
        "status": chosen[2],
        "reason": "highest_priority_then_latest",
    }
    output = parse_structured_response(row.expected_response)
    if (
        canonical_json_bytes(_json(output.content)) != canonical_json_bytes(expected)
        or output.next_action != "finish"
    ):
        counts["reference_mismatches"] += 1
    minority_count = min(status_counts.values())
    minorities = [status for status, count in status_counts.items() if count == minority_count]
    if len(minorities) == 1 and minorities[0] == chosen[2]:
        counts["conflict_unique_minority_status_success"] += 1
        minority_records = [item for item in reports if item[2] == minorities[0]]
        if len(minority_records) == 1 and minority_records[0][0] == best_source:
            counts["conflict_unique_minority_source_and_status_success"] += 1
    if len({item[2] for item in candidates}) == 1:
        counts["conflict_winning_source_recency_irrelevant"] += 1


def _tool(row: IrisExample, counts: Counter[str], actions: Counter[str]) -> str:
    facts = row.hidden_facts
    keys = (
        "symbol_located",
        "target_read",
        "patch_applied",
        "tests_after_patch",
        "tests_passed",
        "failure_inspected",
    )
    if any(type(facts[key]) is not bool for key in keys):
        raise ValueError("Invalid workflow flags")
    known, read, edited, tested, passed, inspected = (facts[key] for key in keys)
    if (
        (read and not known)
        or (edited and not read)
        or (tested and not edited)
        or (passed and not tested)
        or (inspected and (not tested or passed))
    ):
        counts["tool_incoherent_workflow_states"] += 1
    first_match = next(
        action
        for condition, action in (
            (not known, "search_symbol"),
            (not read, "read_file"),
            (not edited, "edit_file"),
            (not tested, "run_tests"),
            (passed, "finish"),
            (not inspected, "inspect_error"),
            (True, "edit_file"),
        )
        if condition
    )
    actions[first_match] += 1
    output = parse_structured_response(row.expected_response)
    if (
        _json(output.content) != {"target": facts["target"], "action": first_match}
        or output.next_action != first_match
    ):
        counts["reference_mismatches"] += 1
    return first_match


def audit_synthetic_plan(
    plan: SyntheticDatasetPlan,
    *,
    max_rows: int,
    max_logical_bytes: int,
    template_id: SyntheticTemplate = "iris-synthetic-01-v1",
) -> dict[str, Any]:
    """Audit explicit development ranges without publishing or selecting final test data.

    At most 100000 rows and 512 MiB logical bytes; processing is streaming except
    bounded hash indices. Duplicate metrics count repeated rows, not pairs. A row
    can repeat both within and across splits. Full-model tokenizer/context fit,
    template independence and unrestricted Python equivalence are not certified.
    """
    try:
        if type(plan) is not SyntheticDatasetPlan:
            raise TypeError("Exact plan required")
        owned = SyntheticDatasetPlan.model_validate(plan.model_dump(mode="python"))
        if template_id not in ("iris-synthetic-01-v1", "iris-synthetic-01-v2"):
            raise ValueError("Unknown synthetic template profile")
        if template_id == "iris-synthetic-01-v2" and any(
            split.first_seed + split.count > 2**26 for split in owned.splits
        ):
            raise ValueError("Synthetic v2 seed domain exceeded")
        if type(max_rows) is not int or not 1 <= max_rows <= 100000:
            raise ValueError("Invalid row budget")
        if type(max_logical_bytes) is not int or not 1 <= max_logical_bytes <= 512 * 1024**2:
            raise ValueError("Invalid byte budget")
        if sum(split.count for split in owned.splits) > max_rows:
            raise ValueError("Plan exceeds row budget")
        report: dict[str, Any] = {}
        logical_bytes = 0
        digest = sha256(b"iris/synthetic-01/logical-rows/v1\0")
        seen: dict[str, dict[str, int]] = {"problem": {}, "repair_function": {}}
        for split_index, split in enumerate(owned.splits):
            counts: Counter[str] = Counter()
            families: Counter[str] = Counter()
            difficulties: Counter[str] = Counter()
            actions: Counter[str] = Counter()
            stratified_actions: dict[str, Counter[str]] = {
                str(level): Counter() for level in (1, 2, 3)
            }
            max_prompt = max_response = 0
            rows = (
                generate_synthetic_split(split)
                if template_id == "iris-synthetic-01-v1"
                else generate_synthetic_split(split, template_id=template_id)
            )
            for row in rows:
                if counts["rows"] >= split.count:
                    raise ValueError("Generator exceeds declared count")
                row = IrisExample.model_validate(row.model_dump(mode="python"))
                if row.template_id != template_id:
                    raise ValueError("Generator returned the wrong template profile")
                if row.split != split.split or row.seed != split.first_seed + counts["rows"]:
                    raise ValueError("Generator row order differs from the explicit plan")
                raw = canonical_json_bytes(row.model_dump(mode="json"))
                logical_bytes += len(raw)
                if logical_bytes > max_logical_bytes:
                    raise ValueError("Logical data exceeds audit budget")
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
                counts["rows"] += 1
                families[row.task_type] += 1
                difficulties[str(row.difficulty)] += 1
                max_prompt = max(max_prompt, len(row.prompt.encode()))
                max_response = max(max_response, len(row.expected_response.encode()))
                identities = {"problem": row.problem_sha256}
                if row.task_type == "split_spec_repair":
                    identities["repair_function"] = _repair(row, counts)
                elif row.task_type == "delayed_goal":
                    _delayed(row, counts)
                elif row.task_type == "conflict_resolution":
                    _conflict(row, counts)
                else:
                    action = _tool(row, counts, actions)
                    stratified_actions[str(row.difficulty)][action] += 1
                bit = 1 << split_index
                for kind, identity in identities.items():
                    prior = seen[kind].get(identity, 0)
                    counts[kind + "_repeated_within_split"] += bool(prior & bit)
                    counts[kind + "_repeated_across_splits"] += bool(prior & ~bit)
                    seen[kind][identity] = prior | bit
            if counts["rows"] != split.count:
                raise ValueError("Generator returned an incomplete split")
            report[split.split] = {
                "counts": dict(sorted(counts.items())),
                "families": dict(sorted(families.items())),
                "difficulties": dict(sorted(difficulties.items())),
                "tool_actions": dict(sorted(actions.items())),
                "tool_actions_by_difficulty": {
                    level: dict(sorted(values.items()))
                    for level, values in stratified_actions.items()
                },
                "max_prompt_utf8_bytes": max_prompt,
                "max_reference_utf8_bytes": max_response,
            }
        return {
            "kind": "iris-synthetic-construction-audit-v1",
            "template_id": template_id,
            "plan": owned.model_dump(mode="json"),
            "plan_sha256": domain_sha256(
                "iris/synthetic-audit-plan/v1", canonical_json_bytes(owned.model_dump(mode="json"))
            ),
            "logical_sha256": digest.hexdigest(),
            "logical_bytes": logical_bytes,
            "unique_problem_identities": len(seen["problem"]),
            "unique_repair_function_identities": len(seen["repair_function"]),
            "splits": report,
            "scientific_release": False,
            "python_sources_executed": False,
            "tokenizer_context_fit_verified": False,
        }
    except Exception:
        raise ValueError("Synthetic construction audit could not be completed") from None


__all__ = ["audit_synthetic_plan"]
