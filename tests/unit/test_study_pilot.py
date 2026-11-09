"""Independent prompt-derived pilot oracles; never execute emitted Python source."""

import ast
import random
import re
from itertools import permutations
from pathlib import Path

import pytest

from iris.application import study_pilot
from iris.application.errors import IrisStudyBenchmarkError
from iris.application.study_pilot import build_synthetic_study_pilot_v1
from iris.artifacts.study_benchmark_v1 import StudyPythonIntegerListRuleV1
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, domain_sha256


@pytest.mark.parametrize("seed", [0, 1907, 2**63 - 1])
def test_generation_is_reproducible_group_disjoint_and_leaves_global_rng_untouched(
    seed: int,
) -> None:
    state = random.getstate()
    benchmark = build_synthetic_study_pilot_v1(seed=seed, problems_per_cell=2)
    assert benchmark == build_synthetic_study_pilot_v1(seed=seed, problems_per_cell=2)
    assert random.getstate() == state
    dataset = benchmark.payload.dataset.payload
    assert dataset.purpose == "operational_fixture"
    assert dataset.manifest.payload.split_counts.model_dump() == {
        "train": 10,
        "validation": 10,
        "test": 10,
    }
    assert len({item.source_item_id for item in dataset.assignments}) == 30
    assert dataset.construction.source_sha256 == bytes_sha256(
        Path(study_pilot.__file__).read_bytes()
    )
    assert dataset.construction.generation_seed == seed
    assert all(item.source_item_id == item.leakage_group_id for item in dataset.assignments)
    assert all(":train:" not in item.prompt for item in dataset.manifest.payload.examples)


def _route_oracle(prompt: str) -> str:
    start, goal = re.findall(r"from (N\d+) to (N\d+)", prompt)[0]
    edges = set(re.findall(r"(N\d+)->(N\d+)", prompt))
    middle = sorted({node for edge in edges for node in edge} - {start, goal})
    candidates: list[tuple[str, ...]] = []
    for length in range(len(middle) + 1):
        for internal in permutations(middle, length):
            path = (start, *internal, goal)
            if all((path[index], path[index + 1]) in edges for index in range(len(path) - 1)):
                candidates.append(path)
        if candidates:
            return "->".join(min(candidates))
    raise AssertionError("pilot graph is disconnected")


@pytest.mark.parametrize("seed", [0, 13, 1907])
def test_every_text_reference_is_correct_under_an_independent_prompt_derived_oracle(
    seed: int,
) -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=seed, problems_per_cell=3)
    for example, assignment in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples,
        benchmark.payload.dataset.payload.assignments,
        strict=True,
    ):
        prompt = example.prompt
        if assignment.task_family == "split_information":
            package = re.findall(r"Which depot receives (P\d+)", prompt)[0]
            routes = dict(re.findall(r"(P\d+) uses (R\d+)", prompt))
            depots = dict(re.findall(r"(R\d+) ends at (D\d+)", prompt))
            expected = depots[routes[package]]
        elif assignment.task_family == "delayed_planning":
            expected = _route_oracle(prompt)
        elif assignment.task_family == "hidden_goal_retention":
            color = re.findall(r"select a (\w+) job", prompt)[0]
            jobs = re.findall(r"(J\d+)/(\w+)/(\d+)", prompt)
            expected = sorted(
                (job for job in jobs if job[1] == color), key=lambda job: (-int(job[2]), job[0])
            )[0][0]
        elif assignment.task_family == "module_conflict":
            priority = re.findall(r"highest first: (.*?)\.", prompt)[0].split(" > ")
            reports = re.findall(r"(S\d+)/(\d+)/(\w+)", prompt)
            expected = sorted(reports, key=lambda row: (priority.index(row[0]), -int(row[1])))[0][2]
        else:
            continue
        assert example.target == expected, example.example_id


def test_python_references_parse_and_hidden_cases_match_prompt_semantics_without_exec() -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=47, problems_per_cell=12)
    for example, scoring in zip(
        benchmark.payload.dataset.payload.manifest.payload.examples,
        benchmark.payload.scoring,
        strict=True,
    ):
        if not isinstance(scoring.rule, StudyPythonIntegerListRuleV1):
            continue
        tree = ast.parse(example.target)
        assert isinstance(tree.body[0], ast.FunctionDef)
        assert tree.body[0].name == "solve"
        lower, upper = map(int, re.findall(r"(-?\d+) <= x < (-?\d+)", example.prompt)[0])
        divisor, residue = map(int, re.findall(r"x % (\d+) == (\d+)", example.prompt)[0])
        operation = re.findall(r"Return the (sum of squares|sum|count) of", example.prompt)[0]
        assert any(not case.inputs for case in scoring.rule.cases)
        assert any(len(case.inputs) != len(set(case.inputs)) for case in scoring.rule.cases)
        for case in scoring.rule.cases:
            expected = 0
            for value in case.inputs:
                if lower <= value < upper and value % divisor == residue:
                    expected += (
                        1
                        if operation == "count"
                        else value**2
                        if operation == "sum of squares"
                        else value
                    )
            assert case.expected == expected


def test_code_groups_use_additive_function_behavior_not_surface_syntax() -> None:
    benchmark = build_synthetic_study_pilot_v1(seed=53, problems_per_cell=12)
    identities: set[str] = set()
    dataset = benchmark.payload.dataset.payload
    for example, assignment, scoring in zip(
        dataset.manifest.payload.examples,
        dataset.assignments,
        benchmark.payload.scoring,
        strict=True,
    ):
        if not isinstance(scoring.rule, StudyPythonIntegerListRuleV1):
            continue
        lower, upper = map(int, re.findall(r"(-?\d+) <= x < (-?\d+)", example.prompt)[0])
        divisor, residue = map(int, re.findall(r"x % (\d+) == (\d+)", example.prompt)[0])
        operation = re.findall(r"Return the (sum of squares|sum|count) of", example.prompt)[0]
        signature: list[tuple[int, int]] = []
        for value in range(lower, upper):
            if value % divisor != residue:
                continue
            if operation == "count":
                weight = 1
            elif operation == "sum":
                weight = value
            else:
                weight = value * value
            if weight:
                signature.append((value, weight))
        digest = domain_sha256(
            "iris/synthetic-pilot/v1/problem",
            canonical_json_bytes(
                {
                    "family": "code_generation_and_verification",
                    "facts": signature,
                }
            ),
        )
        assert assignment.source_item_id == f"problem-{digest}"
        assert digest not in identities
        identities.add(digest)
    assert len(identities) == 36


@pytest.mark.parametrize(
    ("seed", "count"), [(True, 1), (-1, 1), (2**63, 1), (1, True), (1, 0), (1, 257)]
)
def test_invalid_pilot_configuration_is_rejected(seed: int, count: int) -> None:
    with pytest.raises(IrisStudyBenchmarkError):
        build_synthetic_study_pilot_v1(seed=seed, problems_per_cell=count)
