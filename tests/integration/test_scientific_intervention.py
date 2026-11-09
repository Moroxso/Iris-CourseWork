"""Best-checkpoint paired captures using actual local random Qwen, never labels."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from iris.application import scientific_generation as generation
from iris.application import scientific_intervention as storage
from iris.application.request import IrisRequestOrchestrator
from iris.application.scientific_command import ScientificParquetInputV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
from iris.artifacts.scientific_intervention_v1 import (
    IrisScientificInterventionV1,
    ScientificInterventionPayloadV1,
    intervention_plan_for_row_v1,
)
from iris.data.schemas import IrisExample
from iris.interventions import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_command import _files
from tests.integration.test_scientific_command import loaders as loaders
from tests.integration.test_scientific_generation import _test_rows, _test_spec
from tests.integration.test_scientific_generation import completed as completed

pytestmark = pytest.mark.integration


def plan(*specs: dict[str, Any]) -> WorkspaceInterventionPlanV1:
    return WorkspaceInterventionPlanV1(
        interventions=tuple(WorkspaceInterventionSpecV1.model_validate(spec) for spec in specs)
    )


def generate(
    source: tuple[Path, Path],
    test: ScientificParquetInputV1,
    selected: WorkspaceInterventionPlanV1,
    *,
    sampling: bool = False,
) -> IrisScientificInterventionV1:
    return generation.generate_scientific_intervention_v1(
        config_path=source[0],
        execution_path=source[1],
        test=test,
        decoding=ScientificDecodingPolicyV1(
            strategy="sampling" if sampling else "greedy",
            seed=1337,
            temperature=0.7 if sampling else 1.0,
            top_p=0.8 if sampling else 1.0,
        ),
        plan=selected,
    )


def test_empty_pair_is_exact_original_best_capture_and_round_trips_privately(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    rows = _test_rows()[-2:]
    test = _test_spec(tmp_path / "test.parquet", rows)
    before_files = _files(completed[1])
    before_rng = capture_runtime_state(torch.device("cpu"))
    paired = generate(completed, test, plan())
    baseline = generation.generate_scientific_test_v1(
        config_path=completed[0],
        execution_path=completed[1],
        test=test,
        decoding=ScientificDecodingPolicyV1(strategy="greedy", seed=1337),
    )
    assert paired.payload.baseline == baseline
    assert (baseline.payload.selected_step, baseline.payload.terminal_step) == (1, 3)
    assert _files(completed[1]) == before_files
    assert _same_runtime(before_rng, capture_runtime_state(torch.device("cpu")))
    assert paired.payload.intervened_target_nll == "not_measured"
    assert tuple(row.example_id for row in paired.payload.rows) == tuple(row.id for row in rows)
    for changed, control in zip(paired.payload.rows, baseline.payload.rows, strict=True):
        assert changed.token_ids == control.token_ids
        assert changed.output_text == control.output_text
        assert changed.initial_sampling_rng_sha256 is None
        assert changed.bound_plan_sha256 == paired.payload.plan_sha256
    destination = storage.write_scientific_intervention_v1(paired, root=tmp_path / "captures")
    assert destination.parent.name == "scientific-intervention-v1"
    assert storage.read_scientific_intervention_v1(destination) == paired
    assert (
        storage.write_scientific_intervention_v1(paired, root=tmp_path / "captures") == destination
    )
    content = (destination / "intervention.json").read_text(encoding="utf-8")
    for private in ('"prompt"', '"expected_response"', '"hidden_facts"', '"test_code"'):
        assert private not in content
    assert rows[0].prompt not in repr(paired)
    with pytest.raises(generation.IrisScientificGenerationError):
        generation.write_scientific_generation_v1(cast(Any, paired), root=tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_sampling_and_noise_are_row_bound_repeatable_and_independent_of_labels_and_order(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    rows = _test_rows()[-2:]
    selected = plan({"kind": "gaussian_noise", "seed": 42, "sigma": 0.25})
    first = generate(
        completed, _test_spec(tmp_path / "test.parquet", rows), selected, sampling=True
    )
    repeated = generate(
        completed, _test_spec(tmp_path / "test.parquet", rows), selected, sampling=True
    )
    assert first == repeated
    reordered = generate(
        completed, _test_spec(tmp_path / "test.parquet", rows[::-1]), selected, sampling=True
    )
    assert {row.example_id: row for row in first.payload.rows} == {
        row.example_id: row for row in reordered.payload.rows
    }
    assert len({row.bound_plan_sha256 for row in first.payload.rows}) == 2
    assert all(row.initial_sampling_rng_sha256 is not None for row in first.payload.rows)
    raw = rows[0].model_dump(mode="python")
    answer = json.loads(raw["expected_response"])
    answer["content"] = "deliberately changed reference only"
    raw["expected_response"] = json.dumps(answer)
    changed_rows = (IrisExample.model_validate(raw), rows[1])
    changed = generate(
        completed, _test_spec(tmp_path / "test.parquet", changed_rows), selected, sampling=True
    )
    assert changed.payload.rows[0].example_sha256 != first.payload.rows[0].example_sha256
    for original, updated in zip(first.payload.rows, changed.payload.rows, strict=True):
        assert original.token_ids == updated.token_ids
        assert original.output_text == updated.output_text
        assert original.bound_plan_sha256 == updated.bound_plan_sha256
        assert original.initial_sampling_rng_sha256 == updated.initial_sampling_rng_sha256


def test_schema_rejects_missing_reordered_foreign_and_unlabelled_counterfactuals(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    paired = generate(completed, _test_spec(tmp_path / "test.parquet", _test_rows()[-2:]), plan())
    payload = paired.payload
    for change in (
        {"rows": payload.rows[:1]},
        {"rows": payload.rows[::-1]},
        {"plan_sha256": "0" * 64},
        {"semantics": "unknown"},
        {"intervened_target_nll": "same_as_control"},
        {"baseline": payload.baseline.model_copy(update={"sha256": "0" * 64})},
    ):
        with pytest.raises(ValidationError):
            ScientificInterventionPayloadV1.model_validate(
                payload.model_copy(update=change).model_dump(mode="python")
            )
    for row_change in (
        {"example_sha256": "0" * 64},
        {"prompt_sha256": "0" * 64},
        {"sampling_seed": 42},
        {"initial_sampling_rng_sha256": "0" * 64},
        {"bound_plan_sha256": "0" * 64},
        {"output_text": "unlabelled change"},
        {"token_ids": (payload.baseline.payload.vocabulary_size,)},
    ):
        invalid_row = payload.rows[0].model_copy(update=row_change)
        with pytest.raises(ValidationError):
            ScientificInterventionPayloadV1.model_validate(
                payload.model_copy(
                    update={
                        "rows": (invalid_row, *payload.rows[1:]),
                    }
                ).model_dump(mode="python")
            )


def test_dimension_error_is_rejected_before_loading_the_model(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    test = _test_spec(tmp_path / "test.parquet", _test_rows()[:1])
    with (
        patch.object(generation, "_composed", side_effect=AssertionError("must not load")) as load,
        pytest.raises(generation.IrisScientificGenerationError),
    ):
        generate(completed, test, plan({"kind": "zero_slot", "slot": 999}))
    load.assert_not_called()


def test_forged_execution_receipt_aborts_complete_capture_and_preserves_source_and_rng(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    test = _test_spec(tmp_path / "test.parquet", _test_rows()[-1:])
    original = IrisRequestOrchestrator.execute_intervened
    before_files = _files(completed[1])
    before_rng = capture_runtime_state(torch.device("cpu"))

    def forged(owner: IrisRequestOrchestrator, *args: Any, **kwargs: Any) -> Any:
        result = original(owner, *args, **kwargs)
        return replace(result, receipt=replace(result.receipt, graph_sha256="0" * 64))

    with (
        patch.object(IrisRequestOrchestrator, "execute_intervened", new=forged),
        pytest.raises(generation.IrisScientificGenerationError),
    ):
        generate(completed, test, plan({"kind": "zero_workspace"}), sampling=True)
    assert _files(completed[1]) == before_files
    assert _same_runtime(before_rng, capture_runtime_state(torch.device("cpu")))


def test_publication_rejects_tampering_and_preserves_foreign_staging(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    paired = generate(
        completed,
        _test_spec(tmp_path / "test.parquet", _test_rows()[-1:]),
        plan({"kind": "zero_workspace"}),
    )
    with pytest.raises(generation.IrisScientificGenerationError):
        storage.write_scientific_intervention_v1(
            paired.model_copy(update={"sha256": "0" * 64}), root=tmp_path / "wrong"
        )
    with pytest.raises(generation.IrisScientificGenerationError):
        storage.write_scientific_intervention_v1(
            paired, root=tmp_path / "quota", maximum_storage_bytes=1
        )
    fixed = type("Fixed", (), {"hex": "f" * 32})()
    foreign = (
        tmp_path / "collision/scientific-intervention-v1" / f".iris-intervention-{fixed.hex}.tmp"
    )
    foreign.mkdir(parents=True)
    sentinel = foreign / "foreign.txt"
    sentinel.write_text("do not touch", encoding="utf-8")
    with (
        patch.object(storage, "uuid4", return_value=fixed),
        pytest.raises(generation.IrisScientificGenerationError),
    ):
        storage.write_scientific_intervention_v1(paired, root=tmp_path / "collision")
    assert sentinel.read_text(encoding="utf-8") == "do not touch"
    published = storage.write_scientific_intervention_v1(paired, root=tmp_path / "good")
    raw = json.loads((published / "intervention.json").read_bytes())
    raw["payload"]["rows"][0]["bound_plan_sha256"] = "0" * 64
    raw["sha256"] = domain_sha256(
        "iris/scientific-intervention/v1", canonical_json_bytes(raw["payload"])
    )
    (published / "intervention.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(generation.IrisScientificGenerationError):
        storage.read_scientific_intervention_v1(published)


def test_row_operator_seed_uses_only_declared_seed_kind_and_input_identity() -> None:
    base = plan({"kind": "gaussian_noise", "seed": 42, "sigma": 0.1})
    larger = plan({"kind": "gaussian_noise", "seed": 42, "sigma": 0.9})
    first = intervention_plan_for_row_v1(base, example_id="test-1", prompt_sha256="a" * 64)
    paired = intervention_plan_for_row_v1(larger, example_id="test-1", prompt_sha256="a" * 64)
    assert first.interventions[0].seed == paired.interventions[0].seed
    assert intervention_plan_sha256_v1(first) != intervention_plan_sha256_v1(paired)
    for identity in (("test-2", "a" * 64), ("test-1", "b" * 64)):
        other = intervention_plan_for_row_v1(
            base, example_id=identity[0], prompt_sha256=identity[1]
        )
        assert other.interventions[0].seed != first.interventions[0].seed
    assert base.interventions[0].seed == 42
