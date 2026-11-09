"""Five-family dataset membership and leakage checks without model execution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris.application.experiment_manifest import build_experiment_manifest_v1
from iris.artifacts.experiment_v1 import ExperimentExampleV1
from iris.artifacts.study_dataset_v1 import (
    IrisStudyDatasetV1,
    StudyDatasetConstructionV1,
    StudyDatasetPayloadV1,
    StudyDatasetSchemaV1,
    StudyExampleAssignmentV1,
    StudyTaskDefinitionV1,
)
from tests.unit.test_study_dataset import _construction, _dataset, _definitions


@pytest.mark.parametrize("purpose", ["operational_fixture", "study_candidate"])
def test_purpose_never_promotes_dataset_validation_to_scientific_readiness(purpose: str) -> None:
    raw = _dataset().model_dump(mode="python")
    raw["payload"]["purpose"] = purpose
    dataset = IrisStudyDatasetV1.model_validate(raw)

    assert dataset.schema_identity.validation_scope == (
        "input_membership_and_exact_duplicate_checks_only"
    )
    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(
            {**raw["payload"], "scientific_launch_status": "ready"}
        )
    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate({**raw["payload"], "purpose": "validated_benchmark"})


@pytest.mark.parametrize("mutation", ["reorder", "missing", "duplicate", "unknown"])
def test_assignments_must_match_exact_manifest_id_order(mutation: str) -> None:
    raw = _dataset().payload.model_dump(mode="python")
    assignments = list(raw["assignments"])
    if mutation == "reorder":
        assignments[0], assignments[1] = assignments[1], assignments[0]
    elif mutation == "missing":
        assignments.pop()
    elif mutation == "duplicate":
        assignments[1]["example_id"] = assignments[0]["example_id"]
    else:
        assignments[0]["example_id"] = "unknown-example"
    raw["assignments"] = tuple(assignments)

    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(raw)


@pytest.mark.parametrize("remove", ["family", "train", "validation", "test"])
def test_all_five_families_require_nonempty_train_validation_and_test(remove: str) -> None:
    dataset = _dataset()
    pairs = tuple(
        (example, assignment)
        for example, assignment in zip(
            dataset.payload.manifest.payload.examples, dataset.payload.assignments, strict=True
        )
        if not (
            assignment.task_family == "code_generation_and_verification"
            and (remove == "family" or example.split == remove)
        )
    )
    manifest = build_experiment_manifest_v1(
        dataset_id=dataset.payload.manifest.payload.dataset_id,
        examples=tuple(example for example, _ in pairs),
    )
    raw = dataset.payload.model_dump(mode="python")
    raw["manifest"] = manifest
    raw["assignments"] = tuple(assignment for _, assignment in pairs)

    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(raw)


def test_one_leakage_group_cannot_cross_splits_even_between_different_families() -> None:
    raw = _dataset().payload.model_dump(mode="python")
    # Index 0 is split_information/train, index 6 delayed_planning/test.
    raw["assignments"][6]["leakage_group_id"] = raw["assignments"][0]["leakage_group_id"]
    # Keep every source-item/group relation coherent so the group leak is isolated.
    raw["assignments"][21]["leakage_group_id"] = raw["assignments"][0]["leakage_group_id"]

    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(raw)


def test_same_source_item_cannot_be_relabelled_into_different_leakage_groups() -> None:
    raw = _dataset().payload.model_dump(mode="python")
    raw["assignments"][15]["leakage_group_id"] = "a-different-training-group"

    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(raw)


def test_distinct_source_items_can_share_a_group_within_the_same_split() -> None:
    raw = _dataset().payload.model_dump(mode="python")
    raw["assignments"][1]["leakage_group_id"] = raw["assignments"][0]["leakage_group_id"]
    raw["assignments"][16]["leakage_group_id"] = raw["assignments"][0]["leakage_group_id"]

    accepted = StudyDatasetPayloadV1.model_validate(raw)

    assert accepted.assignments[0].source_item_id != accepted.assignments[1].source_item_id
    assert accepted.assignments[0].leakage_group_id == accepted.assignments[1].leakage_group_id


@pytest.mark.parametrize("duplicate_index", [5, 10])
def test_exact_duplicate_prompt_across_splits_is_rejected_even_if_targets_differ(
    duplicate_index: int,
) -> None:
    dataset = _dataset()
    examples = list(dataset.payload.manifest.payload.examples)
    examples[duplicate_index] = ExperimentExampleV1(
        example_id=examples[duplicate_index].example_id,
        split=examples[duplicate_index].split,
        prompt=examples[0].prompt,
        target="A different target does not prevent prompt leakage.",
    )
    raw = dataset.payload.model_dump(mode="python")
    raw["manifest"] = build_experiment_manifest_v1(
        dataset_id=dataset.payload.manifest.payload.dataset_id, examples=tuple(examples)
    )

    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(raw)


def test_exact_duplicate_prompts_within_split_are_retained_without_silent_deduplication() -> None:
    dataset = _dataset()
    examples = list(dataset.payload.manifest.payload.examples)
    examples[15] = ExperimentExampleV1(
        example_id=examples[15].example_id,
        split=examples[15].split,
        prompt=examples[0].prompt,
        target=examples[0].target,
    )
    raw = dataset.payload.model_dump(mode="python")
    raw["manifest"] = build_experiment_manifest_v1(
        dataset_id=dataset.payload.manifest.payload.dataset_id, examples=tuple(examples)
    )

    accepted = StudyDatasetPayloadV1.model_validate(raw)

    assert len(accepted.manifest.payload.examples) == 30
    assert accepted.manifest.payload.examples[15].prompt == examples[0].prompt
    assert accepted.manifest.payload.examples[15].example_id != examples[0].example_id


def test_one_example_per_family_split_is_the_minimal_valid_membership() -> None:
    dataset = _dataset()
    raw = dataset.payload.model_dump(mode="python")
    raw["manifest"] = build_experiment_manifest_v1(
        dataset_id=dataset.payload.manifest.payload.dataset_id,
        examples=dataset.payload.manifest.payload.examples[:15],
    )
    raw["assignments"] = raw["assignments"][:15]

    assert len(StudyDatasetPayloadV1.model_validate(raw).assignments) == 15


def test_duplicate_check_is_exact_utf8_not_a_claim_of_semantic_or_normalized_deduplication() -> (
    None
):
    dataset = _dataset()
    examples = list(dataset.payload.manifest.payload.examples)
    for index, prompt in ((0, "caf\u00e9"), (5, "cafe\u0301"), (10, " caf\u00e9 ")):
        examples[index] = ExperimentExampleV1(
            example_id=examples[index].example_id,
            split=examples[index].split,
            prompt=prompt,
            target=examples[index].target,
        )
    raw = dataset.payload.model_dump(mode="python")
    raw["manifest"] = build_experiment_manifest_v1(
        dataset_id=dataset.payload.manifest.payload.dataset_id, examples=tuple(examples)
    )

    accepted = StudyDatasetPayloadV1.model_validate(raw)

    assert tuple(accepted.manifest.payload.examples[index].prompt for index in (0, 5, 10)) == (
        "caf\u00e9",
        "cafe\u0301",
        " caf\u00e9 ",
    )


@pytest.mark.parametrize("mutation", ["missing", "reorder", "duplicate"])
def test_task_definitions_require_exact_source_order_and_all_five_families(mutation: str) -> None:
    raw = _dataset().payload.model_dump(mode="python")
    definitions = list(raw["task_definitions"])
    if mutation == "missing":
        definitions.pop()
    elif mutation == "reorder":
        definitions[0], definitions[1] = definitions[1], definitions[0]
    else:
        definitions[1] = definitions[0]
    raw["task_definitions"] = tuple(definitions)

    with pytest.raises(ValidationError):
        StudyDatasetPayloadV1.model_validate(raw)


@pytest.mark.parametrize("family_index", range(5))
def test_task_measurement_scope_cannot_claim_another_family_mechanism(family_index: int) -> None:
    raw = _definitions()[family_index].model_dump(mode="python")
    raw["measurement_scope"] = _definitions()[(family_index + 1) % 5].measurement_scope

    with pytest.raises(ValidationError):
        StudyTaskDefinitionV1.model_validate(raw)


@pytest.mark.parametrize("field", ["description", "target_semantics"])
@pytest.mark.parametrize(
    "value",
    ["", " \t\r\n", "bad\0text", "\ud800", 4, True, "я" * 8193],
    ids=["empty", "blank", "nul", "surrogate", "integer", "boolean", "utf8-limit"],
)
def test_task_descriptions_require_bounded_exact_unicode_text(field: str, value: object) -> None:
    raw = _definitions()[0].model_dump(mode="python")
    raw[field] = value

    with pytest.raises(ValidationError):
        StudyTaskDefinitionV1.model_validate(raw)


@pytest.mark.parametrize("field", ["example_id", "source_item_id", "leakage_group_id"])
@pytest.mark.parametrize("value", ["", "white space", "é", "-leading", "a/b", "a" * 129, True])
def test_assignment_ids_require_portable_exact_strings(field: str, value: object) -> None:
    raw = _dataset().payload.assignments[0].model_dump(mode="python")
    raw[field] = value

    with pytest.raises(ValidationError):
        StudyExampleAssignmentV1.model_validate(raw)


@pytest.mark.parametrize(
    ("method", "seed"),
    [
        ("manual_curation", 0),
        ("deterministic_generator", None),
        ("deterministic_generator", True),
        ("deterministic_generator", 1.0),
        ("deterministic_generator", "1"),
        ("deterministic_generator", -1),
        ("deterministic_generator", 2**63),
        ("unspecified", None),
    ],
)
def test_construction_requires_consistent_exact_method_and_seed(method: str, seed: object) -> None:
    raw = _construction().model_dump(mode="python")
    raw.update(method=method, generation_seed=seed)

    with pytest.raises(ValidationError):
        StudyDatasetConstructionV1.model_validate(raw)


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_generated_construction_accepts_explicit_seed_boundaries(seed: int) -> None:
    raw = _construction().model_dump(mode="python")
    raw.update(method="deterministic_generator", generation_seed=seed)

    assert StudyDatasetConstructionV1.model_validate(raw).generation_seed == seed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", ""),
        ("source_revision", "revision/with/slash"),
        ("source_sha256", "A" * 64),
        ("source_sha256", "a" * 63),
        ("source_sha256", 3),
        ("split_policy", "random_examples"),
        ("split_policy_description", ""),
        ("split_policy_description", "bad\0text"),
        pytest.param("split_policy_description", "я" * 8193, id="description-utf8-limit"),
    ],
)
def test_construction_provenance_and_split_policy_are_explicit(field: str, value: object) -> None:
    raw = _construction().model_dump(mode="python")
    raw[field] = value

    with pytest.raises(ValidationError):
        StudyDatasetConstructionV1.model_validate(raw)


def test_manual_construction_requires_explicit_null_seed_not_omission() -> None:
    raw = _construction().model_dump(mode="python")
    del raw["generation_seed"]

    with pytest.raises(ValidationError):
        StudyDatasetConstructionV1.model_validate(raw)


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_schema_version_is_exact_integer_one(version: object) -> None:
    raw = _dataset().schema_identity.model_dump(mode="python")
    raw["version"] = version

    with pytest.raises(ValidationError):
        StudyDatasetSchemaV1.model_validate(raw)


def test_models_are_frozen_forbid_extra_fields_and_revalidate_bypassed_mutations() -> None:
    dataset = _dataset()
    with pytest.raises(ValidationError):
        dataset.payload.purpose = "study_candidate"
    with pytest.raises(ValidationError):
        IrisStudyDatasetV1.model_validate({**dataset.model_dump(mode="python"), "unknown": True})
    with pytest.raises(ValidationError):
        StudyExampleAssignmentV1.model_validate(
            {**dataset.payload.assignments[0].model_dump(mode="python"), "unknown": True}
        )

    object.__setattr__(dataset.payload.assignments[0], "task_family", "unknown")
    with pytest.raises(ValidationError):
        IrisStudyDatasetV1.model_validate(dataset)
