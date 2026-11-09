"""Task-labelled dataset integrity, ownership, and training projection contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from iris.application.errors import IrisStudyDatasetError
from iris.application.experiment_manifest import build_experiment_manifest_v1
from iris.application.study_dataset import (
    build_study_dataset_v1,
    project_study_experiment_manifest_v1,
    select_study_examples_v1,
    validate_study_dataset_v1,
)
from iris.application.study_declaration import build_four_arm_protocol_declaration_v1
from iris.artifacts.experiment_v1 import (
    ExperimentExampleV1,
    ExperimentSplitV1,
    IrisExperimentManifestV1,
)
from iris.artifacts.study_dataset_v1 import (
    IrisStudyDatasetV1,
    StudyDatasetConstructionV1,
    StudyExampleAssignmentV1,
    StudyTaskDefinitionV1,
)
from iris.artifacts.study_v1 import StudyTaskFamilyV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256

_FAMILIES: tuple[StudyTaskFamilyV1, ...] = (
    "split_information",
    "delayed_planning",
    "code_generation_and_verification",
    "hidden_goal_retention",
    "module_conflict",
)
_SCOPES = (
    "cross_fragment_information",
    "delayed_instruction_planning",
    "functional_code_generation",
    "within_prompt_goal_retention",
    "within_prompt_conflict_resolution_proxy",
)
_SPLITS: tuple[ExperimentSplitV1, ...] = ("train", "test", "validation")


def _construction() -> StudyDatasetConstructionV1:
    return StudyDatasetConstructionV1(
        method="manual_curation",
        source_id="iris-unit-test-fixture",
        source_revision="v1",
        source_sha256="a" * 64,
        generation_seed=None,
        split_policy="preassigned_group_disjoint_v1",
        split_policy_description="Operational test data grouped by family and split.",
    )


def _definitions() -> tuple[StudyTaskDefinitionV1, ...]:
    return tuple(
        StudyTaskDefinitionV1.model_validate(
            {
                "task_family": family,
                "measurement_scope": scope,
                "description": f"Operational schema fixture for {family}; not a benchmark.",
                "target_semantics": "Exact fixture text, not a scientifically validated answer.",
            }
        )
        for family, scope in zip(_FAMILIES, _SCOPES, strict=True)
    )


def _manifest_and_assignments() -> tuple[
    IrisExperimentManifestV1, tuple[StudyExampleAssignmentV1, ...]
]:
    examples: list[ExperimentExampleV1] = []
    assignments: list[StudyExampleAssignmentV1] = []
    # Deliberately interleave families and splits, retaining two examples per cell.
    for repetition in (0, 1):
        for split in _SPLITS:
            for family in _FAMILIES:
                example_id = f"{family}:{split}:{repetition}"
                examples.append(
                    ExperimentExampleV1(
                        example_id=example_id,
                        split=split,
                        prompt=f"  Fixture {example_id}: Привет 👁️\r\n",
                        target=f"\tExact target {repetition}\n",
                    )
                )
                assignments.append(
                    StudyExampleAssignmentV1(
                        example_id=example_id,
                        task_family=family,
                        source_item_id=f"source:{family}:{split}",
                        leakage_group_id=f"group:{family}:{split}",
                    )
                )
    return (
        build_experiment_manifest_v1(dataset_id="iris-study-fixture-v1", examples=tuple(examples)),
        tuple(assignments),
    )


def _dataset() -> IrisStudyDatasetV1:
    """Reusable operational-only artifact for schema and filesystem tests."""
    manifest, assignments = _manifest_and_assignments()
    return build_study_dataset_v1(
        manifest=manifest,
        construction=_construction(),
        task_definitions=_definitions(),
        assignments=assignments,
        purpose="operational_fixture",
    )


def test_builder_binds_protocol_and_preserves_exact_manifest_and_assignment_order() -> None:
    dataset = _dataset()
    manifest, assignments = _manifest_and_assignments()

    assert dataset.payload.manifest == manifest
    assert dataset.payload.assignments == assignments
    assert dataset.payload.protocol_declaration_payload_sha256 == (
        build_four_arm_protocol_declaration_v1().payload_sha256
    )
    assert dataset.payload.manifest.payload.examples[0].prompt.endswith("Привет 👁️\r\n")
    assert tuple(item.example_id for item in dataset.payload.assignments) == tuple(
        item.example_id for item in manifest.payload.examples
    )
    assert tuple(item.task_family for item in dataset.payload.task_definitions) == _FAMILIES
    assert dataset.payload_sha256 == domain_sha256(
        "iris/study-dataset/v1/payload",
        canonical_json_bytes(dataset.payload.model_dump(mode="json")),
    )


def test_builder_is_deterministic_and_pairwise_reordering_changes_identity() -> None:
    first = _dataset()
    manifest, assignments = _manifest_and_assignments()
    reordered = build_study_dataset_v1(
        manifest=build_experiment_manifest_v1(
            dataset_id=manifest.payload.dataset_id,
            examples=tuple(reversed(manifest.payload.examples)),
        ),
        construction=_construction(),
        task_definitions=_definitions(),
        assignments=tuple(reversed(assignments)),
        purpose="operational_fixture",
    )

    assert first == _dataset()
    assert reordered.payload_sha256 != first.payload_sha256
    assert reordered.payload.manifest.payload.examples == tuple(
        reversed(first.payload.manifest.payload.examples)
    )


def test_builder_takes_deep_ownership_of_all_supplied_models() -> None:
    manifest, assignments = _manifest_and_assignments()
    construction = _construction()
    definitions = _definitions()
    dataset = build_study_dataset_v1(
        manifest=manifest,
        construction=construction,
        task_definitions=definitions,
        assignments=assignments,
        purpose="operational_fixture",
    )
    before = canonical_json_bytes(dataset.model_dump(mode="json"))

    object.__setattr__(manifest.payload.examples[0], "prompt", "external mutation")
    object.__setattr__(assignments[0], "source_item_id", "external-mutation")
    object.__setattr__(construction, "source_id", "external-mutation")
    object.__setattr__(definitions[0], "description", "external mutation")

    assert canonical_json_bytes(dataset.model_dump(mode="json")) == before
    assert validate_study_dataset_v1(dataset) == dataset


def test_validator_returns_deeply_owned_dataset() -> None:
    dataset = _dataset()
    validated = validate_study_dataset_v1(dataset)

    assert validated == dataset
    assert validated is not dataset
    assert validated.payload is not dataset.payload
    assert validated.payload.manifest is not dataset.payload.manifest
    assert (
        validated.payload.manifest.payload.examples[0]
        is not (dataset.payload.manifest.payload.examples[0])
    )
    assert validated.payload.assignments[0] is not dataset.payload.assignments[0]
    assert validated.payload.construction is not dataset.payload.construction
    assert validated.payload.task_definitions[0] is not dataset.payload.task_definitions[0]


def test_projection_is_the_identical_existing_training_manifest_with_owned_examples() -> None:
    dataset = _dataset()
    projected = project_study_experiment_manifest_v1(dataset)

    assert type(projected) is IrisExperimentManifestV1
    assert projected == dataset.payload.manifest
    assert projected is not dataset.payload.manifest
    assert projected.payload.examples[0] is not dataset.payload.manifest.payload.examples[0]
    object.__setattr__(projected.payload.examples[0], "prompt", "projection mutation")
    assert validate_study_dataset_v1(dataset) == dataset


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("split", _SPLITS)
def test_selection_preserves_cell_order_exact_text_and_owned_assignments(
    family: StudyTaskFamilyV1, split: ExperimentSplitV1
) -> None:
    dataset = _dataset()
    selected = select_study_examples_v1(dataset, task_family=family, split=split)
    expected = tuple(
        (example, assignment)
        for example, assignment in zip(
            dataset.payload.manifest.payload.examples, dataset.payload.assignments, strict=True
        )
        if example.split == split and assignment.task_family == family
    )

    assert type(selected) is tuple
    assert tuple((item.example, item.assignment) for item in selected) == expected
    assert len(selected) == 2
    assert selected[0].example.example_id.endswith(":0")
    assert selected[1].example.example_id.endswith(":1")
    assert selected[0].example is not expected[0][0]
    assert selected[0].assignment is not expected[0][1]
    object.__setattr__(selected[0].example, "prompt", "selection mutation")
    object.__setattr__(selected[0].assignment, "source_item_id", "selection-mutation")
    assert validate_study_dataset_v1(dataset) == dataset


@pytest.mark.parametrize("field", ["outer_hash", "nested_hash", "nested_text", "protocol"])
def test_every_consumer_revalidates_integrity_including_rehashed_outer_payload(field: str) -> None:
    dataset = _dataset()
    raw = dataset.model_dump(mode="python")
    if field == "outer_hash":
        raw["payload_sha256"] = "0" * 64
    else:
        if field == "nested_hash":
            raw["payload"]["manifest"]["payload_sha256"] = "0" * 64
        elif field == "nested_text":
            raw["payload"]["manifest"]["payload"]["examples"][0]["prompt"] = "changed plaintext"
        else:
            raw["payload"]["protocol_declaration_payload_sha256"] = "0" * 64
        raw["payload_sha256"] = domain_sha256(
            "iris/study-dataset/v1/payload", canonical_json_bytes(raw["payload"])
        )
    forged = IrisStudyDatasetV1.model_validate(raw)

    with pytest.raises(IrisStudyDatasetError):
        validate_study_dataset_v1(forged)
    with pytest.raises(IrisStudyDatasetError):
        project_study_experiment_manifest_v1(forged)
    with pytest.raises(IrisStudyDatasetError):
        select_study_examples_v1(forged, task_family="split_information", split="test")


def test_builder_rejects_nested_manifest_with_forged_digest() -> None:
    manifest, assignments = _manifest_and_assignments()
    with pytest.raises(IrisStudyDatasetError):
        build_study_dataset_v1(
            manifest=manifest.model_copy(update={"payload_sha256": "0" * 64}),
            construction=_construction(),
            task_definitions=_definitions(),
            assignments=assignments,
            purpose="operational_fixture",
        )


@pytest.mark.parametrize("field", ["assignments", "task_definitions"])
def test_builder_rejects_list_instead_of_exact_tuple(field: str) -> None:
    manifest, assignments = _manifest_and_assignments()
    definitions = _definitions()
    with pytest.raises(IrisStudyDatasetError):
        build_study_dataset_v1(
            manifest=manifest,
            construction=_construction(),
            task_definitions=(
                cast(tuple[StudyTaskDefinitionV1, ...], list(definitions))
                if field == "task_definitions"
                else definitions
            ),
            assignments=(
                cast(tuple[StudyExampleAssignmentV1, ...], list(assignments))
                if field == "assignments"
                else assignments
            ),
            purpose="operational_fixture",
        )


def test_builder_rejects_assignment_subclasses() -> None:
    class AssignmentSubclass(StudyExampleAssignmentV1):
        pass

    manifest, assignments = _manifest_and_assignments()
    child = AssignmentSubclass.model_validate(assignments[0].model_dump(mode="python"))
    with pytest.raises(IrisStudyDatasetError):
        build_study_dataset_v1(
            manifest=manifest,
            construction=_construction(),
            task_definitions=_definitions(),
            assignments=(child, *assignments[1:]),
            purpose="operational_fixture",
        )


def test_builder_rejects_construction_and_definition_subclasses() -> None:
    class ConstructionSubclass(StudyDatasetConstructionV1):
        pass

    class DefinitionSubclass(StudyTaskDefinitionV1):
        pass

    manifest, assignments = _manifest_and_assignments()
    construction = ConstructionSubclass.model_validate(_construction().model_dump(mode="python"))
    definitions = _definitions()
    definition = DefinitionSubclass.model_validate(definitions[0].model_dump(mode="python"))
    for candidate_construction, candidate_definitions in (
        (construction, definitions),
        (_construction(), (definition, *definitions[1:])),
    ):
        with pytest.raises(IrisStudyDatasetError):
            build_study_dataset_v1(
                manifest=manifest,
                construction=candidate_construction,
                task_definitions=candidate_definitions,
                assignments=assignments,
                purpose="operational_fixture",
            )


def test_selected_pair_is_frozen() -> None:
    selected = select_study_examples_v1(_dataset(), task_family="module_conflict", split="test")

    with pytest.raises(FrozenInstanceError):
        selected[0].example = selected[1].example  # type: ignore[misc]


def test_invalid_private_text_is_not_exposed_in_public_service_error() -> None:
    dataset = _dataset()
    private_text = "SENSITIVE PROMPT CONTENT"
    object.__setattr__(dataset.payload.manifest.payload.examples[0], "prompt", private_text + "\0")

    with pytest.raises(IrisStudyDatasetError) as caught:
        validate_study_dataset_v1(dataset)

    assert private_text not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("invalid", [None, {}, 1, True])
def test_consumers_reject_non_dataset_values(invalid: object) -> None:
    with pytest.raises(IrisStudyDatasetError):
        validate_study_dataset_v1(cast(IrisStudyDatasetV1, invalid))
    with pytest.raises(IrisStudyDatasetError):
        project_study_experiment_manifest_v1(cast(IrisStudyDatasetV1, invalid))
    with pytest.raises(IrisStudyDatasetError):
        select_study_examples_v1(
            cast(IrisStudyDatasetV1, invalid), task_family="module_conflict", split="train"
        )


@pytest.mark.parametrize("family", ["unknown", "Module_conflict", "", True, None])
def test_selection_rejects_invalid_task_family(family: object) -> None:
    with pytest.raises(IrisStudyDatasetError):
        select_study_examples_v1(
            _dataset(), task_family=cast(StudyTaskFamilyV1, family), split="train"
        )


@pytest.mark.parametrize("split", ["eval", "Train", "", True, None])
def test_selection_rejects_invalid_split(split: object) -> None:
    with pytest.raises(IrisStudyDatasetError):
        select_study_examples_v1(
            _dataset(), task_family="split_information", split=cast(ExperimentSplitV1, split)
        )
