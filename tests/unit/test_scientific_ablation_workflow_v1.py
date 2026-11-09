"""DATA plan/phase domains only; no source, factory, model, or native proof."""

from pathlib import Path

import pytest

from iris.application.scientific_ablation_workflow_io import (
    build_scientific_ablation_workflow_plan_v1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    WORKFLOW_ATTEMPT_DOMAIN,
    IrisScientificAblationWorkflowAttemptV1,
    ScientificAblationWorkflowAttemptPayloadV1,
    ScientificAblationWorkflowIntentV1,
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_bytes, domain_sha256
from tests.unit._workflow_data_setup import intent, plan


def test_complete_closed_data_intent_is_stable_and_does_not_touch_source(tmp_path: Path) -> None:
    original = intent(tmp_path)
    first = build_scientific_ablation_workflow_plan_v1(original)
    assert first == build_scientific_ablation_workflow_plan_v1(original)
    assert first.payload is not original and first.payload.resources is not original.resources
    assert first.payload.scientific_acceptance == "not_assessed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "problem", ("extra", "missing", "private", "boolean_quota", "mutable_controls", "subclass")
)
def test_unchecked_metadata_is_not_sanitized_into_a_valid_plan(
    tmp_path: Path, problem: str
) -> None:
    value = intent(tmp_path)
    if problem == "extra":
        value.__dict__["native_completed"] = True
    elif problem == "missing":
        del value.__dict__["resources"]
    elif problem == "private":
        object.__setattr__(value, "__pydantic_private__", {"native": True})
    elif problem == "boolean_quota":
        value.resources.__dict__["maximum_storage_bytes"] = True
    elif problem == "mutable_controls":
        value.__dict__["controls"] = []
    else:

        class Other(ScientificAblationWorkflowIntentV1):
            pass

        value = Other.model_construct(**value.__dict__)
    with pytest.raises(ValueError):
        build_scientific_ablation_workflow_plan_v1(value)


@pytest.mark.parametrize("controls", (0, 1, 12, 64))
def test_every_declared_control_gets_a_full_capture_and_score_phase(
    tmp_path: Path, controls: int
) -> None:
    policies = tuple(
        WorkspaceInterventionPlanV1(
            interventions=(
                WorkspaceInterventionSpecV1(kind="scale_module_bid", module=0, factor=float(i + 1)),
            )
        )
        for i in range(controls)
    )
    value = build_scientific_ablation_workflow_plan_v1(intent(tmp_path, policies))
    phases = scientific_ablation_workflow_phases_v1(value)
    assert len(phases) == 6 + 2 * controls
    assert [p.index for p in phases] == list(range(len(phases)))
    assert len({p.name for p in phases}) == len(phases)
    assert [p.kind for p in phases[-4:]] == ["diagnostics", "workspace", "message", "publish"]
    assert [(p.kind, p.stream_index) for p in phases[2:-4]] == [
        (kind, i) for i in range(controls) for kind in ("capture", "score")
    ]


def test_duplicate_control_is_rejected_not_silently_deduplicated(tmp_path: Path) -> None:
    control = WorkspaceInterventionPlanV1(interventions=())
    with pytest.raises(ValueError):
        intent(tmp_path, (control, control))


@pytest.mark.parametrize(
    "name", ("../diagnostics", "diagnostics\n", "ordinary-score", "control-00-capture")
)
def test_phase_label_cannot_select_another_kind_or_path(name: str) -> None:
    with pytest.raises(ValueError):
        ScientificAblationWorkflowPhaseV1(index=2, name=name, kind="diagnostics", stream_index=None)


@pytest.mark.parametrize(
    "kind", ("capture", "score", "diagnostics", "workspace", "message", "publish")
)
def test_attempt_recovery_policy_is_not_a_caller_chosen_native_permission(
    tmp_path: Path, kind: str
) -> None:
    phase = next(
        p for p in scientific_ablation_workflow_phases_v1(plan(tmp_path)) if p.kind == kind
    )
    correct = (
        "acknowledged_kernel_prefix_only"
        if kind in ("capture", "score")
        else "data_publication_only"
        if kind == "publish"
        else "never_auto_remeasure"
    )
    wrong = (
        "data_publication_only" if correct != "data_publication_only" else "never_auto_remeasure"
    )
    payload = ScientificAblationWorkflowAttemptPayloadV1.model_validate(
        dict(
            plan_sha256="0" * 64,
            phase=phase,
            predecessor_sha256=None if phase.index == 0 else "1" * 64,
            recovery=wrong,
        )
    )
    with pytest.raises(ValueError):
        IrisScientificAblationWorkflowAttemptV1(
            sha256=domain_sha256(
                WORKFLOW_ATTEMPT_DOMAIN, canonical_json_bytes(payload.model_dump(mode="json"))
            ),
            payload=payload,
        )
