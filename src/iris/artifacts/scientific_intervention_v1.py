"""Private paired held-out outputs; changed predictions are never baseline captures."""

from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.artifacts.scientific_generation_v1 import IrisScientificGenerationV1
from iris.artifacts.scientific_inference_v1 import IrisScientificInferenceManifestV1
from iris.interventions.policy import (
    WorkspaceInterventionPlanV1,
    intervention_plan_sha256_v1,
    validate_intervention_plan_v1,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


def intervention_plan_for_row_v1(
    plan: WorkspaceInterventionPlanV1, *, example_id: str, prompt_sha256: str
) -> WorkspaceInterventionPlanV1:
    """Pair operator RNG across amplitudes/order using inputs, never answer labels."""
    plan = validate_intervention_plan_v1(plan)
    if (
        type(example_id) is not str
        or fullmatch(r"test-[0-9]+", example_id) is None
        or type(prompt_sha256) is not str
        or fullmatch(r"[0-9a-f]{64}", prompt_sha256) is None
    ):
        raise ValueError("Exact held-out input identities required for intervention RNG")
    operations = []
    for spec in plan.interventions:
        if spec.seed is not None:
            digest = domain_sha256(
                "iris/scientific-intervention-row-seed/v1",
                canonical_json_bytes([spec.kind, spec.seed, example_id, prompt_sha256]),
            )
            spec = spec.model_copy(update={"seed": int(digest[:16], 16) & (2**63 - 1)})
        operations.append(spec)
    return validate_intervention_plan_v1(
        WorkspaceInterventionPlanV1(interventions=tuple(operations))
    )


class ScientificInterventionGeneratedRowV1(_Strict):
    example_id: str = Field(pattern=r"^test-[0-9]+$")
    example_sha256: Digest
    prompt_sha256: Digest
    bound_plan_sha256: Digest
    sampling_seed: int | None = Field(ge=0, le=2**63 - 1)
    initial_sampling_rng_sha256: Digest | None
    output_text: str = Field(max_length=1048576, repr=False)
    token_ids: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1, max_length=65536)
    stop_reason: Literal["stop_token", "max_new_tokens"]
    matched_stop_token_id: int | None = Field(ge=0)


class ScientificInterventionPayloadV1(_Strict):
    kind: Literal["iris-scientific-intervention-v1"] = "iris-scientific-intervention-v1"
    arm: Literal["D"] = "D"
    data_scope: Literal["candidate_not_launch_approval"] = "candidate_not_launch_approval"
    semantics: Literal["iris-workspace-intervention-semantics-v1"] = (
        "iris-workspace-intervention-semantics-v1"
    )
    operator_rng_policy: Literal["iris-scientific-intervention-row-seed-v1"] = (
        "iris-scientific-intervention-row-seed-v1"
    )
    intervened_target_nll: Literal["not_measured"] = "not_measured"
    baseline: IrisScientificGenerationV1 = Field(repr=False)
    inference_manifest: IrisScientificInferenceManifestV1 = Field(repr=False)
    plan: WorkspaceInterventionPlanV1 = Field(repr=False)
    plan_sha256: Digest
    request_graph_sha256: Digest
    request_sidecar_sha256: Digest
    num_modules: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    state_dim: int = Field(ge=1)
    cycles: int = Field(ge=1)
    winners_per_cycle: int = Field(ge=1)
    rows: tuple[ScientificInterventionGeneratedRowV1, ...] = Field(
        min_length=1, max_length=5000, repr=False
    )

    @model_validator(mode="after")
    def check_pairing(self) -> Self:
        baseline = self.baseline.payload
        if self.baseline.sha256 != domain_sha256(
            "iris/scientific-generation/v1", canonical_json_bytes(baseline.model_dump(mode="json"))
        ):
            raise ValueError("Baseline capture identity differs")
        inference = self.inference_manifest.payload
        model = inference.model
        if (
            self.inference_manifest.sha256 != baseline.inference_sha256
            or self.inference_manifest.sha256
            != domain_sha256(
                "iris/scientific-inference/v1/payload",
                canonical_json_bytes(inference.model_dump(mode="json")),
            )
            or inference.source_checkpoint_sha256 != baseline.best_checkpoint_sha256
            or inference.source_teacher_configuration_sha256
            != baseline.teacher_configuration_sha256
            or inference.backbone_runtime_sha256 != baseline.backbone_runtime_sha256
            or inference.completed_steps != baseline.selected_step
            or inference.purpose != baseline.purpose
            or model.runtime.vocabulary_size != baseline.vocabulary_size
            or model.runtime.max_target_tokens != baseline.max_new_tokens
            or self.request_sidecar_sha256 != model.sha256
            or self.request_graph_sha256
            != domain_sha256(
                "iris/intervened-request-graph/v1",
                canonical_json_bytes(
                    {
                        "cognitive": model.cognitive.model_dump(mode="json"),
                        "workspace": model.workspace.model_dump(mode="json"),
                    }
                ),
            )
            or (
                self.num_modules,
                self.num_slots,
                self.state_dim,
                self.cycles,
                self.winners_per_cycle,
            )
            != (
                model.cognitive.num_modules,
                model.workspace.num_slots,
                model.cognitive.state_dim,
                model.workspace.cycles,
                model.workspace.winners_per_cycle,
            )
        ):
            raise ValueError("Paired graph/export identities differ from the selected source")
        self.plan.validate_dimensions(
            num_slots=self.num_slots,
            num_modules=self.num_modules,
            state_dim=self.state_dim,
            cycles=self.cycles,
            winners_per_cycle=self.winners_per_cycle,
        )
        if self.plan_sha256 != intervention_plan_sha256_v1(self.plan):
            raise ValueError("Intervention plan identity differs")
        if len(self.rows) != len(baseline.rows):
            raise ValueError("Both branches must contain every declared held-out row")
        for row, control in zip(self.rows, baseline.rows, strict=True):
            if (
                row.example_id != control.example_id
                or row.example_sha256 != control.example_sha256
                or row.prompt_sha256 != control.prompt_sha256
                or row.sampling_seed != control.sampling_seed
                or (row.initial_sampling_rng_sha256 is None) != (row.sampling_seed is None)
                or row.bound_plan_sha256
                != intervention_plan_sha256_v1(
                    intervention_plan_for_row_v1(
                        self.plan, example_id=row.example_id, prompt_sha256=row.prompt_sha256
                    )
                )
            ):
                raise ValueError("Intervened row identity/order/RNG differs from its control")
            if len(row.token_ids) > baseline.max_new_tokens or any(
                token >= baseline.vocabulary_size for token in row.token_ids
            ):
                raise ValueError("Intervened token output exceeds declared model limits")
            if any(token in baseline.eos_token_ids for token in row.token_ids[:-1]):
                raise ValueError("Intervened generation continued after EOS")
            if row.stop_reason == "stop_token":
                if (
                    row.matched_stop_token_id != row.token_ids[-1]
                    or row.token_ids[-1] not in baseline.eos_token_ids
                ):
                    raise ValueError("Intervened EOS metadata differs from final token")
            elif (
                row.matched_stop_token_id is not None
                or len(row.token_ids) != baseline.max_new_tokens
                or row.token_ids[-1] in baseline.eos_token_ids
            ):
                raise ValueError("Intervened length termination is inconsistent")
            row.output_text.encode("utf-8", errors="strict")
            if not self.plan.interventions and (
                row.output_text != control.output_text
                or row.token_ids != control.token_ids
                or row.stop_reason != control.stop_reason
                or row.matched_stop_token_id != control.matched_stop_token_id
            ):
                raise ValueError("Empty intervention plan must reproduce its paired control")
        return self


class IrisScientificInterventionV1(_Strict):
    sha256: Digest
    payload: ScientificInterventionPayloadV1 = Field(repr=False)
