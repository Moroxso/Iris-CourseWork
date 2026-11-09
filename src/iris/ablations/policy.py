"""Explicit source-section17 cognitive ablation intent, not a scientific receipt."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris.utils.hashing import canonical_json_bytes, domain_sha256


class CognitiveAblationPolicyV1(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    kind: Literal["iris-cognitive-ablation-policy-v1"] = "iris-cognitive-ablation-policy-v1"
    routing: Literal["learned", "random", "uniform"] = "learned"
    random_routing: Literal["prompt-keyed-soft-normal-logits-v1"] = (
        "prompt-keyed-soft-normal-logits-v1"
    )
    routing_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    broadcast: Literal["enabled", "disabled"] = "enabled"
    goal: Literal["enabled", "disabled"] = "enabled"
    maximum_routing_scalars: int = Field(default=65536, ge=1, le=4_000_000)

    @model_validator(mode="after")
    def exact_seed(self) -> Self:
        if (self.routing == "random") != (self.routing_seed is not None):
            raise ValueError("Only random routing requires an explicit routing seed")
        return self


def validate_cognitive_ablation_policy_v1(value: object) -> CognitiveAblationPolicyV1:
    if type(value) is not CognitiveAblationPolicyV1:
        raise TypeError("Exact cognitive ablation policy required")
    return CognitiveAblationPolicyV1.model_validate(value)


def cognitive_ablation_policy_sha256_v1(value: CognitiveAblationPolicyV1) -> str:
    policy = validate_cognitive_ablation_policy_v1(value)
    return domain_sha256(
        "iris/cognitive-ablation-policy/v1", canonical_json_bytes(policy.model_dump(mode="json"))
    )
