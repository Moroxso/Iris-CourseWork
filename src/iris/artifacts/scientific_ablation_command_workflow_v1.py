"""Explicit command-workflow namespace DATA, never a producer/receiver ACK.

The legacy workflow layout remains closed. This definition selects a DIFFERENT
storage root bound to one original operator/execution/export/plan combination.
It neither authenticates a native result nor permits native crash recovery.
"""

from typing import Literal, Self

from pydantic import model_validator

from iris.artifacts.scientific_checkpoint_v1 import Digest, _Strict
from iris.utils.hashing import canonical_json_bytes, domain_sha256

COMMAND_WORKFLOW_DEFINITION_DOMAIN = "iris/scientific-ablation-command-workflow-definition/v1"
COMMAND_WORKFLOW_NAMESPACE = "scientific-ablation-command-workflow-v1"


class ScientificAblationCommandWorkflowDefinitionPayloadV1(_Strict):
    kind: Literal["iris-scientific-ablation-command-workflow-definition-v1"] = (
        "iris-scientific-ablation-command-workflow-definition-v1"
    )
    operator_sha256: Digest
    execution_sha256: Digest
    export_request_sha256: Digest
    export_result_sha256: Digest
    workflow_plan_sha256: Digest
    authority: Literal["namespace_data_only"] = "namespace_data_only"


class IrisScientificAblationCommandWorkflowDefinitionV1(_Strict):
    sha256: Digest
    payload: ScientificAblationCommandWorkflowDefinitionPayloadV1

    @model_validator(mode="after")
    def matches_definition(self) -> Self:
        if self.sha256 != domain_sha256(
            COMMAND_WORKFLOW_DEFINITION_DOMAIN,
            canonical_json_bytes(self.payload.model_dump(mode="json")),
        ):
            raise ValueError("Command workflow namespace definition checksum differs")
        return self
