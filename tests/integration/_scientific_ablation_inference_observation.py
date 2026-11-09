"""Pure output observations shared with an executable worker, no training factories."""

import json
from dataclasses import asdict, fields
from typing import Any
from unittest.mock import patch

import torch

from iris.application.scientific_ablation_receipt import IrisScientificAblationRequestOutputV1
from iris.application.scientific_ablation_request import IrisScientificAblationRequestV1
from iris.backbone import BackboneAdapter
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import DecodingStrategy
from iris.utils.hashing import bytes_sha256, canonical_json_bytes, tensor_sha256
from iris.workspace.state import WorkspaceState


def workspace_observation(trace: tuple[WorkspaceState, ...]) -> list[dict[str, str]]:
    return [
        {field.name: tensor_sha256(getattr(cycle, field.name)) for field in fields(cycle)}
        for cycle in trace
    ]


def output_observation(
    output: IrisScientificAblationRequestOutputV1, generator: torch.Generator
) -> dict[str, Any]:
    observed: dict[str, Any] = json.loads(
        canonical_json_bytes(
            {
                "text": output.generation.text,
                "tokens": output.generation.token_ids.tolist(),
                "stop_reason": output.generation.stop_reason,
                "matched_stop_token_id": output.generation.matched_stop_token_id,
                "workspace": workspace_observation(output.generation.workspace_trace),
                "receipt": asdict(output.receipt),
                "caller_sampling_state_sha256": tensor_sha256(generator.get_state()),
            }
        )
    )
    return observed


def compact_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep receipt/tokens/workspace, deduplicate actual identical prefill arrays."""
    assert len(outputs) == 4
    prefill = outputs[0]["prefill_logits"]
    compact = []
    for output in outputs:
        assert output["prefill_logits"] == prefill
        phases = output["forward_phases"]
        compact.append(
            {
                **{
                    key: value
                    for key, value in output.items()
                    if key not in {"prefill_logits", "forward_phases"}
                },
                "prefill_logits_sha256": bytes_sha256(
                    canonical_json_bytes(output["prefill_logits"])
                ),
                "forward_phases_sha256": bytes_sha256(canonical_json_bytes(phases)),
                "prefill_count": phases.count("prefill"),
                "cached_count": phases.count("cached"),
            }
        )
    return {"outputs": compact, "prefill_logits": prefill}


def execute_observation(
    request: IrisScientificAblationRequestV1,
    backbone: BackboneAdapter,
    prompt: str,
    strategy: DecodingStrategy,
    generator: torch.Generator,
) -> dict[str, Any]:
    native = backbone.forward_embeddings
    phases: list[str] = []
    logits: list[list[list[float]]] = []

    def actual(*args: Any, **kwargs: Any) -> BackboneCausalOutput:
        result = native(*args, **kwargs)
        phases.append("prefill" if kwargs.get("past_key_values") is None else "cached")
        if len(phases) == 1:
            logits.append(result.logits[:, -1, :].tolist())
        return result

    with patch.object(backbone, "forward_embeddings", side_effect=actual):
        output = request.execute(
            prompt, strategy=strategy, generator=generator if strategy == "sampling" else None
        )
    observed = output_observation(output, generator)
    observed["forward_phases"] = phases
    observed["prefill_logits"] = logits[0]
    assert output.receipt.max_new_tokens == 256
    assert phases == ["prefill", *(["cached"] * 255)]
    assert output.generation.token_ids.shape == (1, 256)
    assert output.generation.stop_reason == "max_new_tokens"
    return observed
