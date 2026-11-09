"""Actual OLD immutable NN-produced replies after later progress; no native replay.

The explicit retained DATA fixture is read-only. Fresh full-suite runs may use
the original fixture producer lazily; they do not relabel copied DATA as native.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

from iris.application import scientific_ablation_operator_workflow as operator
from iris.application import scientific_ablation_workflow as owner
from iris.application import scientific_ablation_workflow_replay as replay
from iris.application.scientific_ablation_workflow_io import ScientificAblationWorkflowStoreV1
from iris.artifacts.scientific_ablation_operator_export_v1 import (
    IrisScientificAblationExportRequestV1,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    IrisScientificAblationWorkflowCommandIntentV1,
)
from iris.artifacts.scientific_ablation_workflow_v1 import IrisScientificAblationWorkflowPlanV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_operator_workflow import bound as bound
from tests.integration.test_scientific_ablation_operator_workflow import cache_ready as cache_ready
from tests.integration.test_scientific_ablation_operator_workflow import completed as completed
from tests.integration.test_scientific_ablation_workflow_handoff import (
    test_actual_concrete_producer_is_received_and_operator_ack_precedes_return as produce_original,
)
from tests.integration.test_scientific_ablation_workflow_producer_result import _wire

pytestmark = pytest.mark.integration


def test_all_four_old_original_results_replay_after_later_progress_without_owner_or_claim(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    retained = os.environ.get("IRIS_HANDOFF_REPLAY_DATA_ROOT")
    if retained is None:
        actual = request.getfixturevalue("bound")
        source = tmp_path / "original-source"
        source.mkdir()
        produce_original(actual, source)
        witness_path = source / "actual-original-handoff-witness.json"
    else:
        witness_path = (
            Path(retained)
            / "pytest/test_actual_concrete_producer_0/actual-original-handoff-witness.json"
        )
    w = json.loads(witness_path.read_bytes())
    plan = IrisScientificAblationWorkflowPlanV1.model_validate_json(
        canonical_json_file_bytes(w["plan"])
    )
    store_path = Path(w["workflow_path"])
    assert (store_path / "phase-001/complete.json").is_file() and len(w["phase_anchors"]) == 2
    inference = Path(plan.payload.inference_path)
    exported = json.loads((inference.parents[2] / "receipt.json").read_bytes())
    export_request = IrisScientificAblationExportRequestV1.model_validate_json(
        canonical_json_file_bytes(exported["payload"]["request"])
    )
    saved = []
    with (
        patch.object(
            operator,
            "create_scientific_ablation_command_run_v1",
            side_effect=AssertionError("No factory"),
        ) as factory,
        patch.object(
            AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("No model loader")
        ) as model,
        patch.object(
            AutoTokenizer, "from_pretrained", side_effect=AssertionError("No tokenizer loader")
        ) as tokenizer,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
        ) as forward,
        patch.object(
            ScientificAblationWorkflowStoreV1,
            "claim",
            side_effect=AssertionError("No mutation claim"),
        ) as claim,
        patch.object(owner, "_advance", side_effect=AssertionError("No advance")) as advance,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No generated Python"),
        ) as candidate,
    ):
        for digest, records in w["saved_records"].items():
            command = IrisScientificAblationWorkflowCommandIntentV1.model_validate_json(
                canonical_json_file_bytes(records["intent.json"])
            )
            assert command.sha256 == digest
            result = replay.read_scientific_ablation_operator_workflow_command_result_v1(
                Path(w["operator_path"]),
                request=export_request,
                export_root=Path(w["export_root"]),
                plan=plan,
                base=Path(w["base"]),
                root=store_path.parents[1],
                command=command,
                expected_operator_ack_sha256=records["operator-ack.json"]["sha256"],
            )
            wire = _wire(result)
            original = [
                v
                for v in w["original_returns"]
                if v["requested_phase"] == wire["requested_phase"]
                and v["expected_chunk"] == wire["expected_chunk"]
            ]
            assert canonical_json_file_bytes(original) == canonical_json_file_bytes([wire])
            saved.append(wire)
            # A syntactically valid, but independently WRONG final ACK must not
            # be learned/replaced from the command's own on-disk metadata.
            with pytest.raises(replay.IrisScientificAblationWorkflowReplayError):
                replay.read_scientific_ablation_operator_workflow_command_result_v1(
                    Path(w["operator_path"]),
                    request=export_request,
                    export_root=Path(w["export_root"]),
                    plan=plan,
                    base=Path(w["base"]),
                    root=store_path.parents[1],
                    command=command,
                    expected_operator_ack_sha256="a" * 64,
                )
        assert len(saved) == 4
        assert all(
            p.call_count == 0
            for p in (factory, model, tokenizer, graph, forward, claim, advance, candidate)
        )
    assert any(not v["progress"]["completed_phase_sha256"] for v in saved)
    (tmp_path / "actual-public-old-result-replay-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                original_witness=str(witness_path),
                results=saved,
                phase_anchors_current=w["phase_anchors"],
                no_factory_graph_forward_claim_advance_candidate=True,
                original_saved_results_not_moving_head=True,
                independently_wrong_ack_rejections=4,
                cold_native_ownership_recovery_or_scientific_acceptance=False,
            )
        )
    )
