"""Genuine concrete public dispatch, live ACK delivery and old DATA replay.

Two ordinary original kernels/two non-Python rows; no factory/producer/result is
replaced. NOT controlled/all4kernels/multirow/OS-cold/race/installed/science proof.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_handoff as handoff
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as workflow
from iris.application import scientific_ablation_workflow_command as public
from iris.application.scientific_ablation_workflow_io import ScientificAblationWorkflowStoreV1
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1 as build,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1 as Kernel,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_operator_export import _intent
from tests.integration.test_scientific_ablation_operator_workflow import bound as bound
from tests.integration.test_scientific_ablation_operator_workflow import cache_ready as cache_ready
from tests.integration.test_scientific_ablation_operator_workflow import completed as completed
from tests.integration.test_scientific_ablation_workflow_producer_result import _next, _wire
from tests.integration.test_scientific_ablation_workflow_public_chunk import _first_request

pytestmark = pytest.mark.integration


def test_real_public_dispatch_delivers_live_ack_before_release_and_replays_exact_old_response(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1], tmp_path: Path
) -> None:
    path, base, exports, plan = bound
    facade = public.create_scientific_ablation_operator_command_workflow_v1(
        path,
        request=_intent(path),
        export_root=exports,
        plan=plan,
        base=base,
        root=tmp_path / "commands",
    )
    # Observe the actual original owner in the test, never inject a factory or a
    # constructible owner. Public request planning remains explicit source work.
    original = facade._seal.job
    returns, requests, replies = [], [], []
    minted: list[handoff._Completion] = []
    original_finish, original_reply = handoff._Receiver._finish, public._reply

    def observe_in_claim(self: Any, result: Any, seal: Any, claim: Any) -> None:
        original_finish(self, result, seal, claim)
        claim.lock.check()
        assert self._completion is not None and self._completion.result is result
        assert self._completion.operator_ack_json == (self._path / "operator-ack.json").read_bytes()
        minted.append(self._completion)

    def lose_delivery(seal: Any, command: Any, completed: Any) -> Any:
        original_reply(seal, command, completed)
        raise RuntimeError("Caller lost the actual public reply after its real original ACK")

    with (
        patch.object(steps, "execute_produced", wraps=steps.execute_produced) as producer,
        patch.object(handoff._Receiver, "_finish", observe_in_claim),
    ):
        for index in range(2):
            phase = scientific_ablation_workflow_phases_v1(plan)[index]
            operation: public.Operation = "generate" if index == 0 else "score"
            interval = _first_request(original, phase)
            for part in range(2):
                command = build(
                    Intent(
                        namespace=facade.definition,
                        phase=phase,
                        completed_phase_sha256=facade.progress.completed_phase_sha256,
                        expected_chunk=interval,
                        rows_per_call=1,
                    )
                )
                if index == 0 and part == 1:
                    before = producer.call_count
                    with (
                        patch.object(public, "_reply", lose_delivery),
                        pytest.raises(public.IrisScientificAblationWorkflowCommandError),
                    ):
                        facade.dispatch(command, operation=operation)
                    assert producer.call_count == before + 1
                    completed = original._seal.completed_handoff
                    assert completed is not None
                    with patch.object(
                        workflow,
                        "_advance",
                        side_effect=AssertionError("No second advance after original completion"),
                    ) as advance:
                        reply = facade.dispatch(command, operation=operation)
                    assert advance.call_count == 0 and reply.original_result is completed.result
                else:
                    reply = facade.dispatch(command, operation=operation)
                assert reply.delivery == "original_reference"
                assert (
                    reply.request_sha256
                    == command.sha256
                    == reply.operator_ack.payload.request_sha256
                )
                completion = minted[-1]
                assert completion.result is reply.original_result
                assert completion is original._seal.completed_handoff
                requests.append(command)
                replies.append(reply)
                returns.append(_wire(reply.original_result))
                assert reply.original_result.kernel_progress_json is not None
                interval = (
                    _next(
                        interval,
                        Kernel.model_validate_json(reply.original_result.kernel_progress_json),
                    )
                    if part == 0
                    else interval
                )
        assert producer.call_count == 4 and len(minted) == 4
    assert len(facade.progress.completed_phase_sha256) == 2
    assert (original._seal.path / "phase-001/complete.json").is_file()
    with (
        patch.object(
            workflow, "_advance", side_effect=AssertionError("No old command execution")
        ) as advance,
        patch.object(
            ScientificAblationWorkflowStoreV1,
            "claim",
            side_effect=AssertionError("No replay mutation claim"),
        ) as claim,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No replay graph")) as graph,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No replay forward")
        ) as forward,
        patch.object(
            AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("No replay model")
        ) as model,
        patch.object(
            AutoTokenizer, "from_pretrained", side_effect=AssertionError("No replay tokenizer")
        ) as tokenizer,
    ):
        old = facade.dispatch(
            requests[0],
            operation="generate",
            expected_operator_ack_sha256=replies[0].operator_ack.sha256,
        )
        assert old.delivery == "saved_original_data"
        assert canonical_json_file_bytes(_wire(old.original_result)) == canonical_json_file_bytes(
            returns[0]
        )
        assert old.operator_ack_json == replies[0].operator_ack_json
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            facade.dispatch(requests[0], operation="generate")
        with pytest.raises(public.IrisScientificAblationWorkflowCommandError):
            facade.dispatch(
                requests[0],
                operation="score",
                expected_operator_ack_sha256=replies[0].operator_ack.sha256,
            )
        assert all(p.call_count == 0 for p in (advance, claim, graph, forward, model, tokenizer))
    roots = sorted((original._seal.path / "handoffs").glob("*/*"))
    assert len(roots) == 4
    (tmp_path / "actual-public-dispatch-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(path),
                base=str(base),
                export_root=str(exports),
                workflow_path=str(original._seal.path),
                plan=plan.model_dump(mode="json"),
                original_returns=returns,
                saved_records={
                    folder.name: {p.name: json.loads(p.read_bytes()) for p in folder.iterdir()}
                    for folder in roots
                },
                phase_anchors=facade.progress.completed_phase_sha256,
                actual_original_producer_calls=4,
                actual_in_claim_completions=len(minted),
                public_reply_operator_ack=[json.loads(v.operator_ack_json) for v in replies],
                original_reference_after_lost_public_delivery=True,
                old_partial_result_after_later_physical_phase=True,
                no_cold_multirow_all4kernel_installed_or_scientific_acceptance=True,
            )
        )
    )
