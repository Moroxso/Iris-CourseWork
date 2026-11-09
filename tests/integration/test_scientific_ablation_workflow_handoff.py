"""Genuine concrete operator/native capture and DATA-only score handoff.

Two non-Python original fixture rows; not all4kernels/diagnostic/writer/cold
coverage. No original result/factory is replaced. Lost response is in-process.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from iris.application import _scientific_ablation_workflow_handoff as handoff
from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_operator_workflow as operator
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    _Claim,
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
from tests.integration.test_scientific_ablation_workflow_producer_result import (
    _call,
    _committed,
    _next,
    _wire,
)
from tests.integration.test_scientific_ablation_workflow_public_chunk import _first_request

pytestmark = pytest.mark.integration


def test_actual_concrete_producer_is_received_and_operator_ack_precedes_return(
    bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1], tmp_path: Path
) -> None:
    path, base, exports, plan = bound
    job = operator._create_operator_workflow(
        path,
        request=_intent(path),
        export_root=exports,
        plan=plan,
        base=base,
        root=tmp_path / "commands",
        command_namespace=True,
    )
    returns = []
    original_write = handoff._Receiver._write
    original_produce = steps.execute_produced
    lost: list[str] = []

    def lose_after_real_ack(
        self: handoff._Receiver, name: str, raw: bytes, seal: Any, claim: _Claim
    ) -> None:
        original_write(self, name, raw, seal, claim)
        if name == "operator-ack.json" and not lost:
            lost.append(self._sha)
            raise RuntimeError("Actual durable ACK, then lost caller response")

    with patch.object(steps, "execute_produced", wraps=original_produce) as native:
        for _ in range(2):
            phase = scientific_ablation_workflow_phases_v1(job._seal.store.plan)[
                len(job.progress.completed_phase_sha256)
            ]
            request = _first_request(job, phase)
            partial = _call(job, phase, request)
            returns.append(_wire(partial))
            assert partial.kernel_progress_json is not None
            request = _next(request, Kernel.model_validate_json(partial.kernel_progress_json))
            before_calls = native.call_count
            with (
                patch.object(handoff._Receiver, "_write", lose_after_real_ack),
                pytest.raises(IrisScientificAblationWorkflowStorageError),
            ):
                _call(job, phase, request)
            assert native.call_count == before_calls + 1 and job._seal.ready is not None
            pending = job._seal.ready.receiver
            assert pending is not None and pending._returned is not None
            actual_return = pending._returned
            with patch.object(steps, "execute_produced", side_effect=AssertionError("No replay")):
                returned = _call(job, phase, request)
            assert returned is actual_return
            _committed(job, phase, returned)
            returns.append(_wire(returned))
            assert (pending._path / "operator-ack.json").is_file()
            lost.clear()
        assert native.call_count == 4
    command_roots = sorted((job._seal.path / "handoffs").glob("*/*"))
    assert len(command_roots) == 4 and len(returns) == 4
    assert not tuple(job._seal.path.glob(".iris-workflow-*.tmp"))
    assert not tuple(job._seal.path.glob(".iris-handoff-*.tmp"))
    records = {
        folder.name: {file.name: json.loads(file.read_bytes()) for file in folder.iterdir()}
        for folder in command_roots
    }
    (tmp_path / "actual-original-handoff-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(path),
                base=str(base),
                export_root=str(exports),
                workflow_path=str(job._seal.path),
                plan=plan.model_dump(mode="json"),
                original_returns=returns,
                saved_records=records,
                actual_original_producer_calls=4,
                phase_anchors=job.progress.completed_phase_sha256,
                generated_python_cold_multirow_or_scientific_acceptance=False,
            )
        )
    )
