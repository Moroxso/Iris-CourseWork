"""Genuine four-kernel original-stop recovery after actual FIRST row publication.

Original operator/export/factory, actual generation/scoring and real row/ACK files.
The interruption is an in-process primary signal, NOT a hard exit/cold/race proof.
Four non-executable rows and one nonempty zero-workspace control, original16MiB.
"""

import json
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_operator as operator
from iris.application import scientific_ablation_operator_export as export
from iris.application import scientific_ablation_operator_workflow as binding
from iris.application import scientific_ablation_workflow_command as public
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.application.scientific_ablation_workflow_io import (
    ScientificAblationWorkflowStoreV1,
    build_scientific_ablation_workflow_plan_v1,
)
from iris.application.scientific_ablation_workflow_replay import (
    build_scientific_ablation_workflow_command_intent_v1 as build,
)
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_command_v1 import (
    ScientificAblationWorkflowCommandIntentPayloadV1 as Intent,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowIntentV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import SyntheticSplitPlan
from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration.test_scientific_ablation_operator_export import _intent
from tests.integration.test_scientific_ablation_operator_workflow import cache_ready as cache_ready
from tests.integration.test_scientific_ablation_operator_workflow import completed as completed
from tests.integration.test_scientific_ablation_workflow_producer_result import _wire
from tests.integration.test_scientific_ablation_workflow_public_chunk import _binding
from tests.integration.test_scientific_generation import _test_spec
from tests.unit._workflow_data_setup import intent as data_intent

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def multirow_bound(
    completed: tuple[Path, Path], cache_ready: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1]:
    path, base = completed
    root = tmp_path_factory.mktemp("multirow-operator-export")
    request = _intent(path)
    receipt = export.export_scientific_ablation_operator_v1(
        path, request=request, base=base, root=root
    )
    source = binding.read_scientific_ablation_operator_generation_source_v1(
        path, request=request, export_root=root
    )
    recipe = operator._recipe(operator.inspect_scientific_ablation_operator_v1(path).descriptor)
    rows = tuple(
        row
        for row in generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
        if row.task_type == "next_tool_prediction" and row.difficulty == 1
    )[:4]
    assert len(rows) == 4 and not any(row.executable for row in rows)
    test = _test_spec(base / "multirow-workflow-test.parquet", rows)
    control = WorkspaceInterventionPlanV1(
        interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
    )
    raw = data_intent(base).model_dump(mode="python") | dict(
        source_sha256=source.bundle.manifest.sha256,
        lifecycle_sha256=request.payload.lifecycle_sha256,
        inference_sha256=receipt.result.payload.inference_sha256,
        selected_checkpoint_sha256=request.payload.source_checkpoint_sha256,
        lifecycle_path=str(source.lifecycle_path),
        inference_path=str(source.inference_path),
        inputs_root=str(base),
        training=recipe.train.model_dump(mode="python"),
        validation=recipe.validation.model_dump(mode="python"),
        test=test.model_dump(mode="python"),
        controls=(control,),
    )
    plan = build_scientific_ablation_workflow_plan_v1(
        ScientificAblationWorkflowIntentV1.model_validate(raw)
    )
    assert plan.payload.resources.maximum_storage_bytes == 16 * 1024**2
    assert cache_ready.is_dir()
    return path, base, root, plan


def _row_pins(path: Path) -> dict[str, tuple[int, int, int, str]]:
    return {
        file.name: (
            file.stat().st_dev,
            file.stat().st_ino,
            file.stat().st_size,
            sha256(file.read_bytes()).hexdigest(),
        )
        for file in path.iterdir()
    }


def test_genuine_all_four_kernels_resume_first_ack_to_same_stop_without_row_replay(
    multirow_bound: tuple[Path, Path, Path, IrisScientificAblationWorkflowPlanV1], tmp_path: Path
) -> None:
    path, base, exports, plan = multirow_bound
    facade = public.create_scientific_ablation_operator_command_workflow_v1(
        path,
        request=_intent(path),
        export_root=exports,
        plan=plan,
        base=base,
        root=tmp_path / "commands",
    )
    original = facade._seal.job
    witnesses: list[dict[str, Any]] = []
    for phase in scientific_ablation_workflow_phases_v1(plan)[:4]:
        assert phase.kind in ("capture", "score")
        module, builder, _ = _binding(phase)
        before = facade.progress.completed_phase_sha256
        if phase.kind == "capture":
            heldout = steps._owner(plan, phase, original._seal.factory)
            kernel_plan = getattr(module, builder)(
                heldout, expected_input_sha256=heldout.manifest.sha256
            )
        else:
            prefix = tuple(
                original._seal.store.read(p, expected_sha256=d)
                for p, d in zip(scientific_ablation_workflow_phases_v1(plan), before, strict=False)
            )
            kernel_plan = getattr(module, builder)(
                **steps._score_args(original._seal.store, prefix[-1])
            )
        assert len(kernel_plan.payload.rows) == 4
        interval = Chunk(
            kernel_plan_sha256=kernel_plan.sha256,
            acknowledgement_head_sha256=module._genesis(kernel_plan.sha256),
            acknowledged_rows=0,
            total_rows=4,
            rows_per_call=3,
            stop_row_index=3,
        )
        command = build(
            Intent(
                namespace=facade.definition,
                phase=phase,
                completed_phase_sha256=before,
                expected_chunk=interval,
                rows_per_call=3,
            )
        )
        command_raw = steps.encoded(command)
        kernel_path = (
            steps._kernel(original._seal.store, phase) / module._STORE / kernel_plan.sha256
        )
        prefix_name = "row" if phase.kind == "capture" else "score"
        called: list[int] = []
        published: list[int] = []
        signalled: list[int] = []
        publish = module._publish

        def publish_then_interrupt(
            stage: Any,
            target: Path,
            files: Any,
            claim: Any,
            *,
            publish: Any = publish,
            prefix_name: str = prefix_name,
            published: list[int] = published,
            signalled: list[int] = signalled,
        ) -> None:
            publish(stage, target, files, claim)
            if target.name.startswith(prefix_name + "-"):
                index = int(target.name.split("-")[1])
                published.append(index)
                if index == 0 and not signalled:
                    assert (target / "ack.json").is_file()
                    signalled.append(index)
                    raise KeyboardInterrupt(
                        "Actual first row ACK published, then primary interruption"
                    )

        with ExitStack() as stack:
            if phase.kind == "capture":
                cls = (
                    IrisScientificAblationHeldoutV1
                    if phase.stream_index is None
                    else IrisScientificAblationControlledHeldoutV1
                )
                execute = cls.execute_row

                def execute_observed(
                    self: Any, index: int, *, execute: Any = execute, called: list[int] = called
                ) -> Any:
                    called.append(index)
                    return execute(self, index)

                stack.enter_context(patch.object(cls, "execute_row", new=execute_observed))
            else:
                score_row = module._score_row

                def score_observed(
                    inputs: Any,
                    index: int,
                    *,
                    score_row: Any = score_row,
                    called: list[int] = called,
                ) -> Any:
                    called.append(index)
                    return score_row(inputs, index)

                stack.enter_context(patch.object(module, "_score_row", new=score_observed))
            stack.enter_context(patch.object(module, "_publish", new=publish_then_interrupt))
            operation: public.Operation = "generate" if phase.kind == "capture" else "score"
            with pytest.raises(KeyboardInterrupt):
                facade.dispatch(command, operation=operation)
            assert called == published == signalled == [0]
            assert facade.progress.completed_phase_sha256 == before
            observed = facade.inspect_kernel(
                expected_phase=phase, expected_kernel_plan_sha256=kernel_plan.sha256
            )
            assert observed.acknowledged_rows == 1
            row0 = kernel_path / f"{prefix_name}-00000"
            row0_before = _row_pins(row0)
            recovered = facade.recover(command, operation=operation)
            payload = recovered.result.payload
            assert payload.route == "continued_original_interval" and payload.resumed_from_row == 1
            assert payload.original_request_sha256 == command.sha256
            assert payload.original_return == "unavailable_not_reconstructed"
            assert payload.phase_record is None and not payload.phase_files
            assert payload.kernel_progress.acknowledged_rows == 3
            assert called == published == [0, 1, 2]
            assert not (kernel_path / f"{prefix_name}-00003").exists()
            assert _row_pins(row0) == row0_before
            assert steps.encoded(command) == command_raw
            assert facade.progress.completed_phase_sha256 == before
            handoff = original._seal.path / "handoffs" / command.sha256[:2] / command.sha256
            assert {p.name for p in handoff.iterdir()} == {"intent.json"}
            assert (handoff / "intent.json").read_bytes() == command_raw
            with (
                patch.object(
                    steps, "execute_produced", side_effect=AssertionError("DATA only")
                ) as work,
                patch.object(
                    steps, "_owner", side_effect=AssertionError("No DATA factory")
                ) as owner,
                patch.object(
                    torch.nn.Module, "__init__", side_effect=AssertionError("No DATA graph")
                ) as graph,
                patch.object(
                    Qwen3ForCausalLM, "forward", side_effect=AssertionError("No DATA forward")
                ) as forward,
                patch.object(
                    AutoModelForCausalLM,
                    "from_pretrained",
                    side_effect=AssertionError("No DATA model"),
                ) as model,
                patch.object(
                    AutoTokenizer,
                    "from_pretrained",
                    side_effect=AssertionError("No DATA tokenizer"),
                ) as tokenizer,
            ):
                again = facade.recover(
                    command,
                    operation=operation,
                    expected_operator_ack_sha256=recovered.operator_ack.sha256,
                )
                assert again == recovered
                assert all(
                    v.call_count == 0 for v in (work, owner, graph, forward, model, tokenizer)
                )
                with (
                    patch.object(
                        ScientificAblationWorkflowStoreV1,
                        "claim",
                        side_effect=AssertionError(
                            "Stale strict dispatch is rejected BEFORE claim"
                        ),
                    ) as claim,
                    pytest.raises(public.IrisScientificAblationWorkflowCommandError),
                ):
                    facade.dispatch(command, operation=operation)
                assert claim.call_count == 0
            # A deliberately NEW, explicit one-row request completes the phase.
            # This is NOT a rebased retry or a different budget disguised as the old command.
            final_interval = Chunk(
                kernel_plan_sha256=kernel_plan.sha256,
                acknowledgement_head_sha256=payload.kernel_progress.acknowledgement_head_sha256,
                acknowledged_rows=3,
                total_rows=4,
                rows_per_call=1,
                stop_row_index=4,
            )
            final_command = build(
                Intent(
                    namespace=facade.definition,
                    phase=phase,
                    completed_phase_sha256=before,
                    expected_chunk=final_interval,
                    rows_per_call=1,
                )
            )
            assert final_command.sha256 != command.sha256
            finished = facade.dispatch(final_command, operation=operation)
            assert len(finished.original_result.progress.completed_phase_sha256) == phase.index + 1
            assert called == published == [0, 1, 2, 3]
            assert _row_pins(row0) == row0_before
        witnesses.append(
            dict(
                phase=phase.model_dump(mode="json"),
                original_command=json.loads(command_raw),
                first_ack_observation=observed.model_dump(mode="json"),
                row0_before=row0_before,
                recovery_intent=json.loads(recovered.intent_json),
                recovery_result=json.loads(recovered.result_json),
                recovery_operator_ack=json.loads(recovered.operator_ack_json),
                actual_row_calls=called,
                actual_published_rows=published,
                no_row3_before_new_explicit_command=True,
                completed_command=final_command.model_dump(mode="json"),
                completed_original_return=_wire(finished.original_result),
                completed_operator_ack=json.loads(finished.operator_ack_json),
            )
        )
    assert len(witnesses) == len(facade.progress.completed_phase_sha256) == 4
    assert {v["phase"]["kind"] for v in witnesses} == {"capture", "score"}
    (tmp_path / "actual-all4-multirow-recovery-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                operator_path=str(path),
                base=str(base),
                export_root=str(exports),
                workflow_path=str(original._seal.path),
                plan=plan.model_dump(mode="json"),
                phase_anchors=facade.progress.completed_phase_sha256,
                actual_kernel_cases=witnesses,
                genuine_original_factory_and_rows=True,
                primary_signal_only_not_os_cold_hard_exit_or_race=True,
                native_scientific_or_installed_acceptance=False,
            )
        )
    )
