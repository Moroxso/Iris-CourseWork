"""Actual whole PUBLIC reports on occupied/expiry/wide original source graphs.

No manufactured DTOs, repeated effect tuple, forward replacement, downloads or
generated Python execution. These tiny random-Qwen CPU tests are not science.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as source_module
from iris.application import scientific_ablation_writer_reports as service
from iris.application.scientific_ablation_causal_controls import (
    ScientificAblationCausalCompleteLimitsV1,
)
from iris.application.scientific_ablation_heldout import build_scientific_ablation_heldout_v1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.artifacts.scientific_ablation_causal_observed_budget_v1 import (
    build_scientific_ablation_causal_observed_budget_v1,
)
from iris.artifacts.scientific_ablation_writer_reports_v1 import (
    ScientificAblationWriterCPURuntimeV1,
    scientific_ablation_writer_report_bytes_v1,
)
from iris.artifacts.scientific_evaluation_v1 import TASKS
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes
from iris.utils.seeds import capture_runtime_state
from tests.integration.scientific_ablation_writer_geometry_fixture import Geometry, case, context
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_complete_source_preflight import ready
from tests.integration.test_scientific_ablation_diagnostics import _experiment
from tests.integration.test_scientific_ablation_generation_source import Routing

pytestmark = pytest.mark.integration


def _measure(
    tmp_path: Path,
    geometry: Geometry,
    routing: Routing,
    goal: str,
    broadcast: str,
    rows: tuple[IrisExample, ...],
    *,
    wide: bool = False,
) -> dict[str, Any]:
    # Windows host NEVER executes generated Python, including an accidental valid
    # generated envelope. This trap is installed during original scoring as well.
    with patch(
        "iris.application.synthetic_scoring_v2.execute_python_job_v2",
        side_effect=AssertionError("No candidate Python execution on host"),
    ) as candidate:
        ex = _experiment(
            case(tmp_path / "case", geometry, routing, rows, goal=goal, broadcast=broadcast),
            tmp_path / "scores",
        )
        candidate.assert_not_called()
    owner_context = context(ex.case.source, geometry, routing, goal=goal, broadcast=broadcast)
    owner = build_scientific_ablation_heldout_v1(
        owner_context,
        expected_reconstruction_sha256=owner_context.reconstruction.sha256,
        test=ex.case.test,
        base=ex.case.root,
        decoding=ex.case.capture.plan.payload.input.payload.decoding,
    )
    run = owner_context._run
    model = run._optimizer._model
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    modes = tuple((child, child.training) for child in model.modules())
    dtype = torch.get_default_dtype()
    events: list[dict[str, Any]] = []
    reservations: list[Any] = []
    original_numeric = build_scientific_ablation_causal_observed_budget_v1
    original_bytes = layout.build_scientific_ablation_causal_report_byte_preflight_v1

    def before(stage: str) -> None:
        ready(owner)
        assert all(child.training is was for child, was in modes)
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        assert _same_runtime(private, run._runtime)
        events.append(
            dict(
                stage=stage,
                states=[owner._state, owner_context._state, owner_context._request._state],
                run_busy=run._busy,
                original_modes_restored=True,
                caller_rng_unchanged=True,
                private_rng_unchanged=True,
            )
        )

    def numeric(*args: Any, **kwargs: Any) -> Any:
        before("numeric")
        with patch.object(
            model.teacher, "_execute", side_effect=AssertionError("No preflight forward")
        ):
            return original_numeric(*args, **kwargs)

    def byte_preflight(*args: Any, **kwargs: Any) -> Any:
        before("bytes")
        with patch.object(
            model.teacher, "_execute", side_effect=AssertionError("No preflight forward")
        ):
            result = original_bytes(*args, **kwargs)
        reservations.append(result)
        return result

    summaries: dict[str, Any] = {}
    runtime = ScientificAblationWriterCPURuntimeV1(
        torch_version=str(torch.__version__),
        torch_git_version=torch.version.git_version,
    )
    with (
        patch.object(
            source_module, "build_scientific_ablation_causal_observed_budget_v1", new=numeric
        ),
        patch.object(
            source_module,
            "build_scientific_ablation_causal_report_byte_preflight_v1",
            new=byte_preflight,
        ),
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_controlled") as legacy,
        patch.object(type(owner_context._request), "execute") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        for message in (False, True):
            protocol = "message" if message else "workspace"
            actual_before = execute.call_count
            policy = _policy(
                message,
                examples_per_stratum=16 if wide else (2 if message else 1),
                maximum_examples=192 if wide else (64 if message else 768),
                maximum_trace_scalars=2_000_000,
                maximum_total_numeric_scalars=64_000_000,
                maximum_executions=8192,
            )
            measure = (
                service.measure_scientific_ablation_message_writer_report_v1
                if message
                else service.measure_scientific_ablation_workspace_writer_report_v1
            )
            report = measure(
                owner,
                input=_input(ex),
                policy=policy,
                limits=ScientificAblationCausalCompleteLimitsV1(
                    maximum_observation_scalars=2
                    * geometry.cycles
                    * geometry.slots
                    * (geometry.width + 6),
                ),
            )
            ready(owner)
            assert all(child.training is was for child, was in modes)
            assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
            assert _same_runtime(private, run._runtime)
            assert torch.get_default_dtype() is dtype
            pf = reservations[-1]
            content = scientific_ablation_writer_report_bytes_v1(
                report, preflight=pf, runtime=runtime
            )
            p = report.payload
            assert len(content) <= pf.report_byte_preflight_bound <= pf.maximum_report_bytes
            assert execute.call_count - actual_before == p.actual_teacher_executions
            assert ex.progress.evaluation is not None
            source = ex.progress.evaluation.payload.plan.payload.source.payload
            assert (source.selected_step, source.terminal_step) == (1, 3)
            data = json.loads(content)
            frames = data["payload"]["controlled_passes"]
            count = len(p.ordinary_rows)
            per_row = 2 + (geometry.cycles - 1) * (
                2 * geometry.writers if message else geometry.writers + 1
            )
            assert p.actual_teacher_executions == count * per_row
            assert len(p.effects) == count * (geometry.cycles - 1) * geometry.writers * (
                geometry.modules - 1
            )
            assert len(p.statistics.pairs) == geometry.modules * (geometry.modules - 1)
            empty = {frame["row_index"]: frame for frame in frames["empty"]}
            occupied_blocks = nonwinning_donors = expirations = 0
            if message:
                for frame in frames["trials"]["block"]:
                    cycle = frame["write_cycle"] - 1
                    output = frame["output"]
                    before_state = output["observations"][cycle]["before_write"]
                    post = output["controlled"]["states"][cycle]
                    if before_state["occupancy"]["values"][0][frame["slot"]]:
                        occupied_blocks += 1
                        assert canonical_json_bytes(post) == canonical_json_bytes(before_state)
                        assert canonical_json_bytes(before_state) != canonical_json_bytes(
                            output["observations"][cycle]["native_written"]
                        )
                for frame in frames["trials"]["donor"]:
                    cycle = frame["write_cycle"] - 1
                    donor = empty[frame["donor_row_index"]]["output"]["controlled"]
                    nonwinning_donors += (
                        frame["writer"] not in donor["trace"]["cycles"][cycle]["selected_modules"]
                    )
            for frame in frames["empty"]:
                output = frame["output"]
                for cycle in range(1, geometry.cycles):
                    previous = output["controlled"]["states"][cycle - 1]
                    before_state = output["observations"][cycle]["before_write"]
                    for slot in range(geometry.slots):
                        expirations += (
                            previous["occupancy"]["values"][0][slot]
                            and previous["ttl"]["values"][0][slot] <= 1.0
                            and not before_state["occupancy"]["values"][0][slot]
                        )
            metrics = dict(
                actual_teacher_executions=p.actual_teacher_executions,
                effects=len(p.effects),
                pairs=len(p.statistics.pairs),
                occupied_block_frames=occupied_blocks,
                nonwinning_donor_frames=nonwinning_donors,
                actual_expired_records=expirations,
                overwrite_effects=sum(e.overwrote_existing_record for e in p.effects),
                nonsurviving_effects=sum(not e.survives_to_next_read for e in p.effects),
            )
            if broadcast == "disabled":
                assert all(
                    e.genuine_distance == e.negative_control_distance == 0.0 for e in p.effects
                )
            original_root = dict(pf._shape.members)
            original_payload = original_root["payload"]
            assert isinstance(original_payload, layout._Object)
            original_source = dict(original_payload.members)["source_metadata"]
            assert isinstance(original_source, layout._Fixed)
            reserved = dict(
                kind="iris-tests-original-complete-source-before-busy-v1",
                scope="trusted-ready-numeric-and-json-reservation-not-full-effects",
                actual_preflight_teacher_executions=0,
                before_observations=events[-2:],
                complete_layout=layout._descriptor(pf._shape),
                source_metadata=json.loads(original_source.encoded),
                **{
                    f.name: getattr(pf, f.name)
                    for f in fields(pf)
                    if f.name.endswith("_bound") or f.name.endswith("_bytes")
                },
                layout_sha256=pf.layout_sha256,
                scientific_acceptance="not_assessed",
                full_r015_complete=False,
            )
            (tmp_path / f"public-writer-{protocol}.json").write_bytes(content)
            (tmp_path / f"preflight-{protocol}.json").write_bytes(
                canonical_json_file_bytes(reserved)
            )
            summaries[protocol] = metrics
        for mocked in (legacy, generate, candidate, loader, tokenizer):
            mocked.assert_not_called()
    (tmp_path / "geometry-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-tests-public-writer-original-geometry-witness-v1",
                geometry=dict(
                    modules=geometry.modules,
                    slots=geometry.slots,
                    width=geometry.width,
                    cycles=geometry.cycles,
                    writers=geometry.writers,
                    empty_threshold=geometry.empty_threshold,
                ),
                metrics=summaries,
                scientific_acceptance="not_assessed",
                full_iris_gw_01_complete=False,
            )
        )
    )
    return summaries


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
@pytest.mark.parametrize("scenario", ("occupied", "expiry", "nonsurvival"))
def test_all12_both_public_original_occupied_expiry_and_nonsurvival_geometry(
    tmp_path: Path,
    routing: Routing,
    goal: str,
    broadcast: str,
    scenario: str,
) -> None:
    geometry = (
        Geometry()
        if scenario == "occupied"
        else Geometry(slots=4, cycles=5, empty_threshold=1.0)
        if scenario == "expiry"
        else Geometry(empty_threshold=4.0)
    )
    summaries = _measure(tmp_path, geometry, routing, goal, broadcast, _rows())
    if scenario == "occupied":
        assert summaries["message"]["occupied_block_frames"] > 0
        assert all(summary["overwrite_effects"] > 0 for summary in summaries.values())
    elif scenario == "expiry":
        assert any(summary["actual_expired_records"] > 0 for summary in summaries.values())
    else:
        assert any(summary["nonsurviving_effects"] > 0 for summary in summaries.values())


def test_actual_whole_wide_both_reports_12096_effects_4032_directed_pairs(tmp_path: Path) -> None:
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=3000),
            template_id="iris-synthetic-01-v2",
        )
    )
    tasks = TASKS
    selected = tuple(
        row
        for task in tasks
        for difficulty in (1, 2, 3)
        for row in tuple(r for r in pool if r.task_type == task and r.difficulty == difficulty)[:16]
    )
    assert len(selected) == 192
    result = _measure(
        tmp_path,
        Geometry(modules=64, width=1, cycles=2),
        "learned",
        "enabled",
        "enabled",
        selected,
        wide=True,
    )
    for metrics in result.values():
        assert metrics["actual_teacher_executions"] == 768
        assert metrics["effects"] == 12096 > 8192 and metrics["pairs"] == 4032
