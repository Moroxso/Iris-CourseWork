"""Genuine full native writer schedules; NOT public/scientific answer acceptance."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_causal_controls as source_module
from iris.application import scientific_ablation_writer_effects as module
from iris.application.scientific_ablation_causal_receipts import _observed_json
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts import scientific_ablation_causal_report_layout_v1 as layout
from iris.training.ablation_observation import (
    _invocation,
    ablation_workspace_state_sha256_v1,
)
from iris.training.model import IrisTeacherForcingModel
from iris.utils.hashing import canonical_json_bytes, canonical_json_file_bytes, domain_sha256
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_causal_controls import _input, _policy, _rows
from tests.integration.test_scientific_ablation_complete_source_preflight import ready, transaction
from tests.integration.test_scientific_ablation_diagnostics import Experiment, _experiment, _heldout
from tests.integration.test_scientific_ablation_evaluation_v2 import _make
from tests.integration.test_scientific_ablation_generation_source import Routing

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Experiment:
    root = tmp_path_factory.mktemp("native-writer-effects")
    return _experiment(_make(root / "case", rows=_rows()), root / "scores")


def projection(source: source_module._Source, value: module._WriterMeasurements) -> dict[str, Any]:
    """Real measured bytes against frozen layout, not a public report constructor."""
    pf = source.complete_preflight
    assert pf is not None
    root = dict(pf._shape.members)
    payload = root["payload"]
    assert isinstance(payload, layout._Object)
    source_metadata = dict(payload.members)["source_metadata"]
    assert isinstance(source_metadata, layout._Fixed)
    import json

    message = bool(source.design.payload.donor_pairs)
    protocol = "message" if message else "workspace"
    ordinary = []
    for row in value.rows:
        current = row.row.model_dump(mode="json")
        ordinary.append(
            {
                k: v
                for k, v in current.items()
                if k
                in (
                    "row_index",
                    "capture_row_sha256",
                    "evaluation_row_sha256",
                    "evaluation_ack_sha256",
                    "ordinary_teacher_invocation_sha256",
                    "target_sha256",
                    "target_token_count",
                    "target_nll_sum",
                    "ordinary_mean_target_nll",
                    "ordinary_trace",
                )
            }
        )
    passes = [
        dict(
            row_index=p.row_index,
            role=p.role,
            write_cycle=p.write_cycle,
            writer=p.writer,
            slot=p.slot,
            donor_row_index=p.donor_row_index,
            donor_pair_sha256=p.donor_pair_sha256,
            output=_observed_json(p.output),
        )
        for p in value.passes
    ]
    data = dict(
        kind=f"iris-scientific-ablation-{protocol}-writer-report-v1",
        scope="complete-teacher-writer-effects-not-generated-answer-quality",
        scientific_acceptance="not_assessed",
        source_metadata=json.loads(source_metadata.encoded),
        byte_preflight=dict(
            layout_sha256=pf.layout_sha256,
            maximum_report_bytes=pf.maximum_report_bytes,
            report_byte_preflight_bound=pf.report_byte_preflight_bound,
        ),
        actual_teacher_executions=value.actual_teacher_executions,
        actual_trace_proposal_observation_scalar_bound=value.actual_trace_proposal_observation_scalar_bound,
        actual_post_intervention_scalar_bound=value.actual_post_intervention_scalar_bound,
        ordinary_rows=ordinary,
        controlled_passes=dict(
            empty=[p for p in passes if p["role"] == "empty"],
            trials={
                role: [p for p in passes if p["role"] == role]
                for role in (("block", "donor") if message else ("shuffle", "zero"))
            },
        ),
        effects=[e.model_dump(mode="json") for e in value.effects],
        statistics=value.statistics.model_dump(mode="json"),
    )
    return dict(
        sha256=domain_sha256(
            f"iris/tests/native-writer-projection/{protocol}/v1", canonical_json_bytes(data)
        ),
        payload=data,
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_all12_both_full_original_best_writer_schedules(
    tmp_path: Path, routing: Routing, goal: str, broadcast: str
) -> None:
    ex = _experiment(
        _make(tmp_path / "case", routing, goal=goal, broadcast=broadcast, rows=_rows()),
        tmp_path / "scores",
    )
    owner = _heldout(ex, routing, goal=goal, broadcast=broadcast)
    context = owner._guard.context
    model, run = context._run._optimizer._model, context._run
    caller, private = capture_runtime_state(torch.device("cpu")), run._runtime
    modes = tuple((m, m.training) for m in model.modules())
    counts = {"ordinary": 0, "observed": 0}
    original_ordinary = IrisScientificAblationTeacherV1.evaluation_forward
    original_observed = IrisTeacherForcingModel.evaluation_forward_ablation_observed

    def busy() -> None:
        assert owner._state == context._state == context._request._state == "busy"
        assert run._busy and not any(m.training for m in model.modules())

    def ordinary(self: Any, *args: Any, **kwargs: Any) -> Any:
        busy()
        counts["ordinary"] += 1
        return original_ordinary(self, *args, **kwargs)

    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        busy()
        counts["observed"] += 1
        assert _same_runtime(private, capture_runtime_state(torch.device("cpu")))
        return original_observed(self, *args, **kwargs)

    with (
        patch.object(IrisScientificAblationTeacherV1, "evaluation_forward", new=ordinary),
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_observed", new=observed),
        patch.object(model.teacher, "_execute", wraps=model.teacher._execute) as execute,
        patch.object(IrisTeacherForcingModel, "evaluation_forward_ablation_controlled") as legacy,
        patch.object(type(context._request), "execute") as generate,
        patch("iris.application.synthetic_scoring_v2.execute_python_job_v2") as candidate,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as loader,
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer,
    ):
        for message in (False, True):
            before_calls = execute.call_count
            before_counts = counts.copy()
            protocol = "message" if message else "workspace"
            with transaction(ex, owner, policy=_policy(message)) as source:
                pf = source.complete_preflight
                assert pf is not None
                assert ex.progress.evaluation is not None
                anchors = ex.progress.evaluation.payload.plan.payload.source.payload
                assert (anchors.selected_step, anchors.terminal_step) == (1, 3)
                result = module._measure_writer_effects(source)
                n = 2 if message else 3
                assert len(result.rows) == n
                assert len(result.passes) == (9 if message else 7) * n
                assert (
                    execute.call_count - before_calls
                    == result.actual_teacher_executions
                    == ((10 if message else 8) * n)
                )
                assert counts["ordinary"] - before_counts["ordinary"] == n
                assert counts["observed"] - before_counts["observed"] == len(result.passes)
                assert len(result.effects) == n * 2 * 2 * 3
                assert len(result.statistics.pairs) == 12
                assert len(source.design.payload.members) == 4
                assert len(source.design.payload.groups) == 20
                assert (
                    result.actual_post_intervention_scalar_bound
                    == pf.post_intervention_scalar_bound
                )
                if broadcast == "disabled":
                    assert all(
                        e.genuine_distance == e.negative_control_distance == 0.0
                        for e in result.effects
                    )
                # All actual native post/before/native-written fields are projected.
                data = projection(source, result)
                encoded = layout.causal_report_layout_bytes_v1(data, pf)
                assert len(encoded) <= pf.report_byte_preflight_bound <= pf.maximum_report_bytes
                (tmp_path / f"native-writer-{protocol}.json").write_bytes(
                    canonical_json_file_bytes(
                        dict(
                            kind="iris-tests-actual-native-writer-schedule-v1",
                            native_measurements=True,
                            source_design=source.design.model_dump(mode="json"),
                            mode=[routing, goal, broadcast],
                            actual_execute_calls=execute.call_count - before_calls,
                            projected_report=data,
                            scientific_acceptance="not_assessed",
                            full_r015_complete=False,
                        )
                    )
                )
            ready(owner)
            assert all(m.training is state for m, state in modes)
            assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
            assert _same_runtime(private, run._runtime)
        assert counts == dict(ordinary=5, observed=39)
        legacy.assert_not_called()
        generate.assert_not_called()
        candidate.assert_not_called()
        loader.assert_not_called()
        tokenizer.assert_not_called()


@pytest.mark.parametrize("field", ("loss", "post", "before", "native", "target", "rewrite_back"))
def test_immediate_retention_denies_late_even_self_rehashed_native_changes(
    experiment: Experiment, field: str
) -> None:
    owner = _heldout(experiment)
    with transaction(experiment, owner, policy=_policy(False)) as source:
        pf = source.complete_preflight
        assert pf is not None
        stages = source_module._observed_controls(source, budget=pf.observed_budget)
        row = stages[0]
        sealed = row.retention_seal
        assert sealed is not None and row.prepared is not None
        output = sealed.output
        tensors = {
            "loss": output.controlled.loss,
            "post": output.controlled.workspace_trace[0].confidence,
            "before": output.observations[1].before_write.confidence,
            "native": output.observations[0].native_written.confidence,
            "target": row.prepared.target_input_ids,
            "rewrite_back": output.controlled.workspace_trace[0].confidence,
        }
        tensor = tensors[field]
        if field == "target":
            tensor.flatten()[0] += 1
        elif field == "loss":
            tensor.add_(0.1)
        else:
            old = tensor.clone()
            state = (
                output.observations[1].before_write
                if field == "before"
                else output.observations[0].native_written
                if field == "native"
                else output.controlled.workspace_trace[0]
            )
            assert bool(state.occupancy.any())
            tensor.masked_fill_(state.occupancy, 0.125)
            if field == "rewrite_back":
                tensor.copy_(old)
        if field in ("before", "native"):
            # Self-rehashing keeps the DTO locally valid, but not its ORIGINAL seal.
            for observation in output.observations:
                object.__setattr__(
                    observation,
                    "before_write_sha256",
                    ablation_workspace_state_sha256_v1(observation.before_write),
                )
                object.__setattr__(
                    observation,
                    "native_written_sha256",
                    ablation_workspace_state_sha256_v1(observation.native_written),
                )
            object.__setattr__(
                output,
                "invocation_sha256",
                _invocation(
                    output.controlled,
                    output.observations,
                    output.maximum_observation_scalars,
                    output.actual_observation_scalars,
                ),
            )
        output.__post_init__()
        with pytest.raises(ValueError, match="Original retained native pass/input changed"):
            sealed.audit()


@pytest.mark.parametrize("message", (False, True))
def test_complete_stage_denies_legacy_unreserved_transaction_without_forward(
    experiment: Experiment, message: bool
) -> None:
    owner = _heldout(experiment)
    with (
        source_module._source_transaction(
            owner,
            input=_input(experiment),
            policy=_policy(message),
            maximum_report_bytes=64 * 1024**2,
        ) as source,
        patch.object(source.model.teacher, "_execute") as execute,
    ):
        with pytest.raises(ValueError, match="Full numeric AND canonical JSON preflight"):
            module._measure_writer_effects(source)
        execute.assert_not_called()
    ready(owner)


@pytest.mark.parametrize("mutation", ("missing", "order", "count", "effect", "pair", "statistics"))
def test_complete_native_stage_audit_rederives_inventory_effects_statistics(
    experiment: Experiment, mutation: str
) -> None:
    owner = _heldout(experiment)
    with transaction(experiment, owner, policy=_policy(True)) as source:
        result = module._measure_writer_effects(source)
        if mutation == "missing":
            broken = replace(result, passes=result.passes[:-1])
        elif mutation == "order":
            broken = replace(result, passes=tuple(reversed(result.passes)))
        elif mutation == "count":
            broken = replace(result, actual_teacher_executions=result.actual_teacher_executions - 1)
        elif mutation == "effect":
            effect = result.effects[0].model_copy(update={"genuine_distance": 0.9})
            broken = replace(result, effects=(effect, *result.effects[1:]))
        elif mutation == "pair":
            donor = next(p for p in result.passes if p.role == "donor")
            altered = replace(donor, donor_pair_sha256="0" * 64)
            broken = replace(
                result, passes=tuple(altered if p is donor else p for p in result.passes)
            )
        else:
            stats = result.statistics.model_copy(update={"pairs": result.statistics.pairs[:-1]})
            broken = replace(result, statistics=stats)
        with pytest.raises((ValueError, KeyError)):
            module._audit_writer_measurements(source, broken)
        # Catching an internal diagnostic denial does not mutate original owners.
        module._audit_writer_measurements(source, result)
    ready(owner)
