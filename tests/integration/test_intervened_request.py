"""Actual D-cycle interventions, separate receipts and genuine offline tiny Qwen."""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from iris.application import (
    IrisIntervenedRequestExecutionOutput,
    IrisIntervenedRequestExecutionReceiptV1,
    IrisRequestOrchestrator,
    build_request_orchestrator,
)
from iris.application.bootstrap import IrisArmRequestRuntimeV2, IrisRequestRuntime
from iris.application.errors import IrisRequestArtifactError, IrisRequestError
from iris.application.request import IrisRequestExecutionOutput, IrisRequestExecutionReceiptV1
from iris.application.request_artifact import build_request_artifact_v1
from iris.application.request_artifact_v2 import build_request_artifact_v2
from iris.application.request_artifact_v3 import build_request_artifact_v3
from iris.application.request_artifact_v4 import build_request_artifact_v4
from iris.backbone import BackboneAdapter
from iris.generation import TemperatureTopPSamplingConfig
from iris.interventions import (
    WorkspaceInterventionPlanV1,
    WorkspaceInterventionSpecV1,
    intervention_plan_sha256_v1,
)
from iris.interventions.session import _build_operation
from iris.schemas import BackboneConfig
from iris.utils.hashing import module_state_sha256, tensor_sha256
from tests.integration.test_request_sidecar import (
    _build_harness,
    _cognitive_config,
    _workspace_config,
)
from tests.unit.test_bootstrap import CPU_INFO
from tests.unit.test_request_artifact_v3 import _runtime_property_patches, artifact_inputs
from tests.unit.test_request_artifact_v4 import _transfer
from tests.unit.test_workspace_intervention_state import assert_same, clone

pytestmark = pytest.mark.integration


def plan(*specifications: dict[str, Any]) -> WorkspaceInterventionPlanV1:
    return WorkspaceInterventionPlanV1(
        interventions=tuple(
            WorkspaceInterventionSpecV1.model_validate(spec) for spec in specifications
        )
    )


def harness() -> Any:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        return _build_harness()


def test_empty_plan_is_identical_to_baseline_but_has_a_distinct_required_receipt() -> None:
    model = harness()
    before_rng = torch.get_rng_state().clone()
    baseline = model.orchestrator.execute("paired control", strategy="greedy")
    prefix = model.iris_generator.prefixes[-1]
    empty = plan()
    with patch(
        "iris.application.request._capture_request_execution_receipt_v1",
        side_effect=AssertionError("No baseline receipt on intervention path"),
    ):
        changed = model.orchestrator.execute_intervened(
            "paired control", plan=empty, strategy="greedy"
        )
    assert type(baseline) is IrisRequestExecutionOutput
    assert type(baseline.receipt) is IrisRequestExecutionReceiptV1
    assert type(changed) is IrisIntervenedRequestExecutionOutput
    assert type(changed.receipt) is IrisIntervenedRequestExecutionReceiptV1
    assert not isinstance(changed, IrisRequestExecutionOutput)
    assert not isinstance(changed.receipt, IrisRequestExecutionReceiptV1)
    assert changed.receipt.intervention_plan_sha256 == intervention_plan_sha256_v1(empty)
    assert changed.receipt.prompt_sha256 == baseline.receipt.prompt_sha256
    assert torch.equal(changed.prompt_input_ids, baseline.prompt_input_ids)
    assert torch.equal(changed.prompt_attention_mask, baseline.prompt_attention_mask)
    assert torch.equal(changed.generation.token_ids, baseline.generation.token_ids)
    assert torch.equal(model.iris_generator.prefixes[-1].inputs_embeds, prefix.inputs_embeds)
    for left, right in zip(
        changed.generation.workspace_trace, baseline.generation.workspace_trace, strict=True
    ):
        assert_same(left, right)
    assert torch.equal(torch.get_rng_state(), before_rng)


@pytest.mark.parametrize(
    "specification",
    [
        {"kind": "zero_workspace"},
        {"kind": "zero_slot", "slot": 0},
        {"kind": "swap_slots", "slot": 0, "other_slot": 1},
        {"kind": "shuffle_slots", "seed": 42},
        {"kind": "gaussian_noise", "seed": 42, "sigma": 0.25},
        {
            "kind": "replace_slot",
            "slot": 0,
            "values": (1.0, -1.0, 0.5, 0.25),
            "source_module": 2,
            "confidence": 0.75,
            "ttl": 4.0,
        },
    ],
)
def test_state_intervention_is_applied_to_the_real_first_cycle_output(
    specification: dict[str, Any],
) -> None:
    model = harness()
    baseline = model.orchestrator.execute("state phase", strategy="greedy")
    selected = plan({**specification, "cycles": (1,)})
    expected = _build_operation(selected.interventions[0]).apply(
        baseline.generation.workspace_trace[0], 1
    )
    before_rng = torch.get_rng_state().clone()
    changed = model.orchestrator.execute_intervened("state phase", plan=selected, strategy="greedy")
    assert_same(changed.generation.workspace_trace[0], expected)
    assert changed.receipt.intervention_plan_sha256 == intervention_plan_sha256_v1(selected)
    assert len(changed.generation.workspace_trace) == 3
    assert torch.equal(before_rng, torch.get_rng_state())
    assert changed.receipt.sidecar_state_sha256 == model.orchestrator.sidecar_state_sha256


def test_freeze_restores_every_record_field_and_each_request_gets_a_fresh_snapshot() -> None:
    model = harness()
    selected = plan({"kind": "freeze_slot", "slot": 0})
    first = model.orchestrator.execute_intervened("freeze", plan=selected, strategy="greedy")
    expected = clone(first.generation.workspace_trace[0])
    for state in first.generation.workspace_trace:
        for field in fields(state):
            assert torch.equal(
                getattr(state, field.name)[:, 0], getattr(expected, field.name)[:, 0]
            )
    with torch.inference_mode():
        first.generation.workspace_trace[0].slots.zero_()
    again = model.orchestrator.execute_intervened("freeze", plan=selected, strategy="greedy")
    assert_same(again.generation.workspace_trace[0], expected)


def test_capacity_is_enforced_before_read_allocation_and_after_later_replacement() -> None:
    model = harness()
    owner = cast(Any, model.orchestrator)
    selected = plan(
        {"kind": "reduce_workspace_capacity", "capacity": 1},
        {
            "kind": "replace_slot",
            "slot": 1,
            "values": (1.0, 2.0, 3.0, 4.0),
            "source_module": 2,
            "confidence": 1.0,
            "ttl": 4.0,
        },
    )
    with patch.object(
        owner._workspace_reader, "forward", wraps=owner._workspace_reader.forward
    ) as reads:
        changed = owner.execute_intervened("capacity", plan=selected, strategy="greedy")
    assert reads.call_count == 3
    for call in reads.call_args_list:
        assert not bool(call.args[0].occupancy[:, 1].any())
    for state in changed.generation.workspace_trace:
        assert state.slots.shape == (1, 2, 4)
        assert bool(state.occupancy[:, 0].all())
        assert not bool(state.occupancy[:, 1].any())
        assert bool((state.source_module[:, 1] == -1).all())
    assert model.iris_generator.prefixes[-1].inputs_embeds.shape[1] == 5


def test_blocked_read_is_the_actual_input_received_by_cognitive_modules() -> None:
    model = harness()
    owner = cast(Any, model.orchestrator)
    with patch.object(
        owner._cognitive_modules, "forward", wraps=owner._cognitive_modules.forward
    ) as calls:
        owner.execute_intervened(
            "read control", plan=plan({"kind": "block_module_read", "module": 1}), strategy="greedy"
        )
    assert calls.call_count == 3
    for call in calls.call_args_list:
        assert not bool(call.args[1][:, 1].any())
    assert bool(calls.call_args_list[1].args[1][:, 0].any())


def test_bid_scaling_reaches_the_actual_arbiter_without_changing_other_first_cycle_bids() -> None:
    model = harness()
    owner = cast(Any, model.orchestrator)
    with patch.object(
        owner._workspace_arbiter, "forward", wraps=owner._workspace_arbiter.forward
    ) as normal:
        owner.execute("bid control", strategy="greedy")
    with patch.object(
        owner._workspace_arbiter, "forward", wraps=owner._workspace_arbiter.forward
    ) as changed:
        owner.execute_intervened(
            "bid control",
            plan=plan({"kind": "scale_module_bid", "module": 0, "factor": 10.0}),
            strategy="greedy",
        )
    original, scaled = normal.call_args_list[0].args[0], changed.call_args_list[0].args[0]
    assert torch.equal(scaled.bid[:, 0], original.bid[:, 0] * 10)
    assert torch.equal(scaled.bid[:, 1:], original.bid[:, 1:])


def test_forced_module_is_a_winner_in_every_real_cycle() -> None:
    model = harness()
    owner = cast(Any, model.orchestrator)
    with patch.object(
        owner._slot_allocator, "forward", wraps=owner._slot_allocator.forward
    ) as allocations:
        output = owner.execute_intervened(
            "forced winner",
            plan=plan({"kind": "force_module_winner", "module": 2}),
            strategy="greedy",
        )
    assert allocations.call_count == 3
    for call in allocations.call_args_list:
        assert bool(call.args[2].winner_mask[:, 2].all())
    for state in output.generation.workspace_trace:
        assert bool((state.source_module == 2).any(dim=-1).all())


def test_blocking_all_module_writes_keeps_every_real_cycle_canonically_empty() -> None:
    model = harness()
    selected = plan(*({"kind": "block_module_write", "module": module} for module in range(3)))
    result = model.orchestrator.execute_intervened(
        "write controls", plan=selected, strategy="greedy"
    )
    for state in result.generation.workspace_trace:
        assert not bool(state.occupancy.any())
        assert not bool(state.slots.any())
        assert not bool(state.write_count.any())
        assert bool((state.source_module == -1).all())


def test_invalid_graph_plan_fails_before_first_pass_and_does_not_poison_next_request() -> None:
    model = harness()
    with pytest.raises(IrisRequestError):
        model.orchestrator.execute_intervened(
            "invalid", plan=plan({"kind": "zero_slot", "slot": 2}), strategy="greedy"
        )
    assert not model.backbone.forward_prompts
    baseline = model.orchestrator.execute("fresh", strategy="greedy")
    with pytest.raises(IrisRequestError):
        model.orchestrator.execute_intervened(
            "invalid",
            plan=plan({"kind": "gaussian_noise", "sigma": 1e100, "seed": 1}),
            strategy="greedy",
        )
    recovered = model.orchestrator.execute_intervened("fresh", plan=plan(), strategy="greedy")
    for left, right in zip(
        recovered.generation.workspace_trace, baseline.generation.workspace_trace, strict=True
    ):
        assert_same(left, right)


def test_old_artifact_builder_cannot_relabel_an_intervened_output_as_baseline() -> None:
    model = harness()
    output = model.orchestrator.execute_intervened("identity", plan=plan(), strategy="greedy")
    # This exact-type shell intentionally has no runtime fields. The execution
    # type gate must reject the counterfactual before touching those fields.
    runtime = object.__new__(IrisRequestRuntime)
    with pytest.raises(IrisRequestArtifactError) as failure:
        build_request_artifact_v1(
            prompt="identity",
            strategy="greedy",
            stop_token_ids=(),
            sampling=model.iris_generator.sampling,
            sampling_seed=None,
            allow_download=False,
            runtime=runtime,
            execution=cast(IrisRequestExecutionOutput, output),
        )
    assert isinstance(failure.value.__cause__, TypeError)
    assert "exact IrisRequestExecutionOutput" in str(failure.value.__cause__)


def test_artifact_v2_also_rejects_intervened_execution_before_runtime_inspection() -> None:
    model = harness()
    output = model.orchestrator.execute_intervened("identity", plan=plan(), strategy="greedy")
    with pytest.raises(IrisRequestArtifactError) as failure:
        build_request_artifact_v2(
            prompt="identity",
            strategy="greedy",
            stop_token_ids=(),
            sampling=model.iris_generator.sampling,
            sampling_seed=None,
            allow_download=False,
            runtime=object.__new__(IrisArmRequestRuntimeV2),
            execution=cast(IrisRequestExecutionOutput, output),
        )
    assert isinstance(failure.value.__cause__, TypeError)
    assert "exact IrisRequestExecutionOutput" in str(failure.value.__cause__)


@pytest.mark.parametrize("version", [3, 4])
def test_later_baseline_artifact_versions_reject_intervened_output(
    tmp_path: Path, version: int
) -> None:
    model = harness()
    output = model.orchestrator.execute_intervened("identity", plan=plan(), strategy="greedy")
    runtime, _, kwargs = artifact_inputs(tmp_path, "D")
    kwargs["execution"] = output
    with _runtime_property_patches("D"), pytest.raises(IrisRequestArtifactError) as failure:
        if version == 3:
            build_request_artifact_v3(**kwargs)  # type: ignore[arg-type]
        else:
            build_request_artifact_v4(
                **kwargs,  # type: ignore[arg-type]
                checkpoint_transfer=_transfer(runtime),
            )
    cause: BaseException = failure.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, TypeError)
    assert "arm-D request artifact output has the wrong type" in str(cause)


def real_qwen_request() -> tuple[IrisRequestOrchestrator, Any]:
    words = {"[UNK]": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "workspace": 4}
    words.update({f"word{index}": index for index in range(5, 16)})
    backend = Tokenizer(WordLevel(words, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="[UNK]",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        lm = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
            Qwen3Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=64,
                bos_token_id=2,
                eos_token_id=1,
                tie_word_embeddings=True,
                attention_dropout=0.0,
            )
        )
        config = BackboneConfig.model_validate(
            {
                "name": "intervention-random-qwen",
                "repository": "offline/random-qwen3",
                "revision": "random-fixture-v1",
                "tokenizer_revision": "wordlevel-fixture-v1",
                "trust_remote_code": False,
                "local_files_only": True,
                "output_hidden_states": True,
                "use_cache": False,
                "dtype": "fp32",
                "tokenizer": {"use_fast": True, "padding_side": "left", "truncation_side": "left"},
                "limits": {"max_prompt_tokens": 16, "max_new_tokens": 3},
            }
        )
        with (
            patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=lm),
        ):
            backbone = BackboneAdapter(config, CPU_INFO)
        owner = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=_cognitive_config(),
            workspace_config=_workspace_config(),
            stop_token_ids=(),
            sampling=TemperatureTopPSamplingConfig(temperature=0.75, top_p=0.9),
        )
    return owner, lm


def test_genuine_tiny_qwen_has_control_parity_and_causal_second_pass_logit_change() -> None:
    owner, lm = real_qwen_request()
    initial_weights = module_state_sha256((("qwen", lm),))
    initial_rng = torch.get_rng_state().clone()
    outputs: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    original = owner.backbone.forward_embeddings

    def record(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        outputs.append(result.logits.detach().clone())
        embeddings.append(args[0].detach().clone())
        return result

    with patch.object(owner.backbone, "forward_embeddings", side_effect=record):
        baseline = owner.execute("hello workspace", strategy="greedy")
        baseline_logits, baseline_prefix = outputs[0], embeddings[0]
        outputs.clear()
        embeddings.clear()
        control = owner.execute_intervened("hello workspace", plan=plan(), strategy="greedy")
        assert torch.equal(outputs[0], baseline_logits)
        assert torch.equal(embeddings[0], baseline_prefix)
        assert torch.equal(control.generation.token_ids, baseline.generation.token_ids)
        outputs.clear()
        embeddings.clear()
        zeroed = owner.execute_intervened(
            "hello workspace", plan=plan({"kind": "zero_workspace"}), strategy="greedy"
        )
        assert outputs[0].shape == baseline_logits.shape
        assert embeddings[0].shape == baseline_prefix.shape
        assert not torch.equal(outputs[0][:, -1], baseline_logits[:, -1])
        assert not torch.equal(embeddings[0], baseline_prefix)
        assert all(not bool(state.occupancy.any()) for state in zeroed.generation.workspace_trace)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert module_state_sha256((("qwen", lm),)) == initial_weights
    assert owner.backbone.trainable_parameter_count == 0
    assert not owner.backbone.has_parameter_gradients


def test_real_sampling_control_binds_initial_rng_state_not_just_seed() -> None:
    owner, _ = real_qwen_request()
    before_global = torch.get_rng_state().clone()
    baseline = owner.execute(
        "hello workspace", strategy="sampling", generator=torch.Generator().manual_seed(42)
    )
    generator = torch.Generator().manual_seed(42)
    expected_hash = tensor_sha256(
        generator.get_state(), domain="iris/intervened-request/sampling-rng/v1"
    )
    control = owner.execute_intervened(
        "hello workspace", plan=plan(), strategy="sampling", generator=generator
    )
    assert torch.equal(control.generation.token_ids, baseline.generation.token_ids)
    assert control.receipt.sampling_seed == 42
    assert control.receipt.sampling_rng_state_sha256 == expected_hash
    advanced = torch.Generator().manual_seed(42)
    torch.rand(4, generator=advanced)
    other = owner.execute_intervened(
        "hello workspace", plan=plan(), strategy="sampling", generator=advanced
    )
    assert other.receipt.sampling_seed == 42
    assert other.receipt.sampling_rng_state_sha256 != control.receipt.sampling_rng_state_sha256
    assert torch.equal(before_global, torch.get_rng_state())


@pytest.mark.parametrize("when", ["before", "during"])
def test_sidecar_weight_drift_prevents_publishing_an_intervened_result(when: str) -> None:
    model = harness()
    owner = cast(Any, model.orchestrator)
    weight = next(owner._workspace_reader.parameters())
    original_weight = weight.detach().clone()
    original = model.iris_generator.generate

    def drift(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        with torch.no_grad():
            weight.add_(0.125)
        return result

    try:
        if when == "before":
            with torch.no_grad():
                weight.add_(0.125)
        with (
            patch.object(model.iris_generator, "generate", side_effect=drift),
            pytest.raises(IrisRequestError) as failure,
        ):
            owner.execute_intervened("weight drift", plan=plan(), strategy="greedy")
        assert "sidecar state changed" in str(failure.value.__cause__).lower()
        if when == "before":
            assert not model.backbone.forward_prompts
    finally:
        with torch.no_grad():
            weight.copy_(original_weight)
    # Test-injected damage is explicitly restored; no automatic rollback is claimed.
    recovered = owner.execute_intervened("weight drift", plan=plan(), strategy="greedy")
    assert recovered.receipt.sidecar_state_sha256 == owner.sidecar_state_sha256


@pytest.mark.parametrize("field", ["sampling", "stop_token_ids"])
def test_decoding_policy_drift_cannot_publish_a_mislabeled_receipt(field: str) -> None:
    model = harness()
    original = model.iris_generator.generate

    def drift(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if field == "sampling":
            # Mutate the originally frozen object to prove receipt storage is owned.
            object.__setattr__(model.iris_generator.sampling, "top_p", 0.5)
        else:
            model.iris_generator.stop_token_ids = (15,)
        return result

    with (
        patch.object(model.iris_generator, "generate", side_effect=drift),
        pytest.raises(IrisRequestError) as failure,
    ):
        model.orchestrator.execute_intervened("policy drift", plan=plan(), strategy="greedy")
    assert "Decoding policy changed" in str(failure.value.__cause__)


def test_sampling_receipt_owns_its_policy_snapshot() -> None:
    model = harness()
    output = model.orchestrator.execute_intervened("owned policy", plan=plan(), strategy="greedy")
    assert output.receipt.sampling is not model.iris_generator.sampling
    object.__setattr__(model.iris_generator.sampling, "temperature", 0.25)
    assert output.receipt.sampling.temperature == 1.0


@pytest.mark.parametrize("mutation", ["reseed", "advance"])
def test_foreign_rng_mutation_is_rejected_after_genuine_qwen_execution(mutation: str) -> None:
    owner, _ = real_qwen_request()
    generator = torch.Generator().manual_seed(42)
    original = owner.backbone.forward_embeddings

    def reseed(*args: Any, **kwargs: Any) -> Any:
        if mutation == "reseed":
            generator.manual_seed(1337)
        else:
            torch.rand(4, generator=generator)
        return original(*args, **kwargs)

    with (
        patch.object(owner.backbone, "forward_embeddings", side_effect=reseed),
        pytest.raises(IrisRequestError) as failure,
    ):
        owner.execute_intervened(
            "hello workspace", plan=plan(), strategy="sampling", generator=generator
        )
    assert "Caller sampling RNG changed" in str(failure.value.__cause__)


def test_post_generation_validation_failure_does_not_commit_caller_sampling_rng() -> None:
    owner, _ = real_qwen_request()
    generator = torch.Generator().manual_seed(42)
    initial = generator.get_state().clone()
    internal = cast(Any, owner)
    original = internal._iris_generator.generate

    def drift(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        next(internal._workspace_reader.parameters()).add_(0.125)
        return output

    with (
        patch.object(internal._iris_generator, "generate", side_effect=drift),
        pytest.raises(IrisRequestError),
    ):
        owner.execute_intervened(
            "hello workspace", plan=plan(), strategy="sampling", generator=generator
        )
    assert torch.equal(generator.get_state(), initial)


def test_success_commits_exact_same_sampling_rng_as_baseline() -> None:
    owner, _ = real_qwen_request()
    baseline_rng = torch.Generator().manual_seed(42)
    intervened_rng = torch.Generator().manual_seed(42)
    initial = intervened_rng.get_state().clone()
    owner.execute("hello workspace", strategy="sampling", generator=baseline_rng)
    owner.execute_intervened(
        "hello workspace", plan=plan(), strategy="sampling", generator=intervened_rng
    )
    assert torch.equal(baseline_rng.get_state(), intervened_rng.get_state())
    assert not torch.equal(initial, intervened_rng.get_state())


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "baseline"),
        ("semantics", "unknown"),
        ("cycles", True),
        ("cycles", 0),
        ("intervention_plan_sha256", "x" * 64),
        ("graph_sha256", "A" * 64),
        ("sidecar_state_sha256", 1),
        ("sampling_seed", 1),
        ("sampling_rng_state_sha256", "a" * 64),
        ("stop_token_ids", (True,)),
    ],
)
def test_receipt_rejects_ambiguous_provenance(field: str, value: Any) -> None:
    receipt = IrisIntervenedRequestExecutionReceiptV1(
        prompt_sha256="a" * 64,
        strategy="greedy",
        stop_token_ids=(),
        sampling=TemperatureTopPSamplingConfig(),
        sampling_seed=None,
        sampling_rng_state_sha256=None,
        intervention_plan_sha256="b" * 64,
        graph_sha256="c" * 64,
        sidecar_state_sha256="d" * 64,
        cycles=3,
    )
    with pytest.raises((TypeError, ValueError)):
        replace(receipt, **{field: value})
