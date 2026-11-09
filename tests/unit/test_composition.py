"""Contract tests for validated Iris request-graph composition."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import iris.application as application
import iris.application.composition as composition
from iris.application import (
    IrisCompositionError,
    IrisRequestError,
    IrisRequestOrchestrator,
    build_request_orchestrator,
)
from iris.backbone import BackboneAdapter
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import (
    SecondPassGenerationPrefixAssembler,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.workspace import (
    LatentBridge,
    SlotAllocator,
    WorkspaceArbiter,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceWriter,
)

CPU = torch.device("cpu")
BACKBONE_HIDDEN_SIZE = 6
NUM_HIDDEN_LAYERS = 4
VOCABULARY_SIZE = 17
MAX_PROMPT_TOKENS = 5
MAX_NEW_TOKENS = 2
MAX_CONTEXT_TOKENS = 9
STATE_DIM = 4
NUM_MODULES = 3
NUM_SLOTS = 2
CYCLES = 3
WINNERS_PER_CYCLE = 2
LAYER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
STOP_TOKEN_IDS = (2, 7)


class OfflineMetadataBackbone(BackboneAdapter):
    """Already-loaded adapter double exposing only stable public metadata."""

    def __init__(
        self,
        *,
        hidden_size: object = BACKBONE_HIDDEN_SIZE,
        num_hidden_layers: object = NUM_HIDDEN_LAYERS,
        vocabulary_size: object = VOCABULARY_SIZE,
        max_position_embeddings: object = MAX_CONTEXT_TOKENS,
        max_prompt_tokens: object = MAX_PROMPT_TOKENS,
        max_new_tokens: object = MAX_NEW_TOKENS,
        device: object = CPU,
        dtype: object = torch.float32,
    ) -> None:
        self._reported: dict[str, object] = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "vocabulary_size": vocabulary_size,
            "max_position_embeddings": max_position_embeddings,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "dtype": dtype,
        }
        self.read_counts: dict[str, int] = {}
        self.getter_errors: dict[str, Exception] = {}
        self.on_read: Any = None
        self.eos_reads = 0

    def _read(self, name: str) -> object:
        count = self.read_counts.get(name, 0) + 1
        self.read_counts[name] = count
        error = self.getter_errors.get(name)
        if error is not None:
            raise error
        if self.on_read is not None:
            self.on_read(name, count)
        return self._reported[name]

    @property
    def hidden_size(self) -> int:
        return cast(int, self._read("hidden_size"))

    @property
    def num_hidden_layers(self) -> int:
        return cast(int, self._read("num_hidden_layers"))

    @property
    def vocabulary_size(self) -> int:
        return cast(int, self._read("vocabulary_size"))

    @property
    def max_position_embeddings(self) -> int:
        return cast(int, self._read("max_position_embeddings"))

    @property
    def max_prompt_tokens(self) -> int:
        return cast(int, self._read("max_prompt_tokens"))

    @property
    def max_new_tokens(self) -> int:
        return cast(int, self._read("max_new_tokens"))

    @property
    def device(self) -> torch.device:
        return cast(torch.device, self._read("device"))

    @property
    def dtype(self) -> torch.dtype:
        return cast(torch.dtype, self._read("dtype"))

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        self.eos_reads += 1
        raise AssertionError("composition must not add implicit EOS stop IDs")


class ThrowingCognitiveConfig(CognitiveConfig):
    """Validated schema whose scalar getter can fail after type validation."""

    state_dim_failure: ClassVar[Exception | None] = None

    def __getattribute__(self, name: str) -> Any:
        if name == "state_dim":
            failure = type(self).state_dim_failure
            if failure is not None:
                raise failure
        return super().__getattribute__(name)


def cognitive_config(*, num_modules: int = NUM_MODULES) -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "tiny-composition-cognitive",
            "state_dim": STATE_DIM,
            "num_modules": num_modules,
            "layer_sampler": {"layer_fractions": LAYER_FRACTIONS},
        }
    )


def workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "tiny-composition-workspace",
            "num_slots": NUM_SLOTS,
            "cycles": CYCLES,
            "winners_per_cycle": WINNERS_PER_CYCLE,
            "decay_rate": 0.5,
            "empty_threshold": 0.01,
            "arbiter": {
                "confidence_weight": 0.25,
                "novelty_weight": 0.1,
                "relevance_weight": 0.25,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def sampling_config() -> TemperatureTopPSamplingConfig:
    return TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)


def build_graph(
    backbone: BackboneAdapter,
    *,
    cognitive: CognitiveConfig | None = None,
    workspace: WorkspaceConfig | None = None,
    stop_token_ids: tuple[int, ...] = STOP_TOKEN_IDS,
    sampling: TemperatureTopPSamplingConfig | None = None,
) -> IrisRequestOrchestrator:
    return build_request_orchestrator(
        backbone=backbone,
        cognitive_config=cognitive if cognitive is not None else cognitive_config(),
        workspace_config=workspace if workspace is not None else workspace_config(),
        stop_token_ids=stop_token_ids,
        sampling=sampling if sampling is not None else sampling_config(),
    )


def sidecar_modules(
    orchestrator: IrisRequestOrchestrator,
) -> tuple[tuple[str, nn.Module], ...]:
    instance = cast(Any, orchestrator)
    return (
        ("layer_sampler", instance._layer_sampler),
        ("goal_encoder", instance._goal_encoder),
        ("module_input_router", instance._module_input_router),
        ("workspace_reader", instance._workspace_reader),
        ("cognitive_modules", instance._cognitive_modules),
        ("proposal_head", instance._proposal_head),
        ("workspace_arbiter", instance._workspace_arbiter),
        ("slot_allocator", instance._slot_allocator),
        ("workspace_writer", instance._workspace_writer),
        ("workspace_decay", instance._workspace_decay),
        ("latent_bridge", instance._latent_bridge),
        ("prefix_assembler", instance._prefix_assembler),
    )


def sidecar_state(orchestrator: IrisRequestOrchestrator) -> dict[str, torch.Tensor]:
    return {
        f"{module_name}.{state_name}": tensor.detach().clone()
        for module_name, module in sidecar_modules(orchestrator)
        for state_name, tensor in module.state_dict().items()
    }


def test_factory_maps_every_real_constructor_and_preserves_exact_identities() -> None:
    backbone = OfflineMetadataBackbone()
    cognitive = cognitive_config()
    workspace = workspace_config()
    sampling = sampling_config()
    constructors: dict[str, type[Any]] = {
        "LayerSampler": LayerSampler,
        "GoalEncoder": GoalEncoder,
        "ModuleInputRouter": ModuleInputRouter,
        "WorkspaceReader": WorkspaceReader,
        "CognitiveModule": CognitiveModule,
        "ProposalHead": ProposalHead,
        "WorkspaceArbiter": WorkspaceArbiter,
        "SlotAllocator": SlotAllocator,
        "WorkspaceWriter": WorkspaceWriter,
        "WorkspaceDecay": WorkspaceDecay,
        "LatentBridge": LatentBridge,
        "SecondPassGenerationPrefixAssembler": SecondPassGenerationPrefixAssembler,
        "IrisRequestOrchestrator": IrisRequestOrchestrator,
    }

    with ExitStack() as stack:
        spies = {
            name: cast(
                MagicMock,
                stack.enter_context(patch.object(composition, name, wraps=constructor)),
            )
            for name, constructor in constructors.items()
        }
        orchestrator = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=cognitive,
            workspace_config=workspace,
            stop_token_ids=STOP_TOKEN_IDS,
            sampling=sampling,
        )

    assert isinstance(orchestrator, IrisRequestOrchestrator)
    spies["LayerSampler"].assert_called_once_with(
        cognitive.layer_sampler,
        backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
        state_dim=STATE_DIM,
    )
    spies["GoalEncoder"].assert_called_once_with(state_dim=STATE_DIM)
    spies["ModuleInputRouter"].assert_called_once_with(
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
    )
    spies["WorkspaceReader"].assert_called_once_with(
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
        num_slots=NUM_SLOTS,
    )
    spies["CognitiveModule"].assert_called_once_with(
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
    )
    spies["ProposalHead"].assert_called_once_with(
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
        cycles=CYCLES,
    )
    spies["WorkspaceArbiter"].assert_called_once_with(
        workspace.arbiter,
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
        num_slots=NUM_SLOTS,
        winners_per_cycle=WINNERS_PER_CYCLE,
    )
    spies["SlotAllocator"].assert_called_once_with(
        workspace.allocator,
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
        num_slots=NUM_SLOTS,
        winners_per_cycle=WINNERS_PER_CYCLE,
    )
    spies["WorkspaceWriter"].assert_called_once_with(
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
        num_slots=NUM_SLOTS,
        winners_per_cycle=WINNERS_PER_CYCLE,
    )
    spies["WorkspaceDecay"].assert_called_once_with(
        state_dim=STATE_DIM,
        num_modules=NUM_MODULES,
        num_slots=NUM_SLOTS,
        decay_rate=workspace.decay_rate,
        empty_threshold=workspace.empty_threshold,
    )
    spies["LatentBridge"].assert_called_once_with(
        state_dim=STATE_DIM,
        backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
        num_slots=NUM_SLOTS,
    )
    spies["SecondPassGenerationPrefixAssembler"].assert_called_once_with(
        hidden_size=BACKBONE_HIDDEN_SIZE,
        num_slots=NUM_SLOTS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    wiring = spies["IrisRequestOrchestrator"].call_args.kwargs
    assert wiring["backbone"] is backbone
    assert wiring["cognitive_config"] is cognitive
    assert wiring["workspace_config"] is workspace
    assert wiring["iris_generator"] is cast(Any, orchestrator)._iris_generator
    assert wiring["iris_generator"].backbone is backbone
    assert wiring["iris_generator"].max_new_tokens == MAX_NEW_TOKENS
    assert wiring["iris_generator"].stop_token_ids is STOP_TOKEN_IDS
    assert wiring["iris_generator"].sampling is sampling
    for name, module in sidecar_modules(orchestrator):
        assert wiring[name] is module


def test_factory_places_real_sidecar_in_eval_with_trainable_ordinary_parameters() -> None:
    backbone = OfflineMetadataBackbone()

    with torch.inference_mode():
        orchestrator = build_graph(backbone)

    expected_types: tuple[type[nn.Module], ...] = (
        LayerSampler,
        GoalEncoder,
        ModuleInputRouter,
        WorkspaceReader,
        CognitiveModule,
        ProposalHead,
        WorkspaceArbiter,
        SlotAllocator,
        WorkspaceWriter,
        WorkspaceDecay,
        LatentBridge,
        SecondPassGenerationPrefixAssembler,
    )
    modules = sidecar_modules(orchestrator)
    assert tuple(type(module) for _, module in modules) == expected_types

    parameter_count = 0
    for _, module in modules:
        assert all(child.training is False for child in module.modules())
        for parameter in module.parameters():
            parameter_count += parameter.numel()
            assert parameter.device == CPU
            assert parameter.dtype == torch.float32
            assert parameter.requires_grad
            assert not parameter.is_inference()
            assert bool(torch.isfinite(parameter).all())
        for buffer in module.buffers():
            assert buffer.device == CPU
            assert not buffer.is_inference()
            if torch.is_floating_point(buffer):
                assert buffer.dtype == torch.float32
                assert bool(torch.isfinite(buffer).all())
    assert parameter_count > 0
    assert cast(Any, orchestrator)._backbone is backbone
    assert cast(Any, orchestrator)._iris_generator.backbone is backbone


def test_factory_is_offline_does_not_execute_backbone_and_does_not_read_eos() -> None:
    backbone = OfflineMetadataBackbone()
    cognitive = cognitive_config()
    workspace = workspace_config()
    cognitive_snapshot = cognitive.model_dump_json()
    workspace_snapshot = workspace.model_dump_json()

    with (
        patch("iris.backbone.adapter.AutoTokenizer.from_pretrained") as tokenizer_loader,
        patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained") as model_loader,
        patch.object(BackboneAdapter, "forward_prompt") as first_pass,
        patch.object(BackboneAdapter, "embed_tokens") as embed_tokens,
        patch.object(BackboneAdapter, "decode_tokens") as decode_tokens,
        patch.object(BackboneAdapter, "forward_embeddings") as causal_forward,
    ):
        orchestrator = build_graph(
            backbone,
            cognitive=cognitive,
            workspace=workspace,
        )

    assert isinstance(orchestrator, IrisRequestOrchestrator)
    tokenizer_loader.assert_not_called()
    model_loader.assert_not_called()
    first_pass.assert_not_called()
    embed_tokens.assert_not_called()
    decode_tokens.assert_not_called()
    causal_forward.assert_not_called()
    assert backbone.eos_reads == 0
    assert cognitive.model_dump_json() == cognitive_snapshot
    assert workspace.model_dump_json() == workspace_snapshot


@pytest.mark.parametrize(
    "case",
    [
        "backbone_type",
        "cognitive_type",
        "workspace_type",
        "sampling_type",
        "stop_container",
        "stop_bool",
        "stop_duplicate",
        "stop_oov",
        "winner_capacity",
        "hidden_size_bool",
        "invalid_sampled_depths",
        "device_type",
        "cuda_indexless",
        "dtype_value",
        "context_capacity",
        "mutated_infinite_threshold",
        "constructed_infinite_threshold",
    ],
)
def test_invalid_preflight_is_rng_neutral_and_constructs_no_sidecar(case: str) -> None:
    kwargs: dict[str, Any] = {
        "backbone": OfflineMetadataBackbone(),
        "cognitive_config": cognitive_config(),
        "workspace_config": workspace_config(),
        "stop_token_ids": STOP_TOKEN_IDS,
        "sampling": sampling_config(),
    }
    if case == "backbone_type":
        kwargs["backbone"] = object()
    elif case == "cognitive_type":
        kwargs["cognitive_config"] = {"state_dim": STATE_DIM}
    elif case == "workspace_type":
        kwargs["workspace_config"] = {"num_slots": NUM_SLOTS}
    elif case == "sampling_type":
        kwargs["sampling"] = object()
    elif case == "stop_container":
        kwargs["stop_token_ids"] = list(STOP_TOKEN_IDS)
    elif case == "stop_bool":
        kwargs["stop_token_ids"] = (True,)
    elif case == "stop_duplicate":
        kwargs["stop_token_ids"] = (2, 2)
    elif case == "stop_oov":
        kwargs["stop_token_ids"] = (VOCABULARY_SIZE,)
    elif case == "winner_capacity":
        kwargs["cognitive_config"] = cognitive_config(num_modules=1)
    elif case == "hidden_size_bool":
        kwargs["backbone"] = OfflineMetadataBackbone(hidden_size=True)
    elif case == "invalid_sampled_depths":
        kwargs["backbone"] = OfflineMetadataBackbone(num_hidden_layers=2)
    elif case == "device_type":
        kwargs["backbone"] = OfflineMetadataBackbone(device="cpu")
    elif case == "cuda_indexless":
        kwargs["backbone"] = OfflineMetadataBackbone(device=torch.device("cuda"))
    elif case == "dtype_value":
        kwargs["backbone"] = OfflineMetadataBackbone(dtype=torch.float64)
    elif case == "context_capacity":
        kwargs["backbone"] = OfflineMetadataBackbone(max_position_embeddings=8)
    elif case == "mutated_infinite_threshold":
        mutated_workspace = workspace_config()
        mutated_workspace.empty_threshold = float("inf")
        kwargs["workspace_config"] = mutated_workspace
    else:
        assert case == "constructed_infinite_threshold"
        valid_workspace = workspace_config()
        kwargs["workspace_config"] = WorkspaceConfig.model_construct(
            name=valid_workspace.name,
            num_slots=valid_workspace.num_slots,
            cycles=valid_workspace.cycles,
            winners_per_cycle=valid_workspace.winners_per_cycle,
            decay_rate=valid_workspace.decay_rate,
            empty_threshold=float("inf"),
            arbiter=valid_workspace.arbiter,
            allocator=valid_workspace.allocator,
        )

    with patch.object(composition, "LayerSampler", wraps=LayerSampler) as constructor:
        rng_before = torch.get_rng_state().clone()
        with pytest.raises(IrisCompositionError, match="preflight validation") as raised:
            build_request_orchestrator(**kwargs)

        assert torch.equal(torch.get_rng_state(), rng_before)
        constructor.assert_not_called()
    assert raised.value.__cause__ is not None


def test_throwing_backbone_metadata_is_chained_without_sidecar_construction() -> None:
    backbone = OfflineMetadataBackbone()
    failure = RuntimeError("metadata exploded")
    backbone.getter_errors["hidden_size"] = failure

    with patch.object(composition, "LayerSampler", wraps=LayerSampler) as constructor:
        with pytest.raises(IrisCompositionError, match="preflight validation") as raised:
            build_graph(backbone)

        constructor.assert_not_called()
    assert raised.value.__cause__ is failure


def test_throwing_validated_config_getter_is_chained_during_preflight() -> None:
    config = ThrowingCognitiveConfig.model_validate(cognitive_config().model_dump())
    failure = RuntimeError("validated scalar getter exploded")
    ThrowingCognitiveConfig.state_dim_failure = failure
    try:
        with (
            patch.object(composition, "LayerSampler", wraps=LayerSampler) as constructor,
            pytest.raises(IrisCompositionError, match="preflight validation") as raised,
        ):
            build_graph(OfflineMetadataBackbone(), cognitive=config)
    finally:
        ThrowingCognitiveConfig.state_dim_failure = None

    constructor.assert_not_called()
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    ("target", "stage"),
    [
        ("LayerSampler", "LayerSampler construction and placement"),
        ("IrisGenerator", "IrisGenerator construction"),
        ("IrisRequestOrchestrator", "IrisRequestOrchestrator wiring"),
    ],
)
def test_construction_failures_are_chained_at_the_exact_stage(target: str, stage: str) -> None:
    failure = RuntimeError(f"{target} exploded")

    with (
        patch.object(composition, target, side_effect=failure),
        pytest.raises(IrisCompositionError, match=stage) as raised,
    ):
        build_graph(OfflineMetadataBackbone())

    assert raised.value.__cause__ is failure


def test_placement_failure_is_chained_and_stops_later_construction() -> None:
    failure = RuntimeError("placement exploded")

    with (
        patch.object(LayerSampler, "to", side_effect=failure),
        patch.object(composition, "GoalEncoder", wraps=GoalEncoder) as later_constructor,
        pytest.raises(
            IrisCompositionError,
            match="LayerSampler construction and placement",
        ) as raised,
    ):
        build_graph(OfflineMetadataBackbone())

    assert raised.value.__cause__ is failure
    later_constructor.assert_not_called()


def test_backbone_metadata_drift_is_rejected_after_placement() -> None:
    backbone = OfflineMetadataBackbone()

    def drift_hidden_size(name: str, count: int) -> None:
        if name == "hidden_size" and count == 2:
            backbone._reported["hidden_size"] = BACKBONE_HIDDEN_SIZE + 1

    backbone.on_read = drift_hidden_size

    with pytest.raises(IrisCompositionError, match="post-placement validation") as raised:
        build_graph(backbone)

    assert isinstance(raised.value.__cause__, ValueError)
    assert "metadata changed" in str(raised.value.__cause__)


def test_config_mutation_by_backbone_metadata_getter_is_rejected_in_preflight() -> None:
    backbone = OfflineMetadataBackbone()
    cognitive = cognitive_config()

    def mutate_config(name: str, count: int) -> None:
        if name == "hidden_size" and count == 1:
            cognitive.name = "mutated-by-getter"

    backbone.on_read = mutate_config

    with pytest.raises(IrisCompositionError, match="preflight validation") as raised:
        build_graph(backbone, cognitive=cognitive)

    assert isinstance(raised.value.__cause__, ValueError)
    assert "mutated cognitive_config" in str(raised.value.__cause__)


def test_config_drift_during_component_construction_is_rejected_post_placement() -> None:
    cognitive = cognitive_config()
    real_constructor = GoalEncoder

    def mutate_then_construct(*, state_dim: int) -> GoalEncoder:
        cognitive.name = "mutated-by-constructor"
        return real_constructor(state_dim=state_dim)

    with (
        patch.object(composition, "GoalEncoder", side_effect=mutate_then_construct),
        pytest.raises(IrisCompositionError, match="post-placement validation") as raised,
    ):
        build_graph(OfflineMetadataBackbone(), cognitive=cognitive)

    assert isinstance(raised.value.__cause__, ValueError)
    assert "cognitive_config changed" in str(raised.value.__cause__)


def test_caller_seed_fully_determines_sidecar_initialization_without_hidden_reseed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        first = sidecar_state(build_graph(OfflineMetadataBackbone()))
        torch.manual_seed(1729)
        second = sidecar_state(build_graph(OfflineMetadataBackbone()))
        torch.manual_seed(1730)
        different = sidecar_state(build_graph(OfflineMetadataBackbone()))

    assert first.keys() == second.keys() == different.keys()
    assert first
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], different[name]) for name in first)


def test_sidecar_fingerprint_is_seeded_stable_and_detects_post_composition_mutation() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        first = build_graph(OfflineMetadataBackbone())
        torch.manual_seed(1729)
        second = build_graph(OfflineMetadataBackbone())
        torch.manual_seed(1730)
        different = build_graph(OfflineMetadataBackbone())

    assert len(first.sidecar_state_sha256) == 64
    assert first.sidecar_state_sha256 == second.sidecar_state_sha256
    assert first.sidecar_state_sha256 != different.sidecar_state_sha256

    mutable_module = next(
        module
        for _, module in sidecar_modules(first)
        if next(module.parameters(), None) is not None
    )
    parameter = next(mutable_module.parameters())
    with torch.no_grad():
        parameter.view(-1)[0].add_(1.0)

    with pytest.raises(IrisRequestError, match="state changed"):
        _ = first.sidecar_state_sha256


def test_factory_and_error_are_canonical_public_application_exports() -> None:
    assert application.build_request_orchestrator is build_request_orchestrator
    assert application.IrisCompositionError is IrisCompositionError
    assert "build_request_orchestrator" in application.__all__
    assert "IrisCompositionError" in application.__all__
