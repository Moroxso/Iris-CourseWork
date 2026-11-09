import inspect
from typing import Any, cast

import pytest
import torch

import iris.application as application
import iris.application.shared_buffer_training_composition as composition
from iris.application.errors import IrisTeacherCompositionError
from iris.application.shared_buffer_request import build_shared_buffer_request_orchestrator
from iris.application.shared_buffer_training_composition import (
    build_shared_buffer_teacher_forcing_model,
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
    SecondPassSequenceAssembler,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig
from iris.shared_buffer import (
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferWriter,
    build_shared_buffer_policy_v1,
    shared_buffer_policy_sha256_v1,
)
from iris.training.shared_buffer_model import IrisSharedBufferTeacherForcingModel

CPU = torch.device("cpu")
COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "shared_buffer_reader",
    "cognitive_modules",
    "proposal_head",
    "shared_buffer_writer",
    "shared_buffer_latent_bridge",
    "sequence_assembler",
)
SHARED_REQUEST_COMPONENT_NAMES = COMPONENT_NAMES[:-1]


class _FrozenBackbone(BackboneAdapter):
    def __init__(
        self,
        *,
        max_position_embeddings: int = 10,
        trainable_parameter_count: int = 0,
        has_parameter_gradients: bool = False,
    ) -> None:
        self._max_context = max_position_embeddings
        self._trainable = trainable_parameter_count
        self._gradients = has_parameter_gradients
        self.callbacks = 0

    @property
    def hidden_size(self) -> int:
        return 6

    @property
    def num_hidden_layers(self) -> int:
        return 4

    @property
    def vocabulary_size(self) -> int:
        return 16

    @property
    def max_position_embeddings(self) -> int:
        return self._max_context

    @property
    def max_prompt_tokens(self) -> int:
        return 5

    @property
    def max_new_tokens(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return CPU

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def trainable_parameter_count(self) -> int:
        return self._trainable

    @property
    def has_parameter_gradients(self) -> bool:
        return self._gradients

    def forward_prompt(self, text: str) -> Any:
        self.callbacks += 1
        raise AssertionError(f"composition executed prompt callback for {text!r}")

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.callbacks += 1
        raise AssertionError("composition executed embedding callback")

    def forward_embeddings(self, *args: object, **kwargs: object) -> Any:
        self.callbacks += 1
        raise AssertionError("composition executed causal callback")


def _config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "arm-c-teacher-composition",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def _build(
    backbone: BackboneAdapter | None = None,
) -> IrisSharedBufferTeacherForcingModel:
    return build_shared_buffer_teacher_forcing_model(
        backbone=backbone or _FrozenBackbone(),
        cognitive_config=_config(),
    )


def test_factory_builds_exact_policy_bound_nine_component_teacher_graph() -> None:
    backbone = _FrozenBackbone()
    model = _build(backbone)

    assert (
        application.build_shared_buffer_teacher_forcing_model
        is build_shared_buffer_teacher_forcing_model
    )
    assert type(model) is IrisSharedBufferTeacherForcingModel
    assert model.backbone is backbone
    assert backbone.callbacks == 0
    assert tuple(model._modules) == COMPONENT_NAMES
    assert tuple(name for name, _ in model._sidecar_modules) == COMPONENT_NAMES
    expected_types = (
        LayerSampler,
        GoalEncoder,
        ModuleInputRouter,
        SharedBufferReader,
        CognitiveModule,
        ProposalHead,
        SharedBufferWriter,
        SharedBufferLatentBridge,
        SecondPassSequenceAssembler,
    )
    for name, expected_type in zip(COMPONENT_NAMES, expected_types, strict=True):
        assert type(getattr(model, name)) is expected_type

    assert model.cycles == 3
    assert model.shared_buffer_policy_sha256 == shared_buffer_policy_sha256_v1()
    assert model._policy == build_shared_buffer_policy_v1()
    assert model.training
    assert all(module.training for module in model.modules())
    assert "_backbone" not in model._modules
    assert not hasattr(model, "workspace_arbiter")
    assert not hasattr(model, "slot_allocator")
    assert not hasattr(model, "workspace_decay")

    parameters = tuple(model.parameters())
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(parameter.device == CPU for parameter in parameters)
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert all(not parameter.is_inference() for parameter in parameters)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    assert model.shared_buffer_reader.max_entries == 9
    assert model.shared_buffer_writer.max_entries == 9
    assert model.shared_buffer_latent_bridge.final_entry_count == 9
    assert model.sequence_assembler.num_slots == 3
    assert model.sequence_assembler.max_context_tokens == 10


def test_factory_uses_only_cognitive_config_and_token_matched_m_context() -> None:
    signature = inspect.signature(build_shared_buffer_teacher_forcing_model)
    assert tuple(signature.parameters) == ("backbone", "cognitive_config")

    # 5 prompt + M=3 latents + 2 targets exactly fills the context. A 3M
    # second-pass prefix would reject this otherwise valid composition.
    model = _build(_FrozenBackbone(max_position_embeddings=10))
    assert model.sequence_assembler.num_slots == 3
    assert model.sequence_assembler.max_context_tokens == 10


def test_same_seed_matches_request_sidecar_parameter_initialization() -> None:
    sampling = TemperatureTopPSamplingConfig(temperature=0.7, top_p=0.9)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        teacher = _build()
        teacher_components = dict(teacher._sidecar_modules)

        torch.manual_seed(456)
        request = build_shared_buffer_request_orchestrator(
            backbone=_FrozenBackbone(),
            cognitive_config=_config(),
            stop_token_ids=(1,),
            sampling=sampling,
        )
        request_components = dict(request._sidecar_modules)

    for name in SHARED_REQUEST_COMPONENT_NAMES:
        teacher_state = teacher_components[name].state_dict()
        request_state = request_components[name].state_dict()
        assert tuple(teacher_state) == tuple(request_state)
        assert all(torch.equal(teacher_state[key], request_state[key]) for key in teacher_state)
    assert type(teacher_components["sequence_assembler"]) is SecondPassSequenceAssembler
    assert request_components["prefix_assembler"].state_dict() == {}


def test_same_seed_reproduces_teacher_state_and_different_seed_changes_it() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77)
        first = _build().state_dict()
        torch.manual_seed(77)
        second = _build().state_dict()
        torch.manual_seed(78)
        third = _build().state_dict()

    assert tuple(first) == tuple(second) == tuple(third)
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], third[name]) for name in first)


def test_factory_materializes_ordinary_parameters_under_inference_mode() -> None:
    with torch.inference_mode():
        model = _build()
    assert all(not parameter.is_inference() for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    ("backbone", "match"),
    [
        (_FrozenBackbone(max_position_embeddings=9), "context"),
        (_FrozenBackbone(trainable_parameter_count=1), "frozen"),
        (_FrozenBackbone(has_parameter_gradients=True), "gradients"),
    ],
)
def test_factory_rejects_invalid_frozen_backbone_boundary(
    backbone: BackboneAdapter,
    match: str,
) -> None:
    with pytest.raises(IrisTeacherCompositionError, match="preflight validation") as captured:
        _build(backbone)
    assert captured.value.__cause__ is not None
    assert match in str(captured.value.__cause__)


def test_factory_rejects_non_exact_cognitive_config() -> None:
    with pytest.raises(IrisTeacherCompositionError, match="preflight validation") as captured:
        build_shared_buffer_teacher_forcing_model(
            backbone=_FrozenBackbone(),
            cognitive_config=cast(CognitiveConfig, object()),
        )
    assert captured.value.__cause__ is not None
    assert "exactly CognitiveConfig" in str(captured.value.__cause__)


def test_constructor_failure_is_typed_and_preserves_exact_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = RuntimeError("arm-c constructor sentinel")

    def reject_constructor(**_: object) -> IrisSharedBufferTeacherForcingModel:
        raise sentinel

    monkeypatch.setattr(
        composition,
        "IrisSharedBufferTeacherForcingModel",
        reject_constructor,
    )
    with pytest.raises(IrisTeacherCompositionError, match="model wiring") as captured:
        _build()
    assert captured.value.__cause__ is sentinel
