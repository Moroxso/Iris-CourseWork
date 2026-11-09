from typing import Any, cast

import pytest
import torch

import iris.application as application
from iris.application.errors import IrisTeacherCompositionError
from iris.application.no_workspace_request import build_no_workspace_request_orchestrator
from iris.application.no_workspace_training_composition import (
    build_no_workspace_teacher_forcing_model,
)
from iris.backbone import BackboneAdapter
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ModuleLatentBridge,
)
from iris.generation import (
    ModuleSecondPassSequenceAssembler,
    TemperatureTopPSamplingConfig,
)
from iris.schemas import CognitiveConfig
from iris.training import IrisNoWorkspaceTeacherForcingModel

CPU = torch.device("cpu")
COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "cognitive_modules",
    "module_latent_bridge",
    "sequence_assembler",
)


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
            "name": "arm-b-teacher-composition",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def _build(
    backbone: BackboneAdapter | None = None,
) -> IrisNoWorkspaceTeacherForcingModel:
    return build_no_workspace_teacher_forcing_model(
        backbone=backbone or _FrozenBackbone(),
        cognitive_config=_config(),
    )


def test_public_factory_builds_exact_no_workspace_teacher_graph() -> None:
    backbone = _FrozenBackbone()
    model = _build(backbone)
    assert application.build_no_workspace_teacher_forcing_model is (
        build_no_workspace_teacher_forcing_model
    )
    assert "build_no_workspace_teacher_forcing_model" in application.__all__
    assert type(model) is IrisNoWorkspaceTeacherForcingModel
    assert model.backbone is backbone
    assert backbone.callbacks == 0
    assert tuple(model._modules) == COMPONENT_NAMES
    expected_types = (
        LayerSampler,
        GoalEncoder,
        ModuleInputRouter,
        CognitiveModule,
        ModuleLatentBridge,
        ModuleSecondPassSequenceAssembler,
    )
    for name, expected_type in zip(COMPONENT_NAMES, expected_types, strict=True):
        assert type(getattr(model, name)) is expected_type
    assert model.training
    assert all(module.training for module in model.modules())
    assert "_backbone" not in model._modules
    forbidden = (
        "workspace_reader",
        "proposal_head",
        "workspace_arbiter",
        "slot_allocator",
        "workspace_writer",
        "workspace_decay",
        "latent_bridge",
    )
    assert all(not hasattr(model, name) for name in forbidden)
    parameters = tuple(model.parameters())
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(parameter.device == CPU for parameter in parameters)
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert all(not parameter.is_inference() for parameter in parameters)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    assert model.sequence_assembler.num_modules == 3
    assert model.sequence_assembler.max_context_tokens == 10


def test_same_seed_matches_request_sidecar_parameter_initialization() -> None:
    sampling = TemperatureTopPSamplingConfig(temperature=0.7, top_p=0.9)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        teacher = _build()
        teacher_components = dict(teacher._sidecar_modules)

        torch.manual_seed(456)
        request = build_no_workspace_request_orchestrator(
            backbone=_FrozenBackbone(),
            cognitive_config=_config(),
            stop_token_ids=(1,),
            sampling=sampling,
        )
        request_components = dict(request._sidecar_modules)

    shared_names = COMPONENT_NAMES[:-1]
    for name in shared_names:
        teacher_state = teacher_components[name].state_dict()
        request_state = request_components[name].state_dict()
        assert tuple(teacher_state) == tuple(request_state)
        assert all(torch.equal(teacher_state[key], request_state[key]) for key in teacher_state)
    assert type(teacher_components["sequence_assembler"]) is (ModuleSecondPassSequenceAssembler)
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
        (_FrozenBackbone(max_position_embeddings=9), "maxima"),
        (_FrozenBackbone(trainable_parameter_count=1), "frozen"),
        (_FrozenBackbone(has_parameter_gradients=True), "gradients"),
    ],
)
def test_factory_rejects_invalid_backbone_boundary(
    backbone: BackboneAdapter,
    match: str,
) -> None:
    with pytest.raises(IrisTeacherCompositionError, match="preflight validation") as captured:
        _build(backbone)
    assert captured.value.__cause__ is not None
    assert match in str(captured.value.__cause__)


def test_factory_rejects_non_exact_cognitive_config() -> None:
    with pytest.raises(IrisTeacherCompositionError, match="preflight validation") as captured:
        build_no_workspace_teacher_forcing_model(
            backbone=_FrozenBackbone(),
            cognitive_config=cast(CognitiveConfig, object()),
        )
    assert captured.value.__cause__ is not None
    assert "exactly CognitiveConfig" in str(captured.value.__cause__)
