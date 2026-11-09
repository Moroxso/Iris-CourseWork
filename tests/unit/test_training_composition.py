"""Composition contract for the registered teacher-forcing sidecar."""

from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

import iris.application as application
from iris.application.composition import build_request_orchestrator
from iris.application.errors import IrisTeacherCompositionError
from iris.application.training_composition import build_teacher_forcing_model
from iris.backbone import BackboneAdapter
from iris.cognitive import (
    CognitiveModule,
    GoalEncoder,
    LayerSampler,
    ModuleInputRouter,
    ProposalHead,
)
from iris.generation import SecondPassSequenceAssembler, TemperatureTopPSamplingConfig
from iris.schemas import CognitiveConfig, WorkspaceConfig
from iris.training import IrisTeacherForcingModel
from iris.utils.hashing import module_state_sha256
from iris.workspace import (
    LatentBridge,
    SlotAllocator,
    WorkspaceArbiter,
    WorkspaceDecay,
    WorkspaceReader,
    WorkspaceWriter,
)

CPU = torch.device("cpu")
COMPONENT_NAMES = (
    "layer_sampler",
    "goal_encoder",
    "module_input_router",
    "workspace_reader",
    "cognitive_modules",
    "proposal_head",
    "workspace_arbiter",
    "slot_allocator",
    "workspace_writer",
    "workspace_decay",
    "latent_bridge",
    "sequence_assembler",
)
REQUEST_OVERLAP = COMPONENT_NAMES[:-1]


class _FrozenBackbone(BackboneAdapter):
    """Public-metadata-only adapter; composition must execute no model callback."""

    def __init__(
        self,
        *,
        max_position_embeddings: int = 9,
        trainable_parameter_count: int = 0,
        has_parameter_gradients: bool = False,
    ) -> None:
        self._test_max_position_embeddings = max_position_embeddings
        self._test_trainable_parameter_count = trainable_parameter_count
        self._test_has_parameter_gradients = has_parameter_gradients

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
        return self._test_max_position_embeddings

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
        return self._test_trainable_parameter_count

    @property
    def has_parameter_gradients(self) -> bool:
        return self._test_has_parameter_gradients

    def forward_prompt(self, text: str) -> Any:
        raise AssertionError(f"composition executed prompt callback for {text!r}")

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("composition executed embedding callback")

    def forward_embeddings(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("composition executed causal callback")


def _cognitive_config() -> CognitiveConfig:
    return CognitiveConfig.model_validate(
        {
            "name": "tiny-cognitive",
            "state_dim": 4,
            "num_modules": 3,
            "layer_sampler": {"layer_fractions": (0.25, 0.5, 0.75, 1.0)},
        }
    )


def _workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "name": "tiny-workspace",
            "num_slots": 2,
            "cycles": 3,
            "winners_per_cycle": 2,
            "decay_rate": 0.5,
            "empty_threshold": 1e-4,
            "arbiter": {
                "confidence_weight": 1.0,
                "novelty_weight": 1.0,
                "relevance_weight": 1.0,
            },
            "allocator": {
                "empty_bonus": 1.0,
                "allocation_age_penalty": 0.1,
                "retention_age_penalty": 0.1,
            },
        }
    )


def _build(backbone: BackboneAdapter | None = None) -> IrisTeacherForcingModel:
    return build_teacher_forcing_model(
        backbone=backbone or _FrozenBackbone(),
        cognitive_config=_cognitive_config(),
        workspace_config=_workspace_config(),
    )


def _components(model: IrisTeacherForcingModel) -> tuple[tuple[str, nn.Module], ...]:
    return tuple((name, getattr(model, name)) for name in COMPONENT_NAMES)


def test_public_factory_composes_exact_registered_training_graph() -> None:
    backbone = _FrozenBackbone()
    model = _build(backbone)

    assert isinstance(model, IrisTeacherForcingModel)
    assert application.build_teacher_forcing_model is build_teacher_forcing_model
    assert application.IrisTeacherCompositionError is IrisTeacherCompositionError
    assert "build_teacher_forcing_model" in application.__all__
    assert "IrisTeacherCompositionError" in application.__all__
    assert model.training
    assert all(module.training for module in model.modules())
    assert id(backbone) not in {id(module) for module in model.modules()}
    assert "_backbone" not in model._modules

    expected_types = (
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
        SecondPassSequenceAssembler,
    )
    for name, expected_type in zip(COMPONENT_NAMES, expected_types, strict=True):
        assert isinstance(getattr(model, name), expected_type)

    parameters = tuple(model.parameters())
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(parameter.device == CPU for parameter in parameters)
    assert all(not parameter.is_inference() for parameter in parameters)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    state_keys = tuple(model.state_dict())
    assert state_keys
    assert all(
        not key.startswith("backbone") and not key.startswith("_backbone") for key in state_keys
    )
    assert any(key.startswith("layer_sampler.") for key in state_keys)
    assert any(key.startswith("latent_bridge.") for key in state_keys)

    assert model.layer_sampler.backbone_hidden_size == 6
    assert model.layer_sampler.state_dim == 4
    assert model.sequence_assembler.hidden_size == 6
    assert model.sequence_assembler.num_slots == 2
    assert model.sequence_assembler.vocabulary_size == 16
    assert model.sequence_assembler.max_prompt_tokens == 5
    assert model.sequence_assembler.max_target_tokens == 2
    assert model.sequence_assembler.max_context_tokens == 9


def test_same_seed_reproduces_sidecar_and_different_seed_changes_it() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _build()
        first_hash = module_state_sha256(_components(first))
        first_following_draw = torch.rand(4)

        torch.manual_seed(123)
        second = _build()
        second_hash = module_state_sha256(_components(second))
        second_following_draw = torch.rand(4)

        torch.manual_seed(124)
        third = _build()
        third_hash = module_state_sha256(_components(third))

    assert first_hash == second_hash
    assert first_hash != third_hash
    assert torch.equal(first_following_draw, second_following_draw)


def test_overlapping_training_and_request_components_share_seeded_state() -> None:
    sampling = TemperatureTopPSamplingConfig(temperature=1.0, top_p=1.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        teacher = _build()
        torch.manual_seed(456)
        request = build_request_orchestrator(
            backbone=_FrozenBackbone(),
            cognitive_config=_cognitive_config(),
            workspace_config=_workspace_config(),
            stop_token_ids=(),
            sampling=sampling,
        )

    for name in REQUEST_OVERLAP:
        teacher_state = getattr(teacher, name).state_dict()
        request_state = getattr(request, f"_{name}").state_dict()
        assert teacher_state.keys() == request_state.keys()
        for key in teacher_state:
            assert torch.equal(teacher_state[key], request_state[key])


def test_ambient_inference_mode_cannot_create_inference_parameters() -> None:
    with torch.inference_mode():
        model = _build()

    assert model.training
    assert all(not parameter.is_inference() for parameter in model.parameters())


@pytest.mark.parametrize(
    ("backbone", "cause_match"),
    [
        (_FrozenBackbone(max_position_embeddings=8), "context"),
        (_FrozenBackbone(trainable_parameter_count=1), "trainable parameters"),
        (_FrozenBackbone(has_parameter_gradients=True), "parameters must not contain gradients"),
    ],
)
def test_preflight_rejects_invalid_frozen_backbone_contract(
    backbone: BackboneAdapter,
    cause_match: str,
) -> None:
    with pytest.raises(
        IrisTeacherCompositionError,
        match="preflight validation",
    ) as captured:
        _build(backbone)

    assert isinstance(captured.value.__cause__, ValueError)
    assert cause_match in str(captured.value.__cause__)


def test_constructor_failure_is_typed_and_preserves_exact_cause() -> None:
    failure = RuntimeError("constructor exploded")
    with (
        patch(
            "iris.application.training_composition.LayerSampler",
            side_effect=failure,
        ),
        pytest.raises(
            IrisTeacherCompositionError,
            match="LayerSampler construction and placement",
        ) as captured,
    ):
        _build()

    assert captured.value.__cause__ is failure
