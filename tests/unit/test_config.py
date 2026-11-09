import re
from importlib.resources import files

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from iris.schemas import BackboneConfig, CognitiveConfig, IrisBootstrapConfig, WorkspaceConfig


def test_base_config_composes_for_cpu_smoke() -> None:
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        config = compose(config_name="config")

    assert config.project.name == "iris"
    assert config.project.version == "0.1.0"
    assert config.runtime.seed == 42
    assert config.backbone.name == "qwen3_0_6b"
    assert config.backbone.repository == "Qwen/Qwen3-0.6B"
    assert config.backbone.trust_remote_code is False
    assert config.backbone.output_hidden_states is True
    assert config.backbone.use_cache is False
    assert config.cognitive.name == "iris_gw_0_1"
    assert config.cognitive.state_dim == 256
    assert config.cognitive.num_modules == 8
    assert list(config.cognitive.layer_sampler.layer_fractions) == [0.25, 0.5, 0.75, 1.0]
    assert config.workspace.name == "iris_gw_0_1"
    assert config.workspace.num_slots == 8
    assert config.workspace.cycles == 3
    assert config.workspace.winners_per_cycle == 2
    assert config.workspace.decay_rate == 0.85
    assert config.workspace.empty_threshold == 1.0e-3
    assert config.workspace.arbiter.confidence_weight == 0.25
    assert config.workspace.arbiter.novelty_weight == 0.10
    assert config.workspace.arbiter.relevance_weight == 0.25
    assert config.workspace.allocator.empty_bonus == 1.0
    assert config.workspace.allocator.allocation_age_penalty == 0.10
    assert config.workspace.allocator.retention_age_penalty == 0.10
    assert config.hardware.device == "cpu"
    assert config.hardware.precision == "fp32"


def test_packaged_backbone_config_matches_hydra_default() -> None:
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        config = compose(config_name="config")

    packaged_resource = files("iris.configs").joinpath("backbone", "qwen3_0_6b.yaml")
    packaged_config = OmegaConf.create(packaged_resource.read_text(encoding="utf-8"))

    assert OmegaConf.to_container(config.backbone) == OmegaConf.to_container(packaged_config)

    validated = BackboneConfig.model_validate(OmegaConf.to_container(config.backbone))
    assert validated.revision == validated.tokenizer_revision
    assert re.fullmatch(r"[0-9a-f]{40}", validated.revision)
    assert validated.limits.max_prompt_tokens == 256
    assert validated.limits.max_new_tokens == 32

    cognitive = CognitiveConfig.model_validate(OmegaConf.to_container(config.cognitive))
    assert cognitive.state_dim == 256
    assert cognitive.num_modules == 8
    assert cognitive.layer_sampler.layer_fractions == (0.25, 0.5, 0.75, 1.0)

    workspace = WorkspaceConfig.model_validate(OmegaConf.to_container(config.workspace))
    assert workspace.num_slots == 8
    assert workspace.cycles == 3
    assert workspace.winners_per_cycle == 2
    assert workspace.decay_rate == 0.85
    assert workspace.empty_threshold == 1.0e-3
    assert workspace.arbiter.confidence_weight == 0.25
    assert workspace.arbiter.novelty_weight == 0.10
    assert workspace.arbiter.relevance_weight == 0.25
    assert workspace.allocator.empty_bonus == 1.0
    assert workspace.allocator.allocation_age_penalty == 0.10
    assert workspace.allocator.retention_age_penalty == 0.10


def test_pinned_main_backbone_profile_composes_without_changing_debug_default() -> None:
    with initialize_config_module(version_base=None, config_module="iris.configs"):
        main = compose(config_name="config", overrides=["backbone=qwen3_1_7b"])
        debug = compose(config_name="config")

    owned = IrisBootstrapConfig.model_validate(OmegaConf.to_container(main))
    assert owned.backbone.name == "qwen3_1_7b"
    assert owned.backbone.repository == "Qwen/Qwen3-1.7B"
    assert owned.backbone.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert owned.backbone.tokenizer_revision == owned.backbone.revision
    assert owned.backbone.trust_remote_code is False
    assert owned.backbone.output_hidden_states is True
    assert owned.backbone.use_cache is False
    assert owned.backbone.dtype == owned.hardware.precision == "fp32"
    assert owned.backbone.limits.max_prompt_tokens == 256
    assert owned.backbone.limits.max_new_tokens == 32

    packaged_resource = files("iris.configs").joinpath("backbone", "qwen3_1_7b.yaml")
    packaged_config = OmegaConf.create(packaged_resource.read_text(encoding="utf-8"))
    assert OmegaConf.to_container(main.backbone) == OmegaConf.to_container(packaged_config)
    assert debug.backbone.name == "qwen3_0_6b"
    assert debug.backbone.repository == "Qwen/Qwen3-0.6B"
