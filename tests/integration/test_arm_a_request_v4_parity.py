"""Evaluation-parity coverage for the checkpoint-free arm-A baseline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import patch

import pytest
import torch
from transformers.cache_utils import Cache

import iris.application.arm_request_command_v3 as command_v3
import iris.application.arm_request_command_v4 as command_v4
import iris.application.bootstrap as bootstrap_v3
import iris.application.request_checkpoint_bootstrap as bootstrap_v4
from iris.application import (
    IrisArmRequestBootstrapResultV4,
    bootstrap_arm_request_runtime_v3,
    bootstrap_arm_request_runtime_v4,
    build_request_artifact_v3,
    build_request_artifact_v4,
    read_request_artifact_v3,
    read_request_artifact_v4,
)
from iris.backbone import BackboneAdapter
from iris.backbone.outputs import BackboneCausalOutput
from iris.generation import DecodingStrategy, TemperatureTopPSamplingConfig
from iris.schemas import IrisBootstrapConfig
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.unit.test_backbone_adapter import FakeTokenizer as _PromptFakeTokenizer
from tests.unit.test_backbone_second_pass import FakeModel as _AdapterFakeModel
from tests.unit.test_bootstrap import (
    CPU,
    CPU_INFO,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    valid_bootstrap_data,
    validated_config,
)
from tests.unit.test_greedy_decoder import FakeCache, token_scores

_PROMPT = "arm A parity prompt"
_STOP_TOKEN_IDS = (2, 7)
_HIDDEN_SIZE = 4
_VOCABULARY_SIZE = 17
_MAX_CONTEXT_TOKENS = MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 8


class _ProductionParityTokenizer(_PromptFakeTokenizer):
    """Prompt tokenizer double completed with the production decode boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.decode_calls: list[tuple[list[int], bool, bool]] = []

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        self.decode_calls.append(
            (
                list(token_ids),
                skip_special_tokens,
                clean_up_tokenization_spaces,
            )
        )
        return "tokens:" + ",".join(str(token_id) for token_id in token_ids)


class _ProductionParityModel(_AdapterFakeModel):
    """Tiny frozen model with an explicit two-token deterministic decode script."""

    def forward(self, **kwargs: object) -> SimpleNamespace:
        output = cast(SimpleNamespace, super().forward(**kwargs))
        call_index = len(self.calls) - 1
        if call_index >= 2:
            raise AssertionError("production parity model received too many decode calls")
        scores = torch.full((1, 1, 8), -100.0)
        scores[0, 0, (5, 6)[call_index]] = 100.0
        output.logits = scores
        return output


class _ArmAParityBackbone(BackboneAdapter):
    """Deterministic frozen adapter double supporting the full cached decode path."""

    def __init__(self) -> None:
        self.forward_calls: list[tuple[int | None, torch.Tensor, torch.Tensor]] = []
        self._scores = token_scores(5, 2, vocabulary_size=_VOCABULARY_SIZE)

    @property
    def hidden_size(self) -> int:
        return _HIDDEN_SIZE

    @property
    def vocabulary_size(self) -> int:
        return _VOCABULARY_SIZE

    @property
    def max_position_embeddings(self) -> int:
        return _MAX_CONTEXT_TOKENS

    @property
    def max_prompt_tokens(self) -> int:
        return MAX_PROMPT_TOKENS

    @property
    def max_new_tokens(self) -> int:
        return MAX_NEW_TOKENS

    @property
    def device(self) -> torch.device:
        return CPU

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def trainable_parameter_count(self) -> int:
        return 0

    @property
    def has_parameter_gradients(self) -> bool:
        return False

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        if text != _PROMPT:
            raise AssertionError("parity test received an unexpected prompt")
        return {
            "input_ids": torch.tensor([[3, 11, 13]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
        }

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (
            input_ids.to(dtype=torch.float32)
            .unsqueeze(-1)
            .expand(-1, -1, _HIDDEN_SIZE)
            .clone()
        )

    def forward_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        logits_to_keep: Literal[0, 1] = 0,
    ) -> BackboneCausalOutput:
        if labels is not None or not use_cache or logits_to_keep != 1:
            raise AssertionError("arm-A parity decode violated its cached inference contract")
        call_index = len(self.forward_calls)
        if call_index >= len(self._scores):
            raise AssertionError("arm-A parity decode exceeded the deterministic script")
        incoming_length = (
            None if past_key_values is None else past_key_values.get_seq_length()
        )
        self.forward_calls.append(
            (
                incoming_length,
                attention_mask.detach().clone(),
                position_ids.detach().clone(),
            )
        )
        if past_key_values is None:
            cache: Cache = FakeCache(inputs_embeds.shape[1])
        else:
            if not isinstance(past_key_values, FakeCache):
                raise AssertionError("arm-A parity decode returned an unexpected cache")
            past_key_values.sequence_length += inputs_embeds.shape[1]
            cache = past_key_values
        logits = self._scores[call_index].to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        return BackboneCausalOutput(
            logits=logits.reshape(1, 1, _VOCABULARY_SIZE),
            loss=None,
            past_key_values=cache,
        )

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        values = tuple(int(token_id) for token_id in token_ids.reshape(-1).tolist())
        if values != (5, 2):
            raise AssertionError("arm-A parity decode returned unexpected token IDs")
        return "deterministic arm A response"


@pytest.fixture(autouse=True)
def preserve_process_runtime_state() -> Iterator[None]:
    snapshot = capture_runtime_state(CPU)
    try:
        yield
    finally:
        restore_runtime_state(snapshot)


def _generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@pytest.mark.parametrize(
    ("strategy", "sampling_seed"),
    [("greedy", None), ("sampling", 811)],
)
def test_checkpoint_aware_arm_a_preserves_v3_evaluation_and_artifact_projection(
    strategy: DecodingStrategy,
    sampling_seed: int | None,
) -> None:
    config = validated_config(seed=90210)
    sampling = TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)
    v3_backbone = _ArmAParityBackbone()
    v4_backbone = _ArmAParityBackbone()

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setattr(
            bootstrap_v3,
            "_resolve_device",
            lambda _config: (CPU_INFO, torch.float32),
        )
        monkeypatch.setattr(bootstrap_v3, "_create_backbone", lambda *_args: v3_backbone)
        runtime_v3 = bootstrap_arm_request_runtime_v3(
            "A",
            config,
            stop_token_ids=_STOP_TOKEN_IDS,
            sampling=sampling,
        )

        monkeypatch.setattr(
            bootstrap_v4,
            "_resolve_device",
            lambda _config: (CPU_INFO, torch.float32),
        )
        monkeypatch.setattr(bootstrap_v4, "_create_backbone", lambda *_args: v4_backbone)
        result_v4 = bootstrap_arm_request_runtime_v4(
            "A",
            config,
            checkpoint_identity=None,
            stop_token_ids=_STOP_TOKEN_IDS,
            sampling=sampling,
        )

    assert type(result_v4) is IrisArmRequestBootstrapResultV4
    assert result_v4.checkpoint_transfer is None
    runtime_v4 = result_v4.runtime
    assert runtime_v4.profile == runtime_v3.profile
    assert runtime_v4.execution_profile_set_sha256 == runtime_v3.execution_profile_set_sha256
    assert runtime_v4.configuration_json == runtime_v3.configuration_json
    assert runtime_v4.configuration_sha256 == runtime_v3.configuration_sha256
    assert runtime_v4.device_info == runtime_v3.device_info
    assert runtime_v4.dtype == runtime_v3.dtype
    assert runtime_v4.seed == runtime_v3.seed
    assert runtime_v4.deterministic is runtime_v3.deterministic
    assert runtime_v4.sidecar_state_sha256 is runtime_v3.sidecar_state_sha256 is None

    generator_v3 = _generator(sampling_seed)
    generator_v4 = _generator(sampling_seed)
    execution_v3 = runtime_v3.orchestrator.execute(
        _PROMPT,
        strategy=strategy,
        generator=generator_v3,
    )
    execution_v4 = runtime_v4.orchestrator.execute(
        _PROMPT,
        strategy=strategy,
        generator=generator_v4,
    )

    assert execution_v4.generation.text == execution_v3.generation.text
    assert torch.equal(execution_v4.generation.token_ids, execution_v3.generation.token_ids)
    assert execution_v4.generation.stop_reason == execution_v3.generation.stop_reason
    assert (
        execution_v4.generation.matched_stop_token_id
        == execution_v3.generation.matched_stop_token_id
    )
    assert execution_v4.generation.workspace_trace == execution_v3.generation.workspace_trace == ()
    assert torch.equal(execution_v4.prompt_input_ids, execution_v3.prompt_input_ids)
    assert torch.equal(execution_v4.prompt_attention_mask, execution_v3.prompt_attention_mask)
    assert execution_v4.receipt == execution_v3.receipt
    if generator_v3 is not None and generator_v4 is not None:
        assert torch.equal(generator_v4.get_state(), generator_v3.get_state())

    assert len(v4_backbone.forward_calls) == len(v3_backbone.forward_calls) == 2
    for v4_call, v3_call in zip(v4_backbone.forward_calls, v3_backbone.forward_calls, strict=True):
        assert v4_call[0] == v3_call[0]
        assert torch.equal(v4_call[1], v3_call[1])
        assert torch.equal(v4_call[2], v3_call[2])

    artifact_v3 = build_request_artifact_v3(
        prompt=_PROMPT,
        strategy=strategy,
        stop_token_ids=_STOP_TOKEN_IDS,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=False,
        runtime=runtime_v3,
        execution=execution_v3,
    )
    artifact_v4 = build_request_artifact_v4(
        prompt=_PROMPT,
        strategy=strategy,
        stop_token_ids=_STOP_TOKEN_IDS,
        sampling=sampling,
        sampling_seed=sampling_seed,
        allow_download=False,
        runtime=runtime_v4,
        execution=execution_v4,
        checkpoint_transfer=None,
    )
    assert artifact_v4.payload.checkpoint_transfer is None
    assert artifact_v4.payload.model_dump(
        mode="json",
        exclude={"checkpoint_transfer"},
    ) == artifact_v3.payload.model_dump(mode="json")


def test_packaged_arm_a_commands_preserve_production_adapter_and_artifact_parity(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    config_data = valid_bootstrap_data(seed=8917)
    backbone_data = cast(dict[str, object], config_data["backbone"])
    backbone_data["limits"] = {"max_prompt_tokens": 3, "max_new_tokens": 2}
    runtime_data = cast(dict[str, object], config_data["runtime"])
    runtime_data["output_dir"] = str(output_root)
    config = IrisBootstrapConfig.model_validate(config_data)
    sampling = TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9)
    models: list[_ProductionParityModel] = []
    tokenizers: list[_ProductionParityTokenizer] = []

    def load_model(**kwargs: object) -> _ProductionParityModel:
        del kwargs
        model = _ProductionParityModel()
        models.append(model)
        return model

    def load_tokenizer(**kwargs: object) -> _ProductionParityTokenizer:
        del kwargs
        tokenizer = _ProductionParityTokenizer()
        tokenizers.append(tokenizer)
        return tokenizer

    with (
        patch.object(command_v3, "_compose_request_config", return_value=config),
        patch.object(command_v4, "_compose_request_config", return_value=config),
        patch.object(bootstrap_v3, "inspect_device", return_value=CPU_INFO),
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=load_tokenizer,
        ),
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=load_model,
        ),
    ):
        output_v3 = command_v3.generate_arm_request_v3(
            arm_id="A",
            prompt=_PROMPT,
            strategy="greedy",
            stop_token_ids=(7,),
            sampling=sampling,
            sampling_seed=None,
            allow_download=False,
        )
        output_v4 = command_v4.generate_arm_request_v4(
            arm_id="A",
            checkpoint_identity=None,
            prompt=_PROMPT,
            strategy="greedy",
            stop_token_ids=(7,),
            sampling=sampling,
            sampling_seed=None,
            allow_download=False,
        )

    assert torch.equal(output_v4.token_ids, output_v3.token_ids)
    assert torch.equal(output_v3.token_ids, torch.tensor([[5, 6]], dtype=torch.int64))
    assert output_v4.text == output_v3.text == "tokens:5,6"
    assert output_v4.stop_reason == output_v3.stop_reason == "max_new_tokens"
    assert output_v4.matched_stop_token_id is output_v3.matched_stop_token_id is None
    assert output_v4.workspace_trace == output_v3.workspace_trace == ()

    assert len(models) == len(tokenizers) == 2
    assert len(models[0].calls) == len(models[1].calls) == 2
    for model in models:
        assert all(call["use_cache"] is True for call in model.calls)
        assert all(call["logits_to_keep"] == 1 for call in model.calls)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(not parameter.requires_grad for parameter in model.parameters())
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    for name, tensor in models[0].state_dict().items():
        torch.testing.assert_close(
            tensor,
            models[1].state_dict()[name],
            rtol=0,
            atol=0,
        )
    for tokenizer in tokenizers:
        assert tokenizer.decode_calls == [([5, 6], True, False)]

    v3_paths = tuple((output_root / "requests" / "v3").glob("*.json"))
    v4_paths = tuple((output_root / "requests" / "v4").glob("*.json"))
    assert len(v3_paths) == len(v4_paths) == 1
    artifact_v3 = read_request_artifact_v3(v3_paths[0])
    artifact_v4 = read_request_artifact_v4(v4_paths[0])
    assert artifact_v4.payload.checkpoint_transfer is None
    assert artifact_v4.payload.model_dump(
        mode="json",
        exclude={"checkpoint_transfer"},
    ) == artifact_v3.payload.model_dump(mode="json")
