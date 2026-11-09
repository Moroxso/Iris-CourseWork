"""Focused genuine-owner denial/default/commit invariants, not new task metrics."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_ablation_controlled_request as runtime
from iris.application.errors import IrisRequestError
from iris.application.scientific_ablation_checkpoint import build_scientific_ablation_checkpoint_v1
from iris.application.scientific_ablation_inference import build_scientific_ablation_inference_v1
from iris.application.scientific_optimization import _same_runtime
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_ablation_controlled_request import (
    Native,
    control,
    generator,
    load,
)
from tests.integration.test_scientific_ablation_inference_projection import _fresh, _source
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trusted() -> Iterator[Native]:
    with (
        patch(
            "iris.backbone.adapter.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("No model load"),
        ) as models,
        patch(
            "iris.backbone.adapter.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("No tokenizer load"),
        ) as tokenizers,
        patch(
            "iris.application.synthetic_scoring_v2.execute_python_job_v2",
            side_effect=AssertionError("No candidate execution"),
        ) as candidates,
    ):
        source, controller = _source()
        bundle = build_scientific_ablation_inference_v1(
            source, expected_lifecycle_sha256=source.manifest.sha256, make_fresh_run=_fresh
        )
        anchors = (
            bundle.manifest.sha256,
            bundle.manifest.payload.source_checkpoint_sha256,
            source.manifest.sha256,
        )
        before = build_scientific_ablation_checkpoint_v1(controller._run).manifest.sha256
        yield Native(bundle, anchors, _fresh()._backbone, controller, before)
        assert build_scientific_ablation_checkpoint_v1(controller._run).manifest.sha256 == before
        models.assert_not_called()
        tokenizers.assert_not_called()
        candidates.assert_not_called()


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, GeneratorExit, RuntimeError))
def test_malformed_guard_getter_never_masks_primary_denial_or_runs_callback(
    trusted: Native, signal: type[BaseException]
) -> None:
    current = load(trusted)
    original_owner = current._owner
    accessed: list[str] = []

    class ForeignGuard:
        @property
        def owner(self) -> object:
            accessed.append("owner")
            raise signal("foreign getter")

    object.__setattr__(current, "_guard", ForeignGuard())
    with (
        patch.object(runtime, "capture_runtime_state") as capture,
        patch.object(trusted.backbone, "forward_prompt") as forward,
        pytest.raises(BaseException) as raised,
    ):
        current.execute(_rows()[0].prompt, control=control("empty"), strategy="greedy")
    assert isinstance(raised.value, IrisRequestError)
    assert accessed == []
    assert original_owner._state == "poisoned"
    capture.assert_not_called()
    forward.assert_not_called()


@pytest.mark.parametrize(
    "slot", ("_guard", "_request", "_owner", "_binding", "_metadata", "_state")
)
@pytest.mark.parametrize("missing", (False, True))
def test_missing_or_foreign_wrapper_slots_poison_real_owner_before_runtime(
    trusted: Native, slot: str, missing: bool
) -> None:
    current = load(trusted)
    owner = current._owner
    if missing:
        object.__delattr__(current, slot)
    else:
        object.__setattr__(current, slot, object())
    with (
        patch.object(runtime, "capture_runtime_state") as capture,
        patch.object(trusted.backbone, "forward_prompt") as forward,
        pytest.raises(IrisRequestError),
    ):
        runtime.IrisScientificAblationControlledRequestV1.execute(
            current, _rows()[0].prompt, control=control("empty"), strategy="greedy"
        )
    assert owner._state == "poisoned"
    capture.assert_not_called()
    forward.assert_not_called()


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_primary_forward_signal_restores_default_dtype_and_sampling(
    trusted: Native, signal: type[BaseException]
) -> None:
    current = load(trusted)
    prompt = _rows()[0].prompt
    sampled = generator(prompt, "sampling")
    assert sampled is not None
    initial = sampled.get_state().clone()
    caller = capture_runtime_state(torch.device("cpu"))
    dtype = torch.get_default_dtype()
    primary = signal("forward with dtype drift")

    def fail(*args: object, **kwargs: object) -> object:
        torch.set_default_dtype(torch.float64 if dtype != torch.float64 else torch.float32)
        raise primary

    try:
        with (
            patch.object(trusted.backbone, "forward_prompt", side_effect=fail),
            pytest.raises(BaseException) as raised,
        ):
            current.execute(prompt, control=control("zero"), strategy="sampling", generator=sampled)
        assert raised.value is primary
        assert torch.get_default_dtype() == dtype
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
        assert torch.equal(sampled.get_state(), initial)
        assert current._state == current._owner._state == "poisoned"
    finally:
        torch.set_default_dtype(dtype)


def test_silent_sampling_commit_return_cannot_emit_inconsistent_receipt(trusted: Native) -> None:
    current = load(trusted)
    prompt = _rows()[0].prompt
    sampled = generator(prompt, "sampling")
    assert sampled is not None
    initial = sampled.get_state().clone()
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(runtime, "_commit_sampling_state", return_value=None),
        pytest.raises(IrisRequestError),
    ):
        current.execute(prompt, control=control("empty"), strategy="sampling", generator=sampled)
    assert torch.equal(sampled.get_state(), initial)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert current._state == current._owner._state == "poisoned"
