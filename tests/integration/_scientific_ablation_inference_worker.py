"""Executable offline inference witness. Never imports producer test factories.

Arguments are authored test inputs, not a public CLI or an untrusted code runner.
Fresh Python -I/-B, explicit package root and independently supplied job digest.
"""

import gc
import hashlib
import json
import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch


def main() -> None:
    assert len(sys.argv) == 4
    package_root = Path(sys.argv[1]).resolve(strict=True)
    job_path = Path(sys.argv[2]).resolve(strict=True)
    assert job_path.stat().st_size <= 1024**2
    raw = job_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sys.argv[3]
    job: dict[str, Any] = json.loads(raw)
    assert set(job) == {
        "backbone_weights",
        "backbone_weights_sha256",
        "backbone_configuration",
        "tokenizer_json",
        "lm_configuration",
        "sampling_seed",
        "sampling_state_hex",
        "cases",
    }
    assert type(job["cases"]) is list and len(job["cases"]) == 12
    assert all(
        set(case) == {"mode", "path", "anchors", "prompt"} and len(case["anchors"]) == 3
        for case in job["cases"]
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(package_root))

    import random

    import torch
    from safetensors.torch import load as load_tensors
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    import iris.application as application
    import iris.application.scientific_ablation_inference as projection
    import iris.application.scientific_ablation_inference_io as io
    import iris.application.scientific_ablation_request as request_module
    from iris.application.scientific_optimization import _same_runtime
    from iris.application.scientific_teacher import IrisScientificTeacherV1
    from iris.backbone import BackboneAdapter
    from iris.generation import TemperatureTopPSamplingConfig
    from iris.schemas import BackboneConfig
    from iris.training.model import IrisTeacherForcingModel
    from iris.utils.devices import inspect_device
    from iris.utils.hashing import canonical_json_bytes, tensor_sha256
    from iris.utils.seeds import capture_runtime_state
    from tests.integration._scientific_ablation_inference_observation import (
        compact_outputs,
        execute_observation,
    )

    torch.set_num_threads(1)
    forbidden = (
        application.IrisScientificAblationRunV1,
        application.IrisScientificAblationOptimizerV1,
        application.IrisScientificAblationTeacherV1,
        application.IrisScientificAblationLifecycleV1,
        application.IrisScientificRunV1,
        application.IrisScientificOptimizerV1,
        IrisScientificTeacherV1,
        IrisTeacherForcingModel,
        torch.optim.AdamW,
    )

    def no_owners() -> None:
        assert not any(type(value) in forbidden for value in gc.get_objects())

    no_owners()
    weights_path = Path(job["backbone_weights"])
    assert weights_path.stat().st_size <= 8 * 1024**2
    weights = weights_path.read_bytes()
    assert hashlib.sha256(weights).hexdigest() == job["backbone_weights_sha256"]
    with torch.device("cpu"), torch.inference_mode(False):
        tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
            tokenizer_object=Tokenizer.from_str(job["tokenizer_json"]),
            eos_token="<eos>",
            bos_token="<bos>",
        )
        lm = Qwen3ForCausalLM(Qwen3Config.from_dict(job["lm_configuration"]))  # type: ignore[no-untyped-call]
        restored = lm.load_state_dict(load_tensors(weights), strict=True)
        assert not restored.missing_keys and not restored.unexpected_keys
        with (
            patch("iris.backbone.adapter.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("iris.backbone.adapter.AutoModelForCausalLM.from_pretrained", return_value=lm),
        ):
            backbone = BackboneAdapter(
                BackboneConfig.model_validate_json(job["backbone_configuration"]),
                inspect_device("cpu"),
            )
    runtime = backbone.checkpoint_runtime_sha256
    random.seed(9203)
    random.gauss(0, 1)
    torch.manual_seed(9209)
    cases: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    denials: list[int] = []
    with ExitStack() as guards:
        constructor_guards = [
            guards.enter_context(
                patch.object(
                    cls,
                    "__init__",
                    side_effect=AssertionError("training owner construction forbidden"),
                )
            )
            for cls in forbidden
        ]
        for index, case in enumerate(job["cases"]):
            no_owners()
            anchors = case["anchors"]
            expected = dict(
                zip(
                    (
                        "expected_sha256",
                        "expected_source_checkpoint_sha256",
                        "expected_lifecycle_sha256",
                    ),
                    anchors,
                    strict=True,
                )
            )
            generator = torch.Generator(device="cpu")
            generator.set_state(
                torch.tensor(
                    list(bytes.fromhex(job["sampling_state_hex"])), dtype=torch.uint8, device="cpu"
                )
            )
            assert generator.initial_seed() == job["sampling_seed"]
            before = capture_runtime_state(torch.device("cpu"))
            default = "meta" if index % 2 else "cpu"
            inference = bool(index % 3)
            with torch.device(default), torch.inference_mode(inference):
                bundle = application.read_scientific_ablation_inference_v1(
                    Path(case["path"]), **expected
                )
                assert bundle.manifest.payload.backbone_runtime_sha256 == runtime
                assert bundle.manifest.payload.completed_steps == 1
                assert bundle.manifest.payload.selection.current_completed_steps == 3
                assert (
                    bundle.manifest.payload.source_checkpoint_sha256
                    != bundle.manifest.payload.selection.current_checkpoint_sha256
                )
                request = application.build_scientific_ablation_request_v1(
                    bundle,
                    backbone=backbone,
                    stop_token_ids=(),
                    sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
                    **expected,
                )
                outputs = [
                    execute_observation(request, backbone, case["prompt"], strategy, generator)
                    for strategy in ("greedy", "greedy", "sampling", "sampling")
                ]
                assert torch.get_default_device() == torch.device(default)
                assert torch.is_inference_mode_enabled() is inference
            assert outputs[0] == outputs[1]
            assert (
                outputs[0]["caller_sampling_state_sha256"]
                != outputs[2]["caller_sampling_state_sha256"]
            )
            assert (
                outputs[2]["caller_sampling_state_sha256"]
                != outputs[3]["caller_sampling_state_sha256"]
            )
            assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
            assert backbone.checkpoint_runtime_sha256 == runtime
            cases.append(
                {
                    "mode": case["mode"],
                    **compact_outputs(outputs),
                    "default": default,
                    "inference": inference,
                }
            )
            no_owners()

            if index:
                continue
            for anchor in range(3):
                wrong = {**expected, list(expected)[anchor]: "0" * 64}
                with (
                    patch.object(io, "_read_file", wraps=io._read_file) as read,
                    patch.object(projection, "load_tensors") as decode,
                ):
                    try:
                        application.read_scientific_ablation_inference_v1(
                            Path(case["path"]), **wrong
                        )
                    except application.IrisScientificAblationInferenceError:
                        pass
                    else:
                        raise AssertionError("Bad trusted anchor accepted")
                assert all(call.args[0].name == "manifest.json" for call in read.call_args_list)
                decode.assert_not_called()
                denials.append(anchor)
            for kind in ("interrupt", "exit", "group"):
                for secondary in (False, True):
                    primary: BaseException = (
                        KeyboardInterrupt("first")
                        if kind == "interrupt"
                        else SystemExit(73)
                        if kind == "exit"
                        else BaseExceptionGroup("first", [SystemExit(74), ValueError("ordinary")])
                    )
                    request = application.build_scientific_ablation_request_v1(
                        bundle,
                        backbone=backbone,
                        stop_token_ids=(),
                        sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
                        **expected,
                    )
                    initial = generator.get_state().clone()
                    caller = capture_runtime_state(torch.device("cpu"))
                    native = request_module._restore_sampling_state
                    cleanup = ValueError("secondary restore acknowledgement")

                    def restore_then_error(
                        value: torch.Generator,
                        state: torch.Tensor,
                        *,
                        native_restore: Callable[[torch.Generator, torch.Tensor], None] = native,
                        has_secondary: bool = secondary,
                        error: BaseException = cleanup,
                    ) -> None:
                        native_restore(value, state)
                        if has_secondary:
                            raise error

                    with (
                        patch.object(request_module, "_receipt", side_effect=primary),
                        patch.object(
                            request_module,
                            "_restore_sampling_state",
                            side_effect=restore_then_error,
                        ),
                        patch.object(
                            backbone, "forward_embeddings", wraps=backbone.forward_embeddings
                        ) as forwards,
                    ):
                        try:
                            request.execute(
                                case["prompt"], strategy="sampling", generator=generator
                            )
                        except BaseException as actual:
                            assert actual is primary
                            if secondary:
                                assert isinstance(actual.__cause__, BaseExceptionGroup)
                                assert actual.__cause__.exceptions == (cleanup,)
                        else:
                            raise AssertionError("Failed request returned output")
                    assert forwards.call_count == 256
                    assert request._state == "poisoned"
                    assert torch.equal(generator.get_state(), initial)
                    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
                    faults.append(
                        {
                            "kind": kind,
                            "secondary": secondary,
                            "full_state_restored": tensor_sha256(initial),
                        }
                    )
        for guard in constructor_guards:
            guard.assert_not_called()
    no_owners()
    loaded = []
    for name, module in sorted(sys.modules.items()):
        if name == "iris" or name.startswith("iris."):
            assert module.__file__ is not None
            file = Path(module.__file__).resolve(strict=True)
            assert file.is_relative_to(package_root)
            loaded.append(
                {
                    "module": name,
                    "path": file.relative_to(package_root).as_posix(),
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            )
        if name.startswith("tests."):
            assert name in {
                "tests.integration",
                "tests.integration._scientific_ablation_inference_observation",
            }
    result = {
        "kind": "iris-mode-bound-inference-fresh-process-test-v1",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "cwd": str(Path.cwd()),
        "package_root": str(package_root),
        "isolated": sys.flags.isolated,
        "training_owner_constructors_called": 0,
        "training_owner_instances": 0,
        "backbone_runtime_sha256": runtime,
        "cases": cases,
        "primary_signal_faults": faults,
        "trust_anchor_denials": denials,
        "loaded_package_files": loaded,
    }
    encoded = canonical_json_bytes(result)
    assert len(encoded) <= 512 * 1024, f"Evidence byte budget exceeded: {len(encoded)}"
    print(encoded.decode())


if __name__ == "__main__":
    main()
