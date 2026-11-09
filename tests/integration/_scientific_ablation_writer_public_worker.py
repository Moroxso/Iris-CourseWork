"""Fixed trusted cold public-import worker; no candidate-generated Python."""

import importlib
import json
import os
import sys
from pathlib import Path
from types import FrameType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("application", "artifacts"):
        raise ValueError("Exact fixed first namespace required")
    first = sys.argv[1]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

    before = torch.get_rng_state().clone()
    defaults = (
        torch.get_default_dtype(),
        torch.get_default_device(),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
    )
    previous = sys.getprofile()

    def forbid(frame: FrameType, event: str, argument: object) -> None:
        if (
            event == "call"
            and frame.f_code.co_name
            in (
                "execute_python_job_v2",
                "_measure_writer_effects",
                "_source_transaction",
            )
            and Path(frame.f_code.co_filename).resolve().is_relative_to(ROOT)
        ):
            raise AssertionError("Public import must not execute native/candidate work")

    try:
        with (
            patch.object(
                torch.nn.Module, "__init__", side_effect=AssertionError("No import model")
            ),
            patch.object(
                torch.nn.Module, "__call__", side_effect=AssertionError("No import forward")
            ),
            patch.object(
                Qwen3ForCausalLM, "__init__", side_effect=AssertionError("No import Qwen")
            ),
            patch.object(
                AutoModelForCausalLM,
                "from_pretrained",
                side_effect=AssertionError("No import weights"),
            ),
            patch.object(
                AutoTokenizer,
                "from_pretrained",
                side_effect=AssertionError("No import tokenizer"),
            ),
        ):
            sys.setprofile(forbid)
            importlib.import_module(f"iris.{first}")
            application = importlib.import_module("iris.application")
            artifacts = importlib.import_module("iris.artifacts")
            assert (
                application.read_scientific_ablation_writer_bundle_v1
                is importlib.import_module(
                    "iris.application.scientific_ablation_writer_io"
                ).read_scientific_ablation_writer_bundle_v1
            )
            assert (
                artifacts.IrisScientificAblationWriterSourceV1
                is importlib.import_module(
                    "iris.artifacts.scientific_ablation_writer_source_v1"
                ).IrisScientificAblationWriterSourceV1
            )
    finally:
        sys.setprofile(previous)
    assert torch.equal(before, torch.get_rng_state())
    assert defaults == (
        torch.get_default_dtype(),
        torch.get_default_device(),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
    )
    count = 0
    for name, module in tuple(sys.modules.items()):
        if name == "iris" or name.startswith("iris."):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                assert Path(filename).resolve().is_relative_to(ROOT), name
                count += 1
    assert not any(name == "tests" or name.startswith("tests.") for name in sys.modules)
    sys.stdout.buffer.write(
        (
            json.dumps(
                dict(
                    kind="iris-writer-public-cold-import-v1",
                    first_namespace=first,
                    pid=os.getpid(),
                    parent_pid=os.getppid(),
                    project_bindings=count,
                    all_project_bindings_staged=True,
                    no_model_native_candidate_work=True,
                    caller_torch_state_unchanged=True,
                    test_package_modules_imported=False,
                    full_main_or_scientific_acceptance=False,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
