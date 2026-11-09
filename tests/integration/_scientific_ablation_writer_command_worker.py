"""Fixed trusted actual cold CLI entrypoint worker; never candidate-generated code."""

import json
import os
import sys
from pathlib import Path
from types import FrameType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main() -> int:
    job_path = Path(sys.argv[1]).resolve(strict=True)
    data = job_path.read_bytes()
    assert 1 <= len(data) <= 64 * 1024
    job = json.loads(data)
    assert set(job) == {"phase", "input_root", "output_root", "artifact_path", "bindings"}
    assert job["phase"] in ("publish", "inspect")
    paths = {
        key: Path(job[key]).resolve() for key in ("input_root", "output_root", "artifact_path")
    }
    assert all(path.is_relative_to(job_path.parent) for path in paths.values())
    bindings = job["bindings"]
    assert set(bindings) == {
        "expected_protocol",
        "expected_report_sha256",
        "expected_source_sha256",
        "runtime",
        "maximum_source_bytes",
        "maximum_report_bytes",
    }
    assert set(bindings["runtime"]) == {"torch_version", "torch_git_version"}
    options = [
        "--protocol",
        bindings["expected_protocol"],
        "--expected-report-sha256",
        bindings["expected_report_sha256"],
        "--expected-source-sha256",
        bindings["expected_source_sha256"],
        "--expected-torch-version",
        bindings["runtime"]["torch_version"],
        "--expected-torch-git-version",
        bindings["runtime"]["torch_git_version"],
        "--maximum-source-bytes",
        str(bindings["maximum_source_bytes"]),
        "--maximum-report-bytes",
        str(bindings["maximum_report_bytes"]),
    ]
    argv = (
        [
            "scientific",
            "publish-writer",
            "--source",
            str(paths["input_root"] / "source.json"),
            "--report",
            str(paths["input_root"] / "report.json"),
            "--output-root",
            str(paths["output_root"]),
            *options,
        ]
        if job["phase"] == "publish"
        else ["scientific", "inspect-writer", str(paths["artifact_path"]), *options]
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

    before = torch.get_rng_state().clone()
    defaults = (
        torch.get_default_dtype(),
        torch.get_default_device(),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
    )
    old_profile = sys.getprofile()
    denied = 0

    def forbid(frame: FrameType, event: str, argument: object) -> None:
        nonlocal denied
        if (
            event == "call"
            and frame.f_code.co_name
            in ("execute_python_job_v2", "_measure_writer_effects", "_source_transaction")
            and Path(frame.f_code.co_filename).resolve().is_relative_to(ROOT)
        ):
            denied += 1
            raise AssertionError("Cold CLI cannot run native/candidate work")

    code = 0
    try:
        with (
            patch.object(
                torch.nn.Module, "__init__", side_effect=AssertionError("No CLI model")
            ) as construct,
            patch.object(
                torch.nn.Module, "__call__", side_effect=AssertionError("No CLI forward")
            ) as call,
            patch.object(
                Qwen3ForCausalLM, "__init__", side_effect=AssertionError("No CLI Qwen")
            ) as qwen,
            patch.object(
                AutoModelForCausalLM,
                "from_pretrained",
                side_effect=AssertionError("No CLI weights"),
            ) as weights,
            patch.object(
                AutoTokenizer, "from_pretrained", side_effect=AssertionError("No CLI tokenizer")
            ) as tokenizer,
        ):
            sys.setprofile(forbid)
            from iris.cli import app

            try:
                app(args=argv, prog_name="iris")
            except SystemExit as result:
                code = (
                    result.code if type(result.code) is int else (0 if result.code is None else 1)
                )
            assert denied == 0
            assert all(mock.call_count == 0 for mock in (construct, call, qwen, weights, tokenizer))
    finally:
        sys.setprofile(old_profile)
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
    print(
        "IRIS_WRITER_COMMAND_EVENT "
        + json.dumps(
            dict(
                kind="iris-writer-command-cold-cli-v1",
                phase=job["phase"],
                actual_cli_exit_code=code,
                pid=os.getpid(),
                parent_pid=os.getppid(),
                project_bindings=count,
                all_project_bindings_staged=True,
                no_model_native_candidate_attempts=True,
                caller_torch_state_unchanged=True,
                test_package_modules_imported=False,
                full_main_or_scientific_acceptance=False,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
