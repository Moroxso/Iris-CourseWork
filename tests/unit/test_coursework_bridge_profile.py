"""Academic teacher gradient alignment and profile admission, no model execution."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_profile_test", ROOT / "scripts" / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_labels_and_latent_only_gradient() -> None:
    import torch

    prompt = torch.randn(1, 4, 3, requires_grad=True)
    latent = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.randn(1, 2, 3, requires_grad=True)
    ids = torch.tensor([[7, 9]])
    module = load_script("coursework_prefix")
    sequence = module.chat_boundary_teacher_sequence(prompt, latent, 2, ids, target)
    assert sequence.labels.tolist() == [[-100] * 6 + [7, 9]]
    # HF shifts labels internally: the last prefix position predicts the first target.
    assert sequence.labels[:, 1:].tolist() == [[-100] * 5 + [7, 9]]
    assert torch.equal(sequence.inputs_embeds[:, 2:4], latent)
    assert torch.equal(sequence.inputs_embeds[:, 6:], target)
    assert sequence.attention_mask.tolist() == [[True] * 8]
    assert sequence.position_ids.tolist() == [list(range(8))]
    sequence.inputs_embeds.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))
    assert prompt.grad is None and target.grad is None
    assert not module.chat_boundary_prefix(prompt, latent, 2).inputs_embeds.requires_grad


@pytest.mark.parametrize("invalid", ["dtype", "shape", "nan", "empty"])
def test_teacher_rejects_invalid_target(invalid: str) -> None:
    import torch

    prompt = torch.zeros(1, 4, 3)
    latent = torch.zeros(1, 2, 3)
    ids = torch.tensor([[7, 9]])
    target = torch.zeros(1, 2, 3)
    if invalid == "dtype":
        ids = ids.float()
    elif invalid == "shape":
        target = torch.zeros(1, 2, 4)
    elif invalid == "nan":
        target[0, 0, 0] = float("nan")
    elif invalid == "empty":
        ids = ids[:, :0]
        target = target[:, :0]
    with pytest.raises(ValueError):
        load_script("coursework_prefix").chat_boundary_teacher_sequence(
            prompt, latent, 2, ids, target
        )


def test_memory_snapshot_reports_own_process() -> None:
    memory = load_script("coursework_bridge_profile").memory_snapshot()
    assert memory["rss_bytes"] > 0
    assert memory["peak_working_set_bytes"] >= memory["rss_bytes"]
    assert memory["available_ram_bytes"] > 0


def test_low_memory_refuses_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("coursework_bridge_profile")
    monkeypatch.setattr(module, "memory_snapshot", lambda: dict(available_ram_bytes=1024))
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--snapshot",
            str(tmp_path / "c1899de289a04d12100db370d81485cdf75e47ca"),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="6 GiB"):
        module.main()
    assert not output.exists()


def test_training_prompt_does_not_reuse_fixed_demonstration_input() -> None:
    profile = load_script("coursework_bridge_profile")
    cases = load_script("coursework_suite").load_cases(
        ROOT / "docs/examples/coursework-cases-v1.json"
    )
    assert all(c["prompt"] != profile.TRAIN_PROMPT for c in cases)


@pytest.mark.parametrize("steps", [0, 5, -1])
def test_step_budget_refuses_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: int,
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile", "--snapshot", str(tmp_path), "--out", str(output), "--steps", str(steps)],
    )
    with pytest.raises(ValueError, match="four total"):
        load_script("coursework_bridge_profile").main()
    assert not output.exists()


def test_relative_resume_refuses_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--snapshot",
            str(tmp_path),
            "--out",
            str(output),
            "--resume",
            "relative-checkpoint",
        ],
    )
    with pytest.raises(ValueError, match="Absolute resume"):
        load_script("coursework_bridge_profile").main()
    assert not output.exists()
