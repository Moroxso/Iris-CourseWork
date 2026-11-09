"""Tutorial split, training selection and target-free generation; no LLM load."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_training_test", ROOT / "scripts/coursework_bridge_train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_split_and_no_heldout_selection_for_updates() -> None:
    module = script()
    rows = module.load_data(module.DATA)
    assert len([r for r in rows if r["split"] == "train"]) == 3
    assert len([r for r in rows if r["split"] == "heldout"]) == 3
    assert [module.training_case(rows, i)["id"] for i in range(1, 7)] == [
        "train-color",
        "train-password",
        "train-addition",
    ] * 2
    assert all(module.training_case(rows, i)["split"] == "train" for i in range(1, 13))


@pytest.mark.parametrize("failure", ["overlap", "id", "missing-split", "empty-target"])
def test_refuses_invalid_split(tmp_path: Path, failure: str) -> None:
    module = script()
    data = json.loads(module.DATA.read_text("utf-8"))
    if failure == "overlap":
        data["examples"][3]["prompt"] = "  " + data["examples"][0]["prompt"].upper() + "\n"
    elif failure == "id":
        data["examples"][0]["id"] = "../unsafe"
    elif failure == "missing-split":
        for row in data["examples"]:
            row["split"] = "train"
    else:
        data["examples"][0]["target"] = " "
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_data(path)


def test_generation_does_not_receive_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    embeds = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    latent = torch.full((1, 2, 3), 99.0)
    states = ("actual-workspace-trace",)
    context = dict(
        ids=torch.tensor([[1, 2, 3, 4]]),
        mask=torch.ones(1, 4, dtype=torch.bool),
        workspace="workspace",
        boundary=2,
        states=states,
    )

    # There are deliberately no targets in the generation context.
    class Backbone:
        def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
            assert torch.equal(ids, context["ids"])
            return embeds

    def bridge(workspace: object, mask: torch.Tensor) -> Any:
        assert workspace == "workspace" and not torch.is_grad_enabled()
        return SimpleNamespace(workspace_embeddings=latent)

    class Generator:
        def generate(self, prefix: Any, **options: Any) -> str:
            assert options == dict(strategy="greedy", workspace_trace=states)
            assert torch.equal(prefix.inputs_embeds[:, 2:4], latent)
            assert prefix.inputs_embeds.shape == (1, 6, 3)
            return "generated-not-a-label"

    assert (
        script().generate_from_context(Backbone(), Generator(), bridge, context)
        == "generated-not-a-label"
    )


@pytest.mark.parametrize("flags", [["--steps", "0"], ["--steps", "13"], ["--evaluate-only"]])
def test_invalid_cli_refuses_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys, "argv", ["train", "--snapshot", str(tmp_path), "--out", str(output), *flags]
    )
    with pytest.raises(ValueError):
        script().main()
    assert not output.exists()
