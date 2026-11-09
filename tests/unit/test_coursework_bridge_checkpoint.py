"""CPU bridge resume contract on tiny real modules, without loading any LLM."""

import importlib.util
import json
import random
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from iris.workspace import LatentBridge

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = dict(model="local-fixture", layout="chat-boundary", dataset="train-fixture-v1")


def api() -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_checkpoint_test", ROOT / "scripts/coursework_bridge_checkpoint.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner() -> tuple[LatentBridge, torch.optim.AdamW]:
    bridge = LatentBridge(state_dim=3, backbone_hidden_size=4, num_slots=2)
    return bridge, torch.optim.AdamW(bridge.parameters(), lr=0.0001, weight_decay=0.0)


def update(bridge: LatentBridge, optimizer: torch.optim.AdamW) -> float:
    # Consume both RNGs so exact continuation cannot pass by just copying weights.
    loss = sum(
        (
            (p * (torch.randn_like(p) + random.random())).square().mean()
            for p in bridge.parameters()
        ),
        torch.zeros(()),
    )
    value = float(loss.detach())
    torch.autograd.backward((loss,))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return value


def seeded_checkpoint(tmp_path: Path) -> tuple[Any, Path, LatentBridge, torch.optim.AdamW]:
    torch.manual_seed(42)
    random.seed(42)
    bridge, optimizer = owner()
    update(bridge, optimizer)
    checkpoint = tmp_path / "checkpoint"
    module = api()
    module.write_checkpoint(checkpoint, bridge, optimizer, completed_steps=1, provenance=PROVENANCE)
    return module, checkpoint, bridge, optimizer


def snapshot(
    bridge: LatentBridge, optimizer: torch.optim.AdamW
) -> tuple[dict[str, torch.Tensor], bytes, object]:
    tensors = {n: p.detach().clone() for n, p in bridge.named_parameters()}
    for i, p in enumerate(bridge.parameters()):
        for key, value in optimizer.state.get(p, {}).items():
            tensors[str(i) + key] = value.detach().clone()
    return tensors, torch.get_rng_state().numpy().tobytes(), random.getstate()


def assert_unchanged(
    before: tuple[dict[str, torch.Tensor], bytes, object],
    bridge: LatentBridge,
    optimizer: torch.optim.AdamW,
) -> None:
    after = snapshot(bridge, optimizer)
    assert before[0].keys() == after[0].keys()
    assert all(torch.equal(before[0][n], after[0][n]) for n in before[0])
    assert before[1:] == after[1:]


def test_exact_next_step_and_rng_restore(tmp_path: Path) -> None:
    module, checkpoint, bridge, optimizer = seeded_checkpoint(tmp_path)
    expected_loss = update(bridge, optimizer)
    expected = {n: t.clone() for n, t in bridge.state_dict().items()}
    expected_torch, expected_python = torch.rand(5), random.random()
    # Restore into unrelated initial values/RNGs, not a second view of the owner.
    torch.manual_seed(991)
    random.seed(992)
    restored, restored_optimizer = owner()
    assert (
        module.restore_checkpoint(
            checkpoint, restored, restored_optimizer, expected_provenance=PROVENANCE
        )
        == 1
    )
    assert update(restored, restored_optimizer) == expected_loss
    assert all(torch.equal(expected[n], t) for n, t in restored.state_dict().items())
    assert torch.equal(torch.rand(5), expected_torch)
    assert random.random() == expected_python
    for a, b in zip(bridge.parameters(), restored.parameters(), strict=True):
        assert optimizer.state[a].keys() == restored_optimizer.state[b].keys()
        assert all(
            torch.equal(v, restored_optimizer.state[b][k]) for k, v in optimizer.state[a].items()
        )


@pytest.mark.parametrize("failure", ["checksum", "provenance", "runtime", "cursor", "nan"])
def test_invalid_checkpoint_refuses_without_mutation(tmp_path: Path, failure: str) -> None:
    module, checkpoint, _, _ = seeded_checkpoint(tmp_path)
    bridge, optimizer = owner()
    before = snapshot(bridge, optimizer)
    expected = PROVENANCE.copy()
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if failure == "checksum":
        path = checkpoint / "state.safetensors"
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif failure == "provenance":
        expected["dataset"] = "wrong-data"
    elif failure == "runtime":
        manifest["cpu_threads"] += 1
    elif failure == "cursor":
        manifest["completed_steps"] = 2
    elif failure == "nan":
        path = checkpoint / "state.safetensors"
        tensors = load_tensors(path.read_bytes())
        tensors["model.normalization.bias"][0] = float("nan")
        raw = save_tensors(tensors)
        path.write_bytes(raw)
        manifest["state_file"].update(bytes=len(raw), sha256=sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        module.restore_checkpoint(checkpoint, bridge, optimizer, expected_provenance=expected)
    assert_unchanged(before, bridge, optimizer)


def test_never_overwrites_and_refuses_uninitialized_state(tmp_path: Path) -> None:
    module, checkpoint, bridge, optimizer = seeded_checkpoint(tmp_path)
    raw = (checkpoint / "state.safetensors").read_bytes()
    with pytest.raises(ValueError, match="Fresh absolute"):
        module.write_checkpoint(
            checkpoint, bridge, optimizer, completed_steps=1, provenance=PROVENANCE
        )
    assert (checkpoint / "state.safetensors").read_bytes() == raw
    fresh, fresh_optimizer = owner()
    path = tmp_path / "not-created"
    with pytest.raises(ValueError, match="initialized"):
        module.write_checkpoint(
            path, fresh, fresh_optimizer, completed_steps=1, provenance=PROVENANCE
        )
    assert not path.exists()


def test_refuses_optimizer_that_does_not_own_bridge(tmp_path: Path) -> None:
    module, checkpoint, _, _ = seeded_checkpoint(tmp_path)
    bridge, _ = owner()
    _, unrelated = owner()
    with pytest.raises(ValueError, match="own every"):
        module.restore_checkpoint(checkpoint, bridge, unrelated, expected_provenance=PROVENANCE)


def test_restore_rolls_back_if_optimizer_installation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, checkpoint, _, _ = seeded_checkpoint(tmp_path)
    bridge, optimizer = owner()
    before = snapshot(bridge, optimizer)
    original = optimizer.load_state_dict
    calls = []

    def fail_once(state: Any) -> None:
        calls.append(state)
        if len(calls) == 1:
            raise RuntimeError("Injected optimizer installation failure")
        original(state)

    monkeypatch.setattr(optimizer, "load_state_dict", fail_once)
    with pytest.raises(RuntimeError, match="Injected"):
        module.restore_checkpoint(checkpoint, bridge, optimizer, expected_provenance=PROVENANCE)
    assert len(calls) == 2
    assert_unchanged(before, bridge, optimizer)
