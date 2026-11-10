"""Trained demo admission/provenance and truthful HTML labels; no LLM load."""

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name: str = "coursework_trained_demo") -> Any:
    spec = importlib.util.spec_from_file_location(
        "coursework_trained_demo_test", ROOT / "scripts" / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("prompt", ["", " ", "x" * 32769], ids=["empty", "blank", "too-long"])
def test_invalid_prompt_refuses_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--snapshot",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path),
            "--out",
            str(output),
            "--prompt",
            prompt,
        ],
    )
    with pytest.raises(ValueError, match="prompt"):
        module().main()
    assert not output.exists()


@pytest.mark.parametrize("field", ["model", "data", "configs", "sources"])
def test_provenance_mismatch_never_adopts_checkpoint_inputs(field: str) -> None:
    expected = {k: "actual" for k in ("model", "data", "configs", "sources")}
    recorded = dict(expected, prepared=[], producer_sha256="producer")
    module().verify_base_provenance(expected, recorded)
    recorded[field] = "checkpoint-supplied-different-value"
    with pytest.raises(ValueError, match="provenance mismatch"):
        module().verify_base_provenance(expected, recorded)


def trained_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    result = dict(
        kind="iris-coursework-trained-bridge-demo-v1",
        completed=True,
        bridge_trained=True,
        cognitive_components_trained=False,
        sidecar_fully_trained=False,
        backbone_frozen=True,
        source_provenance_verified=True,
        optimizer_updates_performed=0,
        checkpoint=dict(
            completed_steps=6, manifest=dict(sha256="a" * 64), state=dict(sha256="b" * 64)
        ),
        text="</script><script>do not execute</script>",
    )
    cycle = {
        k: [0]
        for k in (
            "slot_sources",
            "slot_occupancy",
            "slot_norms",
            "slot_ages",
            "slot_ttl",
            "module_bids",
            "module_confidence",
        )
    }
    trace = dict(
        kind="iris-workspace-diagnostic-trace-v1",
        cycles=[cycle],
        cycle_count=1,
        num_slots=1,
        num_modules=1,
    )
    result["seed"] = 42
    trace["options"] = dict(checkpoint_id="a" * 64, seed=42)
    raw = (
        json.dumps(trace, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    result["workspace_trace"] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    return result, trace


def test_viewer_accepts_explicit_partial_training_and_escapes_text() -> None:
    result, trace = trained_fixture()
    html = module("coursework_trace_viewer").render_report(result, trace)
    embedded = html.split('<script id="trace-data" type="application/json">')[1].split("</script>")[
        0
    ]
    assert json.loads(embedded) == dict(result=result, trace=trace)
    assert result["text"] not in html
    assert 'id="training-status"' in html and 'id="checkpoint"' in html


@pytest.mark.parametrize(
    "field,value",
    [
        ("bridge_trained", False),
        ("cognitive_components_trained", True),
        ("sidecar_fully_trained", True),
        ("source_provenance_verified", False),
        ("optimizer_updates_performed", 1),
        ("checkpoint", None),
    ],
)
def test_viewer_refuses_unproven_or_mislabelled_training(field: str, value: object) -> None:
    result, trace = trained_fixture()
    result[field] = value
    with pytest.raises(ValueError, match="training provenance"):
        module("coursework_trace_viewer").render_report(result, trace)


def test_old_untrained_kind_cannot_claim_trained_bridge() -> None:
    result, trace = trained_fixture()
    result["kind"] = "iris-coursework-two-pass-demo-v1"
    with pytest.raises(ValueError, match="Untrained result kind"):
        module("coursework_trace_viewer").render_report(result, trace)


def test_viewer_refuses_checkpoint_without_content_hash() -> None:
    result, trace = trained_fixture()
    result["checkpoint"]["state"]["sha256"] = "unknown"
    with pytest.raises(ValueError, match="content hashes"):
        module("coursework_trace_viewer").render_report(result, trace)


@pytest.mark.parametrize("failure", ["content", "checkpoint-label", "seed-label"])
def test_viewer_refuses_mixed_trace_and_result(failure: str) -> None:
    result, trace = trained_fixture()
    if failure == "content":
        trace["cycles"][0]["slot_norms"][0] = 12.0
    else:
        trace["options"]["checkpoint_id" if failure == "checkpoint-label" else "seed"] = (
            "b" * 64 if failure == "checkpoint-label" else 999
        )
        # A recomputed digest must not hide a mismatched checkpoint/seed label.
        raw = (
            json.dumps(trace, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        result["workspace_trace"] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="trace content/checkpoint binding"):
        module("coursework_trace_viewer").render_report(result, trace)


def latent_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    result, trace = trained_fixture()
    trace["cycles"][0]["slot_occupancy"] = [True]
    raw = (
        json.dumps(trace, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    result["workspace_trace"] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    metadata = dict(
        kind="iris-coursework-latent-token-metadata-v1",
        observed_at="IrisGenerator.generate-input",
        source_cycle=1,
        num_tokens=1,
        embedding_dim=3,
        dtype="torch.float32",
        vocabulary_token_ids=False,
        span_start=1,
        span_end_exclusive=2,
        prompt_length=2,
        prefix_length=3,
        positions=[1],
        slot_indices=[0],
        slot_sources=[0],
        slot_occupancy=[True],
        attention_mask=[True],
        norms=[5.0],
        embedding_sha256="c" * 64,
    )
    result["latent_tokens"] = metadata
    raw = (
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    result["latent_trace"] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    return result, trace


def test_latent_metadata_is_embedded_without_changing_workspace_trace() -> None:
    result, trace = latent_fixture()
    html = module("coursework_trace_viewer").render_report(result, trace)
    assert 'id="latents"' in html and 'id="latent-summary"' in html
    embedded = html.split('<script id="trace-data" type="application/json">')[1].split("</script>")[
        0
    ]
    assert json.loads(embedded) == dict(result=result, trace=trace)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_cycle", 2),
        ("num_tokens", True),
        ("positions", [2]),
        ("attention_mask", [1]),
        ("slot_occupancy", [False]),
        ("slot_sources", [-1]),
        ("norms", [float("nan")]),
        ("norms", [-1.0]),
        ("norms", []),
        ("embedding_sha256", "unknown"),
        ("vocabulary_token_ids", True),
    ],
)
def test_invalid_latent_metadata_refuses(field: str, value: object) -> None:
    result, trace = latent_fixture()
    result["latent_tokens"][field] = value
    with pytest.raises(ValueError):
        module("coursework_trace_viewer").render_report(result, trace)


def test_latent_metadata_digest_is_required() -> None:
    result, trace = latent_fixture()
    result["latent_trace"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content binding"):
        module("coursework_trace_viewer").render_report(result, trace)


def test_generator_observation_is_read_only_and_uses_actual_prefix() -> None:
    import torch

    values = torch.tensor([[[9.0, 9.0, 9.0], [3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [8.0, 8.0, 8.0]]])
    prefix = SimpleNamespace(
        inputs_embeds=values,
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        position_ids=torch.arange(4).unsqueeze(0),
    )
    workspace = SimpleNamespace(
        slots=torch.zeros(1, 2, 2),
        occupancy=torch.tensor([[True, False]]),
        source_module=torch.tensor([[3, -1]]),
    )
    context = dict(workspace=workspace, ids=torch.tensor([[1, 2]]), boundary=1, states=(workspace,))
    before = values.clone()
    returned = object()

    class Delegate:
        def generate(self, actual: Any, **options: Any) -> Any:
            assert actual is prefix and options == dict(
                strategy="greedy", workspace_trace=(workspace,)
            )
            return returned

    observer = module().ObservedGenerator(Delegate(), context)
    assert observer.generate(prefix, strategy="greedy", workspace_trace=(workspace,)) is returned
    assert torch.equal(values, before)
    assert observer.observation["norms"] == [5.0, 2.0]
    assert observer.observation["positions"] == [1, 2]
    assert observer.observation["slot_sources"] == [3, -1]
    assert observer.observation["attention_mask"] == [True, True]
    assert (
        observer.observation["embedding_sha256"]
        == sha256(values[:, 1:3].contiguous().numpy().tobytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="One generation"):
        observer.generate(prefix)
