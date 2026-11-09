"""Admission checks for the offline coursework entrypoint; no model execution."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def demo(name: str = "coursework_demo") -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("coursework_demo_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tokens,threads", [(0, 4), (129, 4), (8, 0), (8, 9)])
def test_invalid_budgets_refuse_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokens: int, threads: int
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--snapshot",
            str(tmp_path),
            "--out",
            str(output),
            "--max-new-tokens",
            str(tokens),
            "--threads",
            str(threads),
        ],
    )
    with pytest.raises(ValueError, match="Bounded"):
        demo().main()
    assert not output.exists()


def test_academic_prefix_preserves_real_embeddings_and_positions() -> None:
    import torch

    prompt = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    latents = torch.full((1, 2, 3), 99.0)
    assemble = demo("coursework_prefix").chat_boundary_prefix
    prefix = assemble(prompt, latents, 2)
    assert torch.equal(prefix.inputs_embeds[:, :2], prompt[:, :2])
    assert torch.equal(prefix.inputs_embeds[:, 2:4], latents)
    assert torch.equal(prefix.inputs_embeds[:, 4:], prompt[:, 2:])
    assert prefix.attention_mask.tolist() == [[True] * 6]
    assert prefix.position_ids.tolist() == [list(range(6))]
    for invalid in (0, 4, True):
        with pytest.raises(ValueError, match="boundary"):
            assemble(prompt, latents, invalid)


def test_missing_local_weights_never_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = demo()
    snapshot = tmp_path / module.REVISION
    snapshot.mkdir()
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--snapshot",
            str(snapshot),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Incomplete local"):
        module.main()
    assert not output.exists()


def test_result_writer_never_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        demo().save_new(path, {"completed": True})
    assert path.read_bytes() == b"retained"


def test_chat_template_is_explicit_and_raw_path_does_not_call_it() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: object, **options: object) -> str:
            assert messages == [{"role": "user", "content": "hello"}]
            assert options == dict(
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            return "rendered"

    module = demo()
    assert module.render_prompt(Tokenizer(), "hello", raw=False) == "rendered"
    assert module.render_prompt(object(), "hello", raw=True) == "hello"


def test_system_prompt_is_a_separate_message_not_a_replacement_for_user() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: object, **options: object) -> str:
            assert messages == [
                dict(role="system", content="be concise"),
                dict(role="user", content="hello"),
            ]
            assert options["enable_thinking"] is False
            return "rendered with system"

    module = demo()
    assert module.render_prompt(Tokenizer(), "hello", raw=False, system_prompt="be concise") == (
        "rendered with system"
    )
    with pytest.raises(ValueError, match="requires"):
        module.render_prompt(object(), "hello", raw=True, system_prompt="be concise")


@pytest.mark.parametrize(
    "instruction,flags",
    [
        (" ", []),
        ("x" * 8193, []),
        ("instruction", ["--raw-prompt"]),
    ],
)
def test_invalid_system_prompt_refuses_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
    flags: list[str],
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--snapshot",
            str(tmp_path),
            "--out",
            str(output),
            "--system-prompt",
            instruction,
            *flags,
        ],
    )
    with pytest.raises(ValueError, match="system prompt"):
        demo().main()
    assert not output.exists()


@pytest.mark.parametrize("scale", ["nan", "inf", "0", "1.01"])
def test_invalid_bridge_scale_refuses_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scale: str
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--snapshot",
            str(tmp_path),
            "--out",
            str(output),
            "--bridge-init-scale",
            scale,
        ],
    )
    with pytest.raises(ValueError, match="Bridge initialization"):
        demo().main()
    assert not output.exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--prefix-layout", "chat-boundary", "--raw-prompt"],
        ["--check-workspace-influence"],
    ],
)
def test_incompatible_modes_refuse_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--snapshot",
            str(tmp_path),
            "--out",
            str(output),
            *flags,
        ],
    )
    with pytest.raises(ValueError, match="requires"):
        demo().main()
    assert not output.exists()


def viewer_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    result = dict(
        kind="iris-coursework-two-pass-demo-v1",
        completed=True,
        text="</script><script>alert('model text')</script>&",
    )
    trace = dict(
        kind="iris-workspace-diagnostic-trace-v1",
        cycle_count=1,
        num_slots=1,
        num_modules=1,
        cycles=[
            dict(
                slot_sources=[0],
                slot_occupancy=[True],
                slot_norms=[1.0],
                slot_ages=[0],
                slot_ttl=[2.0],
                module_bids=[0.1],
                module_confidence=[0.5],
            )
        ],
    )
    return result, trace


def test_viewer_preserves_data_without_injecting_model_html() -> None:
    result, trace = viewer_fixture()
    html = demo("coursework_trace_viewer").render_report(result, trace)
    data = html.split('<script id="trace-data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(data) == dict(result=result, trace=trace)
    assert result["text"] not in html
    assert "default-src" in html


@pytest.mark.parametrize("field", ["slot_norms", "module_bids", "module_confidence"])
def test_viewer_rejects_inconsistent_trace_vectors(field: str) -> None:
    result, trace = viewer_fixture()
    trace["cycles"][0][field] = []
    with pytest.raises(ValueError, match="vector length"):
        demo("coursework_trace_viewer").render_report(result, trace)


def test_viewer_rejects_unfinished_result() -> None:
    result, trace = viewer_fixture()
    result["completed"] = False
    with pytest.raises(ValueError, match="Completed"):
        demo("coursework_trace_viewer").render_report(result, trace)
