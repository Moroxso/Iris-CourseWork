"""Offline arbitrary-prompt demo with an explicitly verified trained bridge."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GeneratePrompt = Callable[[str, Path], dict[str, Any]]
ProgressSink = Callable[[str, dict[str, Any]], None]


@contextmanager
def watch_workspace(
    request: Any, progress: ProgressSink, component: Callable[[str], None] | None = None
) -> Iterator[None]:
    """Expose validated cycle metadata from this one owned request only.

    Adapt the pinned producer's existing diagnostic boundary without changing
    the producer sources bound into the trained checkpoint or patching a class.
    """
    from coursework_live_viewer import validate_vectors
    from coursework_visuals import component_for_stage

    from iris.workspace.trace import WorkspaceTraceRecorderV1

    if "_run_stage" in vars(request):
        raise ValueError("Request stage observer already present")
    run_stage = request._run_stage

    def observe(name: str, operation: Callable[[], Any], **options: Any) -> Any:
        if component is not None and (node := component_for_stage(name)) is not None:
            component(node)
        result = run_stage(name, operation, **options)
        if name.startswith("diagnostic trace for cycle "):
            if not isinstance(operation, partial):
                raise ValueError("Unexpected coursework diagnostic operation")
            method = operation.func
            recorder = getattr(method, "__self__", None)
            if (
                type(recorder) is not WorkspaceTraceRecorderV1
                or getattr(method, "__func__", None) is not WorkspaceTraceRecorderV1.record
                or not recorder.rows
            ):
                raise ValueError("Unexpected coursework diagnostic boundary")
            row = recorder.rows[-1].model_dump(
                mode="json",
                include={
                    "cycle",
                    "module_bids",
                    "module_confidence",
                    "selected_modules",
                    "selected_slots",
                    "write_modules",
                    "slot_ages",
                    "slot_ttl",
                    "slot_norms",
                    "slot_sources",
                    "slot_occupancy",
                    "read_slot_occupancy",
                    "read_slot_sources",
                    "read_attention",
                },
            )
            arguments = operation.keywords
            proposed = arguments["proposal"].content.detach()
            stored = arguments["workspace"].slots.detach()
            row["winning_vectors"] = [
                dict(
                    module=module,
                    slot=slot,
                    proposal=proposed[0, module].tolist(),
                    stored=stored[0, slot].tolist(),
                )
                for module, slot in zip(row["write_modules"], row["selected_slots"], strict=True)
            ]
            validate_vectors(row, row["winning_vectors"])
            progress("cycle", dict(row=row, cycle_count=recorder.count))
        return result

    request._run_stage = observe
    try:
        yield
    finally:
        del request._run_stage


class PromptValidationError(ValueError):
    """An input rejected before executing or allocating a request."""


def validate_prompt(prompt: str) -> None:
    if not prompt.strip() or len(prompt.encode("utf-8")) > 32768:
        raise PromptValidationError("Nonempty prompt at most 32768 UTF-8 bytes required")


def snapshot_latent_metadata(prefix: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Observe the exact generation prefix, not a second bridge computation."""
    import torch

    values = prefix.inputs_embeds
    slots = context["workspace"].slots
    start = context["boundary"]
    prompt_length = int(context["ids"].shape[1])
    count = int(slots.shape[1])
    if (
        values.ndim != 3
        or values.shape[0] != 1
        or values.dtype != torch.float32
        or values.device.type != "cpu"
        or values.shape[1] != prompt_length + count
        or type(start) is not int
        or not 0 < start < prompt_length
    ):
        raise ValueError("Exact single-request CPU academic prefix required")
    stop = start + count
    vectors = values[:, start:stop].detach().contiguous()
    positions = prefix.position_ids[:, start:stop].detach().tolist()[0]
    mask = prefix.attention_mask[:, start:stop].detach().tolist()[0]
    if (
        vectors.shape[1] != count
        or not bool(torch.isfinite(vectors).all())
        or positions != list(range(start, stop))
        or mask != [True] * count
    ):
        raise ValueError("Finite observed latent span/positions/mask required")
    return dict(
        kind="iris-coursework-latent-token-metadata-v1",
        observed_at="IrisGenerator.generate-input",
        source_cycle=len(context["states"]),
        num_tokens=count,
        embedding_dim=int(values.shape[2]),
        dtype=str(values.dtype),
        prompt_length=prompt_length,
        prefix_length=int(values.shape[1]),
        span_start=start,
        span_end_exclusive=stop,
        positions=positions,
        attention_mask=mask,
        slot_indices=list(range(count)),
        slot_occupancy=context["workspace"].occupancy.tolist()[0],
        slot_sources=context["workspace"].source_module.tolist()[0],
        norms=torch.linalg.vector_norm(vectors.double(), dim=-1).tolist()[0],
        embedding_sha256=sha256(vectors.numpy().tobytes()).hexdigest(),
        vocabulary_token_ids=False,
    )


class ObservedGenerator:
    """Read-only decorator: preserve the same prefix object, options and output."""

    def __init__(
        self, delegate: Any, context: dict[str, Any], *, progress: ProgressSink | None = None
    ) -> None:
        self.delegate = delegate
        self.context = context
        self.observation: dict[str, Any] | None = None
        self.progress = progress

    def generate(self, prefix: Any, **options: Any) -> Any:
        if self.observation is not None:
            raise ValueError("One generation observation per request required")
        self.observation = snapshot_latent_metadata(prefix, self.context)
        if self.progress is not None:
            self.progress("generation", dict(latents=self.observation))
        return self.delegate.generate(prefix, **options)


def verify_base_provenance(expected: dict[str, Any], recorded: dict[str, Any]) -> None:
    """Compare independently measured inputs; never adopt checkpoint-provided paths/configs."""
    base = {k: v for k, v in recorded.items() if k not in {"prepared", "producer_sha256"}}
    if json.dumps(base, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Checkpoint model/data/config/source provenance mismatch")


def main(
    argv: list[str] | None = None,
    *,
    serve: Callable[[GeneratePrompt], None] | None = None,
    quiet: bool = False,
    progress: ProgressSink | None = None,
    comparison_budget: int | None = None,
) -> int:
    if comparison_budget is not None and (
        type(comparison_budget) is not int or not 1 <= comparison_budget <= 32 or serve is None
    ):
        raise ValueError("Comparison requires a session and a bounded shared decoding budget")
    process_began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", required=serve is None)
    args = parser.parse_args(argv)
    if args.prompt is not None:
        validate_prompt(args.prompt)
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute output required")
    if not args.checkpoint.is_absolute():
        raise ValueError("Explicit absolute checkpoint required")
    manifest_path = args.checkpoint / "manifest.json"
    if manifest_path.stat().st_size > 512 * 1024:
        raise ValueError("Bounded checkpoint manifest required")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    recorded = manifest.get("provenance")
    if not isinstance(recorded, dict):
        raise ValueError("Explicit training provenance required")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from coursework_bridge_profile import memory_snapshot
    from coursework_bridge_train import DATA, generate_from_context, load_data
    from coursework_demo import REVISION, render_prompt, save_new
    from coursework_suite import CASES, file_identity

    if not args.snapshot.is_absolute() or args.snapshot.name != REVISION:
        raise ValueError("Pinned absolute snapshot required")
    if memory_snapshot()["available_ram_bytes"] < 6 * 1024**3:
        raise ValueError("At least 6 GiB available RAM required")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    import torch
    import yaml  # type: ignore[import-untyped]
    from coursework_bridge_checkpoint import restore_checkpoint
    from coursework_prefix import chat_boundary_prefix
    from coursework_trace_viewer import render_report
    from transformers import AutoTokenizer

    from iris.application.composition import build_request_orchestrator
    from iris.backbone import BackboneAdapter
    from iris.generation import TemperatureTopPSamplingConfig
    from iris.generation.facade import IrisGenerator
    from iris.generation.outputs import SecondPassGenerationPrefixOutput
    from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
    from iris.utils.devices import inspect_device
    from iris.workspace.state import WorkspaceState
    from iris.workspace.trace import WorkspaceTraceOptionsV1

    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    config_root = ROOT / "src/iris/configs"
    configs = {
        group: yaml.safe_load((config_root / group / (name + ".yaml")).read_text("utf-8"))
        for group, name in (
            ("backbone", "qwen3_0_6b"),
            ("cognitive", "iris_gw_0_1"),
            ("workspace", "iris_gw_0_1"),
        )
    }
    configs["backbone"].update(
        repository=str(args.snapshot),
        local_files_only=True,
        limits=dict(max_prompt_tokens=1024, max_new_tokens=1024),
    )
    rows = load_data(DATA)
    paths = [ROOT / "scripts/coursework_bridge_train.py", DATA, CASES]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "coursework_demo.py",
            "coursework_suite.py",
            "coursework_prefix.py",
            "coursework_bridge_profile.py",
            "coursework_bridge_checkpoint.py",
        )
    ]
    paths += sorted((ROOT / "src/iris").rglob("*.py"))
    paths += sorted(config_root.rglob("*.yaml"))
    expected = dict(
        kind="iris-coursework-bridge-tutorial-v1",
        examples=rows,
        data=file_identity(DATA),
        model_revision=REVISION,
        snapshot_files={
            name: file_identity(args.snapshot / name)
            for name in (
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            )
        },
        sources={str(p.relative_to(ROOT)): file_identity(p) for p in paths},
        configs=configs,
        versions={n: version(n) for n in ("torch", "transformers", "safetensors", "PyYAML")},
        seed=42,
        cpu_threads=4,
        prefix_layout="chat-boundary",
        bridge_init_scale=0.025,
        data_order="train-only-round-robin-no-shuffle",
        learning_rate=0.0001,
        weight_decay=0.0,
        clip_norm=1.0,
    )
    verify_base_provenance(expected, recorded)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.snapshot), local_files_only=True, trust_remote_code=False
    )
    if args.prompt is not None:
        live_prompt = render_prompt(tokenizer, args.prompt, raw=False)
        if len(tokenizer.encode(live_prompt, add_special_tokens=False)) > 1024:
            raise PromptValidationError(
                "Rendered prompt exceeds 1024 tokens; no truncation allowed"
            )
    args.out.mkdir(parents=True)
    began = time.perf_counter()
    stages: list[dict[str, Any]] = []

    def stage(name: str) -> None:
        sample: dict[str, Any] = dict(
            stage=name, seconds=time.perf_counter() - began, **memory_snapshot()
        )
        stages.append(sample)
        save_new(args.out / (name + ".json"), sample)
        if not quiet:
            print(json.dumps(sample), flush=True)
        if int(sample["rss_bytes"]) > 8 * 1024**3:
            raise MemoryError("Observed RSS exceeded 8 GiB")

    try:
        backbone = BackboneAdapter(
            BackboneConfig.model_validate(configs["backbone"]), inspect_device("cpu")
        )
        versions = [p._version for p in backbone._model.parameters()]
        request = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=CognitiveConfig.model_validate(configs["cognitive"]),
            workspace_config=WorkspaceConfig.model_validate(configs["workspace"]),
            stop_token_ids=backbone.eos_token_ids,
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )
        fingerprint = request.sidecar_state_sha256
        comparison_generator = (
            IrisGenerator(
                backbone,
                max_new_tokens=comparison_budget,
                stop_token_ids=backbone.eos_token_ids,
                sampling=request._iris_generator.sampling,
            )
            if comparison_budget is not None
            else None
        )
        stage("loaded")

        def context_for(
            prompt: str, identifier: str, checkpoint_id: str
        ) -> tuple[dict[str, Any], Any]:
            execution = request.execute(
                prompt,
                strategy="greedy",
                diagnostics=WorkspaceTraceOptionsV1(
                    mode="metadata", example_id=identifier, checkpoint_id=checkpoint_id, seed=42
                ),
            )
            head, marker, tail = prompt.rpartition("<|im_start|>assistant\n")
            head_ids = tokenizer.encode(head, add_special_tokens=False)
            if (
                not marker
                or head_ids + tokenizer.encode(marker + tail, add_special_tokens=False)
                != execution.prompt_input_ids[0].tolist()
            ):
                raise ValueError("Exact chat token boundary required")
            state = WorkspaceState(
                **{
                    f.name: getattr(execution.generation.workspace_trace[-1], f.name)
                    .detach()
                    .clone()
                    for f in fields(WorkspaceState)
                }
            )
            return dict(
                ids=execution.prompt_input_ids.detach().clone(),
                mask=execution.prompt_attention_mask.detach().clone(),
                workspace=state,
                boundary=len(head_ids),
                states=execution.generation.workspace_trace,
            ), execution

        # Reconstruct all pinned training contexts to validate producer/data bindings.
        # Targets are tokenized for verification only and never given to generation.
        prepared = []
        for row in rows:
            prompt = render_prompt(tokenizer, row["prompt"], raw=False)
            context, execution = context_for(prompt, row["id"], "untrained-producer")
            assert execution.diagnostic_trace is not None
            trace_path = args.out / ("verification-" + row["id"] + ".json")
            save_new(trace_path, execution.diagnostic_trace.model_dump(mode="json"))
            prepared.append(
                dict(
                    id=row["id"],
                    split=row["split"],
                    rendered_prompt=prompt,
                    prompt_ids=context["ids"][0].tolist(),
                    target_ids=[
                        *tokenizer.encode(row["target"], add_special_tokens=False),
                        tokenizer.eos_token_id,
                    ],
                    boundary=context["boundary"],
                    trace=file_identity(trace_path),
                )
            )
        expected.update(producer_sha256=fingerprint, prepared=prepared)
        bridge = copy.deepcopy(request._latent_bridge)
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=0.0001, weight_decay=0.0)
        completed_steps = restore_checkpoint(
            args.checkpoint, bridge, optimizer, expected_provenance=expected
        )
        bridge_versions = [p._version for p in bridge.parameters()]
        checkpoint: dict[str, Any] = dict(
            path=str(args.checkpoint),
            completed_steps=completed_steps,
            manifest=file_identity(manifest_path),
            state=file_identity(args.checkpoint / "state.safetensors"),
        )
        save_new(args.out / "checkpoint-receipt.json", checkpoint)
        stage("verified")
        startup_seconds = time.perf_counter() - process_began

        def generate_prompt(prompt: str, out: Path) -> dict[str, Any]:
            validate_prompt(prompt)
            if not out.is_absolute() or ".." in out.parts:
                raise ValueError("Fresh absolute request output required")
            if out.exists() and not (serve is None and out == args.out):
                raise ValueError("Fresh request output required")
            rendered = render_prompt(tokenizer, prompt, raw=False)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) > 1024:
                raise PromptValidationError(
                    "Rendered prompt exceeds 1024 tokens; no truncation allowed"
                )
            request_began = time.perf_counter()
            out.mkdir(parents=True, exist_ok=serve is None and out == args.out)
            cycle_vectors: list[list[dict[str, Any]]] = []
            components = ["input"]

            def component(name: str) -> None:
                if name not in components:
                    components.append(name)
                if progress is not None:
                    progress("component", dict(name=name))

            def capture(event: str, payload: dict[str, Any]) -> None:
                cycle_vectors.append(payload["row"]["winning_vectors"])
                if progress is not None:
                    progress(event, payload)

            if progress is not None:
                progress("first-pass", dict(prompt=prompt))
            with watch_workspace(request, capture, component):
                context, execution = context_for(
                    rendered, "coursework-trained-demo", checkpoint["manifest"]["sha256"]
                )
            assert execution.diagnostic_trace is not None
            save_new(
                out / "workspace-trace.json", execution.diagnostic_trace.model_dump(mode="json")
            )
            component("bridge")
            observed = ObservedGenerator(
                comparison_generator or request._iris_generator, context, progress=progress
            )
            generation = generate_from_context(backbone, observed, bridge, context)
            components.append("decode")
            if observed.observation is None:
                raise ValueError("Actual generation prefix observation missing")
            save_new(out / "latent-tokens.json", observed.observation)
            controls: dict[str, Any] = {}
            if comparison_generator is not None:
                # No extra positions for Qwen; the empty control keeps the same
                # trained bridge, positions, prompt, decoder and decoding budget.
                with torch.inference_mode():
                    ids = context["ids"]
                    mask = context["mask"].to(torch.bool)
                    embeddings = backbone.embed_tokens(ids)
                    raw = SecondPassGenerationPrefixOutput(
                        inputs_embeds=embeddings,
                        attention_mask=mask,
                        position_ids=torch.arange(ids.shape[-1], device=ids.device).unsqueeze(0),
                    )
                    state = context["workspace"]
                    empty = WorkspaceState.empty(
                        batch_size=1,
                        num_slots=state.slots.shape[1],
                        state_dim=state.slots.shape[2],
                        dtype=state.slots.dtype,
                        device=state.slots.device,
                    )
                    empty_vectors = bridge(empty, mask).workspace_embeddings
                    empty_prefix = chat_boundary_prefix(
                        embeddings, empty_vectors, context["boundary"]
                    )
                    for label, prefix in (("qwen", raw), ("empty_workspace", empty_prefix)):
                        control = comparison_generator.generate(
                            prefix, strategy="greedy", workspace_trace=()
                        )
                        controls[label] = dict(
                            text=control.text,
                            stop_reason=control.stop_reason,
                            token_ids=control.token_ids.tolist(),
                            positions=prefix.inputs_embeds.shape[1],
                        )
                    controls["settings"] = dict(
                        max_new_tokens=comparison_budget,
                        prompt_tokens=ids.shape[1],
                        iris_positions=observed.observation["positions"][-1]
                        + (ids.shape[1] - context["boundary"])
                        + 1,
                        filled_slots=int(state.occupancy.sum()),
                        empty_latents_sha256=sha256(
                            empty_vectors.contiguous().cpu().numpy().tobytes()
                        ).hexdigest(),
                        iris_latents_sha256=observed.observation["embedding_sha256"],
                    )
            if (
                request.sidecar_state_sha256 != fingerprint
                or backbone.trainable_parameter_count != 0
                or backbone.has_parameter_gradients
                or versions != [p._version for p in backbone._model.parameters()]
                or any(
                    p.grad is not None for _, m in request._sidecar_modules for p in m.parameters()
                )
                or any(p.grad is not None for p in bridge.parameters())
                or bridge_versions != [p._version for p in bridge.parameters()]
            ):
                raise ValueError("Original request/backbone boundary changed")
            sample: dict[str, Any] = dict(
                stage="generated", seconds=time.perf_counter() - began, **memory_snapshot()
            )
            save_new(out / "generated.json", sample)
            if int(sample["rss_bytes"]) > 8 * 1024**3:
                raise MemoryError("Observed RSS exceeded 8 GiB")
            report = dict(
                kind="iris-coursework-trained-bridge-demo-v1",
                completed=True,
                prompt=prompt,
                rendered_prompt=rendered,
                system_prompt="",
                text=generation.text,
                token_ids=generation.token_ids.tolist(),
                stop_reason=generation.stop_reason,
                seed=42,
                prefix_layout="chat-boundary",
                device="cpu",
                dtype="fp32",
                backbone_pretrained=True,
                backbone_frozen=True,
                bridge_trained=True,
                cognitive_components_trained=False,
                sidecar_fully_trained=False,
                optimizer_updates_performed=0,
                workspace_trace=file_identity(out / "workspace-trace.json"),
                workspace_vectors=cycle_vectors,
                architecture_components=components,
                latent_tokens=observed.observation,
                latent_trace=file_identity(out / "latent-tokens.json"),
                checkpoint=checkpoint,
                original_request_unchanged=True,
                trace_producer="untrained-cognitive-modules",
                scientific_acceptance=False,
                quality_improvement_proven=False,
                revision=REVISION,
                source_provenance_verified=True,
                resource_stages=[*stages, sample],
                preflight_seconds=began - process_began,
                inference_phase_seconds=time.perf_counter()
                - (began if serve is None else request_began),
                total_seconds=time.perf_counter()
                - (process_began if serve is None else request_began),
                session_reused=serve is not None,
                session_startup_seconds=startup_seconds,
                consumer_sources={
                    p.name: file_identity(p)
                    for p in (
                        Path(__file__),
                        ROOT / "scripts/coursework_trace_viewer.py",
                        ROOT / "scripts/coursework_live_viewer.py",
                        ROOT / "scripts/coursework_visuals.py",
                    )
                },
            )
            if controls:
                report["comparison"] = controls
            page = render_report(report, execution.diagnostic_trace.model_dump(mode="json"))
            with (out / "viewer.html").open("x", encoding="utf-8") as stream:
                stream.write(page)
            if out != args.out:
                save_new(out / "checkpoint-receipt.json", checkpoint)
            save_new(out / "result.json", report)
            return report

        if serve is not None:
            serve(generate_prompt)
        else:
            report = generate_prompt(args.prompt, args.out)
            print(
                json.dumps(
                    dict(completed=True, text=report["text"], completed_steps=completed_steps),
                    ensure_ascii=True,
                ),
                flush=True,
            )
        return 0
    except BaseException as error:
        save_new(
            args.out / "error.json",
            dict(type=type(error).__name__, message=str(error), stages=stages),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
