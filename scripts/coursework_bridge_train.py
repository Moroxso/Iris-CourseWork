"""Bounded academic bridge training and honest before/after held-out generation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/examples/coursework-bridge-data-v1.json"


def load_data(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > 65536:
        raise ValueError("Bounded tutorial data required")
    data = json.loads(path.read_text("utf-8"))
    rows = data.get("examples")
    if data.get("kind") != "iris-coursework-bridge-data-v1" or not isinstance(rows, list):
        raise ValueError("Versioned tutorial data required")
    if not 2 <= len(rows) <= 8:
        raise ValueError("Two to eight examples required")
    ids, prompts = set(), set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "split", "prompt", "target"}
            or not all(isinstance(v, str) for v in row.values())
        ):
            raise ValueError("Exact textual example fields required")
        if (
            re.fullmatch(r"[a-z][a-z0-9-]{0,39}", row["id"]) is None
            or row["id"] in ids
            or row["split"] not in {"train", "heldout"}
        ):
            raise ValueError("Unique safe IDs and explicit splits required")
        normalized = " ".join(row["prompt"].casefold().split())
        if (
            not normalized
            or normalized in prompts
            or len(row["prompt"].encode()) > 8192
            or not row["target"].strip()
            or len(row["target"].encode()) > 256
        ):
            raise ValueError("Disjoint bounded nonempty prompts and targets required")
        ids.add(row["id"])
        prompts.add(normalized)
    if {row["split"] for row in rows} != {"train", "heldout"}:
        raise ValueError("Both train and heldout splits required")
    return rows


def training_case(rows: list[dict[str, str]], completed_step: int) -> dict[str, str]:
    if type(completed_step) is not int or completed_step < 1:
        raise ValueError("Positive exact step required")
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("Training split required")
    return train[(completed_step - 1) % len(train)]


def generate_from_context(
    backbone: Any, generator: Any, bridge: Any, context: dict[str, Any]
) -> Any:
    """Generation deliberately receives no target IDs/embeddings or teacher labels."""
    import torch
    from coursework_prefix import chat_boundary_prefix

    with torch.inference_mode():
        latent = bridge(context["workspace"], context["mask"]).workspace_embeddings
        prefix = chat_boundary_prefix(
            backbone.embed_tokens(context["ids"]), latent, context["boundary"]
        )
        return generator.generate(prefix, strategy="greedy", workspace_trace=context["states"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 12:
        raise ValueError("One to twelve total training steps required")
    if args.evaluate_only and args.resume is None:
        raise ValueError("Evaluate-only requires a checkpoint")
    if args.resume is not None and not args.resume.is_absolute():
        raise ValueError("Absolute checkpoint required")
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute output required")
    rows = load_data(DATA)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from coursework_bridge_profile import memory_snapshot
    from coursework_demo import REVISION, render_prompt, save_new
    from coursework_suite import CASES, file_identity, load_cases, normalize_answer

    original = {" ".join(c["prompt"].casefold().split()) for c in load_cases(CASES)}
    if any(" ".join(r["prompt"].casefold().split()) in original for r in rows):
        raise ValueError("Tutorial must not reuse the original demonstration inputs")
    if not args.snapshot.is_absolute() or args.snapshot.name != REVISION:
        raise ValueError("Pinned absolute snapshot required")
    if memory_snapshot()["available_ram_bytes"] < 6 * 1024**3:
        raise ValueError("At least 6 GiB available RAM required")
    metadata = json.loads((args.snapshot / "config.json").read_text("utf-8"))
    if (
        metadata.get("model_type"),
        metadata.get("hidden_size"),
        metadata.get("num_hidden_layers"),
    ) != ("qwen3", 1024, 28) or (args.snapshot / "model.safetensors").stat().st_size < 1024**3:
        raise ValueError("Actual pretrained Qwen3-0.6B required")
    args.out.mkdir(parents=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    began = time.perf_counter()
    resources = []

    def stage(name: str) -> None:
        sample: dict[str, Any] = dict(
            stage=name, seconds=time.perf_counter() - began, **memory_snapshot()
        )
        resources.append(sample)
        save_new(args.out / (name + ".json"), sample)
        print(json.dumps(sample), flush=True)
        if sample["rss_bytes"] > 8 * 1024**3:
            raise MemoryError("Observed RSS exceeded 8 GiB; no next step allowed")

    try:
        import torch
        import yaml  # type: ignore[import-untyped]
        from coursework_bridge_checkpoint import restore_checkpoint, write_checkpoint
        from coursework_prefix import chat_boundary_teacher_sequence
        from transformers import AutoTokenizer

        from iris.application.composition import build_request_orchestrator
        from iris.backbone import BackboneAdapter
        from iris.generation import TemperatureTopPSamplingConfig
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
            limits=dict(max_prompt_tokens=128, max_new_tokens=32),
        )
        paths = [Path(__file__), DATA, CASES]
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
        provenance = dict(
            kind="iris-coursework-bridge-tutorial-v1",
            examples=rows,
            data=file_identity(DATA),
            model_revision=REVISION,
            snapshot_files={
                n: file_identity(args.snapshot / n)
                for n in (
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
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.snapshot), local_files_only=True, trust_remote_code=False
        )
        backbone = BackboneAdapter(
            BackboneConfig.model_validate(configs["backbone"]), inspect_device("cpu")
        )
        stage("loaded")
        versions = [p._version for p in backbone._model.parameters()]
        request = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=CognitiveConfig.model_validate(configs["cognitive"]),
            workspace_config=WorkspaceConfig.model_validate(configs["workspace"]),
            stop_token_ids=backbone.eos_token_ids,
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )
        fingerprint = request.sidecar_state_sha256
        reference = copy.deepcopy(request._latent_bridge)
        with torch.no_grad():
            reference.normalization.weight.mul_(0.025)
            reference.normalization.bias.mul_(0.025)
            reference.workspace_type_embedding.mul_(0.025)
        trained = copy.deepcopy(reference)
        optimizer = torch.optim.AdamW(trained.parameters(), lr=0.0001, weight_decay=0.0)
        contexts: dict[str, dict[str, Any]] = {}
        targets: dict[str, torch.Tensor] = {}
        prepared: list[dict[str, Any]] = []
        for row in rows:
            prompt = render_prompt(tokenizer, row["prompt"], raw=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = [
                *tokenizer.encode(row["target"], add_special_tokens=False),
                tokenizer.eos_token_id,
            ]
            if not 1 <= len(prompt_ids) <= 128 or not 1 <= len(target_ids) <= 16:
                raise ValueError("Untruncated prompt<=128/target<=16 required")
            execution = request.execute(
                prompt,
                strategy="greedy",
                diagnostics=WorkspaceTraceOptionsV1(
                    mode="metadata",
                    example_id=row["id"],
                    checkpoint_id="untrained-producer",
                    seed=42,
                ),
            )
            assert execution.diagnostic_trace is not None
            save_new(
                args.out / (row["id"] + ".trace.json"),
                execution.diagnostic_trace.model_dump(mode="json"),
            )
            head, marker, tail = prompt.rpartition("<|im_start|>assistant\n")
            head_ids = tokenizer.encode(head, add_special_tokens=False)
            if (
                not marker
                or head_ids + tokenizer.encode(marker + tail, add_special_tokens=False)
                != prompt_ids
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
            contexts[row["id"]] = dict(
                ids=execution.prompt_input_ids.detach().clone(),
                mask=execution.prompt_attention_mask.detach().clone(),
                workspace=state,
                boundary=len(head_ids),
                states=execution.generation.workspace_trace,
            )
            targets[row["id"]] = torch.tensor([target_ids], dtype=torch.long)
            prepared.append(
                dict(
                    id=row["id"],
                    split=row["split"],
                    rendered_prompt=prompt,
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    boundary=len(head_ids),
                    trace=file_identity(args.out / (row["id"] + ".trace.json")),
                )
            )
        provenance.update(producer_sha256=fingerprint, prepared=prepared)
        save_new(
            args.out / "inputs.json",
            dict(
                provenance=provenance,
                total_steps=args.steps,
                resume=str(args.resume) if args.resume else None,
                evaluate_only=args.evaluate_only,
            ),
        )
        start = (
            restore_checkpoint(args.resume, trained, optimizer, expected_provenance=provenance)
            if args.resume
            else 0
        )
        if (args.evaluate_only and start != args.steps) or (
            not args.evaluate_only and start >= args.steps
        ):
            raise ValueError("Checkpoint cursor disagrees with requested operation")
        stage("prepared")

        def loss_for(bridge: Any, row: dict[str, str]) -> torch.Tensor:
            context, target = contexts[row["id"]], targets[row["id"]]
            latent = bridge(context["workspace"], context["mask"]).workspace_embeddings
            seq = chat_boundary_teacher_sequence(
                backbone.embed_tokens(context["ids"]),
                latent,
                context["boundary"],
                target,
                backbone.embed_tokens(target),
            )
            output = backbone.forward_embeddings(
                seq.inputs_embeds, seq.attention_mask, seq.position_ids, labels=seq.labels
            )
            assert output.loss is not None and bool(torch.isfinite(output.loss))
            return output.loss

        def measure(bridge: Any) -> list[dict[str, Any]]:
            observations = []
            for row in rows:
                with torch.no_grad():
                    nll = float(loss_for(bridge, row))
                generated = generate_from_context(
                    backbone, request._iris_generator, bridge, contexts[row["id"]]
                )
                observations.append(
                    dict(
                        id=row["id"],
                        split=row["split"],
                        nll=nll,
                        text=generated.text,
                        token_ids=generated.token_ids.tolist(),
                        stop_reason=generated.stop_reason,
                        exact_answer=normalize_answer(generated.text)
                        == normalize_answer(row["target"]),
                        finished=generated.stop_reason == "stop_token",
                    )
                )
            return observations

        before = measure(reference)
        save_new(args.out / "untrained.json", before)
        stage("before-measured")
        history, checkpoints = [], []
        if not args.evaluate_only:
            for step in range(start + 1, args.steps + 1):
                row = training_case(rows, step)
                loss = loss_for(trained, row)
                torch.autograd.backward((loss,))
                if any(
                    p.grad is None or not bool(torch.isfinite(p.grad).all())
                    for p in trained.parameters()
                ):
                    raise ValueError("Finite connected bridge gradients required")
                history.append(
                    dict(step=step, id=row["id"], split=row["split"], nll=float(loss.detach()))
                )
                torch.nn.utils.clip_grad_norm_(
                    list(trained.parameters()), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del loss
                if (
                    step % len([r for r in rows if r["split"] == "train"]) == 0
                    or step == args.steps
                ):
                    name = f"checkpoint-step-{step:03}"
                    write_checkpoint(
                        args.out / name,
                        trained,
                        optimizer,
                        completed_steps=step,
                        provenance=provenance,
                    )
                    checkpoints.append(name)
                stage(f"step-{step:03}")
        after = measure(trained)
        save_new(args.out / "trained.json", after)
        baseline = []
        for row in rows:
            context = contexts[row["id"]]
            with torch.inference_mode():
                tokens = backbone._model.generate(
                    input_ids=context["ids"],
                    attention_mask=context["mask"],
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=list(backbone.eos_token_ids),
                    pad_token_id=backbone.eos_token_ids[0],
                    use_cache=True,
                )[:, context["ids"].shape[-1] :]
            text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            if not isinstance(text, str):
                raise ValueError("A single decoded baseline string required")
            baseline.append(
                dict(
                    id=row["id"],
                    split=row["split"],
                    text=text,
                    token_ids=tokens.tolist(),
                    finished=bool(tokens.numel() and int(tokens[0, -1]) in backbone.eos_token_ids),
                    exact_answer=normalize_answer(text) == normalize_answer(row["target"]),
                )
            )
        save_new(args.out / "baseline.json", baseline)
        if (
            request.sidecar_state_sha256 != fingerprint
            or backbone.trainable_parameter_count != 0
            or backbone.has_parameter_gradients
            or versions != [p._version for p in backbone._model.parameters()]
            or any(p.grad is not None for _, m in request._sidecar_modules for p in m.parameters())
        ):
            raise ValueError("Original producer/backbone boundary changed")
        stage("completed")

        def summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                split: dict(
                    total=len([r for r in observations if r["split"] == split]),
                    exact=sum(
                        r["exact_answer"] and r["finished"]
                        for r in observations
                        if r["split"] == split
                    ),
                )
                for split in ("train", "heldout")
            }

        save_new(
            args.out / "result.json",
            dict(
                kind="iris-coursework-bridge-tutorial-result-v1",
                completed=True,
                restored_step=start,
                completed_steps=args.steps,
                optimizer_steps_performed=0 if args.evaluate_only else args.steps - start,
                gradient_history=history,
                checkpoints=checkpoints,
                untrained_summary=summary(before),
                trained_summary=summary(after),
                baseline_summary=summary(baseline),
                backbone_frozen=True,
                cognitive_components_trained=False,
                original_request_unchanged=True,
                trained_component="copy-of-latent-bridge-only",
                cached_workspace_reused=True,
                scientific_acceptance=False,
                quality_improvement_proven=False,
                resource_stages=resources,
                elapsed_seconds=time.perf_counter() - began,
                files={
                    str(p.relative_to(args.out)): file_identity(p)
                    for p in sorted(args.out.rglob("*"))
                    if p.is_file()
                },
            ),
        )
        return 0
    except BaseException as error:
        save_new(
            args.out / "error.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                elapsed_seconds=time.perf_counter() - began,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
