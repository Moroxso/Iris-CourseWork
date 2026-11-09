"""One bounded bridge-only autograd/optimizer profile, not scientific training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PROMPT = "The badge is violet. What color is the badge? Answer with one word."
TRAIN_TARGET = "violet"


def memory_snapshot() -> dict[str, int]:
    import psutil  # type: ignore[import-untyped]

    info = psutil.Process().memory_info()
    return dict(
        rss_bytes=int(info.rss),
        private_bytes=int(getattr(info, "private", info.vms)),
        peak_working_set_bytes=int(getattr(info, "peak_wset", info.rss)),
        available_ram_bytes=int(psutil.virtual_memory().available),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1, help="Total completed steps, 1..4")
    parser.add_argument("--resume", type=Path, help="Academic bridge+AdamW checkpoint")
    args = parser.parse_args()
    if not 1 <= args.steps <= 4:
        raise ValueError("One to four total steps required")
    if args.resume is not None and not args.resume.is_absolute():
        raise ValueError("Absolute resume checkpoint required")
    if not args.out.is_absolute() or args.out.exists() or ".." in args.out.parts:
        raise ValueError("Fresh absolute output required")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from coursework_demo import REVISION, render_prompt, save_new
    from coursework_suite import CASES, file_identity, load_cases

    snapshot: Path = args.snapshot
    if not snapshot.is_absolute() or snapshot.name != REVISION:
        raise ValueError("Pinned absolute snapshot required")
    if any(c["prompt"] == TRAIN_PROMPT for c in load_cases(CASES)):
        raise ValueError("Training prompt must not be a demonstration input")
    memory_before = memory_snapshot()
    if memory_before["available_ram_bytes"] < 6 * 1024**3:
        raise ValueError("At least 6 GiB available RAM required before model loading")
    model_config = json.loads((snapshot / "config.json").read_text("utf-8"))
    if (
        model_config.get("model_type"),
        model_config.get("hidden_size"),
        model_config.get("num_hidden_layers"),
    ) != ("qwen3", 1024, 28):
        raise ValueError("Actual pretrained Qwen3-0.6B geometry required")
    if (snapshot / "model.safetensors").stat().st_size < 1024**3:
        raise ValueError("Actual pretrained weights required, not a tiny fixture")
    args.out.mkdir(parents=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    began = time.perf_counter()
    stages: list[dict[str, Any]] = []

    def stage(name: str) -> None:
        values = memory_snapshot()
        row = dict(stage=name, elapsed_seconds=time.perf_counter() - began, **values)
        stages.append(row)
        save_new(args.out / (name + ".json"), row)
        print(json.dumps(row), flush=True)
        if values["rss_bytes"] > 8 * 1024**3:
            raise MemoryError("Observed process RSS exceeded 8 GiB; no further step allowed")

    try:
        import torch
        import yaml  # type: ignore[import-untyped]
        from coursework_bridge_checkpoint import restore_checkpoint, write_checkpoint
        from coursework_prefix import chat_boundary_teacher_sequence
        from safetensors.torch import save as save_safetensors
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

        def config(group: str) -> Any:
            filename = "qwen3_0_6b" if group == "backbone" else "iris_gw_0_1"
            return yaml.safe_load((config_root / group / (filename + ".yaml")).read_text("utf-8"))

        source_paths = [
            Path(__file__),
            ROOT / "scripts/coursework_prefix.py",
            ROOT / "scripts/coursework_demo.py",
            ROOT / "scripts/coursework_suite.py",
            ROOT / "scripts/coursework_bridge_checkpoint.py",
            CASES,
        ]
        source_paths += sorted((ROOT / "src/iris").rglob("*.py"))
        source_paths += sorted(config_root.rglob("*.yaml"))
        save_new(
            args.out / "inputs.json",
            dict(
                kind="iris-coursework-bridge-profile-inputs-v1",
                training_prompt=TRAIN_PROMPT,
                training_target=TRAIN_TARGET,
                revision=REVISION,
                snapshot=str(snapshot),
                model_weights=file_identity(snapshot / "model.safetensors"),
                snapshot_files={
                    name: file_identity(snapshot / name)
                    for name in (
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                    )
                },
                python_version=sys.version,
                software_versions={
                    name: version(name)
                    for name in ("torch", "transformers", "safetensors", "psutil", "PyYAML")
                },
                source_files={str(p.relative_to(ROOT)): file_identity(p) for p in source_paths},
                seed=42,
                cpu_threads=4,
                prefix_layout="chat-boundary",
                bridge_scale=0.025,
                optimizer=dict(type="AdamW", learning_rate=0.0001, weight_decay=0.0),
                planned_steps=args.steps,
                resume_checkpoint=str(args.resume) if args.resume is not None else None,
                memory_before=memory_before,
                scientific_acceptance=False,
            ),
        )
        cfg = config("backbone")
        cfg.update(
            repository=str(snapshot),
            local_files_only=True,
            limits=dict(max_prompt_tokens=128, max_new_tokens=16),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        prompt = render_prompt(tokenizer, TRAIN_PROMPT, raw=False)
        ids_list = tokenizer.encode(prompt, add_special_tokens=False)
        target_list = tokenizer.encode(TRAIN_TARGET, add_special_tokens=False)
        target_list.append(tokenizer.eos_token_id)
        if not 1 <= len(ids_list) <= 128 or not 1 <= len(target_list) <= 16:
            raise ValueError("Bounded prompt/target required without truncation")
        save_new(
            args.out / "prepared-tokens.json",
            dict(
                rendered_prompt=prompt,
                prompt_token_ids=ids_list,
                target_token_ids=target_list,
                target_includes_eos=True,
            ),
        )
        backbone = BackboneAdapter(BackboneConfig.model_validate(cfg), inspect_device("cpu"))
        stage("loaded")
        versions = [p._version for p in backbone._model.parameters()]
        request = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=CognitiveConfig.model_validate(config("cognitive")),
            workspace_config=WorkspaceConfig.model_validate(config("workspace")),
            stop_token_ids=backbone.eos_token_ids,
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )
        owner_before = request.sidecar_state_sha256
        execution = request.execute(
            prompt,
            strategy="greedy",
            diagnostics=WorkspaceTraceOptionsV1(
                mode="metadata",
                example_id="coursework-training-profile",
                seed=42,
                checkpoint_id="untrained-sidecar",
            ),
        )
        assert execution.diagnostic_trace is not None
        save_new(
            args.out / "workspace-trace.json", execution.diagnostic_trace.model_dump(mode="json")
        )
        workspace = WorkspaceState(
            **{
                f.name: getattr(execution.generation.workspace_trace[-1], f.name).detach().clone()
                for f in fields(WorkspaceState)
            }
        )
        ids = execution.prompt_input_ids.detach().clone()
        mask = execution.prompt_attention_mask.detach().clone()
        head, marker, tail = prompt.rpartition("<|im_start|>assistant\n")
        head_ids = tokenizer.encode(head, add_special_tokens=False)
        if (
            not marker
            or head_ids + tokenizer.encode(marker + tail, add_special_tokens=False) != ids_list
        ):
            raise ValueError("Exact chat boundary required")
        bridge = copy.deepcopy(request._latent_bridge)
        with torch.no_grad():
            bridge.normalization.weight.mul_(0.025)
            bridge.normalization.bias.mul_(0.025)
            bridge.workspace_type_embedding.mul_(0.025)
        initial = {name: tensor.detach().clone() for name, tensor in bridge.state_dict().items()}
        parameters = dict(bridge.named_parameters())
        optimizer = torch.optim.AdamW(parameters.values(), lr=0.0001, weight_decay=0.0)
        inputs = json.loads((args.out / "inputs.json").read_text("utf-8"))
        provenance = {
            k: v
            for k, v in inputs.items()
            if k not in {"memory_before", "planned_steps", "resume_checkpoint"}
        }
        provenance.update(
            original_sidecar_sha256=owner_before,
            workspace_trace=file_identity(args.out / "workspace-trace.json"),
            prepared_tokens=file_identity(args.out / "prepared-tokens.json"),
            data_order="one-training-pair-no-shuffle",
        )
        start_step = 0
        if args.resume is not None:
            start_step = restore_checkpoint(
                args.resume, bridge, optimizer, expected_provenance=provenance
            )
            if start_step >= args.steps:
                raise ValueError("Resume cursor must precede requested total steps")
            initial = {
                name: tensor.detach().clone() for name, tensor in bridge.state_dict().items()
            }
        target_ids = torch.tensor([target_list], dtype=torch.long)
        prompt_embeddings = backbone.embed_tokens(ids)
        target_embeddings = backbone.embed_tokens(target_ids)

        def teacher_loss() -> torch.Tensor:
            latent = bridge(workspace, mask).workspace_embeddings
            sequence = chat_boundary_teacher_sequence(
                prompt_embeddings, latent, len(head_ids), target_ids, target_embeddings
            )
            output = backbone.forward_embeddings(
                sequence.inputs_embeds,
                sequence.attention_mask,
                sequence.position_ids,
                labels=sequence.labels,
            )
            assert output.loss is not None and bool(torch.isfinite(output.loss))
            return output.loss

        stage("prepared")
        gradient_history = []
        loss_before = None
        for step in range(start_step + 1, args.steps + 1):
            loss = teacher_loss()
            if not loss.requires_grad:
                raise ValueError("Teacher loss detached from bridge")
            if loss_before is None:
                loss_before = float(loss.detach())
            stage(f"forward-{step:03}")
            torch.autograd.backward((loss,))
            gradients = {
                name: float(torch.linalg.vector_norm(p.grad))
                for name, p in parameters.items()
                if p.grad is not None
            }
            if len(gradients) != len(parameters) or not any(v > 0 for v in gradients.values()):
                raise ValueError("All bridge parameters need finite connected gradients")
            if any(
                p.grad is None or not bool(torch.isfinite(p.grad).all())
                for p in parameters.values()
            ):
                raise ValueError("Invalid bridge gradients")
            stage(f"backward-{step:03}")
            gradient_history.append(
                dict(completed_step=step, loss=float(loss.detach()), gradient_norms=gradients)
            )
            torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del loss
            write_checkpoint(
                args.out / f"checkpoint-step-{step:03}",
                bridge,
                optimizer,
                completed_steps=step,
                provenance=provenance,
            )
        with torch.no_grad():
            loss_after = float(teacher_loss())
        change = max(
            float((p.detach() - initial[name]).abs().max()) for name, p in parameters.items()
        )
        if not change > 0:
            raise ValueError("No bridge parameter changed")
        if (
            backbone.trainable_parameter_count != 0
            or backbone.has_parameter_gradients
            or versions != [p._version for p in backbone._model.parameters()]
        ):
            raise ValueError("Backbone frozen boundary changed")
        if request.sidecar_state_sha256 != owner_before:
            raise ValueError("Original request sidecar changed")
        if any(
            p.grad is not None
            for _, module in request._sidecar_modules
            for p in module.parameters()
        ):
            raise ValueError("Original cognitive/request components received gradients")
        for name, state in (
            ("bridge-before.safetensors", initial),
            ("bridge-after.safetensors", bridge.state_dict()),
        ):
            data = save_safetensors({n: t.detach().contiguous() for n, t in state.items()})
            with (args.out / name).open("xb") as stream:
                stream.write(data)
        stage("completed")
        save_new(
            args.out / "result.json",
            dict(
                kind="iris-coursework-bridge-profile-result-v1",
                completed=True,
                optimizer_steps=args.steps - start_step,
                completed_steps=args.steps,
                restored_step=start_step,
                trained_component="copy-of-latent-bridge-only",
                backbone_frozen=True,
                original_sidecar_unchanged=True,
                cognitive_components_trained=False,
                training_loss_before=loss_before,
                training_loss_after=loss_after,
                parameter_count=sum(p.numel() for p in parameters.values()),
                gradient_history=gradient_history,
                max_parameter_change=change,
                resource_stages=stages,
                top_level_bridge_safetensors_weights_only=True,
                checkpoints_include_optimizer_and_cpu_python_rng=True,
                resumable_checkpoints=[
                    f"checkpoint-step-{s:03}" for s in range(start_step + 1, args.steps + 1)
                ],
                quality_improvement_proven=False,
                scientific_acceptance=False,
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
                error_type=type(error).__name__,
                message=str(error),
                stages=stages,
                elapsed_seconds=time.perf_counter() - began,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
