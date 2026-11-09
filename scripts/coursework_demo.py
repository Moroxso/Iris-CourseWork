"""Offline CPU coursework demo using the existing two-pass Iris request API.

Generated text is data only. Random sidecar weights are explicitly not trained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CONCISE_SYSTEM_PROMPT = (
    "Follow the user's requested output format exactly. Give only the final answer. "
    "Do not include reasoning, introductions, lists, LaTeX, or explanations unless "
    "the user explicitly asks for them."
)


def save_new(path: Path, value: object) -> None:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode()
    with path.open("xb") as stream:
        stream.write(raw + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--snapshot", type=Path, required=True, help="Local pinned Qwen3-0.6B snapshot"
    )
    result.add_argument("--out", type=Path, required=True, help="New output directory")
    result.add_argument("--prompt", default="Explain briefly what shared memory is.")
    result.add_argument("--system-prompt", default="", help="Explicit shared chat instruction")
    result.add_argument("--max-new-tokens", type=int, default=8)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--repeat", type=int, choices=(1, 2), default=1)
    result.add_argument("--raw-prompt", action="store_true", help="Skip the local chat template")
    result.add_argument(
        "--diagnose-bridge",
        action="store_true",
        help="Also run frozen-Qwen-only and observe actual latent norms",
    )
    result.add_argument(
        "--bridge-init-scale",
        type=float,
        default=1.0,
        help="Explicit experimental untrained bridge amplitude (default unchanged)",
    )
    result.add_argument(
        "--diagnose-placement",
        action="store_true",
        help="Compare actual latents after prompt versus before assistant header",
    )
    result.add_argument(
        "--prefix-layout",
        choices=("suffix", "chat-boundary"),
        default="suffix",
        help="Academic layout only; scientific suffix remains unchanged",
    )
    result.add_argument(
        "--check-workspace-influence",
        action="store_true",
        help="Compare next-token logits with real and zero latents at equal positions",
    )
    return result


def render_prompt(tokenizer: Any, prompt: str, *, raw: bool, system_prompt: str = "") -> str:
    if raw:
        if system_prompt:
            raise ValueError("System prompt requires a chat template")
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if type(rendered) is not str or not rendered:
        raise ValueError("Chat template must return nonempty text")
    return rendered


def main() -> int:
    args = parser().parse_args()
    if args.system_prompt and (
        args.raw_prompt or not args.system_prompt.strip() or len(args.system_prompt.encode()) > 8192
    ):
        raise ValueError("Bounded nonempty system prompt requires a chat template")
    if args.prefix_layout == "chat-boundary" and args.raw_prompt:
        raise ValueError("Chat-boundary layout requires a chat template")
    if args.check_workspace_influence and args.prefix_layout != "chat-boundary":
        raise ValueError("Workspace influence check requires chat-boundary layout")
    if not math.isfinite(args.bridge_init_scale) or not 0.001 <= args.bridge_init_scale <= 1:
        raise ValueError("Bridge initialization scale must be finite and in [0.001, 1]")
    if not (1 <= args.max_new_tokens <= 128 and 1 <= args.threads <= 8 and 0 <= args.seed < 2**32):
        raise ValueError("Bounded tokens, CPU threads and seed required")
    if not args.prompt.strip() or len(args.prompt.encode()) > 8192:
        raise ValueError("Nonempty prompt at most 8192 UTF-8 bytes required")
    snapshot, output = args.snapshot, args.out
    if not snapshot.is_absolute() or snapshot.name != REVISION or not snapshot.is_dir():
        raise ValueError("Explicit local snapshot for the pinned Qwen3-0.6B revision required")
    if not output.is_absolute() or ".." in output.parts or output.exists():
        raise ValueError("Fresh absolute output directory required")
    expected = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }
    if not all((snapshot / name).is_file() for name in expected):
        raise ValueError("Incomplete local pretrained snapshot; nothing will be downloaded")
    model_config = json.loads((snapshot / "config.json").read_bytes())
    if (
        model_config.get("model_type"),
        model_config.get("hidden_size"),
        model_config.get("num_hidden_layers"),
    ) != ("qwen3", 1024, 28):
        raise ValueError("Expected actual Qwen3-0.6B geometry, not the tiny test fixture")
    output.mkdir(parents=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    began = time.perf_counter()
    try:
        import torch
        import yaml  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        from iris.application.composition import build_request_orchestrator
        from iris.backbone import BackboneAdapter
        from iris.generation import TemperatureTopPSamplingConfig
        from iris.schemas import BackboneConfig, CognitiveConfig, WorkspaceConfig
        from iris.utils.devices import inspect_device
        from iris.workspace.trace import WorkspaceTraceOptionsV1

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        config_root = ROOT / "src/iris/configs"

        def config(group: str, name: str) -> Any:
            return yaml.safe_load((config_root / group / (name + ".yaml")).read_text("utf-8"))

        cfg = config("backbone", "qwen3_0_6b")
        cfg.update(repository=str(snapshot), local_files_only=True)
        cfg["limits"] = dict(max_prompt_tokens=256, max_new_tokens=args.max_new_tokens)
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        prompt = render_prompt(
            tokenizer, args.prompt, raw=args.raw_prompt, system_prompt=args.system_prompt
        )
        if len(tokenizer.encode(prompt)) > 256:
            raise ValueError("Rendered prompt exceeds 256 tokens; shorten it instead of truncating")
        backbone = BackboneAdapter(BackboneConfig.model_validate(cfg), inspect_device("cpu"))
        loaded = time.perf_counter()
        assert backbone.trainable_parameter_count == 0
        request = build_request_orchestrator(
            backbone=backbone,
            cognitive_config=CognitiveConfig.model_validate(config("cognitive", "iris_gw_0_1")),
            workspace_config=WorkspaceConfig.model_validate(config("workspace", "iris_gw_0_1")),
            stop_token_ids=backbone.eos_token_ids,
            sampling=TemperatureTopPSamplingConfig(temperature=0.8, top_p=0.9),
        )
        if args.bridge_init_scale != 1:
            # Experimental initial parameters, before execution; never alter loaded backbone.
            with torch.no_grad():
                bridge = request._latent_bridge
                bridge.normalization.weight.mul_(args.bridge_init_scale)
                bridge.normalization.bias.mul_(args.bridge_init_scale)
                bridge.workspace_type_embedding.mul_(args.bridge_init_scale)
        baseline = None
        latent_measurements: list[dict[str, object]] = []
        if args.diagnose_bridge:
            encoded = backbone.tokenize(prompt)
            with torch.inference_mode():
                model = backbone._model
                embedding = model.get_input_embeddings()(encoded["input_ids"])
                prompt_norms = torch.linalg.vector_norm(embedding.float(), dim=-1)
                generated = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=list(backbone.eos_token_ids),
                    pad_token_id=backbone.eos_token_ids[0],
                    use_cache=True,
                )[:, encoded["input_ids"].shape[-1] :]
                baseline = dict(
                    label="Frozen Qwen only, not Iris output",
                    token_ids=generated.tolist(),
                    text=tokenizer.decode(generated[0], skip_special_tokens=True),
                    prompt_embedding_norm_mean=float(prompt_norms.mean()),
                    prompt_embedding_norm_max=float(prompt_norms.max()),
                )

            def observe_latent(module: Any, inputs: Any, value: Any) -> None:
                norms = torch.linalg.vector_norm(
                    value.workspace_embeddings.detach().float(), dim=-1
                )
                latent_measurements.append(
                    dict(
                        token_norms=norms.tolist(),
                        active_tokens=int(value.workspace_attention_mask.sum()),
                    )
                )

            # Observe only: the hook returns None and never replaces bridge output.
            hook = request._latent_bridge.register_forward_hook(observe_latent)
        else:
            hook = None
        result = request.execute(
            prompt,
            strategy="greedy",
            diagnostics=WorkspaceTraceOptionsV1(
                mode="metadata",
                example_id="coursework-demo",
                checkpoint_id="untrained-sidecar",
                seed=args.seed,
            ),
        )
        trace = result.diagnostic_trace
        canonical_suffix_text = result.generation.text
        influence: list[dict[str, object]] = []

        def academic_generation(execution: Any) -> Any:
            if args.prefix_layout == "suffix":
                return execution.generation
            if args.raw_prompt:
                raise ValueError("Academic chat-boundary layout requires a chat template")
            from coursework_prefix import chat_boundary_prefix

            head, marker, tail = prompt.rpartition("<|im_start|>assistant\n")
            if not marker:
                raise ValueError("Assistant boundary not found")
            head_ids = tokenizer.encode(head, add_special_tokens=False)
            rest_ids = tokenizer.encode(marker + tail, add_special_tokens=False)
            ids = execution.prompt_input_ids
            if head_ids + rest_ids != ids[0].tolist():
                raise ValueError("Chat split changed tokenization")
            with torch.inference_mode():
                latent = request._latent_bridge(
                    execution.generation.workspace_trace[-1], execution.prompt_attention_mask
                ).workspace_embeddings
                prefix = chat_boundary_prefix(backbone.embed_tokens(ids), latent, len(head_ids))
                if args.check_workspace_influence:
                    zero = chat_boundary_prefix(
                        backbone.embed_tokens(ids), torch.zeros_like(latent), len(head_ids)
                    )
                    actual_logits = (
                        backbone.forward_embeddings(
                            prefix.inputs_embeds,
                            prefix.attention_mask,
                            prefix.position_ids,
                            logits_to_keep=1,
                        )
                        .logits[:, -1]
                        .float()
                    )
                    zero_logits = (
                        backbone.forward_embeddings(
                            zero.inputs_embeds,
                            zero.attention_mask,
                            zero.position_ids,
                            logits_to_keep=1,
                        )
                        .logits[:, -1]
                        .float()
                    )
                    delta = actual_logits - zero_logits
                    influence.append(
                        dict(
                            comparison="actual-versus-zero-latents-same-layout-and-token-count",
                            max_abs_logit_delta=float(delta.abs().max()),
                            logit_delta_l2=float(torch.linalg.vector_norm(delta)),
                            actual_argmax=int(actual_logits.argmax()),
                            zero_argmax=int(zero_logits.argmax()),
                            quality_improvement_proven=False,
                        )
                    )
                return request._iris_generator.generate(
                    prefix,
                    strategy="greedy",
                    workspace_trace=execution.generation.workspace_trace,
                )

        generation = academic_generation(result)
        assert trace is not None and trace.cycle_count == 3 and len(trace.cycles) == 3
        assert (trace.num_modules, trace.num_slots, trace.state_dim) == (8, 8, 256)
        assert len(result.generation.workspace_trace) == 3 and not backbone.has_parameter_gradients
        elapsed = time.perf_counter() - began
        save_new(output / "workspace-trace.json", trace.model_dump(mode="json"))
        repeat_equal = None
        if args.repeat == 2:
            repeated = request.execute(
                prompt,
                strategy="greedy",
                diagnostics=WorkspaceTraceOptionsV1(
                    mode="metadata",
                    example_id="coursework-demo",
                    checkpoint_id="untrained-sidecar",
                    seed=args.seed,
                ),
            )
            assert repeated.diagnostic_trace is not None
            save_new(
                output / "workspace-trace-repeat.json",
                repeated.diagnostic_trace.model_dump(mode="json"),
            )
            repeat_equal = (
                torch.equal(generation.token_ids, academic_generation(repeated).token_ids)
                and trace == repeated.diagnostic_trace
            )
            if not repeat_equal:
                raise ValueError("Identical repeated request changed tokens or workspace trace")
        if hook is not None:
            hook.remove()
        placement = None
        if args.diagnose_placement:
            if args.raw_prompt:
                raise ValueError("Placement comparison requires the explicit chat template")
            head, marker, tail = prompt.rpartition("<|im_start|>assistant\n")
            if not marker:
                raise ValueError("Assistant generation header not found")
            head_ids = tokenizer.encode(head, add_special_tokens=False)
            answer_ids = tokenizer.encode(marker + tail, add_special_tokens=False)
            encoded = backbone.tokenize(prompt)
            if head_ids + answer_ids != encoded["input_ids"][0].tolist():
                raise ValueError("Chat split changed tokenization")
            with torch.inference_mode():
                embeddings = backbone.embed_tokens(encoded["input_ids"])
                latent = request._latent_bridge(
                    result.generation.workspace_trace[-1], encoded["attention_mask"]
                ).workspace_embeddings
                layouts = {
                    "prompt_then_latents": torch.cat((embeddings, latent), dim=1),
                    "user_then_latents_then_assistant": torch.cat(
                        (embeddings[:, : len(head_ids)], latent, embeddings[:, len(head_ids) :]),
                        dim=1,
                    ),
                }
                placement = {}
                for name, values in layouts.items():
                    mask = torch.ones(values.shape[:2], dtype=torch.long, device=values.device)
                    tokens = backbone._model.generate(
                        inputs_embeds=values,
                        attention_mask=mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=list(backbone.eos_token_ids),
                        pad_token_id=backbone.eos_token_ids[0],
                        use_cache=True,
                    )
                    placement[name] = dict(
                        token_ids=tokens.tolist(),
                        text=tokenizer.decode(tokens[0], skip_special_tokens=True),
                        input_tokens=values.shape[1],
                    )
                placement["suffix_reproduces_iris"] = (
                    placement["prompt_then_latents"]["token_ids"]
                    == result.generation.token_ids.tolist()
                )
        report = dict(
            kind="iris-coursework-two-pass-demo-v1",
            completed=True,
            snapshot=str(snapshot),
            revision=REVISION,
            backbone_pretrained=True,
            backbone_frozen=True,
            sidecar_trained=False,
            bridge_initialization_scale=args.bridge_init_scale,
            seed=args.seed,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            rendered_prompt=prompt,
            prompt_format="raw" if args.raw_prompt else "local-chat-template-no-thinking",
            prefix_layout=args.prefix_layout,
            canonical_suffix_text=canonical_suffix_text,
            repeated_requests=args.repeat,
            repeated_tokens_and_trace_equal=repeat_equal,
            frozen_qwen_baseline=baseline,
            latent_measurements=latent_measurements,
            placement_diagnostic=placement,
            workspace_influence=influence,
            prompt_tokens=int(result.prompt_input_ids.shape[-1]),
            text=generation.text,
            token_ids=generation.token_ids.tolist(),
            stop_reason=generation.stop_reason,
            load_seconds=loaded - began,
            first_request_seconds=elapsed,
            total_seconds=time.perf_counter() - began,
            torch_version=torch.__version__,
            device="cpu",
            dtype="fp32",
            source_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            academic_prefix_source_sha256=sha256(
                (ROOT / "scripts/coursework_prefix.py").read_bytes()
            ).hexdigest(),
            scientific_acceptance=False,
            limitation="Random untrained cognitive components; no quality improvement claim.",
        )
        save_new(output / "result.json", report)
        # -I ignores PYTHONIOENCODING; keep console JSON portable on cp1251 hosts.
        print(json.dumps(report, ensure_ascii=True), flush=True)
        return 0
    except BaseException as error:
        save_new(
            output / "error.json",
            dict(
                error_type=type(error).__name__,
                message=str(error),
                elapsed_seconds=time.perf_counter() - began,
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
