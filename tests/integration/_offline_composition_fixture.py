"""One explicit offline tiny random backbone fixture, never a pretrained model.

The caller configures its isolated HF cache before importing Hugging Face. This
helper does not change global cache settings, download, install or delete files.
"""

import json
import os
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from tests._source_boundary import ordinary_directory, regular


def tiny_offline_cache(build: Callable[[], tuple[Any, Any, Any]]) -> Path:
    from huggingface_hub import constants as hf_constants

    configured = os.environ.get("IRIS_ABLATION_COMPOSITION_TEST_CACHE")
    if not configured:
        raise RuntimeError(
            "Configure a fresh IRIS_ABLATION_COMPOSITION_TEST_CACHE and matching HF_HUB_CACHE "
            "before importing Hugging Face; enable HF_HUB_OFFLINE, HF_DATASETS_OFFLINE and "
            "TRANSFORMERS_OFFLINE. Keep the fixture outside pytest's --basetemp."
        )
    cache = Path(configured)
    if (
        not cache.is_absolute()
        or str(cache) != hf_constants.HF_HUB_CACHE
        or not hf_constants.HF_HUB_OFFLINE
        or any(
            os.environ.get(name) != "1"
            for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
    ):
        raise RuntimeError("An explicit absolute isolated cache and offline loaders are required")
    manifest_path = cache.parent / "tiny-hf-cache-fixture-manifest.json"
    for parent in (cache.parent, *cache.parent.parents):
        if os.path.lexists(parent):
            ordinary_directory(parent)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ordinary_directory(cache.parent)
    if not os.path.lexists(cache):
        if os.path.lexists(manifest_path):
            raise ValueError("Orphan fixture manifest preserved; choose a fresh cache directory")
        import torch

        from iris.utils.seeds import capture_runtime_state, restore_runtime_state

        caller = capture_runtime_state(torch.device("cpu"))
        try:
            _, tokenizer, model = build()
            snapshots = cache / "models--offline--random-qwen3/snapshots"
            model_path, tokenizer_path = snapshots / ("1" * 40), snapshots / ("2" * 40)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(tokenizer_path)
            model.config.save_pretrained(tokenizer_path)
            if not (model_path / "model.safetensors").is_file():
                raise ValueError("Actual tiny random safetensors fixture required")
            files = _inventory(cache)
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    dict(kind="random-tiny-fixture-not-pretrained", files=files),
                    stream,
                    sort_keys=True,
                )
        finally:
            restore_runtime_state(caller)
    ordinary_directory(cache)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate tiny-fixture manifest key")
            result[key] = value
        return result

    manifest = json.loads(regular(manifest_path, 64 * 1024), object_pairs_hook=pairs)
    if (
        type(manifest) is not dict
        or set(manifest) != {"kind", "files"}
        or manifest["kind"] != "random-tiny-fixture-not-pretrained"
        or _inventory(cache) != manifest["files"]
    ):
        raise ValueError("Complete original tiny fixture bytes/membership required")
    return cache


def _inventory(cache: Path) -> dict[str, str]:
    ordinary_directory(cache)
    result: dict[str, str] = {}
    total = 0

    def refused(error: OSError) -> None:
        raise error

    for current, folders, names in os.walk(cache, followlinks=False, onerror=refused):
        directory = Path(current)
        for node in (directory, *(directory / name for name in folders)):
            ordinary_directory(node)
        for name in names:
            path = directory / name
            raw = regular(path, 2 * 1024**2)
            total += len(raw)
            result[path.relative_to(cache).as_posix()] = sha256(raw).hexdigest()
    if not 1 <= len(result) <= 16 or total >= 2 * 1024**2:
        raise ValueError("Bounded random tiny fixture, not full model weights")
    return result
