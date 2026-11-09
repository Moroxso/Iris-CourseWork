"""Small local-machine defence kit: references and verification, no model/venv copy."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KIND = "iris-coursework-local-defence-kit-v2"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "PyYAML",
    "pydantic",
    "omegaconf",
    "einops",
    "psutil",
    "numpy",
    "tokenizers",
)
COPIES = {"README.md": "README.md"}


def identity(path: Path) -> dict[str, Any]:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def relative_file(root: Path, value: str) -> Path:
    parts = PurePosixPath(value)
    if (
        not value
        or parts.is_absolute()
        or "\\" in value
        or ":" in value
        or any(p in {"..", ".", ""} for p in value.split("/"))
    ):
        raise ValueError("Strict relative POSIX file path required")
    return root.joinpath(*parts.parts)


def collect(roots: dict[str, Path], layout: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [
        dict(scope=scope, path=name, **identity(relative_file(roots[scope], name)))
        for scope in sorted(layout)
        for name in sorted(layout[scope])
    ]


def verify_files(
    entries: list[dict[str, Any]], roots: dict[str, Path], layout: dict[str, list[str]]
) -> int:
    required = {(scope, name) for scope, names in layout.items() for name in names}
    actual = [(e.get("scope"), e.get("path")) for e in entries]
    if len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError("Exact required inventory membership mismatch")
    failures = []
    for entry in entries:
        path = relative_file(roots[entry["scope"]], entry["path"])
        try:
            observed = identity(path)
            if observed != {k: entry.get(k) for k in ("bytes", "sha256")}:
                failures.append(entry["scope"] + ":" + entry["path"])
        except OSError:
            failures.append(entry["scope"] + ":" + entry["path"])
    if failures:
        raise ValueError("Inventory content mismatch: " + ", ".join(failures[:10]))
    return len(entries)


def runtime() -> dict[str, Any]:
    return dict(
        executable=str(Path(sys.executable).resolve()),
        prefix=str(Path(sys.prefix).resolve()),
        base_prefix=str(Path(sys.base_prefix).resolve()),
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        packages={name: version(name) for name in PACKAGES},
    )


def layout_for(checkpoint: Path) -> dict[str, list[str]]:
    if not checkpoint.is_relative_to(ROOT):
        raise ValueError("Checkpoint inside this project required")
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "scripts/coursework_defence.ps1",
        ROOT / "README.md",
    ]
    paths += sorted((ROOT / "scripts").glob("coursework*.py"))
    paths += sorted((ROOT / "src/iris").rglob("*.py"))
    paths += sorted((ROOT / "src/iris/configs").rglob("*.yaml"))
    paths += sorted((ROOT / "docs/examples").glob("coursework*.json"))
    paths += [checkpoint / "manifest.json", checkpoint / "state.safetensors"]
    env = Path(sys.prefix).resolve()
    return dict(
        project=sorted({p.relative_to(ROOT).as_posix() for p in paths}),
        snapshot=list(SNAPSHOT_FILES),
        kit=list(COPIES),
        environment=["pyvenv.cfg", Path(sys.executable).resolve().relative_to(env).as_posix()],
        python_base=["python.exe", "python312.dll", "Lib/encodings/__init__.py"],
    )


def smoke(snapshot: Path) -> dict[str, Any]:
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    sys.path.insert(0, str(ROOT / "src"))
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "yaml",
        "pydantic",
        "omegaconf",
        "einops",
        "psutil",
        "iris.backbone",
    ):
        importlib.import_module(name)
    import torch
    from transformers import AutoTokenizer

    if float((torch.tensor([1.0], device="cpu") + 1).item()) != 2.0:
        raise ValueError("CPU tensor smoke failed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    text = tokenizer.apply_chat_template(
        [dict(role="user", content="Return the word ready.")],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > 128:
        raise ValueError("Local tokenizer smoke failed")
    import psutil  # type: ignore[import-untyped]

    available = int(psutil.virtual_memory().available)
    if available < 6 * 1024**3:
        raise ValueError("At least 6 GiB available RAM required for the live demo")
    return dict(
        imports_passed=True,
        cpu_tensor_passed=True,
        local_tokenizer_tokens=len(ids),
        available_ram_bytes=available,
        local_files_only=True,
        physical_network_disconnection_tested=False,
        gpu_validation=False,
    )


def write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, indent=2).encode()
        + b"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if not args.snapshot.is_absolute() or args.snapshot.name != REVISION:
        raise ValueError("Explicit pinned absolute snapshot required")
    if not args.checkpoint.is_absolute() or ".." in args.checkpoint.parts:
        raise ValueError("Absolute in-project checkpoint required")
    current_runtime = runtime()
    env = Path(sys.prefix).resolve()
    if (
        sys.platform != "win32"
        or sys.prefix == sys.base_prefix
        or not Path(sys.executable).resolve().is_relative_to(env)
        or not (env / "pyvenv.cfg").is_file()
    ):
        raise ValueError("Use Python 3.12 from a Windows virtual environment")
    layout = layout_for(args.checkpoint)
    if args.action == "build":
        if (
            args.out is None
            or not args.out.is_absolute()
            or args.out.exists()
            or ".." in args.out.parts
        ):
            raise ValueError("Fresh absolute kit output required")
        evidence = smoke(args.snapshot)
        args.out.mkdir(parents=True)
        for name, source in COPIES.items():
            write_new(args.out / name, (ROOT / source).read_bytes())
        roots = dict(
            project=ROOT,
            snapshot=args.snapshot,
            python_base=Path(sys.base_prefix),
            kit=args.out,
            environment=env,
        )
        manifest: dict[str, Any] = dict(
            kind=KIND,
            local_machine_only=True,
            scientific_acceptance=False,
            project=str(ROOT),
            snapshot=str(args.snapshot),
            checkpoint=str(args.checkpoint),
            runtime=current_runtime,
            files=collect(roots, layout),
            build_smoke=evidence,
            model_or_venv_copied=False,
            limitations=[
                "Not a portable installer or full venv file audit.",
                "No scientific quality acceptance or visual browser QA.",
            ],
        )
        write_new(args.out / "manifest.json", json_bytes(manifest))
        result = dict(
            completed=True,
            action="build",
            file_count=len(manifest["files"]),
            manifest=str(args.out / "manifest.json"),
            manifest_identity=identity(args.out / "manifest.json"),
        )
    else:
        if (
            args.manifest is None
            or not args.manifest.is_absolute()
            or args.manifest.stat().st_size > 1024**2
        ):
            raise ValueError("Bounded absolute kit manifest required")
        manifest = json.loads(args.manifest.read_text("utf-8"))
        if (
            manifest.get("kind") != KIND
            or manifest.get("local_machine_only") is not True
            or manifest.get("scientific_acceptance") is not False
            or manifest.get("project") != str(ROOT)
            or manifest.get("snapshot") != str(args.snapshot)
            or manifest.get("checkpoint") != str(args.checkpoint)
            or manifest.get("runtime") != current_runtime
        ):
            raise ValueError("Kit runtime/path binding mismatch; rebuild explicitly, do not bypass")
        roots = dict(
            project=ROOT,
            snapshot=args.snapshot,
            python_base=Path(sys.base_prefix),
            kit=args.manifest.parent,
            environment=env,
        )
        count = verify_files(manifest["files"], roots, layout)
        result = dict(
            completed=True,
            action="verify",
            verified_files=count,
            manifest_identity=identity(args.manifest),
            runtime_matched=True,
            import_smoke=smoke(args.snapshot),
        )
    if args.receipt is not None:
        if not args.receipt.is_absolute() or args.receipt.exists():
            raise ValueError("Fresh absolute verification receipt required")
        write_new(args.receipt, json_bytes(result))
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
