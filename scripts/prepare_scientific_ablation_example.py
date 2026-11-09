"""Prepare bounded operator example DATA; never train, load models or execute answers.

The inputs command is a tiny CPU smoke example, not a scientific dataset release.
The payload command binds an ACTUAL terminal selected-best operator/export through
the original package API. It does not choose a new best, create an owner or run
the sandbox. Both commands refuse existing outputs and preserve partial failures.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from hashlib import sha256
from itertools import islice
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris.application.scientific_command import ScientificParquetInputV1
    from iris.artifacts.scientific_ablation_workflow_v1 import ScientificAblationWorkflowParquetV1
    from iris.data.schemas import IrisExample

BYTE_LIMIT = 1024**2
MAX_JSON = 64 * 1024


class ExamplePreparationError(ValueError):
    """Safe actionable error; raw inputs/answers are never part of the message."""


def local(path: Path) -> Path:
    windows = PureWindowsPath(str(path))
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path).startswith(("//", "\\\\"))
        or (windows.drive and not windows.root)
        or any(":" in part for part in path.parts if part != path.anchor)
    ):
        raise ExamplePreparationError(
            "Use absolute local paths without aliases or parent traversal."
        )
    for parent in (path, *path.parents):
        if os.path.lexists(parent):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ExamplePreparationError("Symlink/reparse paths are not accepted.")
            if parent != path and not stat.S_ISDIR(info.st_mode):
                raise ExamplePreparationError(
                    "An input/output parent is not an ordinary directory."
                )
    return path


def read(path: Path, maximum: int) -> bytes:
    local(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= maximum:
        raise ExamplePreparationError("Expected a bounded regular non-hardlinked input file.")
    raw = path.read_bytes()
    end = path.lstat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if len(raw) != info.st_size or any(getattr(info, k) != getattr(end, k) for k in keys):
        raise ExamplePreparationError("An input changed during reading; preserve it and inspect.")
    return raw


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in items:
        if name in result:
            raise ExamplePreparationError("Duplicate JSON keys are not accepted.")
        result[name] = value
    return result


def nonfinite(value: str) -> Any:
    raise ExamplePreparationError("Nonfinite JSON numbers are not accepted.")


def json_input(path: Path) -> bytes:
    value = json.loads(read(path, MAX_JSON), object_pairs_hook=pairs, parse_constant=nonfinite)
    return encoded(value)


def encoded(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def fresh(path: Path) -> None:
    local(path)
    if os.path.lexists(path):
        raise ExamplePreparationError("Output already exists; preserve it and choose a fresh path.")


def write_new(path: Path, raw: bytes) -> None:
    fresh(path)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def input_spec(name: str, raw: bytes, count: int) -> dict[str, object]:
    return dict(
        path=name,
        sha256=sha256(raw).hexdigest(),
        maximum_rows=count,
        maximum_file_bytes=BYTE_LIMIT,
        maximum_logical_bytes=BYTE_LIMIT,
    )


def prepare_inputs(root: Path) -> dict[str, object]:
    """Fresh five-file example; source-specified synthetic rows, no model/cache writes."""
    from iris.application.scientific_ablation_command_config import (
        IrisScientificAblationCommandConfigV1,
    )
    from iris.artifacts.scientific_evaluation_v1 import TASKS
    from iris.data.generator import generate_synthetic_split
    from iris.data.parquet import encode_synthetic_parquet
    from iris.data.schemas import SyntheticSplitPlan

    fresh(root)
    train = tuple(
        islice(
            generate_synthetic_split(
                SyntheticSplitPlan(split="train", first_seed=0, count=15),
                template_id="iris-synthetic-01-v2",
            ),
            5,
        )
    )
    validation = tuple(
        islice(
            generate_synthetic_split(
                SyntheticSplitPlan(split="validation", first_seed=1000000, count=15),
                template_id="iris-synthetic-01-v2",
            ),
            2,
        )
    )
    pool = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="test", first_seed=2000000, count=100),
            template_id="iris-synthetic-01-v2",
        )
    )
    test = tuple(
        row
        for task in TASKS
        for row in tuple(r for r in pool if r.task_type == task and r.difficulty == 1)[:2]
    )
    if (
        len(test) != 8
        or len({row.task_type for row in test}) != 4
        or sum(r.executable for r in test) != 2
    ):
        raise ExamplePreparationError(
            "The source template did not produce the expected all-four example."
        )
    contents = {
        name: encode_synthetic_parquet(
            rows, max_logical_bytes=BYTE_LIMIT, max_file_bytes=BYTE_LIMIT
        )
        for name, rows in (
            ("train.parquet", tuple(train)),
            ("validation.parquet", tuple(validation)),
            ("test.parquet", test),
        )
    }
    specs = {
        name: input_spec(name + ".parquet", contents[name + ".parquet"], count)
        for name, count in (("train", 5), ("validation", 2), ("test", 8))
    }
    recipe = IrisScientificAblationCommandConfigV1.model_validate_json(
        encoded(
            dict(
                data_scope="candidate_not_launch_approval",
                purpose="smoke",
                device="cpu",
                backbone=dict(
                    name="tiny-example-not-pretrained",
                    repository="offline/random-qwen3",
                    revision="1" * 40,
                    tokenizer_revision="2" * 40,
                    local_files_only=True,
                    dtype="fp32",
                    tokenizer=dict(use_fast=True, padding_side="left", truncation_side="left"),
                    limits=dict(max_prompt_tokens=768, max_new_tokens=256),
                ),
                cognitive=dict(
                    name="tiny-example",
                    state_dim=8,
                    num_modules=4,
                    layer_sampler=dict(layer_fractions=[0.25, 0.5, 0.75, 1.0]),
                ),
                workspace=dict(
                    name="tiny-example",
                    num_slots=4,
                    cycles=3,
                    winners_per_cycle=2,
                    decay_rate=0.1,
                    empty_threshold=0.0001,
                    arbiter=dict(confidence_weight=1.0, novelty_weight=1.0, relevance_weight=1.0),
                    allocator=dict(
                        empty_bonus=1.0, allocation_age_penalty=0.1, retention_age_penalty=0.1
                    ),
                ),
                ablation=dict(routing="learned", goal="enabled", broadcast="enabled"),
                train=specs["train"],
                validation=specs["validation"],
                initialization_seed=1337,
                probe_initialization_seed=42,
                run_seed=2026,
                deterministic=True,
                terminal_token_id=1,
                objective={},
                optimizer=dict(purpose="smoke", minimum_target_tokens=1, max_optimizer_steps=3),
                lifecycle=dict(
                    purpose="smoke",
                    validate_every_steps=1,
                    checkpoint_every_steps=2,
                    minimum_improvement=1e6,
                    maximum_storage_bytes=32 * 1024**2,
                ),
                maximum_journal_bytes=4 * 1024**2,
            )
        ),
        strict=True,
    )
    contents["recipe.json"] = encoded(recipe.model_dump(mode="json"))
    contents["test-input.json"] = encoded(specs["test"])
    if sum(map(len, contents.values())) > 4 * BYTE_LIMIT:
        raise ExamplePreparationError("The bounded example exceeds its total file budget.")
    root.mkdir(parents=False, mode=0o700)
    for name, raw in contents.items():
        write_new(root / name, raw)
    return dict(
        kind="iris-ablation-operator-example-inputs-v1",
        base=str(root),
        recipe=str(root / "recipe.json"),
        test_spec=str(root / "test-input.json"),
        rows=[5, 2, 8],
        python_rows=2,
        file_bytes=sum(map(len, contents.values())),
        files={name: [len(raw), sha256(raw).hexdigest()] for name, raw in contents.items()},
        model_cache_created_or_downloaded=False,
        native_or_scientific_acceptance=False,
    )


def load_rows(
    spec: ScientificParquetInputV1 | ScientificAblationWorkflowParquetV1, base: Path, split: str
) -> tuple[IrisExample, ...]:
    from iris.data.parquet import decode_synthetic_parquet

    raw = read(base / spec.path, spec.maximum_file_bytes)
    if sha256(raw).hexdigest() != spec.sha256:
        raise ExamplePreparationError(
            "Corpus SHA256 differs from the explicit input specification."
        )
    rows = decode_synthetic_parquet(
        raw,
        max_rows=spec.maximum_rows,
        max_logical_bytes=spec.maximum_logical_bytes,
        max_file_bytes=spec.maximum_file_bytes,
    )
    if any(row.split != split for row in rows):
        raise ExamplePreparationError("Corpus rows belong to a different split.")
    return rows


def prepare_payload(
    *,
    operator: Path,
    export_request: Path,
    export_root: Path,
    base: Path,
    test_spec: Path,
    python_policy: Path | None,
    output: Path,
) -> dict[str, object]:
    """Validate actual stopped selected-best/corpora, then write canonical payload only."""
    from iris.application.scientific_ablation_command_config import (
        IrisScientificAblationCommandConfigV1,
    )
    from iris.application.scientific_ablation_operator import (
        inspect_scientific_ablation_operator_v1,
    )
    from iris.application.scientific_ablation_operator_export import (
        read_scientific_ablation_export_request_v1,
    )
    from iris.application.scientific_ablation_operator_workflow import (
        bind_scientific_ablation_operator_workflow_plan_v1,
        read_scientific_ablation_operator_generation_source_v1,
    )
    from iris.application.scientific_ablation_workflow_io import (
        build_scientific_ablation_workflow_plan_v1,
    )
    from iris.artifacts.code_execution_v2 import PythonSandboxPolicyV2
    from iris.artifacts.scientific_ablation_causal_design_v1 import (
        ScientificAblationMessageCausalPolicyV1,
        ScientificAblationWorkspaceCausalPolicyV1,
    )
    from iris.artifacts.scientific_ablation_collapse_v1 import ScientificAblationCollapsePolicyV1
    from iris.artifacts.scientific_ablation_workflow_v1 import (
        ScientificAblationWorkflowIntentV1,
        ScientificAblationWorkflowParquetV1,
        ScientificAblationWorkflowResourcesV1,
    )
    from iris.artifacts.scientific_generation_v1 import ScientificDecodingPolicyV1
    from iris.interventions.policy import WorkspaceInterventionPlanV1, WorkspaceInterventionSpecV1

    for path in (operator, export_request, export_root, base, test_spec, output):
        local(path)
    fresh(output)
    if any(output.is_relative_to(protected) for protected in (operator, export_root)):
        raise ExamplePreparationError("Keep payload output outside operator and export namespaces.")
    request = read_scientific_ablation_export_request_v1(export_request)
    source = read_scientific_ablation_operator_generation_source_v1(
        operator, request=request, export_root=export_root
    )
    snapshot = inspect_scientific_ablation_operator_v1(operator)
    recipe = IrisScientificAblationCommandConfigV1.model_validate_json(
        snapshot.descriptor.payload.recipe_json, strict=True
    )
    test = ScientificAblationWorkflowParquetV1.model_validate_json(
        json_input(test_spec), strict=True
    )
    groups = (
        load_rows(recipe.train, base, "train"),
        load_rows(recipe.validation, base, "validation"),
        load_rows(test, base, "test"),
    )
    seeds = [row.seed for rows in groups for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ExamplePreparationError("Train/validation/test seeds must be disjoint.")
    policy = (
        None
        if python_policy is None
        else PythonSandboxPolicyV2.model_validate_json(json_input(python_policy), strict=True)
    )
    if any(row.executable for row in groups[2]) and policy is None:
        raise ExamplePreparationError(
            "Executable test rows require an explicit source Python policy."
        )
    intent = ScientificAblationWorkflowIntentV1(
        source_sha256=source.bundle.manifest.sha256,
        lifecycle_sha256=request.payload.lifecycle_sha256,
        inference_sha256=source.bundle.inference.manifest.sha256,
        selected_checkpoint_sha256=request.payload.source_checkpoint_sha256,
        lifecycle_path=str(source.lifecycle_path),
        inference_path=str(source.inference_path),
        inputs_root=str(base),
        training=ScientificAblationWorkflowParquetV1.model_validate_json(
            recipe.train.model_dump_json()
        ),
        validation=ScientificAblationWorkflowParquetV1.model_validate_json(
            recipe.validation.model_dump_json()
        ),
        test=test,
        decoding=ScientificDecodingPolicyV1(strategy="greedy", seed=42),
        controls=(
            WorkspaceInterventionPlanV1(
                interventions=(WorkspaceInterventionSpecV1(kind="zero_workspace"),)
            ),
        ),
        collapse=ScientificAblationCollapsePolicyV1(),
        workspace=ScientificAblationWorkspaceCausalPolicyV1(selection_seed=42, shuffle_seed=1337),
        message=ScientificAblationMessageCausalPolicyV1(selection_seed=42, shuffle_seed=1337),
        resources=ScientificAblationWorkflowResourcesV1(
            maximum_storage_bytes=64 * BYTE_LIMIT,
            minimum_free_bytes=1024**3,
            maximum_kernel_storage_bytes=16 * BYTE_LIMIT,
        ),
        python_policy=policy,
    )
    bind_scientific_ablation_operator_workflow_plan_v1(
        operator,
        request=request,
        export_root=export_root,
        plan=build_scientific_ablation_workflow_plan_v1(intent),
        base=base,
    )
    raw = encoded(intent.model_dump(mode="json"))
    if len(raw) > MAX_JSON:
        raise ExamplePreparationError("Payload exceeds the CLI's 64 KiB saved-intent ceiling.")
    write_new(output, raw)
    return dict(
        kind="iris-ablation-operator-example-payload-v1",
        output=str(output),
        payload_sha256=sha256(raw).hexdigest(),
        source_sha256=intent.source_sha256,
        selected_checkpoint_sha256=intent.selected_checkpoint_sha256,
        workflow_phase_count=8,
        rows=[len(rows) for rows in groups],
        python_policy_present=policy is not None,
        native_or_scientific_acceptance=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser(
        "inputs", help="Write a fresh bounded tiny 5/2/8 example; no model."
    )
    inputs.add_argument("--out", type=Path, required=True)
    payload = commands.add_parser(
        "payload", help="Bind real stopped selected-best DATA; no execution."
    )
    for flag in ("operator", "export-request", "export-root", "base", "test-spec", "out"):
        payload.add_argument("--" + flag, type=Path, required=True)
    payload.add_argument("--python-policy", type=Path)
    args = parser.parse_args()
    try:
        package = local(args.package_root)
        if not (package / "iris/__init__.py").is_file():
            raise ExamplePreparationError(
                "Choose the explicit source or matching installed package root."
            )
        sys.path.insert(0, str(package))
        import iris

        if iris.__file__ is None or Path(iris.__file__).parent != package / "iris":
            raise ExamplePreparationError("The loaded Iris package differs from --package-root.")
        result = (
            prepare_inputs(args.out)
            if args.command == "inputs"
            else prepare_payload(
                operator=args.operator,
                export_request=args.export_request,
                export_root=args.export_root,
                base=args.base,
                test_spec=args.test_spec,
                python_policy=args.python_policy,
                output=args.out,
            )
        )
        print(encoded(result).decode("utf-8"), end="")
        return 0
    except ExamplePreparationError as error:
        print(str(error), file=sys.stderr)
    except Exception:
        print(
            "Example preparation failed. Preserve inputs/partial outputs; "
            "check strict recipe, source export and corpus specifications.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
