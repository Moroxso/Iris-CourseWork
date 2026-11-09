"""Prepare fresh paired mode INPUTS for tiny or actual packaged Iris geometry.

This is DATA preparation for the remaining default/all12 package gates, not a
trained fixture, graph allocation, scientific dataset or launch approval. The
backbone stays the explicit offline RANDOM tiny-Qwen fixture, never pretrained.
No payload command: a future real selected export and measured workflow budgets
must be bound separately. A declared storage ceiling is not preallocation.
"""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any


def _mode(routing: str, goal: str, broadcast: str, routing_seed: int | None) -> dict[str, object]:
    from prepare_scientific_ablation_example_v2 import ExamplePreparationError

    if (
        routing not in ("learned", "random", "uniform")
        or goal not in ("enabled", "disabled")
        or broadcast not in ("enabled", "disabled")
    ):
        raise ExamplePreparationError("Choose an explicit supported routing/goal/broadcast mode.")
    if routing == "random":
        if type(routing_seed) is not int or not 0 <= routing_seed <= 2**63 - 1:
            raise ExamplePreparationError(
                "Random routing requires an explicit integer --routing-seed."
            )
    elif routing_seed is not None:
        raise ExamplePreparationError("Only random routing accepts --routing-seed.")
    return dict(routing=routing, goal=goal, broadcast=broadcast, routing_seed=routing_seed)


def _geometry(profile: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[object]]]:
    from prepare_scientific_ablation_example_v2 import MAX_JSON, ExamplePreparationError, read

    if profile == "tiny":
        return (
            dict(
                name="tiny-example",
                state_dim=8,
                num_modules=4,
                layer_sampler=dict(layer_fractions=[0.25, 0.5, 0.75, 1.0]),
            ),
            dict(
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
            {},
        )
    if profile != "iris-default":
        raise ExamplePreparationError("Choose tiny or iris-default geometry explicitly.")
    from omegaconf import OmegaConf

    import iris
    from iris.schemas import CognitiveConfig, WorkspaceConfig

    if iris.__file__ is None:
        raise ExamplePreparationError("A concrete selected package is required.")
    package = Path(iris.__file__).parent
    values: dict[str, dict[str, Any]] = {}
    pins: dict[str, list[object]] = {}
    for group, schema in (("cognitive", CognitiveConfig), ("workspace", WorkspaceConfig)):
        name = "configs/" + group + "/iris_gw_0_1.yaml"
        raw = read(package / name, MAX_JSON)
        # Packaged DATA only; do not initialize/mutate global Hydra composition.
        config = OmegaConf.create(raw.decode("utf-8"))
        plain = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        values[group] = schema.model_validate(plain).model_dump(mode="json")
        pins[name] = [len(raw), sha256(raw).hexdigest()]
    return values["cognitive"], values["workspace"], pins


def prepare_inputs(
    root: Path,
    *,
    geometry: str = "tiny",
    routing: str = "learned",
    goal: str = "enabled",
    broadcast: str = "enabled",
    routing_seed: int | None = None,
) -> dict[str, object]:
    """Fresh five-file mode recipe; no model, source-owner or workflow execution."""
    from prepare_scientific_ablation_example_v2 import (
        BYTE_LIMIT,
        ExamplePreparationError,
        encoded,
        fresh,
        input_spec,
        write_new,
    )

    from iris.application.scientific_ablation_command_config import (
        IrisScientificAblationCommandConfigV1,
    )
    from iris.artifacts.scientific_evaluation_v1 import TASKS
    from iris.data.generator import generate_synthetic_split
    from iris.data.parquet import encode_synthetic_parquet
    from iris.data.schemas import SyntheticSplitPlan

    ablation = _mode(routing, goal, broadcast, routing_seed)
    cognitive, workspace, resource_pins = _geometry(geometry)
    lifecycle_ceiling = 1024**3 if geometry == "iris-default" else 32 * 1024**2
    parameter_hint = 64 * 1024**2 if geometry == "iris-default" else 16 * 1024**2
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
                cognitive=cognitive,
                workspace=workspace,
                ablation=ablation,
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
                    maximum_storage_bytes=lifecycle_ceiling,
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
        kind="iris-ablation-mode-inputs-v1",
        geometry_profile=geometry,
        routing=routing,
        goal=goal,
        broadcast=broadcast,
        routing_seed=routing_seed,
        config_resource_pins=resource_pins,
        maximum_parameter_bytes_hint=parameter_hint,
        lifecycle_storage_ceiling=lifecycle_ceiling,
        real_default_graph_allocated=False,
        full_default_mode_workflow_acceptance=False,
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


def main() -> int:
    from prepare_scientific_ablation_example_v2 import ExamplePreparationError, encoded, local

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[1] / "src"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry", choices=("tiny", "iris-default"), default="tiny")
    parser.add_argument("--routing", choices=("learned", "random", "uniform"), default="learned")
    parser.add_argument("--goal", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--broadcast", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--routing-seed", type=int)
    args = parser.parse_args()
    try:
        package = local(args.package_root)
        if not (package / "iris/__init__.py").is_file():
            raise ExamplePreparationError(
                "Choose the explicit source or matching installed package."
            )
        sys.path.insert(0, str(package))
        import iris

        if iris.__file__ is None or Path(iris.__file__).parent != package / "iris":
            raise ExamplePreparationError("Loaded Iris differs from --package-root.")
        result = prepare_inputs(
            args.out,
            geometry=args.geometry,
            routing=args.routing,
            goal=args.goal,
            broadcast=args.broadcast,
            routing_seed=args.routing_seed,
        )
        print(encoded(result).decode("utf-8"), end="")
        return 0
    except ExamplePreparationError as error:
        print(str(error), file=sys.stderr)
    except Exception:
        print(
            "Mode input preparation failed. Preserve partial files; "
            "inspect strict package/config/data inputs.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
