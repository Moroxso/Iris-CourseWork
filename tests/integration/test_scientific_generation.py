"""Actual tiny-Qwen held-out decoding; no substituted reference completions."""

import json
from math import fsum
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from iris.application import scientific_command as command
from iris.application import scientific_generation as generation
from iris.application.scientific_checkpoint import (
    build_scientific_checkpoint_v1,
    restore_scientific_checkpoint_v1,
)
from iris.application.scientific_execution import (
    _checkpoint_for_event,
    _selected_best,
    read_scientific_execution_v1,
)
from iris.application.scientific_lifecycle import _bundle_parts
from iris.application.scientific_optimization import _same_runtime
from iris.artifacts.scientific_generation_v1 import (
    IrisScientificGenerationV1,
    ScientificDecodingPolicyV1,
    ScientificGenerationPayloadV1,
)
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplit, SyntheticSplitPlan
from iris.utils.directory_publication import rename_directory_with_windows_retry
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_command import _edit, _files, _input, _recipe
from tests.integration.test_scientific_command import loaders as loaders

pytestmark = pytest.mark.integration


def _test_rows() -> tuple[IrisExample, ...]:
    rows = generate_synthetic_split(
        SyntheticSplitPlan(split="test", first_seed=2000000, count=15),
        template_id="iris-synthetic-01-v2",
    )
    return tuple({row.task_type: row for row in rows}.values())


def _test_spec(path: Path, rows: tuple[IrisExample, ...]) -> command.ScientificParquetInputV1:
    return command.ScientificParquetInputV1.model_validate(_input(path, rows))


@pytest.fixture
def completed(tmp_path: Path, loaders: Any) -> tuple[Path, Path]:
    recipe = _recipe(tmp_path)
    _edit(
        recipe,
        lambda raw: raw["lifecycle"].update(
            minimum_improvement=1e6,
            early_stopping_patience=2,
        ),
    )
    created = command.create_scientific_command_v1(
        config_path=recipe, output_root=tmp_path / "runs"
    )
    done = command.resume_scientific_command_v1(
        config_path=recipe, execution_path=created.path, phases=1000
    )
    assert done.state.phase == "stopped" and done.state.step == 3 and done.state.best_step == 1
    return recipe, done.path


def _generate(
    completed: tuple[Path, Path],
    spec: command.ScientificParquetInputV1,
    *,
    strategy: str = "greedy",
) -> IrisScientificGenerationV1:
    return generation.generate_scientific_test_v1(
        config_path=completed[0],
        execution_path=completed[1],
        test=spec,
        decoding=ScientificDecodingPolicyV1.model_validate(
            {
                "strategy": strategy,
                "seed": 1337,
                "temperature": 0.7 if strategy == "sampling" else 1.0,
                "top_p": 0.8 if strategy == "sampling" else 1.0,
            }
        ),
    )


def test_actual_multitoken_capture_uses_best_and_preserves_source_and_rng(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    rows = _test_rows()
    spec = _test_spec(tmp_path / "test.parquet", rows)
    before_files = _files(completed[1])
    before = capture_runtime_state(torch.device("cpu"))
    result = _generate(completed, spec)
    payload = result.payload
    assert (payload.selected_step, payload.terminal_step) == (1, 3)
    assert payload.eos_token_ids == (1,)
    assert tuple(row.example_id for row in payload.rows) == tuple(row.id for row in rows)
    assert len({row.task_type for row in payload.rows}) == 4
    assert any(len(row.token_ids) > 1 for row in payload.rows)
    assert all(len(row.token_ids) <= 256 for row in payload.rows)
    assert all(row.sampling_seed is None for row in payload.rows)
    assert all(
        row.output_text != source.expected_response
        for row, source in zip(payload.rows, rows, strict=True)
    )
    assert _files(completed[1]) == before_files
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))

    # Independently measure the saved best; the later training state is NOT used.
    snapshot = read_scientific_execution_v1(completed[1])
    selected = None
    for event in snapshot.events:
        selected = _selected_best(selected, event)
    current, best = _bundle_parts(
        _checkpoint_for_event(
            snapshot.path,
            snapshot.header,
            snapshot.events[-1],
            best_event=selected,
        )
    )
    assert best is not None and current.state_bytes != best.state_bytes
    assert payload.best_checkpoint_sha256 == best.manifest.sha256
    assert selected is not None and payload.best_validation_event_sha256 == selected.sha256
    config = command.read_scientific_command_config_v1(completed[0])
    with command._composed(config, base=tmp_path) as lifecycle:
        run = lifecycle._run
        restore_scientific_checkpoint_v1(best, run=run)
        model = run._optimizer._model
        model.eval()
        sums, counts = [], []
        try:
            for row, captured in zip(rows, payload.rows, strict=True):
                out = model.evaluation_forward(row, terminal_token_id=config.terminal_token_id)
                assert captured.target_nll_sum == float(out.losses.language_model)
                assert captured.target_token_count == out.supervised_token_count
                sums.append(float(out.losses.language_model))
                counts.append(out.supervised_token_count)
        finally:
            model.train(True)
        assert build_scientific_checkpoint_v1(run) == best
    assert payload.target_nll_per_token == fsum(sums) / sum(counts)
    assert payload.target_token_count == sum(counts)
    assert len(set(counts)) > 1

    path = generation.write_scientific_generation_v1(result, root=tmp_path / "captures")
    assert generation.read_scientific_generation_v1(path) == result
    assert generation.write_scientific_generation_v1(result, root=tmp_path / "captures") == path
    content = (path / "generation.json").read_text(encoding="utf-8")
    assert '"expected_response"' not in content and '"hidden_facts"' not in content
    assert '"prompt"' not in content and '"test_code"' not in content
    assert rows[0].prompt not in repr(result)


def test_sampling_is_reproducible_input_only_and_independent_of_row_order(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    rows = _test_rows()[-2:]
    first = _generate(completed, _test_spec(tmp_path / "test.parquet", rows), strategy="sampling")
    torch.manual_seed(987654321)
    repeated = _generate(
        completed, _test_spec(tmp_path / "test.parquet", rows), strategy="sampling"
    )
    assert repeated == first
    reordered = _generate(
        completed, _test_spec(tmp_path / "test.parquet", rows[::-1]), strategy="sampling"
    )
    assert reordered.sha256 != first.sha256
    assert {row.example_id: row for row in reordered.payload.rows} == {
        row.example_id: row for row in first.payload.rows
    }
    raw = rows[0].model_dump(mode="python")
    answer = json.loads(raw["expected_response"])
    answer["content"] = "deliberately changed reference only"
    raw["expected_response"] = json.dumps(answer)
    changed = (IrisExample.model_validate(raw), rows[1])
    output = _generate(
        completed, _test_spec(tmp_path / "test.parquet", changed), strategy="sampling"
    )
    for original, updated in zip(first.payload.rows, output.payload.rows, strict=True):
        assert original.sampling_seed == updated.sampling_seed
        assert (
            original.token_ids == updated.token_ids and original.output_text == updated.output_text
        )
    assert output.payload.rows[0].target_nll_sum != first.payload.rows[0].target_nll_sum
    assert output.payload.rows[0].example_sha256 != first.payload.rows[0].example_sha256


@pytest.mark.parametrize("phases", [0, 1, 2, 3, 4, 5, 6, 7])
def test_incomplete_source_rejects_before_model_loading(
    tmp_path: Path,
    loaders: Any,
    phases: int,
) -> None:
    recipe = _recipe(tmp_path)
    report = command.create_scientific_command_v1(config_path=recipe, output_root=tmp_path / "runs")
    if phases:
        command.resume_scientific_command_v1(
            config_path=recipe, execution_path=report.path, phases=phases
        )
    spec = _test_spec(tmp_path / "test.parquet", _test_rows()[:1])
    with (
        patch.object(generation, "_composed", side_effect=AssertionError("must not load")) as load,
        pytest.raises(generation.IrisScientificGenerationError, match="did not complete"),
    ):
        _generate((recipe, report.path), spec)
    load.assert_not_called()


@pytest.mark.parametrize("kind", ["train", "validation", "duplicate", "v1", "digest", "oversize"])
def test_invalid_test_input_rejects_before_model_loading(
    tmp_path: Path,
    completed: tuple[Path, Path],
    kind: str,
) -> None:
    rows = _test_rows()[:1]
    if kind in {"train", "validation"}:
        rows = tuple(
            generate_synthetic_split(
                SyntheticSplitPlan(split=cast(SyntheticSplit, kind), first_seed=3000000, count=15),
                template_id="iris-synthetic-01-v2",
            )
        )[:1]
    elif kind == "duplicate":
        rows = rows * 2
    elif kind == "v1":
        rows = tuple(
            generate_synthetic_split(
                SyntheticSplitPlan(split="test", first_seed=2000000, count=15),
                template_id="iris-synthetic-01-v1",
            )
        )[:1]
    # Parquet itself rejects duplicate rows; exercise the service's ownership too.
    if kind == "duplicate":
        with pytest.raises(ValueError):
            generation._own_test_rows(rows, excluded_seeds=set(), scientific=False)
        return
    spec = _test_spec(tmp_path / "test.parquet", rows)
    if kind == "digest":
        spec = spec.model_copy(update={"sha256": "0" * 64})
    if kind == "oversize":
        spec = spec.model_copy(update={"maximum_rows": 5001})
    with (
        patch.object(generation, "_composed", side_effect=AssertionError("must not load")) as load,
        pytest.raises(generation.IrisScientificGenerationError),
    ):
        _generate(completed, spec)
    load.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    ["train_seed", "validation_seed", "validation_recipe", "validation_order", "initial_seed"],
)
def test_context_mismatch_or_seed_leakage_rejects_before_request(
    tmp_path: Path,
    completed: tuple[Path, Path],
    kind: str,
) -> None:
    rows = _test_rows()[:1]
    if kind in {"train_seed", "validation_seed"}:
        seed = 0 if kind == "train_seed" else 1000000
        rows = tuple(
            generate_synthetic_split(
                SyntheticSplitPlan(split="test", first_seed=seed, count=15),
                template_id="iris-synthetic-01-v2",
            )
        )[:1]
    elif kind == "validation_recipe":
        _edit(completed[0], lambda raw: raw["lifecycle"].update(minimum_improvement=42.0))
    elif kind == "validation_order":
        config = command.read_scientific_command_config_v1(completed[0])
        validation = command._rows(config.validation, tmp_path)
        replacement = _input(tmp_path / "validation.parquet", validation[::-1])
        _edit(completed[0], lambda raw: raw.update(validation=replacement))
    else:
        _edit(completed[0], lambda raw: raw.update(initialization_seed=5678))
    spec = _test_spec(tmp_path / "test.parquet", rows)
    before = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(
            generation, "build_request_orchestrator", side_effect=AssertionError("no request")
        ) as request,
        pytest.raises(generation.IrisScientificGenerationError),
    ):
        _generate(completed, spec)
    request.assert_not_called()
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))


def test_failure_is_redacted_restores_rng_and_leaves_source_retryable(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    spec = _test_spec(tmp_path / "test.parquet", _test_rows()[-1:])
    before, files = capture_runtime_state(torch.device("cpu")), _files(completed[1])
    with (
        patch(
            "iris.application.request.IrisRequestOrchestrator.generate",
            side_effect=RuntimeError("PRIVATE PROMPT"),
        ),
        pytest.raises(generation.IrisScientificGenerationError) as error,
    ):
        _generate(completed, spec)
    assert "PRIVATE" not in str(error.value) and error.value.__cause__ is None
    assert _same_runtime(before, capture_runtime_state(torch.device("cpu")))
    assert _files(completed[1]) == files
    assert len(_generate(completed, spec).payload.rows) == 1


def test_capture_quota_corruption_and_uncertain_publication(
    tmp_path: Path,
    completed: tuple[Path, Path],
) -> None:
    result = _generate(completed, _test_spec(tmp_path / "test.parquet", _test_rows()[-1:]))
    with pytest.raises(generation.IrisScientificGenerationError):
        generation.write_scientific_generation_v1(
            result, root=tmp_path / "quota", maximum_storage_bytes=1
        )
    assert list((tmp_path / "quota" / "scientific-generation-v1").iterdir()) == []
    rename = rename_directory_with_windows_retry

    def uncertain(source: Path, target: Path) -> None:
        rename(source, target)
        raise OSError("lost acknowledgement")

    with patch.object(generation, "rename_directory_with_windows_retry", side_effect=uncertain):
        path = generation.write_scientific_generation_v1(result, root=tmp_path / "captures")
    assert generation.read_scientific_generation_v1(path) == result
    file = path / "generation.json"
    content = file.read_bytes()
    file.write_bytes(content + b" ")
    with pytest.raises(generation.IrisScientificGenerationError):
        generation.read_scientific_generation_v1(path)
    with pytest.raises(generation.IrisScientificGenerationError):
        generation.write_scientific_generation_v1(result, root=tmp_path / "captures")
    assert file.read_bytes() == content + b" "


def test_schema_rejects_contradictory_tokens_and_seed(
    tmp_path: Path, completed: tuple[Path, Path]
) -> None:
    result = _generate(completed, _test_spec(tmp_path / "test.parquet", _test_rows()[-1:]))
    for update in (
        {"sampling_seed": 12},
        {"token_ids": (1, 2)},
        {"target_nll_sum": float("nan")},
        {"matched_stop_token_id": 99},
    ):
        row = result.payload.rows[0].model_copy(update=update)
        corrupt = result.model_copy(
            update={"payload": result.payload.model_copy(update={"rows": (row,)})}
        )
        with pytest.raises(generation.IrisScientificGenerationError):
            generation.write_scientific_generation_v1(corrupt, root=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    overflow = result.payload.model_dump(mode="python")
    first, second = (dict(overflow["rows"][0]) for _ in range(2))
    first.update(example_id="test-9000000", target_nll_sum=1e308)
    second.update(example_id="test-9000001", target_nll_sum=1e308)
    overflow["rows"] = (first, second)
    with pytest.raises(ValueError, match="aggregation overflowed"):
        ScientificGenerationPayloadV1.model_validate(overflow)


def test_public_api_resolves_to_canonical_service() -> None:
    import iris.application as public

    assert public.generate_scientific_test_v1 is generation.generate_scientific_test_v1
    assert public.read_scientific_generation_v1 is generation.read_scientific_generation_v1
    assert public.write_scientific_generation_v1 is generation.write_scientific_generation_v1
    assert public.ScientificDecodingPolicyV1 is ScientificDecodingPolicyV1
    assert public.IrisScientificGenerationV1 is IrisScientificGenerationV1
