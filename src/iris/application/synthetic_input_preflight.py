"""Streaming cache-only token-budget diagnostics for explicit four-family data.

No model weights, candidate execution, publication, profile selection or launch
approval. Latent reservations are caller-owned scenarios, not arm attestations.
"""

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from iris.application.errors import IrisSyntheticInputPreflightError
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticDatasetPlan, SyntheticTemplate
from iris.integrations.tokenizer_semantics import (
    encode_raw_token_ids,
    tokenizer_semantics_sha256,
    tokenizer_special_ids,
)
from iris.utils.hashing import canonical_json_bytes, domain_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class SyntheticTokenizerSpec(_StrictModel):
    """Explicit immutable asset revisions. Loading is always fast and cache-only."""

    repository: Literal["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    padding_side: Literal["left", "right"]
    truncation_side: Literal["left", "right"]


class SyntheticContextBudget(_StrictModel):
    """One explicit envelope; names do not certify hardware or B/C/D semantics."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    max_prompt_tokens: int = Field(ge=1, le=1048576)
    max_target_tokens_including_terminal: int = Field(ge=2, le=1048576)
    reserved_latent_tokens: int = Field(ge=0, le=65536)


@dataclass
class _Totals:
    rows: int = 0
    max_prompt: int = 0
    max_target: int = 0
    content_violations: Counter[str] = field(default_factory=Counter)
    invalid_rows: int = 0
    budget_violations: dict[str, Counter[str]] = field(default_factory=dict)
    budget_invalid_rows: Counter[str] = field(default_factory=Counter)

    def observe(
        self,
        prompt: int,
        target: int,
        content_violations: tuple[str, ...],
        budget_violations: dict[str, tuple[str, ...]],
    ) -> None:
        self.rows += 1
        self.max_prompt = max(self.max_prompt, prompt)
        self.max_target = max(self.max_target, target)
        self.content_violations.update(content_violations)
        self.invalid_rows += bool(content_violations)
        for name, violations in budget_violations.items():
            self.budget_violations.setdefault(name, Counter()).update(violations)
            self.budget_invalid_rows[name] += bool(content_violations or violations)

    def report(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "max_prompt_tokens": self.max_prompt,
            "max_target_tokens_including_terminal": self.max_target,
            "invalid_content_rows": self.invalid_rows,
            "content_violations": dict(sorted(self.content_violations.items())),
            "budgets": {
                name: {
                    "invalid_rows": self.budget_invalid_rows[name],
                    "violations": dict(sorted(violations.items())),
                }
                for name, violations in sorted(self.budget_violations.items())
            },
        }


def _positive_integer(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid positive model metadata")
    return value


def _model_metadata(config: Any) -> dict[str, Any]:
    if config.model_type != "qwen3":
        raise ValueError("Unexpected model type")
    vocabulary_size = _positive_integer(config.vocab_size)
    positions = _positive_integer(config.max_position_embeddings)
    eos = config.eos_token_id
    eos_ids = (eos,) if type(eos) is int else tuple(eos) if type(eos) in (list, tuple) else ()
    if (
        not eos_ids
        or any(type(item) is not int or not 0 <= item < vocabulary_size for item in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("Invalid model EOS metadata")
    return {
        "model_type": "qwen3",
        "vocab_size": vocabulary_size,
        "max_position_embeddings": positions,
        "eos_token_ids": list(eos_ids),
    }


def _budget_violations(
    budget: SyntheticContextBudget, prompt: int, target: int, positions: int
) -> tuple[str, ...]:
    violations = []
    if prompt > budget.max_prompt_tokens:
        violations.append("prompt_over_limit")
    if target > budget.max_target_tokens_including_terminal:
        violations.append("target_over_limit")
    if prompt + budget.reserved_latent_tokens + target > positions:
        violations.append("observed_context_over_limit")
    return tuple(violations)


def preflight_synthetic_inputs(
    plan: SyntheticDatasetPlan,
    *,
    template_id: SyntheticTemplate,
    tokenizer_spec: SyntheticTokenizerSpec,
    terminal_token_id: int,
    budgets: tuple[SyntheticContextBudget, ...],
    max_rows: int,
    max_logical_bytes: int,
) -> dict[str, Any]:
    """Check every declared row; overflow returns complete diagnostics, never truncation.

    Processing retains one row and bounded aggregate counters, not a full corpus
    or per-row report. Logical input bounds are not peak-RSS guarantees. The report
    contains private corpus identities but no text or token IDs. It is not an
    authenticated receipt, hardware measurement or scientific release decision.
    """
    try:
        if type(plan) is not SyntheticDatasetPlan:
            raise TypeError("Exact plan required")
        owned = SyntheticDatasetPlan.model_validate(plan.model_dump(mode="python"))
        if type(tokenizer_spec) is not SyntheticTokenizerSpec:
            raise TypeError("Exact tokenizer specification required")
        assets = SyntheticTokenizerSpec.model_validate(tokenizer_spec.model_dump(mode="python"))
        if type(budgets) is not tuple or not 1 <= len(budgets) <= 8:
            raise ValueError("One to eight explicit budgets required")
        if any(type(item) is not SyntheticContextBudget for item in budgets):
            raise TypeError("Exact context budgets required")
        limits = tuple(
            SyntheticContextBudget.model_validate(item.model_dump(mode="python"))
            for item in budgets
        )
        if len({item.name for item in limits}) != len(limits):
            raise ValueError("Duplicate context budget names")
        if type(template_id) is not str or template_id not in (
            "iris-synthetic-01-v1",
            "iris-synthetic-01-v2",
        ):
            raise ValueError("Unknown synthetic template")
        if template_id == "iris-synthetic-01-v2" and any(
            split.first_seed + split.count > 2**26 for split in owned.splits
        ):
            raise ValueError("Synthetic v2 seed domain exceeded")
        if type(max_rows) is not int or not 1 <= max_rows <= 100000:
            raise ValueError("Invalid row budget")
        if type(max_logical_bytes) is not int or not 1 <= max_logical_bytes <= 512 * 1024**2:
            raise ValueError("Invalid logical byte budget")
        if sum(split.count for split in owned.splits) > max_rows:
            raise ValueError("Plan exceeds row budget")
        if type(terminal_token_id) is not int or terminal_token_id < 0:
            raise ValueError("Invalid terminal token ID")
        declared = {
            "plan": owned.model_dump(mode="json"),
            "template_id": template_id,
            "tokenizer_spec": assets.model_dump(mode="json"),
            "terminal_token_id": terminal_token_id,
            "budgets": [item.model_dump(mode="json") for item in limits],
        }
        declared_bytes = canonical_json_bytes(declared)
        model_config: Any = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=assets.repository,
            revision=assets.model_revision,
            trust_remote_code=False,
            local_files_only=True,
        )
        metadata = _model_metadata(model_config)
        vocabulary_size = metadata["vocab_size"]
        positions = metadata["max_position_embeddings"]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=assets.repository,
            revision=assets.tokenizer_revision,
            trust_remote_code=False,
            local_files_only=True,
            use_fast=True,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("Loader did not return a fast tokenizer")
        tokenizer.padding_side = assets.padding_side
        tokenizer.truncation_side = assets.truncation_side
        specials = tokenizer_special_ids(tokenizer, vocabulary_size)
        if (
            type(tokenizer.eos_token_id) is not int
            or terminal_token_id != tokenizer.eos_token_id
            or terminal_token_id not in specials
            or terminal_token_id not in metadata["eos_token_ids"]
        ):
            raise ValueError("Explicit terminal must be a tokenizer and model EOS")
        fingerprint = tokenizer_semantics_sha256(tokenizer)
        total = _Totals()
        strata: dict[str, dict[str, _Totals]] = {}
        splits: dict[str, Any] = {}
        logical_bytes = 0
        logical_digest = sha256(b"iris/synthetic-01/logical-rows/v1\0")
        token_digest = sha256(b"iris/synthetic-input-token-stream/v1\0")
        for split in owned.splits:
            split_total = _Totals()
            cells: dict[str, _Totals] = {}
            for row in generate_synthetic_split(split, template_id=template_id):
                if split_total.rows >= split.count or type(row) is not IrisExample:
                    raise ValueError("Invalid generator row or excess count")
                row = IrisExample.model_validate(row.model_dump(mode="python"))
                if (
                    row.template_id != template_id
                    or row.split != split.split
                    or row.seed != split.first_seed + split_total.rows
                ):
                    raise ValueError("Generator membership differs from declared plan")
                raw = canonical_json_bytes(row.model_dump(mode="json"))
                logical_bytes += len(raw)
                if logical_bytes > max_logical_bytes:
                    raise ValueError("Logical data exceeds byte budget")
                logical_digest.update(len(raw).to_bytes(8, "big"))
                logical_digest.update(raw)
                prompt_ids = encode_raw_token_ids(
                    tokenizer, row.prompt, target=False, vocabulary_size=vocabulary_size
                )
                content_ids = encode_raw_token_ids(
                    tokenizer, row.expected_response, target=True, vocabulary_size=vocabulary_size
                )
                target_ids = (*content_ids, terminal_token_id)
                encoded = canonical_json_bytes(
                    {
                        "id": row.id,
                        "logical_row_sha256": sha256(raw).hexdigest(),
                        "prompt_ids": prompt_ids,
                        "target_ids": target_ids,
                    }
                )
                token_digest.update(len(encoded).to_bytes(8, "big"))
                token_digest.update(encoded)
                violations = []
                if not prompt_ids:
                    violations.append("empty_prompt_encoding")
                if not content_ids:
                    violations.append("empty_target_encoding")
                if any(item in specials for item in content_ids):
                    violations.append("target_contains_special_tokens")
                per_budget = {
                    item.name: _budget_violations(item, len(prompt_ids), len(target_ids), positions)
                    for item in limits
                }
                cell = cells.setdefault(f"{row.task_type}/difficulty-{row.difficulty}", _Totals())
                for aggregate in (total, split_total, cell):
                    aggregate.observe(
                        len(prompt_ids), len(target_ids), tuple(violations), per_budget
                    )
            if split_total.rows != split.count:
                raise ValueError("Generator returned an incomplete split")
            splits[split.split] = split_total.report()
            strata[split.split] = cells
        if (
            tokenizer_semantics_sha256(tokenizer) != fingerprint
            or tokenizer_special_ids(tokenizer, vocabulary_size) != specials
            or _model_metadata(model_config) != metadata
            or canonical_json_bytes(
                {
                    "plan": owned.model_dump(mode="json"),
                    "template_id": template_id,
                    "tokenizer_spec": assets.model_dump(mode="json"),
                    "terminal_token_id": terminal_token_id,
                    "budgets": [item.model_dump(mode="json") for item in limits],
                }
            )
            != declared_bytes
        ):
            raise ValueError("Tokenizer, metadata or declared inputs changed during preflight")
        budget_results = {}
        for item in limits:
            envelope = (
                item.max_prompt_tokens
                + item.reserved_latent_tokens
                + item.max_target_tokens_including_terminal
            )
            budget_results[item.name] = {
                "configured_context_tokens": envelope,
                "configured_context_envelope_fits": envelope <= positions,
                "inputs_fit": envelope <= positions and not total.budget_invalid_rows[item.name],
            }
        return {
            "kind": "iris-synthetic-input-preflight-v1",
            **declared,
            "declaration_sha256": domain_sha256(
                "iris/synthetic-input-preflight-declaration/v1", declared_bytes
            ),
            "plan_sha256": domain_sha256(
                "iris/synthetic-audit-plan/v1", canonical_json_bytes(declared["plan"])
            ),
            "tokenizer_semantics_sha256": fingerprint,
            "model_context_metadata": metadata,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
            "logical_sha256": logical_digest.hexdigest(),
            "logical_bytes": logical_bytes,
            "token_stream_sha256": token_digest.hexdigest(),
            "totals": total.report(),
            "splits": splits,
            "strata": {
                split_name: {key: cell.report() for key, cell in sorted(cells.items())}
                for split_name, cells in strata.items()
            },
            "budget_results": budget_results,
            "inputs_fit": all(item["inputs_fit"] for item in budget_results.values()),
            "scientific_launch_status": "not_assessed",
            "hardware_fit_verified": False,
            "model_weights_loaded": False,
            "python_sources_executed": False,
        }
    except Exception:
        raise IrisSyntheticInputPreflightError(
            "Iris synthetic input preflight could not complete"
        ) from None


__all__ = ["SyntheticContextBudget", "SyntheticTokenizerSpec", "preflight_synthetic_inputs"]
