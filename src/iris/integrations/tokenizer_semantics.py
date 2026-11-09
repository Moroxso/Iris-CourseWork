"""Shared raw-text fast-tokenizer semantics; no model loading or corpus policy.

The historical fingerprint domain and encoding kwargs are intentionally preserved
for study-v1 compatibility. Callers own offline loading, budgets and metadata
stability checks. This helper does not select task families or latent counts.
"""

import json
from collections.abc import Mapping
from importlib.metadata import version

from transformers import PreTrainedTokenizerFast

from iris.utils.hashing import canonical_json_bytes, domain_sha256


def tokenizer_special_ids(
    tokenizer: PreTrainedTokenizerFast, vocabulary_size: int
) -> tuple[int, ...]:
    values = tokenizer.all_special_ids
    if type(values) is not list or not values:
        raise ValueError("missing special token metadata")
    if any(type(value) is not int or not 0 <= value < vocabulary_size for value in values):
        raise ValueError("invalid special token metadata")
    if len(set(values)) != len(values):
        raise ValueError("duplicate special token metadata")
    return tuple(sorted(values))


def tokenizer_semantics_sha256(tokenizer: PreTrainedTokenizerFast) -> str:
    split_special_tokens = tokenizer.split_special_tokens
    if type(split_special_tokens) is not bool:
        raise ValueError("special-token splitting policy must be an exact boolean")
    # HF mutates backend truncation/padding settings during encode calls. Those
    # are explicit per-call policies here, not changes to vocabulary/normalization.
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("truncation", None)
    backend.pop("padding", None)
    return domain_sha256(
        "iris/study-tokenizer-semantics/v1",
        canonical_json_bytes(
            {
                "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "backend": backend,
                "special_tokens_map": tokenizer.special_tokens_map,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "eos_token_id": tokenizer.eos_token_id,
                "split_special_tokens": split_special_tokens,
                "prompt_policy": "raw_text_add_special_tokens_no_padding_no_truncation",
                "target_policy": "raw_text_no_special_tokens_no_padding_no_truncation_plus_eos",
                "transformers_version": version("transformers"),
                "tokenizers_version": version("tokenizers"),
            }
        ),
    )


def encode_raw_token_ids(
    tokenizer: PreTrainedTokenizerFast, text: str, *, target: bool, vocabulary_size: int
) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=not target,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    if not isinstance(encoded, Mapping) or set(encoded) != {"input_ids", "attention_mask"}:
        raise ValueError("unexpected tokenizer output fields")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if type(ids) is not list or type(mask) is not list or len(ids) != len(mask):
        raise ValueError("invalid unbatched tokenizer output")
    if any(type(value) is not int or not 0 <= value < vocabulary_size for value in ids):
        raise ValueError("invalid token ID")
    if any(type(value) is not int or value != 1 for value in mask):
        raise ValueError("untruncated unpadded text must be fully attended")
    return tuple(ids)


__all__ = ["encode_raw_token_ids", "tokenizer_semantics_sha256", "tokenizer_special_ids"]
