"""Explicit academic chat-boundary prefix; not the scientific suffix contract."""

import torch

from iris.generation.outputs import SecondPassGenerationPrefixOutput, SecondPassSequenceOutput


def chat_boundary_prefix(
    prompt_embeddings: torch.Tensor,
    latents: torch.Tensor,
    boundary: int,
    *,
    preserve_latent_gradients: bool = False,
) -> SecondPassGenerationPrefixOutput:
    """Preserve all tokens, placing real latents before the assistant header."""
    if type(preserve_latent_gradients) is not bool:
        raise ValueError("Explicit gradient mode required")
    if prompt_embeddings.ndim != 3 or latents.ndim != 3:
        raise ValueError("Three-dimensional embeddings required")
    if prompt_embeddings.shape[0] != 1 or latents.shape[0] != 1:
        raise ValueError("One request required")
    if type(boundary) is not int or not 0 < boundary < prompt_embeddings.shape[1]:
        raise ValueError("Interior chat boundary required")
    if latents.shape[1] == 0 or prompt_embeddings.shape[2] != latents.shape[2]:
        raise ValueError("Nonempty matching latent width required")
    if prompt_embeddings.dtype != latents.dtype or prompt_embeddings.device != latents.device:
        raise ValueError("Matching embedding dtype/device required")
    if not torch.isfinite(prompt_embeddings).all() or not torch.isfinite(latents).all():
        raise ValueError("Finite embeddings required")
    values = torch.cat(
        (
            prompt_embeddings.detach()[:, :boundary],
            latents,
            prompt_embeddings.detach()[:, boundary:],
        ),
        dim=1,
    )
    if not preserve_latent_gradients:
        values = values.detach()
    length = values.shape[1]
    return SecondPassGenerationPrefixOutput(
        inputs_embeds=values,
        attention_mask=torch.ones((1, length), dtype=torch.bool, device=values.device),
        position_ids=torch.arange(length, device=values.device).unsqueeze(0),
    )


def chat_boundary_teacher_sequence(
    prompt_embeddings: torch.Tensor,
    latents: torch.Tensor,
    boundary: int,
    target_ids: torch.Tensor,
    target_embeddings: torch.Tensor,
) -> SecondPassSequenceOutput:
    """Teacher labels align to target tokens; gradients flow only through latents."""
    prefix = chat_boundary_prefix(
        prompt_embeddings, latents, boundary, preserve_latent_gradients=True
    )
    if (
        target_ids.ndim != 2
        or target_ids.shape[0] != 1
        or target_ids.shape[1] == 0
        or target_ids.dtype != torch.long
        or target_ids.device != latents.device
    ):
        raise ValueError("Nonempty int64 target IDs on the latent device required")
    if (
        target_embeddings.shape != (*target_ids.shape, latents.shape[-1])
        or target_embeddings.dtype != latents.dtype
        or target_embeddings.device != latents.device
        or not torch.isfinite(target_embeddings).all()
    ):
        raise ValueError("Matching finite target embeddings required")
    values = torch.cat((prefix.inputs_embeds, target_embeddings.detach()), dim=1)
    labels = torch.cat(
        (
            torch.full(prefix.attention_mask.shape, -100, dtype=torch.long, device=latents.device),
            target_ids.detach().clone(),
        ),
        dim=1,
    )
    length = values.shape[1]
    return SecondPassSequenceOutput(
        inputs_embeds=values,
        labels=labels,
        attention_mask=torch.ones((1, length), dtype=torch.bool, device=latents.device),
        position_ids=torch.arange(length, device=latents.device).unsqueeze(0),
    )
