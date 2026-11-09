"""Teacher-forced second-pass sequence assembly for Iris-GW-0.1."""

import torch
from torch import nn

from iris.generation.outputs import SecondPassSequenceOutput
from iris.workspace.outputs import LatentBridgeOutput


class SecondPassSequenceAssemblyError(RuntimeError):
    """Raised when prompt, workspace, and target tensors cannot form one sequence."""


class SecondPassSequenceAssembler(nn.Module):
    """Assemble `[prompt][workspace][target]` without shifting target labels."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_slots: int,
        vocabulary_size: int,
        max_prompt_tokens: int,
        max_target_tokens: int,
        max_context_tokens: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_slots": num_slots,
            "vocabulary_size": vocabulary_size,
            "max_prompt_tokens": max_prompt_tokens,
            "max_target_tokens": max_target_tokens,
            "max_context_tokens": max_context_tokens,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if max_prompt_tokens + num_slots + max_target_tokens > max_context_tokens:
            message = "Configured prompt, workspace, and target maxima exceed the context window"
            raise ValueError(message)

        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.vocabulary_size = vocabulary_size
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.max_context_tokens = max_context_tokens

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        workspace: LatentBridgeOutput,
        target_input_ids: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> SecondPassSequenceOutput:
        """Return one teacher-forced sequence using the Hugging Face label convention."""
        batch_size, prompt_length, dtype, device = self._validate_prompt_embeddings(
            prompt_embeddings
        )
        prompt_mask = self._validate_prompt_mask(
            prompt_attention_mask,
            batch_size=batch_size,
            prompt_length=prompt_length,
            device=device,
        )
        prompt_logical_lengths = prompt_mask.to(dtype=torch.int64).sum(dim=-1)
        self._validate_workspace(
            workspace,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            prompt_logical_lengths=prompt_logical_lengths,
        )
        target_length = self._validate_target_embeddings(
            target_embeddings,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        target_mask = self._validate_target_mask(
            target_attention_mask,
            batch_size=batch_size,
            target_length=target_length,
            device=device,
        )
        self._validate_target_ids(
            target_input_ids,
            batch_size=batch_size,
            target_length=target_length,
            device=device,
        )
        total_length = prompt_length + self.num_slots + target_length
        if total_length > self.max_context_tokens:
            message = "Physical second-pass sequence exceeds the context window"
            raise SecondPassSequenceAssemblyError(message)

        prompt_values = self._materialize(prompt_embeddings)
        workspace_values = self._materialize(workspace.workspace_embeddings)
        target_values = self._materialize(target_embeddings)
        inputs_embeds = torch.cat(
            (prompt_values, workspace_values, target_values),
            dim=1,
        )
        attention_mask = torch.cat(
            (prompt_mask, workspace.workspace_attention_mask.clone(), target_mask),
            dim=1,
        )
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        position_ids = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        if not torch.equal(
            position_ids[:, prompt_length : prompt_length + self.num_slots],
            workspace.workspace_position_ids,
        ):
            message = "Workspace positions disagree with the assembled logical sequence"
            raise SecondPassSequenceAssemblyError(message)
        if not bool((position_ids < self.max_context_tokens).all()):
            message = "Logical second-pass positions exceed the context window"
            raise SecondPassSequenceAssemblyError(message)

        ignored_prefix = torch.full(
            (batch_size, prompt_length + self.num_slots),
            -100,
            dtype=torch.int64,
            device=device,
        )
        target_labels = torch.where(
            target_mask,
            target_input_ids,
            torch.full_like(target_input_ids, -100),
        )
        labels = torch.cat((ignored_prefix, target_labels), dim=1)
        output = SecondPassSequenceOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        self._validate_output(
            output,
            batch_size=batch_size,
            total_length=total_length,
            dtype=dtype,
            device=device,
        )
        return output

    def _validate_prompt_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[int, int, torch.dtype, torch.device]:
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            message = "Prompt embeddings must have shape [batch, prompt, hidden_size]"
            raise SecondPassSequenceAssemblyError(message)
        batch_size, prompt_length, hidden_size = embeddings.shape
        if (
            batch_size <= 0
            or prompt_length <= 0
            or prompt_length > self.max_prompt_tokens
            or hidden_size != self.hidden_size
        ):
            message = "Prompt embedding shape exceeds the assembler contract"
            raise SecondPassSequenceAssemblyError(message)
        self._validate_frozen_embeddings(embeddings, name="Prompt")
        return batch_size, prompt_length, embeddings.dtype, embeddings.device

    def _validate_target_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> int:
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            message = "Target embeddings must have shape [batch, target, hidden_size]"
            raise SecondPassSequenceAssemblyError(message)
        expected_prefix = (batch_size,)
        target_length = embeddings.shape[1]
        if (
            embeddings.shape[0:1] != expected_prefix
            or target_length <= 0
            or target_length > self.max_target_tokens
            or embeddings.shape[2] != self.hidden_size
        ):
            message = "Target embedding shape exceeds the assembler contract"
            raise SecondPassSequenceAssemblyError(message)
        if embeddings.dtype != dtype or embeddings.device != device:
            message = "Prompt and target embeddings must share one dtype and device"
            raise SecondPassSequenceAssemblyError(message)
        self._validate_frozen_embeddings(embeddings, name="Target")
        return target_length

    @staticmethod
    def _validate_frozen_embeddings(embeddings: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(embeddings):
            message = f"{name} embeddings must use a floating dtype"
            raise SecondPassSequenceAssemblyError(message)
        try:
            finite = bool(torch.isfinite(embeddings).all())
        except Exception as error:
            message = f"{name} embeddings cannot be inspected on {embeddings.device}"
            raise SecondPassSequenceAssemblyError(message) from error
        if not finite:
            message = f"{name} embeddings contain NaN or Inf"
            raise SecondPassSequenceAssemblyError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = f"{name} token embeddings must come from the frozen backbone"
            raise SecondPassSequenceAssemblyError(message)

    def _validate_prompt_mask(
        self,
        mask: torch.Tensor,
        *,
        batch_size: int,
        prompt_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompt_mask = self._validate_binary_mask(
            mask,
            expected_shape=(batch_size, prompt_length),
            device=device,
            name="Prompt",
        )
        transitions = self._mask_transitions(prompt_mask)
        contiguous_left = (transitions >= 0).all(dim=1)
        contiguous_right = (transitions <= 0).all(dim=1)
        if not bool(torch.logical_or(contiguous_left, contiguous_right).all()):
            message = "Prompt attention must use contiguous left or right padding"
            raise SecondPassSequenceAssemblyError(message)
        return prompt_mask

    def _validate_target_mask(
        self,
        mask: torch.Tensor,
        *,
        batch_size: int,
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        target_mask = self._validate_binary_mask(
            mask,
            expected_shape=(batch_size, target_length),
            device=device,
            name="Target",
        )
        if not bool(target_mask[:, 0].all()) or not bool(
            (self._mask_transitions(target_mask) <= 0).all()
        ):
            message = "Target attention must use non-empty right padding"
            raise SecondPassSequenceAssemblyError(message)
        return target_mask

    @staticmethod
    def _validate_binary_mask(
        mask: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        integer_dtypes = {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected_shape:
            message = f"{name} attention mask has an invalid shape"
            raise SecondPassSequenceAssemblyError(message)
        if mask.dtype not in integer_dtypes or mask.device != device:
            message = f"{name} attention mask has an invalid dtype or device"
            raise SecondPassSequenceAssemblyError(message)
        if not bool(torch.logical_or(mask == 0, mask == 1).all()):
            message = f"{name} attention mask must contain only zeros and ones"
            raise SecondPassSequenceAssemblyError(message)
        if not bool(mask.any(dim=1).all()):
            message = f"Every {name.lower()} must contain an attended token"
            raise SecondPassSequenceAssemblyError(message)
        return mask.detach().clone().to(dtype=torch.bool)

    @staticmethod
    def _mask_transitions(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, 1:].to(dtype=torch.int8) - mask[:, :-1].to(dtype=torch.int8)

    def _validate_workspace(
        self,
        workspace: LatentBridgeOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        prompt_logical_lengths: torch.Tensor,
    ) -> None:
        if not isinstance(workspace, LatentBridgeOutput):
            message = "Workspace prefix must be LatentBridgeOutput"
            raise SecondPassSequenceAssemblyError(message)
        embeddings = workspace.workspace_embeddings
        expected_shape = (batch_size, self.num_slots, self.hidden_size)
        if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != expected_shape:
            message = "Workspace embeddings have an invalid shape"
            raise SecondPassSequenceAssemblyError(message)
        if not torch.is_floating_point(embeddings) or embeddings.dtype != dtype:
            message = "Workspace embeddings must share the token-embedding dtype"
            raise SecondPassSequenceAssemblyError(message)
        if embeddings.device != device or not bool(torch.isfinite(embeddings).all()):
            message = "Workspace embeddings must be finite on the token-embedding device"
            raise SecondPassSequenceAssemblyError(message)
        expected_matrix = (batch_size, self.num_slots)
        workspace_mask = workspace.workspace_attention_mask
        if (
            not isinstance(workspace_mask, torch.Tensor)
            or tuple(workspace_mask.shape) != expected_matrix
            or workspace_mask.dtype != torch.bool
            or workspace_mask.device != device
            or not bool(workspace_mask.all())
        ):
            message = "Every fixed workspace token must be attended"
            raise SecondPassSequenceAssemblyError(message)
        positions = workspace.workspace_position_ids
        expected_positions = prompt_logical_lengths.unsqueeze(1) + torch.arange(
            self.num_slots,
            dtype=torch.int64,
            device=device,
        )
        if (
            not isinstance(positions, torch.Tensor)
            or tuple(positions.shape) != expected_matrix
            or positions.dtype != torch.int64
            or positions.device != device
            or not torch.equal(positions, expected_positions)
        ):
            message = "Workspace position IDs must follow the logical prompt"
            raise SecondPassSequenceAssemblyError(message)

    def _validate_target_ids(
        self,
        input_ids: torch.Tensor,
        *,
        batch_size: int,
        target_length: int,
        device: torch.device,
    ) -> None:
        if (
            not isinstance(input_ids, torch.Tensor)
            or tuple(input_ids.shape) != (batch_size, target_length)
            or input_ids.dtype != torch.int64
            or input_ids.device != device
        ):
            message = "Target input IDs must be int64 [batch, target] on the embedding device"
            raise SecondPassSequenceAssemblyError(message)
        if not bool(torch.logical_and(input_ids >= 0, input_ids < self.vocabulary_size).all()):
            message = "Target input IDs contain an out-of-vocabulary index"
            raise SecondPassSequenceAssemblyError(message)

    def _validate_output(
        self,
        output: SecondPassSequenceOutput,
        *,
        batch_size: int,
        total_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(output.inputs_embeds.shape) != (
            batch_size,
            total_length,
            self.hidden_size,
        ):
            message = "Assembled input embeddings have an invalid shape"
            raise SecondPassSequenceAssemblyError(message)
        if output.inputs_embeds.dtype != dtype or output.inputs_embeds.device != device:
            message = "Assembled embeddings changed dtype or device"
            raise SecondPassSequenceAssemblyError(message)
        if not bool(torch.isfinite(output.inputs_embeds).all()):
            message = "Assembled embeddings contain NaN or Inf"
            raise SecondPassSequenceAssemblyError(message)
        matrix_shape = (batch_size, total_length)
        for name, tensor, expected_dtype in (
            ("attention mask", output.attention_mask, torch.bool),
            ("position IDs", output.position_ids, torch.int64),
            ("labels", output.labels, torch.int64),
        ):
            if (
                tuple(tensor.shape) != matrix_shape
                or tensor.dtype != expected_dtype
                or tensor.device != device
            ):
                message = f"Assembled {name} has an invalid shape, dtype, or device"
                raise SecondPassSequenceAssemblyError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor
