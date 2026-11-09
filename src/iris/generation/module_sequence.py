"""Teacher-forced ``[prompt][module states][target]`` assembly for arm B."""

import torch
from torch import nn

from iris.cognitive import ModuleLatentBridgeOutput
from iris.generation.outputs import SecondPassSequenceOutput


class ModuleSecondPassSequenceAssemblyError(RuntimeError):
    """Raised when arm-B prompt, module, and target tensors cannot form a sequence."""


class ModuleSecondPassSequenceAssembler(nn.Module):
    """Assemble arm B without constructing or emulating a workspace segment."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_modules: int,
        vocabulary_size: int,
        max_prompt_tokens: int,
        max_target_tokens: int,
        max_context_tokens: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_modules": num_modules,
            "vocabulary_size": vocabulary_size,
            "max_prompt_tokens": max_prompt_tokens,
            "max_target_tokens": max_target_tokens,
            "max_context_tokens": max_context_tokens,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if max_prompt_tokens + num_modules + max_target_tokens > max_context_tokens:
            message = "Configured prompt, module, and target maxima exceed context"
            raise ValueError(message)

        self.hidden_size = hidden_size
        self.num_modules = num_modules
        self.vocabulary_size = vocabulary_size
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.max_context_tokens = max_context_tokens

    def forward(
        self,
        *,
        prompt_embeddings: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        modules: ModuleLatentBridgeOutput,
        target_input_ids: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> SecondPassSequenceOutput:
        """Return one exact teacher sequence with labels only on target tokens."""
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
        self._validate_modules(
            modules,
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
        total_length = prompt_length + self.num_modules + target_length
        if total_length > self.max_context_tokens:
            message = "Physical arm-B teacher sequence exceeds the context window"
            raise ModuleSecondPassSequenceAssemblyError(message)

        inputs_embeds = torch.cat(
            (
                self._materialize(prompt_embeddings),
                self._materialize(modules.module_embeddings),
                self._materialize(target_embeddings),
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                prompt_mask,
                modules.module_attention_mask.detach().clone(),
                target_mask,
            ),
            dim=1,
        )
        raw_positions = attention_mask.to(dtype=torch.int64).cumsum(dim=-1) - 1
        position_ids = torch.where(
            attention_mask,
            raw_positions,
            torch.zeros_like(raw_positions),
        )
        module_slice = position_ids[:, prompt_length : prompt_length + self.num_modules]
        if not torch.equal(module_slice, modules.module_position_ids):
            message = "Module positions disagree with the assembled logical sequence"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if not bool((position_ids < self.max_context_tokens).all()):
            message = "Logical arm-B teacher positions exceed the context window"
            raise ModuleSecondPassSequenceAssemblyError(message)

        ignored_prefix = torch.full(
            (batch_size, prompt_length + self.num_modules),
            -100,
            dtype=torch.int64,
            device=device,
        )
        target_labels = torch.where(
            target_mask,
            target_input_ids,
            torch.full_like(target_input_ids, -100),
        )
        output = SecondPassSequenceOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=torch.cat((ignored_prefix, target_labels), dim=1),
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
        if type(embeddings) is not torch.Tensor or embeddings.ndim != 3:
            message = "Prompt embeddings must have shape [batch, prompt, hidden_size]"
            raise ModuleSecondPassSequenceAssemblyError(message)
        batch_size, prompt_length, hidden_size = embeddings.shape
        if (
            batch_size <= 0
            or prompt_length <= 0
            or prompt_length > self.max_prompt_tokens
            or hidden_size != self.hidden_size
        ):
            message = "Prompt embedding shape exceeds the arm-B sequence contract"
            raise ModuleSecondPassSequenceAssemblyError(message)
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
        if type(embeddings) is not torch.Tensor or embeddings.ndim != 3:
            message = "Target embeddings must have shape [batch, target, hidden_size]"
            raise ModuleSecondPassSequenceAssemblyError(message)
        target_length = embeddings.shape[1]
        if (
            embeddings.shape[0] != batch_size
            or target_length <= 0
            or target_length > self.max_target_tokens
            or embeddings.shape[2] != self.hidden_size
        ):
            message = "Target embedding shape exceeds the arm-B sequence contract"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if embeddings.dtype != dtype or embeddings.device != device:
            message = "Prompt and target embeddings must share dtype and device"
            raise ModuleSecondPassSequenceAssemblyError(message)
        self._validate_frozen_embeddings(embeddings, name="Target")
        return target_length

    @staticmethod
    def _validate_frozen_embeddings(embeddings: torch.Tensor, *, name: str) -> None:
        if not torch.is_floating_point(embeddings):
            message = f"{name} embeddings must use a floating dtype"
            raise ModuleSecondPassSequenceAssemblyError(message)
        try:
            finite = bool(torch.isfinite(embeddings).all())
        except Exception as error:
            message = f"{name} embeddings cannot be inspected on {embeddings.device}"
            raise ModuleSecondPassSequenceAssemblyError(message) from error
        if not finite:
            message = f"{name} embeddings contain NaN or Inf"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if embeddings.requires_grad or embeddings.grad_fn is not None:
            message = f"{name} token embeddings must come from the frozen backbone"
            raise ModuleSecondPassSequenceAssemblyError(message)

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
            raise ModuleSecondPassSequenceAssemblyError(message)
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
            raise ModuleSecondPassSequenceAssemblyError(message)
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
        if type(mask) is not torch.Tensor or tuple(mask.shape) != expected_shape:
            message = f"{name} attention mask has an invalid shape"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if mask.dtype not in integer_dtypes or mask.device != device:
            message = f"{name} attention mask has an invalid dtype or device"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if not bool(torch.logical_or(mask == 0, mask == 1).all()):
            message = f"{name} attention mask must contain only zeros and ones"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if not bool(mask.any(dim=1).all()):
            message = f"Every {name.lower()} must contain an attended token"
            raise ModuleSecondPassSequenceAssemblyError(message)
        return mask.detach().clone().to(dtype=torch.bool)

    @staticmethod
    def _mask_transitions(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, 1:].to(dtype=torch.int8) - mask[:, :-1].to(dtype=torch.int8)

    def _validate_modules(
        self,
        modules: ModuleLatentBridgeOutput,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        prompt_logical_lengths: torch.Tensor,
    ) -> None:
        if type(modules) is not ModuleLatentBridgeOutput:
            message = "Module prefix must be exactly ModuleLatentBridgeOutput"
            raise ModuleSecondPassSequenceAssemblyError(message)
        expected_embeddings = (batch_size, self.num_modules, self.hidden_size)
        embeddings = modules.module_embeddings
        if type(embeddings) is not torch.Tensor or tuple(embeddings.shape) != expected_embeddings:
            message = "Module embeddings have an invalid shape"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if (
            not torch.is_floating_point(embeddings)
            or embeddings.dtype != dtype
            or embeddings.device != device
            or not bool(torch.isfinite(embeddings).all())
        ):
            message = "Module embeddings have an invalid dtype, device, or value"
            raise ModuleSecondPassSequenceAssemblyError(message)
        expected_matrix = (batch_size, self.num_modules)
        module_mask = modules.module_attention_mask
        if (
            type(module_mask) is not torch.Tensor
            or tuple(module_mask.shape) != expected_matrix
            or module_mask.dtype != torch.bool
            or module_mask.device != device
            or not bool(module_mask.all())
        ):
            message = "Every fixed module token must be attended"
            raise ModuleSecondPassSequenceAssemblyError(message)
        expected_positions = prompt_logical_lengths.unsqueeze(1) + torch.arange(
            self.num_modules,
            dtype=torch.int64,
            device=device,
        )
        positions = modules.module_position_ids
        if (
            type(positions) is not torch.Tensor
            or tuple(positions.shape) != expected_matrix
            or positions.dtype != torch.int64
            or positions.device != device
            or not torch.equal(positions, expected_positions)
        ):
            message = "Module position IDs must follow the logical prompt"
            raise ModuleSecondPassSequenceAssemblyError(message)

    def _validate_target_ids(
        self,
        input_ids: torch.Tensor,
        *,
        batch_size: int,
        target_length: int,
        device: torch.device,
    ) -> None:
        if (
            type(input_ids) is not torch.Tensor
            or tuple(input_ids.shape) != (batch_size, target_length)
            or input_ids.dtype != torch.int64
            or input_ids.device != device
        ):
            message = "Target input IDs must be int64 [batch, target] on embedding device"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if not bool(torch.logical_and(input_ids >= 0, input_ids < self.vocabulary_size).all()):
            message = "Target input IDs contain an out-of-vocabulary index"
            raise ModuleSecondPassSequenceAssemblyError(message)

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
            message = "Assembled arm-B embeddings have an invalid shape"
            raise ModuleSecondPassSequenceAssemblyError(message)
        if (
            output.inputs_embeds.dtype != dtype
            or output.inputs_embeds.device != device
            or not bool(torch.isfinite(output.inputs_embeds).all())
        ):
            message = "Assembled arm-B embeddings changed dtype, device, or value"
            raise ModuleSecondPassSequenceAssemblyError(message)
        matrix_shape = (batch_size, total_length)
        for name, tensor, expected_dtype in (
            ("attention mask", output.attention_mask, torch.bool),
            ("position IDs", output.position_ids, torch.int64),
            ("labels", output.labels, torch.int64),
        ):
            if (
                type(tensor) is not torch.Tensor
                or tuple(tensor.shape) != matrix_shape
                or tensor.dtype != expected_dtype
                or tensor.device != device
            ):
                message = f"Assembled arm-B {name} is invalid"
                raise ModuleSecondPassSequenceAssemblyError(message)

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_inference() and not torch.is_inference_mode_enabled():
            return tensor.detach().clone()
        return tensor


__all__ = [
    "ModuleSecondPassSequenceAssembler",
    "ModuleSecondPassSequenceAssemblyError",
]
