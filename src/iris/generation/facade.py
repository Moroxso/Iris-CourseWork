"""Thin text-decoding facade over validated Iris cached decoders."""

import torch

from iris.backbone.adapter import BackboneAdapter
from iris.generation.greedy import GreedyIrisDecoder
from iris.generation.outputs import (
    DecodingStrategy,
    IrisGenerationOutput,
    IrisGeneratorOutput,
    SecondPassGenerationPrefixOutput,
)
from iris.generation.sampling import (
    SamplingIrisDecoder,
    TemperatureTopPSamplingConfig,
    _clone_owned_generator,
)
from iris.workspace.state import WorkspaceState


class IrisGeneratorError(RuntimeError):
    """Raised when token generation cannot be finalized as validated text."""


class IrisGenerator:
    """Dispatch one proven decoder and preserve its exact trace provenance."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        *,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...],
        sampling: TemperatureTopPSamplingConfig,
    ) -> None:
        self._backbone = backbone
        self._greedy_decoder = GreedyIrisDecoder(
            backbone,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        self._sampling_decoder = SamplingIrisDecoder(
            backbone,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
        )
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids
        self.sampling = sampling
        self._stop_token_set = frozenset(stop_token_ids)

    @property
    def backbone(self) -> BackboneAdapter:
        """Return the exact frozen adapter owned by both decoding strategies."""
        return self._backbone

    @torch.inference_mode()
    def generate(
        self,
        prefix: SecondPassGenerationPrefixOutput,
        *,
        strategy: DecodingStrategy,
        workspace_trace: tuple[WorkspaceState, ...],
        generator: torch.Generator | None = None,
    ) -> IrisGeneratorOutput:
        """Generate and decode text; sampled RNG commits only after full success."""
        if type(strategy) is not str or strategy not in {"greedy", "sampling"}:
            message = "strategy must be either 'greedy' or 'sampling'"
            raise IrisGeneratorError(message)
        if strategy == "greedy":
            if generator is not None:
                message = "Greedy generation does not accept a random generator"
                raise IrisGeneratorError(message)
            try:
                raw_output = self._greedy_decoder.generate(
                    prefix,
                    workspace_trace=workspace_trace,
                )
            except Exception as error:
                message = "Greedy Iris token generation failed"
                raise IrisGeneratorError(message) from error
            try:
                return self._finalize(raw_output, workspace_trace=workspace_trace)
            except Exception as error:
                message = "Failed to finalize greedy Iris text"
                raise IrisGeneratorError(message) from error

        if not isinstance(generator, torch.Generator):
            message = "Sampling generation requires an explicit torch.Generator"
            raise IrisGeneratorError(message)
        expected_device = self._prefix_device(prefix)
        try:
            local_generator = _clone_owned_generator(
                generator,
                expected_device=expected_device,
            )
        except Exception as error:
            message = "Failed to start the facade sampling transaction"
            raise IrisGeneratorError(message) from error
        try:
            raw_output = self._sampling_decoder.generate(
                prefix,
                workspace_trace=workspace_trace,
                generator=local_generator,
            )
        except Exception as error:
            message = "Sampled Iris token generation failed"
            raise IrisGeneratorError(message) from error
        try:
            output = self._finalize(raw_output, workspace_trace=workspace_trace)
        except Exception as error:
            message = "Failed to finalize sampled Iris text"
            raise IrisGeneratorError(message) from error
        try:
            generator.set_state(local_generator.get_state().detach().clone())
        except Exception as error:
            message = "Failed to commit the facade sampling transaction"
            raise IrisGeneratorError(message) from error
        return output

    @staticmethod
    def _prefix_device(prefix: SecondPassGenerationPrefixOutput) -> torch.device:
        if not isinstance(prefix, SecondPassGenerationPrefixOutput) or not isinstance(
            prefix.inputs_embeds,
            torch.Tensor,
        ):
            message = "IrisGenerator requires a validated generation prefix"
            raise IrisGeneratorError(message)
        return prefix.inputs_embeds.device

    def _finalize(
        self,
        raw_output: IrisGenerationOutput,
        *,
        workspace_trace: tuple[WorkspaceState, ...],
    ) -> IrisGeneratorOutput:
        if not isinstance(raw_output, IrisGenerationOutput):
            message = "Decoder returned an invalid IrisGenerationOutput"
            raise IrisGeneratorError(message)
        if raw_output.workspace_trace is not workspace_trace:
            message = "Decoder did not preserve the exact workspace trace"
            raise IrisGeneratorError(message)
        token_ids = raw_output.token_ids
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.ndim != 2
            or token_ids.shape[0] != 1
            or token_ids.shape[1] <= 0
            or token_ids.shape[1] > self.max_new_tokens
            or token_ids.dtype != torch.int64
        ):
            message = "Decoder returned invalid generated token IDs"
            raise IrisGeneratorError(message)
        try:
            generated_token_ids = tuple(
                int(token_id) for token_id in token_ids[0].detach().to(device="cpu").tolist()
            )
        except Exception as error:
            message = "Failed to inspect the generated token sequence"
            raise IrisGeneratorError(message) from error
        last_token_id = generated_token_ids[-1]
        earlier_stop_token = any(
            token_id in self._stop_token_set for token_id in generated_token_ids[:-1]
        )
        if raw_output.stop_reason == "stop_token":
            matched = raw_output.matched_stop_token_id
            if type(matched) is not int or matched not in self._stop_token_set:
                message = "Decoder returned invalid stop-token metadata"
                raise IrisGeneratorError(message)
            if last_token_id != matched:
                message = "Matched stop token is not the final generated token"
                raise IrisGeneratorError(message)
            if earlier_stop_token:
                message = "Decoder continued generation after an earlier stop token"
                raise IrisGeneratorError(message)
        elif raw_output.stop_reason == "max_new_tokens":
            if raw_output.matched_stop_token_id is not None:
                message = "Maximum-length output cannot report a matched stop token"
                raise IrisGeneratorError(message)
            if token_ids.shape[1] != self.max_new_tokens:
                message = "Maximum-length output has not reached max_new_tokens"
                raise IrisGeneratorError(message)
            if any(token_id in self._stop_token_set for token_id in generated_token_ids):
                message = "Maximum-length output contains an unreported stop token"
                raise IrisGeneratorError(message)
        else:
            message = "Decoder returned an unknown stop reason"
            raise IrisGeneratorError(message)

        text = self._backbone.decode_tokens(token_ids)
        return IrisGeneratorOutput(
            text=text,
            token_ids=token_ids,
            stop_reason=raw_output.stop_reason,
            matched_stop_token_id=raw_output.matched_stop_token_id,
            workspace_trace=workspace_trace,
        )
