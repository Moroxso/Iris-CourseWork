"""Private one-use request ownership kernel, not a public loader/RNG transaction."""

from dataclasses import dataclass, fields
from re import fullmatch
from typing import NoReturn

import torch

from iris.ablations.cognitive import CognitiveAblationSessionV1
from iris.ablations.policy import CognitiveAblationPolicyV1, cognitive_ablation_policy_sha256_v1
from iris.application.errors import IrisRequestCheckpointError, IrisRequestError
from iris.application.request import _ScientificAblationOwnedRequest
from iris.application.request_sidecar_state import _target_state
from iris.generation import DecodingStrategy, TemperatureTopPSamplingConfig
from iris.utils.hashing import domain_sha256
from iris.workspace.trace import WorkspaceTraceOptionsV1


@dataclass(frozen=True, slots=True)
class _AblationRequestBinding:
    artifact_sha256: str
    source_checkpoint_sha256: str
    lifecycle_sha256: str
    policy_sha256: str
    sidecar_sha256: str


def _binding_values(binding: _AblationRequestBinding) -> tuple[str, ...]:
    if type(binding) is not _AblationRequestBinding:
        raise IrisRequestError("Exact owned artifact binding required")
    values = tuple(getattr(binding, field.name) for field in fields(binding))
    if any(type(value) is not str or fullmatch(r"[0-9a-f]{64}", value) is None for value in values):
        raise IrisRequestError("Canonical owned artifact/policy/sidecar bindings required")
    return values


def _graph_references(request: _ScientificAblationOwnedRequest) -> tuple[object, ...]:
    return (
        request._backbone,
        request._iris_generator,
        request._cognitive_config,
        request._workspace_config,
        request._sidecar_modules,
        request._iris_generator.sampling,
        request._iris_generator.stop_token_ids,
        *(getattr(request, "_" + name) for name, _ in request._sidecar_modules),
    )


def _decoding_values(request: _ScientificAblationOwnedRequest) -> tuple[float, float, int]:
    sampling = request._iris_generator.sampling
    if type(sampling) is not TemperatureTopPSamplingConfig:
        raise IrisRequestError("Exact owned sampling policy required")
    return sampling.temperature, sampling.top_p, request._iris_generator.max_new_tokens


def _tensor_signature(request: _ScientificAblationOwnedRequest) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(value),
            value._version,
            value.untyped_storage().data_ptr(),
            tuple(value.shape),
            value.dtype,
            value.device,
        )
        for name, value in _target_state(request._sidecar_modules).items()
    )


def _session_binding_values(session: CognitiveAblationSessionV1) -> tuple[object, ...]:
    """Stable per-invocation facts, rechecked BEFORE serialization at completion."""
    try:
        if type(session) is not CognitiveAblationSessionV1 or set(vars(session)) != {
            "_policy",
            "policy_sha256",
            "prompt_sha256",
            "modules",
            "slots",
            "width",
            "cycles",
            "_phase",
            "_failed",
            "_finished",
            "_length",
        }:
            raise IrisRequestError("Completed owned session fields changed")
        policy = session._policy
        if (
            type(policy) is not CognitiveAblationPolicyV1
            or set(vars(policy)) != set(CognitiveAblationPolicyV1.model_fields)
            or policy.__pydantic_extra__ is not None
            or type(session.policy_sha256) is not str
            or type(session.prompt_sha256) is not str
            or any(
                type(value) is not int
                for value in (session.modules, session.slots, session.width, session.cycles)
            )
        ):
            raise IrisRequestError("Completed owned policy/session structure changed")
        return (
            id(policy),
            cognitive_ablation_policy_sha256_v1(policy),
            session.policy_sha256,
            session.prompt_sha256,
            session.modules,
            session.slots,
            session.width,
            session.cycles,
        )
    except Exception as error:
        if isinstance(error, IrisRequestError):
            raise
        raise IrisRequestError(
            "Completed owned policy/session binding could not be checked"
        ) from error


@dataclass(frozen=True, slots=True, eq=False)
class _BootstrapCapability:
    owner: "_AblationRequestOwner"
    request: _ScientificAblationOwnedRequest
    wrapper: object
    binding: _AblationRequestBinding
    consumed: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class _ExecuteCapability:
    owner: "_AblationRequestOwner"
    request: _ScientificAblationOwnedRequest
    wrapper: object
    binding: _AblationRequestBinding
    session: CognitiveAblationSessionV1
    prompt: str
    strategy: DecodingStrategy
    generator: torch.Generator | None
    diagnostics: WorkspaceTraceOptionsV1 | None
    session_binding: tuple[object, ...]
    consumed: bool = False


class _AblationRequestOwner:
    """Concrete exclusive anchor; public facade must finish its outer transaction.

    No registry or ambient authorization. A reserved capability is not entered
    until its exact operation consumes it; successful core remains busy until
    explicit postcheck completion. Methods here are private implementation seams.
    """

    __slots__ = (
        "_binding",
        "_binding_values",
        "_decoding",
        "_entered",
        "_graph",
        "_initial_sha256",
        "_pending",
        "_request",
        "_state",
        "_tensors",
        "_wrapper",
        "_wrapper_seal",
    )

    def __init__(
        self,
        request: _ScientificAblationOwnedRequest,
        *,
        wrapper: object,
        binding: _AblationRequestBinding,
    ) -> None:
        if type(request) is not _ScientificAblationOwnedRequest or wrapper is None:
            raise IrisRequestError("Fresh exact private request and explicit wrapper required")
        if hasattr(request, "_ablation_anchor") or hasattr(request, "_ablation_binding"):
            raise IrisRequestError("An owned request cannot be reattached")
        values = _binding_values(binding)
        request._validate_runtime_boundary()
        self._initial_sha256 = request.sidecar_state_sha256
        self._request, self._wrapper, self._binding = request, wrapper, binding
        self._wrapper_seal: object | None = None
        self._binding_values = values
        self._graph = _graph_references(request)
        self._decoding = _decoding_values(request)
        self._tensors = _tensor_signature(request)
        self._state = "new"
        self._pending: _BootstrapCapability | _ExecuteCapability | None = None
        self._entered = False
        request._ablation_anchor = self
        request._ablation_binding = binding

    def _poison(self) -> None:
        self._state = "poisoned"
        self._pending = None
        self._entered = False

    def _live(self, wrapper: object) -> None:
        try:
            request = self._request
            if (
                type(self) is not _AblationRequestOwner
                or wrapper is not self._wrapper
                or type(request) is not _ScientificAblationOwnedRequest
                or request._ablation_anchor is not self
                or request._ablation_binding is not self._binding
                or _binding_values(self._binding) != self._binding_values
                or self._state not in {"new", "ready", "bootstrapping", "executing"}
                or type(self._entered) is not bool
            ):
                raise IrisRequestError("Owned request anchor/binding/state changed")
            if self._state in {"new", "ready"}:
                if self._pending is not None or self._entered:
                    raise IrisRequestError("Idle owner has inconsistent delegation state")
            elif type(self._pending) is not (
                _BootstrapCapability if self._state == "bootstrapping" else _ExecuteCapability
            ):
                raise IrisRequestError("Busy owner lacks its exact operation capability")
            actual = _graph_references(request)
            if (
                len(actual) != len(self._graph)
                or any(
                    item is not expected for item, expected in zip(actual, self._graph, strict=True)
                )
                or _decoding_values(request) != self._decoding
            ):
                raise IrisRequestError("Owned request graph or decoding policy changed")
            request._validate_runtime_boundary()
        except BaseException as error:
            self._poison()
            if isinstance(error, Exception) and not isinstance(error, IrisRequestError):
                raise IrisRequestError("Failed to verify owned anchor/graph boundary") from error
            raise

    def _check_sidecar(self, expected: str) -> None:
        try:
            if self._request.sidecar_state_sha256 != expected or (
                _tensor_signature(self._request) != self._tensors
            ):
                raise IrisRequestError("Owned sidecar value/version/storage changed")
        except BaseException:
            self._poison()
            raise

    def _reserve_bootstrap(self, wrapper: object) -> _BootstrapCapability:
        self._live(wrapper)
        if self._state != "new":
            raise IrisRequestError("Bootstrap is one-use and only valid for a new owner")
        self._check_sidecar(self._initial_sha256)
        cap = _BootstrapCapability(self, self._request, wrapper, self._binding)
        self._state, self._pending = "bootstrapping", cap
        return cap

    def _reserve_execution(
        self,
        wrapper: object,
        *,
        prompt: str,
        session: CognitiveAblationSessionV1,
        strategy: DecodingStrategy,
        generator: torch.Generator | None = None,
        diagnostics: WorkspaceTraceOptionsV1 | None = None,
    ) -> _ExecuteCapability:
        self._live(wrapper)
        if self._state != "ready":
            raise IrisRequestError("Owned request is not ready or is already busy")
        self._request._validate_request(prompt, strategy=strategy, generator=generator)
        _validate_session(self._request, session, prompt, self._binding.policy_sha256)
        self._check_sidecar(self._binding.sidecar_sha256)
        cap = _ExecuteCapability(
            self,
            self._request,
            wrapper,
            self._binding,
            session,
            prompt,
            strategy,
            generator,
            diagnostics,
            _session_binding_values(session),
        )
        self._state, self._pending = "executing", cap
        return cap

    def _complete_bootstrap(self, cap: object) -> None:
        self._live(self._wrapper)
        _require_cap(self, cap, _BootstrapCapability, consumed=True)
        if self._request.sidecar_state_sha256 != self._binding.sidecar_sha256:
            raise IrisRequestError("Bootstrap did not install the bound artifact")
        self._tensors = _tensor_signature(self._request)
        self._state, self._pending, self._entered = "ready", None, False

    def _verify_execution(self, wrapper: object, cap: object) -> None:
        """Verify completed core while retaining the exclusive outer transaction."""
        try:
            self._live(wrapper)
            _require_cap(self, cap, _ExecuteCapability, consumed=True)
            assert type(cap) is _ExecuteCapability
            if _session_binding_values(cap.session) != cap.session_binding:
                raise IrisRequestError("Owned session policy/prompt/geometry binding changed")
            if (
                cap.session._finished is not True
                or cap.session._failed is not False
                or type(cap.session._phase) is not int
                or cap.session._phase != self._request._cycles + 3
            ):
                raise IrisRequestError("Owned cognitive session did not finish successfully")
            self._check_sidecar(self._binding.sidecar_sha256)
        except BaseException:
            self._poison()
            raise

    def _complete_execution(self, wrapper: object, cap: object) -> None:
        self._verify_execution(wrapper, cap)
        self._state, self._pending, self._entered = "ready", None, False


def _validate_session(
    request: _ScientificAblationOwnedRequest, session: object, prompt: object, policy_sha256: str
) -> None:
    if type(prompt) is not str or type(session) is not CognitiveAblationSessionV1:
        raise IrisRequestError("Owned core requires its exact fresh cognitive session/prompt")
    session._validate_start(
        prompt_sha256=domain_sha256("iris/cognitive-ablation-prompt/v1", prompt.encode("utf-8")),
        num_modules=request._num_modules,
        num_slots=request._num_slots,
        state_dim=request._state_dim,
        cycles=request._cycles,
    )
    if session.policy_sha256 != policy_sha256:
        raise IrisRequestError("Owned cognitive session policy differs from bound artifact")


def _anchor(request: _ScientificAblationOwnedRequest) -> _AblationRequestOwner:
    try:
        owner = request._ablation_anchor
    except AttributeError as error:
        raise IrisRequestError("Owned request is missing its anchor") from error
    if type(owner) is not _AblationRequestOwner or owner._request is not request:
        raise IrisRequestError("Owned request anchor identity differs")
    owner._live(owner._wrapper)
    return owner


def _require_cap(
    owner: _AblationRequestOwner,
    cap: object,
    kind: type[_BootstrapCapability] | type[_ExecuteCapability],
    *,
    consumed: bool,
) -> None:
    if (
        type(cap) is not kind
        or not isinstance(cap, (_BootstrapCapability, _ExecuteCapability))
        or owner._pending is not cap
        or cap.owner is not owner
        or cap.request is not owner._request
        or cap.wrapper is not owner._wrapper
        or cap.binding is not owner._binding
        or cap.consumed is not consumed
        or owner._entered is not consumed
    ):
        raise IrisRequestError("Exact one-use operation/owner/request/binding capability required")


def _consume_bootstrap(
    request: _ScientificAblationOwnedRequest, cap: object
) -> _AblationRequestOwner:
    owner = _anchor(request)
    _require_cap(owner, cap, _BootstrapCapability, consumed=False)
    if owner._state != "bootstrapping":
        raise IrisRequestError("Wrong bootstrap operation state")
    owner._check_sidecar(owner._initial_sha256)
    assert type(cap) is _BootstrapCapability
    object.__setattr__(cap, "consumed", True)
    owner._entered = True
    return owner


def _consume_execution(
    request: _ScientificAblationOwnedRequest,
    cap: object,
    *,
    prompt: object,
    strategy: object,
    generator: object,
    session: object,
    intervention: object,
    diagnostics: object,
) -> _AblationRequestOwner:
    owner = _anchor(request)
    _require_cap(owner, cap, _ExecuteCapability, consumed=False)
    assert type(cap) is _ExecuteCapability
    if (
        owner._state != "executing"
        or type(prompt) is not str
        or prompt != cap.prompt
        or type(strategy) is not str
        or strategy != cap.strategy
        or generator is not cap.generator
        or session is not cap.session
        or diagnostics is not cap.diagnostics
        or intervention is not None
    ):
        raise IrisRequestError("Owned capability invocation or operation differs")
    _validate_session(request, session, prompt, owner._binding.policy_sha256)
    owner._check_sidecar(owner._binding.sidecar_sha256)
    object.__setattr__(cap, "consumed", True)
    owner._entered = True
    return owner


def _raise_bootstrap_failure(error: BaseException) -> NoReturn:
    # The historical copy kernel wraps an uncertain rollback in a typed error.
    # Preserve its actual first SIGNAL object while retaining secondary failures.
    cause = error.__cause__
    if (
        isinstance(error, IrisRequestCheckpointError)
        and type(cause) is BaseExceptionGroup
        and cause.message == "Request sidecar installation and rollback both failed"
        and len(cause.exceptions) == 2
        and not isinstance(cause.exceptions[0], Exception)
    ):
        primary = cause.exceptions[0]
        primary.add_note("Owned bootstrap was poisoned; secondary rollback failure is chained")
        raise primary from error
    raise error
