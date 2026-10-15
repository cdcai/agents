"""Local request estimates and execution deadlines, with no metadata network calls.

Limits supplied by a deployment owner take precedence over best-effort metadata.
The default input cutoff is a configurable guess, not a model capability claim.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .exceptions import ContextLengthExceededError, TerminalAgentError

logger = logging.getLogger(__name__)
DEFAULT_INPUT_CUTOFF = 272_000


def _positive(value: int | None, name: str, *, allow_zero: bool = False) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (0 if allow_zero else 1)
    ):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer"
        )


@dataclass(frozen=True)
class RequestBudget:
    """Independent input/output/context constraints and their provenance.

    ``output_reservation`` reserves visible output and ``reasoning_reservation``
    reserves additional reasoning tokens. Their sum must fit the output cap.
    A request's ``max_completion_tokens``/``max_tokens`` already includes
    reasoning; the larger of that value and these reservations is used.
    """

    model_name: str = ""
    model: str | None = None
    model_version: str | None = None
    tokenizer: str | None = "o200k_base"
    context_window: int | None = None
    max_input_tokens: int | None = DEFAULT_INPUT_CUTOFF
    max_output_tokens: int | None = None
    output_reservation: int = 16_384
    reasoning_reservation: int = 0
    safety_margin: int = 1_024
    provenance: tuple[str, ...] = ("configurable fallback input cutoff estimate",)
    token_counter: Callable[[str], int] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("context_window", "max_input_tokens", "max_output_tokens"):
            _positive(getattr(self, name), name)
        for name in ("output_reservation", "reasoning_reservation", "safety_margin"):
            _positive(getattr(self, name), name, allow_zero=True)
        if self.max_input_tokens is None and self.context_window is None:
            raise ValueError("Configure max_input_tokens or context_window")
        if (
            self.max_output_tokens is not None
            and self.reserved_output > self.max_output_tokens
        ):
            raise ValueError(
                "Output and reasoning reservations exceed max_output_tokens"
            )
        if self.input_limit <= 0:
            raise ValueError("Reservations and safety margin leave no input capacity")

    @property
    def reserved_output(self) -> int:
        return self.output_reservation + self.reasoning_reservation

    @property
    def input_limit(self) -> int:
        return self.input_limit_for(self.reserved_output)

    def input_limit_for(self, output_tokens: int) -> int:
        limits = []
        if self.max_input_tokens is not None:
            limits.append(self.max_input_tokens)
        if self.context_window is not None:
            limits.append(
                self.context_window - max(output_tokens, self.reserved_output)
            )
        return min(limits) - self.safety_margin


def resolve_request_budget(
    model_name: str,
    *,
    model: str | None = None,
    model_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    output_reservation: int = 16_384,
    reasoning_reservation: int = 0,
    safety_margin: int = 1_024,
) -> RequestBudget:
    """Resolve optional metadata plus explicit settings without guessing aliases.

    Metadata is an already fetched mapping; missing capability fields are normal.
    No management-plane privileges or network discovery are required. Metadata is
    used per call and never cached here. Only recognized numeric limit fields are
    accepted; TPM/quota/file-size metadata is intentionally ignored.
    """
    values: dict[str, Any] = {
        "model_name": model_name,
        "model": model,
        "model_version": model_version,
        "output_reservation": output_reservation,
        "reasoning_reservation": reasoning_reservation,
        "safety_margin": safety_margin,
    }
    sources = ["configurable fallback input cutoff estimate"]
    if model is not None:
        sources.append(f"configured underlying model: {model}")
    if metadata:
        capabilities = metadata.get("capabilities", {})
        for source in (capabilities, metadata):
            if not isinstance(source, Mapping):
                continue
            for name in ("context_window", "max_input_tokens", "max_output_tokens"):
                value = source.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    values[name] = value
            for name in ("tokenizer", "model_version"):
                if isinstance(source.get(name), str):
                    values[name] = source[name]
        sources.append("caller-supplied best-effort metadata (not cached)")
    if overrides:
        allowed = set(RequestBudget.__dataclass_fields__) - {"model_name", "provenance"}
        invalid = set(overrides) - allowed
        if invalid:
            raise ValueError(f"Unknown request budget settings: {sorted(invalid)}")
        values.update(overrides)
        sources.append("explicit deployment/application overrides")
    if "max_input_tokens" in values:
        sources = [source for source in sources if "fallback" not in source]
    values["provenance"] = tuple(sources)
    return RequestBudget(**values)


@lru_cache(maxsize=8)
def _encoding(name: str):
    try:
        import tiktoken  # type: ignore[import-not-found]

        return tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 - optional third-party tokenizer/cache failures
        # An unavailable package, encoding, or offline cache cannot disable the
        # preflight. Byte counts err toward smaller requests instead.
        logger.warning(
            "Tokenizer %s unavailable; using conservative UTF-8 byte estimates", name
        )
        return None


def _counter(budget: RequestBudget) -> tuple[Callable[[str], int], str]:
    if budget.token_counter is not None:
        return budget.token_counter, "configured token counter estimate"
    encoding = _encoding(budget.tokenizer) if budget.tokenizer else None
    if encoding is not None:
        return lambda value: len(
            encoding.encode(value, disallowed_special=())
        ), f"{budget.tokenizer} estimate"
    return lambda value: len(value.encode("utf-8")), "conservative UTF-8 byte estimate"


def _serialized(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class RequestEstimate:
    input_tokens: int
    input_limit: int
    output_tokens: int
    components: dict[str, int]
    estimator: str
    is_estimate: bool = True

    @property
    def exceeds_budget(self) -> bool:
        return self.input_tokens > self.input_limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "input_limit": self.input_limit,
            "output_tokens": self.output_tokens,
            "components": dict(self.components),
            "estimator": self.estimator,
            "is_estimate": True,
        }


class RequestBudgetExceededError(ContextLengthExceededError):
    """An estimated final request exceeds the configured local cutoff."""

    def __init__(
        self, budget: RequestBudget, estimate: RequestEstimate, reason: str = "input"
    ):
        self.budget = budget
        self.estimate = estimate
        self.reason = reason
        super().__init__(
            f"Estimated request {reason} exceeds budget: "
            f"{estimate.input_tokens} estimated input tokens, cutoff {estimate.input_limit}; "
            f"reserved output {estimate.output_tokens}",
            code="request_budget_exceeded",
            details={
                **estimate.as_dict(),
                "reason": reason,
                "provenance": list(budget.provenance),
            },
        )


def estimate_request(
    request: Mapping[str, Any], budget: RequestBudget
) -> RequestEstimate:
    """Estimate the complete rendered text request, including history and schemas.

    Chat framing is estimated separately. This is not a server token count;
    compare it with returned usage to assess error. Image/audio payloads require
    an application-supplied estimator and are rejected rather than undercounted.
    """
    count, estimator = _counter(budget)
    components = dict.fromkeys(
        ("system", "user", "history", "tools", "schema", "other", "framing"), 0
    )
    messages = request.get("messages", [])
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            part.get("type") not in ("text", "input_text")
            for part in content
            if isinstance(part, Mapping)
        ):
            raise ValueError(
                "Multimodal token estimation requires an application-specific budget strategy"
            )
        role = message.get("role")
        component = (
            "system"
            if role in ("system", "developer")
            else "user"
            if role == "user"
            else "history"
        )
        components[component] += count(_serialized(message))
        components["framing"] += 12
    for name, value in request.items():
        if name == "messages":
            continue
        component = (
            "tools"
            if name in ("tools", "functions")
            else "schema"
            if name == "response_format"
            else "other"
        )
        components[component] += count(_serialized({name: value}))
    components["framing"] += 8
    requested = request.get("max_completion_tokens", request.get("max_tokens"))
    if requested is not None:
        _positive(requested, "request output limit")
    output = max(budget.reserved_output, requested or 0)
    return RequestEstimate(
        input_tokens=sum(components.values()),
        input_limit=budget.input_limit_for(output),
        output_tokens=output,
        components=components,
        estimator=estimator,
    )


def check_request_budget(
    request: Mapping[str, Any], budget: RequestBudget
) -> RequestEstimate:
    estimate = estimate_request(request, budget)
    if (
        budget.max_output_tokens is not None
        and estimate.output_tokens > budget.max_output_tokens
    ):
        raise RequestBudgetExceededError(budget, estimate, "output")
    if estimate.exceeds_budget:
        raise RequestBudgetExceededError(budget, estimate)
    return estimate


class ExecutionDeadlineExceeded(TerminalAgentError):
    """Local elapsed deadline; remote work may still be running."""

    def __init__(self, seconds: float, *, scope: str = "agent"):
        self.seconds = seconds
        self.scope = scope
        super().__init__(
            f"Local {scope} elapsed-time budget of {seconds:g} seconds expired; remote work may continue",
            code="execution_deadline_exceeded",
            details={
                "scope": scope,
                "seconds": seconds,
                "remote_work_may_continue": True,
            },
        )


@dataclass(frozen=True)
class ExecutionDeadline:
    """Monotonic deadline with injectable clock for deterministic checks."""

    seconds: float
    clock: Callable[[], float] = field(
        default=time.monotonic, repr=False, compare=False
    )
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.seconds, bool)
            or not math.isfinite(self.seconds)
            or self.seconds <= 0
        ):
            raise ValueError("Elapsed-time budget must be finite and positive")
        object.__setattr__(self, "started_at", self.clock())

    @property
    def remaining(self) -> float:
        return max(0.0, self.started_at + self.seconds - self.clock())

    def check(self) -> None:
        if self.remaining <= 0:
            raise ExecutionDeadlineExceeded(self.seconds)
