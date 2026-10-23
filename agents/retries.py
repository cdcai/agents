"""Bounded retries at the failed operation, never around callbacks or agents."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic

import openai

from .exceptions import (
    AgentError,
    ContextLengthExceededError,
    RetryableAgentError,
    TerminalAgentError,
)


@dataclass(frozen=True)
class RetryPolicy:
    """Attempts include the initial operation; elapsed time includes backoff."""

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    elapsed_time_budget: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
            or not math.isfinite(self.initial_delay)
            or not math.isfinite(self.max_delay)
            or self.initial_delay < 0
            or self.max_delay < 0
        ):
            raise ValueError("Retry attempts must be positive and delays nonnegative")
        if self.elapsed_time_budget is not None and (
            not math.isfinite(self.elapsed_time_budget) or self.elapsed_time_budget <= 0
        ):
            raise ValueError("Retry elapsed_time_budget must be positive")


def normalize_exception(exc: Exception) -> Exception:
    """Normalize SDK errors without treating ordinary application errors as transient."""
    if isinstance(exc, AgentError):
        return exc
    if isinstance(exc, openai.APIError):
        body = getattr(exc, "body", None)
        code = getattr(exc, "code", None)
        if isinstance(body, dict):
            body = body.get("error", body)
            if isinstance(body, dict):
                code = body.get("code") or code
        if code == "context_length_exceeded":
            return ContextLengthExceededError(str(exc), code=code, details=body)
        status = getattr(exc, "status_code", None)
        transient = isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))
        transient = (
            transient
            or status in (408, 409, 429)
            or (status is not None and status >= 500)
        )
        error_class = RetryableAgentError if transient else TerminalAgentError
        error = error_class(str(exc), code=code, details=body)
        error.remote_outcome_unknown = isinstance(exc, openai.APIConnectionError)
        return error
    return exc


def is_retryable_exception(exc: Exception, *, agent_scope: bool = False) -> bool:
    error = normalize_exception(exc)
    return isinstance(error, RetryableAgentError) and (
        not agent_scope or not error.operation_scoped
    )


async def retry_operation[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retry: bool = True,
) -> T:
    """Repeat only known transient failures; cancellation always propagates.

    Elapsed budgets bound the operation as well as delays. A local timeout says
    nothing about whether a previously submitted remote request completed.
    """
    started = monotonic()
    for attempt in range(policy.max_attempts):
        try:
            if policy.elapsed_time_budget is None:
                return await operation()
            remaining = policy.elapsed_time_budget - (monotonic() - started)
            if remaining <= 0:
                raise TerminalAgentError(
                    "Operation retry elapsed-time budget exhausted",
                    code="retry_deadline_exceeded",
                )
            timeout = asyncio.timeout(remaining)
            try:
                async with timeout:
                    return await operation()
            except TimeoutError as exc:
                if not timeout.expired():
                    raise
                deadline_error = TerminalAgentError(
                    "Operation retry elapsed-time budget exhausted; remote outcome may be unknown",
                    code="retry_deadline_exceeded",
                    details={"remote_outcome_unknown": True},
                )
                raise deadline_error from exc
        except Exception as exc:
            error = normalize_exception(exc)
            if (
                not retry
                or not is_retryable_exception(error)
                or attempt + 1 >= policy.max_attempts
            ):
                if error is exc:
                    raise
                raise error from exc
            delay = min(policy.max_delay, policy.initial_delay * (2**attempt))
            if policy.elapsed_time_budget is not None:
                remaining = policy.elapsed_time_budget - (monotonic() - started)
                if delay >= remaining:
                    if error is exc:
                        raise
                    raise error from exc
            await asyncio.sleep(delay)
    raise AssertionError("Retry loop exited without a result")
