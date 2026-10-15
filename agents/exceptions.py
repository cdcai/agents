"""Structured failures shared by providers, applications, and processors."""

from typing import Any


class AgentError(RuntimeError):
    """An execution failure with an explicit retry classification.

    Provider failures are operation scoped: a processor must not restart the
    entire agent after the provider has exhausted its own retry budget.
    """

    retryable = False
    operation_scoped = True
    remote_outcome_unknown = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: Any = None,
        operation_scoped: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details
        self.operation_scoped = operation_scoped


class TerminalAgentError(AgentError):
    """The unchanged operation cannot succeed by retrying."""


class RetryableAgentError(AgentError):
    """A transient failure, retried only within the owning retry budget."""

    retryable = True


class BatchRequestError(AgentError):
    """A single request failure, preserving the original provider payload."""

    def __init__(
        self,
        message: str,
        *,
        custom_id: str | None = None,
        batch_id: str | None = None,
        status_code: int | None = None,
        code: str | None = None,
        details: Any = None,
        remote_outcome_unknown: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.custom_id = custom_id
        self.batch_id = batch_id
        self.status_code = status_code
        self.remote_outcome_unknown = remote_outcome_unknown


class TerminalBatchRequestError(BatchRequestError, TerminalAgentError):
    pass


class RetryableBatchRequestError(BatchRequestError, RetryableAgentError):
    retryable = True


class ContextLengthExceededError(TerminalBatchRequestError):
    """The application may explicitly rebuild a smaller request context."""


class BatchProtocolError(TerminalBatchRequestError):
    """A missing, malformed, or ambiguous result cannot be safely replayed."""


class BatchPollingDeadlineExceeded(TerminalAgentError):
    """Local tracking stopped; remote completion/cancellation is uncertain."""

    def __init__(self, batch: Any, seconds: float) -> None:
        super().__init__(
            f"Batch [{batch.id}] exceeded its {seconds:g}s local polling deadline; "
            "the remote request may still complete.",
            code="batch_polling_deadline_exceeded",
            details={
                "batch_id": batch.id,
                "seconds": seconds,
                "remote_outcome_unknown": True,
            },
        )
        self.batch = batch
        self.batch_id = batch.id
        self.remote_outcome_unknown = True
