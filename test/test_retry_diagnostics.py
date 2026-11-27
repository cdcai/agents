"""Keep transport identity and ambiguity available after SDK normalization."""

from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from agents.exceptions import RetryableAgentError, TerminalAgentError
from agents.retries import RetryPolicy, normalize_exception, retry_operation
from test.test_batch_scheduler import provider
from test.test_openai_batch_helper import make_batch_file


@pytest.mark.parametrize(
    "status,unknown", [(400, False), (408, True), (429, False), (504, True)]
)
def test_normalized_status_retains_request_id_and_remote_ambiguity(status, unknown):
    error = openai.APIStatusError(
        "synthetic status",
        response=httpx.Response(
            status,
            headers={"x-request-id": "request-123"},
            request=httpx.Request("POST", "https://example.test"),
        ),
        body={"code": "synthetic"},
    )
    normalized = normalize_exception(error)
    assert normalized.status_code == status
    assert normalized.request_id == "request-123"
    assert normalized.remote_outcome_unknown is unknown
    assert normalized.details == {"code": "synthetic"}


@pytest.mark.asyncio
async def test_ambiguous_creation_preserves_uploaded_file_without_resubmitting():
    p = provider()
    p.llm.batches.create.side_effect = openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.test")
    )
    with pytest.raises(RetryableAgentError) as failure:
        await p.create_batch_task(make_batch_file())
    assert failure.value.input_file_id == "file_123"
    assert failure.value.details["operation"] == "batches.create"
    assert failure.value.remote_outcome_unknown
    p.llm.batches.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_operation_deadline_exposes_unknown_remote_outcome():
    import asyncio

    async def blocked():
        await asyncio.Future()

    operation = AsyncMock(side_effect=blocked)
    with pytest.raises(TerminalAgentError) as failure:
        await retry_operation(operation, RetryPolicy(elapsed_time_budget=0.01))
    assert failure.value.code == "retry_deadline_exceeded"
    assert failure.value.remote_outcome_unknown
    operation.assert_awaited_once()


@pytest.mark.asyncio
async def test_prompt_creation_timeout_never_requeues_batch():
    import asyncio
    from types import SimpleNamespace

    p = provider()
    p.model_name = "synthetic"
    p.request_budget = None
    p.batch_idx = 1
    p.batch_idx_lock = asyncio.Lock()
    p.endpoint_fn = p.query_batch_mode
    p.llm.batches.create.side_effect = openai.APITimeoutError(
        request=httpx.Request("POST", "https://example.test")
    )
    with pytest.raises(RetryableAgentError):
        await p.prompt_agent(
            SimpleNamespace(record_request=lambda request: None),
            [{"role": "user", "content": "synthetic"}],
        )
    p.llm.files.create.assert_awaited_once()
    p.llm.batches.create.assert_awaited_once()
    assert p.batch_idx == 2


@pytest.mark.asyncio
async def test_batch_context_rebuild_does_not_retry_rebuilt_queue_call():
    from types import SimpleNamespace

    from agents.exceptions import ContextLengthExceededError

    p = provider()
    p.model_name = "synthetic"
    p.request_budget = None
    p.endpoint_fn = AsyncMock(
        side_effect=[
            ContextLengthExceededError("context"),
            RetryableAgentError("tracking lost"),
        ]
    )
    p._rebuild_context = AsyncMock(
        return_value=[{"role": "user", "content": "rebuilt"}]
    )
    with pytest.raises(RetryableAgentError):
        await p.prompt_agent(
            SimpleNamespace(record_request=lambda request: None),
            [{"role": "user", "content": "synthetic"}],
        )
    assert p.endpoint_fn.await_count == 2
    p._rebuild_context.assert_awaited_once()
