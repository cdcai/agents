"""Recover known remote work without re-uploading or re-creating requests."""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from agents.exceptions import ContextLengthExceededError, RetryableAgentError
from agents.retries import RetryPolicy
from test.test_batch_failures import azure_context_error
from test.test_batch_scheduler import provider
from test.test_openai_batch_helper import (
    make_batch,
    make_batch_file,
    make_batch_input,
    make_dispatch_helper,
    make_result,
)


def completed_batch(*, output="output", error=None, completed=1, failed=0):
    batch = make_batch()
    batch.output_file_id = output
    batch.error_file_id = error
    batch.request_counts = openai.types.batch.BatchRequestCounts(
        total=completed + failed, completed=completed, failed=failed
    )
    return batch


def content(*rows):
    return SimpleNamespace(content="\n".join(json.dumps(row) for row in rows).encode())


def unavailable():
    return openai.InternalServerError(
        "unavailable",
        response=httpx.Response(
            500,
            headers={"x-request-id": "request-123"},
            request=httpx.Request("GET", "https://example.test"),
        ),
        body=None,
    )


@pytest.mark.asyncio
async def test_error_file_retry_preserves_success_and_exposes_context_error():
    p = provider(attempts=2)
    batch = completed_batch(error="error", failed=1)
    p.llm.batches.retrieve.return_value = batch
    calls = Counter()

    async def download(file_id):
        calls[file_id] += 1
        if file_id == "output":
            return content(make_result("good"))
        if calls[file_id] == 1:
            raise unavailable()
        return content(azure_context_error())

    p.llm.files.content.side_effect = download
    rows = await p.get_batch_results(batch)
    assert {row["custom_id"] for row in rows} == {"good", "bad"}
    assert calls == {"output": 1, "error": 2}
    helper, _ = make_dispatch_helper([])
    helper.provider = p
    p.send_batch = AsyncMock(return_value=make_batch_file())
    p.create_batch_task = AsyncMock(return_value=batch)
    p.get_batch_results = AsyncMock(return_value=rows)
    p.batch_out = {
        key: asyncio.get_running_loop().create_future() for key in ("good", "bad")
    }
    await helper._batch_handler(make_batch_input("good", "bad"), ["good", "bad"])
    assert p.batch_out["good"].result()
    assert isinstance(p.batch_out["bad"].exception(), ContextLengthExceededError)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["no-files", "empty-file", "partial-file"])
async def test_refreshes_incomplete_results_using_request_counts(initial):
    p = provider(attempts=2)
    batch = completed_batch(
        output=None if initial == "no-files" else "output", completed=2
    )
    p.llm.batches.retrieve.return_value = completed_batch(completed=2)
    first = content(make_result("one")) if initial == "partial-file" else content()
    p.llm.files.content.side_effect = (
        [content(make_result("one"), make_result("two"))]
        if initial == "no-files"
        else [first, content(make_result("one"), make_result("two"))]
    )
    rows = await p.get_batch_results(batch)
    assert {row["custom_id"] for row in rows} == {"one", "two"}
    p.llm.batches.retrieve.assert_awaited_once_with("batch_123")
    p.llm.batches.create.assert_not_awaited()
    p.llm.files.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_file_failure_exhaustion_retains_success_and_diagnostics(caplog):
    p = provider(attempts=2)
    batch = completed_batch(error="error", failed=1)
    p.llm.batches.retrieve.return_value = batch
    calls = Counter()

    async def download(file_id):
        calls[file_id] += 1
        if file_id == "output":
            return content(make_result("good"))
        raise unavailable()

    p.llm.files.content.side_effect = download
    rows = await p.get_batch_results(batch)
    assert rows[0]["custom_id"] == "good"
    marker = rows[1]["_batch_protocol_error"]
    assert marker["batch_id"] == batch.id
    assert marker["file_id"] == "error"
    assert marker["retryable"]
    assert calls == {"output": 1, "error": 2}
    assert "request-123" in caplog.text
    assert "Result recovery exhausted" in caplog.text


@pytest.mark.asyncio
async def test_recovery_elapsed_budget_retains_already_read_success():
    p = provider(attempts=3)
    p.retry_policy = RetryPolicy(initial_delay=0, elapsed_time_budget=0.02)
    batch = completed_batch(error="error", failed=1)
    p.llm.batches.retrieve.return_value = batch

    async def download(file_id):
        if file_id == "output":
            return content(make_result("good"))
        raise unavailable()

    async def refresh(batch_id):
        await asyncio.Future()

    p.llm.files.content.side_effect = download
    p.llm.batches.retrieve.side_effect = refresh
    rows = await asyncio.wait_for(p.get_batch_results(batch), 1)
    assert rows[0]["custom_id"] == "good"
    assert rows[-1]["_batch_protocol_error"]["code"] == "retry_deadline_exceeded"
    assert not p.batch_handler._jobs


@pytest.mark.asyncio
async def test_result_recovery_cancellation_propagates_and_settles_download():
    p = provider()
    started = asyncio.Event()

    async def download(file_id):
        started.set()
        await asyncio.Future()

    p.llm.files.content.side_effect = download
    task = asyncio.create_task(p.get_batch_results(completed_batch()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not p.batch_handler._jobs
    p.llm.batches.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_later_partial_snapshot_does_not_erase_readable_rows():
    p = provider(attempts=3)
    batch = completed_batch(completed=2)
    p.llm.batches.retrieve.return_value = batch
    p.llm.files.content.side_effect = [
        content(make_result("one")),
        content(),
        content(make_result("two")),
    ]
    rows = await p.get_batch_results(batch)
    assert {row["custom_id"] for row in rows} == {"one", "two"}
    assert p.llm.files.content.await_count == 3


@pytest.mark.asyncio
async def test_exhausted_refresh_does_not_discard_success_or_file_error():
    p = provider(attempts=3)
    batch = completed_batch(error="error", failed=1)
    p.llm.batches.retrieve.side_effect = unavailable()
    p.llm.files.content.side_effect = [content(make_result("good")), unavailable()]
    rows = await p.get_batch_results(batch)
    assert rows[0]["custom_id"] == "good"
    assert rows[1]["_batch_protocol_error"]["file_id"] == "error"
    assert p.llm.batches.retrieve.await_count == 2
    assert p.llm.files.content.await_count == 2
    assert rows[-1]["_batch_protocol_error"]["output_file_id"] == "output"


@pytest.mark.asyncio
async def test_known_result_file_not_found_is_retried_but_bad_request_is_terminal():
    p = provider(attempts=2)
    batch = completed_batch()
    p.llm.batches.retrieve.return_value = batch
    request = httpx.Request("GET", "https://example.test")
    p.llm.files.content.side_effect = [
        openai.NotFoundError(
            "not visible yet", response=httpx.Response(404, request=request), body=None
        ),
        content(make_result("good")),
    ]
    rows = await p.get_batch_results(batch)
    assert rows[0]["custom_id"] == "good"
    assert p.llm.files.content.await_count == 2
    p.llm.files.content.reset_mock()
    p.llm.batches.retrieve.reset_mock()
    p.llm.files.content.side_effect = openai.BadRequestError(
        "invalid file", response=httpx.Response(400, request=request), body=None
    )
    rows = await p.get_batch_results(batch)
    assert not rows[0]["_batch_protocol_error"]["retryable"]
    p.llm.files.content.assert_awaited_once()
    p.llm.batches.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_download_is_classified_as_retrieval_failure_without_replay():
    p = provider(attempts=2)
    batch = completed_batch()
    p.llm.batches.retrieve.return_value = batch
    p.llm.files.content.side_effect = unavailable()
    p.send_batch = AsyncMock(return_value=make_batch_file())
    p.create_batch_task = AsyncMock(return_value=batch)
    p.batch_out["missing"] = asyncio.get_running_loop().create_future()
    helper, _ = make_dispatch_helper([])
    helper.provider = p
    await helper._batch_handler(make_batch_input("missing"), ["missing"])
    error = p.batch_out["missing"].exception()
    assert isinstance(error, RetryableAgentError)
    assert error.code == "batch_result_retrieval_failed"
    assert error.batch_id == batch.id and error.custom_id == "missing"
    assert error.remote_outcome_unknown and error.operation_scoped
    assert p.llm.files.content.await_count == 2
    p.create_batch_task.assert_awaited_once()
    p.send_batch.assert_awaited_once()
