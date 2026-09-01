"""
Test OpenAI Batch API lifecycle handling.
"""

import asyncio
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai.types import Batch, FileObject

from agents.providers.openai import (
    OPENAI_BATCH_ACTIVE_STATUSES,
    OPENAI_BATCH_SUCCESS_STATUSES,
    OPENAI_BATCH_TERMINAL_STATUSES,
    AzureOpenAIBatchProvider,
)


BatchStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]


def make_batch(status: BatchStatus) -> Batch:
    return Batch(
        id="batch_123",
        completion_window="24h",
        created_at=0,
        endpoint="/v1/chat/completions",
        input_file_id="file_123",
        object="batch",
        output_file_id="file_123" if status == "completed" else None,
        status=status,
    )


def make_batch_file() -> FileObject:
    return FileObject(
        id="file_123",
        bytes=1024,
        created_at=0,
        filename="batch.jsonl",
        object="file",
        purpose="batch",
        status="uploaded",
    )


def make_provider(batches) -> AzureOpenAIBatchProvider:
    provider = object.__new__(AzureOpenAIBatchProvider)
    provider.llm = SimpleNamespace(batches=batches)
    return provider


def test_openai_batch_statuses_are_exhaustive_and_disjoint():
    assert OPENAI_BATCH_ACTIVE_STATUSES == {
        "validating",
        "in_progress",
        "finalizing",
        "cancelling",
    }
    assert OPENAI_BATCH_TERMINAL_STATUSES == {
        "completed",
        "failed",
        "expired",
        "cancelled",
    }
    assert OPENAI_BATCH_SUCCESS_STATUSES == {"completed"}
    assert OPENAI_BATCH_ACTIVE_STATUSES.isdisjoint(OPENAI_BATCH_TERMINAL_STATUSES)
    assert OPENAI_BATCH_SUCCESS_STATUSES <= OPENAI_BATCH_TERMINAL_STATUSES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status", ["completed", "failed", "expired", "cancelled"]
)
async def test_create_batch_task_reports_updates_and_stops_at_terminal_status(
    terminal_status: BatchStatus,
):
    batches = SimpleNamespace(
        create=AsyncMock(return_value=make_batch("validating")),
        retrieve=AsyncMock(
            side_effect=[make_batch("in_progress"), make_batch(terminal_status)]
        ),
        cancel=AsyncMock(),
    )
    provider = make_provider(batches)
    callback = Mock()

    with patch(
        "agents.providers.openai.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        result = await provider.create_batch_task(
            make_batch_file(), timeout=30, status_callback=callback
        )

    assert result.status == terminal_status
    assert [call.args[0].status for call in callback.call_args_list] == [
        "validating",
        "in_progress",
        terminal_status,
    ]
    assert batches.retrieve.await_count == 2
    sleep.assert_awaited_once_with(30)
    batches.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_batch_task_accepts_no_status_callback():
    batches = SimpleNamespace(
        create=AsyncMock(return_value=make_batch("completed")),
        retrieve=AsyncMock(),
        cancel=AsyncMock(),
    )
    provider = make_provider(batches)

    result = await provider.create_batch_task(make_batch_file())

    assert result.status == "completed"
    batches.retrieve.assert_not_awaited()
    batches.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_error_is_not_masked_by_remote_cancellation_error():
    callback_error = RuntimeError("callback failed")
    batches = SimpleNamespace(
        create=AsyncMock(return_value=make_batch("in_progress")),
        retrieve=AsyncMock(),
        cancel=AsyncMock(side_effect=ValueError("cancel failed")),
    )
    provider = make_provider(batches)

    with pytest.raises(RuntimeError) as exc_info:
        await provider.create_batch_task(
            make_batch_file(), status_callback=Mock(side_effect=callback_error)
        )

    assert exc_info.value is callback_error
    batches.cancel.assert_awaited_once_with("batch_123")


@pytest.mark.asyncio
async def test_local_task_cancellation_cancels_remote_batch_without_masking_error():
    retrieve_started = asyncio.Event()

    async def retrieve(batch_id: str):
        retrieve_started.set()
        await asyncio.Future()

    batches = SimpleNamespace(
        create=AsyncMock(return_value=make_batch("in_progress")),
        retrieve=AsyncMock(side_effect=retrieve),
        cancel=AsyncMock(side_effect=RuntimeError("cancel failed")),
    )
    provider = make_provider(batches)
    task = asyncio.create_task(provider.create_batch_task(make_batch_file()))
    await retrieve_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    batches.cancel.assert_awaited_once_with("batch_123")
