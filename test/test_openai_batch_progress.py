"""Tests for OpenAI batch progress integration."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai.types import Batch

from agents.batch_progress import BatchProgressState
from agents.providers.openai import (
    AzureOpenAIBatchProvider,
    OpenAIBatchAPIHelper,
)

TestBatchStatus = Literal[
    "validating",
    "in_progress",
    "completed",
    "expired",
    "cancelled",
]


def make_request(custom_id: str = "task-1") -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {},
    }


def make_batch(
    status: TestBatchStatus,
    *,
    completed: int = 0,
    failed: int = 0,
    name: str | None = None,
) -> Batch:
    return Batch(
        id="batch_123",
        completion_window="24h",
        created_at=1_756_742_400,
        endpoint="/v1/chat/completions",
        input_file_id="file_123",
        metadata={"name": name} if name is not None else None,
        object="batch",
        output_file_id="file_456" if status == "completed" else None,
        request_counts={
            "total": 10,
            "completed": completed,
            "failed": failed,
        },
        status=status,
    )


def make_result(custom_id: str = "task-1") -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "id": f"chatcmpl-{custom_id}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": "Done",
                            "role": "assistant",
                        },
                    }
                ],
                "created": 0,
                "model": "gpt-4o",
                "object": "chat.completion",
            }
        },
    }


class RecordingRenderer:
    def __init__(self, *, fail_refresh: bool = False):
        self.fail_refresh = fail_refresh
        self.states: list[BatchProgressState] = []
        self.close_count = 0

    def refresh(self, state: BatchProgressState) -> None:
        self.states.append(state)
        if self.fail_refresh:
            raise RuntimeError("render failed")

    def close(self) -> None:
        self.close_count += 1


def make_provider(
    batches: list[Batch],
    *,
    quiet: bool = False,
    create_error: BaseException | None = None,
):
    async def create_batch_task(batch_file, timeout, status_callback):
        for batch in batches:
            status_callback(batch)
        if create_error is not None:
            raise create_error
        return batches[-1]

    return SimpleNamespace(
        quiet=quiet,
        batch_q=asyncio.Queue(),
        batch_out={},
        send_batch=AsyncMock(return_value=object()),
        create_batch_task=AsyncMock(side_effect=create_batch_task),
        get_batch_results=AsyncMock(return_value=[make_result()]),
    )


async def run_batch_handler(helper: OpenAIBatchAPIHelper) -> None:
    await helper.lock.acquire()
    await helper._batch_handler([make_request()])


@pytest.mark.asyncio
async def test_batch_callback_tracks_openai_snapshots_and_request_counts():
    renderer = RecordingRenderer()
    provider = make_provider(
        [
            make_batch("validating", completed=2, name="nightly-run"),
            make_batch("completed", completed=9, failed=1, name="nightly-run"),
        ]
    )
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(
        batch_size=10,
        progress_renderer=renderer,
    )
    helper.register_provider(provider)

    try:
        await run_batch_handler(helper)

        active_states = [state for state in renderer.states if state.active]
        assert active_states
        active = active_states[0].active[0]
        assert active.id == "batch_123"
        assert active.name == "nightly-run"
        assert active.created_at == datetime.fromtimestamp(
            1_756_742_400, tz=timezone.utc
        )
        assert active.request_counts is not None
        assert active.request_counts.total == 10
        assert active.request_counts.completed == 2
        assert active.request_counts.failed == 0

        assert helper.batch_progress.active == ()
        assert dict(helper.batch_progress.finished_counts) == {"completed": 1}
        assert future.result().choices[0].message.content == "Done"
        assert provider.create_batch_task.call_args.kwargs["status_callback"]
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_renderer_failure_does_not_fail_batch_requests():
    renderer = RecordingRenderer(fail_refresh=True)
    provider = make_provider(
        [make_batch("in_progress"), make_batch("completed", completed=10)]
    )
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(batch_size=10, progress_renderer=renderer)
    helper.register_provider(provider)

    try:
        await run_batch_handler(helper)

        assert future.result().choices[0].message.content == "Done"
        assert dict(helper.batch_progress.finished_counts) == {"completed": 1}
        assert len(renderer.states) == 1
        assert renderer.states[0].active[0].name == "batch_123"
    finally:
        await helper.close()


@pytest.mark.parametrize("status", ["expired", "cancelled"])
def test_observed_terminal_statuses_are_aggregated(status: TestBatchStatus):
    helper = OpenAIBatchAPIHelper(batch_size=10)

    helper._record_batch_status(make_batch(status))

    assert helper.batch_progress.active == ()
    assert dict(helper.batch_progress.finished_counts) == {status: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("progress_max_items", "quiet"),
    [(0, False), (10, True)],
)
async def test_zero_limit_and_quiet_provider_suppress_rendering_but_keep_state(
    progress_max_items: int,
    quiet: bool,
):
    renderer = RecordingRenderer()
    provider = make_provider(
        [make_batch("in_progress"), make_batch("completed", completed=10)],
        quiet=quiet,
    )
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(
        batch_size=10,
        progress_max_items=progress_max_items,
        progress_renderer=renderer,
    )
    helper.register_provider(provider)

    try:
        await run_batch_handler(helper)

        assert renderer.states == []
        assert dict(helper.batch_progress.finished_counts) == {"completed": 1}
    finally:
        await helper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_error",
    [RuntimeError("poll failed"), asyncio.CancelledError()],
)
async def test_handler_error_reconciles_last_active_batch(create_error: BaseException):
    renderer = RecordingRenderer()
    provider = make_provider(
        [make_batch("in_progress")],
        create_error=create_error,
    )
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(batch_size=10, progress_renderer=renderer)
    helper.register_provider(provider)

    try:
        with pytest.raises(type(create_error)):
            await run_batch_handler(helper)

        assert helper.batch_progress.active == ()
        assert renderer.states[-1].active == ()
        if isinstance(create_error, Exception):
            assert dict(helper.batch_progress.finished_counts) == {"tracking_failed": 1}
            assert future.exception() is create_error
        else:
            assert dict(helper.batch_progress.finished_counts) == {
                "tracking_cancelled": 1
            }
            assert future.cancelled()
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_helper_close_owns_renderer_and_closes_it_once():
    renderer = RecordingRenderer()
    helper = OpenAIBatchAPIHelper(batch_size=10, progress_renderer=renderer)
    helper.register_provider(make_provider([]))

    await helper.close()
    await helper.close()

    assert renderer.close_count == 1


@pytest.mark.asyncio
async def test_default_renderer_uses_limit_and_quiet_setting():
    renderer = Mock()
    with patch(
        "agents.providers.openai.TqdmBatchProgressRenderer",
        return_value=renderer,
    ) as renderer_type:
        helper = OpenAIBatchAPIHelper(batch_size=10, progress_max_items=4)
        helper.register_provider(make_provider([], quiet=True))
        await helper.close()

    renderer_type.assert_called_once_with(max_items=4, disable=True)
    renderer.close.assert_called_once_with()

def test_provider_forwards_progress_limit_only_to_default_helper():
    helper = Mock()
    progress = BatchProgressState(finished_counts={"completed": 2})
    helper.batch_progress = progress
    with (
        patch.object(AzureOpenAIBatchProvider, "authenticate"),
        patch.object(
            AzureOpenAIBatchProvider,
            "_bearer_token_generator",
            "test-token",
            create=True,
        ),
        patch("agents.providers.openai.openai.AsyncAzureOpenAI") as async_azure_openai,
        patch(
            "agents.providers.openai.OpenAIBatchAPIHelper",
            return_value=helper,
        ) as helper_type,
    ):
        provider = AzureOpenAIBatchProvider(
            "gpt-4o-batch",
            batch_size=25,
            n_workers=3,
            progress_max_items=4,
            api_version="2026-01-01",
        )

    helper_type.assert_called_once_with(
        batch_size=25,
        n_workers=3,
        progress_max_items=4,
    )
    helper.register_provider.assert_called_once_with(provider)
    assert "progress_max_items" not in async_azure_openai.call_args.kwargs
    assert provider.batch_progress is progress


def test_provider_rejects_progress_limit_with_custom_helper():
    helper = Mock()

    with pytest.raises(ValueError, match="custom batch_handler"):
        AzureOpenAIBatchProvider(
            "gpt-4o-batch",
            batch_handler=helper,
            progress_max_items=4,
        )

    helper.register_provider.assert_not_called()
