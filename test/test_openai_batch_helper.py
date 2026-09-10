"""
Test OpenAI Batch API helper lifecycle and result dispatch.
"""

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from openai.types import Batch, FileObject

from agents.providers import openai as openai_provider
from agents.providers.openai import (
    AzureOpenAIBatchProvider,
    OpenAIBatchAPIHelper,
    RequestTooLargeError,
)

TestBatchStatus = Literal["completed", "in_progress", "cancelling"]


def make_request(custom_id: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {},
    }


def make_batch_input(*custom_ids: str) -> BytesIO:
    return BytesIO(
        b"".join(
            AzureOpenAIBatchProvider._serialize_request(make_request(custom_id))
            for custom_id in custom_ids
        )
    )


def make_batch(status: TestBatchStatus = "completed") -> Batch:
    return Batch(
        id="batch_123",
        completion_window="24h",
        created_at=0,
        endpoint="/v1/chat/completions",
        input_file_id="file_123",
        object="batch",
        output_file_id="file_456" if status == "completed" else None,
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


def make_result(custom_id: str, content: str = "Done") -> dict:
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
                            "content": content,
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


def make_queue_provider(queue=None, **kwargs):
    defaults = {
        "quiet": True,
        "batch_q": queue if queue is not None else asyncio.Queue(),
        "batch_out": {},
        "_serialize_request": AzureOpenAIBatchProvider._serialize_request,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_recording_batch_provider():
    submitted_payloads = []
    submitted_files = []

    async def send_batch(file_content):
        submitted_files.append(file_content)
        assert not file_content.closed
        payload = file_content.read()
        submitted_payloads.append(payload)
        return SimpleNamespace(
            id=f"file-{len(submitted_payloads)}",
            payload=payload,
        )

    async def create_batch_task(batch_file, **_kwargs):
        return SimpleNamespace(
            id=f"batch-{batch_file.id}",
            errors=None,
            payload=batch_file.payload,
        )

    async def get_batch_results(batch_task):
        return [
            make_result(json.loads(line)["custom_id"])
            for line in batch_task.payload.splitlines()
        ]

    provider = make_queue_provider(
        send_batch=send_batch,
        create_batch_task=create_batch_task,
        get_batch_results=get_batch_results,
    )
    return provider, submitted_payloads, submitted_files


def make_dispatch_helper(results):
    provider = make_queue_provider(
        send_batch=AsyncMock(return_value=object()),
        create_batch_task=AsyncMock(return_value=make_batch()),
        get_batch_results=AsyncMock(return_value=results),
    )
    helper = OpenAIBatchAPIHelper(batch_size=10)
    helper.provider = provider
    helper.lock = asyncio.Semaphore(0)
    return helper, provider


def make_remote_cancel_provider():
    retrieve_started = asyncio.Event()
    cancel_started = asyncio.Event()
    cancel_release = asyncio.Event()
    cancel_completed = asyncio.Event()

    async def retrieve(batch_id: str):
        retrieve_started.set()
        await asyncio.Future()

    async def cancel(batch_id: str):
        cancel_started.set()
        await cancel_release.wait()
        cancel_completed.set()
        return make_batch("cancelling")

    batches = SimpleNamespace(
        create=AsyncMock(return_value=make_batch("in_progress")),
        retrieve=AsyncMock(side_effect=retrieve),
        cancel=AsyncMock(side_effect=cancel),
    )
    provider = object.__new__(AzureOpenAIBatchProvider)
    provider.quiet = True
    provider.batch_q = asyncio.Queue()
    provider.batch_out = {}
    provider.llm = SimpleNamespace(batches=batches)
    provider.send_batch = AsyncMock(return_value=make_batch_file())
    provider.get_batch_results = AsyncMock()
    events = SimpleNamespace(
        retrieve_started=retrieve_started,
        cancel_started=cancel_started,
        cancel_release=cancel_release,
        cancel_completed=cancel_completed,
    )
    return provider, batches, events


class NotifyingQueue(asyncio.Queue):
    def __init__(self):
        super().__init__()
        self.received = asyncio.Event()

    async def get(self):
        item = await super().get()
        self.received.set()
        return item


class BlockingSemaphore:
    def __init__(self):
        self.acquire_started = asyncio.Event()

    async def acquire(self):
        self.acquire_started.set()
        await asyncio.Future()

    def release(self):
        pass


class BlockingLock:
    def __init__(self):
        self.acquire_started = asyncio.Event()
        self.release_acquire = asyncio.Event()

    async def __aenter__(self):
        self.acquire_started.set()
        await self.release_acquire.wait()

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass


@pytest.mark.asyncio
async def test_close_during_batch_collection_accounts_for_first_request(monkeypatch):
    batch_files = []

    def make_spooled_file(**_kwargs):
        batch_file = BytesIO()
        batch_files.append(batch_file)
        return batch_file

    monkeypatch.setattr(openai_provider, "SpooledTemporaryFile", make_spooled_file)
    queue = NotifyingQueue()
    provider = make_queue_provider(queue)
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(batch_size=2)
    helper.timeout = 60
    helper.register_provider(provider)

    try:
        queue.put_nowait(make_request("task-1"))
        await asyncio.wait_for(queue.received.wait(), timeout=1)
        await asyncio.wait_for(queue.join(), timeout=1)

        await helper.close()

        assert helper.task is not None and helper.task.done()
        assert future.cancelled()
        assert provider.batch_out == {}
        assert batch_files and all(batch_file.closed for batch_file in batch_files)
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_close_during_semaphore_wait_cancels_unsubmitted_requests():
    provider = make_queue_provider()
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(batch_size=1)
    helper.register_provider(provider)
    semaphore = BlockingSemaphore()
    helper.lock = semaphore

    try:
        provider.batch_q.put_nowait(make_request("task-1"))
        await asyncio.wait_for(semaphore.acquire_started.wait(), timeout=1)

        await helper.close()

        await asyncio.wait_for(provider.batch_q.join(), timeout=1)
        assert helper.task is not None and helper.task.done()
        assert helper.batch_tasks == set()
        assert future.cancelled()
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_close_awaits_in_flight_handlers_and_is_repeatable():
    upload_started = asyncio.Event()

    async def send_batch(batch):
        upload_started.set()
        await asyncio.Future()

    provider = make_queue_provider(send_batch=send_batch)
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(batch_size=1)
    helper.register_provider(provider)

    try:
        provider.batch_q.put_nowait(make_request("task-1"))
        await asyncio.wait_for(upload_started.wait(), timeout=1)
        handler_tasks = set(helper.batch_tasks)

        await helper.close()
        await helper.close()

        assert all(task.done() for task in handler_tasks)
        assert helper.batch_tasks == set()
        assert future.cancelled()
        await asyncio.wait_for(provider.batch_q.join(), timeout=1)
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_concurrent_close_does_not_interrupt_remote_cancellation():
    provider, batches, events = make_remote_cancel_provider()
    future = asyncio.get_running_loop().create_future()
    provider.batch_out["task-1"] = future
    helper = OpenAIBatchAPIHelper(batch_size=1)
    helper.register_provider(provider)

    try:
        provider.batch_q.put_nowait(make_request("task-1"))
        await asyncio.wait_for(events.retrieve_started.wait(), timeout=1)
        first_close = asyncio.create_task(helper.close())
        await asyncio.wait_for(events.cancel_started.wait(), timeout=1)

        second_close = asyncio.create_task(helper.close())
        await asyncio.sleep(0)

        assert not first_close.done()
        assert not second_close.done()
        events.cancel_release.set()
        await asyncio.wait_for(asyncio.gather(first_close, second_close), timeout=1)
        assert events.cancel_completed.is_set()
        batches.cancel.assert_awaited_once_with("batch_123")
        assert future.cancelled()
    finally:
        events.cancel_release.set()
        await helper.close()


@pytest.mark.asyncio
async def test_cancelled_close_waiter_still_cleans_local_requests():
    provider, batches, events = make_remote_cancel_provider()
    in_flight = asyncio.get_running_loop().create_future()
    queued = asyncio.get_running_loop().create_future()
    provider.batch_out.update({"task-1": in_flight, "task-2": queued})
    helper = OpenAIBatchAPIHelper(batch_size=1)
    helper.register_provider(provider)

    try:
        provider.batch_q.put_nowait(make_request("task-1"))
        await asyncio.wait_for(events.retrieve_started.wait(), timeout=1)
        close_waiter = asyncio.create_task(helper.close())
        await asyncio.wait_for(events.cancel_started.wait(), timeout=1)
        provider.batch_q.put_nowait(make_request("task-2"))

        close_waiter.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_waiter, timeout=1)
        await asyncio.wait_for(provider.batch_q.join(), timeout=1)
        assert provider.batch_q.empty()
        assert provider.batch_out == {}
        assert in_flight.cancelled()
        assert queued.cancelled()
        assert helper._close_task is not None and not helper._close_task.done()

        events.cancel_release.set()
        await asyncio.wait_for(helper.close(), timeout=1)
        assert events.cancel_completed.is_set()
        batches.cancel.assert_awaited_once_with("batch_123")
    finally:
        events.cancel_release.set()
        await helper.close()


@pytest.mark.asyncio
async def test_close_drains_queued_requests_and_guards_registration():
    provider = make_queue_provider()
    futures = {
        f"task-{idx}": asyncio.get_running_loop().create_future() for idx in range(1, 3)
    }
    provider.batch_out.update(futures)
    helper = OpenAIBatchAPIHelper(batch_size=2)
    helper.register_provider(provider)

    try:
        for custom_id in futures:
            provider.batch_q.put_nowait(make_request(custom_id))

        with pytest.raises(RuntimeError, match="already registered"):
            helper.register_provider(provider)

        await helper.close()
        await asyncio.wait_for(provider.batch_q.join(), timeout=1)

        assert provider.batch_q.empty()
        assert provider.batch_out == {}
        assert all(future.cancelled() for future in futures.values())
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_batcher_splits_requests_before_jsonl_file_exceeds_limit(
    monkeypatch,
):
    requests = [make_request("task-1"), make_request("task-2")]
    request_size = len(AzureOpenAIBatchProvider._serialize_request(requests[0]))
    assert all(
        len(AzureOpenAIBatchProvider._serialize_request(request)) <= request_size
        for request in requests
    )
    monkeypatch.setattr(openai_provider, "MAX_BATCH_FILE_SIZE", request_size)

    provider, submitted_payloads, submitted_files = make_recording_batch_provider()
    futures = {
        request["custom_id"]: asyncio.get_running_loop().create_future()
        for request in requests
    }
    provider.batch_out.update(futures)
    helper = OpenAIBatchAPIHelper(batch_size=len(requests))
    helper.timeout = 0.01
    helper.register_provider(provider)

    try:
        for request in requests:
            provider.batch_q.put_nowait(request)

        await asyncio.wait_for(asyncio.gather(*futures.values()), timeout=1)

        assert all(
            len(payload) <= openai_provider.MAX_BATCH_FILE_SIZE
            for payload in submitted_payloads
        )
        assert [
            json.loads(line)["custom_id"]
            for payload in submitted_payloads
            for line in payload.splitlines()
        ] == ["task-1", "task-2"]
        assert all(batch_file.closed for batch_file in submitted_files)
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_batcher_rejects_oversized_request_and_processes_next(monkeypatch):
    valid_request = make_request("task-valid")
    limit = len(AzureOpenAIBatchProvider._serialize_request(valid_request))
    oversized_request = make_request("task-oversized")
    oversized_request["body"] = {"input": "x" * limit}
    monkeypatch.setattr(openai_provider, "MAX_BATCH_FILE_SIZE", limit)

    provider, submitted_payloads, submitted_files = make_recording_batch_provider()
    oversized_future = asyncio.get_running_loop().create_future()
    valid_future = asyncio.get_running_loop().create_future()
    provider.batch_out.update(
        {
            "task-oversized": oversized_future,
            "task-valid": valid_future,
        }
    )
    helper = OpenAIBatchAPIHelper(batch_size=1)
    helper.timeout = 0.01
    helper.register_provider(provider)

    try:
        provider.batch_q.put_nowait(oversized_request)
        provider.batch_q.put_nowait(valid_request)

        with pytest.raises(RequestTooLargeError) as exc_info:
            await oversized_future
        await asyncio.wait_for(valid_future, timeout=1)

        assert exc_info.value.custom_id == "task-oversized"
        assert exc_info.value.size > exc_info.value.limit == limit
        assert [
            json.loads(line)["custom_id"]
            for payload in submitted_payloads
            for line in payload.splitlines()
        ] == ["task-valid"]
        assert all(batch_file.closed for batch_file in submitted_files)
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_query_batch_mode_removes_future_when_caller_is_cancelled():
    provider = object.__new__(AzureOpenAIBatchProvider)
    provider.batch_idx = 1
    provider.batch_idx_lock = asyncio.Lock()
    provider.batch_q = asyncio.Queue()
    provider.batch_out = {}
    provider.batch_handler = SimpleNamespace(_closed=False)
    task = asyncio.create_task(provider.query_batch_mode([], model="gpt-4o"))
    request = await asyncio.wait_for(provider.batch_q.get(), timeout=1)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    provider.batch_q.task_done()
    assert request["custom_id"] == "task-1"
    assert provider.batch_out == {}


@pytest.mark.asyncio
async def test_query_waiting_for_id_lock_rechecks_closed_helper():
    provider = object.__new__(AzureOpenAIBatchProvider)
    provider.batch_idx = 1
    provider.batch_idx_lock = BlockingLock()
    provider.batch_q = asyncio.Queue()
    provider.batch_out = {}
    provider.batch_handler = SimpleNamespace(_closed=False)
    task = asyncio.create_task(provider.query_batch_mode([], model="gpt-4o"))
    await asyncio.wait_for(provider.batch_idx_lock.acquire_started.wait(), timeout=1)

    provider.batch_handler._closed = True
    provider.batch_idx_lock.release_acquire.set()

    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(task, timeout=1)
    assert provider.batch_q.empty()
    assert provider.batch_out == {}


@pytest.mark.asyncio
async def test_result_dispatch_skips_abandoned_futures():
    results = [
        make_result("task-cancelled"),
        make_result("task-missing"),
        make_result("task-active", content="Active result"),
    ]
    helper, provider = make_dispatch_helper(results)
    active = asyncio.get_running_loop().create_future()
    cancelled = asyncio.get_running_loop().create_future()
    cancelled.cancel()
    provider.batch_out.update({"task-active": active, "task-cancelled": cancelled})

    batch_ids = ["task-active", "task-cancelled", "task-missing"]
    batch = make_batch_input(*batch_ids)

    await helper._batch_handler(batch, batch_ids)

    assert active.result().choices[0].message.content == "Active result"
    assert cancelled.cancelled()
    assert batch.closed


@pytest.mark.asyncio
async def test_malformed_result_does_not_fail_siblings():
    results = [
        None,
        ["not", "a", "result"],
        {"custom_id": "task-malformed", "response": {"body": {}}},
        make_result("task-valid", content="Valid result"),
    ]
    helper, provider = make_dispatch_helper(results)
    malformed = asyncio.get_running_loop().create_future()
    valid = asyncio.get_running_loop().create_future()
    provider.batch_out.update({"task-malformed": malformed, "task-valid": valid})

    batch_ids = ["task-malformed", "task-valid"]
    batch = make_batch_input(*batch_ids)

    await helper._batch_handler(batch, batch_ids)

    assert malformed.exception() is not None
    assert valid.result().choices[0].message.content == "Valid result"
    assert batch.closed


@pytest.mark.asyncio
async def test_missing_batch_result_sets_request_exception():
    helper, provider = make_dispatch_helper([make_result("task-returned")])
    returned = asyncio.get_running_loop().create_future()
    missing = asyncio.get_running_loop().create_future()
    provider.batch_out.update({"task-returned": returned, "task-missing": missing})

    batch_ids = ["task-returned", "task-missing"]
    batch = make_batch_input(*batch_ids)

    await helper._batch_handler(batch, batch_ids)

    assert returned.result().choices[0].message.content == "Done"
    with pytest.raises(RuntimeError, match="returned no result"):
        await missing
    assert batch.closed


@pytest.mark.asyncio
async def test_batch_error_propagation_skips_missing_futures():
    error = RuntimeError("upload failed")
    helper, provider = make_dispatch_helper([])
    provider.send_batch = AsyncMock(side_effect=error)
    active = asyncio.get_running_loop().create_future()
    provider.batch_out["task-active"] = active
    batch_ids = ["task-missing", "task-active"]
    batch = make_batch_input(*batch_ids)

    with pytest.raises(RuntimeError) as exc_info:
        await helper._batch_handler(batch, batch_ids)

    assert exc_info.value is error
    assert active.exception() is error
    assert batch.closed


@pytest.mark.asyncio
async def test_provider_init_failure_does_not_register_helper():
    helper = OpenAIBatchAPIHelper(batch_size=1)

    with (
        patch.object(
            AzureOpenAIBatchProvider,
            "authenticate",
            side_effect=RuntimeError("authentication failed"),
        ),
        pytest.raises(RuntimeError, match="authentication failed"),
    ):
        AzureOpenAIBatchProvider("gpt-4o", batch_handler=helper, progress_max_items=None, quiet=True)

    assert helper.task is None
    assert helper.batch_tasks == set()
    assert helper._progress_renderer is None
