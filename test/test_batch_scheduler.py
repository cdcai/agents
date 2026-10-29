"""Exercise scheduling through provider forwards and single-attempt primitives."""

import asyncio
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from agents.exceptions import RetryableAgentError
from agents.providers.openai import AzureOpenAIBatchProvider, OpenAIBatchAPIHelper
from agents.retries import RetryPolicy
from test.test_openai_batch_helper import make_batch, make_batch_file


def provider(*, strategy="retrieve", attempts=3):
    result = object.__new__(AzureOpenAIBatchProvider)
    result.quiet = True
    result.batch_q = asyncio.Queue()
    result.batch_out = {}
    result.retry_policy = RetryPolicy(max_attempts=attempts, initial_delay=0)
    result.llm = SimpleNamespace(
        files=SimpleNamespace(
            create=AsyncMock(return_value=make_batch_file()), content=AsyncMock()
        ),
        batches=SimpleNamespace(
            create=AsyncMock(return_value=make_batch("in_progress")),
            retrieve=AsyncMock(return_value=make_batch()),
            list=AsyncMock(),
            cancel=AsyncMock(return_value=make_batch("cancelling")),
        ),
    )
    result.batch_handler = OpenAIBatchAPIHelper(
        batch_size=1, n_workers=2, progress_max_items=0, polling_strategy=strategy
    )
    result.batch_handler.register_provider(result)
    return result


def rate_limit(headers):
    return openai.RateLimitError(
        "busy",
        response=httpx.Response(
            429, headers=headers, request=httpx.Request("POST", "https://example.test")
        ),
        body=None,
    )


@pytest.mark.asyncio
async def test_upload_rewinds_original_offset_and_forwards_keywords():
    p = provider()
    contents = []

    async def upload(**kwargs):
        contents.append(kwargs["file"][1].read())
        assert kwargs["extra_headers"] == {"test": "upload"}
        if len(contents) < 3:
            raise RetryableAgentError("busy")
        return make_batch_file()

    p.llm.files.create.side_effect = upload
    stream = BytesIO(b"skip-payload")
    stream.seek(5)
    await p.send_batch(stream, extra_headers={"test": "upload"})
    assert contents == [b"payload"] * 3
    assert not stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "creation", "download"])
@pytest.mark.parametrize("headers", [{"retry-after-ms": "30"}, {"Retry-After": "0.03"}])
async def test_exhausted_job_preserves_server_cooldown_for_sibling(operation, headers):
    p = provider(attempts=1)
    times = []

    async def request(*args, **kwargs):
        times.append(time.monotonic())
        if len(times) == 1:
            raise rate_limit(headers)
        return make_batch() if operation == "creation" else make_batch_file()

    if operation == "upload":
        p._upload_batch_once = request
        call = lambda: p.send_batch(BytesIO(b"payload"))
    elif operation == "creation":
        p._create_batch_once = request
        call = lambda: p.create_batch_task(make_batch_file(), metadata={"name": "test"})
    else:
        p._download_batch_once = request
        call = lambda: p.batch_handler._download_file("file")
    first = asyncio.create_task(call())
    second = asyncio.create_task(call())
    with pytest.raises(RetryableAgentError):
        await first
    await second
    assert len(times) == 2
    assert times[1] - times[0] >= 0.025


@pytest.mark.asyncio
async def test_creation_single_attempt_and_keywords_forwarded():
    p = provider()
    p.llm.batches.create.side_effect = RetryableAgentError("lost response")
    with pytest.raises(RetryableAgentError):
        await p.create_batch_task(make_batch_file(), metadata={"name": "test"})
    p.llm.batches.create.assert_awaited_once_with(
        input_file_id="file_123",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"name": "test"},
    )


@pytest.mark.asyncio
async def test_cancel_upload_awaits_stream_release_and_worker_survives():
    p = provider()
    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stream = BytesIO(b"payload")

    async def upload(file_content, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopping.set()
            await release.wait()
            assert not file_content.closed

    p._upload_batch_once = upload
    task = asyncio.create_task(p.send_batch(stream))
    await started.wait()
    task.cancel()
    await stopping.wait()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    stream.close()
    p._upload_batch_once = AsyncMock(return_value=make_batch_file())
    await p.send_batch(BytesIO(b"next"))
    assert not p.batch_handler.tasks[0].done()


@pytest.mark.asyncio
async def test_close_drains_bounded_queues_and_blocked_producers():
    p = provider()
    started = asyncio.Event()

    async def upload(*args, **kwargs):
        started.set()
        await asyncio.Future()

    p._upload_batch_once = upload
    tasks = [asyncio.create_task(p.send_batch(BytesIO(b"data"))) for _ in range(12)]
    await started.wait()
    await asyncio.sleep(0)
    assert p.batch_handler.send_batch_q.maxsize == 4
    assert p.batch_handler.send_batch_q.full()
    tasks[2].cancel()
    await p.batch_handler.close()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    await p.batch_handler.send_batch_q.join()
    assert p.batch_handler.send_batch_q.empty()
    assert all(task.done() for task in p.batch_handler.tasks)
    with pytest.raises(RuntimeError, match="closed"):
        await p.send_batch(BytesIO())


@pytest.mark.asyncio
async def test_queued_cancellation_skips_stream_and_failure_does_not_kill_worker():
    p = provider()
    started, release = asyncio.Event(), asyncio.Event()

    async def upload(stream, **kwargs):
        started.set()
        await release.wait()
        raise ValueError("bad upload")

    p._upload_batch_once = upload
    first = asyncio.create_task(p.send_batch(BytesIO(b"first")))
    await started.wait()
    stream = BytesIO(b"cancelled")
    second = asyncio.create_task(p.send_batch(stream))
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    stream.close()
    release.set()
    with pytest.raises(ValueError):
        await first
    p._upload_batch_once = AsyncMock(return_value=make_batch_file())
    await p.send_batch(BytesIO(b"last"))
    p._upload_batch_once.assert_awaited_once()


def batch(batch_id, status="in_progress", created_at=100):
    return make_batch().model_copy(
        update={"id": batch_id, "status": status, "created_at": created_at}
    )


@pytest.mark.asyncio
async def test_list_pagination_ignores_unrelated_ids_and_retrieves_missing():
    p = provider(strategy="list")
    p.llm.batches.create.side_effect = [batch("a"), batch("b")]
    entered, release = asyncio.Event(), asyncio.Event()

    async def listing(**kwargs):
        entered.set()
        await release.wait()
        if "after" not in kwargs:
            return SimpleNamespace(data=[batch("unrelated")], has_more=True)
        return SimpleNamespace(data=[batch("a", "completed")], has_more=False)

    p.llm.batches.list.side_effect = listing
    p.llm.batches.retrieve.return_value = batch("b", "expired")
    callback = Mock()
    first = asyncio.create_task(
        p.create_batch_task(make_batch_file(), status_callback=callback)
    )
    await entered.wait()
    second = asyncio.create_task(p.create_batch_task(make_batch_file()))
    while len(p.batch_handler._polls) != 2:
        await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second)
    assert [result.status for result in results] == ["completed", "expired"]
    assert [call.args[0].status for call in callback.call_args_list] == [
        "in_progress",
        "completed",
    ]
    calls = p.llm.batches.list.call_args_list
    assert calls[0].kwargs == {
        "limit": 100,
        "extra_query": {"$filter": "created_at gt 99", "$orderby": "created_at asc"},
    }
    assert calls[1].kwargs["after"] == "unrelated"
    p.llm.batches.retrieve.assert_awaited_once_with("b")


@pytest.mark.asyncio
async def test_list_page_cap_and_fresh_cursor_on_next_sweep():
    p = provider(strategy="list")
    p.llm.batches.list.side_effect = lambda **kwargs: SimpleNamespace(
        data=[batch("other")], has_more=True
    )
    p.llm.batches.retrieve.side_effect = [make_batch("in_progress"), make_batch()]
    await p.create_batch_task(make_batch_file(), timeout=0)
    calls = p.llm.batches.list.call_args_list
    assert len(calls) == 20
    assert "after" not in calls[0].kwargs
    assert "after" not in calls[10].kwargs
    assert p.llm.batches.retrieve.await_count == 2


@pytest.mark.asyncio
async def test_unsupported_filter_falls_back_once(caplog):
    p = provider(strategy="list")
    p.llm.batches.list.side_effect = openai.BadRequestError(
        "$filter is not supported",
        response=httpx.Response(
            400, request=httpx.Request("GET", "https://example.test")
        ),
        body=None,
    )
    p.llm.batches.retrieve.side_effect = [make_batch("in_progress"), make_batch()]
    await p.create_batch_task(make_batch_file(), timeout=0)
    assert p.batch_handler.polling_strategy == "retrieve"
    p.llm.batches.list.assert_awaited_once()
    assert "filter unsupported" in caplog.text


@pytest.mark.asyncio
async def test_transient_list_failure_retries_without_switching_strategy():
    p = provider(strategy="list")
    p.llm.batches.list.side_effect = [
        rate_limit({"retry-after-ms": "1"}),
        SimpleNamespace(data=[make_batch()], has_more=False),
    ]
    await p.create_batch_task(make_batch_file())
    assert p.llm.batches.list.await_count == 2
    assert p.batch_handler.polling_strategy == "list"
    p.llm.batches.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_single_request_does_not_cancel_remote_batch():
    p = provider()
    started, release = asyncio.Event(), asyncio.Event()

    async def retrieve(batch_id):
        started.set()
        await release.wait()
        return make_batch()

    p.llm.batches.retrieve.side_effect = retrieve
    lifecycle = asyncio.create_task(p.create_batch_task(make_batch_file()))
    await started.wait()
    # query_batch_mode callers own only their result future.
    p.batch_idx, p.batch_idx_lock = 1, asyncio.Lock()
    request = asyncio.create_task(p.query_batch_mode([], model="model"))
    await asyncio.sleep(0)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    p.llm.batches.cancel.assert_not_awaited()
    release.set()
    await lifecycle


@pytest.mark.asyncio
async def test_polling_retry_budget_and_coordinator_survives_failure():
    p = provider(attempts=2)
    p.llm.batches.retrieve.side_effect = RetryableAgentError("busy")
    from agents.exceptions import BatchPollingDeadlineExceeded

    with pytest.raises(BatchPollingDeadlineExceeded):
        await p.create_batch_task(make_batch_file(), polling_deadline=0.05)
    # One operation budget, followed by a single metadata refresh at the deadline.
    assert p.llm.batches.retrieve.await_count == 3
    p.llm.batches.cancel.assert_not_awaited()
    p.llm.batches.retrieve.side_effect = None
    p.llm.batches.retrieve.return_value = make_batch()
    await p.create_batch_task(make_batch_file())
    assert not p.batch_handler.tasks[-1].done()


@pytest.mark.asyncio
async def test_repeated_upload_cancellation_waits_for_stream_cleanup():
    p = provider()
    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def upload(stream, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopping.set()
            await release.wait()
            assert not stream.closed

    p._upload_batch_once = upload
    stream = BytesIO(b"payload")
    task = asyncio.create_task(p.send_batch(stream))
    await started.wait()
    task.cancel()
    await stopping.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    stream.close()


@pytest.mark.asyncio
async def test_close_does_not_interrupt_active_upload_cleanup():
    p = provider()
    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def upload(stream, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopping.set()
            await release.wait()
            assert not stream.closed

    p._upload_batch_once = upload
    stream = BytesIO(b"payload")
    task = asyncio.create_task(p.send_batch(stream))
    await started.wait()
    close = asyncio.create_task(p.batch_handler.close())
    await stopping.wait()
    await asyncio.sleep(0)
    assert not close.done()
    assert not task.done()
    release.set()
    await close
    with pytest.raises(asyncio.CancelledError):
        await task
    stream.close()


@pytest.mark.asyncio
async def test_close_awaits_direct_lifecycle_remote_cleanup():
    p = provider()
    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def retrieve(batch_id):
        started.set()
        await asyncio.Future()

    async def cancel(batch_id):
        stopping.set()
        await release.wait()
        return make_batch("cancelling")

    p.llm.batches.retrieve.side_effect = retrieve
    p.llm.batches.cancel.side_effect = cancel
    task = asyncio.create_task(p.create_batch_task(make_batch_file()))
    await started.wait()
    close = asyncio.create_task(p.batch_handler.close())
    await stopping.wait()
    assert not close.done()
    release.set()
    await close
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not p.batch_handler._polls


@pytest.mark.asyncio
async def test_initial_retrieve_is_immediate_and_subsequent_checks_use_interval():
    p = provider()
    times = []

    async def retrieve(batch_id):
        times.append(time.monotonic())
        return make_batch("in_progress" if len(times) == 1 else "completed")

    p.llm.batches.retrieve.side_effect = retrieve
    started = time.monotonic()
    await p.create_batch_task(make_batch_file(), timeout=0.03)
    assert times[0] - started < 0.02
    assert times[1] - times[0] >= 0.025


@pytest.mark.asyncio
async def test_retry_elapsed_budget_includes_server_cooldown():
    p = provider()
    p.retry_policy = RetryPolicy(
        max_attempts=5, initial_delay=0, elapsed_time_budget=0.02
    )
    p._upload_batch_once = AsyncMock(side_effect=rate_limit({"Retry-After": "60"}))
    started = time.monotonic()
    from agents.exceptions import TerminalAgentError

    with pytest.raises(TerminalAgentError) as failure:
        await p.send_batch(BytesIO(b"payload"))
    assert failure.value.code == "retry_deadline_exceeded"
    assert time.monotonic() - started < 0.2
    p._upload_batch_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_completed_creation_racing_cancellation_cleans_remote_batch():
    p = provider()
    started = asyncio.Event()

    async def create(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            # A transport can deliver a response at the cancellation boundary.
            return make_batch("in_progress")

    p._create_batch_once = create
    task = asyncio.create_task(p.create_batch_task(make_batch_file()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    p.llm.batches.cancel.assert_awaited_once_with("batch_123")
    assert not p.batch_handler._polls


@pytest.mark.asyncio
async def test_cancelled_provider_close_keeps_transport_open_until_helper_finishes():
    p = provider()
    p._credential = None
    p.llm.close = AsyncMock()
    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def retrieve(batch_id):
        started.set()
        await asyncio.Future()

    async def cancel(batch_id):
        stopping.set()
        await release.wait()
        return make_batch("cancelling")

    p.llm.batches.retrieve.side_effect = retrieve
    p.llm.batches.cancel.side_effect = cancel
    lifecycle = asyncio.create_task(p.create_batch_task(make_batch_file()))
    await started.wait()
    close = asyncio.create_task(p.close())
    await stopping.wait()
    close.cancel()
    await asyncio.sleep(0)
    p.llm.close.assert_not_awaited()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await close
    with pytest.raises(asyncio.CancelledError):
        await lifecycle
    p.llm.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_exhausted_list_polling_falls_back_to_known_id_without_cancellation():
    p = provider(strategy="list", attempts=2)
    p.llm.batches.list.side_effect = rate_limit({})
    result = await p.create_batch_task(make_batch_file())
    assert result.status == "completed"
    assert p.llm.batches.list.await_count == 2
    p.llm.batches.retrieve.assert_awaited_once_with("batch_123")
    p.llm.batches.create.assert_awaited_once()
    p.llm.batches.cancel.assert_not_awaited()
    assert p.batch_handler.polling_strategy == "list"


@pytest.mark.asyncio
async def test_polling_failure_reconciles_completed_batch_before_cancellation():
    p = provider(attempts=1)
    p.llm.batches.retrieve.side_effect = [rate_limit({}), make_batch()]
    result = await p.create_batch_task(make_batch_file(), timeout=0.01)
    assert result.status == "completed"
    assert p.llm.batches.retrieve.await_count == 2
    p.llm.batches.cancel.assert_not_awaited()
    p.llm.batches.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_exhausted_polling_preserves_remote_identity_and_original_cause():
    p = provider(attempts=1)
    p.llm.batches.retrieve.side_effect = rate_limit({})
    from agents.exceptions import BatchPollingDeadlineExceeded

    with pytest.raises(BatchPollingDeadlineExceeded) as failure:
        await p.create_batch_task(make_batch_file(), polling_deadline=0.03)
    assert failure.value.batch_id == "batch_123"
    assert failure.value.remote_outcome_unknown
    p.llm.batches.cancel.assert_not_awaited()
    p.llm.batches.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_permanent_observation_failure_never_cancels_remote_work():
    p = provider(attempts=1)
    p.llm.batches.retrieve.side_effect = ValueError("Invalid configuration")
    with pytest.raises(ValueError, match="Invalid configuration"):
        await p.create_batch_task(make_batch_file())
    p.llm.batches.cancel.assert_not_awaited()
