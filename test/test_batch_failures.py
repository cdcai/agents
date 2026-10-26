"""Small synthetic batch files exercise failures without report data or APIs."""

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from agents import Agent, StopOnStep
from agents.abstract import CallbackCondition
from agents.budgets import RequestBudget, RequestBudgetExceededError
from agents.exceptions import (
    BatchPollingDeadlineExceeded,
    BatchProtocolError,
    ContextLengthExceededError,
    RetryableAgentError,
    RetryableBatchRequestError,
    TerminalAgentError,
)
from agents.processors import BatchProcessorIterable, ProcessorIterable
from agents.providers.openai import AzureOpenAIBatchProvider, AzureOpenAIProvider
from agents.retries import RetryPolicy, retry_operation
from test.test_openai_batch_helper import (
    make_batch,
    make_batch_input,
    make_dispatch_helper,
    make_result,
)


def azure_context_error(custom_id="bad"):
    return {
        "custom_id": custom_id,
        "response": None,
        "error": {
            "code": None,
            "message": {
                "error": {
                    "message": "Input tokens exceed the configured limit of 272000 tokens.",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded",
                }
            },
        },
    }


def file_provider(batch):
    provider = object.__new__(AzureOpenAIBatchProvider)
    provider.retry_policy = RetryPolicy(max_attempts=2, initial_delay=0)
    provider.llm = SimpleNamespace(files=SimpleNamespace(content=AsyncMock()))
    provider.quiet = True
    provider.batch_q = asyncio.Queue()
    provider.batch_out = {}
    from agents.providers.openai import OpenAIBatchAPIHelper

    provider.batch_handler = OpenAIBatchAPIHelper(batch_size=1, progress_max_items=0)
    provider.batch_handler.register_provider(provider)
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "mixed", "all-error", "no-files"])
async def test_reads_success_and_error_files_including_all_error_batches(mode):
    batch = make_batch()
    batch.output_file_id = "output" if mode in {"success", "mixed"} else None
    batch.error_file_id = "error" if mode in {"mixed", "all-error"} else None
    provider = file_provider(batch)
    rows = {"output": make_result("good"), "error": azure_context_error()}

    async def content(file_id):
        return SimpleNamespace(content=(json.dumps(rows[file_id]) + "\n").encode())

    provider.llm.files.content.side_effect = content
    results = await provider.get_batch_results(batch)
    assert {row["custom_id"] for row in results} == {
        rows[file_id]["custom_id"]
        for file_id in (batch.output_file_id, batch.error_file_id)
        if file_id
    }
    assert provider.llm.files.content.await_count == bool(batch.output_file_id) + bool(
        batch.error_file_id
    )


@pytest.mark.asyncio
async def test_mixed_batch_propagates_normalized_error_without_losing_success():
    raw = azure_context_error()
    helper, provider = make_dispatch_helper([make_result("good"), raw])
    provider.batch_out = {
        key: asyncio.get_running_loop().create_future() for key in ("good", "bad")
    }
    await helper._batch_handler(make_batch_input("good", "bad"), ["good", "bad"])
    assert provider.batch_out["good"].result().choices[0].message.content == "Done"
    error = provider.batch_out["bad"].exception()
    assert isinstance(error, ContextLengthExceededError)
    assert error.code == "context_length_exceeded"
    assert error.custom_id == "bad" and error.batch_id == "batch_123"
    assert error.details == raw and not error.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_row",
    [
        {
            "custom_id": "bad",
            "response": None,
            "error": {"code": None, "message": "The request timed out. Please retry."},
        },
        {
            "custom_id": "bad",
            "response": {
                "status_code": 503,
                "body": {"error": {"code": "server_error", "message": "Unavailable"}},
            },
        },
    ],
)
async def test_timeout_and_http_service_failures_are_retryable(error_row):
    helper, provider = make_dispatch_helper([error_row])
    future = provider.batch_out["bad"] = asyncio.get_running_loop().create_future()
    await helper._batch_handler(make_batch_input("bad"), ["bad"])
    assert isinstance(future.exception(), RetryableBatchRequestError)
    assert future.exception().details == error_row


@pytest.mark.asyncio
async def test_duplicate_missing_unexpected_and_malformed_rows_resolve_every_future():
    helper, provider = make_dispatch_helper(
        [
            make_result("good"),
            make_result("duplicate"),
            make_result("duplicate"),
            make_result("unexpected"),
            {"custom_id": "malformed", "response": None},
            None,
            ["bad"],
        ]
    )
    ids = ["good", "duplicate", "missing", "malformed"]
    provider.batch_out = {
        key: asyncio.get_running_loop().create_future() for key in ids
    }
    await helper._batch_handler(make_batch_input(*ids), ids)
    assert all(future.done() for future in provider.batch_out.values())
    assert provider.batch_out["good"].result()
    for key, code in [
        ("duplicate", "duplicate_batch_result"),
        ("missing", "missing_batch_result"),
        ("malformed", "malformed_batch_result"),
    ]:
        error = provider.batch_out[key].exception()
        assert isinstance(error, BatchProtocolError) and error.code == code


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["expired", "cancelled", "failed"])
async def test_terminal_partial_batch_retains_completed_requests(status):
    batch = make_batch()
    batch.status = status
    batch.output_file_id = "output"
    batch.error_file_id = "error"
    provider = file_provider(batch)

    async def content(file_id):
        if file_id == "output":
            return SimpleNamespace(
                content=(json.dumps(make_result("good")) + "\ninvalid json\n").encode()
            )
        raise ValueError("broken file")

    provider.llm.files.content.side_effect = content
    rows = await provider.get_batch_results(batch)
    assert rows[0]["custom_id"] == "good"
    assert len([row for row in rows if "_batch_protocol_error" in row]) == 2


@pytest.mark.asyncio
async def test_polling_deadline_preserves_available_successes_and_does_not_resubmit():
    initial = make_batch("in_progress")
    partial = make_batch()
    partial.status = "cancelled"
    partial.output_file_id = "output"
    retrieve_calls = 0

    async def retrieve(_batch_id):
        nonlocal retrieve_calls
        retrieve_calls += 1
        if retrieve_calls == 1:
            await asyncio.Future()
        return partial

    provider = file_provider(initial)
    provider.batch_polling_deadline = 0.01
    provider.llm.batches = SimpleNamespace(
        create=AsyncMock(return_value=initial),
        retrieve=AsyncMock(side_effect=retrieve),
        cancel=AsyncMock(return_value=partial),
    )
    provider.llm.files.content.return_value = SimpleNamespace(
        content=json.dumps(make_result("good")).encode()
    )
    provider.send_batch = AsyncMock(return_value=SimpleNamespace(id="file"))
    ids = ["good", "missing"]
    provider.batch_out = {
        key: asyncio.get_running_loop().create_future() for key in ids
    }
    helper, _ = make_dispatch_helper([])
    helper.provider = provider
    await helper._batch_handler(make_batch_input(*ids), ids)
    assert provider.batch_out["good"].result()
    assert isinstance(
        provider.batch_out["missing"].exception(), BatchPollingDeadlineExceeded
    )
    assert provider.batch_out["missing"].exception().remote_outcome_unknown
    provider.llm.batches.create.assert_awaited_once()
    provider.llm.batches.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_operation_retries_only_transient_failure_and_preserves_cancellation():
    operation = AsyncMock(side_effect=[RetryableAgentError("busy"), "done"])
    result = await retry_operation(
        operation, RetryPolicy(max_attempts=2, initial_delay=0)
    )
    assert result == "done" and operation.await_count == 2
    terminal = AsyncMock(side_effect=ContextLengthExceededError("too large"))
    with pytest.raises(ContextLengthExceededError):
        await retry_operation(terminal, RetryPolicy(initial_delay=0))
    terminal.assert_awaited_once()
    cancelled = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await retry_operation(cancelled, RetryPolicy())
    cancelled.assert_awaited_once()


@pytest.mark.asyncio
async def test_sdk_context_error_is_terminal_and_retry_budget_bounds_attempts():
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://example.test")
    )
    error = openai.BadRequestError(
        "too large", response=response, body={"code": "context_length_exceeded"}
    )
    operation = AsyncMock(side_effect=error)
    with pytest.raises(ContextLengthExceededError) as failure:
        await retry_operation(operation, RetryPolicy(initial_delay=0))
    assert failure.value.__cause__ is error
    operation.assert_awaited_once()
    operation = AsyncMock(side_effect=RetryableAgentError("busy"))
    with pytest.raises(RetryableAgentError):
        await retry_operation(operation, RetryPolicy(max_attempts=2, initial_delay=0))
    assert operation.await_count == 2


def prompt_provider(request_budget=None):
    provider = object.__new__(AzureOpenAIProvider)
    provider.model_name = "private-deployment"
    provider.request_budget = request_budget
    provider.retry_policy = RetryPolicy(max_attempts=2, initial_delay=0)
    provider.all_usage = []
    provider.endpoint_fn = AsyncMock(
        return_value=ChatCompletion.model_validate(
            make_result("good")["response"]["body"]
        )
    )
    return provider


class SimpleAgent(Agent):
    BASE_PROMPT = "Process {batch}"

    def __init__(self, provider, **kwargs):
        super().__init__(StopOnStep(0), provider=provider, **kwargs)


@pytest.mark.asyncio
async def test_final_preflight_blocks_known_oversize_before_batch_queue_or_network():
    budget = RequestBudget(
        max_input_tokens=300, output_reservation=0, safety_margin=0, token_counter=len
    )
    provider = prompt_provider(budget)
    agent = SimpleAgent(provider, batch="one")
    initial = [{"role": "user", "content": "small"}]
    await provider.prompt_agent(agent, initial)
    expanded = [*initial, {"role": "tool", "tool_call_id": "id", "content": "x" * 300}]
    with pytest.raises(RequestBudgetExceededError):
        await provider.prompt_agent(agent, expanded)
    provider.endpoint_fn.assert_awaited_once()
    assert len(agent.request_history) == 1


@pytest.mark.asyncio
async def test_context_rebuild_is_checked_and_submitted_while_audit_is_preserved():
    budget = RequestBudget(
        max_input_tokens=300, output_reservation=0, safety_margin=0, token_counter=len
    )
    provider = prompt_provider(budget)
    hook = AsyncMock(return_value=[{"role": "user", "content": "compact"}])
    agent = SimpleAgent(provider, batch="one", context_rebuild_hook=hook)
    original = [{"role": "user", "content": "x" * 500}]
    await provider.prompt_agent(agent, original)
    hook.assert_awaited_once()
    assert provider.endpoint_fn.await_args.kwargs["messages"][0]["content"] == "compact"
    assert original[0]["content"] == "x" * 500
    assert agent.request_history[0]["messages"][0]["content"] == "compact"
    hook.return_value = original
    with pytest.raises(RequestBudgetExceededError):
        await provider.prompt_agent(agent, original)
    assert provider.endpoint_fn.await_count == 1


@pytest.mark.asyncio
async def test_transient_prompt_retry_does_not_repeat_success_callback_or_model_work():
    provider = prompt_provider()
    response = provider.endpoint_fn.return_value
    provider.endpoint_fn.side_effect = [RetryableAgentError("busy"), response]
    callback = AsyncMock()
    callback.condition = CallbackCondition.ON_SUCCESS
    agent = SimpleAgent(provider, batch="one", callbacks=[callback])
    await agent()
    assert agent.answer == "Done" and provider.endpoint_fn.await_count == 2
    callback.assert_awaited_once()
    assert len(agent.request_history) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("processor_type", [BatchProcessorIterable, ProcessorIterable])
async def test_processor_does_not_reset_terminal_requests_and_exposes_failures(
    processor_type,
):
    provider = prompt_provider()

    async def endpoint(**kwargs):
        if "bad" in str(kwargs["messages"]):
            raise ContextLengthExceededError("too large")
        return ChatCompletion.model_validate(make_result("good")["response"]["body"])

    provider.endpoint_fn.side_effect = endpoint
    processor = processor_type(
        ["good", "bad"], SimpleAgent, provider, batch_size=1, n_retry=5
    )
    answers = await processor.process()
    assert answers[0] == "Done" and len(answers) == 2
    assert provider.endpoint_fn.await_count == 2
    assert isinstance(processor.failures[1], ContextLengthExceededError)
    assert processor.agents[1].last_error is processor.failures[1]


@pytest.mark.asyncio
async def test_all_failed_agents_remain_inspectable_before_aggregate_failure():
    provider = prompt_provider()
    provider.endpoint_fn.side_effect = TerminalAgentError("denied")
    processor = BatchProcessorIterable(["bad"], SimpleAgent, provider, batch_size=1)
    with pytest.raises(RuntimeError, match="No agents were successful"):
        await processor.process()
    assert len(processor.agents) == 1 and set(processor.failures) == {0}
    provider.endpoint_fn.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("processor_type", [BatchProcessorIterable, ProcessorIterable])
async def test_cancellation_awaits_owned_workers_and_preserves_agent_inventory(
    processor_type,
):
    provider = prompt_provider()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def endpoint(**kwargs):
        if "good" in str(kwargs["messages"]):
            return ChatCompletion.model_validate(
                make_result("good")["response"]["body"]
            )
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    provider.endpoint_fn.side_effect = endpoint
    processor = processor_type(["good", "stalled"], SimpleAgent, provider, batch_size=1)
    task = asyncio.create_task(processor.process())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert len(processor.agents) == 2
    assert processor.agents[0].answer == "Done"


@pytest.mark.asyncio
async def test_raw_timeout_is_not_mislabeled_as_retry_or_polling_deadline():
    error = TimeoutError("local transport timeout")
    operation = AsyncMock(side_effect=error)
    with pytest.raises(TimeoutError) as failure:
        await retry_operation(operation, RetryPolicy(elapsed_time_budget=10))
    assert failure.value is error
    provider = file_provider(make_batch())
    provider.batch_polling_deadline = 10
    provider.llm.batches = SimpleNamespace(
        create=AsyncMock(return_value=make_batch("in_progress")),
        retrieve=AsyncMock(side_effect=error),
        cancel=AsyncMock(),
    )
    with pytest.raises(TimeoutError) as failure:
        await provider.create_batch_task(SimpleNamespace(id="file"))
    assert failure.value is error


@pytest.mark.asyncio
async def test_ambiguous_batch_creation_does_not_create_duplicate_remote_work():
    provider = file_provider(make_batch())
    provider.llm.batches = SimpleNamespace(
        create=AsyncMock(side_effect=RetryableAgentError("connection lost"))
    )
    with pytest.raises(RetryableAgentError):
        await provider.create_batch_task(SimpleNamespace(id="file"))
    provider.llm.batches.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_context_override_is_supported_and_remote_unchanged_context_is_not_replayed():
    provider = prompt_provider(
        RequestBudget(
            max_input_tokens=300,
            token_counter=len,
            safety_margin=0,
            output_reservation=0,
        )
    )
    agent = SimpleAgent(provider, batch="one")
    agent.rebuild_context = lambda messages, error: [
        {"role": "user", "content": "compact"}
    ]
    await provider.prompt_agent(agent, [{"role": "user", "content": "x" * 500}])
    assert provider.endpoint_fn.await_args.kwargs["messages"][0]["content"] == "compact"
    provider.endpoint_fn.reset_mock()
    provider.endpoint_fn.side_effect = ContextLengthExceededError("remote overflow")
    agent.rebuild_context = lambda messages, error: messages
    with pytest.raises(ContextLengthExceededError):
        await provider.prompt_agent(agent, [{"role": "user", "content": "small"}])
    provider.endpoint_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_override_mutating_rebuilt_messages_preserves_the_original_audit():
    provider = prompt_provider()
    response = provider.endpoint_fn.return_value
    provider.endpoint_fn.side_effect = [
        ContextLengthExceededError("remote overflow"),
        response,
    ]
    agent = SimpleAgent(provider, batch="one")
    original = [{"role": "user", "content": "full original evidence"}]

    def rebuild(messages, error):
        messages[0]["content"] = "compact"
        return messages

    agent.rebuild_context = rebuild
    await provider.prompt_agent(agent, original)
    assert original[0]["content"] == "full original evidence"
    assert (
        agent.request_history[0]["messages"][0]["content"] == "full original evidence"
    )
    assert agent.request_history[1]["messages"][0]["content"] == "compact"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 1.5},
        {"max_attempts": True},
        {"initial_delay": float("nan")},
        {"max_delay": float("inf")},
        {"elapsed_time_budget": float("inf")},
    ],
)
def test_invalid_retry_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_tokenizer_lookup_never_downloads_assets(monkeypatch):
    from agents.budgets import _encoding

    package = ModuleType("tiktoken")
    registry = ModuleType("tiktoken.registry")
    encoding = object()
    registry.ENCODINGS = {"loaded-local-test": encoding}

    def no_download(*args, **kwargs):
        pytest.fail("Preflight attempted to download tokenizer assets")

    package.get_encoding = no_download
    monkeypatch.setitem(sys.modules, "tiktoken", package)
    monkeypatch.setitem(sys.modules, "tiktoken.registry", registry)
    _encoding.cache_clear()
    try:
        assert _encoding("cold-local-test") is None
        assert _encoding("loaded-local-test") is encoding
    finally:
        _encoding.cache_clear()
