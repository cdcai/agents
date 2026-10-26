"""Typed tool history stays local; both transports receive JSON text content."""

import asyncio
import json
from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agents.budgets import RequestBudget
from agents.providers.openai import AzureOpenAIBatchProvider, AzureOpenAIProvider


class Box(IntEnum):
    SYMPTOMS = 7


class ToolResult(BaseModel):
    box: Box = Box.SYMPTOMS
    text: str = "Evidence"


def response():
    return ChatCompletion(
        id="completion",
        created=0,
        model="test",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "next",
                            "type": "function",
                            "function": {"name": "finish", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    )


def provider_for(mode, budget):
    cls = AzureOpenAIBatchProvider if mode == "batch" else AzureOpenAIProvider
    provider = object.__new__(cls)
    provider.model_name = "test"
    provider.all_usage = []
    provider._configure_execution(budget, False, None, None)
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "batch"])
@pytest.mark.parametrize("with_budget", [False, True])
async def test_typed_results_are_rendered_before_budget_and_transport(
    mode, with_budget
):
    configured = (
        RequestBudget(token_counter=len, max_input_tokens=10000)
        if with_budget
        else None
    )
    provider = provider_for(mode, configured)
    result = ToolResult()
    assistant = {"role": "assistant", "content": "Calling tool"}
    tool = {"role": "tool", "tool_call_id": "one", "content": result}
    image = {"role": "user", "content": [{"type": "text", "text": "Context"}]}
    plain_tool = {"role": "tool", "tool_call_id": "two", "content": "Plain text"}
    history = [assistant, tool, image, plain_tool]
    agent = SimpleNamespace(
        tool_res_payload=history,
        all_usage=[],
        scratchpad="",
        record_request=AsyncMock(),
    )
    # record_request is synchronous in real agents.
    recorded = []
    agent.record_request = recorded.append
    sent = []

    async def chat_endpoint(**request):
        sent.append(json.loads(json.dumps(request)))
        return response()

    async def consume_batch():
        task = await provider.batch_q.get()
        sent.append(json.loads(provider._serialize_request(task))["body"])
        provider.batch_out[task["custom_id"]].set_result(response())

    if mode == "batch":
        provider.batch_idx_lock = asyncio.Lock()
        provider.batch_idx = 0
        provider.batch_q = asyncio.Queue()
        provider.batch_out = {}
        provider.batch_handler = SimpleNamespace(_closed=False)
        provider.endpoint_fn = provider.query_batch_mode
        worker = asyncio.create_task(consume_batch())
    else:
        provider.endpoint_fn = chat_endpoint

    await provider.prompt_agent(agent, list(history))
    if mode == "batch":
        await worker

    assert json.loads(sent[0]["messages"][1]["content"]) == {
        "box": 7,
        "text": "Evidence",
    }
    assert (
        ToolResult.model_validate_json(sent[0]["messages"][1]["content"]).box
        is Box.SYMPTOMS
    )
    assert recorded[0]["messages"][1]["content"] == result.model_dump_json()
    assert history[1] is tool and tool["content"] is result
    assert sent[0]["messages"][0] == assistant
    assert sent[0]["messages"][2] == image
    assert sent[0]["messages"][3] == plain_tool
    # Assistant tool-call messages remain dictionaries, usable by stopping conditions.
    assert isinstance(history[-1], dict)
    assert history[-1]["role"] == "assistant"
    if with_budget:
        assert agent.last_request_estimate.input_tokens > 0


@pytest.mark.asyncio
async def test_rebuilt_context_is_normalized_before_second_budget_check():
    configured = RequestBudget(
        token_counter=len,
        max_input_tokens=1000,
        output_reservation=0,
        safety_margin=0,
    )
    provider = provider_for("chat", configured)
    result = ToolResult()
    rebuilt = [{"role": "tool", "tool_call_id": "one", "content": result}]
    agent = SimpleNamespace(rebuild_context=AsyncMock(return_value=rebuilt))
    prepared = await provider._prepare_request(
        agent, [{"role": "user", "content": "x" * 2000}], {}
    )
    agent.rebuild_context.assert_awaited_once()
    assert prepared[0]["content"] == result.model_dump_json()
    assert rebuilt[0]["content"] is result


@pytest.mark.asyncio
async def test_unsupported_tool_values_are_not_silently_stringified():
    provider = provider_for("chat", None)
    prepared = await provider._prepare_request(
        SimpleNamespace(),
        [{"role": "tool", "tool_call_id": "one", "content": object()}],
        {},
    )
    with pytest.raises(TypeError):
        AzureOpenAIBatchProvider._serialize_request({"body": {"messages": prepared}})
