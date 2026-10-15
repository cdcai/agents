import asyncio
import copy
from dataclasses import replace

import pytest

from agents import Agent, StopOnStep, StructuredOutputAgent
from agents.abstract import Callback, CallbackCondition
from agents.budgets import (
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    RequestBudget,
    RequestBudgetExceededError,
    check_request_budget,
    estimate_request,
    resolve_request_budget,
)


def budget(**kwargs):
    return RequestBudget(
        token_counter=len, output_reservation=0, safety_margin=0, **kwargs
    )


def test_cutoff_is_configurable_guess_with_no_alias_inference():
    result = resolve_request_budget("unknown-private-deployment")
    assert result.max_input_tokens == 272_000
    assert result.context_window is None
    assert result.model is None
    assert any("fallback" in source for source in result.provenance)
    assert resolve_request_budget("gpt5.1-batch-dgw-default").context_window is None


def test_metadata_is_optional_and_explicit_deployment_caps_win():
    result = resolve_request_budget(
        "custom-deployment",
        model="gpt-5.1",
        model_version="2025-11-13",
        metadata={
            "capabilities": {
                "context_window": 400_000,
                "max_input_tokens": 399_000,
                "tpm": 10,
            }
        },
        overrides={"max_input_tokens": 272_000, "max_output_tokens": 128_000},
    )
    assert result.model == "gpt-5.1"
    assert result.model_version == "2025-11-13"
    assert result.input_limit == 270_976
    assert result.context_window == 400_000
    assert not any("fallback" in source for source in result.provenance)
    no_capabilities = resolve_request_budget(
        "alias", metadata={"capabilities": {"tpm": 999}}
    )
    assert no_capabilities.max_input_tokens == 272_000
    malformed = resolve_request_budget(
        "alias", metadata={"capabilities": {"context_window": "400000"}}
    )
    assert malformed.context_window is None


def test_input_context_and_output_constraints_are_independent():
    configured = RequestBudget(
        context_window=1_000,
        max_input_tokens=900,
        max_output_tokens=500,
        output_reservation=200,
        reasoning_reservation=100,
        safety_margin=50,
    )
    assert configured.input_limit == 650
    result = estimate_request(
        {"messages": [], "max_completion_tokens": 400}, configured
    )
    assert result.input_limit == 550
    assert result.output_tokens == 400  # Already includes reasoning.
    with pytest.raises(RequestBudgetExceededError) as failure:
        check_request_budget({"messages": [], "max_completion_tokens": 501}, configured)
    assert failure.value.reason == "output"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_input_tokens": 0},
        {"safety_margin": -1},
        {"context_window": 100, "output_reservation": 100},
        {"max_output_tokens": 10, "output_reservation": 20},
        {"max_input_tokens": None, "context_window": None},
    ],
)
def test_invalid_configuration_fails_before_requests(kwargs):
    with pytest.raises(ValueError):
        RequestBudget(**kwargs)


def test_exact_estimated_boundary_and_structured_component_failure():
    request = {"messages": [{"role": "user", "content": "a"}]}
    original = budget(max_input_tokens=10_000)
    estimate = estimate_request(request, original)
    exact = replace(original, max_input_tokens=estimate.input_tokens)
    assert check_request_budget(request, exact).input_tokens == exact.input_limit
    with pytest.raises(RequestBudgetExceededError) as failure:
        check_request_budget(
            request, replace(exact, max_input_tokens=exact.input_limit - 1)
        )
    assert failure.value.estimate.is_estimate is True
    assert failure.value.details["components"]["user"] > 0
    assert failure.value.details["is_estimate"] is True
    assert failure.value.retryable is False


def test_history_tools_schema_and_framing_are_counted_after_initial_valid_input():
    initial = {"messages": [{"role": "system", "content": "instructions"}]}
    configured = budget(max_input_tokens=500)
    assert check_request_budget(initial, configured).input_tokens < 500
    expanded = copy.deepcopy(initial)
    expanded["messages"].extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "x", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "x", "content": "result " * 100},
        ]
    )
    expanded["tools"] = [
        {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object"}},
        }
    ]
    expanded["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "answer"},
    }
    with pytest.raises(RequestBudgetExceededError) as failure:
        check_request_budget(expanded, configured)
    for key in ("history", "tools", "schema", "framing"):
        assert failure.value.estimate.components[key] > 0


def test_unavailable_tokenizer_uses_labeled_utf8_fallback(monkeypatch):
    monkeypatch.setattr("agents.budgets._encoding", lambda _: None)
    result = estimate_request(
        {"messages": [{"role": "user", "content": "症状"}]}, RequestBudget()
    )
    assert "UTF-8" in result.estimator
    assert result.is_estimate


def test_multimodal_requests_fail_explicitly():
    with pytest.raises(ValueError, match="Multimodal"):
        estimate_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    }
                ]
            },
            budget(),
        )


@pytest.mark.asyncio
async def test_context_hook_and_audit_keep_original_history():
    original = [{"role": "tool", "content": "full evidence", "tool_call_id": "one"}]
    seen = []

    async def rebuild(agent, messages, error):
        seen.append(error)
        messages[0]["content"] = "compact evidence"
        return messages

    agent = Agent(StopOnStep(), provider=object(), context_rebuild_hook=rebuild)
    agent.tool_res_payload = copy.deepcopy(original)
    agent.scratchpad = "full audit evidence"
    agent.record_request({"messages": original})
    error = RequestBudgetExceededError(
        budget(), estimate_request({"messages": original}, budget())
    )
    result = await agent.rebuild_context(original, error)
    assert result[0]["content"] == "compact evidence"
    original[0]["content"] = "subsequent mutation"
    assert agent.request_history[0]["messages"][0]["content"] == "full evidence"
    assert agent.tool_res_payload[0]["content"] == "full evidence"
    assert agent.scratchpad == "full audit evidence"
    assert seen == [error]


def test_deadline_fake_clock_and_validation():
    now = [10.0]
    deadline = ExecutionDeadline(3, clock=lambda: now[0])
    assert deadline.remaining == 3
    now[0] = 14
    with pytest.raises(ExecutionDeadlineExceeded) as failure:
        deadline.check()
    assert failure.value.details["remote_work_may_continue"] is True
    assert deadline.remaining == 0
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ExecutionDeadline(value)


@pytest.mark.asyncio
async def test_agent_deadline_cancels_only_owned_work_and_dispatches_terminal_error():
    class RecordingCallback(Callback):
        condition = CallbackCondition.ON_ERROR

        async def __call__(self, agent, error):
            agent.callback_output.append(error)

    class StalledAgent(Agent):
        async def step(self):
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned_up = True

    agent = StalledAgent(
        StopOnStep(),
        provider=object(),
        elapsed_time_budget=30,
        callbacks=[RecordingCallback()],
    )
    # Fake an already expired monotonic deadline: no wall-clock timeout sleep.
    now = [0.0]
    agent.execution_deadline = ExecutionDeadline(1, clock=lambda: now[0])
    now[0] = 2
    with pytest.raises(ExecutionDeadlineExceeded):
        await agent.run()
    assert isinstance(agent.callback_output[0], ExecutionDeadlineExceeded)
    # A live, stalled operation is cancelled and awaited by asyncio.timeout.
    agent.reset()
    agent.elapsed_time_budget = 0.01
    unrelated = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(ExecutionDeadlineExceeded):
            await agent.run()
        assert agent.cleaned_up
        assert not unrelated.cancelled()
    finally:
        unrelated.cancel()
        await asyncio.gather(unrelated, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_timeout_is_not_misclassified_as_agent_deadline():
    class FailedAgent(Agent):
        async def step(self):
            raise TimeoutError("HTTP request timed out")

    agent = FailedAgent(StopOnStep(), provider=object(), elapsed_time_budget=30)
    with pytest.raises(TimeoutError, match="HTTP request"):
        await agent.run()


def test_structured_agent_reset_initializes_deadline_and_request_history():
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    agent = StructuredOutputAgent(Answer, provider=object(), elapsed_time_budget=10)
    assert agent.execution_deadline is None
    assert agent.request_history == []
    assert agent.answer == {}
