from typing import Any, assert_type, cast

import pytest
from pydantic import BaseModel

from agents import (
    Agent,
    AgentCallback,
    PredictionAgent,
    StopOnStep,
    StructuredOutputAgent,
)
from agents.abstract import Callback, CallbackCondition, _Provider


class CallingAgent(Agent):
    def __init__(self) -> None:
        self.callback_output: list[Any] = []
        self.CALLBACKS = []
        self.terminated = True


class CallbackAgent(Agent):
    def __init__(self, *, answer: Any, scratchpad: str) -> None:
        self.input_answer = answer
        self.input_scratchpad = scratchpad
        self.answer = ""

    async def run(self, reset: bool = False, *kwargs: Any) -> None:
        self.answer = "reviewed"


@pytest.mark.asyncio
async def test_agent_callback_preserves_spawned_agent_type() -> None:
    callback = AgentCallback(CallbackAgent)
    calling_agent = CallingAgent()

    calling_agent.answer = {"result": 42}
    calling_agent.scratchpad = "work"

    await callback(calling_agent, None)

    assert_type(callback, AgentCallback[CallbackAgent])
    assert_type(callback.callback_agent, CallbackAgent)
    assert callback.callback_agent.input_answer == {"result": 42}
    assert callback.callback_agent.input_scratchpad == "work"
    assert calling_agent.callback_output == ["reviewed"]


@pytest.mark.asyncio
async def test_completed_agent_stores_callback_exception() -> None:
    callback_error = RuntimeError("callback failed")

    class FailingCallback(Callback[Agent]):
        async def __call__(self, cls: Agent, exc: Exception | None) -> None:
            raise callback_error

    calling_agent = CallingAgent()
    calling_agent.CALLBACKS = [FailingCallback()]
    calling_agent.answer = "completed answer"
    calling_agent.scratchpad = "completed work"
    calling_agent.terminated = True

    await calling_agent.run()

    assert calling_agent.callback_output == [callback_error]


class RecordingCallback(Callback[Agent]):
    def __init__(
        self, condition: CallbackCondition, error: Exception | None = None
    ) -> None:
        self.condition = condition
        self.error = error
        self.calls: list[tuple[Agent, Exception | None]] = []

    async def __call__(self, cls: Agent, exc: Exception | None) -> None:
        self.calls.append((cls, exc))
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_run_dispatches_callbacks_by_condition(fails: bool) -> None:
    run_error = RuntimeError("agent failed")

    class SteppingAgent(CallingAgent):
        async def step(self) -> None:
            if fails:
                raise run_error
            self.terminated = True

    agent = SteppingAgent()
    agent.terminated = False
    callbacks = [RecordingCallback(condition) for condition in CallbackCondition]
    agent.CALLBACKS.extend(callbacks)

    if fails:
        with pytest.raises(RuntimeError) as raised:
            await agent.run()
        assert raised.value is run_error
    else:
        await agent.run()

    outcome = CallbackCondition.ON_ERROR if fails else CallbackCondition.ON_SUCCESS
    for callback in callbacks:
        expected = (
            [(agent, run_error if fails else None)]
            if callback.condition in (CallbackCondition.ALWAYS, outcome)
            else []
        )
        assert callback.calls == expected


@pytest.mark.asyncio
async def test_callback_failure_preserves_run_error_and_remaining_callbacks() -> None:
    run_error = RuntimeError("agent failed")
    callback_error = ValueError("callback failed")

    class FailingAgent(CallingAgent):
        async def step(self) -> None:
            raise run_error

    agent = FailingAgent()
    agent.terminated = False
    failing = RecordingCallback(CallbackCondition.ON_ERROR, callback_error)
    remaining = RecordingCallback(CallbackCondition.ALWAYS)
    agent.CALLBACKS = [failing, remaining]

    with pytest.raises(RuntimeError) as raised:
        await agent.run()

    assert raised.value is run_error
    assert agent.callback_output == [callback_error]
    assert failing.calls == [(agent, run_error)]
    assert remaining.calls == [(agent, run_error)]


@pytest.mark.asyncio
async def test_agent_hierarchy_accepts_callback_sequences() -> None:
    class Response(BaseModel):
        answer: str

    provider = cast(_Provider[Any], object())
    callback = RecordingCallback(CallbackCondition.ON_SUCCESS)
    callbacks = (callback,)
    agents = [
        Agent(StopOnStep(), provider=provider, callbacks=callbacks),
        StructuredOutputAgent(Response, provider=provider, callbacks=callbacks),
        PredictionAgent(["yes", "no"], provider=provider, callbacks=callbacks),
    ]

    for agent in agents:
        agent.terminated = True
        await agent.run()

    assert callback.calls == [(agent, None) for agent in agents]
