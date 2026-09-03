from typing import Any, assert_type

import pytest

from agents import Agent, AgentCallback


class CallingAgent(Agent):
    def __init__(self) -> None:
        self.callback_output: list[Any] = []


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

    await callback(calling_agent, answer={"result": 42}, scratchpad="work")

    assert_type(callback, AgentCallback[CallbackAgent])
    assert_type(callback.callback_agent, CallbackAgent)
    assert callback.callback_agent.input_answer == {"result": 42}
    assert callback.callback_agent.input_scratchpad == "work"
    assert calling_agent.callback_output == ["reviewed"]
