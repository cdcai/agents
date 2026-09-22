from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockFixture

import agents
from agents.abstract import _Provider
from agents.providers import AzureOpenAIProvider
from agents.providers.openai import OpenAIToolCall


class DummyAgent(agents.Agent):
    @agents.agent_callable(
        "A tool the language agent can use",
        {"a": "a variable", "b": "another variable"},
    )
    def blah(self, a: str, b: int) -> str:
        """
        A tool the language agent can use
        """
        return ""


class DummyAgentWithCondition(agents.Agent):
    @agents.agent_callable(
        "A tool the language agent can use",
        variable_description={},
        condition=lambda agent: agent.curr_step >= 2,
    )
    def fizz(self) -> str:
        """
        A tool the language agent can use after step 1
        """
        return "Congrats."


class AsyncDummyAgentWithCondition(agents.Agent):
    @agents.async_agent_callable(
        "An asynchronous conditional tool",
        variable_description={"value": "A value to return"},
        condition=lambda agent: agent.curr_step >= 2,
    )
    async def async_fizz(self, value: str) -> str:
        return value


class AsyncDummyAgent(DummyAgent):
    @agents.async_agent_callable(
        "A function named blech", {"d": "A variable of the letter d"}
    )
    async def blech(self, d: float) -> str:
        return ""


class SteppedToolAgent(agents.Agent):
    SYSTEM_PROMPT = "Test system prompt"
    BASE_PROMPT = "Test prompt"
    tool_enabled: bool = False


def test_json_payload_from_annotations(mocker: MockFixture) -> None:
    """
    Testing that decorated methods correctly
    generate tools that can be used by an agent
    """
    _provider = mocker.Mock(spec=AzureOpenAIProvider)

    my_dummy = DummyAgent(agents.StopNoOp(), provider=_provider)

    assert len(my_dummy.TOOLS) == 1, "Tool length is off!"
    assert set(my_dummy._known_tools) == {"blah"}, (
        f"Only found tools:{my_dummy._known_tools}"
    )


def test_json_payload_from_async_annotations(mocker: MockFixture) -> None:
    """
    Testing that decorated async methods correctly
    generate tools that can be used by an agent
    """
    _provider = mocker.Mock(spec=AzureOpenAIProvider)

    my_dummy = AsyncDummyAgent(agents.StopNoOp(), provider=_provider)

    assert len(my_dummy.TOOLS) == 2, "Tool length is off!"
    assert set(my_dummy._known_tools) == {
        "blah",
        "blech",
    }, f"Only found tools:{my_dummy._known_tools}"


@pytest.mark.asyncio
async def test_sync_decorator_condition_controls_availability(
    mocker: MockFixture,
) -> None:
    provider = mocker.Mock(spec=AzureOpenAIProvider)
    agent = DummyAgentWithCondition(agents.StopNoOp(), provider=provider)

    assert len(agent.TOOLS) == 1
    assert agent._known_tools == []

    agent.curr_step = 2

    assert agent._known_tools == ["fizz"]
    assert await agent.TOOLS[0].invoke() == "Congrats."


@pytest.mark.asyncio
async def test_async_decorator_condition_controls_availability(
    mocker: MockFixture,
) -> None:
    provider = mocker.Mock(spec=AzureOpenAIProvider)
    agent = AsyncDummyAgentWithCondition(agents.StopNoOp(), provider=provider)

    assert len(agent.TOOLS) == 1
    assert agent._known_tools == []

    agent.curr_step = 2

    assert agent._known_tools == ["async_fizz"]
    assert await agent.TOOLS[0].invoke(value="available") == "available"


def test_explicit_tool_condition_cannot_override_decorator_condition(
    mocker: MockFixture,
) -> None:
    provider = mocker.Mock(spec=AzureOpenAIProvider)
    agent = DummyAgentWithCondition(agents.StopNoOp(), provider=provider)
    decorated_call = agent.TOOLS[0].call

    # We shouldn't be able to over-ride the decorator, this would introduce
    # another point of failure I don't want to deal with
    with pytest.raises(match="condition cannot be overwritten"):
        tool: agents.Tool[DummyAgentWithCondition] = agents.Tool(
            call=decorated_call,
            condition=lambda _agent: True,
        )


@pytest.mark.asyncio
async def test_condition_controls_tool_definitions_for_each_step() -> None:
    def second_step_tool() -> str:
        return "available"

    tool: agents.Tool[SteppedToolAgent] = agents.Tool(
        call=second_step_tool,
        description="A tool available from the second step onward",
        variable_description={},
        condition=lambda agent: agent.curr_step >= 2,
    )
    response = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content="continue", tool_calls=None),
    )
    prompt_mock = AsyncMock(return_value=response)
    provider = cast(
        _Provider[Any],
        SimpleNamespace(
            prompt_agent=prompt_mock,
            tool_call_wrapper=OpenAIToolCall,
        ),
    )
    agent = SteppedToolAgent(
        agents.StopOnStep(2),
        provider=provider,
        tools=[tool],
    )

    await agent.step()
    await agent.step()

    first_kwargs = prompt_mock.await_args_list[0].kwargs
    second_kwargs = prompt_mock.await_args_list[1].kwargs
    assert "tools" not in first_kwargs
    assert second_kwargs["tools"] == [tool.json_payload]
    assert isinstance(second_kwargs["tools"][0], dict)


@pytest.mark.asyncio
async def test_tool_call_uses_availability_snapshot() -> None:
    calls: list[str] = []

    def stateful_tool(value: str) -> str:
        calls.append(value)
        return f"handled {value}"

    tool: agents.Tool[SteppedToolAgent] = agents.Tool(
        call=stateful_tool,
        description="Handle a value",
        variable_description={"value": "The value to handle"},
        condition=lambda agent: agent.tool_enabled,
    )
    raw_tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="stateful_tool",
            arguments='{"value": "request"}',
        ),
    )
    response = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content="", tool_calls=[raw_tool_call]),
    )

    async def prompt_agent(agent, prompt, **kwargs):
        assert kwargs["tools"] == [tool.json_payload]
        agent.tool_enabled = False
        return response

    provider = cast(
        _Provider[Any],
        SimpleNamespace(
            prompt_agent=AsyncMock(side_effect=prompt_agent),
            tool_call_wrapper=OpenAIToolCall,
        ),
    )
    agent = SteppedToolAgent(
        agents.StopOnStep(1),
        provider=provider,
        tools=[tool],
    )
    agent.tool_enabled = True

    await agent.step()

    assert calls == ["request"]
    assert agent.tool_res_payload[-1] == {
        "tool_call_id": "call-1",
        "role": "tool",
        "content": "handled request",
    }
    assert agent._known_tools == []


def test_duplicate_tool_names_are_rejected() -> None:
    def repeated() -> str:
        return ""

    first: agents.Tool[Any] = agents.Tool(
        call=repeated,
        description="First",
        variable_description={},
    )
    second: agents.Tool[Any] = agents.Tool(
        call=repeated,
        description="Second",
        variable_description={},
    )
    provider = cast(_Provider[Any], SimpleNamespace())

    with pytest.raises(ValueError, match="Duplicate tool name"):
        SteppedToolAgent(
            agents.StopNoOp(),
            provider=provider,
            tools=[first, second],
        )


def test_class_level_tools_are_rejected() -> None:
    class ClassLevelToolAgent(SteppedToolAgent):
        pass

    ClassLevelToolAgent.TOOLS = []  # type: ignore[misc]

    provider = cast(_Provider[Any], SimpleNamespace())

    with pytest.raises(TypeError, match="TOOLS is no longer supported"):
        ClassLevelToolAgent(agents.StopNoOp(), provider=provider)
