import pytest
from pytest_mock import MockFixture
import agents

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


class AsyncDummyAgent(DummyAgent):
    @agents.async_agent_callable(
        "A function named blech", {"d": "A variable of the letter d"}
    )
    async def blech(self, d: float) -> str:
        return ""


def test_json_payload_from_annotations(mocker: MockFixture) -> None:
    """
    Testing that decorated methods correctly
    generate tools that can be used by an agent
    """
    _provider = mocker.Mock(spec=agents.AzureOpenAIProvider)

    my_dummy = DummyAgent(agents.StopNoOp(), provider=_provider)

    assert len(my_dummy.TOOLS) == 1, "Tool length is off!"
    assert set(my_dummy._known_tools) == {
        "blah"
    }, f"Only found tools:{my_dummy._known_tools}"


def test_json_payload_from_async_annotations(mocker: MockFixture) -> None:
    """
    Testing that decorated async methods correctly
    generate tools that can be used by an agent
    """
    _provider = mocker.Mock(spec=agents.AzureOpenAIProvider)

    my_dummy = AsyncDummyAgent(agents.StopNoOp(), provider=_provider)

    assert len(my_dummy.TOOLS) == 2, "Tool length is off!"
    assert set(my_dummy._known_tools) == {
        "blah",
        "blech",
    }, f"Only found tools:{my_dummy._known_tools}"
