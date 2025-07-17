"""
Test experimental observability classes
"""
import openai
import pydantic
import pytest
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pytest_mock import MockFixture

import agents.observability as agents
from agents import StopNoOp

# A mock OpenAI response
mock_completion = ChatCompletion(
    id="fake_message",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                role="assistant", content="Lorem ipsum, dolor sit amet."
            ),
        )
    ],
    model="gpt-4o",
    created=1234,
    object="chat.completion",
    usage=CompletionUsage(completion_tokens=100, prompt_tokens=7, total_tokens=107),
)


async def dummy_endpoint(messages, **kwargs):
    return mock_completion


@pytest.mark.asyncio
async def test_token_and_turn_tracking(mocker: MockFixture):
    """
    Testing observable classes track tokens and turns correctly
    """
    openai.AsyncAzureOpenAI = mocker.Mock(spec=openai.AsyncAzureOpenAI)
    mocker.patch.object(
        agents.AzureOpenAIProvider,
        "authenticate",
        return_value=None
    )

    prov = agents.AzureOpenAIProvider(
        "super_cool_model", interactive=True
    )

    prov.endpoint_fn = prov.round_trip_increment(dummy_endpoint)
    prov.model_name = "super_cool_model"

    class DummyObservableAgent(agents.Agent):
        BASE_PROMPT = "A dummy prompt"

    ag1 = DummyObservableAgent(StopNoOp(), provider=prov)
    ag2 = DummyObservableAgent(StopNoOp(), provider=prov)

    # One step of first agent
    await ag1.step()

    # Check that token and turn count updated in provider and agent
    expected_counts = agents.LLMUsage(
        input_tok=7, output_tok=100, total_tok=107, round_trips=1
    )
    assert (
        expected_counts == ag1.usage
    ), f"Observable agent didn't correctly track usage! got: {ag1.usage}"
    assert (
        expected_counts == prov.usage
    ), f"Observable provider didn't correctly track usage! got: {prov.usage}"

    # Run one step of another agent
    await ag2.step()

    # Check that usage of first agent stayed the same and that provider now reflects both calls
    expected_counts2 = agents.LLMUsage(
        *map(lambda x: x * 2, expected_counts)
    )
    assert (
        expected_counts == ag1.usage
    ), f"Observable agent should not have been modified! got: {ag1.usage}"
    assert (
        expected_counts == ag2.usage
    ), f"Observable agent didn't correctly track usage! got: {ag2.usage}"
    assert (
        expected_counts2 == prov.usage
    ), f"Observable provider didn't correctly track usage! got: {prov.usage}"


def test_subclassing_observable(mocker: MockFixture):
    """
    Testing that one can subclass ObservableAgents
    """
    _provider = mocker.Mock(spec=agents.AzureOpenAIProvider)

    class ObservableStructuredOutputAgent(
        agents.StructuredOutputAgent, agents.Agent
    ):
        BASE_PROMPT = "Lorem ipsum"

    class Return(pydantic.BaseModel):
        a: int
        b: int

    ag = ObservableStructuredOutputAgent(Return, provider=_provider)

    assert ag.usage == agents.LLMUsage(
        input_tok=0, output_tok=0, total_tok=0, round_trips=0
    )
    assert ag.round_trips == 0
