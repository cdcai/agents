from typing import Any, cast, get_args

from agents import (
    Agent,
    BatchProcessorDF,
    BatchProcessorIterable,
    ProcessorDF,
    ProcessorIterable,
    StopOnStep,
)
from agents.abstract import _Provider


class DummyAgent(Agent):
    def __init__(self, provider: _Provider[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(StopOnStep(), provider=provider, **kwargs)


class TypedIterableProcessor(ProcessorIterable[DummyAgent]):
    pass


def test_public_processors_accept_agent_type_parameter() -> None:
    specialized_processors = (
        ProcessorIterable[DummyAgent],
        ProcessorDF[DummyAgent],
        BatchProcessorIterable[DummyAgent],
        BatchProcessorDF[DummyAgent],
    )

    for specialized_processor in specialized_processors:
        assert get_args(specialized_processor) == (DummyAgent,)


def test_typed_processor_preserves_agent_class() -> None:
    provider = cast(_Provider[DummyAgent], object())
    processor = TypedIterableProcessor(["one"], DummyAgent, provider)

    agent = processor._spawn_agent(["one"])

    assert type(agent) is DummyAgent
    assert processor.agent_class is DummyAgent
