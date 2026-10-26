from typing import Any, cast, get_args

import pytest

from agents import (
    Agent,
    BatchProcessorDF,
    BatchProcessorIterable,
    ProcessorDF,
    ProcessorIterable,
    StopOnStep,
)
from agents.abstract import _Provider
from agents.processors import PROCESSOR_MAX_CONCURRENCY


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


@pytest.mark.parametrize("max_concurrency", [0, -1])
def test_batch_processor_rejects_nonpositive_concurrency(max_concurrency: int) -> None:
    provider = cast(_Provider[DummyAgent], object())

    with pytest.raises(ValueError, match="max_concurrency"):
        BatchProcessorIterable(
            ["one"],
            DummyAgent,
            provider,
            batch_size=1,
            max_concurrency=max_concurrency,
        )


@pytest.mark.parametrize("max_concurrency", [None, 2])
def test_batch_processor_accepts_default_or_positive_concurrency(
    max_concurrency: int | None,
) -> None:
    provider = cast(_Provider[DummyAgent], object())
    processor = BatchProcessorIterable(
        ["one"],
        DummyAgent,
        provider,
        batch_size=1,
        max_concurrency=max_concurrency,
    )

    expected = (
        PROCESSOR_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
    )
    assert processor.max_concurrency == expected
