"""
Base observable classes which track token usage and round trips
"""

import logging
from collections import namedtuple
from collections.abc import Awaitable, Callable
from functools import reduce, wraps
from typing import (
    Any,
)

from openai.types import CompletionUsage

logger = logging.getLogger(__name__)

"""
LLM usage by agent / provider

- input_tok : Tokens sent (prompt + tool)
- output_tok : Tokens in return messages
- total_tok : input_tok + output_tok
- round_trips : How many total queries were sent?
"""
LLMUsage = namedtuple(
    "LLMUsage",
    ["input_tok", "output_tok", "total_tok", "round_trips"],
    defaults=[0, 0, 0, 0],
)


# Simple oneliner to sum up LLMUsage
def sum_llm_usage(x: LLMUsage, y: LLMUsage) -> LLMUsage:
    return LLMUsage(*[i + j for i, j in zip(x, y)])


class Observable[CompletionT: CompletionUsage | LLMUsage]:
    """
    Observability methods and attributes needed for the provider and agent to track
    token and round-trip usage
    """

    all_usage: list[LLMUsage]
    round_trips: int

    def __init__(self, **kwargs):
        self.all_usage = []
        self.round_trips = 0

    @staticmethod
    def usage_adapter(usage: CompletionT | None) -> LLMUsage:
        """
        A method which should be extended to adapt usage for provider-specific token usage types
        """
        if not isinstance(usage, LLMUsage):
            raise TypeError(
                f"Observable class could not convert usage of type {type(usage)}"
            )
        return usage

    @property
    def usage(self) -> LLMUsage:
        return reduce(sum_llm_usage, self.all_usage, LLMUsage())

    def round_trip_increment(self, f: Callable[..., Awaitable[Any]]):
        """
        A simple decorator that will wrap the method we will use as a
        proxy for round trips (which will be different depending on whether we're using chat or batch endpoints)
        """

        @wraps(f)
        async def run_and_inc(*args, **kwds):
            out = await f(*args, **kwds)
            self.round_trips += 1
            return out

        return run_and_inc
