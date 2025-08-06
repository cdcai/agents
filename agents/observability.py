"""
Providers and Agents that track token and turn usage
"""

import logging
from collections import namedtuple
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Union,
    Generic,
    Literal,
)

import backoff
import openai
from openai.types import CompletionUsage
from openai.types.batch import Batch
from openai.types.chat import ChatCompletionMessageParam


from .agent import Agent as BaseAgent
from .agent import StructuredOutputAgent as BaseStructuredOutputAgent
from .agent import PredictionAgent as BasePredictionAgent
from .agent import (
    PredictionAgentWithJustification as BasePredictionAgentWithJustification,
)
from .providers import AzureOpenAIBatchProvider as BaseAzureOpenAIBatchProvider
from .providers import AzureOpenAIProvider as BaseAzureOpenAIProvider
from .providers.openai import _AzureProvider as _BaseAzureProvider
from .providers.openai import ProviderMode

logger = logging.getLogger(__name__)

__all__ = [
    "AzureOpenAIProvider",
    "AzureOpenAIBatchProvider",
    "Agent",
    "StructuredOutputAgent",
    "PredictionAgent",
    "PredictionAgentWithJustification",
]

"""
LLM usage by agent / provider

- input_tok : Tokens sent (prompt + tool)
- output_tok : Tokens in return messages
- total_tok : input_tok + output_tok
- round_trips : How many total queries were sent?
"""
LLMUsage = namedtuple(
    "LLMUsage", ["input_tok", "output_tok", "total_tok", "round_trips"]
)


class Observable:
    """
    Observability methods and attributes needed for the provider and agent to track
    token and round-trip usage
    """

    all_usage: List[CompletionUsage]
    round_trips: int

    @property
    def usage(self) -> LLMUsage:
        """
        Get total token usage as reported by OpenAI
        """
        completion = 0
        prompt = 0
        total = 0

        for use in self.all_usage:
            completion += use.completion_tokens
            prompt += use.prompt_tokens
            total += use.total_tokens

        return LLMUsage(
            input_tok=prompt,
            output_tok=completion,
            total_tok=total,
            round_trips=self.round_trips,
        )

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


class _AzureProvider(
    Generic[ProviderMode], _BaseAzureProvider["Agent", ProviderMode], Observable
):
    """
    An Observable Azure OpenAI Provider for language Agents.

    This provider generally assumes you already have all required environment variables
    set correctly, or will provide them as kwargs which will be passed to AsyncAzureOpenAI at init

    Namely:
    - api_version or OPENAI_API_VERSION
    - azure_endpoint or AZURE_OPENAI_ENDPOINT

    AZURE_OPENAI_API_KEY will be assigned via authentication (either by ClientSecret or Interactive AD Auth depending on `interactive`)

    Usage:

    Usage at the provider level is tracked via the `usage` attribute.
    """

    mode: ProviderMode

    def __init__(self, model_name: str, interactive: bool, **kwargs):
        """
        Args:
        :param str model_name: Model name from the deployments list to use
        :param bool interactive: Should authentication use an Interactive AD Login (T), or ClientSecret (F)?
        :param **kwargs: Any additional kw-args for AsyncAzureOpenAI
        """
        super().__init__(model_name, interactive, **kwargs)
        self.all_usage = []
        self.round_trips = 0

    @backoff.on_exception(
        backoff.expo, (openai.APIError, openai.AuthenticationError), max_tries=3
    )
    async def prompt_agent(
        self,
        ag: "Agent",
        prompt: Union[
            List[ChatCompletionMessageParam],
            ChatCompletionMessageParam,
        ],
        **kwargs,
    ):
        """
        An async version of the main OAI prompting logic.

        :param ag: The calling agent class
        :param prompt: Either a dict or a list of dicts representing the message(s) to send to OAI model
        :param kwargs: Key word arguments passed to completions.create() call (tool calls, etc.)

        :return: An openAI Choice response object
        """

        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if not isinstance(prompt, list):
            prompt = [prompt]

        try:
            res = await self.endpoint_fn(
                messages=prompt, model=self.model_name, **kwargs
            )
        except openai.AuthenticationError as e:
            logger.info("Auth failed, attempting to re-authenticate before retrying")
            self.authenticate()
            raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out = res.choices[0]
        # TODO: Fix how we're accomplishing this eventually, it's hacky to overload the whole method
        # just to add two lines of code tracking tokens.
        # One solution would be just changing the return of prompt_agent to the ChatCompletion object
        # instead of Choice, so the token usage object isn't removed and we can choose to handle it
        # or not via a wrapper. That's probably cleaner.
        if isinstance(res.usage, CompletionUsage):
            self.all_usage.append(res.usage)
            ag.all_usage.append(res.usage)

        # HACK: OpenAI API can't handle None in a roundtrip
        # so we have to patch the message content so it doesn't throw an error.
        if out.message.content is None:
            out.message.content = "<None>"
        ag.scratchpad += "--- Output --------------------------\n"
        ag.scratchpad += "Message:\n"
        ag.scratchpad += out.message.content + "\n"

        if len(ag.TOOLS):
            # attempt to parse tool call arguments
            # BUG: OpenAI sometimes doesn't return a "tool_calls" reason and uses "stop" instead. Annoying.
            if out.finish_reason == "tool_calls" or (
                out.finish_reason == "stop"
                and out.message.tool_calls
                and len(out.message.tool_calls)
            ):
                out.finish_reason = "tool_calls"
                # Append GPT response to next payload
                # NOTE: This has to come before the next step of parsing
                ag.tool_res_payload.append(out.message)

        logger.info(f"Received response: {out.message.content}")

        if out.finish_reason == "length":
            ag.truncated = True
            ag.scratchpad += (
                "Response returned truncated from OpenAI due to token length.\n"
            )
            logger.warning("Message returned truncated.")
        ag.scratchpad += "\n-----------------------------------\n"
        return out


class AzureOpenAIProvider(BaseAzureOpenAIProvider, _AzureProvider[Literal["chat"]]):
    mode = "chat"

    def __init__(self, model_name: str, interactive: bool, **kwargs):
        super().__init__(model_name, interactive, **kwargs)
        self.endpoint_fn = self.round_trip_increment(self.endpoint_fn)


class AzureOpenAIBatchProvider(
    BaseAzureOpenAIBatchProvider, _AzureProvider[Literal["batch"]]
):
    mode = "batch"

    async def get_batch_results(self, batch: Batch) -> List[Dict]:
        """
        A patched version of get_batch_results that increments our round-trip count every call
        """
        return await self.round_trip_increment(super().get_batch_results)(batch)


class Agent(BaseAgent, Observable):
    def __init__(
        self,
        stopping_condition,
        model_name=None,
        provider=None,
        tools=None,
        callbacks=None,
        oai_kwargs=None,
        **fmt_kwargs,
    ):
        self.all_usage = []
        self.round_trips = 0
        super().__init__(
            stopping_condition,
            model_name,
            provider,
            tools,
            callbacks,
            oai_kwargs,
            **fmt_kwargs,
        )

    async def step(self):
        return await self.round_trip_increment(super().step)()


class StructuredOutputAgent(BaseStructuredOutputAgent, Agent): ...


class PredictionAgent(BasePredictionAgent, Agent): ...


class PredictionAgentWithJustification(BasePredictionAgentWithJustification, Agent): ...
