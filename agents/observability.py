"""
Providers and Agents that track token and turn usage
(experimental)
"""

import logging
from collections import namedtuple
from typing import List, Union

import backoff
import openai
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletion,
    ChatCompletionMessage,
)

from .abstract import _Agent
from .providers import AzureOpenAIProvider
from .agent import Agent

logger = logging.getLogger(__name__)

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


class ObservableAzureOpenAIProvider(AzureOpenAIProvider, Observable):
    def __init__(self, model_name: str, interactive: bool, **kwargs):
        self.all_usage = []
        self.round_trips = 0
        super().__init__(model_name, interactive, **kwargs)

    @backoff.on_exception(
        backoff.expo, (openai.APIError, openai.AuthenticationError), max_tries=3
    )
    async def prompt_agent(
        self,
        ag: _Agent,
        prompt: Union[List[ChatCompletionMessageParam], ChatCompletionMessageParam],
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

        ag.scratchpad += "--- Input ---------------------------\n"
        for msg in prompt:
            if "content" in msg and isinstance(msg["content"], str):
                ag.scratchpad += "\n" + msg["content"]
        ag.scratchpad += "\n-----------------------------------\n"

        try:
            res: ChatCompletion = await self.llm.chat.completions.create(
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

        if isinstance(res.usage, CompletionUsage):
            self.all_usage.append(res.usage)

        ag.scratchpad += "--- Output --------------------------\n"
        ag.scratchpad += "Message:\n"
        ag.scratchpad += out.message.content if out.message.content else "<None>" + "\n"

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
                ag.tool_res_payload.append(out.message.model_dump())

        ag.scratchpad += "\n-----------------------------------\n"
        logger.info(f"Received response: {out.message.content}")

        if out.finish_reason == "length":
            ag.truncated = True
            ag.scratchpad += (
                "Response returned truncated from OpenAI due to token length.\n"
            )
            logger.warning("Message returned truncated.")
        return out


class AgentObservable(Agent, Observable):
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
        self.usage_list = []
        self.turns = 0
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
        out = await super().step()
        self.turns += 1
        return out
