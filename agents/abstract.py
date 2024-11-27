"""
All abstract classes
"""
import abc
import logging
import os
from typing import Callable, List, Optional, Union, Any

import openai
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage

logger = logging.getLogger(__name__)

class _StoppingCondition(metaclass=abc.ABCMeta):
    """
    A callable that contains some logic to determine whether a language agent
    has finished it's run.

    The main call should always return the final answer, if we've finished the run, or None otherwise
    """
    @abc.abstractmethod
    def __call__(self, cls: '_Agent', response: str) -> Optional[Any]:
        raise NotImplementedError()

class _Agent(metaclass=abc.ABCMeta):
    terminated: bool = False
    truncated: bool = False
    curr_step: int = 1
    output_len: int = 1
    scratchpad: str = ""
    answer: Any = ""
    BASE_PROMPT: str = ""
    SYSTEM_PROMPT: str = ""
    oai_kwargs : dict
    TOOLS : list
    CALLBACKS : list
    callback_output: list
    tool_res_payload: list[dict]

    def __init__(
        self,
        model_name: str,
        stopping_condition: _StoppingCondition,
        llm: Optional[openai.AsyncOpenAI] = None,
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Callable]] = None,
        oai_kwargs: Optional[dict[str, any]] = None,
        **fmt_kwargs
    ):
        pass
    
    @abc.abstractmethod
    async def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: Optional[int] = None, **addn_oai_kwargs) -> Choice:
        raise NotImplementedError()

    @abc.abstractmethod
    async def step(self):
        """
        Run a single "step" of the agent logic.
        Handles prompting OpenAI, optionally handling tool calls, and determining whether we've
        finished or run out of tokens.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _check_stop_condition(self, response: ChatCompletionMessage) -> None:
        """
        Called from within :func:`step()`.
        Checks whether our stop condition has been met and handles assignment of answer, if so.

        It's broken out this way because we may not always want to use message.content as the answer (tool call output, for instance)
        """

    @abc.abstractmethod
    def format_prompt(self, **kwargs) -> str:
        """
        Method which formats the BASE_PROMPT string, possibly inserting additional content.
        This is usually called within get_next_message() to populate the first user message.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_messages(self) -> list[dict[str, str]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _handle_tool_calls(self, response: Choice) -> None:
        raise NotImplementedError()

    @property
    def is_terminated(self) -> bool:
        return self.terminated

    @property
    def is_truncated(self) -> bool:
        return self.truncated

    @abc.abstractmethod
    def dump(self, outfile: Union[str, os.PathLike]) -> None:
        raise NotImplementedError()

    @staticmethod
    def clean_response(res: str) -> str:
        out = res.strip('\n').strip().replace('\n', '')
        return out

    @staticmethod
    @abc.abstractmethod
    def authenticate() -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()