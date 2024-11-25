"""
All abstract classes
"""
import abc
import logging
import os
from typing import Callable, List, Optional, Union

import openai
from openai.types.chat.chat_completion import Choice

logger = logging.getLogger(__name__)

class _StoppingCondition(metaclass=abc.ABCMeta):
    """
    A callable that contains some logic to determine whether a language agent
    has finished it's run.

    The main call should always return a bool, which will be used to signal
    termination in the Agent class.
    """
    @abc.abstractmethod
    def __call__(self, cls: '_Agent', response: str) -> bool:
        raise NotImplementedError()

class _Agent(metaclass=abc.ABCMeta):
    terminated: bool = False
    truncated: bool = False
    curr_step: int = 1
    output_len: int = 1
    scratchpad: str = ""
    answer: str = ""
    BASE_PROMPT: str = ""
    SYSTEM_PROMPT: str = ""
    oai_kwargs : dict
    TOOLS : list
    CALLBACKS : list
    callback_output: list

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
    async def step():
        """
        Run a single "step" of the agent logic.
        Handles prompting OpenAI, optionally handling tool calls, and determining whether we've
        finished or run out of tokens.
        """
        raise NotImplementedError()

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
    def reset():
        raise NotImplementedError()