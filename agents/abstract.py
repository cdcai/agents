"""
All abstract classes
"""

import abc
import asyncio
import json
import logging
import os
from asyncio import Task, create_task, to_thread
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.chat_completion import ChatCompletion, Choice

from pydantic import BaseModel, ValidationError

from .observability import Observable

logger = logging.getLogger(__name__)

Message = Union[dict[str, Any], "ChatCompletionMessageParam"]


@dataclass
class _ToolCall[AgentT: _Agent](metaclass=abc.ABCMeta):
    """
    A tool call abstract class which handles validation and execution of requested tool calls
    using asyncio.

    Each provider will require different tool call return formats, and the tool call object itself
    may be structured differently. You'll need to define a few methods for extending:

    - id: A property which should return the tool call id from the tool_call object
    - func_name: A property which should return the function name from the tool_call object as a string
    - arg_str: A property which returns the arguments from the tool_call object that will be parsed as JSON to a dict
    - _construct_return_message(): A function which returns the tool call result as a message payload for the langauge agent

    """

    "Agent instance calling the tool"
    agent: AgentT

    "The tool call object containing the tool id, name, and args"
    tool_call: Any

    "A callable requested by the agent via the tool_call"
    func: Callable = field(init=False)

    "Named arguments passed to `func`"
    kwargs: dict[str, Any] = field(default_factory=dict, init=False)

    "Any errors to be returned to the agent (failure to retrieve function or parse args, etc)"
    errors: str | None = field(init=False, default=None)

    "An asyncio task running the tool call"
    task: Task | None = field(init=False, default=None)

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """
        The tool call ID extracted from the `tool_call` object
        which will be used to compose the return message
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def func_name(self) -> str:
        """
        The name of the function being requested by the language agent
        extracted from the `tool_call` object.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def arg_str(self) -> str:
        """
        The unparsed arguments string in JSON format extracted from
        the `tool_call` object.
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def _construct_return_message(
        id: str, response: str | BaseModel
    ) -> dict[str, str | BaseModel]:
        """
        A function that constructs the provider-appropriate return message for the tool call.

        Args:
            id (str): A tool call ID that the language uses to keep track of responses internally
            response (str | BaseModel): The response to the requested tool call to return to the model

        Returns:
            out (dict[str, str | BaseModel]): A properly formatted message payload with the output of the tool call for the language agent to handle
        """
        raise NotImplementedError()

    @property
    def result(self) -> dict[str, str | BaseModel] | None:
        """
        Return tool call result, if available
        - calls the result() method on the task
        """
        if self.task is None:
            return None
        return self.task.result()

    def _check_and_assign_func(self):
        """
        A function to check the validity of the function name before retrieving the method.

        We check:
        - That the requested method is in the agent class
        - then, we assert that tool is allowed to be called for safety

        if either fail, appends the error for re-prompt in the `errors` attribute, and skip evaluation.
        """
        try:
            self.func = getattr(self.agent, self.func_name)
            assert self.func_name in self.agent._known_tools
        except (AttributeError, AssertionError):
            logger.warning(
                f"Agent attempted to apply undefined function: {self.func_name}()"
            )
            self.errors = f"You attempted to apply an undefined function: {self.func_name}, you may only use the following functions as tool calls: {self.agent._known_tools}."

    def _check_and_assign_kwargs(self):
        """
        Either assign kwargs for tool call, if JSON payload is able to be decoded, or append error for re-prompt
        """
        try:
            self.kwargs.update(json.loads(self.arg_str))
        except json.JSONDecodeError as e:
            logger.warning(
                f"Tool call {self.func_name} in response couldn't be decoded: {e!s}"
            )
            self.errors = "The arguments to your previous tool call couldn't be parsed correctly. Please ensure you properly escapse quotes and construct a valid JSON payload."

    def __call__(self) -> Task[dict[str, str | BaseModel]]:
        """
        Return async task to gather later
        - Can only be fired once
        - Resulting task is assigned to the `task` slot for later use
        """
        if self.task is None:
            self.task = create_task(self.handler(), name=self.id)
        return self.task

    async def handler(self) -> dict[str, str | BaseModel]:
        """
        A handler coroutine that wraps a tool call, either awaiting it if it's also a co-routine, or sending
        it to a thread to be handled separately if it's sequential.

        This first checks that the function and arguments provided by the language agent are valid before attempting to call the tool.

        Returns:
            out (dict[str, str | BaseModel]): A tool call reply payload for the language agent
        """
        self._check_and_assign_func()
        self._check_and_assign_kwargs()

        # If we had a validation error, early return
        if self.errors is not None:
            res = self.errors
        else:
            if iscoroutinefunction(self.func):
                call = self.func(**self.kwargs)
            else:
                call = to_thread(self.func, **self.kwargs)

            try:
                res = await call
            except ValidationError as e:
                # Case: Handle pydantic validation errors by passing them back to the
                # model to correct
                logger.debug("Failed Pydantic Validation.")
                res = str(e)

        return self._construct_return_message(self.id, res)


class _Provider[AgentT: _Agent](Observable, metaclass=abc.ABCMeta):
    """
    A LLM Provider which should provide the standard methods for prompting and agent
    authenticating, etc.
    """

    "The tool_call class specific to this provider that will be used to evaluate any tool calls from the model"
    tool_call_wrapper: type[_ToolCall[Any]]
    "The method that will be used to call the OpenAI API, e.g. openai.chat.completions.create"
    endpoint_fn: Callable[..., Awaitable["ChatCompletion"]]

    mode: Literal["chat", "batch"]

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__()

    @abc.abstractmethod
    def authenticate(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    async def prompt_agent(self, ag: AgentT, prompt: Any, **kwargs: Any) -> "Choice":
        raise NotImplementedError()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """
        Close the provider, if necessary.
        """
        return


class _StoppingCondition(metaclass=abc.ABCMeta):
    """
    A callable that contains some logic to determine whether a language agent
    has finished it's run.

    The main call should always return the final answer, if we've finished the run, or None otherwise
    """

    @abc.abstractmethod
    def __call__(self, cls: "_Agent", response: "Choice") -> Any | None:
        raise NotImplementedError()


class _Agent(Observable, metaclass=abc.ABCMeta):
    terminated: bool = False
    truncated: bool = False
    curr_step: int = 1
    output_len: int = 1
    scratchpad: str = ""
    answer: Any = ""
    BASE_PROMPT: str = ""
    SYSTEM_PROMPT: str = ""
    oai_kwargs: dict[str, Any]
    TOOLS: list[Any]
    CALLBACKS: list[Callable[..., Any]]
    callback_output: list[Any]
    tool_res_payload: list[dict[str, Any]]
    provider: _Provider[Any]
    placeholder: Any | None

    def __init__(
        self,
        stopping_condition: _StoppingCondition,
        model_name: str | None = None,
        provider: _Provider[Any] | None = None,
        tools: Sequence[Any] | None = None,
        callbacks: Sequence[Callable[..., Any]] | None = None,
        oai_kwargs: dict[str, Any] | None = None,
        **fmt_kwargs: Any,
    ) -> None:
        super().__init__()

    @abc.abstractmethod
    async def step(self) -> None:
        """
        Run a single "step" of the agent logic.
        Handles prompting OpenAI, optionally handling tool calls, and determining whether we've
        finished or run out of tokens.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _check_stop_condition(self, response: "Choice") -> None:
        """
        Called from within :func:`step()`.
        Checks whether our stop condition has been met and handles assignment of answer, if so.

        It's broken out this way because we may not always want to use message.content as the answer (tool call output, for instance)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def format_prompt(self) -> str:
        """
        Method which formats the BASE_PROMPT string, possibly inserting additional content.
        This is usually called within get_next_message() to populate the first user message.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_messages(self) -> list[Message]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _handle_tool_calls(self, response: "Choice") -> None:
        raise NotImplementedError()

    @property
    def is_terminated(self) -> bool:
        return self.terminated

    @property
    def _known_tools(self) -> list[str]:
        return [tool["function"]["name"] for tool in self.TOOLS]

    @property
    def is_truncated(self) -> bool:
        return self.truncated

    @abc.abstractmethod
    def dump(self, outfile: str | os.PathLike) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()


class _BatchAPIHelper[ProviderT: _Provider[Any]](metaclass=abc.ABCMeta):
    timeout: float = 2.0
    task: Task
    batch_tasks: list[Task]
    provider: ProviderT

    async def close(self):
        """
        Close the batch API helper, canceling any running tasks
        """
        if self.task is not None:
            self.task.cancel()
        for t in self.batch_tasks:
            t.cancel()

        all_tasks = [self.task] + self.batch_tasks
        await asyncio.gather(*all_tasks, return_exceptions=True)

    @abc.abstractmethod
    def register_provider(self, provider: ProviderT) -> None:
        """
        Main method to link provider and start batching task via asyncio
        """
        raise NotImplementedError()


class _Callback[CallingAgentT: _Agent](metaclass=abc.ABCMeta):
    """
    A Callback virtual class
    """

    @abc.abstractmethod
    async def __call__(self, cls: CallingAgentT, answer: Any, scratchpad: str) -> None:
        """
        Primary method called by agent during callback process

        :param Agent cls: Instantitated class of calling agent for possible modification
        :param answer: The final response of the calling Agent
        :param str scratchpad: The full interaction history of the calling Agent

        """
        raise NotImplementedError()
