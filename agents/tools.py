import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, Literal, Optional, Type

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, ValidationError

from .abstract import _Agent

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """
    An encapsulating class for tool calls from a language agent

    Somewhat hacky, since it assumes an OpenAI-style standard.

    Calling this class after init returns an asyncio task which can be gathered to return the result
    """

    agent: _Agent
    tool_call: ChatCompletionMessageToolCall
    id: str = field(init=False)
    func_name: str = field(init=False)
    func: Callable = field(init=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, init=False)
    errors: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        self.id = self.tool_call.id
        self.func_name = self.tool_call.function.name
        self._check_and_assign_func()
        self._check_and_assign_kwargs()

    def _check_and_assign_func(self):
        """
        Check tool that agent requested
        - Check that method is in agent class
        - then, assert that tool is callable for safety

        if either fail, append error for re-prompt
        """
        try:
            self.func: Callable = getattr(self.agent, self.func_name)
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
            self.kwargs.update(json.loads(self.tool_call.function.arguments))
        except json.JSONDecodeError as e:
            logger.warning(
                f"Tool call {self.func_name} in response couldn't be decoded: {str(e)}"
            )
            self.errors = "The arguments to your previous tool call couldn't be parsed correctly. Please ensure you properly escapse quotes and construct a valid JSON payload."

    def __call__(self) -> asyncio.Task[Dict[str, str | BaseModel]]:
        """
        Return async task to gather later
        """
        return asyncio.create_task(self.handler(self.func, **self.kwargs), name=self.id)

    async def handler(self, func: Callable, **kwargs) -> Dict[str, str | BaseModel]:
        """
        A handler coroutine that wraps a tool call, either awaiting it if it's also a co-routine, or sending
        it to a thread to be handled separately if it's sequential.

        Args:
            func (Callable): A function requested to be called by the agent
            kwargs: Named arguments to `func` passed by the language agent after converting to dict

        Returns:
            out (Dict[str, str | BaseModel]): A tool call reply payload for the language agent
        """
        # If we had a validation error, early return
        if self.errors is not None:
            res = self.errors
        else:
            try:
                if iscoroutinefunction(func):
                    res = await func(**kwargs)
                else:
                    res = await asyncio.to_thread(func, **kwargs)
            except ValidationError as e:
                # Case: Handle pydantic validation errors by passing them back to the
                # model to correct
                logger.warning("Failed Pydantic Validation.")
                res = str(e)

        out = {"tool_call_id": self.id, "role": "tool", "content": res}

        return out


async def tool_call_handler(func: Callable, **kwargs) -> str | BaseModel:
    """
    A handler coroutine that wraps a tool call, either awaiting it if it's also a co-routine, or sending
    it to a thread to be handled separately if it's sequential.

    Args:
        func (Callable): A function requested to be called by the agent
        kwargs: Named arguments to `func` passed by the language agent after converting to dict

    Returns:
        out (str | BaseModel): Either a string to pass back to language agent, or BaseModel in the case of a StructuredOutputAgent which will terminate the run.
    """
    try:
        if iscoroutinefunction(func):
            res = await func(**kwargs)
        else:
            res = await asyncio.to_thread(func, **kwargs)
    except ValidationError as e:
        # Case: Handle pydantic validation errors by passing them back to the
        # model to correct
        logger.warning("Failed Pydantic Validation.")
        res = str(e)

    return res


def _subprocess_tool_call_on_file(
    tool_input: str,
    cmd_args: list[str],
    output_type: Literal["stdout", "file"] = "stdout",
) -> str:
    """
    A helper function that writes `tool_input` to a file and runs a python module on that file, either returning stdout+stderr or the contents of the file after the subprocess call.

    Ex. As a tool call for an Agent to use mypy / black on device and return output

    :param tool_input (str): A string to pass as input to the tool (this is likely code)
    :param cmd_args (list[str]): Command-line args between the python -m call and the file name (should include the python module to call and any additional arguments)
    :param output_type (str): The output to return (either stdout+error, or contents of the tempfile, if this is modified)

    :return: Either stdout and stderr concatenated into a string and separated by a newline, or `tool_input` after calling the python module
    """
    with tempfile.TemporaryFile("w", delete=False) as file:
        file.write(tool_input)
        file.close()

        # Run mypy in a subprocess and capture stderr and stdout
        subprocess_output = subprocess.run(
            [sys.executable, "-m", *cmd_args, file.name], capture_output=True, text=True
        )

        if output_type == "stdout":
            return "\n".join([subprocess_output.stdout, subprocess_output.stderr])
        elif output_type == "file":
            with open(file.name, "r", encoding="utf-8") as f:
                out = f.read()

            return out
        else:
            # Shouldn't be reachable
            raise NotImplementedError()
