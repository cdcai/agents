import asyncio
import logging
from copy import copy
from typing import Any, Optional

import openai
from pydantic import BaseModel

from ..abstract import _Agent
from ..providers import AzureOpenAIProvider
from ..stopping_conditions import StopOnDataModel

logger = logging.getLogger(__name__)


class Agent(_Agent):
    """
    Base Class for language agents, which can be initialized directly or subclassed depending on use case.

    :param bool terminated: Whether agent has completed it's run (use :func:`reset`, to reset)
    :param bool truncated: Whether the agent received a truncated response
    :param int curr_step: Current step of the query process
    :param int output_len: Expected length of answer `("", ) * output_len`
    :param str scratchpad: Documents all steps querying and evaluating LLM responses
    :param any answer: Final response from agent (default: str, could be any)
    :param str BASE_PROMPT: Query prompt which should be populated with `fmt_kwargs` via fstr (and possibly other parameters via :func:`format_prompt`). sent as system message
    :param str SYSTEM_PROMPT: System prompt (persona) sent as first message in chat session
    :param dict oai_kwargs: OpenAI arguments passed as-is to API (temperature, top_p, etc.)
    :param list TOOLS: List of tools the agent can use. Can be defined in subclass or at runtime. (see: https://platform.openai.com/docs/guides/function-calling)
    :param list CALLBACKS: List of callbacks to evaluate at completion. Should be a list of callables with a signature `fun(self, answer, scratchpad)`
    :param _StoppingCondition stopping_condition: The StoppingCondition handler class which will be called after each step to determine if the task is completed.


    Tool Use
    --------
    Each added tool must have a corresponding class method that can be invoked during :func:`step()` if the GPT calls it.
    You should subclass accordingly.

    Callback Use
    ------------
    Each callback will have access to class object, scratchpad, and final answer, thus the signature must match.
    This is still quite experimental, but the intended usecase is for reflection / refinement applications.
    """

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
        """
        Base Agent class

        :param _StoppingCondition stopping_condition: A handler that signals when an Agent has completed the task
        :param str model_name: Name of model to use (or deployment name for AzureOpenAI) (optional if provider is passed)
        :param Type[_Provider] provider: Instantiated OpenAI instance to use (optional)
        :param List[dict] tools: List of tools the agent can call via response (optional)
        :param List[Callable] callbacks: List of callbacks to evaluate at end of run (optional)
        :param dict[str, any] oai_kwargs: Dict of additional OpenAI arguments to pass thru to chat call
        :param fmt_kwargs: Additional named arguments which will be inserted into the :func:`BASE_PROMPT` via fstring
        """
        self.fmt_kwargs = fmt_kwargs
        self.stopping_condition = stopping_condition
        # We default to Azure OpenAI here, but
        # we could also use something else as long as it follows the OpenAI API
        if provider is None:
            self.provider = AzureOpenAIProvider(
                model_name=model_name, interactive=False
            )
        else:
            self.provider = provider

        # Handle Tools
        self.TOOLS = getattr(self, "TOOLS", [])

        if tools is not None:
            self.TOOLS.extend(tools)

        # Add any methods defined with decorator
        self.TOOLS.extend(self._check_agent_callable_methods())

        # Handle Callbacks
        self.CALLBACKS = getattr(self, "CALLBACKS", [])

        if callbacks is not None:
            self.CALLBACKS.extend(callbacks)

        self.oai_kwargs = oai_kwargs if oai_kwargs is not None else {}

        if len(self.TOOLS):
            self.oai_kwargs.update({"tools": self.TOOLS})

        self.reset()

    async def run(self, reset: bool = False, *kwargs) -> None:
        if reset:
            self.reset()

        while not (self.is_terminated or self.is_truncated):
            logger.debug(f"Running step {self.curr_step}.")
            await self.step()

        # Evaluate callbacks, if available
        for callback in self.CALLBACKS:
            await callback(self, answer=self.answer, scratchpad=self.scratchpad)

    async def __call__(self, *args, **kwargs) -> str:
        """
        Run the underlying agent logic and returns the final answer.

        :param reset (bool): Passed to run(), whether to reset agent state (else, will return previous answer on subsequent runs)
        :param outfile (str): Passed to dump(), file to dump the full scratchpad
        """
        outfile = kwargs.pop("outfile", None)

        await self.run(*args, **kwargs)

        if outfile is not None:
            self.dump(outfile)

        return self.answer

    def _check_stop_condition(self, response):
        # Check if we've reached a stopping place
        if (answer := self.stopping_condition(self, response)) is not None:
            self.answer = answer
            self.terminated = True
            logger.info("Stopping condition signaled, terminating.")

    async def step(self):
        """
        Run a single "step" of the agent logic.
        Handles prompting OpenAI, optionally handling tool calls, and determining whether we've
        finished or run out of tokens.
        """
        # Pull base query + system messages
        # (abstract)
        llm_prompt_input = self.get_next_messages()

        # Send off messages for reply
        self.scratchpad += f"=== Step {self.curr_step} ===========\n"

        # Attempt to query GPT and handle invalid JSON parsing of args
        response = None
        n_retry = 3
        while response is None and n_retry > 0:
            response = await self.provider.prompt_agent(
                self, llm_prompt_input, **self.oai_kwargs
            )
        if response is None:
            logger.warning("No response after 3 retries, Terminating!")
            self.truncated = True
        else:
            if response.finish_reason == "length":
                # Determine if we're truncated
                self.truncated = True
                logger.warning("Response truncated due to length, Terminating!")
            # Recursive call if tool calls in response
            elif response.message.tool_calls is not None:
                await self._handle_tool_calls(response)

        # Conditionally end run and assign answer
        self._check_stop_condition(response)

        if self.terminated:
            self.scratchpad += "===== Answer ============\n"
            self.scratchpad += str(self.answer)

        # End Step
        self.scratchpad += "==============================\n\n"
        self.curr_step += 1

    @staticmethod
    def clean_response(res: str) -> str:
        """
        Simple helper function to clean response text, if desired
        """
        out = res.strip("\n").strip().replace("\n", "")
        return out

    def format_prompt(self) -> str:
        """
        Method which formats the BASE_PROMPT string, possibly inserting additional content.
        This is usually called within get_next_message() to populate the first user message.
        """
        if len(self.BASE_PROMPT) == 0:
            raise ValueError(
                "You initialized an Agent with no BASE_PROMPT, please define this attribute with your prompt, optionally adding any formatting args in brackets."
            )
        try:
            out = self.BASE_PROMPT.format(**self.fmt_kwargs)
        except KeyError as err:
            raise KeyError(
                f"The following format kwargs were not passed at init time to format the BASE_PROMPT: {err}."
            )

        return out

    def get_next_messages(self) -> list[dict[str, str]]:
        """
        Retrieve next message payload for GPT prompting.
        This defaults to only the SYSTEM_PROMPT and the formatted BASE_PROMPT via format_prompt()
        """
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": self.format_prompt()},
        ]

        # If we have existing tool response messages, append them
        if len(self.tool_res_payload):
            out.extend(self.tool_res_payload)

        return out

    async def _handle_tool_calls(self, response):
        """
        Handle all tool calls in response object

        This gets a method within this class by name and evaluates it with the arguments provided by openai.

        The output of that method is appended to a new message in the tool_res_payload list, for downstream querying.
        """
        # Guarding
        if response.message.tool_calls is None:
            return None

        self.scratchpad += "--- Evaluating Toolcalls -----------------\n"

        # Run all awaitables
        tool_calls = [
            self.provider.tool_call_wrapper(self, tool)
            for tool in response.message.tool_calls
        ]
        tool_call_tasks = [tool_call() for tool_call in tool_calls]

        tool_call_results = await asyncio.gather(
            *tool_call_tasks, return_exceptions=True
        )

        # We might have handled all the calls before we ever dispatched, so only proceed if there were
        # tasks to check
        if len(tool_call_results):
            self.scratchpad += "Tool call results:\n\n"
            for payload, result in zip(tool_calls, tool_call_results):
                # Log it
                toolcall_str = f"{payload.func_name}({str(payload.kwargs)[:100] + '...(trunc)' if len(str(payload.kwargs)) > 100 else str(payload.kwargs)})"
                logger.info(f"Got tool call: {toolcall_str}")
                self.scratchpad += f"\t=> {toolcall_str}\n"
                self.scratchpad += "\t\t"

                # Check if a task errored and raise if so
                if isinstance(result, BaseException):
                    self.scratchpad += str(result)
                    raise result
                else:
                    self.tool_res_payload.append(result)

                    self.scratchpad += "\t\t"
                    self.scratchpad += (
                        result["content"]
                        if isinstance(result["content"], str)
                        else repr(result["content"])
                    )
                    self.scratchpad += "\n\n"

        self.scratchpad += "---------------------------------\n\n"

    def reset(self) -> None:
        """
        Reset agent state for a re-run
        """
        self.scratchpad = ""
        self.answer = ""
        self.tool_res_payload = []
        self.callback_output = []
        self.curr_step = 1
        self.truncated = False
        self.terminated = False

    def dump(self, outfile):
        """
        Dump scratchfile to disk
        """
        with open(outfile, "w", encoding="utf-8") as file:
            file.writelines(elem + "\n" for elem in self.scratchpad.split("\n"))

    def _check_agent_callable_methods(self):
        """
        Scans class for methods that are flagged as agent callable via decorator
        which alleviates some boilerplate for manually defining JSON schema along with method
        """
        payload = []
        for obj in dir(self):
            if callable(getattr(self, obj)) and len(
                getattr(getattr(self, obj), "agent_tool_payload", [])
            ):
                payload.append(getattr(getattr(self, obj), "agent_tool_payload"))

        return payload


class StructuredOutputAgent(Agent):
    """
    An Agent with accepts a pydantic BaseModel to use as a tool / validator for model output

    A class method is constructed at runtime along with a stopping condition which triggers when a `response_model` object is detected in the response.
    """

    answer: dict[str, Any]
    _response_model_warn: bool = True

    def __init__(
        self,
        response_model: type[BaseModel],
        model_name: Optional[str] = None,
        stopping_condition=None,
        provider=None,
        tools=None,
        callbacks=None,
        oai_kwargs=None,
        **fmt_kwargs,
    ):
        """
        Language Agent with structured output

        Handles creating a class method that is used to validate the tool call in the response body at runtime.
        Also constructs it's own stopping condition which triggers when a `response_model` object is detected in the response (and is parsed correctly).

        :param BaseModel response_model: A data model to use for structured output
        :param _StoppingCondition stopping_condition: A handler that signals when an Agent has completed the task
        :param str model_name: Name of model to use (or deployment name for AzureOpenAI) (optional if provider is passed)
        :param Type[_Provider] provider: Instantiated OpenAI instance to use (optional)
        :param List[dict] tools: List of tools the agent can call via response (optional)
        :param List[Callable] callbacks: List of callbacks to evaluate at end of run (optional)
        :param dict[str, any] oai_kwargs: Dict of additional OpenAI arguments to pass thru to chat call
        :param fmt_kwargs: Additional named arguments which will be inserted into the :func:`BASE_PROMPT` via fstring

        """
        self.response_model = response_model
        self.output_len = len(self.response_model.model_fields)

        if stopping_condition is None:
            stopping_condition = StopOnDataModel(response_model)
        elif stopping_condition is not None and self._response_model_warn:
            logger.warning(
                "StructuredOutputAgent assumes a `StopOnBaseModel` stopping condition, but you passed another at runtime which will take precedence. This may lead to errors. This warning will only be displayed once per session."
            )
            StructuredOutputAgent._response_model_warn = False  # Once per session
        else:
            pass

        oai_tool = openai.pydantic_function_tool(response_model)

        # Ensure we don't modify external tools
        tools_internal = copy(tools)

        if tools_internal is not None:
            tools_internal.append(oai_tool)
        else:
            tools_internal = [oai_tool]

        # Assign a class method
        fun_name = oai_tool["function"]["name"]
        setattr(self, fun_name, self.response_model)

        super().__init__(
            stopping_condition=stopping_condition,
            model_name=model_name,
            provider=provider,
            tools=tools_internal,
            callbacks=callbacks,
            oai_kwargs=oai_kwargs,
            **fmt_kwargs,
        )

    def reset(self) -> None:
        """
        Reset agent state for a re-run
        """
        self.scratchpad = ""
        self.answer = {}
        self.tool_res_payload = []
        self.callback_output = []
        self.curr_step = 1
        self.truncated = False
        self.terminated = False
