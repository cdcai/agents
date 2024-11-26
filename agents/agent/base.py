import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Callable, Literal, Optional, Union, List
from copy import deepcopy

import backoff
import openai
from azure.identity import ClientSecretCredential
from openai.types.chat.chat_completion import ChatCompletionMessage
from ..abstract import _Agent

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
    def __init__(self, model_name, stopping_condition, llm = None, tools = None, callbacks = None, oai_kwargs = None, **fmt_kwargs):
        """
        Base Agent class

        :param str model_name: Name of OpenAI model to use (or deployment name for AzureOpenAI)
        :param _StoppingCondition stopping_condition: A handler that signals when an Agent has completed the task
        :param AsyncOpenAI llm: Instantiated OpenAI instance to use (optional)
        :param List[dict] tools: List of tools the agent can call via response (optional)
        :param List[Callable] callbacks: List of callbacks to evaluate at end of run (optional)
        :param dict[str, any] oai_kwargs: Dict of additional OpenAI arguments to pass thru to chat call
        :param fmt_kwargs: Additional named arguments which will be inserted into the :func:`BASE_PROMPT` via fstring
        """
        self.fmt_kwargs = fmt_kwargs
        self.stopping_condition = stopping_condition
        # We default to Azure OpenAI here, but
        # we could also use something else as long as it follows the OpenAI API
        if llm is None:
            self.authenticate()
            self.llm = openai.AsyncAzureOpenAI()
        else:
            self.llm = llm
        self.model_name = model_name
        
        # Handle Tools
        self.TOOLS = getattr(self, "TOOLS", [])

        if tools is not None:
            self.TOOLS.extend(tools)

        # Handle Callbacks
        self.CALLBACKS = getattr(self, "CALLBACKS", [])

        if callbacks is not None:
            self.CALLBACKS.extend(callbacks)

        self.oai_kwargs = oai_kwargs if oai_kwargs is not None else {}
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
        if answer := self.stopping_condition(self, response) is not None:
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
            try:
                response = await self.prompt_agent(llm_prompt_input)
            except json.decoder.JSONDecodeError as e:
                if n_retry == 0:
                    raise e
                else:
                    self.scratchpad += "JSONDecodeError in tool call argumement parsing. Retrying.\n"
                    logger.warning(f"Tool calls in response couldn't be decoded. {n_retry} retries remaining.")
                    llm_prompt_input.append(
                        {
                            "role": "user",
                            "content": "The arguments to your previous tool call couldn't be parsed correctly. Please ensure you properly escapse quotes and construct a valid JSON payload."
                        }
                    )
                    n_retry -= 1
                    continue
        if response is None:
            logger.warning("No response after 3 retries, Terminating!")
            self.truncated = True
        else:
            if response.finish_reason == "length":
                # Determine if we're truncated
                self.truncated = True
                logger.warning("Response truncated due to length, Terminating!")
            # Recursive call if tool calls in response
            elif response.finish_reason == "tool_calls":
                self._handle_tool_calls(response)
        
        # Conditionally end run and assign answer
        self._check_stop_condition(response)

        # End Step
        self.scratchpad += "==============================\n\n"
        self.curr_step += 1

    @staticmethod
    def authenticate():
        """
        Authenticate against Azure OpenAI using Service Principal
        """
        # === Service Principal auth ========================
        credential = ClientSecretCredential(
            tenant_id=os.environ["SP_TENANT_ID"],
            client_id=os.environ["SP_CLIENT_ID"],
            client_secret=os.environ["SP_CLIENT_SECRET"],
        )

        os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token

        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["GPT4_URL"]

    @backoff.on_exception(backoff.expo, (openai.APIError, openai.AuthenticationError), max_tries=3)
    async def prompt_agent(self, prompt, n_tok = None, **addn_oai_kwargs):
        """
        An async version of the main OAI prompting logic.

        :param prompt: Either a dict or a list of dicts representing the message(s) to send to OAI model
        :param n_tok: An optional maximum token length to request of the model response
        :param addn_oai_kwargs: Key word arguments passed to completions.create() call (tool calls, etc.)

        :return: An openAI Choice response object
        """

        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if isinstance(prompt, dict):
            prompt = [prompt]

        self.scratchpad += f"--- Input ---------------------------\n"
        self.scratchpad += "\n".join(msg["content"] for msg in prompt if not isinstance(msg, ChatCompletionMessage))
        self.scratchpad += "\n-----------------------------------\n"
    
        try:
            res = await self.llm.chat.completions.create(
                messages=prompt, model=self.model_name, max_tokens=n_tok,
                **addn_oai_kwargs,
                **self.oai_kwargs
            )
        except openai.AuthenticationError:
            logger.info("Auth failed, attempting to re-authenticate before retrying")
            # HACK: This isn't terrific, but it should work for
            # our current use case (Azure OpenAI with service principal/User creds)
            if isinstance(self.llm, openai.AzureOpenAI):
                self.authenticate()
                self.llm.api_key = os.environ["AZURE_OPENAI_API_KEY"]
            raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out = res.choices[0]

        self.scratchpad += "--- Output --------------------------\n"
        self.scratchpad += "Message:\n"
        self.scratchpad += out.message.content + "\n"

        if len(self.TOOLS):
            # Append GPT response to next payload
            # NOTE: This has to come before the next step of parsing
            self.tool_res_payload.append(deepcopy(out.message))

            # attempt to parse tool call arguments
            if out.finish_reason == "tool_calls":
                self.scratchpad += "Tool calls: \n"
                for i, tool in enumerate(out.message.tool_calls):
                    out.message.tool_calls[i].function.arguments = json.loads(tool.function.arguments)

                    # Log it
                    toolcall_str = f"{tool.function.name}({str(tool.funcion.arguments)[:30] + '...(trunc)' if len(str(tool.funcion.arguments)) > 30 else str(tool.funcion.arguments)})"
                    logger.info(f"Got toolcall: {toolcall_str}")
                    self.scratchpad += f"\t=> {toolcall_str}\n"
        
        self.scratchpad += "\n-----------------------------------\n"
        logger.info(f"Received response: {out.message.content}")

        if out.finish_reason == "length":
            self.truncated = True
            self.scratchpad += "Response returned truncated from OpenAI due to token length.\n"
            logger.warning("Message returned truncated.")
        return out

    def format_prompt(self, **kwargs) -> str:
        """
        Method which formats the BASE_PROMPT string, possibly inserting additional content.
        This is usually called within get_next_message() to populate the first user message.
        """
        return re.sub("\s+", " ", self.BASE_PROMPT).format(**self.fmt_kwargs)

    def get_next_messages(self) -> list[dict[str, str]]:
        """
        Retrieve next message payload for GPT prompting.
        This defaults to only the SYSTEM_PROMPT and the formatted BASE_PROMPT via format_prompt()
        """
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": self.format_prompt()}
        ]

        # If we have existing tool response messages, append them
        if len(self.tool_res_payload):
            out.extend(self.tool_res_payload)
        
        return out

    def _handle_tool_calls(self, response):
        """
        Handle all tool calls in response object
        
        This gets a method within this class by name and evaluates it with the arguments provided by openai.

        The output of that method is appended to a new message in the tool_res_payload list, for downstream querying.
        """
        for tool in response.message.tool_calls:
            self.scratchpad += "--- Evaluating Toolcalls -----------------\n"
            # Try to call tool, if present, else raise.
            try:
                fun : Callable = getattr(self, tool.function.name)
                # OpenAI returns as str, which should hopefully eval to dict
                kwargs : dict[str, any] = tool.function.arguments

                tool_result = fun(**kwargs)

                self.scratchpad += f"\t=> {tool.function.name}()\n"
                self.scratchpad += tool_result + "\n\n"

                self.tool_res_payload.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "content": tool_result
                    }
                )
            except Exception as e:
                logger.error(f"Tool call {tool.function.name} failed.")
                raise e

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

class MultiStepToolAgent(Agent):
    """
    An agent which expects multiple round-trips to complete task before finally calling a :func:`call_submit()` function to finish
    """

    TOOLS : list[dict]
    # Will always be added to TOOLS
    # (required to finalize)
    submit_tool : dict = {
        # Submit (final response)
            "type": "function",
            "function": {
                "name": "call_submit",
                "description": "Submit the final response back to user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Final response to user"
                        }
                    },
                    "required": ["input"]
                }
            }
        }

    # Payload to send back in subsequent steps
    tool_res_payload : list[dict]

    def __init__(self, model_name: str, llm: openai.AsyncOpenAI | None = None, tools: Optional[Union[dict, list[dict]]] = None, submit_tool: bool = True, oai_kwargs: Optional[dict[str, any]] = None, **fmt_kwargs):
        
        if tools is None and submit_tool:
            tools = [self.submit_tool]
        elif submit_tool:
            tools.append(submit_tool)

        super().__init__(model_name, llm, tools=tools, oai_kwargs=oai_kwargs, **fmt_kwargs)

    def call_submit(self, input: str, clean: bool = False) -> None:
        """
        Final response call, which terminates further processing
        """
        out_msg = self.clean_response(input) if clean else input
        logger.info(f"Received final response: {out_msg}")
        self.scratchpad += "===== Answer ==========\n"
        self.scratchpad += out_msg
        self.answer = out_msg
        self.terminated = True
    
    @staticmethod
    def _subprocess_tool_call_on_file(tool_input: str, cmd_args: list[str], output_type: Literal["stdout", "file"] = "stdout") -> str:
        """
        A helper function that writes `tool_input` to a file and runs a python module on that file, either returning stdout+stderr or the contents of the file after the subprocess call.

        :param tool_input (str): A string to pass as input to the tool (this is likely code)
        :param cmd_args (list[str]): Command-line args between the python -m call and the file name (should include the python module to call and any additional arguments)
        :param output_type (str): The output to return (either stdout+error, or contents of the tempfile, if this is modified)
        
        :return: Either stdout and stderr concatenated into a string and separated by a newline, or `tool_input` after calling the python module
        """
        with tempfile.TemporaryFile("w", delete=False) as file:
            file.write(tool_input)
            file.close()

            # Run mypy in a subprocess and capture stderr and stdout
            out = subprocess.run(
                [sys.executable, "-m", *cmd_args, file.name],
                capture_output=True
            )
            if output_type == "stdout":
                return "\n".join([out.stdout.decode("utf-8"), out.stderr.decode("utf-8")])
            elif output_type == "file":
                with open(file.name, "r") as f:
                    out = f.read()
                    return(out)
            else:
                # Shouldn't be reachable
                raise NotImplementedError()
