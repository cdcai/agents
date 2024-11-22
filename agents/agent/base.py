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
import gymnasium as gym
import openai
import tiktoken
from azure.identity import ClientSecretCredential
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage

logger = logging.getLogger(__name__)


class Agent:
    """
    Base Class for language agents, which can be initialized directly or subclassed depending on use case.

    :param bool terminated: Whether agent has completed it's run (use :func:`reset`, to reset)
    :param bool truncated: Whether the agent received a truncated response
    :param int curr_step: Current step of the query process
    :param int output_len: Expected length of answer `("", ) * output_len`
    :param str scratchpad: Documents all steps querying and evaluating LLM responses
    :param any answer: Final response from agent (default: str, could be any)
    :param str BASE_PROMPT: Query prompt which should be populated with `{question}` via fstr (and possibly other parameters via :func:`format_prompt`). sent as system message
    :param str SYSTEM_PROMPT: System prompt (persona) sent as first message in chat session
    :param dict oai_kwargs: OpenAI arguments passed as-is to API (temperature, top_p, etc.)
    :param list TOOLS: List of tools the agent can use. Can be defined in subclass or at runtime. (see: https://platform.openai.com/docs/guides/function-calling)
    :param list CALLBACKS: List of callbacks to evaluate at completion. Should be a list of callables with a signature `fun(self, answer, scratchpad)`

    Tool Use
    --------
    Each added tool must have a corresponding class method that can be invoked during :func:`step()` if the GPT calls it.
    You should subclass accordingly.

    Callback Use
    ------------
    Each callback will have access to class object, scratchpad, and final answer, thus the signature must match.
    This is still quite experimental, but the intended usecase is for reflection / refinement applications.
    """
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

    def __init__(
        self,
        question: str,
        model_name: str,
        llm: Optional[openai.AsyncOpenAI] = None,
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Callable]] = None,
        **oai_kwargs
    ):
        """
        Base Agent class

        :param str question: Final piece of prompt that is inserted into `BASE_PROMPT` via fstr
        :param str model_name: Name of OpenAI model to use (or deployment name for AzureOpenAI)
        :param AsyncOpenAI llm: Instantiated OpenAI instance to use (optional)
        :param List[dict] tools: List of tools the agent can call via response (optional)
        :param List[Callable] callbacks: List of callbacks to evaluate at end of run (optional)
        """
        self.question = question
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

        self.oai_kwargs = {"temperature": 0.0}
        self.oai_kwargs.update(oai_kwargs)
        self.reset()

    async def run(self, reset: bool = False, *kwargs) -> None:
        if reset:
            self.reset()

        while not (self.is_terminated() or self.is_truncated()):
            logger.debug(f"Running step {self.curr_step}.")
            await self.step()
        
        # Evaluate callbacks, if available
        for callback in self.CALLBACKS:
            callback(self, answer=self.answer, scratchpad=self.scratchpad)

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
            
        # End Step
        self.scratchpad += "==============================\n\n"
        self.curr_step += 1

    @backoff.on_exception(backoff.expo, (openai.APIError, openai.AuthenticationError), max_tries=3)
    async def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: Optional[int] = None, **addn_oai_kwargs) -> Choice:
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
        return re.sub("\s+", " ", self.BASE_PROMPT).format(question=self.question)

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

    def _handle_tool_calls(self, response: Choice):
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

                self.scratchpad
                self.tool_res_payload.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "content": tool_result
                    }
                )
            except Exception as e:
                logger.error(f"Tool call {tool} failed.")
                raise e

    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.truncated

    def reset(self) -> None:
        """
        Reset agent state for a re-run
        """
        self.scratchpad = ""
        self.answer = ""
        self.tool_res_payload = []
        self.curr_step = 1
        self.truncated = False
        self.terminated = False

    def dump(self, outfile: Union[str, os.PathLike]) -> None:
        """
        Dump scratchfile to disk
        """
        with open(outfile, "w", encoding="utf-8") as file:
            file.writelines(elem + "\n" for elem in self.scratchpad.split("\n"))

    @staticmethod
    def clean_response(res: str) -> str:
        out = res.strip('\n').strip().replace('\n', '')
        return out

    @staticmethod
    def authenticate() -> None:
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

class PersistentAgent(Agent):
    APPEND_PROMPT : str = "{obs}"
    conversation_cache : list[dict]

    def reset(self) -> None:
        self.conversation_cache = []
        return super().reset()

    async def step(self):
        """
        Full Agent logic. Prompts LLM and saves answer
        """
        llm_prompt_input = self.get_next_messages()
        response = await self.prompt_agent(llm_prompt_input, n_tok=None)
        answer = response.message.content
        self.scratchpad += f"=== Input ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"
        self.scratchpad += f"\n=== Answer =====\n"
        self.scratchpad += "\n".join(answer) + "\n"
        self.scratchpad += "\n===================================\n"

        self.answer = answer
        self.conversation_cache.append({k: response.message.__dict__[k] for k in ["role", "content"]})
        self.terminated = True

    def format_append_prompt(self, obs: str) -> str:
        return self.APPEND_PROMPT.format(obs=obs)

    def add_observation(self, obs: str) -> None:
        """
        Append new message / observation to message cache
        This inserts `obs` the `APPEND_PROMPT` attribute and appends to the conversation
        """
        self.conversation_cache.append({
            "role": "user",
            "content": self.format_append_prompt(obs)
        })

    def get_next_messages(self) -> list[dict[str, str]]:
        out = super().get_next_messages()
        out.extend(self.conversation_cache)

        return(out)

class ReduceAgent(Agent):
    """
    An agent which reduces a list[str] question to a single string output
    """
    question : list[str]

    def __init__(self, question: list[str], model_name: str, llm: openai.AsyncOpenAI | None = None, **oai_kwargs):
        self.question = []
        super().__init__(question, model_name, llm, **oai_kwargs)

    def step(self):
        """
        This will always be a single-step run to summarize the messages
        """
        res = self.prompt_agent(self.get_next_messages(), n_tok=None)
        self.answer = res.message.content
        self.terminated = True

    def get_next_messages(self) -> list[dict[str, str]]:
        out = super().get_next_messages()
        out.append({
            "role": "assistant",
            "content": "\n".join(self.question)
        })
        
        return out
    def format_prompt(self, **kwargs) -> str:
        """Return BASE_PROMPT as-is, no templating"""
        return self.BASE_PROMPT

class ChunkedAgent(Agent):
    """
    A language agent which can handle large prompt input by chunking.
    It will use tiktoken to estimate token usage and ensure that the message payload
    sent each trip is below `chunk_max`, set at init time.

    Each subsequent step() will append the output from the previous run as context.

    After the run is finished, all answers will be appended together into a single string and assigned to the answer attribute.
    """
    answer_cache : list[str] = []
    prompt_len : int
    full_question : str
    chunk_max : int

    def __init__(self, question: str, model_name: str, llm: openai.AsyncOpenAI | None = None, chunk_max : int = 3000, **oai_kwargs):
        # Also save full input to full_question attribute since we'll
        # overwrite self.question if the resulting payload is too large
        self.full_question = question
        self.chunk_max = chunk_max
        # Get tokenizer to handle chunking responses if needed
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Take the base prompt length before we fstring it
        self.prompt_len = len(self.tokenizer.encode(self.BASE_PROMPT.format(question="") + self.SYSTEM_PROMPT))
        
        super().__init__(question, model_name, llm, **oai_kwargs)

    def combine_answer_cache(self) -> None:
        """
        Combine possibly >1 response into a the final answer string
        """
        self.answer = "\n".join(self.answer_cache)

    def fetch_last_response(self) -> Optional[dict[str, str]]:
        """
        If step > 1, returns last response to use as context for the next.
        Otherwise, we'd only include the system query and base query (including possibly only a single chunk of the input)
        """
        if len(self.answer_cache):
            out = {
                "role": "user",
                "content": "Here is the output of previous chunk you worked on, for context:\n {}".format(self.answer_cache[-1])
            }
        else:
            out = None
        
        return out

    def get_next_messages(self) -> list[dict[str, str]]:
        """
        Retrieve next message payload for GPT prompting, and append previous output for context, if needed.
        """
        out = super().get_next_messages()
        if (last_translation_message := self.fetch_last_response()) is not None:
            out.append(last_translation_message)

        return out

    def get_prompt_len(self) -> int:
        """
        Return base prompt length before using fstring to fill in template.
        (also accounting for possible previous translation context)
        """
        if (last_translation_message := self.fetch_last_response()) is not None:
            prompt_len = self.prompt_len + len(self.tokenizer.encode(last_translation_message["content"]))
        else:
            prompt_len = self.prompt_len

        return prompt_len

    def format_prompt(self, split_expr: str = "\n{2,}?", join_str: str = "\n\n", **kwargs) -> str:
        """
        Formatting BASE_QUERY, checking for output length and chunking self.question if necessary

        :param split_expr (str): A string or regex to pass to re.split() to split self.question into chunks.
        :param join_str (str): A string to use to recompose chunks of self.question back together.
        
        NOTE: split_expr and join_str can be different (ex. '\\n{2, }?', and '\\n\\n'),
        but join_str should always produce output that could be split on subsequent calls using split_expr.
        """
        prompt_len = self.get_prompt_len()

        input_chunks = re.split(split_expr, self.question)
        excess = []

       # pop chunk by chunk until we have a message payload less than the requested max
        while len(self.tokenizer.encode(join_str.join(input_chunks))) + prompt_len > self.chunk_max:
            # Store excess message payload in question object
            excess.append(input_chunks.pop())
        
        # Reverse things around and re-join to string
        # to get things the right way around
        self.question = join_str.join(reversed(excess))

        return re.sub("\s+", " ", self.BASE_PROMPT).format(question=join_str.join(input_chunks)).strip()
    
    async def step(self):
        # Prompt LLM
        llm_prompt_input = self.get_next_messages()
        answer = await self.prompt_agent(llm_prompt_input, n_tok = 2 * self.chunk_max).message.content
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"
        self.scratchpad += f"\n=== Answer {self.curr_step} =====\n"
        self.scratchpad += answer + "\n"
        self.scratchpad += "\n===================================\n"

        # Append answer to cache and continue
        self.answer_cache.append(answer)

        # End run
        self.terminated = len(self.question) == 0
        self.curr_step += 1

    async def run(self, reset: bool = False) -> None:
        super().run(reset)
        self.combine_answer_cache()
    
    def reset(self) -> None:
        self.answer_cache = []
        return super().reset()

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

    def __init__(self, question: str, model_name: str, llm: openai.AsyncOpenAI | None = None, tools: Optional[Union[dict, list[dict]]] = None, submit_tool: bool = True, **oai_kwargs):
        
        if tools is None and submit_tool:
            tools = [self.submit_tool]
        elif submit_tool:
            tools.append(submit_tool)

        super().__init__(question, model_name, llm, tools=tools, **oai_kwargs)

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

class EnvAgent(Agent):
    """
    A Base (abstract) class for language agents which interact with an environment
    (as implemented by gymnasium) 
    """
    correct: bool = False

    def __init__(self, question: str, model_name: str, llm: openai.AsyncOpenAI, env: gym.Env):
        self.env = env
        super().__init__(question, model_name, llm)
    
    def is_truncated(self) -> bool:
        # NOTE: I think they also checked that prompt length
        # was under a certain value here, but that'd mean
        # importing tiktoken and computing it each step
        return super().truncated() and not self.correct

    async def prompt_agent(self, prompt: dict[str, str] | list[dict[str, str]], n_tok: int = 100, **oai_kwargs) -> str:
        out_msg = await super().prompt_agent(prompt, n_tok, **oai_kwargs)

        out = self.clean_response(out_msg.message.content) + "\n"

        return out

    def reset(self):
        super().reset()
        self.env.reset()
        self.correct = False
