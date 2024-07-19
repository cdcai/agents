import abc
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Callable, Literal, Optional, Union

import backoff
import gymnasium as gym
import openai
import tiktoken
from azure.identity import ClientSecretCredential
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class Agent(metaclass=abc.ABCMeta):
    terminated: bool = False
    truncated: bool = False
    curr_step: int = 1
    scratchpad: str = ""
    answer: str = ""
    BASE_PROMPT: str = ""
    SYSTEM_PROMPT: str = ""

    def __init__(
        self, question: str, model_name: str, llm: Optional[openai.OpenAI] = None
    ):
        self.question = question

        # We default to Azure OpenAI here, but
        # we could also use something else as long as it follows the OpenAI API
        if llm is None:
            self.authenticate()
            self.llm = openai.AzureOpenAI()
        else:
            self.llm = llm
        self.model_name = model_name
        self.reset()

    def run(self, reset: bool = False) -> None:
        if reset:
            self.reset()

        while not (self.is_terminated() or self.is_truncated()):
            logger.debug(f"Running step {self.curr_step}.")
            self.step()

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError()

    @backoff.on_exception(backoff.expo, openai.APIError, max_tries=3, logger=logger)
    def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: int = 100, **oai_kwargs) -> ChatCompletion:
        
        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if isinstance(prompt, dict):
            prompt = [prompt]

        try:
            res = self.llm.chat.completions.create(
                messages=prompt, model=self.model_name, max_tokens=n_tok,
                **oai_kwargs
            )
        except openai.AuthenticationError:
            logger.info("Auth failed, attempting to re-authenticate before retrying")
            # HACK: This isn't terrific, but it should work for
            # our current use case (Azure OpenAI with service principal/User creds)
            if isinstance(self.llm, openai.AzureOpenAI):
                self.authenticate()
                self.llm.api_key = os.environ["AZURE_OPENAI_API_KEY"]
                res = self.llm.chat.completions.create(
                    messages=prompt,
                    model=self.model_name,
                    max_tokens=n_tok,
                    **oai_kwargs
                )
            else:
                raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out = res.choices[0]
        logger.info(f"Received response: {out.message.content}")

        return out

    @abc.abstractmethod
    def format_prompt(self, **kwargs) -> str:
        """
        Method which formats the BASE_QUERY string, possibly inserting additional content.
        This should also implement chunking beha
        """
        raise NotImplementedError()

    def get_next_messages(self) -> list[dict[str, str]]:
        """
        Retrieve next message payload for GPT prompting
        """
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.format_prompt()}
        ]
        
        return out
    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.truncated

    def reset(self) -> None:
        self.scratchpad = ""
        self.answer = ""
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

    def __init__(self, question: str, model_name: str, llm: openai.OpenAI | None = None, chunk_max : int = 3000):
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
        
        super().__init__(question, model_name, llm)

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
        Formatting BASE_QUERY, checking for output length and chunking if necessary
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

    def run(self, reset: bool = False) -> None:
        super().run(reset)
        self.combine_answer_cache()
    
    def reset(self) -> None:
        self.answer_cache = []
        return super().reset()

class ToolAwareAgent(Agent):
    """
    A base-class for an agent which can utilize OpenAI tool calls.
    Subclasses would be expected to extend the TOOLS attribute to include additional
    tools / functions. Each added tool should be appended to the base attribute at init.
    
    In addition, each added tool must have a corresponding class method that can be invoked
    during step() if the GPT calls it.
    """

    TOOLS : list[dict] = []
    # Will always be added to TOOLS
    # (required to finalize)
    submit_tool : list[dict] = [
        # Submit (final response)
        {
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
    ]

    # Payload to send back in subsequent steps
    tool_res_payload : list[dict] = []

    def __init__(self, question: str, model_name: str, llm: openai.OpenAI | None = None):
        self.TOOLS.extend(self.submit_tool)
        super().__init__(question, model_name, llm)
    def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: Optional[int] = None, tool_use : Literal["required", "auto", "none"] = "auto", **oai_kwargs):
        
        out = super().prompt_agent(prompt, n_tok, tools=self.TOOLS, tool_choice=tool_use, **oai_kwargs)

        # attempt to parse tool call arguments
        if out.finish_reason == "tool_calls":
                for i, tool in enumerate(out.message.tool_calls):
                    out.message.tool_calls[i].function.arguments = json.loads(tool.function.arguments)

        return out

    def get_next_messages(self) -> list[dict[str, str]]:
        """
        Get next message payload for GPT, possibly appending tool call result output, if present.
        """
        out = super().get_next_messages()
        # If we have existing tool response messages, append them
        if len(self.tool_res_payload):
            out.extend(self.tool_res_payload)
        
        return out
    
    def _handle_tool_calls(self, response):
        """
        Handle all tool calls in response object
        """
        for tool in response.message.tool_calls:
            # Try to call tool, if present, else raise.
            try:
                fun : Callable = getattr(self, tool.function.name)
                # OpenAI returns as str, which should hopefully eval to dict
                kwargs : dict[str, any] = tool.function.arguments

                self.scratchpad += f"=> Requested toolcall: {tool.function.name}({str(kwargs)[:30] + '...'} <=\n"
                logger.info(f"Got tool call: {tool.function.name}({str(kwargs)[:30] + '...'})")

                tool_result = fun(**kwargs)
                self.tool_res_payload.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "function",
                        "name": tool.function.name,
                        "content": tool_result
                    }
                )
            except Exception as e:
                logger.error(f"Tool call {tool} failed.")
                raise e

    def step(self):
        # Pull base query + system messages
        # (abstract)
        llm_prompt_input = self.get_next_messages()

        # Send off messages for reply
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"
    
        # Attempt to query GPT and handle invalid JSON parsing of args
        response = None
        n_retry = 3
        while response is None and n_retry > 0:
            try:
                response = self.prompt_agent(llm_prompt_input)
            except json.decoder.JSONDecodeError as e:
                logger.warn(f"Tool calls in response couldn't be decoded. {n_retry} retries remaining.")
                if n_retry == 0:
                    raise e
                else:
                    llm_prompt_input.append(
                        {
                            "role": "user",
                            "content": "The arguments to your previous tool call couldn't be parsed correctly. Please ensure you properly escapse quotes and construct a valid JSON payload."
                        }
                    )
                    n_retry -= 1
                    continue
        # Append GPT response to next payload
        self.tool_res_payload.append(
            {
            "role": "assistant",
            "content": response.message.content if response.message.content is not None else ""
            }
        )

        if response.finish_reason == "length":
            # Determine if we're truncated
            self.truncated = True
            logger.warn("Response truncated due to length, Terminating!")
        # Recursive call if tool calls in response
        elif response.finish_reason == "tool_calls":
            self._handle_tool_calls(response)
        
        # End Step
        self.curr_step += 1
    
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
    
    def reset(self) -> None:
        self.tool_res_payload = []
        return super().reset()

    def format_prompt(self) -> str:
        return re.sub("\s+", " ", self.BASE_PROMPT).format(question=self.question)
    
    @staticmethod
    def _subprocess_tool_call_on_file(tool_input: str, cmd_args: list[str], output_type: Literal["stdout", "file"] = "stdout") -> str:
        """
        A helper function that writes `tool_input` to a file and runs a python module on that file.
        :param tool_input (str): A string to pass as input to the tool (this is likely code)
        :param cmd_args (list[str]): Command-line args between the python -m call and the file name (should include the python module to call and any additional arguments)
        :param output_type (str): The output to return (either stdout+error, or contents of the tempfile, if this is modified)
        Returns stdout and stderr concatenated into a string and separated by a newline
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

    def __init__(self, question: str, model_name: str, llm: openai.OpenAI, env: gym.Env):
        self.env = env
        super().__init__(question, model_name, llm)
    
    def is_truncated(self) -> bool:
        # NOTE: I think they also checked that prompt length
        # was under a certain value here, but that'd mean
        # importing tiktoken and computing it each step
        return super().truncated() and not self.correct

    def prompt_agent(self, prompt: dict[str, str] | list[dict[str, str]], n_tok: int = 100, **oai_kwargs) -> str:
        out_msg = super().prompt_agent(prompt, n_tok, **oai_kwargs)

        out = self.clean_response(out_msg.message.content) + "\n"

        return out

    def reset(self):
        super().reset()
        self.env.reset()
        self.correct = False