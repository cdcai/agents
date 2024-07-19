"""
Agents (React & Relexion)
(based from Reflexion repo, all credit to Noah Shinn and team)
https://github.com/noahshinn/reflexion
"""

import abc
import logging
import json
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
from openai import OpenAI

# Regex to extract python script from OpenAI response
# (allowing multiple cases because GPT-4 isn't consistent)
PYSCRIPT_CONTENT = re.compile(r"```[pP][ython]*\n(.+?)```", re.DOTALL)

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
    def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: int = 100, **oai_kwargs):
        
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
    def format_prompt(self, **kwargs) -> dict[str, str]:
        raise NotImplementedError()

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

class ToolAwareAgent(Agent):
    """
    A base-class for an agent which can utilize tool calls.
    Subclasses would be expected to extend the TOOLS attribute to include additional
    tools / functions. Each added tool should be appended to the base attribute at init.
    
    In addition, each added tool must have a corresponding class method that can be invoked
    during step() if the GPT calls it.
    """

    TOOLS : list[dict] = [
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

    def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: Optional[int] = None, tool_use : Literal["required", "auto", "none"] = "auto", **oai_kwargs):
        
        out = super().prompt_agent(prompt, n_tok, tools=self.TOOLS, tool_choice=tool_use, **oai_kwargs)

        # attempt to parse tool call arguments
        if out.finish_reason == "tool_calls":
                for i, tool in enumerate(out.message.tool_calls):
                    out.message.tool_calls[i].function.arguments = json.loads(tool.function.arguments)

        return out
    def step(self):
        # Pull base query + system messages
        # (abstract)
        llm_prompt_input = self.format_prompt()

        # If we have existing tool response messages, append them
        if len(self.tool_res_payload):
            llm_prompt_input.extend(self.tool_res_payload)
        
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

    def format_prompt(self) -> list[dict[str, str]]:
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": re.sub("\s+", " ", self.BASE_PROMPT).format(question=self.question)}
        ]

        return out
    
    @staticmethod
    def _subprocess_tool_call_on_file(tool_input: str, cmd_args: list[str]) -> str:
        """
        A helper function that writes `tool_input` to a file and runs a python module on that file.
        :param tool_input (str): A string to pass as input to the tool (this is likely code)
        :param cmd_args (list[str]): Command-line args between the python -m call and the file name (should include the python module to call and any additional arguments)
        
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

        return "\n".join([str(out.stdout), str(out.stderr)])

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

class SASConvertAgent(Agent):
    """
    The language agent responsible for producing Python files from input SAS scripts.
    """
    SYSTEM_PROMPT: str = "You are a coding expert in the statistical language, SAS, and Python and able to assist with converting between the two."
    BASE_PROMPT: str = """
    I am converting an existing SAS script into Python. The file is quite long, so you may only send part of it at a time.
    The SAS script performs several ETL steps on data files which contain EHR data.
    Please help me re-write this script using syntactically correct Python and any data science libraries (ex. numpy, pandas, polars) that might be needed.
    If more than one script is needed, please place the file contents in separate ```python blocks.
    Please provide ONLY the output code.

    Here is my SAS file:
    ```sas
    {question}
    ```
    """
    py_scripts : list[str] = []

    def __init__(self, question: str, model_name: str, llm: OpenAI | None = None, chunk_max : int = 2500):
        # Get tokenizer to handle chunking responses if needed
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.prompt_len = len(self.tokenizer.encode(self.BASE_PROMPT.format(question="") + self.SYSTEM_PROMPT))
        # Also save full input to full_question attribute since we'll
        # overwrite self.question if the resulting payload is too large
        self.full_question = question
        self.chunk_max = chunk_max

        super().__init__(question, model_name, llm)

    def run(self, reset: bool = False) -> None:
        super().run(reset)

        self.combine_pyscripts()

    def step(self):
        # Prompt LLM for first-pass
        llm_prompt_input = self.format_prompt()
        first_answer = self.prompt_agent(llm_prompt_input, n_tok = 2 * self.chunk_max).message.content
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"

        # Attempt to parse python scripts in response
        ret_scripts = self.extract_pyscripts(first_answer)
        self.scratchpad += f"\n=== Answer {self.curr_step} =====\n"
        self.scratchpad += "\n".join(ret_scripts)
        self.scratchpad += "\n===================================\n"

        self.py_scripts.extend(ret_scripts)

        # End run
        self.terminated = len(self.question) == 0
        self.curr_step += 1

    def extract_pyscripts(self, answer: str) -> str:

        return [script.group(1) for script in PYSCRIPT_CONTENT.finditer(answer)]

    def combine_pyscripts(self) -> None:
        """
        Combine returned python scripts into a single string for writeout
        """
        self.answer = "\n".join(self.py_scripts)

    def fetch_last_translation(self) -> Optional[dict[str, str]]:
        if len(self.py_scripts):
            out = {
                "role": "user",
                "content": """
                    Here is the last piece of translated code you produced in a previous step for context:
                    ```python
                    {}
                    ```
                    """.format(self.py_scripts[-1]).strip()
            }
        else:
            out = None
        
        return out

    def get_prompt_len(self) -> int:
        """
        Return base prompt length before using fstring to fill in template.
        (also accounting for possible previous translation context)
        """
        if (last_translation_message := self.fetch_last_translation()) is not None:
            prompt_len = self.prompt_len + len(self.tokenizer.encode(last_translation_message["content"]))
        else:
            prompt_len = self.prompt_len
        
        return prompt_len

    def format_prompt(self) -> list[dict[str, str]]:
        """
        Insert SAS code into prompt, and return list of messages
        to send to chatGPT.
        """
        prompt_len = self.get_prompt_len()

        # Split by code block
        # (at least one empty line between next code block)
        # This will theoretically require fewer calls to reach a response of appropriate length
        script_inq = re.split("\n{2,}?", self.question)

        excess_lines = []
        # pop line by line until we have a message payload less than the requested max
        while len(self.tokenizer.encode("\n\n".join(script_inq))) + prompt_len > self.chunk_max:
            # Store excess message payload in question object
            excess_lines.append(script_inq.pop())
        
        # Reverse things around and re-join to string
        # to get things the right way around
        self.question = "\n\n".join(reversed(excess_lines))

        # Construct query
        fmt_prompt = re.sub("\s+", " ", self.BASE_PROMPT).format(question="\n\n".join(script_inq)).strip()
        
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": fmt_prompt}
        ]

        if (last_translation_message := self.fetch_last_translation()) is not None:
            out.append(last_translation_message)

        return out

    def reset(self) -> None:
        self.py_scripts = []
        super().reset()

class PythonRefineAgent(ToolAwareAgent):
    SYSTEM_PROMPT: str = "You are a Python coding expert and can identify and correct syntactial mistakes and fix incomplete code using standard conventions."
    BASE_PROMPT: str = """
    I have converted part of an existing SAS script into Python. Due to length, the script may have been translated in chunks and the final results concatenated into a single script.
    This script may contain syntax errors, be poorly commented, have undefined global references, or duplicative/un-used imports, etc.
    Please read this script and provide any corrections that may be needed. Please provide type hints, function docstrings, and guiding comments as needed.
    You may call the mypy and black tools to assist, and you may call both in parallel. If no changes are needed, provide the script back using the submit tool. Always check the file first before submitting.
    Please provide ONLY the output code marked in a code block, no additional commentary is needed.

    Here is the python file:
    ```python
    {question}
    ```
    """

    TOOLS = [
        # Mypy
        {
            "type": "function",
            "function": {
                "name": "call_mypy",
                "description": "Run the MyPy static typing tool on input code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to check using the mypy static typing tool"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        # Black
        {
            "type": "function",
            "function": {
                "name": "call_black",
                "description": "Run the black python code formater on input python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to check using black tool"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

    def __init__(self, question: str, model_name: str, llm: OpenAI | None = None):
        # Add additional tools from base class(call_submit)
        self.TOOLS.extend(super().TOOLS)

        super().__init__(question, model_name, llm)
    
    @staticmethod
    def call_black(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        try:
            import black
        except ImportError as e:
            logger.error("black is mising, please install it with `pip install black`")
            raise e

        return ToolAwareAgent._subprocess_tool_call_on_file(code, ["black", "--diff"])

    @staticmethod
    def call_mypy(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        try:
            import mypy
        except ImportError as e:
            logger.error("mypy is mising, please install it with `pip install mypy`")
            raise e

        return ToolAwareAgent._subprocess_tool_call_on_file(code, ["mypy", "--install-types", "--non-interactive"])

class ReactAgent(EnvAgent):
    BASE_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps.
    Thought can reason about the current situation, and Action can be three types: 
    (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    (2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
    (3) Finish[answer], which returns the answer and finishes the task.
    You may take as many steps as necessary, but only respond with the step requested at the end of this message.
    Here are some examples:
    {examples}
    (END OF EXAMPLES)

    Question: {question}{scratchpad}"""

    def __init__(
        self,
        question: str,
        examples: str,
        model_name: str,
        llm: openai.OpenAI,
        env: gym.Env,
    ):
        super().__init__(question, model_name, llm, env)
        self.examples = examples

    def step(self) -> None:
        """
        Main Agent interaction logic, each step consists of three sub-steps:
        1. Think
        2. Act
        3. Observe
        """

        # Think
        logger.info("thinking...")
        self.scratchpad += f"\nThought {self.curr_step}: "
        self.scratchpad += self.prompt_agent(self.format_prompt())

        # Act
        logger.info("getting action...")
        self.scratchpad += f"\nAct {self.curr_step}: "
        action = self.prompt_agent(self.format_prompt())
        self.scratchpad += action

        # Observe
        logger.info("executing action and recieving observation...")
        self.scratchpad += f"\nObservation {self.curr_step}: "
        obs, self.correct, self.terminated, self.truncated, self.curr_step = (
            self.env.step(action)
        )
        self.scratchpad += obs + "\n"

    def format_prompt(self) -> dict[str, str]:
        """
        Format the base prompt with dynamic content
        using f-string plus kwargs
        """
        fmt_prompt = self.BASE_PROMPT.format(
            examples=self.examples, question=self.question, scratchpad=self.scratchpad
        )

        return {"role": "user", "content": fmt_prompt}


class ReactandReflectAgent(ReactAgent):
    BASE_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""
    REFELECTION_PROMPT = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""
    FAILED_TRIAL_HEADER = (
        "You have attempted to answer the following question before and failed."
    )
    REFLECTION_HEADER = "The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question."
    LAST_TRIAL_HEADER = (
        "Below is the last trial where you attempted to answer the question."
    )
    reflections: list[str] = []
    reflection_str: str = ""

    def __init__(
        self,
        question: str,
        examples: str,
        reflection_strategy: Literal[
            "last_attempt", "reflexion", "last_attempt_reflexion"
        ],
        model_name: str,
        llm: openai.OpenAI,
        env: gym.Env,
    ):
        super().__init__(question, examples, model_name, llm, env)
        self.strategy = reflection_strategy
        pass

    def run(self, reset: bool = False) -> None:
        """
        Run standard React logic, but add in a reflection step if the agent failed previously
        """
        if (self.is_terminated() or self.is_truncated()) and not self.env.is_correct():
            self.reflect()

        super().run(reset)

    def format_prompt(self) -> dict[str, str]:
        """
        Format the base prompt with dynamic content
        using f-string plus kwargs
        """
        fmt_prompt = self.BASE_PROMPT.format(
            examples=self.examples,
            question=self.question,
            scratchpad=self.scratchpad,
            reflections=self.reflection_str
        )

        return {"role": "user", "content": fmt_prompt}

    def format_reflection_prompt(self) -> dict[str, str]:
        """
        Format the reflection prompt with dynamic content
        using f-string plus kwargs
        """
        fmt_prompt = self.REFELECTION_PROMPT.format(
            examples=self.examples,
            question=self.question,
            scratchpad=self.scratchpad
        )

        return {"role": "user", "content": fmt_prompt}

    def reflect(self) -> None:
        """
        Reflect on failure to hopefully provide clues for
        how to solve the problem in the next iteration
        """

        logger.debug("Reflecting.")

        self.reflection_str = self.FAILED_TRIAL_HEADER + "\n"

        if self.strategy == "last_attempt":
            self.reflection_str += self.LAST_TRIAL_HEADER + "\n"
            self.reflection_str += f"Question: {self.question}\n"
            self.reflection_str += self.scratchpad
            self.reflection_str += "(END PREVIOUS TRIAL)\n"
        elif self.strategy == "reflexion":
            self.reflections.append(
                self.prompt_agent(self.format_reflection_prompt(), n_tok=250)
            )
            self.reflection_str += self.REFLECTION_HEADER + "\n"
            self.reflection_str += "\n- ".join(self.reflections)
        elif self.strategy == "last_attempt_reflexion":
            self.reflection_str += self.LAST_TRIAL_HEADER + "\n"
            self.reflection_str += f"Question: {self.question}\n"
            self.reflection_str += self.scratchpad
            self.reflection_str += "(END PREVIOUS TRIAL)\n"
            self.reflections.append(
                self.prompt_agent(self.format_reflection_prompt(), n_tok=250)
            )
            self.reflection_str += self.REFLECTION_HEADER + "\n"
            self.reflection_str += "\n- ".join(self.reflections)
        else:
            raise NotImplementedError("Unknown Reflexion strategy: {self.strategy}")

        logger.debug(f"got reflection string:\n{self.reflection_str}")

    def reset(self) -> None:
        self.reflections = []
        self.reflection_str = ""
        super().reset()
