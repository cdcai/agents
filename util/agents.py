"""
Agents (React & Relexion)
(based from Reflexion repo, all credit to Noah Shinn and team)
https://github.com/noahshinn/reflexion
"""

import abc
import logging
import os
import re
from typing import Literal, Union, Optional

import backoff
import gymnasium as gym
import openai
from azure.identity import ClientSecretCredential

logger = logging.getLogger(__name__)

class Agent(metaclass=abc.ABCMeta):
    terminated: bool = False
    truncated: bool = False
    curr_step: int = 1
    scratchpad: str = ""
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
    def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: int = 100) -> str:
        
        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if isinstance(prompt, dict):
            prompt = [prompt]

        try:
            res = self.llm.chat.completions.create(
                messages=prompt, model=self.model_name, max_tokens=n_tok
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
                    max_tokens=n_tok
                )
            else:
                raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out_msg = res.choices[0].message

        out = self.clean_response(out_msg.content) + "\n"

        logger.info(f"Received response: {out}")
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
    An agent which can utilize tool calls
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
    answer : str = ""

    @backoff.on_exception(backoff.expo, openai.APIError, max_tries=3, logger=logger)
    def prompt_agent(self, prompt: Union[dict[str, str], list[dict[str, str]]], n_tok: int = 100, tool_use : Literal["required", "auto"] = "required"):
        
        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if isinstance(prompt, dict):
            prompt = [prompt]

        try:
            res = self.llm.chat.completions.create(
                messages=prompt,
                model=self.model_name,
                functions=self.TOOLS,
                tool_choice=tool_use,
                max_tokens=n_tok
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
                    max_tokens=n_tok
                )
            else:
                raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        return res

    def step(self):
        # Pull base query + system messages
        # (abstract)
        llm_prompt_input = self.format_prompt()

        # If we have existing tool response messages, append them
        # and reset before we recieve new data
        if len(self.tool_res_payload):
            llm_prompt_input.extend(self.tool_res_payload)
            self.tool_res_payload = []
        
        # Send off messages for reply
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"
    
        response = self.prompt_agent(llm_prompt_input, n_tok = 2 * self.chunk_max)

        # Determine if we're truncated
        self.truncated = response.finish_reason == "length"

        # Append GPT response to next payload
        self.tool_res_payload.append(response.choices[0].message)

        # Recursive call if tool calls in response
        if response.requires_action:
            for tool in response.required_action.submit_tool_outputs.tool_calls:
                # Try to call tool, if present, else raise.
                try:
                    fun = getattr(self, tool.function.name)
                    kwargs : dict[str, any] = tool.function.arguments
                    kwargs.update({"id": tool["id"]})

                    logger.info(f"Got tool call: {fun}({kwargs})")

                    tool_result = fun(**kwargs)
                    self.tool_res_payload.append(tool_result)
                except Exception as e:
                    logger.error(f"Tool call {tool} failed.")
                    raise e
        
        # End Step
        self.curr_step += 1
    
    def call_submit(self, input: str) -> None:
        """
        Final response call, which terminates further processing
        """
        out_msg = self.clean_response(input)
        logger.info(f"Received final response: {out_msg}")
        self.answer = out_msg
        self.terminated = True
    
    def reset(self) -> None:
        self.answer = ""
        self.tool_res_payload = []
        return super().reset()

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

    def reset(self):
        super().reset()
        self.env.reset()
        self.correct = False

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
