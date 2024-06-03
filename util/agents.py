"""
Agents (React & Relexion)
(based from Reflexion repo, all credit to Noah Shinn and team)
https://github.com/noahshinn/reflexion
"""

import abc
import logging
import os
from typing import Literal, Union

import gymnasium as gym
import openai

logger = logging.getLogger(__name__)


class Agent(metaclass=abc.ABCMeta):
    finished: bool = False
    terminated: bool = False
    truncated: bool = False
    curr_step: int = 1
    scratchpad: str = ""
    BASE_PROMPT: str = ""
    SYSTEM_PROMPT: str = ""

    def __init__(
        self, question: str, model_name: str, llm: openai.OpenAI, env: gym.Env
    ):
        self.question = question
        self.llm = llm
        self.model_name = model_name
        self.env = env
        self.reset()


    def run(self, reset: bool = False) -> None:
        if reset:
            self.reset()

        while not (self.is_terminated() or self.is_truncated()):
            logger.debug(f"Running step {self.curr_step}.")
            self.step()

    @abc.abstractmethod
    def step(self):
        pass

    def prompt_agent(self, prompt: str, n_tok: int = 100) -> str:
        logger.debug(f"Sending prompt to LLM:\n{prompt}")
        try:
            res = self.llm.chat.completions.create(
                prompt, model=self.model_name, max_tokens=n_tok
            )
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out = res["choices"][0]["message"]["content"]

        logger.debug(f"Received response: {out}")
        return out

    @abc.abstractmethod
    def format_prompt(self, **kwargs) -> str:
        pass

    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        # NOTE: I think they also checked that prompt length
        # was under a certain value here, but that'd mean
        # importing tiktoken and computing it each step
        return self.truncated and not self.finished

    def reset(self) -> None:
        self.scratchpad = ""
        self.curr_step = 1
        self.truncated = False
        self.terminated = False
        self.finished = False
        self.env.reset()

    def dump(self, outfile: Union[str, os.PathLike]) -> None:
        """
        Dump scratchfile to disk
        """
        with open(outfile, "w", encoding="utf-8") as file:
            file.writelines(elem + "\n" for elem in self.scratchpad.split("\n"))


class ReactAgent(Agent):
    BASE_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
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
        logger.debug("thinking...")
        self.scratchpad += f"\nThought {self.curr_step}: "
        self.scratchpad += self.prompt_agent(self.format_prompt())

        # Act
        logger.debug("getting action...")
        self.scratchpad += f"\nAct {self.curr_step}: "
        action = self.prompt_agent(self.format_prompt())
        self.scratchpad += action

        # Observe
        logger.debug("executing action and recieving observation...")
        self.scratchpad += f"\nObservation {self.curr_step}: "
        obs, self.finished, self.terminated, self.truncated, self.curr_step = (
            self.env.step(action)
        )
        self.scratchpad += obs

    def format_prompt(self) -> str:
        """
        Format the base prompt with dynamic content
        using f-string plus kwargs
        """
        fmt_prompt = self.BASE_PROMPT.format(
            examples=self.examples, question=self.question, scratchpad=self.scratchpad
        )

        return fmt_prompt


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

    def run(self) -> None:
        """
        Run standard React logic, but add in a reflection step if the agent failed previously
        """
        if (self.is_terminated or self.is_truncated) and not self.env.is_correct():
            self.reflect()

        super().run()

    def format_prompt(self) -> str:
        """
        Format the base prompt with dynamic content
        using f-string plus kwargs
        """
        fmt_prompt = self.BASE_PROMPT.format(
            examples=self.examples, question=self.question, scratchpad=self.scratchpad
        )

        return fmt_prompt

    def format_reflection_prompt(self) -> str:
        """
        Format the reflection prompt with dynamic content
        using f-string plus kwargs
        """
        fmt_prompt = self.REFELECTION_PROMPT.format(
            examples=self.examples,
            question=self.question,
            scratchpad=self.scratchpad,
            reflections=self.reflection_str,
        )

        return fmt_prompt

    def compose_reflections(self) -> None:
        """
        From Reflections (or just prior attempt)
        """

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
                self.prompt_agent(self.format_reflection_prompt, n_tok=250)
            )
            self.reflection_str += self.REFLECTION_HEADER + "\n"
            self.reflection_str += "\n- ".join(self.reflections)
        elif self.strategy == "last_attempt_reflexion":
            self.reflection_str += self.LAST_TRIAL_HEADER + "\n"
            self.reflection_str += f"Question: {self.question}\n"
            self.reflection_str += self.scratchpad
            self.reflection_str += "(END PREVIOUS TRIAL)\n"
            self.reflections.append(
                self.prompt_agent(self.format_reflection_prompt, n_tok=250)
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
