"""
Agents (React & Relexion)
(based from Reflexion repo, all credit to Noah Shinn and team)
https://github.com/noahshinn/reflexion
"""

import logging
from typing import Literal

import gymnasium as gym
import openai

from .base_agents import EnvAgent

logger = logging.getLogger(__name__)

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
        action = self.prompt_agent(self.get_next_messages())
        self.scratchpad += action

        # Observe
        logger.info("executing action and recieving observation...")
        self.scratchpad += f"\nObservation {self.curr_step}: "
        obs, self.correct, self.terminated, self.truncated, self.curr_step = (
            self.env.step(action)
        )
        self.scratchpad += obs + "\n"

    def format_prompt(self) -> str:
        """
        Format the base prompt with dynamic content
        using f-string plus kwargs
        """
        return self.BASE_PROMPT.format(
            examples=self.examples, question=self.question, scratchpad=self.scratchpad
        )


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

    def format_prompt(self) -> str:
        """
        Format the base prompt with dynamic content
        using f-string plus kwargs
        """
        return self.BASE_PROMPT.format(
            examples=self.examples,
            question=self.question,
            scratchpad=self.scratchpad,
            reflections=self.reflection_str
        )

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
