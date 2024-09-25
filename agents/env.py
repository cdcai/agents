"""
A Wikipedia QA Gymnasium / Environment
(almost entirely lifted / based from Reflexion repo, all credit to Noah Shinn and team)
https://github.com/noahshinn/reflexion
"""

import logging
import re
import string
from typing import Union

import gymnasium as gym
from langchain_community.docstore.wikipedia import Wikipedia

from .react import DocstoreExplorer
from .ssl_tools import no_ssl_verification

logger = logging.getLogger(__name__)


class WikiQAEnv(gym.Env):
    def __init__(self, question: str, truth: str, max_steps: int = 6):
        self.explorer = DocstoreExplorer(Wikipedia())
        self.question = question
        self.truth = truth
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        self.curr_step = 1
        self.terminated = False
        self.answer = ""

    def step(self, action: str) -> tuple[str, bool, bool, bool, int]:
        """
        One step running the environment
        """
        reward: bool = False
        action_type, arg = self.parse_action(action)

        logger.debug(f"Step {self.curr_step}: got action, {action_type}; arg {arg}")

        if action_type == "Finish":
            self.answer = arg

            if reward := self.is_correct():
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"

            logger.info(f"Final answer given: {arg}, {obs}")
            self.terminated = True

        elif action_type == "Search":
            try:
                # HACK: CDC Uses self-signed certs in resolution
                with no_ssl_verification():
                    obs = self.explorer.search(arg).strip("\n").strip()
            except Exception as e:
                logger.debug(e)
                obs = "Could not find that page, please try another search."

        elif action_type == "Lookup":
            try:
                obs = self.explorer.lookup(arg).strip("\n").strip()
            except Exception as e:
                logger.debug(e)
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."

        terminated = self.terminated
        truncated = self.is_truncated()

        self.curr_step += 1

        logger.info(f"Observed: {obs}; truncated: {truncated}; finished: {terminated}")
        return (obs, reward, terminated, truncated, self.curr_step)

    def is_truncated(self) -> bool:
        return self.curr_step >= self.max_steps

    @staticmethod
    def parse_action(string: str) -> Union[tuple[str, str], tuple[None, None]]:
        """
        'action[argument]' -> ('action', 'argument')
        """
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, string)

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument

        else:
            return None, None

    def is_correct(self) -> bool:
        return self.normalize_answer(self.truth) == self.normalize_answer(self.answer)

    @staticmethod
    def normalize_answer(ans: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(ans))))


class WikiQAEnvActive(WikiQAEnv):
    """
    WikiQA Gymnasium with a Human-in-the-loop to judge whether response is correct
    when evaluating final anwser
    """

    def is_correct(self) -> bool:
        res = input(
            f"===Answer===\n{self.normalize_answer(self.answer)}\n===\nIs answer correct? ([T]rue, [F]alse)"
        ).lower()

        while True:
            if res in ["true", "t"]:
                res_bool = True
                break
            elif res in ["false", "f"]:
                res_bool = False
                break
            else:
                res = (
                    input("Invalid option, please return either [t]rue/[f]alse")
                    .lower()
                    .strip()
                )

        return res_bool
