"""
Structured prediction agent (predicting from a list of choices)
Sean Browning (oet5)
"""

import logging
from typing import List, Literal, Optional

import openai
import polars as pl
import pydantic

from .base import ToolAwareAgent

logger = logging.getLogger(__name__)


class PredictionAgent(ToolAwareAgent):
    """
    A language agent which returns a structured prediction
    """

    answer: list[str]

    def __init__(
        self,
        prompt: str,
        persona: str,
        model_name: str,
        df: pl.DataFrame,
        labels: list[str],
        llm: openai.OpenAI | None = None,
        **oai_kwargs,
    ):
        """
        A ToolAwareAgent where the output is a prediction for each row of `df` from one of the choices in `labels`

        :param prompt: A prompt which becomes the first user message and should explain how to classify `df`
        :param persona: A system message / persona
        :param model_name: The Azure OpenAI model name to use
        :param df: The data to classify
        :param labels: The set of labels the model will choose from
        :param llm: (optional) An OpenAI instance, otherwise one will be created via ClientSecret credentials internally
        :param oai_kwargs: Additional key word args to pass to the OpenAI Chat Completion API (temperature, top_k, etc.)

        """
        self.labels = labels
        self.expected_n = df.height
        self.df = df
        self.SYSTEM_PROMPT = persona
        self.BASE_PROMPT = prompt
        self._build_pydantic_model()
        super().__init__(
            question=prompt,
            model_name=model_name,
            llm=llm,
            tools=self.response_tool,
            submit_tool=False,
            **oai_kwargs,
        )

    def _build_pydantic_model(self):
        """
        Construct a pydantic model that we'll use to force the LLM to return a structured resposne
        """

        self.response_model = pydantic.create_model(
            "classify",
            labels=(
                List[Literal[tuple(self.labels)]],
                pydantic.Field(
                    alias="labels",
                    description="Classify the input data into one of the possible categories",
                ),
            ),
        )

        self.response_tool = openai.pydantic_function_tool(
            self.response_model,
            name="classify",
            description="Classify the data using one of the possible categories",
        )

    def format_prompt(self) -> str:
        """
        Return our prompt as-is, because we pre-formatted it
        """
        return (
            self.BASE_PROMPT
            + f"\n{self.expected_n} potential cases:\n"
            + self.df.write_ndjson()
        )

    def classify(self, labels: list[str]) -> Optional[str]:
        """
        Function to "classify" ADE, which inspects that the LLM provided a response of the correct length and used only the labels provided
        """
        if len(labels) != self.expected_n:
            logger.warning(f"Invalid return length: {len(labels)}, retrying.")
            return f"Input was of length {self.expected_n} but you returned a response of length {len(labels)}. Please try again."

        try:
            # End our run if we make it through this
            parsed_args = self.response_model(labels=labels)
            self.answer = parsed_args.labels
            self.terminated = True
            self.scratchpad += "===== Answer ==========\n"
            self.scratchpad += str(self.answer)
            logger.info("Got valid response, terminating.")
        except pydantic.ValidationError:
            logger.warning(f"Response didn't pass pydantic validation, retrying.")
            # HACK: the default message from pydantic would be pretty long and lead to context length issues
            # so I'm making my own
            invalid_labels = [val for val in labels if val not in self.labels]
            return f"Pydantic validation of function call failed because you passed the following invalid label(s): {invalid_labels}. Please retry using ONLY the labels allowed."

    def reset(self) -> None:
        self.tool_res_payload = []
        self.scratchpad = ""
        self.answer = []
        self.curr_step = 1
        self.truncated = False
        self.terminated = False

    def run(self, reset: bool = False, steps: Optional[int] = None) -> None:
        """
        Run the agent, optionally running only a fixed number of steps
        """
        if reset:
            self.reset()

        if steps is not None:
            for _ in range(steps):
                logger.debug(f"Running step {self.curr_step}.")
                self.step()
                if self.is_terminated() or self.is_truncated():
                    break
        else:
            super().run(reset)
