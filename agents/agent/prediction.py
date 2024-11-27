"""
Structured prediction agents (predicting from a list of choices)
Sean Browning (oet5)
"""

import logging
from typing import List, Literal, Optional, Callable

import openai
import polars as pl
import pydantic

from .base import Agent
from ..abstract import _StoppingCondition
from ..stopping_conditions import StopNoOp

logger = logging.getLogger(__name__)


class PredictionAgent(Agent):
    """
    A language agent which returns a structured prediction given a polars DataFrame input.
    
    The result from the agent should be a `list[str]` of `len(df.height)`
    """

    answer: list[str]

    def __init__(
        self,
        df: pl.DataFrame,
        labels: list[str],
        model_name: str,
        stopping_condition: Optional[_StoppingCondition] = None,
        llm: Optional[openai.AsyncOpenAI] = None,
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Callable]] = None,
        oai_kwargs: Optional[dict[str, any]] = None,
        **fmt_kwargs
    ):
        """
        An Agent where the output is a prediction for each row of `df` from one of the choices in `labels`
        
        
        :param df: The data to classify (the ndJSON format of this will be available as `df` to format the BASE_PROMPT)
        :param labels: The set of categorical labels the model can choose from
        :param str model_name: Name of OpenAI model to use (or deployment name for AzureOpenAI)
        :param _StoppingCondition stopping_condition: A handler that signals when an Agent has completed the task (optional)
        :param AsyncOpenAI llm: Instantiated OpenAI instance to use (optional)
        :param List[dict] tools: List of tools the agent can call via response (optional)
        :param List[Callable] callbacks: List of callbacks to evaluate at end of run (optional)
        :param dict[str, any] oai_kwargs: Dict of additional OpenAI arguments to pass thru to chat call
        :param fmt_kwargs: Additional named arguments which will be inserted into the :func:`BASE_PROMPT` via fstring
        """
        self.labels = labels
        self.expected_n = df.height
        self.response_tool = self._build_pydantic_model()

        if tools is None:
            tools = []
        
        tools.append(self.response_tool)

        if stopping_condition is not None:
            logger.warning("A stopping condition was passed to Prediction Agent. This may lead to unexpected results.")
        else:
            stopping_condition = StopNoOp()

        super().__init__(
            model_name=model_name,
            stopping_condition=stopping_condition,
            llm=llm,
            tools=tools,
            callbacks=callbacks,
            oai_kwargs=oai_kwargs,
            df=df.write_ndjson(),
            **fmt_kwargs
        )

    def _build_pydantic_model(self) -> dict:
        """
        Construct a pydantic model that we'll use to force the LLM to return a structured response
        """

        response_model = pydantic.create_model(
            "classify",
            labels=(
                List[Literal[tuple(self.labels)]],
                pydantic.Field(
                    alias="labels",
                    description="Classify the input data into one of the possible categories",
                ),
            ),
        )

        response_tool = openai.pydantic_function_tool(
            response_model,
            name="classify",
            description="Classify the data using one of the possible categories",
        )

        return response_tool

    def classify(self, labels: list[str]) -> Optional[str]:
        """
        Check classification. Inspects that the LLM provided a response of the correct length and used only the labels provided.
        
        HACK: The logic here is unfortunately doing double duty, because it's a tool call but also signaling a stop condition.
        I'd like to de-couple that at some point, but I'm not sure the best approach.
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


class PredictionAgentWithJustification(PredictionAgent):
    """
    A PredictionAgent which returns both a structured label output along with a short text justification for the prediction.

    The label and justification are supplied in the same tool call / response body rather than in separate messages to improve coherence.
    """
    output_len: int = 2 # Each sample has two components: a label, and a justification
    def _build_pydantic_model(self):
        """
        Construct a pydantic model that we'll use to force the LLM to return a structured response.
        This will also include a justification for the classification.
        """

        self.response_model = pydantic.create_model(
            "classify",
            labels=(
                List[Literal[tuple(self.labels)]],
                pydantic.Field(
                    alias="labels",
                    description="Classify the input data into one of the possible categories",
                )
            ),
            justification=(
                List[str],
                pydantic.Field(
                    alias="justification",
                    description="SHORT description explaining your reasoning for the classfication"
                )
            )
        )

        self.response_tool = openai.pydantic_function_tool(
            self.response_model,
            name="classify",
            description="Classify the data using one of the possible categories and, for each classification, provide a short description of your reasoning.",
        )

    def classify(self, labels: list[str], justification: list[str]) -> Optional[str]:
        """
        Check classification. Inspects that the LLM provided a response of the correct length and used only the labels provided.
        """
        if len(labels) != self.expected_n:
            logger.warning(f"Invalid return length: {len(labels)}, retrying.")
            return f"Input was of length {self.expected_n} but you returned a response of length {len(labels)}. Please try again."

        if len(justification) != len(labels):
            logger.warning(f"Invalid justification length: {len(justification)} != {len(labels)}, retrying.")
            return f"You should have provided one justification for each classification you provided ({len(labels)}), but I only recieved {len(justification)}. Please try again."
        try:
            # End our run if we make it through this
            parsed_args = self.response_model(labels=labels, justification=justification)
            self.answer = [(ans, just) for ans, just in zip(parsed_args.labels, parsed_args.justification)]
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
