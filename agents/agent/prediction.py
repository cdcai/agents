"""
Structured prediction agents (predicting from a list of choices)
Sean Browning (oet5)
"""

import logging
from typing import Callable, List, Literal, Optional, Any

import openai
import polars as pl
import pydantic

from ..abstract import _StoppingCondition
from .base import Agent, StructuredOutputAgent

logger = logging.getLogger(__name__)


class PredictionAgent(StructuredOutputAgent):
    """
    A language agent which returns a structured prediction given a polars DataFrame input.
    
    The result from the agent should be a `dict[str, list]` with one element, "labels", of `len(df.height)`
    """

    def __init__(
        self,
        df: pl.DataFrame,
        labels: list[str],
        model_name: str,
        stopping_condition: Optional[_StoppingCondition] = None,
        llm: Optional[openai.AsyncOpenAI] = None,
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Callable]] = None,
        oai_kwargs: Optional[dict[str, Any]] = None,
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
        response_model = self._build_pydantic_model()

        super().__init__(
            response_model,
            model_name=model_name,
            expected_len=self.expected_n,
            stopping_condition=stopping_condition,
            llm=llm,
            tools=tools,
            callbacks=callbacks,
            oai_kwargs=oai_kwargs,
            df=df.write_ndjson(),
            **fmt_kwargs
        )

    def _build_pydantic_model(self) -> type[pydantic.BaseModel]:
        """
        Construct a pydantic model that we'll use to force the LLM to return a structured response
        """

        response_model = pydantic.create_model(
            "classify",
            labels=(
                # HACK: This is a bodge to build a model with labels only known at runtime
                # but it will fail static typing in doing so
                List[Literal[tuple(self.labels)]],
                pydantic.Field(
                    alias="labels",
                    description="Classify the input data into one of the possible categories",
                ),
            ),
        )

        return response_model


class PredictionAgentWithJustification(PredictionAgent):
    """
    A PredictionAgent which returns both a structured label output along with a short text justification for the prediction.

    The label and justification are supplied in the same tool call / response body rather than in separate messages to improve coherence.
    """
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