"""
Structured prediction agents (predicting from a list of choices)
Sean Browning (oet5)
"""

import logging
from typing import Callable, List, Literal, Optional, Any, Type

import pydantic

from ..abstract import _StoppingCondition, _Provider
from .base import StructuredOutputAgent

logger = logging.getLogger(__name__)


class PredictionAgent(StructuredOutputAgent):
    """
    A language agent which returns a structured prediction given a set of labels.

    This is a subclass to StructuredOutputAgent, and is provided for convenience
    when output is limited to a set of labels, thus the response model can be built on the fly.

    The result from the agent should be a `dict[str, list]` with one element, "labels", of `len(df.height)`
    """

    def __init__(
        self,
        labels: list[str],
        expected_len: Optional[int] = None,
        stopping_condition: Optional[_StoppingCondition] = None,
        model_name: Optional[str] = None,
        provider: Optional[Type[_Provider]] = None,
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Callable]] = None,
        oai_kwargs: Optional[dict[str, Any]] = None,
        **fmt_kwargs,
    ):
        """
        A language agent which returns a structured prediction given a set of choices in `labels`.


        :param labels: The set of categorical labels the model can choose from
        :param int expected_len: Optional length constraint on the response_model (OpenAI API doesn't allow maxItems parameter in schema so this is checked post-hoc in the Pydantic BaseModel)
        :param _StoppingCondition stopping_condition: A handler that signals when an Agent has completed the task (optional)
        :param str model_name: Name of model to use (or deployment name for AzureOpenAI) (optional if provider is passed)
        :param Type[_Provider] provider: Instantiated OpenAI instance to use (optional)
        :param List[dict] tools: List of tools the agent can call via response (optional)
        :param List[Callable] callbacks: List of callbacks to evaluate at end of run (optional)
        :param dict[str, any] oai_kwargs: Dict of additional OpenAI arguments to pass thru to chat call
        :param fmt_kwargs: Additional named arguments which will be inserted into the :func:`BASE_PROMPT` via fstring
        """
        self.labels = labels
        response_model = self._build_pydantic_model(length_constraint=expected_len)

        super().__init__(
            response_model,
            model_name=model_name,
            stopping_condition=stopping_condition,
            provider=provider,
            tools=tools,
            callbacks=callbacks,
            oai_kwargs=oai_kwargs,
            **fmt_kwargs,
        )

    def _build_pydantic_model(
        self, length_constraint: Optional[int] = None
    ) -> type[pydantic.BaseModel]:
        """
        Construct a pydantic model that we'll use to force the LLM to return a structured response
        """

        class classify(pydantic.BaseModel):
            # NOTE: This is kind of hacky since we're using runtime type-hints
            labels: List[Literal[tuple(self.labels)]] = pydantic.Field( #type: ignore
                description="Classify the input data into one of the possible categories"
            )

            @pydantic.model_validator(mode="after")
            def check_len(self):
                """
                Possibly check len
                """
                if length_constraint and len(self.labels) != length_constraint:
                    raise ValueError(
                        f"Expected labels to be of length {length_constraint} but got {len(self.labels)}"
                    )

                return self

        return classify


class PredictionAgentWithJustification(PredictionAgent):
    """
    A PredictionAgent which returns both a structured label output along with a short text justification for the prediction.

    The label and justification are supplied in the same tool call / response body rather than in separate messages to improve coherence.

    This has the additional stipulation that the number of labels and justifications must agree in number, which is enforced by pydantic post-hoc.
    """

    def _build_pydantic_model(self, length_constraint: Optional[int] = None):
        """
        Construct a pydantic model that we'll use to force the LLM to return a structured response.
        This will also include a justification for the classification.
        """

        class classify(pydantic.BaseModel):
            # NOTE: This is kind of hacky since we're using runtime type-hints
            labels: List[Literal[tuple(self.labels)]] = pydantic.Field( #type: ignore
                description="Classify the input data into one of the possible categories"
            )
            justification: List[str] = pydantic.Field(
                description="SHORT description explaining your reasoning for the classfication"
            )

            @pydantic.model_validator(mode="after")
            def check_len(self):
                """
                Check that justification and labels agree in length
                (and optionally that they match length constraint, if passed)
                """
                if len(self.justification) != len(self.labels):
                    raise ValueError(
                        f"There should be a justfication for each label assigned, but the counts differed: (labels: {len(self.labels)}, justifications: {len(self.justification)})"
                    )
                if length_constraint and len(self.justification) != length_constraint:
                    raise ValueError(
                        f"Expected exactly {length_constraint} labels and justifications, but got: (labels: {len(self.labels)}, justifications: {len(self.justification)})"
                    )

                return self

        return classify
