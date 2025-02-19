"""
Entailment Agent (which provides log probability of choice as well)
Sean Browning (oet5)
"""

import logging
from typing import Tuple
from ..abstract import _StoppingCondition, _Provider
from .base import Agent

logger = logging.getLogger(__name__)


class EntailmentAgent(Agent):
    log_probs: Tuple[str, float]

    """
    An agent that evaluates entailment and returns either true / false given two statements
    along with the log-probability

    The base class has three format variables to provide:
    
    {context} - Context required to make a determination
    {statement_a} - First statement
    {statement_b} - Second statement
    """
    SYSTEM_PROMPT = """
    You are a language agent who determines whether two statements, A & B are entailed (such that A being true presupposes B being true as well).
    You will reply only with a single token, either True or False according to whether the statements are entailed or not.
""".strip().replace("    ", "")
    
    BASE_PROMPT = """
    {context}

    Statement A: {statement_a}

    Statement B: {statement_b}

    Does A entail B?
""".strip().replace("    ", "")

    def __init__(self, stopping_condition, model_name=None, provider=None, tools=None, callbacks=None, oai_kwargs=None, **fmt_kwargs):

        if tools is not None:
            raise RuntimeError("EntailmentAgent returns LogProbs, so tool use isn't available")
        
        entailment_oai_kwargs = {"logprobs": True, "top_logprobs": 2}

        if oai_kwargs is None:
            oai_kwargs = entailment_oai_kwargs
        else:
            oai_kwargs.update(entailment_oai_kwargs)

        super().__init__(stopping_condition, model_name, provider, tools, callbacks, oai_kwargs, **fmt_kwargs)
    
    def _check_stop_condition(self, response):
        super()._check_stop_condition(response)

        # Also append log probs if we're finished
        if self.answer is not None:
            for logprob in response.logprobs.content[0].top_logprobs:
                self.log_probs.append( (logprob.token, logprob.logprob) )
    
    def reset(self):
        self.log_probs = []
        return super().reset()