"""
Callbacks for Agents
"""

from typing import Any

from .abstract import _Callback
from .agent import Agent


class AgentCallback[CallbackAgentT: Agent](_Callback[Agent]):
    """
    Call another agent with the answer and scratchpad of a completed agent
    """

    callback_agent: CallbackAgentT

    def __init__(self, agent_class: type[CallbackAgentT], **agent_kwargs: Any) -> None:
        """
        Create an Agent Callback, i.e. an Agent which will be called at the
        end of an Agent run with the answer and scratchpad.

        The provided `agent_class` will be initialized at the end of the run with
        `answer` and `scratchpad` variables passed to format the `BASE_PROMPT`

        Possible use cases could include reflection/reaction on llm agent feedback,
        summarization of task, etc.

        :param Agent agent_class: Uninitialized Agent class to use in callback
        :param agent_kwargs: All named arguments with which to initialize agent_class
        """
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs

    async def __call__(self, cls: Agent, answer: Any, scratchpad: str) -> None:
        """
        Run new callback agent on calling agent's answer and scratchpad and append output.
        """

        self.callback_agent = self.agent_class(
            **self.agent_kwargs, answer=answer, scratchpad=scratchpad
        )
        await self.callback_agent.run()
        cls.callback_output.append(self.callback_agent.answer)
