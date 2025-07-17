"""
Callbacks for Agents
"""

import abc

from .agent import Agent


class _Callback(metaclass=abc.ABCMeta):
    """
    A Callback virtual class
    """

    @abc.abstractmethod
    def __call__(self, cls: Agent, answer: str, scratchpad: str):
        """
        Primary method called by agent during callback process

        :param Agent cls: Instantitated class of calling agent for possible modification
        :param str answer: The final response of the calling Agent
        :param str scratchpad: The full interaction history of the calling Agent

        """
        raise NotImplementedError()


class AgentCallback(_Callback):
    """
    Call another agent with the answer and scratchpad of a completed agent
    """

    def __init__(self, agent_class: type[Agent], **agent_kwargs):
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

    async def __call__(self, cls: Agent, answer: str, scratchpad: str) -> None:
        """
        Run new callback agent on calling agent's answer and scratchpad and append output.
        """

        self.callback_agent = self.agent_class(
            **self.agent_kwargs, answer=answer, scratchpad=scratchpad
        )
        await self.callback_agent.run()
        cls.callback_output.append(self.callback_agent.answer)
