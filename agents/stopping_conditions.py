"""
Stopping conditions for language agents
"""

from .abstract import _StoppingCondition
from pydantic import BaseModel
from typing import Optional

class StoppingCondition(_StoppingCondition):
    """
    A callable that contains some logic to determine whether a language agent
    has finished it's run. This is called at the end of every turn via the `__call__` method.
    
    This class is abstract and should be subclassed with requisite `__call__` and `__init__` methods.

    The main call should always return the final answer, if we've finished the run, or None otherwise
    """
    pass

class StopOnStep(_StoppingCondition):
    """
    Stop language agent on a given step

    :param int step: Number of steps the Agent should run before terminating
    """

    def __init__(self, step: int = 1):
        self.step = step
    
    def __call__(self, cls, response) -> Optional[str]:
        if cls.curr_step >= self.step:
            return response.message.content
        else:
            return None

class StopOnDataModel(_StoppingCondition):
    """
    A stopping condition that checks whether the final tool call was an instance of the provided class.

    Useful for terminating once we receive correctly parsed / structured output
    """
    def __init__(self, answer_cls: type[BaseModel]):
        self.answer_cls = answer_cls

    def __call__(self, cls, response) -> Optional[dict]:
        if len(cls.tool_res_payload) and isinstance(cls.tool_res_payload[-1]["content"], self.answer_cls):
            return cls.tool_res_payload[-1]["content"].model_dump() # pydantic.BaseModel.model_dump() -> dict[str, Any]
        else:
            return None

class StopNoOp(_StoppingCondition):
    """
    A stopping condition which always returns None.

    This is used when answer / stop handling is done internal to the Agent
    """

    def __init__(self):
        pass

    def __call__(self, cls, response) -> None:
        return None
