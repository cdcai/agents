from .abstract import _StoppingCondition
from pydantic import BaseModel
from typing import Optional

class StopOnStep(_StoppingCondition):
    """
    Stop language agent on a given step

    :param int step: Number of steps the Agent should run before terminating
    """

    def __init__(self, step: int = 1):
        self.step = step
    
    def __call__(self, cls, response) -> str:
        if cls.curr_step >= self.step:
            return response.message.content
        else:
            return None

class StopOnBaseModel(_StoppingCondition):
    """
    A stopping condition that checks whether the final tool call was an instance of the provided class.

    Useful for terminating once we receive correctly parsed / structured output
    """
    def __init__(self, answer_cls: BaseModel):
        self.answer_cls = answer_cls

    def __call__(self, cls, response) -> Optional[dict]:
        if isinstance(cls.tool_res_payload[-1], self.answer_cls):
            return cls.tool_res_payload[-1].model_dump() # pydantic.BaseModel.model_dump() -> dict[str, Any]
        else:
            return None

class StopNoOp(_StoppingCondition):
    """
    A stopping condition which always returns None.

    This is used when answer / stop handling is done internal to the Agent
    """

    def __init__(self):
        pass

    def __call__(self, cls, response):
        return None
