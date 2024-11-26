from .abstract import _StoppingCondition

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

class StopNoOp(_StoppingCondition):
    """
    A stopping condition which always returns None.

    This is used when answer / stop handling is done internal to the Agent
    """

    def __init__(self):
        pass

    def __call__(self, cls, response):
        return None
