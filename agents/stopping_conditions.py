from .abstract import _StoppingCondition

class StopOnStep(_StoppingCondition):
    """
    Stop language agent on a given step

    :param int step: Number of steps the Agent should run before terminating
    """

    def __init__(self, step: int = 1):
        self.step = step
    
    def __call__(self, cls, response) -> bool:
        return cls.curr_step >= self.step