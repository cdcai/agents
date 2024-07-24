"""
Meta-algorithms and arrangements for agent systems
"""

from .agents import Agent, PersistentAgent
import logging

logger = logging.getLogger(__name__)

class Reflexion(Agent):
    """
    Implement a Reflexion-like algorithm for a number of loops.
    This is assumed to be un-supervised, rather than a reinforcement-based approach where the answer is known apriori.

    The process is roughly: actor(start)

    :param start (str): A prompt or starting state to feed to the actor agent
    :param actor (Agent): A language agent which produces the output desired
    :param evaluator (Agent): A language agent that reviews the output of the actor agent each round and provides feedback for improvement
    :param model_name (str): An Azure OpenAI deployment name to use for the two agents
    :param n_rounds (int): The number of rounds that the actor and evaluator loop should be run
    """
    max_steps : int
    actor_proto_class : PersistentAgent
    eval_proto_class : PersistentAgent
    actor : PersistentAgent = None
    evaluator : PersistentAgent = None
    agent_answers : list[str] = []
    eval_answers : list[str] = []
    logger : logging.Logger

    def __init__(self, start: str, actor: PersistentAgent, evaluator: PersistentAgent, model_name: str, n_rounds: int = 3) -> None:
        super().__init__(question=start, model_name=model_name)
        self.max_steps = n_rounds
        self.actor_proto_class = actor
        self.eval_proto_class = evaluator
        self.logger = logger.getChild(self.__class__.__name__)

    def run(self, reset: bool = False) -> None:
        if not self.truncated or self.terminated:
            # Run action on start so each step/round starts with a reflection and ends with an action
            self.logger.info("=== Starting Place ========")
            self.scratchpad += "=== Starting Place ==========\n"
            self.actor = self.actor_proto_class(question=self.format_prompt(), model_name=self.model_name, llm = self.llm)
            self.answer = self.actor()
            self.agent_answers.append(self.answer)
            self.scratchpad += self.actor.scratchpad
            self.scratchpad += "\n=====================\n"

        return super().run(reset)
    def step(self):
        """
        Run 1 round of reflection + action
        """
        self.logger.info(f"=== Round {self.curr_step} ===")
        self.scratchpad += f"=== Round {self.curr_step} ===\n"
        # === Reflect =================
        self.logger.info("Reflecting.")
        self.scratchpad += f"=== Reflecting =========\n"
        if self.evaluator is None:
            # Init if we haven't yet
            self.evaluator = self.eval_proto_class(question=self.answer, model_name=self.model_name, llm = self.llm)
        else:
            self.evaluator.add_observation(self.answer)
        
        self.evaluator.step()
        reflection = self.evaluator.answer

        self.scratchpad += reflection
        self.scratchpad += "\n=====================\n"
        self.eval_answers.append(reflection)

        # === Act =====================
        self.logger.info("Acting.")
        self.scratchpad += f"=== Acting =========\n"
        self.actor.add_observation(reflection)
        self.actor.step()
        self.answer = self.actor.answer
        self.scratchpad += self.answer
        self.scratchpad += "\n=====================\n"
        self.agent_answers.append(self.answer)
        
        self.truncated = self.actor.truncated or self.evaluator.truncated
        self.terminated = self.curr_step == self.max_steps
        self.curr_step += 1

    def format_prompt(self) -> None:
        """
        Not Needed.
        """
        return None
    
    def reset(self) -> None:
        self.agent_answers = []
        self.eval_answers = []
        super().reset()
