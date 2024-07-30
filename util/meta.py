"""
Meta-algorithms and arrangements for agent systems
"""

from .agents import Agent, ChunkedAgent, PersistentAgent
import logging
import openai
import re

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

    def __init__(self, start: str, actor: PersistentAgent, evaluator: PersistentAgent, model_name: str, n_rounds: int = 3, llm: openai.OpenAI | None = None) -> None:
        super().__init__(question=start, model_name=model_name, llm=llm)
        self.max_steps = n_rounds
        self.actor_proto_class = actor
        self.eval_proto_class = evaluator
        self.actor = self.actor_proto_class(question=self.format_prompt(), model_name=self.model_name, llm = self.llm)
        self.logger = logger.getChild(self.__class__.__name__)

    def run(self, reset: bool = False) -> None:
        if not self.truncated or self.terminated:
            # Run action on start so each step/round starts with a reflection and ends with an action
            self.logger.info("=== Starting Place ========")
            self.scratchpad += "=== Starting Place ==========\n"
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

    def format_prompt(self) -> str:
        """
        No formatting needed here, just return the original input.
        """
        return self.question
    
    def reset(self) -> None:
        self.agent_answers = []
        self.eval_answers = []
        super().reset()

class ChunkedReflexion(ChunkedAgent):
    max_steps : int
    actor_proto_class : PersistentAgent
    eval_proto_class : PersistentAgent
    logger : logging.Logger
    def __init__(self, start: str, actor: PersistentAgent, evaluator: PersistentAgent, model_name: str, n_rounds: int = 2, llm: openai.OpenAI | None = None, chunk_max: int = 3000):
        super().__init__(question=start, model_name=model_name, llm=llm, chunk_max=chunk_max)

        self.actor_proto_class = actor
        self.eval_proto_class = evaluator
        self.max_steps = n_rounds
        self.BASE_PROMPT = self.actor_proto_class.BASE_PROMPT
        self.SYSTEM_PROMPT = self.actor_proto_class.SYSTEM_PROMPT

        self.logger = logger.getChild(self.__class__.__name__)
    
    def step(self):
        """
        One step of the alg. Processes on one chunk at a time, running a Reflexion-style algorithm over each chunk.
        """
        self.scratchpad += f"=== Chunk {self.curr_step} ========\n"
        self.logger.info(f"=== Chunk {self.curr_step} =============")
        step_input = self.format_prompt()
        reflection_agent = Reflexion(step_input, self.actor_proto_class, self.eval_proto_class, self.model_name, n_rounds=self.max_steps, llm=self.llm) 
        
        # Add previous chunk's translation as context
        if (last_response := self.fetch_last_response()) is not None:
            reflection_agent.actor.conversation_cache.append({
                "role": "user",
                "content": "Here is the translation of the previous code chunk, for context:\n```python\n{}\n```".format(last_response["content"])
                })
        
        step_answer = reflection_agent()

        self.scratchpad += reflection_agent.scratchpad
        self.scratchpad += "\n=======================\n"
        self.answer_cache.append(step_answer)

        self.terminated = len(self.question) == 0
        self.curr_step += 1
            
    def format_prompt(self, split_expr: str = "\n{2,}?", join_str: str = "\n\n", **kwargs) -> str:
            """
            Formatting BASE_QUERY, checking for output length and chunking self.question if necessary

            :param split_expr (str): A string or regex to pass to re.split() to split self.question into chunks.
            :param join_str (str): A string to use to recompose chunks of self.question back together.
            
            NOTE: split_expr and join_str can be different (ex. '\\n{2, }?', and '\\n\\n'),
            but join_str should always produce output that could be split on subsequent calls using split_expr.
            """
            prompt_len = self.get_prompt_len()

            input_chunks = re.split(split_expr, self.question)
            excess = []

        # pop chunk by chunk until we have a message payload less than the requested max
            while len(self.tokenizer.encode(join_str.join(input_chunks))) + prompt_len > self.chunk_max:
                # Store excess message payload in question object
                excess.append(input_chunks.pop())
            
            # Reverse things around and re-join to string
            # to get things the right way around
            self.question = join_str.join(reversed(excess))

            return join_str.join(input_chunks)
        