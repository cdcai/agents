"""
Demo of simple Python code production Agent with an Evaluator Agent as a callback
Sean Browning
"""
import asyncio
import agents
import logging

logger = logging.basicConfig()

class DummyAgent(agents.Agent):
    """
    A code producing agent with a simple prompt
    """
    SYSTEM_PROMPT = "You are a language agent proficient in producing expressive and syntactically correct Python code."
    BASE_PROMPT = "Write a simple program to fit an ordinary least squares regression model to a polars DataFrame input. Include a worked example in your response and provide only the code in code fences (ex. ```python)"

class DummyEvaluatorAgent(agents.Agent):
    """
    An evaluator agent that will be called with the output of the above agent and give feedback on the response and the prompt
    """
    SYSTEM_PROMPT = "You are a skilled project manager and evaluator of Python programs. You provide expert evaluation of whether code meets stated goals and reflect on how to improve the approach."
    BASE_PROMPT = """
    The following Python program was produced by an AI language agent:

    {answer}

    Here is the full history of the requested program and the response from the AI languge agent in response:

    {scratchpad}

    Reflect on the performance of the AI language model and provide feedback on how the initial prompt and resulting python code could be improved.
    """

if __name__ == "__main__":
    ag = DummyAgent(
        model_name = "edav-api-share-gpt4-api-nofilter",
        stopping_condition=agents.StopOnStep(1),
        callbacks=[
            agents.AgentCallback(
                DummyEvaluatorAgent,
                model_name="edav-api-share-gpt4-api-nofilter",
                stopping_condition=agents.StopOnStep(1)
            )
        ],
        oai_kwargs={"temperature": 1.0}
    )

    asyncio.run(ag())

    print(f"Answer:\n{ag.answer}\n\nFeedback:\n{ag.callback_output[0]}")