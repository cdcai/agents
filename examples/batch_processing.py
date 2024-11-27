"""
Demo of batch processing

An extension of the prior structured prediction example where we produce 20 Q this time
and send to agents to solve in batches of 5

Sean Browning
"""

import asyncio
import agents
from pydantic import BaseModel, Field
from typing import List, Literal
import logging
import dotenv
import openai

N_QUESTIONS = 20
BATCH_SIZE = 5

# NOTE: This loads in env vars for openAI
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class Answer(BaseModel):
    """
    Response we expect from question answering agent
    """
    answer: List[Literal["A", "B", "C", "D"]] = Field(description="Answer to the question")

class QuestionandAnswer(Answer):
    """
    The response body we expect question producing agent (Q + A)
    """
    question: List[str] = Field(description="Question text that includes the question and response options, but NOT the answer")

class QAGenerator(agents.StructuredOutputAgent):
    """
    A language agent which produces questions to be answered
    """

    SYSTEM_PROMPT = """
    You are a language agent proficient in crafting multiple choice questions.
    You are competing against another lanaguage agent, so you only produce difficult questions that you think would stump another AI model.
    """

    BASE_PROMPT = """
    Write {n_questions} short multiple choice questions on whatever topic you choose and supply their correct answers.

    For each question, there should be 4 possible answer choices: A, B, C, and D.

    You will supply the question text and the answer as parameters to a function call.
    The function call accepts an array for each argument, so pass all questions and answers in the same function call and ensure their indices align.
    
    For this task, perform two sequential actions:
    1. Think through step-by-step how you would craft a question to stump an AI model of the same level as yourself
    2. Send your questions and answers via function call

    Send each action as a separate response
    """

class QAnswerer(agents.StructuredOutputAgent):
    SYSTEM_PROMPT = "You are a language agent proficient in answering multiple choice questions."
    # NOTE: {batch} is important here, since it's passed implicitly by the processor
    BASE_PROMPT = """
    Answer each multiple choice question using a function call.
    The function call accepts an array for each argument, so pass all answers in the same function call and ensure their indices align.

    {batch}
    """

if __name__ == "__main__":
    # Run this with Interactive OAuth
    agents.openai_creds_ad("Interactive")

    llm = openai.AsyncAzureOpenAI()

    ag = QAGenerator(
        response_model=QuestionandAnswer,
        model_name = "edav-api-share-gpt4-api-nofilter",
        llm = llm,
        oai_kwargs={"temperature": 1.0},
        n_questions=N_QUESTIONS
    )

    # Generate our questions
    asyncio.run(ag())

    print(ag.answer)

    proc = agents.BatchProcessor(
        data=ag.answer["question"],
        agent_class=QAnswerer,
        batch_size=BATCH_SIZE,
        n_workers=3,
        response_model=Answer,
        model_name = "edav-api-share-gpt4-api-nofilter",
        oai_kwargs={"temperature": 0.0}
    )

    predicted_answers_raw = asyncio.run(proc.process())

    predicted_answers = []

    for batch in predicted_answers_raw:
        predicted_answers.extend(batch["answer"])

    print(predicted_answers)
    
    n_correct = sum(expected == predicted for expected, predicted in zip(ag.answer["answer"], predicted_answers))
    print(f"{n_correct} / {len(ag.answer['answer'])} Correct")