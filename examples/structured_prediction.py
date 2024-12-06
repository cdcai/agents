"""
Demo of structured returns and predictions using two agents:
- One QAGenerator agent which returns some questions to be answered along with their correct answers
- One QAnswerer agent, which attempts to answer the questions

Sean Browning
"""
import asyncio
import agents
from pydantic import BaseModel, Field
from typing import List, Literal
import logging
import dotenv
import openai

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

    SYSTEM_PROMPT = "You are a language agent proficient in crafting multiple choice questions"
    BASE_PROMPT = """
    Write {n_questions} short multiple choice questions on a general topic and supply their correct answers.

    For each question, there should be 4 possible answer choices: A, B, C, and D.

    You will supply the question text and the answer as parameters to a function call.
    The function call accepts an array for each argument, so pass all questions and answers in the same function call and ensure their indices align.
    """

class QAnswerer(agents.StructuredOutputAgent):
    SYSTEM_PROMPT = "You are a language agent proficient in answering multiple choice questions."
    BASE_PROMPT = """
    Answer each multiple choice question using a function call.
    The function call accepts an array for each argument, so pass all answers in the same function call and ensure their indices align.

    {questions}
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
        n_questions=5
    )

    asyncio.run(ag())

    print(ag.answer)

    ag2 = QAnswerer(
        response_model=Answer,
        model_name = "edav-api-share-gpt4-api-nofilter",
        llm = llm,
        oai_kwargs={"temperature": 1.0},
        questions=ag.answer["question"]
    )

    asyncio.run(ag2())

    print(ag2.answer)