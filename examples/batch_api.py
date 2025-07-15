"""
Testing the OpenAI Batch API in a few different scenarios.
"""

import asyncio
import logging
from typing import List

import pydantic
from dotenv import load_dotenv

import agents

load_dotenv()

logging.basicConfig(filename="batch_api.log", filemode="w", level=logging.INFO)

# A knock-knock joke return model
class KnockKnock(pydantic.BaseModel):
    setup: str = pydantic.Field(description="The setup for the knock-knock joke")
    punchline: str = pydantic.Field(
        description="The punchline for the knock-knock joke"
    )

    def __str__(self) -> str:
        return f"- Knock-Knock.\n- Who's There?\n- {self.setup}\n- {self.setup} who?\n- {self.punchline}"

    def __repr__(self) -> str:
        return f"KnockKnock(setup={self.setup!r}, punchline={self.punchline!r})"

# Define an agent
class KnockKnockAgent(agents.StructuredOutputAgent):
    "An agent that writes good Knock Knock jokes."

    BASE_PROMPT = """
    You are a top-comedian that specializes in writing knock-knock jokes.

    You're in a competition to write the best knock-knock joke, and you have to really think of a unique joke to win.

    You'll have two tasks:
    1. Plan for how you'd write a knock-knock joke that would win in a competition using the plan() tool
    2. Write the joke using the KnockKnock model, which should consist of a setup (the part that follows "who's there?") and a punchline (the part that follows "who?").
    """

    # Making temp higher
    oai_kwargs = {"temperature": 0.9, "parallel_tool_calls": False}

    def __init__(
        self,
        model_name: str | None = None,
        stopping_condition=None,
        provider=None,
        tools=None,
        callbacks=None,
        oai_kwargs=None,
        **fmt_kwargs,
    ):

        super().__init__(
            response_model=KnockKnock,
            model_name=model_name,
            stopping_condition=stopping_condition,
            provider=provider,
            tools=tools,
            callback=callbacks,
            oai_kwargs=oai_kwargs,
            **fmt_kwargs,
        )

    @agents.agent_callable(
        "Plan for how you'd write a knock-knock joke that would win in a competition.",
        {"text": "Your plan for the joke."},
    )
    def plan(self, text: str) -> str:
        return "Good thinking. Now send your joke."

class KnockKnockJudge(agents.PredictionAgent):
    """
    The Knock-knock joke contest Judge
    """

    BASE_PROMPT = """
    You are the judge of a knock-knock joke competition.
    You know a good joke when you hear it, and it's time to crown a winner.

    Read the following knock-knock jokes and crown a winner using a tool call with the joke number:

    {jokes}
    """.strip()

    def __init__(self, jokes: List[str], provider=None, **fmt_kwargs):
        labels = [str(i) for i, _ in enumerate(jokes)]

        if fmt_kwargs is None:
            fmt_kwargs = {}

        fmt_kwargs.update(
            {
                "jokes": "\n".join(
                    f"Joke {i}:\n{joke}" for i, joke in enumerate(jokes)
                )
            }
        )

        super().__init__(labels=labels, provider=provider, **fmt_kwargs)

async def agents_example():
    async with agents.AzureOpenAIBatchProvider(
        "gpt-4o-batch", batch_size=5, n_workers=2
    ) as provider:
        # Kind of a hacky way to use this, but just for demonstration purposes
        proc = agents.BatchProcessorIterable(
            [i for i in range(10)], KnockKnockAgent, batch_size=1, provider=provider
        )

        jokes = await proc.process()

    print("Got the following entries:")
    print(
        "\n".join(
            f"Joke {i}:\n{str(KnockKnock.model_validate(res))}"
            for i, res in enumerate(jokes)
        )
    )

    # Judge the jokes
    judge = KnockKnockJudge(
        [str(KnockKnock.model_validate(joke)) for joke in jokes],
        # Chat provider since it's a single call
        provider=agents.AzureOpenAIProvider(
            model_name="gpt-4o-nofilter", interactive=False
        ),
    )

    await judge()
    best_joke_idx = judge.answer["labels"][0]

    print(
        "The judge crowned a winner!\n\n{}".format(
            str(KnockKnock.model_validate(jokes[int(best_joke_idx)]))
        )
    )


if __name__ == "__main__":
    asyncio.run(agents_example())
