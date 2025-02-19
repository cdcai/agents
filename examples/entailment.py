"""
Demo of evaluating entailment of two statements and returning the log probability from the model
Sean Browning
"""
import asyncio
import agents
import logging
import dotenv

import agents.ssl_tools

# NOTE: This loads in env vars for openAI
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    prov = agents.AzureOpenAIProvider(
        "gpt-4o-mini-nofilter",
        interactive=True
    )

    agent = agents.EntailmentAgent(
        stopping_condition=agents.StopOnStep(1),
        provider=prov,
        context = "",
        statement_a = "The king of france is bald.",
        statement_b = "France has a king."
    )

    with agents.ssl_tools.no_ssl_verification():
        asyncio.run(agent())

    print(agent.answer)