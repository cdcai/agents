import argparse
import os
import logging

import openai
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv

import util

load_dotenv()

# === Service Principal auth ========================
credential = ClientSecretCredential(
    tenant_id=os.environ["SP_TENANT_ID"],
    client_id=os.environ["SP_CLIENT_ID"],
    client_secret=os.environ["SP_CLIENT_SECRET"],
)

os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token(
    "https://cognitiveservices.azure.com/.default"
).token

# openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["GPT4_URL"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", required=True, type=str)
    parser.add_argument("--truth", "-t", required=True, type=str)
    parser.add_argument("--examples", "-e", required=True, type=str)
    parser.add_argument("--n_trials", "-n", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument(
        "--model", "-m", type=str, default="edav-api-share-gpt4-api-nofilter"
    )
    parser.add_argument(
        "--agent", "-a", choices=["react", "reflexion"], type=str, default="react"
    )
    parser.add_argument(
        "--reflection_strategy",
        "-s",
        choices=["last_attempt", "reflexion", "last_attempt_reflexion"],
        default="last_attempt_reflexion",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--log_path",
        default="out.log",
        type=str
    )
    args = parser.parse_args()

    if args.debug:
        # Add debug logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handle = logging.FileHandler(args.log_path)
        handle.setLevel(logging.DEBUG)
        logger.addHandler(handle)

    # Initialize environment / gymnasium
    envir = util.WikiQAEnv(
        question=args.question, truth=args.truth, max_steps=args.n_steps
    )

    # Initialize LLM
    llm = openai.AzureOpenAI()

    # Initialize Agent
    if args.agent == "react":
        agent = util.ReactAgent(
            question=args.question,
            examples=args.examples,
            model_name=args.model,
            llm=llm,
            env=envir,
        )
    elif args.agent == "reflexion":
        agent = util.ReactandReflectAgent(
            question=args.question,
            examples=args.examples,
            reflection_strategy=args.reflection_strategy,
            model_name=args.model,
            llm=llm,
            env=envir,
        )
    else:
        raise NotImplementedError(f"Unknown agent class, {args.agent}")

    trial_results = []

    for i in range(args.n_trials):
        if args.agent != "reflexion":
            # Only Reflexion stores
            # memory between trials
            agent.reset()
        agent.run()

        if args.debug:
            # Dump full scratchpad
            logger.debug("Trial complete. Full Scratchpad:\n" + agent.scratchpad)
        
        trial_results.append(agent.correct)
    
    print(f"Successful trials: {sum(trial_results)} / {len(trial_results)}")


if __name__ == "__main__":
    main()
