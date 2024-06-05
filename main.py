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
    parser.add_argument("--examples", "-e", type=str)
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
        default=False
    )
    parser.add_argument(
        "--log_path",
        default="out.log",
        type=str
    )
    args = parser.parse_args()

    # Add hard-coded example option if none are passed
    if args.examples is None:
        examples = """Question: Which of Jonny Craig and Pete Doherty has been a member of more bands ?
Thought 1: I need to search Jonny Craig and Pete Doherty, find the number of bands they have been a member of, then find which one has been a member of more bands.
Action 1: Search[Jonny Craig]
Observation 1: Jonathan Monroe Craig (born March 26, 1986) is a Canadian-American singer and songwriter, best known as the former lead vocalist of bands Dance Gavin Dance, Emarosa, Rain City Drive (previously known as Slaves), as well as the co-lead vocalist of the short-lived supergroup Isles & Glaciers. As a solo artist, he has released two studio albums, two EPs and a live album to date, and his distinct type of soul-based singing has earned him considerable acclaim.
Thought 2: Jonny Craig has been a member of four bands. I need to search Pete Doherty next and find the number of bands he has been a member of.
Action 2: Search[Pete Doherty]
Observation 2: Peter Doherty (born 12 March 1979) is an English musician, songwriter, actor, poet, writer, and artist. He is best known for being co-frontman of The Libertines, which he formed with Carl Barât in 1997. His other musical projects are indie band Babyshambles and Peter Doherty and the Puta Madres.
Thought 3: Pete Doherty has been a member of three bands. So Jonny Craig has been a member of more bands.
Action 3: Finish[Jonny Craig]
Observation 3: Answer is CORRECT
Correct answer: Jonny" Craig

Question: The creator of "Wallace and Gromit" also created what animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes? 
Thought 1: I need to search "Wallace and Gromit" and find the animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes.
Action 1: Search["Wallace and Gromit"]
Observation 1: Could not find ["Wallace and Gromit"]. Similar: ['Wallace and Gromit', 'Wallace & Gromit: The Curse of the Were-Rabbit', 'A Close Shave', 'A Grand Day Out', 'The Wrong Trousers', 'A Matter of Loaf and Death', 'Nick Park', 'Aardman Animations', 'Timmy Time', "Wallace and Gromit's Cracking Contraptions"]
Thought 2: To find the animation comedy, I can search Nick Park.
Action 2: Search[Nick Park]
Observation 2: Nicholas Wulstan Park  (born 6 December 1958) is a  British animator who created Wallace and Gromit, Creature Comforts, Chicken Run, Shaun the Sheep, and Early Man. Park has been nominated for an Academy Award a total of six times and won four with Creature Comforts (1989), The Wrong Trousers (1993), A Close Shave (1995) and Wallace & Gromit: The Curse of the Were-Rabbit (2005).He has also received five BAFTA Awards, including the BAFTA for Best Short Animation for A Matter of Loaf and Death, which was also the most watched television programme in the United Kingdom in 2008. His 2000 film Chicken Run is the highest-grossing stop motion animated film.For his work in animation, in 2012, Park was among the British cultural icons selected by artist Peter Blake to appear in a new version of Blake's most famous artwork—the Beatles' Sgt. Pepper's Lonely Hearts Club Band album cover—to celebrate the British cultural figures of his life.
Thought 3: Nick Park created Wallace and Gromit, Creature Comforts, Chicken Run, Shaun the Sheep, and Early Man. The animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes is Creature Comforts.
Action 3: Finish[Creature Comforts]
Observation 3: Answer is CORRECT
Correct answer: Creature Comforts"""
    else:
        examples = args.examples
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
            examples=examples,
            model_name=args.model,
            llm=llm,
            env=envir,
        )
    elif args.agent == "reflexion":
        agent = util.ReactandReflectAgent(
            question=args.question,
            examples=examples,
            reflection_strategy=args.reflection_strategy,
            model_name=args.model,
            llm=llm,
            env=envir,
        )
    else:
        raise NotImplementedError(f"Unknown agent class, {args.agent}")

    trial_results = []

    for i in range(args.n_trials):
        # If we got it right, stop running
        if agent.correct:
            break
        # else, keep working
        if args.agent != "reflexion":
            # Only Reflexion stores
            # memory between trials
            agent.reset()
        agent.run()

        if args.debug:
            # Dump full scratchpad
            logger.debug("Trial complete. Full Scratchpad:\n" + agent.scratchpad)
        else:
            print("Trial complete. Full Scratchpad:\n" + agent.scratchpad)
        
        trial_results.append(agent.correct)
    
    print(f"Successful trials: {sum(trial_results)} / {len(trial_results)}")


if __name__ == "__main__":
    main()
