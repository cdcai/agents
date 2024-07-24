"""
Testing Conversion of SAS files to Python using system of language agents
"""
import logging
from argparse import ArgumentParser
from dotenv import load_dotenv

from util.agents import *
from util.meta import Reflexion
load_dotenv()

logger = logging.basicConfig(filename="sas_conversion.log", filemode="w", level=logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="SAS -> Python Conversion via LLM agents",
        description="""
        This program converts an input SAS file into a rough outline of a Python script that might replace the functionality.
        It's unlikely that the resulting script will work out-of-the-box, but it should provide a basic structure to start with
        """
    )
    parser.add_argument("-i", "--input", type=str, default="data/just_bili.sas", help="SAS file input path")
    parser.add_argument("-o", "--output", type=str, default="out.py", help="Python file output path (default: out.py)")
    parser.add_argument("-m", "--model", type=str, default="edav-chatapp-share-gpt4-32k-tpm25kplus-v0613-dfilter", help="Name of model deployment to use for language agents (default: edav-chatapp-share-gpt4-32k-tpm25kplus-v0613-dfilter)")
    parser.add_argument("--device_code", action="store_true", help="Instead of using Service Principal creds, utilize Azure Device Code authorization flow for Azure OpenAI")
    parser.add_argument("--reflection_rounds", type=int, default=2, help="Number of rounds of Reflexion to refine the output")
    args = parser.parse_args()

    with open(args.input, "r") as file:
        sas_file_content = file.read().strip()
    
    if args.device_code:
        # TODO: Implement
        pass

    reflection_pipeline = Reflexion(
        sas_file_content,
        actor=SAStoPythonAgent,
        evaluator=PythonReflexionAgent,
        model_name=args.model,
        n_rounds=args.reflection_rounds
    )

    refine_agent = PythonRefineAgent(reflection_pipeline(), args.model)

    with open(args.output, "w") as file:
        file.write(refine_agent())
