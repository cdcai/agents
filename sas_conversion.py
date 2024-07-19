"""
Testing Conversion of SAS files to Python using system of language agents
"""
import logging
from argparse import ArgumentParser
from dotenv import load_dotenv

from util import SASConvertAgent, PythonRefineAgent

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
    parser.add_argument("-i", "--input", type=str, default="data/INPUT - GEN LAB 1 initial LOINC pull and data clean (5).sas", help="SAS file input path")
    parser.add_argument("-o", "--output", type=str, default="out.py", help="Python file output path (default: out.py)")
    parser.add_argument("-m", "--model", type=str, default="edav-chatapp-share-gpt4-32k-tpm25kplus-v0613-dfilter", help="Name of model deployment to use for language agents (default: edav-chatapp-share-gpt4-32k-tpm25kplus-v0613-dfilter)")
    parser.add_argument("--device_code", action="store_true", help="Instead of using Service Principal creds, utilize Azure Device Code authorization flow for Azure OpenAI")
    args = parser.parse_args()

    with open(args.input, "r") as file:
        sas_file_content = file.read().strip()
    
    if args.device_code:
        # TODO: Implement
        pass

    sas_agent = SASConvertAgent(sas_file_content, model_name=args.model, chunk_max=3000)

    sas_agent.run()

    with open(args.output, "w") as file:
        file.write(sas_agent.answer)

    with open("sas_agent_scratchpad.txt", "w") as file:
        file.write(sas_agent.scratchpad)

    refine_agent = PythonRefineAgent(sas_agent.answer, model_name=args.model)

    refine_agent.run()

    with open(args.output.replace(".py", "_refined.py"), "w") as file:
        file.write(refine_agent.answer)

    with open("python_refine_scratchpad.txt", "w") as file:
        file.write(refine_agent.scratchpad)