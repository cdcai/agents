"""
Testing Conversion of SAS files to Python using system of language agents
"""
from argparse import ArgumentParser

import os
import re
from typing import Union
from openai import OpenAI
import tiktoken
import logging
from dotenv import load_dotenv

from util import Agent

load_dotenv()

PYSCRIPT_CONTENT = re.compile(r"```python(.+?)```", re.DOTALL)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SASConvertAgent(Agent):
    """
    The language agent responsible for producing Python files from input SAS scripts.
    """
    SYSTEM_PROMPT: str = "You are a coding expert in the statistical language, SAS, and Python and able to assist with converting between the two."
    BASE_PROMPT: str = """
    I am converting an existing SAS script into Python. The SAS script performs several ETL steps on data files which contain EHR data.
    Please re-write this script using syntactically correct Python and any data science libraries (ex. numpy, pandas, polars) that might be needed.
    If more than one script is needed, please place the file contents in separate ```py blocks.

    Here is my SAS file:
    ```sas
    {question}
    ```
    """

    py_scripts : list[str] = []
    answer : list[str] = []

    def __init__(self, question: str, model_name: str, llm: OpenAI | None = None):
        # Get tokenizer to handle chunking responses if needed
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Also save full input to full_question attribute since we'll
        # overwrite self.question if the resulting payload is too large
        self.full_question = question
    
        super().__init__(question, model_name, llm)
    def run(self, reset: bool = False) -> None:
        super().run(reset)

        self.dump_pyscripts()

    def step(self):
        # Prompt LLM
        self.answer.append(self.prompt_agent(self.format_prompt()))
        self.scratchpad += f"\n=== Answer {self.curr_step} =====\n"
        self.scratchpad += self.answer[-1]
        self.scratchpad += "\n===================================\n"

        # Attempt to parse python scripts in response
        self.extract_pyscripts()

        # End run
        self.terminated = len(self.question) > 0
        self.curr_step += 1

    def extract_pyscripts(self):

        for script in PYSCRIPT_CONTENT.finditer(self.answer[-1]):
            self.py_scripts.append(script)

    def dump_pyscripts(self, out_dir: Union[str, os.PathLike] = "data") -> None:
        out_path = Path(out_dir)

        for i, script in enumerate(self.py_scripts, 1):
            with open(out_path / f"py_script_{i}.py", "w") as file:
                file.write(script)


    def format_prompt(self) -> list[dict[str, str]]:
        """
        Insert SAS code into prompt, and return list of messages
        to send to chatGPT.
        """
        prompt_len = len(self.tokenizer.encode(self.BASE_PROMPT.format(question="") + self.SYSTEM_PROMPT))

        # Split by line
        script_inq = self.question.split("\n")

        excess_lines = []
        # pop line by line until we have a message payload less than GPT-4 Max
        while len(self.tokenizer.encode(script_inq)) + prompt_len > self.GPT4_TOKEN_MAX:
            # Store excess message payload in question object
            excess_lines.append(script_inq.pop())
        
        # We popped from the end, so we need to reverse and re-join to string
        # to get things the right way around
        self.question = "\n".join(reversed(excess_lines))

        # Construct query
        fmt_prompt = re.sub("\s+", " ", self.BASE_PROMPT).format(question="\n".join(script_inq))
        
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": fmt_prompt}
        ]

        return out

    def reset(self) -> None:
        self.answer = []
        self.py_scripts = []
        super().reset()

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
    sas_agent = SASConvertAgent(sas_file_content, model_name=args.model)

    sas_agent.run()

    print(sas_agent.py_scripts)

