"""
Testing Conversion of SAS files to Python using system of language agents
"""
import logging
import os
import re
import sys
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import PureWindowsPath
from typing import Union

import tiktoken
from black import FileMode, format_str
from dotenv import load_dotenv
from openai import OpenAI

from util import Agent, ToolAwareAgent

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
    I am converting an existing SAS script into Python. The file is quite long, so you may only send part of it at a time.
    The SAS script performs several ETL steps on data files which contain EHR data.
    Please help me re-write this script using syntactically correct Python and any data science libraries (ex. numpy, pandas, polars) that might be needed.
    If more than one script is needed, please place the file contents in separate ```python blocks.
    Please provide ONLY the output code.

    Here is my SAS file:
    ```sas
    {question}
    ```
    """
    py_scripts : list[str] = []
    answer : list[str] = []

    def __init__(self, question: str, model_name: str, llm: OpenAI | None = None, chunk_max : int = 2500):
        # Get tokenizer to handle chunking responses if needed
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Also save full input to full_question attribute since we'll
        # overwrite self.question if the resulting payload is too large
        self.full_question = question
        self.chunk_max = chunk_max

        super().__init__(question, model_name, llm)

    def run(self, reset: bool = False) -> None:
        super().run(reset)

        self.dump_pyscripts()

    def step(self):
        # Prompt LLM for first-pass
        llm_prompt_input = self.format_prompt()
        first_answer = self.prompt_agent(llm_prompt_input, n_tok = 2 * self.chunk_max)
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"

        # Attempt to parse python scripts in response
        self.extract_pyscripts(first_answer)
        self.scratchpad += f"\n=== First Answer {self.curr_step} =====\n"
        self.scratchpad += self.py_scripts[-1]
        self.scratchpad += "\n===================================\n"

        # Send output script to another agent for refinement
        refine_agent = PythonRefineAgent(self.py_scripts[-1], self.model_name, self.llm)
        refine_agent.run()
        self.answer.append(refine_agent.answer)
        self.scratchpad += f"\n=== Refined Answer {self.curr_step} =====\n"
        self.scratchpad += self.answer[-1]
        self.scratchpad += "\n===================================\n"

        # End run
        self.terminated = len(self.question) == 0
        self.curr_step += 1

    @staticmethod
    def clean_response(res: str) -> str:
        """
        Over-rides base clean_response method that strips new lines
        (which we want for code blocks)
        """
        return res

    def extract_pyscripts(self, answer: str):

        for script in PYSCRIPT_CONTENT.finditer(answer):
            self.py_scripts.append(script.group(1))

    def dump_pyscripts(self, out_path: Union[str, os.PathLike] = "out.py") -> None:
        """
        Combine returned python scripts into a single file and write out to disk
        :param out_path: Path to the output python file to write out
        """

        full_pyscript = ""
        for i, script in enumerate(self.py_scripts, 1):
            full_pyscript += f"# === Chunk {i} ==== \n"
            full_pyscript += script
            full_pyscript += "\n ================== \n\n"

        with open(out_path, "w") as file:
            file.write(full_pyscript)


    def format_prompt(self) -> list[dict[str, str]]:
        """
        Insert SAS code into prompt, and return list of messages
        to send to chatGPT.
        """
        prompt_len = len(self.tokenizer.encode(self.BASE_PROMPT.format(question="") + self.SYSTEM_PROMPT))

        # Split by code block
        # (at least one empty line between next code block)
        # This will theoretically require fewer calls to reach a response of appropriate length
        script_inq = re.split("\n{2,}?", self.question)

        excess_lines = []
        # pop line by line until we have a message payload less than the requested max
        while len(self.tokenizer.encode("\n\n".join(script_inq))) + prompt_len > self.chunk_max:
            # Store excess message payload in question object
            excess_lines.append(script_inq.pop())
        
        # Reverse things around and re-join to string
        # to get things the right way around
        self.question = "\n\n".join(reversed(excess_lines))

        # Construct query
        fmt_prompt = re.sub("\s+", " ", self.BASE_PROMPT).format(question="\n\n".join(script_inq))
        
        out = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": fmt_prompt}
        ]

        return out

    def reset(self) -> None:
        self.answer = []
        self.py_scripts = []
        super().reset()

class PythonRefineAgent(ToolAwareAgent):
    SYSTEM_PROMPT: str = "You are a coding expert in Python and can identify and correct syntactial mistakes"
    BASE_PROMPT: str = """
    I have converted part of an existing SAS script into Python. This script may contain small syntax errors, be poorly commented, or have undefined global references.
    Please read this script and provide any corrections that may be needed. Please provide type hints, function docstrings, and guiding comments as needed.
    You may call the mypy and black tools to check the code. If no changes are needed, provide the script back using the submit tool. Always check the file first before submitting.
    Please provide ONLY the output code marked in a code block, no additional commentary is needed.

    Here is the python file:
    ```python
    {question}
    ```
    """

    TOOLS = [
        # Mypy
        {
            "type": "function",
            "function": {
                "name": "call_mypy",
                "description": "Run the MyPy static typing tool on input code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to check using the mypy static typing tool"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        # Black
        {
            "type": "function",
            "function": {
                "name": "call_black",
                "description": "Run the black python code formater on input python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to check using black tool"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

    def __init__(self, question: str, model_name: str, llm: OpenAI | None = None):
        # Add additional tools from base class(call_submit)
        self.TOOLS.extend(super().TOOLS)

        super().__init__(question, model_name, llm)
    
    @staticmethod
    def call_black(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        black_res = format_str(code, mode = FileMode())

        return black_res

    @staticmethod
    def call_mypy(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        with tempfile.TemporaryFile("w", suffix=".py", delete=False) as file:
            file.write(code)
            file.close()

            # Run mypy in a subprocess and capture stderr and stdout
            out = subprocess.run(
                [sys.executable, "-m", "mypy", "--install-types", "--non-interactive", file.name],
                capture_output=True
            )

        return "\n".join([str(out.stdout), str(out.stderr)])


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

