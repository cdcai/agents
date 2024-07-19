import logging
import re

from openai import OpenAI
from .base_agents import ChunkedAgent, ToolAwareAgent

logger = logging.getLogger(__name__)

class CodeOutlinerAgent(ChunkedAgent):
    """
    A language agent which, given input code, provides an outline of inputs, outputs, and transformations in plain language
    for general understanding and downstream translation tasks
    """

    SYSTEM_PROMPT: str = "You are a coding expert capable of reading code in any langauge and providing coherent outlines of the underlying purpose and logic"
    BASE_PROMPT: str = """
    Please outline the following code using bullets and plain language.
    In your response, please provide a short description of what the program accomplishes, a list of inputs and outputs, and a step-by-step outline of any algorithms or transformations contained in the code.
    Your response will be used downstream by other agents producing code in a different language, so please provide clear instructions that could be followed to produce the same script.

    {question}
    """

    def step(self):
        # Prompt LLM for first-pass
        llm_prompt_input = self.get_next_messages()
        first_answer = self.prompt_agent(llm_prompt_input, n_tok = 2 * self.chunk_max).message.content
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"

        # Attempt to parse python scripts in response
        ret_scripts = self.extract_pyscripts(first_answer)
        self.scratchpad += f"\n=== Answer {self.curr_step} =====\n"
        self.scratchpad += "\n".join(ret_scripts)
        self.scratchpad += "\n===================================\n"

        self.answer_cache.extend(ret_scripts)

        # End run
        self.terminated = len(self.question) == 0
        self.curr_step += 1

class SASConvertAgent(ChunkedAgent):
    """
    The language agent responsible for producing Python files from input SAS scripts.
    """
    SYSTEM_PROMPT: str = "You are a coding expert in the statistical language, SAS, and Python and able to assist with converting between the two."
    BASE_PROMPT: str = """
    I am converting an existing SAS script into Python. The file is quite long, so you may only see part of it at a time.
    The SAS script performs several ETL steps on data files which contain EHR data.
    Please help me re-write this script using syntactically correct Python and any data science libraries (ex. numpy, pandas, polars) that might be needed.
    If more than one script is needed, please place the file contents in separate ```python blocks.
    Please provide ONLY the output code.

    Here is my SAS file:
    ```sas
    {question}
    ```
    """
    # Regex to extract python script from OpenAI response
    # (allowing multiple cases because GPT-4 isn't consistent)
    PYSCRIPT_CONTENT = re.compile(r"```[pP][ython]*\n(.+?)```", re.DOTALL)

    def step(self):
        # Prompt LLM for first-pass
        llm_prompt_input = self.get_next_messages()
        first_answer = self.prompt_agent(llm_prompt_input, n_tok = 2 * self.chunk_max).message.content
        self.scratchpad += f"=== Input {self.curr_step} ==========\n"
        self.scratchpad += "\n".join(msg["content"] for msg in llm_prompt_input)
        self.scratchpad += "\n===================================\n"

        # Attempt to parse python scripts in response
        ret_scripts = self.extract_pyscripts(first_answer)
        self.scratchpad += f"\n=== Answer {self.curr_step} =====\n"
        self.scratchpad += "\n".join(ret_scripts)
        self.scratchpad += "\n===================================\n"

        self.answer_cache.extend(ret_scripts)

        # End run
        self.terminated = len(self.question) == 0
        self.curr_step += 1

    def extract_pyscripts(self, answer: str) -> list[str]:
        py_scripts = [script.group(1) for script in self.PYSCRIPT_CONTENT.finditer(answer)]
        logger.info(f"Extracted {len(py_scripts)} from response.")
        
        return(py_scripts)

class PythonRefineAgent(ToolAwareAgent):
    SYSTEM_PROMPT: str = "You are a Python coding expert and can identify and correct syntactical mistakes and fix incomplete code using standard conventions."
    BASE_PROMPT: str = """
    I have converted part of an existing SAS script into Python. Due to length, the script may have been translated in chunks and the final results concatenated into a single script.
    This script may contain syntax errors, be poorly commented, have undefined global references, or duplicative/un-used imports, etc.
    Please read this script and provide any corrections and re-organization as needed. Please also provide type hints, function docstrings, and guiding comments as needed.
    You may call the mypy and black tools to assist, and you may call both in parallel. 
    If no changes are needed, provide the script back using the call_submit tool. Always check the file first before submitting your final answer using call_submit.
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

    @staticmethod
    def call_black(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        try:
            import black
        except ImportError as e:
            logger.error("black is mising, please install it with `pip install black`")
            raise e

        return ToolAwareAgent._subprocess_tool_call_on_file(code, ["black"], output_type="file")

    @staticmethod
    def call_mypy(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        try:
            import mypy
        except ImportError as e:
            logger.error("mypy is mising, please install it with `pip install mypy`")
            raise e

        return ToolAwareAgent._subprocess_tool_call_on_file(code, ["mypy", "--install-types", "--non-interactive"])
