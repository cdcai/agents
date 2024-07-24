import logging
import re

from openai import OpenAI

from .base import PersistentAgent, ReduceAgent, ChunkedAgent, ToolAwareAgent

logger = logging.getLogger(__name__)

class OutlineSummarizeAgent(ReduceAgent):
    """
    A simple agent that summarizes all the outlines provided by CodeOutlinerAgent
    """
    SYSTEM_PROMPT : str = "You are an expert at creating highly technical and thorough summaries and outlines of computer programs"
    BASE_PROMPT : str = """
    The following messages are all technical outlines another AI assistant produced pertaining to sections of the same computer program.
    Please read and combine them to produce a detailed summary of the full program described in the same style.
    You should retain all pertinent data so this outline can be used as a guide to write a program from scratch without seeing the underlying script.
    """

class SAStoPythonAgent(PersistentAgent):
    """
    A very simple, no frills, SAS -> Python code conversion agent.
    """
    SYSTEM_PROMPT : str = "You are an expert at statistical and computer programming and can translate input SAS code to Python. You return only code blocks as output."
    BASE_PROMPT : str = "Convert the following SAS code into Python code. You may use any library imports necessary to complete the task.\n```sas\n{question}\n```"
    APPEND_PROMPT : str = "Refine your answer using the following reflections\n{obs}"
    # Regex to extract python script from OpenAI response
    # (allowing multiple cases because GPT-4 isn't consistent)
    PYSCRIPT_CONTENT = re.compile(r"```[pP][ython]*\n(.+?)```", re.DOTALL)

    def step(self):
        """
        Full Agent logic. Prompts LLM and saves answer
        """
        super().step()
        # Extract just the Python code from the response
        self.answer = "\n".join(self.extract_pyscripts(self.answer))

    def format_prompt(self) -> str:
        return self.BASE_PROMPT.format(question=self.question).strip()

    def extract_pyscripts(self, answer: str) -> list[str]:
        py_scripts = [script.group(1) for script in self.PYSCRIPT_CONTENT.finditer(answer)]
        logger.info(f"Extracted {len(py_scripts)} python chunk from response.")
        
        return(py_scripts)

class PythonReflexionAgent(PersistentAgent):
    SYSTEM_PROMPT : str = "You are an advanced reasoning agent that can improve Python code based on self refection."
    BASE_PROMPT : str = "A languge agent was given a task to translate an ETL/cleaning pipeline from SAS code into Python code. Review the code below and devise a high-level plan to improve the output script using best Python data science coding practices. Use complete sentences.\n{question}"
    APPEND_PROMPT : str = "The script was modified according to your previous reflections. Review this new code and devise a new plan\n```python\n{obs}\n```"

    def format_prompt(self) -> str:
        return self.BASE_PROMPT.format(question=self.question).strip()

class PseudocodeSummarizeAgent(ReduceAgent):
    """
    An agent which produces whole program pseudocode from several plain-language program outlines
    """
    SYSTEM_PROMPT : str = "You are an expert in computer science and can produce legible pseudo-code from plain-text descriptions of computer programs"
    BASE_PROMPT : str = """
    The following messages are all technical outlines another AI assistant produced pertaining to sections of the same computer program.
    Please use these descriptions to re-create the full program being described using pseudo-code. You may generalize functions as necessary, but be sure that all funcationality makes it to the final product.
    Your output should contain only psuedocode, no plaintext.
    
    ex.

    Input: [
        "The the program defines two integer variables, x = 2 and y = 3",
        "The program defines a function, sum, that takes two integer parameters, a and b, and adds them together and returns an integer sum"
        "The program runs the sum() function on x and y and assigns the value to variable z"
        ]

    Output: ```
    let x : integer = 2;
    let y : integer = 3;
    def sum( x : integer, y : integer) -> integer { return (x + y) } 

    z = sum(x,y)
    ```
"""

class PythonSummarizeAgent(ReduceAgent):
    """A simple agent that summarizes the several python scripts already generated"""
    SYSTEM_PROMPT : str = "You are an expert python programmer that can reorganize and piece together a functional script from several fragments"
    BASE_PROMPT : str = "A previous AI Agent translated a SAS script into Python in several chunks. Please combine all these chunks into a single valid python program. Please ensure that all the pieces are incorperated in the final script. Please return only the script as output in a ```python block"
    
    def get_next_messages(self) -> list[dict[str, str]]:
        out = super().get_next_messages()
        out.append({
            "role": "assistant",
            "content": "\n".join(self.question)
        })
        
        return out
class CodeOutlinerAgent(ChunkedAgent):
    """
    A language agent which, given input code, provides an outline of inputs, outputs, and transformations in plain language
    for general understanding and downstream translation tasks
    """

    SYSTEM_PROMPT: str = "You are a coding expert capable of reading code in any langauge and providing coherent outlines of the underlying purpose and logic"
    BASE_PROMPT: str = """
    Please outline the following code using bullets and plain language. You may only see part of the script at a time, so use any provided context as a guide to previously viewed code to incorperate into your response.
    In your response, please provide a short description of what the program accomplishes, a list of inputs and outputs, and a step-by-step outline of any algorithms or transformations contained in the code.
    Your response will be used downstream by other agents producing code in a different language, so please provide clear instructions that could be followed to produce the same script.

    ```sas
    {question}
    ```
    """
    answer : list[str] = []

    def combine_answer_cache(self) -> None:
        """
        Overload this method to return all answers as a list
        instead of appended together
        """
        self.answer = self.answer_cache

class SASConvertAgent(ChunkedAgent):
    """
    The language agent responsible for producing Python files from input SAS scripts.
    """
    SYSTEM_PROMPT: str = "You are a coding expert in the statistical language, SAS, and Python and able to assist with converting between the two."
    BASE_PROMPT: str = """
    I am converting an existing SAS script into Python. The file is quite long, so you may only see part of it at a time.
    Please help me re-write this script using syntactically correct Python and any data science libraries (ex. numpy, pandas, polars) that might be needed.
    If more than one script is needed, please place the file contents in separate ```python blocks.
    Please provide ONLY the output code.

    Here's an outline of the script with some context that might be helpful:
    {context}

    Here is my SAS file:
    ```sas
    {{question}}
    ```
    """
    # Regex to extract python script from OpenAI response
    # (allowing multiple cases because GPT-4 isn't consistent)
    PYSCRIPT_CONTENT = re.compile(r"```[pP][ython]*\n(.+?)```", re.DOTALL)
    
    answer : list[str] = []

    def __init__(self, question: str, context: str, model_name: str, llm: OpenAI | None = None, chunk_max: int = 3000):
        self.context = context
        self.BASE_PROMPT = self.BASE_PROMPT.format(context=self.context)
        super().__init__(question, model_name, llm, chunk_max)

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

    def combine_answer_cache(self) -> None:
        """
        Overload this method to return all answers as a list
        instead of appended together
        """
        self.answer = self.answer_cache

    def extract_pyscripts(self, answer: str) -> list[str]:
        py_scripts = [script.group(1) for script in self.PYSCRIPT_CONTENT.finditer(answer)]
        logger.info(f"Extracted {len(py_scripts)} python chunk from response.")
        
        return(py_scripts)

class PythonRefineAgent(ToolAwareAgent):
    """
    A Language agent that attempts to refine input Python code using coding knowledge and local tool calls to mypy and black
    """
    SYSTEM_PROMPT: str = "You are a Python coding expert and can identify and correct syntactical mistakes and fix incomplete code using standard conventions."
    BASE_PROMPT: str = """
    I have converted part of an existing SAS script into Python. Due to length, the script may have been translated in chunks and the final results concatenated into a single script.
    Please read this script and provide any corrections and re-organization as needed. Please also provide type hints, function docstrings, and guiding comments as needed.
    You may call the mypy, black, and ruff tools to assist, and you may call both in parallel. 
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
        },
        # Ruff
        {
            "type": "function",
            "function": {
                "name": "call_ruff",
                "description": "Run the ruff python code linter on input python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to check using ruff tool"
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
    def call_ruff(code: str) -> str:
        """
        Tool usable by language agent to return formatting help from Black
        :param code: Python code to check against black
        """
        try:
            import ruff
        except ImportError as e:
            logger.error("ruff is mising, please install it with `pip install ruff`")
            raise e

        return ToolAwareAgent._subprocess_tool_call_on_file(code, ["ruff", "check"], output_type="stdout")

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
