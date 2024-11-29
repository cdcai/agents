import subprocess
import sys
import tempfile
from typing import Literal


def _subprocess_tool_call_on_file(tool_input: str, cmd_args: list[str], output_type: Literal["stdout", "file"] = "stdout") -> str:
    """
    A helper function that writes `tool_input` to a file and runs a python module on that file, either returning stdout+stderr or the contents of the file after the subprocess call.

    Ex. As a tool call for an Agent to use mypy / black on device and return output

    :param tool_input (str): A string to pass as input to the tool (this is likely code)
    :param cmd_args (list[str]): Command-line args between the python -m call and the file name (should include the python module to call and any additional arguments)
    :param output_type (str): The output to return (either stdout+error, or contents of the tempfile, if this is modified)
    
    :return: Either stdout and stderr concatenated into a string and separated by a newline, or `tool_input` after calling the python module
    """
    with tempfile.TemporaryFile("w", delete=False) as file:
        file.write(tool_input)
        file.close()

        # Run mypy in a subprocess and capture stderr and stdout
        subprocess_output = subprocess.run(
            [sys.executable, "-m", *cmd_args, file.name],
            capture_output=True,
            text=True
        )

        if output_type == "stdout":
            return "\n".join([subprocess_output.stdout, subprocess_output.stderr])
        elif output_type == "file":
            with open(file.name, "r", encoding="utf-8") as f:
                out = f.read()
            
            return(out)
        else:
            # Shouldn't be reachable
            raise NotImplementedError()
