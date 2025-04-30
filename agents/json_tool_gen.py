"""
Automated generation of function calling JSON payload for OpenAI
using python type hints and a thin decorator

Sean Browning
"""
from functools import update_wrapper
import inspect
from typing import (
    List,
    Union,
    Callable,
    Dict,
    Any,
    Literal,
    get_type_hints,
    get_origin,
    get_args,
)
from types import NoneType

PYTHON_TO_OAI_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    List: "array",
    Dict: "object",
    NoneType: "null",
}

class AgentCallable:
    def __init__(self, func: Callable, description: str, variable_description: Dict[str, str]):
        self.description = description
        self.variable_description = variable_description
        self.func = func
        self.agent_json_payload = self.generate_tool_json_payload()

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        return result

    def generate_tool_json_payload(self) -> Dict[str, Any]:
        """
        Internal function used to generate OpenAI Function Calling JSON payload from function type hints.
        """
        hints = get_type_hints(self.func)
        hints.pop("return", None) # Ignore return type
        sig = inspect.signature(self.func)
        missing_annotations = (set(sig.parameters.keys()) ^ {"self"}) - set(hints.keys())
        missing_descriptions = (set(sig.parameters.keys()) ^ {"self"}) - set(
            self.variable_description.keys()
        )

        if len(missing_annotations) > 0:
            raise ValueError(
                "agent_callable requires type hints for every argument! Missing type hints for {}: {}.".format(
                    self.func.__name__, ", ".join(missing_annotations)
                )
            )
        if len(missing_descriptions):
            raise ValueError(
                "agent_callable requires descriptions for every argument! Missing description for {}: {}.".format(
                    self.func.__name__, ", ".join(missing_descriptions)
                )
            )

        tool_json = {
            "type": "function",
            "function": {
                "name": self.func.__name__,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [variable for variable in hints.keys()],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        for arg, hint in hints.items():
            try:
                arg_properties = self.arg_to_oai_type(hint)
            except KeyError as e:
                raise KeyError("Processing arg {} failed. {}".format(arg, str(e)))
            arg_properties["description"] = self.variable_description[arg]
            tool_json["function"]["parameters"]["properties"].update({arg: arg_properties})

        return tool_json
    
    @staticmethod
    def arg_to_oai_type(arg: Any) -> Dict[str, Any]:
        """
        Converting Python type hint to OpenAI type for JSON payload.

        Args:
            arg (Any): A type hint for a specific argument of a function
        Returns:
            Dict[str, str]: Detailing the argument type and any possible choices (if a Literal or List)
        Raises:
            KeyError if type is not an interpretable type for OpenAI
        """
        origin = get_origin(arg)
        args = get_args(arg)

        if origin is list or origin is List:
            item_type = AgentCallable.arg_to_oai_type(args[0]) if args else {"type": "any"}
            return {"type": "array", "items": item_type}
        elif origin is dict or origin is Dict:
            return {"type": "object"}
        elif origin is Literal:
            return {"type": "string", "enum": list(args)}
        elif origin is Union:
            # If >1 option, we have to run multiple times and aggregate results
            union_types = [AgentCallable.arg_to_oai_type(py_type) for py_type in args]
            out: Dict[str, Union[List[str], str]] = {}

            for union_type in union_types:
                for key, value in union_type.items():
                    if key in out:
                        if isinstance(out[key], list):
                            out[key].append(value)
                        else:
                            out[key] = [out[key], value]
                    else:
                        out[key] = value

            return out
        elif arg in PYTHON_TO_OAI_SCHEMA:
            return {"type": PYTHON_TO_OAI_SCHEMA[arg]}
        else:
            raise KeyError(
                "Type {} is not an interpretable type for OpenAI.".format(str(arg))
            )

class AsyncAgentCallable(AgentCallable):
    async def __call__(self, *args, **kwargs):
        result = await self.func(*args, **kwargs)
        return result

def agent_callable(description: str, variable_description: dict[str, str]):
    """
    Marks a method as accessible to a language agent
    and generates required JSON payload by extracting type hints.

    Args:
        description (str): A description of the function which will be shared with the language agent
        variable_description (dict[str, str]): A dict with entries for each variable of the function describing what each variable is
    """
    def agent_callable_wrapper(func: Callable):
        # Handle async
        if inspect.iscoroutinefunction(func) is False:
            AC = AgentCallable
        else:
            AC = AsyncAgentCallable

        wrapper = update_wrapper(AC(func, description, variable_description), func)
        return wrapper

    return agent_callable_wrapper
