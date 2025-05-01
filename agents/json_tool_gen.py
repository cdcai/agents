"""
Automated generation of function calling JSON payload for OpenAI
using python type hints and a thin decorator

Sean Browning
"""

import functools
import inspect
from types import NoneType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

PYTHON_TO_OAI_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    List: "array",
    Dict: "object",
    NoneType: "null",
}


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
        item_type = arg_to_oai_type(args[0]) if args else {"type": "any"}
        return {"type": "array", "items": item_type}
    elif origin is dict or origin is Dict:
        return {"type": "object"}
    elif origin is Literal:
        return {"type": "string", "enum": list(args)}
    elif origin is Union:
        # If >1 option, we have to run multiple times and aggregate results
        union_types = [arg_to_oai_type(py_type) for py_type in args]
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


def generate_tool_json_payload(
    func: Callable, description: str, variable_description: Dict[str, str]
) -> Dict[str, Any]:
    """
    Internal function used to generate OpenAI Function Calling JSON payload from function type hints.

    Args:
        func (Callable): The function to extract type hints from
        description (str): A Description of the function
        variable_description (Dict[str, str]): A dict with entries for each variable of the function describing what each variable is
    Returns:
        Dict[str, Any] A JSON payload to provide in the request body for OpenAI Function Calling
    """
    hints = get_type_hints(func)
    hints.pop("return", None)  # Ignore return type
    sig = inspect.signature(func)
    missing_annotations = (set(sig.parameters.keys()) ^ {"self"}) - set(hints.keys())
    missing_descriptions = (set(sig.parameters.keys()) ^ {"self"}) - set(
        variable_description.keys()
    )

    if len(missing_annotations) > 0:
        raise ValueError(
            "agent_callable requires type hints for every argument! Missing type hints for {}: {}.".format(
                func.__name__, ", ".join(missing_annotations)
            )
        )
    if len(missing_descriptions):
        raise ValueError(
            "agent_callable requires descriptions for every argument! Missing description for {}: {}.".format(
                func.__name__, ", ".join(missing_descriptions)
            )
        )

    tool_json = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
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
            arg_properties = arg_to_oai_type(hint)
        except KeyError as e:
            raise KeyError("Processing arg {} failed. {}".format(arg, str(e)))
        arg_properties["description"] = variable_description[arg]
        tool_json["function"]["parameters"]["properties"].update({arg: arg_properties})

    return tool_json


def agent_callable(description: str, variable_description: dict[str, str]):
    """
    Marks a method as accessible to a language agent
    and generates required JSON payload by extracting type hints.

    Args:
        description (str): A description of the function which will be shared with the language agent
        variable_description (dict[str, str]): A dict with entries for each variable of the function describing what each variable is
    """

    def agent_callable_wrapper(func: Callable):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        # Generate the JSON payload needed for OpenAI Function Calling API
        # and assign it to an attribute we can extract at Agent init time
        json_payload = generate_tool_json_payload(
            func, description, variable_description
        )
        setattr(wrapper, "agent_tool_payload", json_payload)

        return wrapper

    return agent_callable_wrapper
