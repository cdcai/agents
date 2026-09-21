"""
Automated generation of function calling JSON payload for OpenAI
using python type hints and a thin decorator

TODO: make generic and split out eventually

Sean Browning
"""

import functools
import inspect
from asyncio import to_thread
from dataclasses import dataclass, field

try:
    from types import NoneType as TypeNone
except ImportError:
    # Fix: py3.9
    TypeNone = type(None)  # type: ignore

from collections.abc import Callable
from typing import (
    Any,
    Literal,
    Protocol,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

PYTHON_TO_OAI_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    TypeNone: "null",
}

__all__ = ["Tool", "agent_callable", "async_agent_callable"]

ToolParameterType = Literal[
    "string", "integer", "number", "boolean", "array", "null", "object", "any"
]


@dataclass
class Tool[AgentT]:
    """An executable tool, its model-facing definition, and availability policy."""

    call: Callable[..., Any]
    json_payload: "ToolDefinition | None" = None
    description: str | None = None
    variable_description: dict[str, str] | None = None
    condition: Callable[[AgentT], bool] | None = None
    name: str = field(init=False)

    def __post_init__(self):

        if self.json_payload is None:
            self.json_payload = getattr(self.call, "agent_tool_payload", None)

        if self.condition is None:
            self.condition = getattr(self.call, "agent_tool_condition", None)

        if self.json_payload is None:
            if self.description is None:
                raise TypeError(
                    "`description` cannot be None if `json_payload` is None "
                    "and `call` doesn't have a JSON payload"
                )
            elif self.variable_description is None:
                raise TypeError(
                    "`variable_description` cannot be None if `json_payload` "
                    "is None and `call` doesn't have a JSON payload"
                )
            else:
                # Generate tool payload from call, description, and variable description
                self.json_payload = generate_tool_json_payload(
                    self.call,
                    self.description,
                    self.variable_description,
                )

        # Just making it easier on myself
        self.name = self.json_payload["function"]["name"]

    def is_available(self, agent: AgentT) -> bool:
        """
        Return whether the tool is available to ``agent`` for the next step.

        Agents evaluate this once per step and use that same snapshot for both
        the model-facing definitions and subsequent call authorization.
        """
        return self.condition(agent) if self.condition else True

    @property
    def definition(self) -> "ToolDefinition":
        """Return the provider-facing tool definition."""
        # ``__post_init__`` always populates this value.
        assert self.json_payload is not None
        return self.json_payload

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the tool without blocking the event loop for sync callables."""
        if inspect.iscoroutinefunction(self.call):
            return await self.call(*args, **kwargs)

        result = await to_thread(self.call, *args, **kwargs)
        # Support callable objects and wrappers which return an awaitable but
        # are not themselves recognized as coroutine functions.
        if inspect.isawaitable(result):
            return await result
        return result

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.invoke(*args, **kwargs)


class ToolParameterProperties(TypedDict, total=False):
    type: ToolParameterType | list[ToolParameterType]
    description: str
    enum: list[Any]
    items: "ToolParameterProperties"


class ToolParameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, ToolParameterProperties]
    required: list[str]
    additionalProperties: bool


class ToolFunction(TypedDict):
    name: str
    description: str | None
    parameters: ToolParameters
    strict: bool


class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: ToolFunction


@runtime_checkable
class _AgentToolPayloadCarrier(Protocol):
    agent_tool_payload: ToolDefinition
    agent_tool_condition: Callable[[Any], bool] | None


def arg_to_oai_type(arg: Any) -> ToolParameterProperties:
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

    if origin is list or origin is list:
        item_type = arg_to_oai_type(args[0]) if args else {"type": "any"}
        return {"type": "array", "items": item_type}  # type: ignore
    elif origin is dict or origin is dict:
        return {"type": "object"}
    elif origin is Literal:
        return {"type": "string", "enum": list(args)}
    elif origin is Union:
        # If >1 option, we have to run multiple times and aggregate results
        union_types = [arg_to_oai_type(py_type) for py_type in args]
        out: dict[str, list[str] | str] = {}

        for union_type in union_types:
            for key, value in union_type.items():
                if key in out:
                    if isinstance(out[key], list):
                        out[key].append(value)  # type: ignore
                    elif isinstance(out[key], str):
                        out[key] = [out[key], value]  # type: ignore
                else:
                    out[key] = value  # type: ignore

        return out  # type: ignore
    elif arg in PYTHON_TO_OAI_SCHEMA:
        return {"type": PYTHON_TO_OAI_SCHEMA[arg]}  # type: ignore
    else:
        raise KeyError(f"Type {arg!s} is not an interpretable type for OpenAI.")


def generate_tool_json_payload(
    func: Callable, description: str, variable_description: dict[str, str]
) -> ToolDefinition:
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
    parameter_names = set(sig.parameters)
    parameter_names.discard("self")
    missing_annotations = parameter_names - set(hints)
    missing_descriptions = parameter_names - set(variable_description)

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

    tool_json: ToolDefinition = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [variable for variable in hints],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    for arg, hint in hints.items():
        try:
            arg_properties = arg_to_oai_type(hint)
        except KeyError as e:
            raise KeyError(f"Processing arg {arg} failed. {e!s}")
        arg_properties["description"] = variable_description[arg]
        tool_json["function"]["parameters"]["properties"].update({arg: arg_properties})

    return tool_json


def agent_callable(description: str, variable_description: dict[str, str], condition: Callable[[Any], bool] | None = None):
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
        cast(_AgentToolPayloadCarrier, wrapper).agent_tool_payload = json_payload
        cast(_AgentToolPayloadCarrier, wrapper).agent_tool_condition = condition
        return wrapper

    return agent_callable_wrapper


def async_agent_callable(description: str, variable_description: dict[str, str], condition: Callable[[Any], bool] | None = None):
    """
    Marks a coroutine as accessible to a language agent
    and generates required JSON payload by extracting type hints.

    Args:
        description (str): A description of the function which will be shared with the language agent
        variable_description (dict[str, str]): A dict with entries for each variable of the function describing what each variable is
    """

    def agent_callable_wrapper(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            return result

        # Generate the JSON payload needed for OpenAI Function Calling API
        # and assign it to an attribute we can extract at Agent init time
        json_payload = generate_tool_json_payload(
            func, description, variable_description
        )
        cast(_AgentToolPayloadCarrier, wrapper).agent_tool_payload = json_payload
        cast(_AgentToolPayloadCarrier, wrapper).agent_tool_condition = condition
        return wrapper

    return agent_callable_wrapper
