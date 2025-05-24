import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Literal, Union

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

from .abstract import _ToolCall

logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall(_ToolCall):
    """
    An encapsulating class for tool calls from an OpenAI lanaguge agent
    """

    tool_call: ChatCompletionMessageToolCall

    @property
    def id(self) -> str:
        return self.tool_call.id

    @property
    def func_name(self) -> str:
        return self.tool_call.function.name

    @property
    def arg_str(self) -> str:
        return self.tool_call.function.arguments

    @staticmethod
    def _construct_return_message(
        id: str, response: Union[str, BaseModel]
    ) -> Dict[str, Union[str, BaseModel]]:
        return {"tool_call_id": id, "role": "tool", "content": response}
