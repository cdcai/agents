"""
Using models locally served via huggingface via an OpenAI interface
"""

import asyncio
import logging
from multiprocessing import Process
from typing import Iterable, List, Dict

import httpx

from .openai import OpenAIProvider

try:
    import openai
except ImportError as e:
    raise ImportError(
        f"OpenAI package must be installed to use an transformers provider!\n{str(e)}"
    )

from openai.types.chat.chat_completion import Choice
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

try:
    import transformers
except ImportError as e:
    raise ImportError(
        f"The transformers package must be installed to use transformers provider!\n{str(e)}"
    )

from transformers.commands.serving import ServeArguments, ServeCommand

logger = logging.getLogger(__name__)


class TransformersProvider(OpenAIProvider):
    """
    A provider that spins up a local LLM using HuggingFace transformers serving in a new process
    and uses the OpenAI interface for completions
    """

    serving_process: Process

    def __init__(
        self, model_name: str, host: str = "localhost", port: int = 8000, **kwargs
    ):
        """
        A provider that spins up a local LLM using HuggingFace transformers serving in a new process
        and uses the OpenAI interface for completions. Note that you'll need pytorch and ample GPU resources
        for this provider, as everything runs locally.

        :param str model_name: The name of a huggingface model instance
        :param str host: The name of the host for the transformers serving instance
        :param int port: The port number of the host to serve the API
        :param kwargs: Additional named arguments passed to transformers serving (see `transformers serve --help`)
        """

        xformers_kwargs = kwargs
        xformers_kwargs.update({"host": host, "port": port})
        self.xformers_kwargs = ServeArguments(**xformers_kwargs)
        self.base_url = f"http://{host}:{port}/v1"
        super().__init__(model_name, base_url=self.base_url, api_key="n/a")
        self.endpoint_fn = self.round_trip_increment(self.chat_stream_to_response)

    async def chat_stream_to_response(
        self, messages: Iterable[ChatCompletionMessageParam], **kwargs
    ):
        stream_res = await self.llm.chat.completions.create(
            messages=messages, stream=True, **kwargs
        )

        final_response_content = ""
        final_tool_calls: Dict[int, Dict] = {}

        async for chunk in stream_res:
            if chunk.choices[0].delta.content is not None:
                final_response_content += chunk.choices[0].delta.content

            for tool_call in chunk.choices[0].delta.tool_calls or []:
                index = tool_call.index

                if index not in final_tool_calls:
                    final_tool_calls[index] = tool_call.model_dump()

                if final_tool_calls[index]["function"] is not None:
                    if final_tool_calls[index]["function"]["arguments"] is None:
                        final_tool_calls[index]["function"]["arguments"] = ""
                    if tool_call.function is not None:
                        final_tool_calls[index]["function"]["arguments"] += (
                            tool_call.function.arguments
                            if tool_call.function.arguments is not None
                            else ""
                        )

        final_choice = chunk.choices[0].model_dump()

        final_choice["message"] = {
            "content": final_response_content,
            "role": "assistant",
            "tool_calls": [v for _, v in final_tool_calls.items()],
        }

        out = ChatCompletion(
            id=chunk.id,
            model=chunk.model,
            created=chunk.created,
            object="chat.completion",
            choices=[Choice.model_validate(final_choice)],
        )

        return out

    def _serving_heatbeat(self):
        """
        Simple loop to check that our server has started
        """
        logger.info("[TransformersProvider] Waiting for server process to start.")
        while True:
            with httpx.Client() as client:
                try:
                    res = client.get(self.base_url)
                except httpx.ConnectError:
                    continue
                # The status code actually doesn't matter
                # so much as it does that we make a connection
                # since there isn't a / GET endpoint anyways
                break
        logger.info("[TransformersProvider] Process started.")

    @staticmethod
    def _serving(args: ServeArguments):
        """
        Serve command that runs in separate process
        """
        serve_process = ServeCommand(args)
        serve_process.run()

    def __aenter__(self):
        """
        Start the serving process
        """
        # Start server
        self.serving_process = Process(
            target=self._serving,
            name="Transformers-Serving",
            args=[self.xformers_kwargs],
        )
        self.serving_process.start()

        # Check server has started
        self._serving_heatbeat()

        return super().__aenter__()

    def __aexit__(self, exc_type, exc_value, traceback):
        """
        End serving process
        """
        logger.info("[TransformersProvider] Closing server.")
        self.serving_process.terminate()
        self.serving_process.join()
        return super().__aexit__(exc_type, exc_value, traceback)
