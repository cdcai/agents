import asyncio
import logging
import os
import json
from typing import List, Union, Tuple, Dict, Optional
from io import StringIO, BytesIO

import backoff
import openai
from azure.identity import ClientSecretCredential, InteractiveBrowserCredential
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from openai.types.file_object import FileObject
from openai.types.batch import Batch
from ..abstract import _Agent, _Provider, _BatchAPIHelper
from ..tools import OpenAIToolCall

DEFAULT_BATCH_SIZE = 1000

logger = logging.getLogger(__name__)

class OpenAIBatchAPIHelper(_BatchAPIHelper):

    provider : "AzureOpenAIBatchProvider"

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def register_provider(self, provider: "AzureOpenAIBatchProvider"):
        """
        Store provider as an attribute and start up batching task
        """
        self.provider = provider
        self.task = asyncio.create_task(self._batcher(), name="OpenAIBatchHelper")

    async def _batcher(self):
        """
        Batch loop

        This is a co-routine that puts all of our messages together into a batch, sends them off,
        and stores the results back into a dict for the individual agents to handle.

        It's started at init time if we select batch mode, and persists for the duration of the session.
        """
        while True:
            # Define our batch and start the clock
            batch = []
            batch_poll_start = asyncio.get_running_loop().time()

            # Await new messages to load into the batch
            # - Until we hit our max batch size, or
            # - Until we've waited for the time indicated (default 2s)
            while len(batch) <= self.batch_size:
                time_remaining = self.timeout - (
                    asyncio.get_running_loop().time() - batch_poll_start
                )
                if time_remaining <= 0.0:
                    # We've waited long enough, send what we have in a batch
                    break
                try:
                    req = await asyncio.wait_for(self.provider.batch_q.get(), timeout=0.1)
                    batch.append(req)
                    self.provider.batch_q.task_done()
                except asyncio.TimeoutError:
                    continue

            # Case: Nothing to submit
            # - Just re-run until we do
            if len(batch) == 0:
                continue

            # Otherwise: create our batch payload, send off to OpenAI, and wait for the results
            batch_file = await self.provider.send_batch(batch)
            logger.debug(f"Batch file {batch_file.id} sent to OpenAI")

            batch_task = await self.provider.create_batch_task(batch_file)

            # Get results
            results = await self.provider.get_batch_results(batch_task)
            
            # Write out results to dict for agents to pick up
            for result in results:
                self.provider.batch_out[result["custom_id"]] = ChatCompletion.model_validate(result["response"]["body"])

class AzureOpenAIProvider(_Provider):
    """
    An Azure OpenAI Provider for language Agents.

    This provider generally assumes you already have all required environment variables
    set correctly, or will provide them as kwargs which will be passed to AsyncAzureOpenAI at init

    Namely:
    - api_version or OPENAI_API_VERSION
    - azure_endpoint or AZURE_OPENAI_ENDPOINT

    AZURE_OPENAI_API_KEY will be assigned via authentication (either by ClientSecret or Interactive AD Auth depending on `interactive`)

    :param str model_name: Model name from the deployments list to use
    :param bool interactive: Should authentication use an Interactive AD Login (T), or ClientSecret (F)?
    :param **kwargs: Any additional kw-args for AsyncAzureOpenAI
    """

    tool_call_wrapper = OpenAIToolCall
    llm: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI]
    mode : str = "chat"

    def __init__(
        self,
        model_name: str,
        interactive: bool,
        **kwargs,
    ):
        self.model_name = model_name
        self.interactive = interactive
        self.authenticate()
        self.llm = openai.AsyncAzureOpenAI(**kwargs)
        # Monkey-patching depending on selected mode
        if self.mode == "chat":
            self.endpoint_fn = self.llm.chat.completions.create

    def authenticate(self) -> None:
        """
        Retrieve Azure OpenAI API key via Interactive AD Login or ClientSecret authentication
        (Interactive is not suggested for long-running tasks, since key expires every hour)

        :returns: API key assigned to `AZURE_OPENAI_API_KEY` and `OPENAI_API_KEY` environ variables
        """
        credential: Union[InteractiveBrowserCredential, ClientSecretCredential]

        if self.interactive:
            credential = InteractiveBrowserCredential()
        else:
            credential = ClientSecretCredential(
                tenant_id=os.environ["SP_TENANT_ID"],
                client_id=os.environ["SP_CLIENT_ID"],
                client_secret=os.environ["SP_CLIENT_SECRET"],
            )

        os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token
        os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

        if getattr(self, "llm", None) is not None:
            self.llm.api_key = os.environ["AZURE_OPENAI_API_KEY"]

    @backoff.on_exception(
        backoff.expo, (openai.APIError, openai.AuthenticationError), max_tries=3
    )
    async def prompt_agent(
        self,
        ag: _Agent,
        prompt: Union[List[ChatCompletionMessageParam], ChatCompletionMessageParam],
        **kwargs,
    ):
        """
        An async version of the main OAI prompting logic.

        :param ag: The calling agent class
        :param prompt: Either a dict or a list of dicts representing the message(s) to send to OAI model
        :param kwargs: Key word arguments passed to completions.create() call (tool calls, etc.)

        :return: An openAI Choice response object
        """

        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if not isinstance(prompt, list):
            prompt = [prompt]

        ag.scratchpad += "--- Input ---------------------------\n"
        for msg in prompt:
            if "content" in msg and isinstance(msg["content"], str):
                ag.scratchpad += "\n" + msg["content"]
        ag.scratchpad += "\n-----------------------------------\n"

        try:
            res = await self.endpoint_fn(
                messages=prompt, model=self.model_name, **kwargs
            )
        except openai.AuthenticationError as e:
            logger.info("Auth failed, attempting to re-authenticate before retrying")
            self.authenticate()
            raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out = res.choices[0]

        ag.scratchpad += "--- Output --------------------------\n"
        ag.scratchpad += "Message:\n"
        ag.scratchpad += out.message.content if out.message.content else "<None>" + "\n"

        if len(ag.TOOLS):
            # attempt to parse tool call arguments
            # BUG: OpenAI sometimes doesn't return a "tool_calls" reason and uses "stop" instead. Annoying.
            if out.finish_reason == "tool_calls" or (
                out.finish_reason == "stop"
                and out.message.tool_calls
                and len(out.message.tool_calls)
            ):
                out.finish_reason = "tool_calls"
                # Append GPT response to next payload
                # NOTE: This has to come before the next step of parsing
                ag.tool_res_payload.append(out.message)

        ag.scratchpad += "\n-----------------------------------\n"
        logger.info(f"Received response: {out.message.content}")

        if out.finish_reason == "length":
            ag.truncated = True
            ag.scratchpad += (
                "Response returned truncated from OpenAI due to token length.\n"
            )
            logger.warning("Message returned truncated.")
        return out

class AzureOpenAIBatchProvider(AzureOpenAIProvider):
    """
    Azure OpenAI using the Batch API
    """
    
    mode = "batch"

    def __init__(self, model_name: str, *, interactive: bool = False, batch_size: int = DEFAULT_BATCH_SIZE, batch_handler: Optional[OpenAIBatchAPIHelper] = None, **kwargs):
        """
        Using AzureOpenAI with the Batch API mode. Each batch typically takes several minutes or longer to be evaluated, but many requests
        can be sent at once, and the price/request is generally around half of the standard chat endpoint.

        :param str model_name: The name of an Azure OpenAI deployment
        :param bool interactive: Should the requests be run in interactive mode using EntraID credentials
        :param int batch_size: The maximum size of batches that should be sent to OpenAI at a time
        :param OpenAIBatchAPIHelper batch_handler: (optional) An initialized batch handler which will be used to handle the inqueue of requests to send to openAI
        :param kwargs: Any keyword arguments to pass to OpenAI class
        
        """
        self.batch_size = batch_size
        self.batch_idx = 1
        self.endpoint_fn = self.query_batch_mode
        self.batch_q: asyncio.Queue[Dict] = asyncio.Queue()
        self.batch_out: Dict[str, ChatCompletion] = {}

        if batch_handler is None:
            self.batch_handler = OpenAIBatchAPIHelper(batch_size=batch_size)
        self.batch_handler.register_provider(self)

        super().__init__(model_name, interactive, **kwargs)

    async def query_batch_mode(
        self, messages: List[ChatCompletionMessageParam], model: str, **kwargs
    ) -> ChatCompletion:
        task_id = f"task-{self.batch_idx}"
        self.batch_idx += 1

        task = {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, **kwargs, "messages": messages},
        }

        await self.batch_q.put(task)

        task_done = False
        while not task_done:
            try:
                out = self.batch_out.pop(task_id)
                task_done = True
                if isinstance(out, Exception):
                    # Propagate errors if we encountered any
                    raise out
            except KeyError:
                # DEBUG: Remove this when batch size is >>
                logger.debug(f"Still waiting for batch response [{task_id}]")
                # Poll every 5s for the response
                await asyncio.sleep(5)
                continue

        return out

    @staticmethod
    def _create_batch_file(tasks: List[dict]) -> Tuple[str, bytes, str]:
        """
        Create a batch file for the OpenAI Batch API

        :param tasks: List of task dictionaries to be sent to OpenAI
        :return: Tuple containing the file name, file content, and MIME type to send as an API payload
        """

        with StringIO() as batch_file:
            for task in tasks:
                batch_file.write(json.dumps(task) + "\n")

            return (
                "batch_tasks.jsonl",
                batch_file.getvalue().encode("utf-8"),
                "application/json",
            )

    async def send_batch(
        self,
        tasks: List[dict],
        **kwargs,
    ) -> FileObject:
        """
        Send a batch file to OpenAI pending further processing.

        :param tasks: List of task dictionaries to be sent to OpenAI
        :param kwargs: Additional keyword arguments for the file upload (see OpenAI API documentation)

        :return: An OpenAI File object representing the uploaded batch file
        """
        file_name, file_content, mime_type = self._create_batch_file(tasks)
        return await self.llm.files.create(
            file=(file_name, file_content, mime_type), purpose="batch", **kwargs
        )

    async def create_batch_task(
        self, batch_file: FileObject, timeout: int = 30, **kwargs
    ) -> Batch:
        """
        Create a batch from an existing batch file object.

        :param FileObject batch_file: An OpenAI File object representing the batch file
        :param int timeout: polling timeout waiting for response
        :param kwargs: Additional keyword arguments for the batch creation (see OpenAI API documentation)

        :return: An OpenAI File object representing the created batch
        """
        try:
            logger.info(f"Executing batch task [{batch_file.id}]")
            batch = await self.llm.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                **kwargs,
            )
        except:
            logger.error(f"Attempt to process batch {batch_file.id} failed!")
            raise

        while batch.status not in ["completed", "failed"]:
            logger.info(f"Batch status [{batch.id}]: {batch.status}. Waiting for completion...")
            batch = await self.llm.batches.retrieve(batch.id)
            await asyncio.sleep(timeout)

        return batch

    async def get_batch_results(self, batch: Batch) -> List[dict]:
        """
        Retrieve the results of a completed batch.

        :param batch: An OpenAI Batch object representing the completed batch

        :return: A list of results from the batch
        """
        if batch.status != "completed" or batch.output_file_id is None:
            raise ValueError("Batch status was not 'completed'! Got: " + batch.status)

        results = []
        result_stream = await self.llm.files.content(batch.output_file_id)

        with BytesIO() as buffer:
            buffer.write(result_stream.content)
            result_text = buffer.getvalue().decode("utf-8")
            for line in result_text.splitlines():
                results.append(json.loads(line))

        return results

class OpenAIProvider(AzureOpenAIProvider):
    """
    Standard (non-Azure) OpenAI provider

    Requires `api_key` passed as a kwarg, or OPENAI_API_KEY set as an environment variable
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.authenticate()
        self.llm = openai.AsyncOpenAI(**kwargs)

    def authenticate(self):
        pass
