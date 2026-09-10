import asyncio
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from io import BytesIO, StringIO
from typing import (
    Literal,
    TypedDict,
    cast,
)

import backoff

try:
    import openai
except ImportError as e:
    raise ImportError(
        f"OpenAI package must be installed to use an OpenAI provider!\n{e!s}"
    )

try:
    from azure.identity.aio import ClientSecretCredential, get_bearer_token_provider
except ImportError as e:
    raise ImportError(f"azure.identity is required for OpenAI providers!\n{e!s}")

from openai.types import (
    Batch,
    CompletionCreateParams,
    CompletionUsage,
    EmbeddingCreateParams,
    FileObject,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ValidationError

from ..abstract import _Agent, _BatchAPIHelper, _Provider, _ToolCall
from ..batch_progress import (
    DEFAULT_BATCH_PROGRESS_MAX_ITEMS,
    BatchProgressRenderer,
    BatchProgressState,
    BatchSnapshot,
    BatchTracker,
    TqdmBatchProgressRenderer,
    validate_max_items,
)
from ..batch_progress import (
    BatchRequestCounts as ProgressRequestCounts,
)
from ..observability import LLMUsage, Observable

DEFAULT_BATCH_SIZE = 1000
logger = logging.getLogger(__name__)

OPENAI_BATCH_ACTIVE_STATUSES: frozenset[str] = frozenset(
    {"validating", "in_progress", "finalizing", "cancelling"}
)
OPENAI_BATCH_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)
OPENAI_BATCH_SUCCESS_STATUSES: frozenset[str] = frozenset({"completed"})
BATCH_TRACKING_CANCELLED_STATUS = "tracking_cancelled"
BATCH_TRACKING_FAILED_STATUS = "tracking_failed"

BatchStatusCallback = Callable[[Batch], None]


def _noop_batch_status_callback(_batch: Batch) -> None:
    pass


# HACK: OpenAI does not (yet) implement batch request input type
# See: https://github.com/openai/openai-python/issues/1937
class BatchRequestInput(TypedDict):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]
    body: EmbeddingCreateParams | CompletionCreateParams


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
        id: str, response: str | BaseModel
    ) -> dict[str, str | BaseModel]:
        return {"tool_call_id": id, "role": "tool", "content": response}


class OpenAIBatchAPIHelper(_BatchAPIHelper["AzureOpenAIBatchProvider"]):
    # The time in seconds to wait before checking if a batch has been completed
    api_timeout: int = 30

    def __init__(
        self,
        batch_size: int,
        n_workers: int = 1,
        *,
        progress_max_items: int = DEFAULT_BATCH_PROGRESS_MAX_ITEMS,
        progress_renderer: BatchProgressRenderer | None = None,
    ):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.progress_max_items = validate_max_items(progress_max_items)
        self.batch_tasks = set()
        self.task = None
        self._closed = False
        self._close_task = None
        self._batch_tracker = BatchTracker()
        self._progress_renderer = progress_renderer
        self._progress_enabled = self.progress_max_items != 0
        self._progress_renderer_closed = False

    @property
    def batch_progress(self) -> BatchProgressState:
        """Return the latest provider-neutral batch progress snapshot."""

        return self._batch_tracker.snapshot()

    def register_provider(self, provider: "AzureOpenAIBatchProvider"):
        """
        Store provider as an attribute and start up batching task
        """
        if self.task is not None or self._closed:
            raise RuntimeError("OpenAIBatchAPIHelper is already registered.")

        self.provider = provider
        self._progress_enabled = self._progress_enabled and not provider.quiet
        if self._progress_renderer is None:
            try:
                self._progress_renderer = TqdmBatchProgressRenderer(
                    max_items=self.progress_max_items,
                    disable=not self._progress_enabled,
                )
            except Exception:
                self._progress_enabled = False
                logger.exception("Unable to initialize batch progress display:")

        # Create a Semaphore to ensure only n_workers batches running concurrently
        self.lock = asyncio.Semaphore(self.n_workers)
        self.task = asyncio.create_task(self._batcher(), name="OpenAIBatchHelper")

    @staticmethod
    def _batch_snapshot(batch: Batch) -> BatchSnapshot:
        """Translate an OpenAI batch object into provider-neutral progress."""

        request_counts = None
        if batch.request_counts is not None:
            request_counts = ProgressRequestCounts(
                total=batch.request_counts.total,
                completed=batch.request_counts.completed,
                failed=batch.request_counts.failed,
            )

        name = batch.id
        if isinstance(batch.metadata, dict):
            metadata_name = batch.metadata.get("name")
            if isinstance(metadata_name, str) and metadata_name:
                name = metadata_name
        return BatchSnapshot(
            id=batch.id,
            name=name,
            created_at=datetime.fromtimestamp(batch.created_at, tz=UTC),
            status=batch.status,
            terminal=batch.status in OPENAI_BATCH_TERMINAL_STATUSES,
            request_counts=request_counts,
        )

    def _refresh_batch_progress(self) -> None:
        if not self._progress_enabled or self._progress_renderer is None:
            return

        try:
            self._progress_renderer.refresh(self.batch_progress)
        except Exception:
            self._progress_enabled = False
            logger.exception("Unable to refresh batch progress display")

    def _record_batch_status(self, batch: Batch) -> BatchSnapshot | None:
        try:
            snapshot = self._batch_snapshot(batch)
            self._batch_tracker.update(snapshot)
        except Exception:
            logger.exception("Unable to track batch progress")
            return None

        self._refresh_batch_progress()
        return snapshot

    def _finish_interrupted_batch(
        self, batch: BatchSnapshot | None, status: str
    ) -> None:
        if batch is None or batch.terminal:
            return

        try:
            self._batch_tracker.finish(replace(batch, status=status, terminal=True))
        except Exception:
            logger.exception("Unable to finish batch progress tracking")
            return

        self._refresh_batch_progress()

    def _close_progress_renderer(self) -> None:
        if self._progress_renderer_closed:
            return

        self._progress_renderer_closed = True
        if self._progress_renderer is None:
            return

        try:
            self._progress_renderer.close()
        except Exception:
            logger.exception("Unable to close batch progress display")

    def _batch_handler_callback(self, task: asyncio.Task):
        """
        Simple callback handler for batch tasks
        """
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Batch task was cancelled.")
        except Exception:
            logger.exception("Batch task resulted in an error")
        finally:
            self.batch_tasks.discard(task)

    def _cleanup_local_requests(self) -> None:
        """
        Drain queued requests and cancel futures that cannot receive a result.
        """
        if not hasattr(self, "provider"):
            return

        while True:
            try:
                self.provider.batch_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self.provider.batch_q.task_done()

        for future in list(self.provider.batch_out.values()):
            if not future.done():
                future.cancel()
        self.provider.batch_out.clear()

    async def _close(self) -> None:
        try:
            await super().close()
        finally:
            try:
                self._cleanup_local_requests()
            finally:
                self._close_progress_renderer()

    async def close(self):
        """
        Close the helper and cancel any requests that have not received a result.
        """
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close(), name="OpenAIBatchHelperClose"
            )

        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            # The shared close operation continues, but local callers must not hang.
            self._cleanup_local_requests()
            raise

    async def _batcher(self):
        """
        Batch loop

        This is a co-routine that puts all of our messages together into a batch, sends them off,
        and stores the results back into a dict for the individual agents to handle.

        It's started at init time if we select batch mode, and persists for the duration of the session.
        """
        logger.info("OpenAIBatchHelper opening.")
        while True:
            try:
                # Define our batch and start the clock
                batch = []
                # Wait until first query comes in
                req = await self._get_batch_request()
                batch.append(req)

                # Await new messages to load into the batch
                # - Until we hit our max batch size, or
                # - Until we've waited for the time indicated (default 2s)
                while len(batch) < self.batch_size:
                    try:
                        req = await self._get_batch_request(timeout=self.timeout)
                    except TimeoutError:
                        break
                    batch.append(req)

                # Wait for semaphore to send off batch task
                await self.lock.acquire()

                batch_task = asyncio.create_task(self._batch_handler(batch))
                self.batch_tasks.add(batch_task)

                # Batch task should remove itself from the list once it's done
                batch_task.add_done_callback(self._batch_handler_callback)

            except (asyncio.CancelledError, GeneratorExit):
                # If the task was cancelled, we should exit the loop
                logger.info("OpenAIBatchHelper closing.")

                break

    async def _get_batch_request(
        self, timeout: float | None = None
    ) -> BatchRequestInput:
        """Retrieve and account for one queued request, including cancellation."""
        request_task = asyncio.create_task(self.provider.batch_q.get())
        try:
            if timeout is None:
                request = await request_task
            else:
                request = await asyncio.wait_for(request_task, timeout=timeout)
        except BaseException:
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            if not request_task.cancelled() and request_task.exception() is None:
                self.provider.batch_q.task_done()
            raise

        self.provider.batch_q.task_done()
        return request

    async def _batch_handler(self, batch: list[BatchRequestInput]) -> None:
        """
        A handler method that submits the batch of tasks to OpenAI and retrieves the results
        when finished.
        """
        latest_snapshot: BatchSnapshot | None = None

        def status_callback(batch_status: Batch) -> None:
            nonlocal latest_snapshot
            snapshot = self._record_batch_status(batch_status)
            if snapshot is not None:
                latest_snapshot = snapshot

        # Create batch file, send to OpenAI and execute
        try:
            batch_file = await self.provider.send_batch(batch)
            batch_task = await self.provider.create_batch_task(
                batch_file,
                timeout=self.api_timeout,
                status_callback=status_callback,
            )

            if batch_task.errors is not None and batch_task.errors.data is not None:
                # Batch returned an error. Raise
                errors = "\n".join(
                    f"[{err.code}]: {err.message}" for err in batch_task.errors.data
                )
                logger.error(f"Batch {batch_task.id} returned an error:\n{errors}")
                raise RuntimeError(
                    f"Batch {batch_task.id} returned an error:\n{errors}"
                )

            # Get results
            results = await self.provider.get_batch_results(batch_task)

            # Write out results to dict for agents to pick up
            expected_ids = {batch_item["custom_id"] for batch_item in batch}
            returned_ids = set()
            for result in results:
                if not isinstance(result, dict):
                    logger.warning(
                        f"Batch [{batch_task.id}] returned an unexpected result."
                    )
                    continue
                custom_id = result.get("custom_id")
                if not isinstance(custom_id, str) or custom_id not in expected_ids:
                    logger.warning(
                        f"Batch [{batch_task.id}] returned an unexpected result."
                    )
                    continue

                returned_ids.add(custom_id)
                future = self.provider.batch_out.get(custom_id)
                if future is None or future.done():
                    continue

                try:
                    response = ChatCompletion.model_validate(result["response"]["body"])
                except (KeyError, TypeError, ValidationError) as e:
                    future.set_exception(e)
                else:
                    future.set_result(response)

            for custom_id in expected_ids - returned_ids:
                future = self.provider.batch_out.get(custom_id)
                if future is None or future.done():
                    continue
                future.set_exception(
                    RuntimeError(
                        f"Batch [{batch_task.id}] returned no result for "
                        f"request [{custom_id}]."
                    )
                )

            # Log that we're done
            logger.info(f"Batch [{batch_task.id}] completed.")

        except asyncio.CancelledError:
            self._finish_interrupted_batch(
                latest_snapshot, BATCH_TRACKING_CANCELLED_STATUS
            )
            for batch_item in batch:
                future = self.provider.batch_out.get(batch_item["custom_id"])
                if future is not None and not future.done():
                    future.cancel()
            raise

        except Exception as e:
            self._finish_interrupted_batch(
                latest_snapshot, BATCH_TRACKING_FAILED_STATUS
            )
            # propagate the exception to the futures
            for batch_item in batch:
                fut = self.provider.batch_out.get(batch_item["custom_id"])
                if fut is not None and not fut.done():
                    # If the future is not done, set it to an exception
                    fut.set_exception(e)

            # Signal to batcher as well
            raise

        finally:
            self.lock.release()


class OpenAIObservable(Observable[CompletionUsage]):
    @staticmethod
    def usage_adapter(usage: CompletionUsage | None) -> LLMUsage:
        if usage is None:
            out = LLMUsage()
        else:
            out = LLMUsage(
                input_tok=usage.prompt_tokens,
                output_tok=usage.completion_tokens,
                total_tok=usage.total_tokens,
                round_trips=1,
            )
        return out


class _AzureProvider[AgentT: _Agent, ProviderModeT: Literal["chat", "batch"]](
    _Provider[AgentT], OpenAIObservable
):
    """
    An Azure OpenAI Provider for language Agents.

    This provider generally assumes you already have all required environment variables
    set correctly, or will provide them as kwargs which will be passed to AsyncAzureOpenAI at init

    Namely:
    - api_version or OPENAI_API_VERSION
    - azure_endpoint or AZURE_OPENAI_ENDPOINT

    A bearer token generator will be passed to the AsyncAzureOpenAI constructor so long running tasks will not fail due to token expiration.

    :param str model_name: Model name from the deployments list to use
    :param bool interactive: Should authentication use an Interactive AD Login (T), or ClientSecret (F)?
    :param str resource_endpoint: The Azure API endpoint used to retrieve a bearer token in the auth flow
    :param **kwargs: Any additional kw-args for AsyncAzureOpenAI
    """

    tool_call_wrapper = OpenAIToolCall
    llm: openai.AsyncAzureOpenAI | openai.AsyncOpenAI
    mode: ProviderModeT
    model_name: str
    interactive: bool
    resource_endpoint: str

    def __init__(
        self,
        model_name: str,
        interactive: bool,
        resource_endpoint: str = "https://cognitiveservices.azure.com/.default",
        **kwargs,
    ):
        super().__init__(model_name)
        self.model_name = model_name
        self.interactive = interactive
        self.resource_endpoint = resource_endpoint
        self.authenticate()
        self.llm = openai.AsyncAzureOpenAI(
            azure_ad_token_provider=self._bearer_token_generator, **kwargs
        )

    def authenticate(self) -> None:
        """
        Retrieve Azure OpenAI API key via ClientSecret authentication and
        """

        credential = ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        )

        self._bearer_token_generator = get_bearer_token_provider(
            credential, self.resource_endpoint
        )

    @backoff.on_exception(backoff.expo, openai.APIError, max_tries=3)
    async def prompt_agent(
        self,
        ag: AgentT,
        prompt: list[ChatCompletionMessageParam] | ChatCompletionMessageParam,
        **kwargs,
    ) -> Choice:
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

        res = await self.endpoint_fn(messages=prompt, model=self.model_name, **kwargs)

        out = res.choices[0]

        # Track usage at the provider level and agent level
        self.all_usage.append(self.usage_adapter(res.usage))
        ag.all_usage.append(self.usage_adapter(res.usage))

        # HACK: OpenAI API can't handle None in a roundtrip
        # so we have to patch the message content so it doesn't throw an error.
        if out.message.content is None:
            out.message.content = "<None>"
        ag.scratchpad += "--- Output --------------------------\n"
        ag.scratchpad += "Message:\n"
        ag.scratchpad += out.message.content + "\n"

        # attempt to parse tool call arguments
        # BUG: OpenAI sometimes doesn't return a "tool_calls" reason and uses "stop" instead. Annoying.
        if len(ag.TOOLS) and (
            out.finish_reason == "tool_calls"
            or (
                out.finish_reason == "stop"
                and out.message.tool_calls
                and len(out.message.tool_calls)
            )
        ):
            # Patch finish_reason if it was actually a tool call but didn't
            # indicate it
            out.finish_reason = "tool_calls"
            # Append GPT response to next payload
            # NOTE: This has to come before the next step of parsing
            ag.tool_res_payload.append(out.message.model_dump())

        logger.debug(f"Received response: {out.message.content}")

        if out.finish_reason == "length":
            ag.truncated = True
            ag.scratchpad += (
                "Response returned truncated from OpenAI due to token length.\n"
            )
            logger.warning("Message returned truncated.")

        ag.scratchpad += "\n-----------------------------------\n"

        return out


class AzureOpenAIProvider[AgentT: _Agent](_AzureProvider[AgentT, Literal["chat"]]):
    mode = "chat"

    def __init__(self, model_name: str, interactive: bool, **kwargs):
        super().__init__(model_name, interactive, **kwargs)
        self.endpoint_fn = self.round_trip_increment(self.llm.chat.completions.create)


class AzureOpenAIBatchProvider[AgentT: _Agent](
    _AzureProvider[AgentT, Literal["batch"]]
):
    """
    Azure OpenAI using the Batch API

    This provider is designed to handle large batches of requests to OpenAI's Batch API, which are completed asynchronously.
    Each batch typically takes several minutes or longer to be evaluated, but many requests can be sent at once,
    and the price/request is generally around half of the standard chat endpoint.

    Additionally, one can send multiple batches at once, further speeding up processing time on large tasks.
    """

    mode = "batch"

    def __init__(
        self,
        model_name: str,
        *,
        interactive: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_workers: int = 1,
        batch_handler: OpenAIBatchAPIHelper | None = None,
        progress_max_items: int | None = DEFAULT_BATCH_PROGRESS_MAX_ITEMS,
        quiet: bool = False,
        resource_endpoint: str = "https://cognitiveservices.azure.com/.default",
        **kwargs,
    ):
        """
        Using AzureOpenAI with the Batch API mode. Each batch typically takes several minutes or longer to be evaluated, but many requests
        can be sent at once, and the price/request is generally around half of the standard chat endpoint.

        :param str model_name: The name of an Azure OpenAI deployment (note: must be a batch-capable model, such as gpt-4o-batch)
        :param bool interactive: Should the requests be run in interactive mode using EntraID credentials
        :param int batch_size: The maximum size of batches that should be sent to OpenAI at a time
        :param int n_workers: If `batch_handler` is not provided, the number of workers to run in parallel to process incoming requests (default: 1)
        :param OpenAIBatchAPIHelper batch_handler: (optional) An initialized batch handler which will be used to handle the inqueue of requests to send to openAI
        :param int | None progress_max_items: (default: 10) Maximum number of active batches to display individually. Zero disables the display.
        :param bool quiet: If True, suppresses batch progress output
        :param kwargs: Any keyword arguments to pass to OpenAI class

        """
        self.batch_size = batch_size
        self.batch_idx = 1
        self.batch_idx_lock = asyncio.Lock()
        self.endpoint_fn = self.query_batch_mode
        self.batch_q: asyncio.Queue[BatchRequestInput] = asyncio.Queue()
        self.batch_out: dict[str, asyncio.Future[ChatCompletion]] = {}
        self.quiet = quiet

        if batch_handler is None and progress_max_items is not None:
            self.batch_handler = OpenAIBatchAPIHelper(
                batch_size=batch_size,
                n_workers=n_workers,
                progress_max_items=progress_max_items,
            )
        elif batch_handler is not None and progress_max_items is None:
            self.batch_handler = batch_handler
        else:
            # odd case: custom batch handler but progress_max_items is also set
            raise ValueError(
                "progress_max_items must be configured on a custom batch_handler."
            )

        super().__init__(model_name, interactive, **kwargs)

        # Register the batch handler only after provider initialization succeeds.
        self.batch_handler.register_provider(self)

    @property
    def batch_progress(self) -> BatchProgressState:
        """Return the latest provider-neutral batch progress snapshot."""

        return self.batch_handler.batch_progress

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Clean up the batch handler and any ongoing tasks
        """
        if hasattr(self, "batch_handler") and self.batch_handler is not None:
            # Cancel the batch processing task
            await self.batch_handler.close()

    async def query_batch_mode(
        self, messages: list[ChatCompletionMessageParam], model: str, **kwargs
    ) -> ChatCompletion:
        async with self.batch_idx_lock:
            if self.batch_handler._closed:
                raise RuntimeError("OpenAIBatchAPIHelper is closed.")
            task_id = f"task-{self.batch_idx}"
            self.batch_idx += 1

        task = cast(
            BatchRequestInput,
            {
                "custom_id": task_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, **kwargs, "messages": messages},
            },
        )

        # Create a future to hold the result
        future = asyncio.get_running_loop().create_future()
        self.batch_out[task_id] = future

        try:
            # Put the task into the queue for processing
            await self.batch_q.put(task)

            # Await the result
            return await future
        finally:
            # Remove the future even if its caller is cancelled.
            self.batch_out.pop(task_id, None)

    @staticmethod
    def _serialize_request(task: BatchRequestInput) -> bytes:
        return (json.dumps(task) + "\n").encode("utf-8")
        
    @classmethod
    def _create_batch_file(
        cls,
        tasks: list[BatchRequestInput],
    ) -> tuple[str, bytes, str]:
        """
        Create a batch file for the OpenAI Batch API

        :param tasks: list of task dictionaries to be sent to OpenAI
        :return: Tuple containing the file name, file content, and MIME type to send as an API payload
        """

        batch_file_content = b"".join(cls._serialize_request(task) for task in tasks)

        return (
            "batch_tasks.jsonl",
            batch_file_content,
            "application/json",
        )

    async def send_batch(
        self,
        tasks: list[BatchRequestInput],
        **kwargs,
    ) -> FileObject:
        """
        Send a batch file to OpenAI pending further processing.

        :param tasks: list of task dictionaries to be sent to OpenAI
        :param kwargs: Additional keyword arguments for the file upload (see OpenAI API documentation)

        :return: An OpenAI File object representing the uploaded batch file
        """
        file_name, file_content, mime_type = await asyncio.to_thread(
            self._create_batch_file, tasks
        )

        file = await self.llm.files.create(
            file=(file_name, file_content, mime_type), purpose="batch", **kwargs
        )

        logger.info(f"Created file [{file.id}] with {len(tasks)} queries.")

        return file

    async def create_batch_task(
        self,
        batch_file: FileObject,
        timeout: int = 30,
        status_callback: BatchStatusCallback | None = None,
        **kwargs,
    ) -> Batch:
        """
        Create a batch from an existing batch file object.

        :param FileObject batch_file: An OpenAI File object representing the batch file
        :param int timeout: polling timeout waiting for response
        :param status_callback: Optional synchronous callback invoked with the created batch
            and every subsequently retrieved batch state
        :param kwargs: Additional keyword arguments for the batch creation

        :return: The terminal OpenAI Batch object
        """
        if status_callback is None:
            status_callback = _noop_batch_status_callback

        try:
            batch = await self.llm.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                **kwargs,
            )
            logger.info(f"Executing batch task [{batch_file.id}] -> [{batch.id}]")
        except Exception:
            logger.error(f"Attempt to process batch {batch_file.id} failed!")
            raise

        try:
            status_callback(batch)
            while batch.status not in OPENAI_BATCH_TERMINAL_STATUSES:
                logger.info(f"Batch [{batch.id}] Status: {batch.status}")
                batch = await self.llm.batches.retrieve(batch.id)
                status_callback(batch)
                if batch.status not in OPENAI_BATCH_TERMINAL_STATUSES:
                    await asyncio.sleep(timeout)

        except (Exception, asyncio.CancelledError):
            # Cancel the remote batch before terminating while preserving the error.
            if batch.status not in OPENAI_BATCH_TERMINAL_STATUSES:
                try:
                    await self.llm.batches.cancel(batch.id)
                except Exception:
                    logger.exception(f"Error cancelling batch [{batch.id}]:")
            logger.exception(f"Error processing batch [{batch.id}]!")
            raise

        return batch

    async def get_batch_results(self, batch: Batch) -> list[dict]:
        """
        Retrieve the results of a completed batch.

        :param batch: An OpenAI Batch object representing the completed batch

        :return: A list of results from the batch
        """
        if (
            batch.status not in OPENAI_BATCH_SUCCESS_STATUSES
            or batch.output_file_id is None
        ):
            raise ValueError("Batch status was not 'completed'! Got: " + batch.status)

        result_stream = await self.llm.files.content(batch.output_file_id)
        results = await asyncio.to_thread(
            self._response_from_bytes, result_stream.content
        )
        # TODO: Now this diverges from how we do it with a chat endpoint
        # but maybe no reason to overcomplicate things.
        self.round_trips += 1

        return results

    @staticmethod
    def _response_from_bytes(stream: bytes) -> list[dict]:
        out = []
        with BytesIO() as buffer:
            buffer.write(stream)
            result_text = buffer.getvalue().decode("utf-8")
            for line in result_text.splitlines():
                out.append(json.loads(line))

        return out


class OpenAIProvider[AgentT: _Agent](AzureOpenAIProvider[AgentT]):
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
