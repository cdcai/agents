import asyncio
import logging
from abc import ABCMeta, abstractmethod
from itertools import islice
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, Type

import polars as pl
from tqdm import tqdm

from .abstract import _Provider
from .agent import Agent
from .providers import AzureOpenAIProvider

logger = logging.getLogger(__name__)


class _BatchProcessor(metaclass=ABCMeta):
    """
    A virtual class for a processor that maps over a large iterable
    where 1 output is expected for every 1 input
    """

    def __init__(
        self,
        data: Iterable,
        agent_class: type[Agent],
        provider: Optional[Type[_Provider]],
        batch_size: int = 5,
        n_workers: int = 1,
        n_retry: int = 5,
        **kwargs,
    ):
        """
        A Processor which operates on chunks of an Iterable.
        Each chunk must be independent, as state will not be maintained between agent calls.

        :param Iterable data: An object which will be split into chunks of `batch_size` and passed as-is to `agent_class`
        :param Agent agent_class: An uninitialized class which will be used to process the batches of `data` (which it takes as a named argument, `batch`, for formatting)
        :param _Provider provider: Optionally, an initialized LLM provider to use with the processor
        :param int batch_size: Number of rows per batch of `data`
        :param int n_workers: Number of workers to use during processing. >1 will spawn parallel workers using concurrent.futures
        :param int n_retry: For each agent, how many round trips to try before giving up
        :param kwargs: Additional named arguments passed to `agent_class` on init
        """
        self.agent_class = agent_class
        self.data = data
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.n_retry = n_retry
        self.agent_kwargs = kwargs

        if provider is not None:
            self.provider = provider
        else:
            # Default to AzureOpenAI
            try:
                self.provider = AzureOpenAIProvider(
                    model_name=kwargs["model_name"], interactive=True
                )
            except KeyError:
                raise RuntimeError(
                    "If `provider` is not passed, `model_name` must be passed to initialize one!"
                )

        # Parallel queues
        #                                idx, retries remaining, batch
        self.in_q: asyncio.PriorityQueue[Tuple[int, int, Any]] = asyncio.PriorityQueue()
        #                                        idx, response
        self.out_q: asyncio.PriorityQueue[Tuple[int, Any]] = asyncio.PriorityQueue()

    def _load_inqueue(self):
        """
        Load inqueue for parallel processing
        """
        logger.info("Loading Inqueue")
        for i, batch in enumerate(self._iter()):
            self.in_q.put_nowait((i, self.n_retry, batch))

        self.error_tasks = 0

    @staticmethod
    def dequeue(q: asyncio.Queue) -> list:
        out = []
        while q.qsize() > 0:
            try:
                it = q.get_nowait()
                out.append(it)
            except asyncio.QueueEmpty:
                break
        return out

    @abstractmethod
    def _iter(self) -> Iterator:
        """
        Abstract method that should return self.data chunked by self.batch_size
        """
        raise NotImplementedError()

    @abstractmethod
    def _placeholder(self, batch: Any) -> Any:
        """
        Abstract method which should return an appropriately sized placholder data piece
        that will be inserted in place of a real prediction if we encounter an error
        """
        raise NotImplementedError()

    async def process(self):
        """
        Process all samples from input data using language agent, splitting by chunk size specified at init

        :return Iterable: The predicted values after mapping over the input iterable (also stored in self.predicted)
        """
        # Either the workers we called for at init or the number of batches we have to process
        # (whichever is fewer)
        self._load_inqueue()
        n_workers = min(self.n_workers, self.in_q.qsize())
        logger.info(
            f"[_process_parallel] processing {self.in_q.qsize()} queries on {n_workers} threads"
        )

        self.pbar = tqdm(total=self.in_q.qsize(), desc="Batch Processing", unit="Batch")
        workers = []

        for i in range(n_workers):
            workers.append(
                asyncio.create_task(
                    self._worker(f"llm_worker-{i}", self.in_q, self.out_q),
                    name=f"llm_worker-{i}",
                )
            )

        # Wait until the queue is processed
        await self.in_q.join()

        # Terminate workers
        for worker in workers:
            worker.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

        self.pbar.close()

        if self.error_tasks > 0:
            logger.warning(
                f"[process] There were {self.error_tasks} unsucessful batches!"
            )

        # De-queue into list from output queue
        out = []
        for _, msg in self.dequeue(self.out_q):
            if isinstance(msg, list):
                out.extend(msg)
            else:
                out.append(msg)

        return out

    async def _worker(
        self,
        worker_name: str,
        in_q: asyncio.PriorityQueue,
        out_q: asyncio.PriorityQueue,
    ):
        """
        Agent worker
        """
        while True:
            try:
                (id, retry_left, data) = await in_q.get()
                errored = False
                agent = self._spawn_agent(data)
                answer = await agent()

                if len(answer) == 0:
                    logger.error(
                        f"[_worker - {worker_name}]: No answer was provided for query {id}"
                    )
                    errored = True

            except asyncio.CancelledError:
                logger.info(
                    f"[_worker - {worker_name}]: Got CancelledError, terminating."
                )
                break

            except Exception as e:
                logger.error(f"[_worker - {worker_name}]: Task {id} failed, {str(e)}")
                errored = True

            if errored:
                retry_left -= 1
                if retry_left < 0:
                    # End retries, fill in data with placeholder
                    logger.error(
                        f"[_worker - {worker_name}]: Task {id} failed {self.n_retry} times and will not be retried"
                    )
                    answer = self._placeholder(data)
                    self.error_tasks += 1
                else:
                    # Send data back to queue to retry processing
                    logger.info(
                        f"[_worker - {worker_name}]: Task {id} - {retry_left} retries remaining"
                    )
                    await in_q.put((id, retry_left, data))
                    in_q.task_done()
                    continue

            await out_q.put((id, answer))
            self.pbar.update()
            in_q.task_done()

    @staticmethod
    def _batch_format(batch: Any) -> str:
        """
        An optional formatter to convert batch into a str
        """
        return str(batch)

    def _spawn_agent(self, batch: Iterable, **kwargs) -> Agent:
        """
        Spawn agent for next run, formatting batch as appropriate

        :param batch: An iterable, the next piece of data to be processed from in_q
        :param kwargs: Any additional named arguments passed to initializer of agent class

        :return: An initialized Agent class
        """
        batch_str = self._batch_format(batch)
        out = self.agent_class(
            provider=self.provider, **self.agent_kwargs, batch=batch_str, **kwargs
        )  # type: ignore[misc]
        return out


class BatchProcessor(_BatchProcessor):
    """
    A batch processor which maps an Agent over elements of an iterable (usually a list[str]).

    Each chunk of the iterable is operated on independently.
    """

    def _iter(self) -> Iterator:
        """
        Just a backport of itertools.batched

        Yields a batch of `self.data` of len `self.batch_size` until data is exhausted.
        """
        iterator = iter(self.data)
        while batch := tuple(islice(iterator, self.batch_size)):
            yield batch

    def _placeholder(self, batch: Sequence):
        """
        Returns a `List[str]` with `len() == len(batch)`
        """
        resp_obj = (
            ""
            if self.agent_class.output_len == 1
            else ("",) * self.agent_class.output_len
        )
        return [resp_obj] * len(batch)


class DFBatchProcessor(_BatchProcessor):
    """
    A Processor which operates on chunks of a polars dataframe.
    Each chunk must be independent, as state will not be maintained between agent calls.

    The main user-facing method after init is :func:`process()`
    """

    data: pl.DataFrame

    def _iter(self) -> Iterator[pl.DataFrame]:
        return self.data.iter_slices(self.batch_size)

    def _placeholder(self, batch: pl.DataFrame):
        """
        Returns a List[str] with len() == batch.height
        """
        resp_obj = (
            ""
            if self.agent_class.output_len == 1
            else ("",) * self.agent_class.output_len
        )
        return [resp_obj] * batch.height

    @staticmethod
    def _batch_format(batch: pl.DataFrame) -> str:
        """
        Write out batch argument as ndJSON formatted str to insert into BASE_PROMPT
        """
        return batch.write_ndjson()
