from abc import ABCMeta, abstractmethod
import logging
import queue
import concurrent.futures
from tqdm import tqdm
import polars as pl
import openai
from typing import Iterator, Iterable

from .agent import Agent
from .generic import openai_creds_ad

logger = logging.getLogger(__name__)


class BatchProcessor(metaclass=ABCMeta):
    """
    A virtual class for a processor that maps over a large iterable
    """
    def __init__(self, data: Iterable, agent_class: Agent, batch_size: int = 5, n_workers : int = 1, n_retry : int = 5, interactive : bool = False, **kwargs):
        """
        A Processor which operates on chunks of an Iterable.
        Each chunk must be independent, as state will not be maintained between agent calls.
        
        :param Iterable data: An object which will be split into chunks of `batch_size` and passed as-is to `agent_class`
        :param Agent agent_class: An uninitialized class which will be used to process the chunks of `data` (which it takes as it's first argument)
        :param int batch_size: Number of rows per batch of `data`
        :param int n_workers: Number of workers to use during processing. >1 will spawn parallel workers using concurrent.futures
        :param int n_retry: For each agent, how many round trips to try before giving up
        :param bool interactive: Use interactive authentication (`True`) instead of ClientSecret authentication (not advised, for debugging)
        :param kwargs: Additional named arguments passed to `agent_class` on init
        """
        self.agent_class = agent_class
        self.data = data
        self.n_workers = n_workers
        self.parallel = n_workers > 1
        self.batch_size = batch_size
        self.n_retry = n_retry
        self.agent_kwargs = kwargs
        self.interactive = interactive
        # Parallel queues
        if self.parallel:
            self.in_q = queue.PriorityQueue()
            self.out_q = queue.PriorityQueue()

    def _load_inqueue(self):
        """
        Load inqueue for parallel processing
        """
        logger.info("Loading Inqueue")
        for i, batch in enumerate(self._iter()):
            self.in_q.put_nowait((i, batch))
        
        self.error_tasks = 0

    @staticmethod
    def dequeue(q: queue.Queue) -> list:
        out = []
        while q.qsize() > 0:
            try:
                it = q.get_nowait()
                out.append(it)
            except queue.Empty:
                break
        return out

    @abstractmethod
    def _iter(self) -> Iterator:
        """
        Abstract method that should return self.data chunked by self.batch_size
        """
        raise NotImplementedError()

    @abstractmethod
    def _placeholder(batch: Iterable) -> Iterable:
        """
        Abstract method which should return an appropriately sized placholder data piece
        that will be inserted in place of a real prediction if we encounter an error
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def n_batches(self) -> int:
        """
        Abstract method which should return the int value for the number of batches
        that will be processed after chunking
        (for parallel, this is in_q.qsize() after calling _load_inqueue())
        """
        raise NotImplementedError()

    def _process_parallel(self):
        # Either the workers we called for at init or the number of batches we have to process
        # (whichever is fewer)
        self._load_inqueue()
        n_workers = min(self.n_workers, self.n_batches)
        logger.info(f"[_process_parallel] processing {self.in_q.qsize()} queries on {n_workers} threads")

        with tqdm(total=self.in_q.qsize(), desc="Batch Processing", unit="Batch") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="llm_worker") as executor:
                futures : dict[concurrent.futures.Future, tuple[int, pl.DataFrame]] = {}
                
                # This may take >1 cycle if we have to re-run, so while()
                while True:
                    while self.in_q.qsize() > 0:
                        id, data = self.in_q.get()
                        futures.update({executor.submit(self._spawn_agent_threadpool, id, data): (id, data)})

                    for future in concurrent.futures.as_completed(futures):
                        id, data = futures[future]
                        try:
                            id_out, answer = future.result()
                        except Exception:
                            # Error indicates that we're returning the data for re-processing, so load back into inq
                            logger.warning(f"[_process_parallel]: Batch {id} failed, re-trying.")
                            self.in_q.put((id, data))
                            continue
                        self.out_q.put((id_out, answer))
                        pbar.update()
                    if self.in_q.qsize() == 0:
                        # No more tasks, kill
                        break

        if self.error_tasks > 0:
            logger.warning(f"[_process_parallel] There were {self.error_tasks} unsucessful batches!")
        # De-queue into list from output queue
        out = []
        for _, msg in self.dequeue(self.out_q):
            out.extend(msg)

        return out

    def _spawn_agent_threadpool(self, id: int, data: pl.DataFrame) -> tuple[int, list[str]]:
        try:
            agent = self._spawn_agent(data)
            answer = agent(steps=self.n_retry)
            if len(answer) == 0:
                logger.error(f"[_spawn_agent_threadpool]: No answer was provided for query {id}, filling with missing values!")
                answer = self._placeholder(data)
                self.error_tasks += 1

            logger.info(f"[_spawn_agent_threadpool]: Processed query {id}.")

            return id, answer
        except Exception as e:
            logger.error(f"[_spawn_agent_threadpool]: Task failed, {str(e)}")
            raise e

    def _process_seq(self) -> list[str]:
        out = []
        for i, batch in tqdm(enumerate(self._iter(), 1), total=self.n_batches, desc="Batch Processing", unit="Batch"):
            logger.info(f"[_process_seq] Running batch {i}")
            agent = self._spawn_agent(batch)
            answer = agent(steps=self.n_retry)
            if len(answer) > 0:
                out.extend(answer)
            else:
                logger.error("[_process_seq] No answer was provided, filling with missing values!")
                out.extend(self._placeholder(batch))
        
        return(out)

    def _spawn_agent(self, batch: Iterable):
        openai_creds_ad("Interactive" if self.interactive else "ClientSecret")

        llm = openai.AzureOpenAI()
        
        out = self.agent_class(batch, llm=llm, **self.agent_kwargs)

        return out

    def process(self) -> Iterable:
        """
        Process all samples from input data using language agent, splitting by chunk size specified at init
        
        :return Iterable: The predicted values after mapping over the input iterable (also stored in self.predicted)
        """
        self.predicted = []

        if self.parallel:
            self.predicted = self._process_parallel()
        else:
            self.predicted = self._process_seq()

        return self.predicted

class DFBatchProcessor(BatchProcessor):
    """
    A Processor which operates on chunks of a polars dataframe.
    Each chunk must be independent, as state will not be maintained between agent calls.

    The main user-facing method after init is .process()
    """
    def __init__(self, data: pl.DataFrame, agent_class: Agent, batch_size: int = 5, n_workers : int = 1, n_retry : int = 5, interactive : bool = False, **kwargs):
        """
        A Processor which operates on chunks of a polars dataframe.
        Each chunk must be independent, as state will not be maintained between agent calls.
        
        :param pl.DataFrame data: A Data Frame object which will be split by `batch_size` rows and passed as-is to `agent_class`
        :param Agent agent_class: An uninitialized class which will be used to process the chunks of `data` (which it takes as it's first argument)
        :param int batch_size: Number of rows per batch of `data`
        :param int n_workers: Number of workers to use during processing. >1 will spawn parallel workers using concurrent.futures
        :param int n_retry: For each agent, how many round trips to try before giving up
        :param bool interactive: Use interactive authentication (`True`) instead of ClientSecret authentication (not advised, for debugging)
        :param kwargs: Additional named arguments passed to `agent_class` on init
        """
        super().__init__(data, agent_class, batch_size, n_workers, n_retry, interactive, **kwargs)

    def _iter(self) -> Iterator[pl.DataFrame]:
        return self.data.iter_slices(self.batch_size)

    def _placeholder(batch: pl.DataFrame):
        """
        Returns a List[str] with len() == batch.height
        """
        return [""] * batch.height
    
    @property
    def n_batches(self) -> int:
        if self.parallel:
            n = self.in_q.qsize()
        else:
            n = -(-self.data.height // self.batch_size)

        return n