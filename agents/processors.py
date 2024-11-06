import logging
import queue
import concurrent.futures
from tqdm import tqdm
import polars as pl
import openai

from .agent import Agent
from .generic import openai_creds_ad

logger = logging.getLogger(__name__)

class DFBatchProcessor:
    """
    A Processor which operates on chunks of a polars dataframe.
    Each chunk must be independent, as state will not be maintained between agent calls.

    The main user-facing method after init is .process()
    """
    def __init__(self, df: pl.DataFrame, agent_class: Agent, batch_size: int = 5, n_workers : int = 1, n_retry : int = 5, interactive : bool = False, **kwargs):
        """
        A Processor which operates on chunks of a polars dataframe.
        Each chunk must be independent, as state will not be maintained between agent calls.
        
        :param pl.DataFrame df: A Data Frame object which will be split by `batch_size` rows and passed as-is to `agent_class`
        :param Agent agent_class: An uninitialized class which will be used to process the chunks of `df` (which it takes as it's first argument)
        :param int batch_size: Number of rows per batch of `df`
        :param int n_workers: Number of workers to use during processing. >1 will spawn parallel workers using concurrent.futures
        :param int n_retry: For each agent, how many round trips to try before giving up
        :param bool interactive: Use interactive authentication (`True`) instead of ClientSecret authentication (not advised for `n_workers` > 1)
        :param kwargs: Additional named arguments passed to `agent_class` on init
        """
        self.agent_class = agent_class
        self.df = df
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
        for i, batch in enumerate(self.df.iter_slices(self.batch_size)):
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

    def _process_parallel(self):
        # Either the workers we called for at init or the number of batches we have to process
        # (whichever is fewer)
        self._load_inqueue()
        n_workers = min(self.n_workers, self.in_q.qsize())
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
                answer = [""] * data.height
                self.error_tasks += 1

            logger.info(f"[_spawn_agent_threadpool]: Processed query {id}.")

            return id, answer
        except Exception as e:
            logger.error(f"[_spawn_agent_threadpool]: Task failed, {str(e)}")
            raise e

    def _process_seq(self) -> list[str]:
        out = []
        n_batches = -(-self.df.height // self.batch_size)
        for i, batch in tqdm(enumerate(self.df.iter_slices(self.batch_size), 1), total=n_batches, desc="Batch Processing", unit="Batch"):
            logger.info(f"---Running batch {i}----")
            agent = self._spawn_agent(batch)
            answer = agent(steps=self.n_retry)
            if len(answer) > 0:
                out.extend(answer)
            else:
                logger.error("No answer was provided, filling with missing values!")
                out.extend([""] * batch.height)
        
        return(out)

    def _spawn_agent(self, batch: pl.DataFrame):
        openai_creds_ad("Interactive" if self.interactive else "ClientSecret")

        llm = openai.AzureOpenAI()
        
        out = self.agent_class(batch, llm=llm, **self.agent_kwargs)

        return out

    def process(self) -> pl.DataFrame:
        """
        Process all samples from input data frame using language agent, splitting by chunk size specified at init

        :return pl.DataFrame: The initial dataframe with a new column `response`
        """
        self.predicted = []

        if self.parallel:
            self.predicted = self._process_parallel()
        else:
            self.predicted = self._process_seq()

        self.pred_df = (
            self.df
            .with_columns(response = pl.Series("predicted", values=self.predicted))
        )

        return self.pred_df