"""Instance-local operation workers and a shared batch status coordinator."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import IO, Any, Literal

from ..exceptions import AgentError, BatchPollingDeadlineExceeded, RetryableAgentError
from ..retries import RetryPolicy, is_retryable_exception, retry_operation

logger = logging.getLogger(__name__)
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


@dataclass(eq=False)
class OperationJob:
    operation: Callable[..., Awaitable[Any]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future
    retry: bool = True
    execution: asyncio.Task | None = None


@dataclass
class PollRegistration:
    batch: Any
    callback: Callable
    interval: float
    next_check: float
    deadline: float | None
    future: asyncio.Future


class BatchScheduler:
    """Scheduling belongs to the helper; all SDK calls belong to the provider."""

    def _init_scheduler(
        self, n_workers: int, polling_strategy: Literal["retrieve", "list"]
    ):
        if n_workers < 1:
            raise ValueError("n_workers must be positive")
        if polling_strategy not in {"retrieve", "list"}:
            raise ValueError("polling_strategy must be 'retrieve' or 'list'")
        self.polling_strategy = polling_strategy
        self._queues: dict[str, asyncio.Queue[OperationJob]] = {
            name: asyncio.Queue(maxsize=max(1, 2 * n_workers))
            for name in ("upload", "creation", "download")
        }
        self.send_batch_q = self._queues["upload"]
        self._jobs: set[OperationJob] = set()
        self._lifecycles: set[asyncio.Task] = set()
        self._puts: set[asyncio.Task] = set()
        self._cooldowns: dict[str, float] = {}
        self._polls: dict[str, PollRegistration] = {}
        self._poll_wakeup = asyncio.Event()
        self._poll_active: asyncio.Task | None = None

    def _start_scheduler(self):
        self.tasks.extend(
            asyncio.create_task(
                self._operation_worker(name, queue), name=f"Batch-{name}"
            )
            for name, queue in self._queues.items()
        )
        self.tasks.append(
            asyncio.create_task(self._poll_coordinator(), name="Batch-polling")
        )

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OpenAIBatchAPIHelper is closed.")

    @staticmethod
    async def _cancel_and_wait(task):
        if not task.done() and not task.cancelling():
            task.cancel()
        # Repeated caller cancellation cannot return a borrowed stream early.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - the result belongs to the job future
                break

    async def _submit(self, name, operation, *args, _retry=True, **kwargs):
        self._check_open()
        job = OperationJob(
            operation, args, kwargs, asyncio.get_running_loop().create_future(), _retry
        )
        self._jobs.add(job)
        # Track blocked producers too, so close can settle them without queue races.
        put = asyncio.create_task(self._queues[name].put(job))
        self._puts.add(put)
        try:
            done, _ = await asyncio.wait(
                {put, job.future}, return_when=asyncio.FIRST_COMPLETED
            )
            if put in done:
                await put
            return await asyncio.shield(job.future)
        finally:
            cleanup = asyncio.create_task(
                self._settle_job(name, job, put, asyncio.current_task())
            )
            interrupted = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    interrupted = True
            cleanup.result()
            if interrupted:
                raise asyncio.CancelledError

    async def _settle_job(self, name, job, put, owner):
        try:
            if not put.done():
                put.cancel()
            await asyncio.gather(put, return_exceptions=True)
            if not job.future.done():
                job.future.cancel()
            if job.execution is not None and not job.execution.done():
                await self._cancel_and_wait(job.execution)
            if (
                name == "creation"
                and owner.cancelling()
                and job.execution is not None
                and job.execution.done()
                and not job.execution.cancelled()
                and job.execution.exception() is None
            ):
                # A completed creation response raced local cancellation.
                batch = job.execution.result()
                await self._cleanup_remote(
                    PollRegistration(batch, lambda batch: None, 0, 0, None, job.future)
                )
            if job.future.done() and not job.future.cancelled():
                job.future.exception()  # Observe errors even if the caller was cancelled.
        finally:
            self._puts.discard(put)
            self._jobs.discard(job)

    async def _operation_worker(self, name, queue):
        while True:
            job = await queue.get()
            try:
                if job.future.done() or self._closed:
                    job.future.cancel()
                    continue
                job.execution = asyncio.create_task(
                    self._request(
                        name,
                        lambda job=job: job.operation(*job.args, **job.kwargs),
                        retry=job.retry and name != "creation",
                    )
                )
                try:
                    result = await job.execution
                except asyncio.CancelledError:
                    job.future.cancel()
                    if asyncio.current_task().cancelling():
                        raise
                except Exception as exc:  # noqa: BLE001 - isolate independent jobs
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                queue.task_done()

    def _retry_delay(self, exc, attempt, policy):
        headers = {
            key.lower(): value
            for key, value in getattr(
                getattr(exc, "response", None), "headers", {}
            ).items()
        }
        try:
            if "retry-after-ms" in headers:
                delay = float(headers["retry-after-ms"]) / 1000
            elif "retry-after" in headers:
                value = headers["retry-after"]
                try:
                    delay = float(value)
                except ValueError:
                    delay = (
                        parsedate_to_datetime(value) - datetime.now(UTC)
                    ).total_seconds()
            else:
                raise ValueError
            if delay >= 0 and delay < float("inf"):
                return delay
        except (ValueError, TypeError, OverflowError):
            pass
        return random.uniform(0.5, 1.0) * min(
            policy.max_delay, policy.initial_delay * 2 ** min(attempt, 30)
        )

    async def _request(self, name, operation, *, retry=True):
        policy = getattr(self.provider, "retry_policy", RetryPolicy())
        attempt = 0

        async def once():
            nonlocal attempt
            delay = self._cooldowns.get(name, 0) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await operation()
            except Exception as exc:
                if is_retryable_exception(exc):
                    self._cooldowns[name] = max(
                        self._cooldowns.get(name, 0),
                        time.monotonic() + self._retry_delay(exc, attempt, policy),
                    )
                attempt += 1
                raise

        # retry_operation bounds elapsed time, including the shared cooldown.
        # Its delay is zero because the operation itself enforces that cooldown.
        zero_delay = RetryPolicy(policy.max_attempts, 0, 0, policy.elapsed_time_budget)
        return await retry_operation(once, zero_delay, retry=retry)

    async def send_batch(self, file_content: IO[bytes], **kwargs):
        self._check_open()
        start = file_content.tell()

        async def upload(stream, offset, **upload_kwargs):
            stream.seek(offset)
            return await self.provider._upload_batch_once(stream, **upload_kwargs)

        return await self._submit("upload", upload, file_content, start, **kwargs)

    async def create_batch_task(self, *args, **kwargs):
        self._check_open()
        task = asyncio.current_task()
        self._lifecycles.add(task)
        try:
            return await self._create_and_poll(*args, **kwargs)
        finally:
            self._lifecycles.discard(task)

    async def _create_and_poll(
        self,
        batch_file,
        timeout=30,
        status_callback=None,
        polling_deadline=None,
        **kwargs,
    ):
        try:
            batch = await self._submit(
                "creation", self.provider._create_batch_once, batch_file, **kwargs
            )
        except AgentError as exc:
            exc.input_file_id = batch_file.id
            exc.details = {
                "operation": "batches.create",
                "input_file_id": batch_file.id,
                "remote_outcome_unknown": exc.remote_outcome_unknown,
                "cause": exc.details,
            }
            raise
        callback = status_callback or (lambda batch: None)
        deadline = polling_deadline
        if deadline is None:
            deadline = getattr(self.provider, "batch_polling_deadline", None)
        registration = PollRegistration(
            batch,
            callback,
            timeout,
            time.monotonic(),
            None if deadline is None else time.monotonic() + deadline,
            asyncio.get_running_loop().create_future(),
        )
        self._notify(registration, batch)
        registration.next_check = time.monotonic()
        if batch.status in TERMINAL:
            return batch
        if self._closed:
            await self._cleanup_remote(registration)
            raise RuntimeError("OpenAIBatchAPIHelper is closed.")
        self._polls[batch.id] = registration
        self._poll_wakeup.set()
        try:
            deadline_context = asyncio.timeout_at(registration.deadline)
            try:
                async with deadline_context:
                    return await asyncio.shield(registration.future)
            except TimeoutError as exc:
                if not deadline_context.expired():
                    raise
                await self._unregister(batch.id, registration)
                await self._cleanup_remote(registration, retrieve=True, cancel=False)
                raise BatchPollingDeadlineExceeded(
                    registration.batch, deadline
                ) from exc
        except BatchPollingDeadlineExceeded:
            raise
        except asyncio.CancelledError:
            await self._unregister(batch.id, registration)
            await self._cleanup_remote(registration)
            raise
        except Exception as exc:
            await self._unregister(batch.id, registration)
            if isinstance(exc, AgentError):
                exc.batch_id = batch.id
                exc.remote_outcome_unknown = True
                exc.details = {
                    "batch_id": batch.id,
                    "batch_status": registration.batch.status,
                    "remote_outcome_unknown": True,
                    "cause": exc.details,
                }
            raise
        finally:
            await self._unregister(batch.id, registration)

    async def _unregister(self, batch_id, registration):
        if self._polls.get(batch_id) is registration:
            self._polls.pop(batch_id)
            # Stop and await the current SDK request before remote cleanup.
            if self._poll_active is not None:
                await self._cancel_and_wait(self._poll_active)
            self._poll_wakeup.set()
        if not registration.future.done():
            registration.future.cancel()
        elif not registration.future.cancelled():
            registration.future.exception()

    async def _cleanup_remote(self, registration, *, retrieve=False, cancel=True):
        if registration.batch.status in TERMINAL:
            return True
        try:
            async with asyncio.timeout(5):
                if cancel:
                    batch = await self.provider._cancel_batch_once(
                        registration.batch.id
                    )
                    if getattr(batch, "id", None) == registration.batch.id:
                        registration.batch = batch
                if retrieve:
                    batch = await self.provider._retrieve_batch_once(
                        registration.batch.id
                    )
                    if getattr(batch, "id", None) == registration.batch.id:
                        registration.batch = batch
        except Exception:
            logger.warning(
                "Unable to confirm remote batch state during cleanup", exc_info=True
            )

    def _notify(self, registration, batch):
        registration.batch = batch
        try:
            registration.callback(batch)
        except Exception:
            logger.exception("Batch status callback failed")
        registration.next_check = time.monotonic() + registration.interval
        if batch.status in TERMINAL and not registration.future.done():
            registration.future.set_result(batch)

    @staticmethod
    def _unsupported_filter(exc):
        message = str(exc).lower()
        return (
            getattr(exc, "status_code", None) in {400, 422, 501}
            and "filter" in message
            and any(
                word in message
                for word in (
                    "unsupported",
                    "not supported",
                    "unrecognized",
                    "unknown query",
                )
            )
        )

    async def _poll_call(self, operation):
        self._poll_active = asyncio.create_task(self._request("polling", operation))
        try:
            return await self._poll_active
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            return None
        finally:
            self._poll_active = None

    async def _poll_due(self, due):
        unresolved = dict(due)
        if self.polling_strategy == "list":
            after = None
            oldest = min(reg.batch.created_at for reg in self._polls.values()) - 1
            try:
                for _ in range(10):
                    if self._closed:
                        return
                    page = await self._poll_call(
                        lambda after=after: self.provider._list_batches_once(
                            oldest, after=after
                        )
                    )
                    if page is None:
                        return
                    for batch in page.data:
                        registration = self._polls.get(batch.id)
                        if registration is not None and batch.id in due:
                            self._notify(registration, batch)
                            unresolved.pop(batch.id, None)
                    if not page.has_more or not page.data:
                        break
                    after = page.data[-1].id
            except Exception as exc:  # noqa: BLE001 - fall back to known IDs
                if self._unsupported_filter(exc.__cause__ or exc):
                    logger.warning(
                        "Azure batch list filter unsupported; using individual retrieval"
                    )
                    self.polling_strategy = "retrieve"
                else:
                    logger.warning(
                        "Batch list polling failed; retrieving %s known batch IDs",
                        len(unresolved),
                    )
        for batch_id, registration in unresolved.items():
            if self._closed:
                return
            if (
                self._polls.get(batch_id) is not registration
                or registration.future.done()
            ):
                continue
            try:
                batch = await self._poll_call(
                    lambda batch_id=batch_id: self.provider._retrieve_batch_once(
                        batch_id
                    )
                )
                if batch is not None and self._polls.get(batch_id) is registration:
                    self._notify(registration, batch)
            except Exception as exc:  # noqa: BLE001 - isolate independent jobs
                if not registration.future.done():
                    if is_retryable_exception(exc) or getattr(
                        exc, "remote_outcome_unknown", False
                    ):
                        # Retain known work during observation loss. The lifecycle's
                        # deadline bounds tracking; the interval prevents busy polling.
                        registration.next_check = time.monotonic() + max(
                            1, registration.interval
                        )
                        logger.warning("Batch [%s] temporarily unobservable", batch_id)
                    else:
                        registration.future.set_exception(exc)

    async def _poll_coordinator(self):
        while not self._closed:
            self._poll_wakeup.clear()
            now = time.monotonic()
            active = {
                key: reg for key, reg in self._polls.items() if not reg.future.done()
            }
            due = {key: reg for key, reg in active.items() if reg.next_check <= now}
            if due:
                try:
                    await self._poll_due(due)
                except Exception as exc:  # noqa: BLE001 - isolate independent jobs
                    # A failed sweep must not kill the coordinator or spin.
                    for reg in due.values():
                        if not reg.future.done():
                            reg.future.set_exception(exc)
                continue
            delay = min((reg.next_check - now for reg in active.values()), default=None)
            try:
                async with asyncio.timeout(delay):
                    await self._poll_wakeup.wait()
            except TimeoutError:
                pass

    async def get_batch_results(self, batch):
        self._check_open()
        results = []
        files = {}
        known_rows = {}
        attempt = 0

        async def download(file_id):
            if file_id not in files:
                # The recovery pass owns retries, so file retries cannot multiply it.
                files[file_id] = await self._submit(
                    "download",
                    self.provider._download_batch_once,
                    file_id,
                    _retry=False,
                )
            return files[file_id]

        async def read():
            nonlocal batch, results, attempt
            attempt += 1
            if attempt > 1:
                refreshed = await self.provider._retrieve_batch_once(batch.id)
                if refreshed.id != batch.id:
                    raise ValueError("Result recovery returned a different batch ID")
                batch = refreshed
            snapshot = await self.provider._read_batch_results(batch, download)
            grouped = {}
            unkeyed = []
            for row in snapshot:
                custom_id = row.get("custom_id") if isinstance(row, dict) else None
                if isinstance(custom_id, str):
                    grouped.setdefault(custom_id, []).append(row)
                else:
                    unkeyed.append(row)
            # A later partial snapshot cannot erase readable rows or hide duplicates.
            for custom_id, rows in grouped.items():
                if len(rows) > len(known_rows.get(custom_id, [])):
                    known_rows[custom_id] = rows
            results = [row for rows in known_rows.values() for row in rows] + unkeyed
            errors = [
                row["_batch_protocol_error"]
                for row in results
                if isinstance(row, dict) and "_batch_protocol_error" in row
            ]
            counts = getattr(batch, "request_counts", None)
            expected = 0 if counts is None else counts.completed + counts.failed
            if batch.status == "completed" and counts is not None:
                expected = counts.total
            received = len(
                {
                    row["custom_id"]
                    for row in results
                    if isinstance(row, dict) and isinstance(row.get("custom_id"), str)
                }
            )
            unavailable = any(error.get("retryable") for error in errors)
            incomplete = not errors and received < expected
            if unavailable or incomplete:
                if incomplete:
                    # A readable but incomplete snapshot may become available later.
                    files.clear()
                raise RetryableAgentError(
                    f"Batch [{batch.id}] results are not yet fully available",
                    code="batch_results_unavailable",
                    details={
                        "batch_id": batch.id,
                        "expected": expected,
                        "received": received,
                        "file_errors": errors,
                    },
                )
            return results

        try:
            return await self._request("results", read)
        except Exception as exc:  # noqa: BLE001 - preserve readable sibling files
            logger.warning(
                "Result recovery exhausted for batch [%s] (output=%s, error=%s, type=%s, code=%s)",
                batch.id,
                batch.output_file_id,
                batch.error_file_id,
                type(exc).__name__,
                getattr(exc, "code", None),
            )
            return [
                *results,
                {
                    "_batch_protocol_error": {
                        "batch_id": batch.id,
                        "output_file_id": batch.output_file_id,
                        "error_file_id": batch.error_file_id,
                        "type": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                        "retryable": is_retryable_exception(exc),
                    }
                },
            ]

    async def _download_file(self, file_id):
        return await self._submit(
            "download", self.provider._download_batch_once, file_id
        )

    async def _stop_scheduler(self):
        lifecycles = list(self._lifecycles)
        for task in lifecycles:
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*lifecycles, return_exceptions=True)
        for job in list(self._jobs):
            job.future.cancel()
        puts = list(self._puts)
        for put in puts:
            put.cancel()
        await asyncio.gather(*puts, return_exceptions=True)
        for registration in list(self._polls.values()):
            registration.future.cancel()
        executions = [job.execution for job in self._jobs if job.execution is not None]
        for execution in executions:
            await self._cancel_and_wait(execution)
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for queue in self._queues.values():
            while not queue.empty():
                job = queue.get_nowait()
                job.future.cancel()
                queue.task_done()
            await queue.join()
