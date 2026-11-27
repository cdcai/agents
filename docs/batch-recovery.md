# Batch scheduling and recovery

Each `OpenAIBatchAPIHelper` runs one upload worker, one creation worker, one
result-download worker, and one status coordinator. Each operation queue holds
up to `max(1, 2 * n_workers)` jobs; `n_workers` separately limits concurrent batch
lifecycles. The provider owns authentication, the SDK client, response parsing,
and single-attempt API calls. The helper owns queues, retries, polling, and
shutdown. Coordination is local to a provider/helper instance; it does not
coordinate other processes or instances.

Status checks use individual retrieval by default. Opt into Azure's
[filtered batch listing](https://learn.microsoft.com/azure/ai-services/openai/how-to/batch)
with `AzureOpenAIBatchProvider(..., polling_strategy="list")`. Each sweep starts
with a fresh cursor, filters from one second before the oldest tracked batch,
and reads at most ten pages of 100 items, including terminal states. Missing IDs
are retrieved individually. An explicit unsupported-filter response switches
the instance to retrieval and logs a warning once.

For a custom helper, configure polling on the helper itself:

```python
helper = OpenAIBatchAPIHelper(
    batch_size=1000, n_workers=4, poll_frequency=60,
    polling_strategy="list", progress_max_items=10,
)
provider = AzureOpenAIBatchProvider(
    "gpt-4o-batch", batch_handler=helper, progress_max_items=None,
)
```

## Operation retries

The provider's `RetryPolicy` bounds upload, download, and polling retries.
Cooldowns honor `retry-after-ms` and `Retry-After`; otherwise they use capped
exponential backoff with jitter. A cooldown also delays the next queued job
after retry exhaustion. Creation always makes a single attempt to avoid
creating duplicate remote work after an ambiguous response.

When list polling exhausts its retry budget, the helper retrieves unresolved
batch IDs individually. Transient retrieval failures retain registrations and
wait at least one second before another observation cycle. Set
`batch_polling_deadline` to bound the total tracking duration. Without a deadline,
tracking continues until remote completion or explicit cancellation. Permanent
observation failures preserve the batch ID and last status without cancelling
remote work.

Batch request queueing has no enclosing retry or operation timeout: uploads,
creation, polling, and retrieval own their individual budgets. This also applies
after an explicit context rebuild.

## Result recovery

Result retrieval uses its own `RetryPolicy` budget across metadata refreshes and
file downloads. Each pass downloads a file at most once, retaining readable
sibling files and previously recovered rows. Transient download failures, HTTP
404s for known result files, and fewer rows than terminal request counts trigger
bounded recovery against the existing batch ID. Recovery never uploads or
creates a new batch. Malformed rows and terminal request errors are not replayed.
If recovery exhausts its budget, readable results still resolve normally; affected
requests receive `batch_result_retrieval_failed` with batch/file diagnostics,
instead of an unexplained missing-result error. These errors remain operation
scoped, so processor retries cannot repeat remote model work.

Use `RetryPolicy(elapsed_time_budget=120)` to bound the entire result recovery,
including downloads, metadata requests, queue waits, and backoff. Without an
elapsed budget, attempt counts are bounded and each SDK call uses the configured
SDK timeout. For longer outages, retain exception batch IDs (or the uploaded
`input_file_id` after an ambiguous creation) for later reconciliation. Recovery
state is local to the running helper; it is not persisted across process restarts.
Normalized SDK errors preserve `status_code`, `request_id`, and
`remote_outcome_unknown`. File-read warnings include these identifiers without
logging result payloads.

## Cancellation and shutdown

`send_batch()` borrows its stream until completion and rewinds to its initial
offset for every attempt. Cancelling it waits until execution releases the
stream. `send_batch()` and `create_batch_task()` forward additional SDK keyword
arguments. Status callback failures are logged without stopping polling.
Cancelling an individual `query_batch_mode()` caller abandons only that result;
cancelling a batch lifecycle cancels the remote batch. Polling deadlines refresh metadata within a bounded cleanup window and recover
available partial results without cancelling remote work. Closing the
provider rejects new work and settles helper work and progress rendering before
closing the SDK transport.
