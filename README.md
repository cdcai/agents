# Agents
[![Test Suite](https://github.com/cdcai/agents/actions/workflows/test.yaml/badge.svg)](https://github.com/cdcai/agents/actions/workflows/test.yaml)  

Sean Browning (NCEZID/OD Data Science Team)

## Background

I wanted to learn how to create and use language agents to solve complex problems. LangChain wasn't cutting it for me, so I made my own library from first principles to suit a few projects we're working on.

This package contains a few classes that can be used as building blocks for language agents and agentic systems. I plan to expand it with additional functionality as I need it, but keep a minimal footprint (ie. if you're already using openai and pydantic, this should bring no additional dependencies).

All code uses asyncio by design, and though I've tried to generalize as I can, I mostly built around OpenAI and specifically Azure OpenAI since that's what we are allowed to work with internally.

## Really? Another agent framework? Should I even use this library?

Maybe you *shouldn't* use this library.

After all, there are whole teams and companies of real software engineers working on similar frameworks that should suit your use-case:

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [smolagents](https://github.com/huggingface/smolagents)
- [AutoGen](https://github.com/microsoft/autogen)

By design, this library was designed to be python-ic, minimally invasive, and not require a lot of boilerplate to do what is essentially just string formatting and API request handling. It serves me well, but may not serve you.

## Installation

This isn't currently on pypi, so just use pip to install directly via GitHub:

```sh
pip install git+https://github.com/cdcai/multiagent.git
```

## Examples

| Example | Link |
| ---- | ---- |
| Taking output from one agent as input to another in a callback | [agent_with_callback.py](examples/agent_with_callback.py) |
| Getting structured output from agent / Text Prediction | [structured_prediction.py](examples/structured_prediction.py) |
| Batch processing large inputs over the same agent in parallel | [batch_processing.py](examples/batch_processing.py) |
| Processing requests with the OpenAI Batch API | [batch_api.py](examples/batch_api.py) |

### Batch API progress

`AzureOpenAIBatchProvider` displays active remote batches by metadata name (falling
back to the batch ID), UTC start time, status, and request counts when they are
available. The display shows up to 10 batches, oldest first, then aggregates
additional batches by status and keeps terminal status totals on the last line.
Local monitoring failures and cancellations are reported separately as
`tracking_failed` and `tracking_cancelled`, rather than as remote API statuses.

Set the display limit explicitly when creating the provider:

```python
provider = AzureOpenAIBatchProvider(
    "gpt-4o-batch",
    batch_size=1000,
    n_workers=20,
    progress_max_items=10,
)
```

Set either value to `0` to disable the remote batch display. `quiet=True`
also suppresses it.

The latest immutable state is available as `provider.batch_progress`. Alternate
displays can implement `BatchProgressRenderer` and be supplied to an
`OpenAIBatchAPIHelper`; configure `progress_max_items` on that custom helper.

### Batch operation scheduling

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

The provider's `RetryPolicy` bounds upload, download, and polling retries.
Cooldowns honor `retry-after-ms` and `Retry-After`; otherwise they use capped
exponential backoff with jitter. A cooldown also delays the next queued job
after retry exhaustion. Creation always makes a single attempt to avoid
creating duplicate remote work after an ambiguous response.

When list polling exhausts its retry budget, the helper retrieves unresolved
batch IDs individually. Before cancelling after a polling failure, it makes one
final retrieval with the same retry policy, within a five-second cleanup window.
A terminal response is returned for result processing; an unresolved tracking
error retains the batch ID, last status, and original cause.

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

`send_batch()` borrows its stream until completion and rewinds to its initial
offset for every attempt. Cancelling it waits until execution releases the
stream. `send_batch()` and `create_batch_task()` forward additional SDK keyword
arguments. Status callback failures are logged without stopping polling.
Cancelling an individual `query_batch_mode()` caller abandons only that result;
cancelling a batch lifecycle cancels the remote batch. Polling deadlines attempt
bounded remote cancellation and recover available partial results. Closing the
provider rejects new work and settles helper work and progress rendering before
closing the SDK transport.

### Request budgets and elapsed deadlines

Providers can check the complete rendered request before a network submission,
including accumulated tool history, tool definitions, and output schemas:

```python
from agents import RequestBudget
from agents.providers import AzureOpenAIBatchProvider

budget = RequestBudget(
    model_name="my-deployment",
    max_input_tokens=272_000,
    output_reservation=32_768,
    safety_margin=8_000,
    provenance=("deployment owner's configured input cutoff",),
)
provider = AzureOpenAIBatchProvider(
    "my-deployment", request_budget=budget,
)
```

The default 272,000-token input cutoff is a configurable guess, not a statement
about a model's context window. `resolve_request_budget` accepts best-effort
metadata and explicit overrides without requiring discovery or interpreting
deployment names. Explicit overrides take precedence. Optional `context_window`
and `max_output_tokens` constraints remain separate from the input cutoff.
Output and additional reasoning reservations reduce available context capacity;
the larger of their sum and a request's output-token limit is reserved. A
completion-token limit already includes reasoning and is not counted twice.

Counts are estimates. To use `tiktoken`, explicitly load its encoding before
starting agents, for example `tiktoken.get_encoding("o200k_base")` at startup
(this can download assets). Preflight uses only encodings already loaded in the
process and never downloads tokenizer data. Otherwise conservative UTF-8 byte
estimates keep preflight available and may require smaller requests.
A custom `token_counter` supports another tokenizer. Estimates report separate
system, user, history, tools, schema, other, and framing components. Compare them
with returned input usage to assess estimator accuracy. Multimodal payloads need
a dedicated estimator and are rejected by the text estimator.

An oversized request raises terminal `RequestBudgetExceededError` with its
structured `estimate`, budget, and provenance. Applications can pass
`context_rebuild_hook=hook` to an `Agent`, or override its async
`rebuild_context(messages, error)` method. A hook receives `(agent, messages,
error)` and returns replacement messages or `None`; sync and async hooks work.
The provider checks the replacement once before submission. Evidence selection
and compression belong to the application. `tool_res_payload` and `scratchpad`
retain the execution audit, while `request_history` stores independent snapshots
of submitted contexts. Resetting an agent starts a new audit.

`Agent(..., elapsed_time_budget=seconds)` bounds the agent's own execution across
steps and continuation calls. Resetting starts a new elapsed budget. Terminal
callbacks run after this timer so they can persist the outcome; callback agents
can have independent budgets. A local `ExecutionDeadlineExceeded` cancels and
awaits only the current agent's pending work. It does not close a shared provider
or cancel an unrelated request's remote batch. The SDK's HTTP `timeout`, the
provider's `batch_polling_deadline`, and operation retry limits are separate
settings. A local deadline cannot prove remote execution stopped; an immediate
retry can duplicate remote work.

### Batch failures and retry ownership

The batch provider reads both output and error files, correlates rows by
`custom_id`, and retains successful responses even when siblings fail or the
batch expires. Nested Azure errors retain their original row in `error.details`;
the normalized error exposes `code`, `custom_id`, `batch_id`, and `status_code`.
Duplicate IDs, malformed responses, and missing rows produce terminal
`BatchProtocolError` for the affected request. Completed all-error batches are
valid batch responses and propagate their individual errors.

Pass `retry_policy=RetryPolicy(max_attempts=3, elapsed_time_budget=120)` from
`agents.retries` to configure bounded retries of failed operations. Context
overflow and invalid requests are terminal for unchanged inputs. Timeouts,
rate limits, and service failures can retry the individual model request or
file operation without replaying prior steps or callbacks. Batch creation is
attempted once because losing its response can leave remote work running. SDK
retries default to zero; retain that setting to keep a single retry budget.

Processors retain every spawned agent in input order, including during
cancellation, and expose terminal errors as `processor.failures[index]` and
`agent.last_error`. `n_retry` counts total allowed agent attempts, including the
initial attempt. Only an explicit application
`RetryableAgentError(..., operation_scoped=False)` permits continuation at the
agent level; provider failures are already handled at the operation level and
are never multiplied by processor retries. Untyped application errors terminate
the agent. Partial failures still produce compatibility placeholders; applications
must reconcile `failures` before declaring their workload successful.

Set `request_budget=budget` to enable preflight or
`require_request_budget=True` to use the configurable fallback cutoff. Legacy
providers without either option do not enforce a cutoff. A report deadline
abandons only its own future. A `batch_polling_deadline` is owned by the helper
for the entire submitted batch: it requests remote cancellation, preserves
available results, and reports `BatchPollingDeadlineExceeded` for unresolved
requests. Cancellation remains best effort and does not guarantee remote work
has stopped.

## Public Domain and CC0 Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

See [LICENSE](LICENSE) for the complete CC0 1.0 legal code. Source code incorporated
from other projects retains its original license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you agree that your contribution will be released under the
[CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
