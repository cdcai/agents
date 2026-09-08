"""
Provider-neutral state and formatting for batch API progress.

Providers are responsible for translating their batch objects into
``BatchSnapshot`` instances and identifying terminal states. This module only
tracks those snapshots and renders their current state.
"""

import os
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol

import tqdm.asyncio as tqdm

DEFAULT_BATCH_PROGRESS_MAX_ITEMS = 10
BATCH_PROGRESS_MAX_ITEMS_ENV_VAR = "AGENTS_BATCH_PROGRESS_MAX_ITEMS"


@dataclass(frozen=True)
class BatchRequestCounts:
    """Request counts reported for a batch."""

    total: int
    completed: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        for field_name in ("total", "completed", "failed"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be greater than or equal to 0")


@dataclass(frozen=True)
class BatchSnapshot:
    """An immutable, provider-neutral view of a remote batch."""

    id: str
    name: str
    created_at: datetime
    status: str
    terminal: bool = False
    request_counts: BatchRequestCounts | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.status:
            raise ValueError("status must not be empty")


@dataclass(frozen=True)
class BatchProgressState:
    """An immutable snapshot of all progress currently known to a tracker."""

    active: tuple[BatchSnapshot, ...] = ()
    finished_counts: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", tuple(self.active))
        object.__setattr__(
            self,
            "finished_counts",
            MappingProxyType(dict(self.finished_counts)),
        )

    @property
    def active_count(self) -> int:
        """Number of active batches."""

        return len(self.active)

    @property
    def finished_count(self) -> int:
        """Number of finished batches."""

        return sum(self.finished_counts.values())


class BatchProgressRenderer(Protocol):
    """Render immutable snapshots of provider-neutral batch progress."""

    def refresh(self, state: BatchProgressState) -> None: ...

    def close(self) -> None: ...


class _TqdmRow(Protocol):
    """The subset of the tqdm API used by the progress renderer."""

    def set_description_str(self, desc: str, refresh: bool = True) -> None: ...

    def refresh(self) -> None: ...

    def close(self) -> None: ...


TqdmFactory = Callable[..., _TqdmRow]


class TqdmBatchProgressRenderer:
    """Render batch progress as reusable, text-only tqdm rows."""

    def __init__(
        self,
        max_items: int | None = None,
        *,
        disable: bool = False,
        tqdm_factory: TqdmFactory | None = None,
    ) -> None:
        self.max_items = resolve_batch_progress_max_items(max_items)
        self.disable = disable or self.max_items == 0
        self._tqdm_factory = tqdm.tqdm if tqdm_factory is None else tqdm_factory
        self._rows: list[_TqdmRow] = []
        self._closed = False

    def refresh(self, state: BatchProgressState) -> None:
        """Update the live rows to reflect a batch progress snapshot."""

        if self.disable or self._closed:
            return

        rendered = format_batch_progress(state, max_items=self.max_items)
        lines = rendered.splitlines()

        while len(self._rows) > len(lines):
            self._rows.pop().close()

        existing_row_count = len(self._rows)
        for index, line in enumerate(lines):
            if index < existing_row_count:
                self._rows[index].set_description_str(line, refresh=False)
                continue

            self._rows.append(
                self._tqdm_factory(
                    desc=line,
                    bar_format="{desc}",
                    leave=False,
                )
            )

        for row in self._rows:
            row.refresh()

    def close(self) -> None:
        """Close all live rows from bottom to top."""

        if self._closed:
            return

        self._closed = True
        rows, self._rows = self._rows, []
        for row in reversed(rows):
            row.close()


class BatchTracker:
    """Track the latest active and terminal snapshots for remote batches."""

    def __init__(self) -> None:
        self._active: dict[str, BatchSnapshot] = {}
        self._finished: dict[str, BatchSnapshot] = {}
        self._finished_counts: Counter[str] = Counter()

    @property
    def active_count(self) -> int:
        """Number of active batches."""

        return len(self._active)

    @property
    def finished_count(self) -> int:
        """Number of finished batches."""

        return len(self._finished)

    def update(self, batch: BatchSnapshot) -> None:
        """
        Store the latest snapshot for a batch.

        Terminal snapshots are passed to ``finish`` so callers may use the same
        method for every status returned by a provider.
        """

        if batch.terminal:
            self.finish(batch)
            return

        if batch.id in self._finished:
            raise ValueError(f"Batch {batch.id!r} has already finished")

        self._active[batch.id] = batch

    def finish(self, batch: BatchSnapshot) -> None:
        """Move a terminal batch out of the active set and count its status."""

        if not batch.terminal:
            raise ValueError("A finished batch must have terminal=True")

        self._active.pop(batch.id, None)

        previous = self._finished.get(batch.id)
        if previous is not None:
            if previous.status == batch.status:
                self._finished[batch.id] = batch
                return

            self._finished_counts[previous.status] -= 1
            if self._finished_counts[previous.status] == 0:
                del self._finished_counts[previous.status]

        self._finished[batch.id] = batch
        self._finished_counts[batch.status] += 1

    def snapshot(self) -> BatchProgressState:
        """Return an immutable, deterministic snapshot of the tracker."""

        return BatchProgressState(
            active=tuple(sorted(self._active.values(), key=_batch_sort_key)),
            finished_counts=dict(sorted(self._finished_counts.items())),
        )


def resolve_batch_progress_max_items(
    max_items: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """
    Resolve the number of individual active batches to display.

    An explicit argument takes precedence over
    ``AGENTS_BATCH_PROGRESS_MAX_ITEMS``. Invalid environment values issue a
    warning and use the default, while invalid explicit values raise.
    """

    if max_items is not None:
        return _validate_max_items(max_items)

    if environ is None:
        environ = os.environ

    raw_value = environ.get(BATCH_PROGRESS_MAX_ITEMS_ENV_VAR)
    if raw_value is None:
        return DEFAULT_BATCH_PROGRESS_MAX_ITEMS

    try:
        env_max_items = int(raw_value)
        return _validate_max_items(env_max_items)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid {BATCH_PROGRESS_MAX_ITEMS_ENV_VAR} value {raw_value!r}; "
            f"using default {DEFAULT_BATCH_PROGRESS_MAX_ITEMS}",
            UserWarning,
            stacklevel=2,
        )
        return DEFAULT_BATCH_PROGRESS_MAX_ITEMS


def format_batch_progress(
    state: BatchProgressState,
    max_items: int = DEFAULT_BATCH_PROGRESS_MAX_ITEMS,
) -> str:
    """Format progress as a deterministic, file-tree-style string."""

    max_items = _validate_max_items(max_items)
    if max_items == 0:
        return ""

    active = sorted(state.active, key=_batch_sort_key)
    visible = active[:max_items]
    hidden = active[max_items:]

    lines = [f"Active Batches ({len(active)}):"]
    active_lines = [_format_batch(batch) for batch in visible]
    if hidden:
        status_counts = Counter(batch.status for batch in hidden)
        statuses = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        )
        active_lines.append(f"… {len(hidden)} hidden: {statuses}")

    for index, line in enumerate(active_lines):
        connector = "└──" if index == len(active_lines) - 1 else "├──"
        lines.append(f"{connector} {line}")

    if state.finished_counts:
        statuses = ", ".join(
            f"{status}={count}"
            for status, count in sorted(state.finished_counts.items())
        )
        lines.append(f"Finished Batches ({state.finished_count}): {statuses}")

    return "\n".join(lines)


def _validate_max_items(max_items: int) -> int:
    if isinstance(max_items, bool) or not isinstance(max_items, int):
        raise TypeError("max_items must be an integer")
    if max_items < 0:
        raise ValueError("max_items must be greater than or equal to 0")
    return max_items


def _batch_sort_key(batch: BatchSnapshot):
    created_at = batch.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)

    return created_at, batch.id


def _format_batch(batch: BatchSnapshot) -> str:
    timestamp = batch.created_at.isoformat(sep=" ", timespec="seconds")
    line = f"{batch.name} [{timestamp}]: {batch.status}"

    if batch.request_counts is not None:
        request_counts = batch.request_counts
        line += (
            f" · requests: {request_counts.completed}/{request_counts.total} completed"
        )
        if request_counts.failed:
            line += f", {request_counts.failed} failed"

    return line


__all__ = [
    "BATCH_PROGRESS_MAX_ITEMS_ENV_VAR",
    "DEFAULT_BATCH_PROGRESS_MAX_ITEMS",
    "BatchProgressRenderer",
    "BatchProgressState",
    "BatchRequestCounts",
    "BatchSnapshot",
    "BatchTracker",
    "TqdmBatchProgressRenderer",
    "format_batch_progress",
    "resolve_batch_progress_max_items",
]
