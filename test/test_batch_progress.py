"""Tests for provider-neutral batch progress tracking and formatting."""

import warnings
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from agents.batch_progress import (
    BATCH_PROGRESS_MAX_ITEMS_ENV_VAR,
    DEFAULT_BATCH_PROGRESS_MAX_ITEMS,
    BatchProgressState,
    BatchRequestCounts,
    BatchSnapshot,
    BatchTracker,
    format_batch_progress,
    resolve_batch_progress_max_items,
)


START_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def make_batch(
    index: int,
    *,
    status: str = "in_progress",
    created_at: datetime = START_TIME,
    terminal: bool = False,
    request_counts: BatchRequestCounts | None = None,
) -> BatchSnapshot:
    """Create a predictable batch snapshot for tests."""

    return BatchSnapshot(
        id=f"id_{index:02}",
        name=f"batch_{index:02}",
        created_at=created_at,
        status=status,
        terminal=terminal,
        request_counts=request_counts,
    )


def test_batch_snapshot_and_request_counts_are_immutable():
    counts = BatchRequestCounts(total=10, completed=6, failed=1)
    batch = make_batch(1, request_counts=counts)

    with pytest.raises(FrozenInstanceError):
        batch.status = "completed"

    with pytest.raises(FrozenInstanceError):
        counts.completed = 7


def test_progress_state_has_immutable_default_finished_counts():
    state = BatchProgressState()

    assert dict(state.finished_counts) == {}
    with pytest.raises(TypeError):
        state.finished_counts["completed"] = 1


@pytest.mark.parametrize("field_name", ["total", "completed", "failed"])
def test_request_counts_reject_negative_values(field_name: str):
    values = {"total": 10, "completed": 5, "failed": 1}
    values[field_name] = -1

    with pytest.raises(ValueError, match=field_name):
        BatchRequestCounts(**values)


def test_tracker_updates_batches_without_duplicates():
    tracker = BatchTracker()
    validating = make_batch(1, status="validating")
    in_progress = replace(
        validating,
        status="in_progress",
        request_counts=BatchRequestCounts(total=100, completed=25),
    )

    tracker.update(validating)
    tracker.update(in_progress)

    state = tracker.snapshot()
    assert tracker.active_count == 1
    assert state.active == (in_progress,)


def test_tracker_orders_oldest_first_with_id_tiebreaker():
    tracker = BatchTracker()
    newer = make_batch(1, created_at=START_TIME + timedelta(minutes=1))
    tied_second = make_batch(3)
    oldest = make_batch(0, created_at=START_TIME - timedelta(minutes=1))
    tied_first = make_batch(2)

    for batch in (newer, tied_second, oldest, tied_first):
        tracker.update(batch)

    assert [batch.id for batch in tracker.snapshot().active] == [
        "id_00",
        "id_02",
        "id_03",
        "id_01",
    ]


def test_tracker_finishes_batches_and_counts_terminal_statuses_once():
    tracker = BatchTracker()
    active = make_batch(1)
    completed = replace(active, status="completed", terminal=True)
    failed = replace(make_batch(2), status="failed", terminal=True)

    tracker.update(active)
    tracker.update(completed)
    tracker.finish(completed)
    tracker.finish(failed)

    state = tracker.snapshot()
    assert tracker.active_count == 0
    assert tracker.finished_count == 2
    assert state.finished_count == 2
    assert dict(state.finished_counts) == {"completed": 1, "failed": 1}


def test_tracker_replaces_a_revised_terminal_status_without_double_counting():
    tracker = BatchTracker()
    failed = replace(make_batch(1), status="failed", terminal=True)
    completed = replace(failed, status="completed")

    tracker.finish(failed)
    tracker.finish(completed)

    assert dict(tracker.snapshot().finished_counts) == {"completed": 1}


def test_tracker_rejects_nonterminal_finish_and_updates_after_finish():
    tracker = BatchTracker()
    active = make_batch(1)

    with pytest.raises(ValueError, match="terminal=True"):
        tracker.finish(active)

    tracker.finish(replace(active, status="cancelled", terminal=True))
    with pytest.raises(ValueError, match="already finished"):
        tracker.update(active)


def test_format_zero_max_items_disables_all_output():
    state = BatchProgressState(
        active=(make_batch(1),), finished_counts={"completed": 3}
    )

    assert format_batch_progress(state, max_items=0) == ""


def test_format_one_item_and_aggregate_hidden_statuses():
    state = BatchProgressState(
        active=(
            make_batch(3, status="validating", created_at=START_TIME + timedelta(3)),
            make_batch(1, status="in_progress", created_at=START_TIME + timedelta(1)),
            make_batch(2, status="validating", created_at=START_TIME + timedelta(2)),
            make_batch(0, status="in_progress", created_at=START_TIME),
        )
    )

    output = format_batch_progress(state, max_items=1)

    assert output.splitlines() == [
        "Active Batches (4):",
        "├── batch_00 [2026-09-01 12:00:00+00:00]: in_progress",
        "└── … 3 hidden: in_progress=1, validating=2",
    ]


def test_format_ten_items_without_aggregation():
    state = BatchProgressState(
        active=tuple(
            make_batch(index, created_at=START_TIME + timedelta(minutes=index))
            for index in reversed(range(10))
        )
    )

    lines = format_batch_progress(state, max_items=10).splitlines()

    assert len(lines) == 11
    assert lines[1].startswith("├── batch_00")
    assert lines[-1].startswith("└── batch_09")
    assert "hidden" not in lines[-1]


def test_format_overflow_limits_individual_rows_and_includes_counts():
    batches = tuple(
        make_batch(
            index,
            status="finalizing" if index >= 10 else "in_progress",
            created_at=START_TIME + timedelta(minutes=index),
            request_counts=(
                BatchRequestCounts(total=100, completed=80, failed=2)
                if index == 0
                else None
            ),
        )
        for index in range(12)
    )
    state = BatchProgressState(
        active=batches,
        finished_counts={"failed": 1, "completed": 4},
    )

    lines = format_batch_progress(state, max_items=10).splitlines()

    assert len([line for line in lines if "[2026-" in line]) == 10
    assert (
        lines[1]
        == "├── batch_00 [2026-09-01 12:00:00+00:00]: in_progress"
        " · requests: 80/100 completed, 2 failed"
    )
    assert lines[-2] == "└── … 2 hidden: finalizing=2"
    assert lines[-1] == "Finished Batches (5): completed=4, failed=1"


def test_format_uses_unicode_tree_connectors():
    state = BatchProgressState(active=(make_batch(1), make_batch(2)))

    output = format_batch_progress(state)

    assert "├──" in output
    assert "└──" in output


def test_resolve_max_items_defaults_and_reads_environment():
    assert resolve_batch_progress_max_items(environ={}) == 10
    assert DEFAULT_BATCH_PROGRESS_MAX_ITEMS == 10
    assert (
        resolve_batch_progress_max_items(
            environ={BATCH_PROGRESS_MAX_ITEMS_ENV_VAR: "0"}
        )
        == 0
    )
    assert (
        resolve_batch_progress_max_items(
            environ={BATCH_PROGRESS_MAX_ITEMS_ENV_VAR: " 7 "}
        )
        == 7
    )


def test_explicit_max_items_takes_precedence_over_environment():
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        value = resolve_batch_progress_max_items(
            3, environ={BATCH_PROGRESS_MAX_ITEMS_ENV_VAR: "invalid"}
        )

    assert value == 3
    assert not warning_records


@pytest.mark.parametrize("value", ["invalid", "", "-1"])
def test_invalid_environment_values_warn_and_use_default(value: str):
    with pytest.warns(UserWarning, match=BATCH_PROGRESS_MAX_ITEMS_ENV_VAR):
        result = resolve_batch_progress_max_items(
            environ={BATCH_PROGRESS_MAX_ITEMS_ENV_VAR: value}
        )

    assert result == DEFAULT_BATCH_PROGRESS_MAX_ITEMS


def test_invalid_explicit_max_items_raise():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        resolve_batch_progress_max_items(-1)

    with pytest.raises(TypeError, match="integer"):
        resolve_batch_progress_max_items(False)

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        format_batch_progress(BatchProgressState(), max_items=-1)
