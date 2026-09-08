"""Tests for the tqdm-backed batch progress renderer."""

from datetime import datetime, timezone
from typing import Any

from agents.batch_progress import (
    BatchProgressState,
    BatchSnapshot,
    TqdmBatchProgressRenderer,
)

START_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def make_batch(index: int, status: str = "in_progress") -> BatchSnapshot:
    """Create a predictable active batch snapshot for renderer tests."""

    return BatchSnapshot(
        id=f"id_{index:02}",
        name=f"batch_{index:02}",
        created_at=START_TIME,
        status=status,
    )


class FakeTqdmRow:
    """Record the tqdm operations performed on one rendered row."""

    def __init__(self, factory: "FakeTqdmFactory", kwargs: dict[str, Any]):
        self.factory = factory
        self.kwargs = kwargs
        self.creation_index = len(factory.rows)
        self.description = kwargs["desc"]
        self.description_updates: list[tuple[str, bool]] = []
        self.refresh_count = 0
        self.close_count = 0

    def set_description_str(self, desc: str, refresh: bool = True):
        self.description = desc
        self.description_updates.append((desc, refresh))

    def refresh(self):
        self.refresh_count += 1

    def close(self):
        self.close_count += 1
        self.factory.close_order.append(self.creation_index)


class FakeTqdmFactory:
    """Create fake rows without writing to a terminal."""

    def __init__(self):
        self.rows: list[FakeTqdmRow] = []
        self.close_order: list[int] = []

    def __call__(self, **kwargs: Any) -> FakeTqdmRow:
        row = FakeTqdmRow(self, kwargs)
        self.rows.append(row)
        return row


def test_renderer_lazily_creates_text_only_rows():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(max_items=10, tqdm_factory=factory)

    assert factory.rows == []

    renderer.refresh(BatchProgressState(active=(make_batch(1),)))

    assert len(factory.rows) == 2
    assert all("position" not in row.kwargs for row in factory.rows)
    assert all(row.kwargs["bar_format"] == "{desc}" for row in factory.rows)
    assert all(row.kwargs["leave"] is False for row in factory.rows)
    assert factory.rows[0].description == "Active Batches (1):"
    assert factory.rows[1].description.startswith("└── batch_01")


def test_renderer_reuses_rows_when_descriptions_change():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(max_items=10, tqdm_factory=factory)
    renderer.refresh(BatchProgressState(active=(make_batch(1),)))
    original_rows = tuple(factory.rows)

    renderer.refresh(BatchProgressState(active=(make_batch(1, status="finalizing"),)))

    assert tuple(factory.rows) == original_rows
    assert len(factory.rows) == 2
    assert factory.rows[0].description_updates == [("Active Batches (1):", False)]
    assert factory.rows[1].description_updates[0][0].endswith(": finalizing")
    assert factory.rows[1].description_updates[0][1] is False
    assert all(row.refresh_count == 2 for row in factory.rows)


def test_renderer_adds_only_new_rows_when_output_grows():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(max_items=10, tqdm_factory=factory)
    renderer.refresh(BatchProgressState(active=(make_batch(1),)))
    original_rows = tuple(factory.rows)

    renderer.refresh(BatchProgressState(active=(make_batch(1), make_batch(2))))

    assert len(factory.rows) == 3
    assert tuple(factory.rows[:2]) == original_rows
    assert factory.rows[2].creation_index == 2
    assert factory.close_order == []


def test_renderer_removes_rows_from_bottom_when_output_shrinks():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(max_items=10, tqdm_factory=factory)
    renderer.refresh(
        BatchProgressState(active=(make_batch(1), make_batch(2), make_batch(3)))
    )

    renderer.refresh(BatchProgressState())

    assert factory.close_order == [3, 2, 1]
    assert factory.rows[0].close_count == 0
    assert factory.rows[0].description == "Active Batches (0):"


def test_renderer_close_is_reverse_ordered_and_idempotent():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(max_items=10, tqdm_factory=factory)
    renderer.refresh(BatchProgressState(active=(make_batch(1), make_batch(2))))

    renderer.close()
    renderer.close()
    renderer.refresh(BatchProgressState(active=(make_batch(3),)))

    assert factory.close_order == [2, 1, 0]
    assert all(row.close_count == 1 for row in factory.rows)
    assert len(factory.rows) == 3


def test_renderer_max_items_zero_never_creates_rows():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(max_items=0, tqdm_factory=factory)

    renderer.refresh(BatchProgressState(active=(make_batch(1),)))
    renderer.close()

    assert factory.rows == []


def test_disabled_renderer_never_creates_rows():
    factory = FakeTqdmFactory()
    renderer = TqdmBatchProgressRenderer(
        max_items=10,
        disable=True,
        tqdm_factory=factory,
    )

    renderer.refresh(BatchProgressState(active=(make_batch(1),)))
    renderer.close()

    assert factory.rows == []
