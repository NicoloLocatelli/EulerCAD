"""Tests for `memory_gate`: what must trip it, and what must not.

Run with `uv run pytest tests/test_memory_gate.py`.

The gate reads /proc/meminfo through `pool.meminfo_gb`, so each test swaps that
name for a stub. Everything here runs in the parent process - no spawn - so
patching the module global does reach the code under test.
"""

from __future__ import annotations

import pytest

from eulercad.abc.cad import pool


class FakeHost:
    """A /proc/meminfo stand-in with a clock the test drives by hand."""

    def __init__(
        self,
        swap_used: float = 0.0,
        available: float = 20.0,
        readable: bool = True,
    ) -> None:
        self.swap_used = swap_used
        self.available = available
        self.readable = readable
        self.now = 0.0

    def _meminfo(self) -> dict[str, float]:
        if not self.readable:
            return {}
        return {
            "SwapUsed": self.swap_used,
            "MemAvailable": self.available,
            "MemFree": 1.0,
        }

    def install(self) -> None:
        # Deliberate monkeypatch: `pool` reads the host through these two module
        # globals, so replacing them is what lets the test drive memory and time.
        pool.meminfo_gb = self._meminfo  # ty: ignore[invalid-assignment]
        pool.time.monotonic = lambda: self.now  # ty: ignore[invalid-assignment]

    def tick(self, seconds: float, swap_delta: float = 0.0) -> None:
        self.now += seconds
        self.swap_used += swap_delta


@pytest.fixture(autouse=True)
def _real_host_afterwards():
    """Put back the real host after each test.

    `install` replaces `time.monotonic` for the whole process, not only for `pool`. Left frozen,
    every later wait in the test session (a process pool's, for one) never ends.

    Yields:
        Nothing; the test runs in between.
    """
    real_meminfo, real_monotonic = pool.meminfo_gb, pool.time.monotonic
    yield
    pool.meminfo_gb = real_meminfo
    pool.time.monotonic = real_monotonic


def test_slow_drift_does_not_trip() -> None:
    """25 MB/min of unrelated drift must never throttle, however long it runs.

    This is the corpus-run regression: with a fixed baseline the same drift
    crossed 0.5 GB after 90 minutes and cut 14 workers to 3.
    """
    host = FakeHost()
    host.install()
    gate = pool.memory_gate(min_free_gb=2.0, swap_growth_gb=0.5, window_s=300.0)
    for _ in range(240):  # 4 hours at one sample per minute
        assert gate(12), f"held at t={host.now}s, swap={host.swap_used:.2f} GB"
        host.tick(60.0, swap_delta=0.025)
    assert host.swap_used > 5.0, "drift should have accumulated well past 0.5 GB"


def test_fast_growth_trips() -> None:
    """1 GB/min - the runaway the gate exists to catch - must hold dispatch."""
    host = FakeHost()
    host.install()
    gate = pool.memory_gate(min_free_gb=2.0, swap_growth_gb=0.5, window_s=300.0)
    assert gate(12)
    host.tick(60.0, swap_delta=1.0)
    assert not gate(12), "1 GB in 60 s must exceed a 0.5 GB threshold"


def test_gate_recovers_after_growth_stops() -> None:
    """Once the burst ages out of the window, dispatch must resume.

    The fixed-baseline version could never do this - that was the bug.
    """
    host = FakeHost()
    host.install()
    gate = pool.memory_gate(min_free_gb=2.0, swap_growth_gb=0.5, window_s=300.0)
    gate(12)
    host.tick(60.0, swap_delta=1.0)
    assert not gate(12)
    for _ in range(6):  # idle past the window
        host.tick(60.0)
        gate(12)
    assert gate(12), "burst aged out of the window; gate should reopen"


def test_available_floor_still_holds() -> None:
    """The secondary signal must still catch genuine exhaustion."""
    host = FakeHost(available=1.0)
    host.install()
    gate = pool.memory_gate(min_free_gb=2.0, swap_growth_gb=0.5, window_s=300.0)
    assert not gate(12), "1 GB available is below the 2 GB floor"


def test_unreadable_meminfo_dispatches() -> None:
    """No signal must mean no throttle - never deadlock the pool on a bad read."""
    FakeHost(readable=False).install()
    gate = pool.memory_gate(min_free_gb=2.0, swap_growth_gb=0.5)
    assert gate(12)
