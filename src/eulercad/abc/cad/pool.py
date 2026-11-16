"""Generic multi-process worker pool: one process per item, a wall-clock timeout per item, and a per-item progress registry the parent can read while an item is still running."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# How often the scheduler checks on active workers (liveness + deadlines).
POLL_INTERVAL = 0.2

# Fixed-size shared buffer each item's registry row is backed by. A worker's
# stage hook writes into it, so the parent can read "stuck at: ..." for a
# process it just force-killed - the killed process is gone and can never
# answer any in-process query itself.
_STAGE_BUF_SIZE = 256


def _write_stage(buf, stage: str) -> None:
    data = stage.encode("utf-8", "replace")[: len(buf)]
    buf.raw = data.ljust(len(buf), b"\x00")


def _read_stage(buf) -> str:
    return buf.raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def meminfo_gb() -> dict[str, float]:
    """Selected /proc/meminfo fields, in GiB.

    Returns:
        `MemFree`, `MemAvailable` and `SwapUsed`; empty if /proc/meminfo is unreadable.
    """
    want = ("MemFree:", "MemAvailable:", "SwapTotal:", "SwapFree:")
    got: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            for key in want:
                if line.startswith(key):
                    got[key.rstrip(":")] = int(line.split()[1]) / 1024**2
    except OSError:
        return {}
    if "SwapTotal" in got and "SwapFree" in got:
        got["SwapUsed"] = got["SwapTotal"] - got["SwapFree"]
    return got


def memory_gate(min_free_gb: float, swap_growth_gb: float, window_s: float = 300.0) -> Callable[[int], bool]:
    """Build a `can_dispatch` predicate that keeps the host out of swap.

    **Swap growth is the primary signal** - it states the constraint directly, and it
    is the one that works. The secondary signal exists only to catch genuine memory
    exhaustion before swapping starts.

    That secondary signal must be `MemAvailable`, not `MemFree`. Both were tried:

    - `MemFree` was chosen first, after a run pushed 3.5 GB into swap while
      `MemAvailable` still read 10.8 GB. But `MemFree` is *meant* to be small on
      Linux - the kernel fills it with page cache - so over a long run it decays
      toward zero no matter how much memory is genuinely reclaimable. Measured on a
      9-hour projection run, `MemFree` fell to 544 MB against 19 GB of
      `MemAvailable`, and the gate throttled 12 workers down to 1 while swap sat
      *below* its starting value the whole time. That cost roughly a 3x slowdown.
    - `MemAvailable` counts reclaimable page cache, so it tracks real headroom. Its
      failure mode - overstating headroom when the kernel would rather swap than
      reclaim - is exactly what the swap-growth signal already catches.

    So: swap growth catches the case `MemAvailable` misses, and `MemAvailable` avoids
    the false positives `MemFree` produces. Keep the floor low; it is a backstop, not
    the throttle.

    Growth must be measured over a *sliding window*, not against a fixed baseline
    taken when the gate was built. `SwapUsed` only falls when swapped pages are
    faulted back in, which for long-idle pages never happens - so a fixed baseline
    makes the measure a ratchet that can only climb. Measured on a corpus run:
    unrelated idle processes drifted ~25 MB/min into swap, crossed a 0.5 GB
    threshold after 90 minutes, and throttled 14 workers to 3 while `MemAvailable`
    read 20.8 GB and load sat at 2.12. The window makes the signal bidirectional and
    states the real constraint - swap is filling *fast*, right now - which is what
    separates that harmless drift from the runaway it is meant to catch.

    Args:
        min_free_gb: hold dispatch below this much *available* memory.
        swap_growth_gb: hold dispatch once swap grows this much within `window_s`.
        window_s: length of the sliding window growth is measured over.

    Returns:
        A predicate taking the current active-worker count, for `run_pool`.
    """
    history: deque[tuple[float, float]] = deque()
    state = {"logged": ""}

    def gate(n_active: int) -> bool:
        info = meminfo_gb()
        if not info:
            return True
        now = time.monotonic()
        used = info.get("SwapUsed", 0.0)
        history.append((now, used))
        # Keep one sample older than the window, so growth is always measured across
        # its full span rather than across however much has elapsed so far.
        while len(history) > 2 and now - history[1][0] > window_s:
            history.popleft()
        grown = used - min(u for _, u in history)
        available = info.get("MemAvailable", float("inf"))
        reason = ""
        if grown > swap_growth_gb:
            reason = f"swap grew {grown:.1f} GB > {swap_growth_gb:.1f} GB in {window_s:.0f}s"
        elif available < min_free_gb:
            reason = f"{available:.1f} GB available < {min_free_gb:.1f} GB"
        if not reason:
            state["logged"] = ""
            return True
        if state["logged"] != reason:
            log.info("holding dispatch at %d workers: %s", n_active, reason)
            state["logged"] = reason
        return False

    return gate


def cap_address_space(gb: float) -> None:
    """Cap the calling process's address space, so a runaway allocation fails it alone.

    Must be called inside the worker, after the spawn: setting it in the parent would
    constrain the scheduler too. `run_pool` does this for you via `memory_cap_gb`.

    A soft dispatch gate cannot help once a worker is already running and allocating
    - measured, a model that passed a complexity check still allocated for 300 s
    inside the mesher - so this is the hard backstop behind it.
    """
    import resource

    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit if hard == resource.RLIM_INFINITY else min(limit, hard), hard))
    except (ValueError, OSError) as e:
        log.warning("could not cap worker address space at %.1f GB: %s", gb, e)


def stage_writer(buf) -> Callable[[str], None]:
    """Build a callback that writes into `buf` - e.g. `projection.set_stage_hook(stage_writer(buf))` in a worker process.

    Returns:
        Callable[[str], None]: a callback that writes its argument into `buf`.
    """
    return lambda stage: _write_stage(buf, stage)


class Registry:
    """Father-owned table of per-item progress rows.

    One row per in-flight item, keyed by an arbitrary integer id (e.g. an
    item's position in the dispatch list). Each row is its own shared-memory
    buffer: a spawned item's process is handed exactly that one buffer -
    never the registry itself - so it can write its own row and cannot see
    or touch any other item's. Rows are never reused across items: create()
    always allocates a fresh buffer, so there is no window where one id's
    row still carries content left over from a different, unrelated item.
    """

    def __init__(self, ctx: mp.context.BaseContext, buf_size: int = _STAGE_BUF_SIZE) -> None:
        self._ctx = ctx
        self._buf_size = buf_size
        self._rows: dict[int, Any] = {}

    def create(self, key: int):
        """Allocate a fresh row for `key`.

        Returns:
            The shared buffer to pass into that item's process - and nothing else.
        """
        buf = self._ctx.Array("c", self._buf_size, lock=False)
        self._rows[key] = buf
        return buf

    def read(self, key: int) -> str:
        """Read the last stage reported for `key`.

        Returns:
            str: the last stage written for `key`, or "" if nothing was written yet (or `key` is unknown).
        """
        buf = self._rows.get(key)
        if buf is None:
            return ""
        return _read_stage(buf)

    def discard(self, key: int) -> None:
        """Drop `key`'s row. Not required for correctness (rows are never reused) - just bounds memory over a long run."""
        self._rows.pop(key, None)


class _Worker:
    __slots__ = ("key", "process", "start_time")

    def __init__(self, process: mp.process.BaseProcess, key: int) -> None:
        self.process = process
        self.key = key
        self.start_time = time.monotonic()


def _dispatch_pending[T](
    ctx: mp.context.BaseContext,
    registry: Registry,
    pending: deque[tuple[int, T]],
    active: dict[int, _Worker],
    target: Callable[..., None],
    make_args: Callable[[int, T, Any], tuple],
    n_workers: int,
    can_dispatch: Callable[[int], bool] | None = None,
) -> None:
    """Spawn queued items into fresh processes until `active` reaches `n_workers`.

    `can_dispatch` gates each spawn on some external resource - memory headroom,
    typically. It is consulted per spawn and may stop the loop early, in which case
    the item stays queued and is retried on the next poll. It is never consulted
    when nothing is running, so a predicate that always refuses cannot wedge the
    pool - progress is guaranteed regardless of what the caller supplies.
    """
    while pending and len(active) < n_workers:
        if active and can_dispatch is not None and not can_dispatch(len(active)):
            return
        key, item = pending.popleft()
        buf = registry.create(key)
        # BaseContext's stub only declares Process on concrete subclasses
        # (SpawnContext, ForkContext, ...); get_context() has no literal-string
        # overload to narrow to one, so ty can't see it here even though every
        # context returned by get_context() has it at runtime.
        process = ctx.Process(target=target, args=make_args(key, item, buf))  # ty: ignore[unresolved-attribute]
        process.start()
        active[key] = _Worker(process, key)


def _poll_once[T](
    active: dict[int, _Worker],
    registry: Registry,
    items: list[T],
    timeout: float,
    is_success: Callable[[int, T], bool],
    label: Callable[[T], str],
) -> list[tuple[int, bool]]:
    """Check every active worker once: alive-and-within-budget, finished, or timed out.

    Returns:
        list[tuple[int, bool]]: (key, ok) for workers that are no longer running.
    """
    now = time.monotonic()
    finished: list[tuple[int, bool]] = []
    for key, worker in active.items():
        if worker.process.is_alive():
            if now - worker.start_time > timeout:
                stage = registry.read(key) or "unknown"
                worker.process.kill()
                worker.process.join()
                log.error(
                    "%s: exceeded %gs wall-clock budget, force-killed (stuck at: %s)",
                    label(items[key]),
                    timeout,
                    stage,
                )
                finished.append((key, False))
            continue

        worker.process.join()
        ok = worker.process.exitcode == 0 and is_success(key, items[key])
        if not ok:
            stage = registry.read(key) or "unknown"
            log.error(
                "%s: worker exited with code %s (stuck at: %s)",
                label(items[key]),
                worker.process.exitcode,
                stage,
            )
        finished.append((key, ok))
    return finished


def run_pool[T](
    items: list[T],
    target: Callable[..., None],
    make_args: Callable[[int, T, Any], tuple],
    *,
    n_workers: int,
    timeout: float,
    is_success: Callable[[int, T], bool] = lambda key, item: True,
    on_finish: Callable[[int, T, bool], None] | None = None,
    label: Callable[[T], str] = str,
    can_dispatch: Callable[[int], bool] | None = None,
) -> list[T]:
    """Run `target` once per item, each in its own process, up to `n_workers` at a time.

    Each item gets a fresh Registry row; `target` is expected to write its
    progress into the shared buffer `make_args` embeds in its args (see
    `stage_writer`). There is no retry: each process handles exactly one
    item, so a failure always points at that one item, not a batch, and an
    unattended re-run is unlikely to succeed where a fresh process just
    failed.

    Args:
        items: work items, in dispatch order. `key` (the int a row/worker is
            addressed by) is each item's position in this list.
        target: process entry point; called as `target(*make_args(key, item, buf))`.
        make_args: builds `target`'s args for one item, given its key, the
            item itself, and the shared stage buffer for that item's row.
        is_success: extra caller-defined check run after a process exits 0
            (e.g. "did it actually write its output"). Not run at all if the
            process crashed or timed out.
        on_finish: called once per item as soon as it's known to be done (ok
            or not) - e.g. to advance a progress bar or clean up partial output.
        label: human-readable name for an item, used in log messages only.
        can_dispatch: called with the current number of active workers before each
            spawn; returning False holds the item back until the next poll. Use it
            to gate on a resource `n_workers` cannot express - notably free memory,
            when per-item cost is wildly uneven and saturating the cores would swap.
            Never consulted when the pool is empty, so it cannot deadlock the pool.

    Returns:
        list[T]: items skipped because their process timed out, crashed, or failed `is_success`.
    """
    n_workers = max(1, min(n_workers, len(items)))
    log.info("Running %d item(s) with %d worker process(es)", len(items), n_workers)

    ctx = mp.get_context("spawn")
    registry = Registry(ctx)
    pending: deque[tuple[int, T]] = deque(enumerate(items))
    active: dict[int, _Worker] = {}
    skipped: list[T] = []

    try:
        while pending or active:
            _dispatch_pending(ctx, registry, pending, active, target, make_args, n_workers, can_dispatch)

            finished = _poll_once(active, registry, items, timeout, is_success, label)
            for key, ok in finished:
                active.pop(key)
                registry.discard(key)
                if not ok:
                    skipped.append(items[key])
                if on_finish is not None:
                    on_finish(key, items[key], ok)

            if not finished:
                time.sleep(POLL_INTERVAL)
    finally:
        # A hung worker never returns, so waiting on it would block forever - kill
        # whatever is still alive (covers KeyboardInterrupt / unexpected exceptions
        # in the scheduler loop itself).
        for worker in active.values():
            if worker.process.is_alive():
                worker.process.kill()
            worker.process.join()

    return skipped
