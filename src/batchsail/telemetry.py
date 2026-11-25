"""Bounded scheduling events and latency distributions."""

from bisect import bisect_left
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from .clock import NS_PER_MS
from .executor.backend import KernelTimings
from .results.model import ResultEnvelope

BOUNDARIES_NS = (1_000, 10_000, 100_000, 500_000, *(ms * NS_PER_MS for ms in (
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 30_000,
)))


class Histogram:
    def __init__(self) -> None:
        self.buckets = [0] * (len(BOUNDARIES_NS) + 1)
        self.count = 0
        self.sum_ns = 0
        self.max_ns = 0

    def add(self, value_ns: int) -> None:
        self.buckets[bisect_left(BOUNDARIES_NS, value_ns)] += 1
        self.count += 1
        self.sum_ns += value_ns
        self.max_ns = max(self.max_ns, value_ns)

    def snapshot(self) -> dict[str, Any]:
        cumulative = 0
        bins = []
        for boundary, amount in zip((*BOUNDARIES_NS, None), self.buckets, strict=True):
            cumulative += amount
            bins.append({"le_ns": boundary, "count": cumulative})
        return {"count": self.count, "sum_ns": self.sum_ns, "max_ns": self.max_ns, "buckets": bins}


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    kind: str
    wall_ms: int
    details: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "wall_ms": self.wall_ms, **self.details}


class Telemetry:
    """Calls from the runtime are serialized by the coordinator."""

    def __init__(self, capacity: int):
        self._events: deque[Event] = deque(maxlen=capacity)
        self._event_sequence = 0
        self._counts: Counter[str] = Counter()
        self._sizes: Counter[int] = Counter()
        self._terminal: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        self._elapsed = Histogram()
        self._queued = Histogram()
        self._service = Histogram()
        self._worker = {
            "dispatch_round_trip": Histogram(),
            "weight_load": Histogram(),
            "input_pack": Histogram(),
            "compute": Histogram(),
        }
        self._kernel_cache = Counter[str]()

    def note(self, kind: str, wall_ms: int, **details: Any) -> None:
        self._event_sequence += 1
        self._counts[kind] += 1
        self._events.append(Event(self._event_sequence, kind, wall_ms, details))

    def dispatch(self, wall_ms: int, *, size: int, **details: Any) -> None:
        self._sizes[size] += 1
        self.note("batch_dispatched", wall_ms, size=size, **details)

    def settlement(self, result: ResultEnvelope, wall_ms: int) -> None:
        self._terminal[result.status.value] += 1
        if result.failure is not None:
            self._failures[result.failure.code.value] += 1
        if result.elapsed_ns is not None:
            self._elapsed.add(result.elapsed_ns)
            if result.queue_ns is not None:
                self._queued.add(result.queue_ns)
                self._service.add(result.elapsed_ns - result.queue_ns)
        self.note(
            "request_settled", wall_ms,
            **result.key.to_json(), **result.model.to_json(), attempt=result.attempt,
            status=result.status.value, batch_id=result.batch_id, slot=result.slot, device=result.device,
            failure=None if result.failure is None else result.failure.code.value,
        )

    def worker(self, backend_ns: int | None, timings: KernelTimings | None) -> None:
        if backend_ns is not None:
            self._worker["dispatch_round_trip"].add(backend_ns)
        if timings is not None:
            self._worker["weight_load"].add(timings.load_ns)
            self._worker["input_pack"].add(timings.pack_ns)
            self._worker["compute"].add(timings.compute_ns)
            self._kernel_cache["hit" if timings.cache_hit else "miss"] += 1

    def page(self, *, after: int, limit: int) -> dict[str, Any]:
        lower = self._events[0].id if self._events else 1
        matches = [event.to_json() for event in self._events if event.id > after][:limit]
        return {
            "events": matches,
            "dropped_before": lower,
            "latest": self._event_sequence,
            "next": matches[-1]["id"] if matches else after,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self._counts.items())),
            "batch_sizes": dict(sorted(self._sizes.items())),
            "terminal": dict(sorted(self._terminal.items())),
            "failures": dict(sorted(self._failures.items())),
            "latency_ns": {
                "elapsed": self._elapsed.snapshot(),
                "queue": self._queued.snapshot(),
                "post_dispatch": self._service.snapshot(),
            },
            "worker_ns": {name: histogram.snapshot() for name, histogram in self._worker.items()},
            "kernel_cache": dict(sorted(self._kernel_cache.items())),
        }
