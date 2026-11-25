"""Summarize settled workload results without retaining request tensors."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from typing import Any

from ..codec import encode
from ..request.model import RequestKey
from .model import ResultEnvelope, ResultStatus


@dataclass(frozen=True, slots=True)
class LatencySummary:
    samples: int
    minimum_ns: int | None
    mean_ns: float | None
    p50_ns: int | None
    p95_ns: int | None
    p99_ns: int | None
    maximum_ns: int | None

    @classmethod
    def from_samples(cls, sample: list[int]) -> "LatencySummary":
        if not sample:
            return cls(0, None, None, None, None, None, None)
        values = sorted(sample)
        count = len(values)
        return cls(
            count, values[0], sum(values) / count,
            values[ceil(0.50 * count) - 1], values[ceil(0.95 * count) - 1],
            values[ceil(0.99 * count) - 1], values[-1],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "minimum_ns": self.minimum_ns,
            "mean_ns": self.mean_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "maximum_ns": self.maximum_ns,
        }


class ResultStatistics:
    def __init__(self) -> None:
        self._count = Counter[str]()
        self._failure = Counter[str]()
        self._attempts = Counter[int]()
        self._devices = Counter[str]()
        self._batches = Counter[tuple[str, str]]()
        self._elapsed: list[int] = []
        self._queue: list[int] = []
        self._dispatched: list[int] = []

    def add(self, result: ResultEnvelope) -> None:
        self._count[result.status.value] += 1
        if result.failure is not None:
            self._failure[result.failure.code.value] += 1
        if result.elapsed_ns is not None:
            self._elapsed.append(result.elapsed_ns)
        if result.attempt:
            self._attempts[result.attempt] += 1
            self._devices[result.device] += 1
            self._batches[(result.device, result.batch_id)] += 1
        if result.queue_ns is not None and result.elapsed_ns is not None:
            self._queue.append(result.queue_ns)
            self._dispatched.append(result.elapsed_ns - result.queue_ns)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": dict(sorted(self._count.items())),
            "failure": dict(sorted(self._failure.items())),
            "attempts": dict(sorted(self._attempts.items())),
            "devices": dict(sorted(self._devices.items())),
            "batches": len(self._batches),
            "slots_per_observed_batch": dict(sorted(Counter(self._batches.values()).items())),
            "latency": {
                "elapsed": LatencySummary.from_samples(self._elapsed).to_json(),
                "queue": LatencySummary.from_samples(self._queue).to_json(),
                "after_dispatch": LatencySummary.from_samples(self._dispatched).to_json(),
            },
        }


class WorkloadStatistics:
    def __init__(self) -> None:
        self._aggregate = ResultStatistics()
        self._by_model: dict[str, ResultStatistics] = defaultdict(ResultStatistics)
        self._by_tenant: dict[str, ResultStatistics] = defaultdict(ResultStatistics)
        self._outputs: dict[RequestKey, str] = {}

    def add(self, result: ResultEnvelope) -> None:
        self._aggregate.add(result)
        self._by_model[result.model.label].add(result)
        self._by_tenant[result.key.tenant].add(result)
        if result.status == ResultStatus.SUCCEEDED and result.output is not None:
            self._outputs[result.key] = sha256(encode(result.output.to_json())).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        fingerprints = sha256()
        for identity, value in sorted(self._outputs.items()):
            record = {**identity.to_json(), "output_sha256": value}
            fingerprints.update(encode(record) + b"\n")
        return {
            "aggregate": self._aggregate.to_json(),
            "by_model": {label: stats.to_json() for label, stats in sorted(self._by_model.items())},
            "by_tenant": {label: stats.to_json() for label, stats in sorted(self._by_tenant.items())},
            "output_fingerprints": len(self._outputs),
            "output_sha256": fingerprints.hexdigest(),
        }
