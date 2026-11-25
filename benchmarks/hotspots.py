"""Measure compatible-shape scheduling and kernel hot paths with tensor verification."""

import argparse
from collections import Counter
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from batchsail.batcher import BatchingPolicy
from batchsail.codec import digest, encode
from batchsail.config import RuntimeConfig
from batchsail.device import DeviceSpec
from batchsail.request import Request, RequestKey
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime
from batchsail.tensor import Tensor, equivalent

ROOT = Path(__file__).resolve().parents[1]


def reference(model, tensor):
    rows = [tensor.row(index) for index in range(tensor.rows)]
    if model == "affine":
        values = [item for x, y, z in rows for item in (0.5 + 2*x - y + 0.5*z, -2 + 3*y + z)]
    else:
        previous = (0.0, 0.0)
        before_previous = (0.0, 0.0)
        values = []
        for (x, y) in rows:
            a, b = previous
            c, d = before_previous
            values.extend((1 + x + .5*y + .25*a - .5*b + .5*c, -1 - x + 2*y + b - .25*c + .25*d))
            previous, before_previous = (x, y), previous
    return Tensor((tensor.rows, 2), tuple(values))


def workload(scenario, count, seed):
    rng = Random(seed)
    for index in range(count):
        model = "affine" if index % 3 else "filter"
        columns = 3 if model == "affine" else 2
        if scenario == "heterogeneous":
            rows = rng.choice((1, 3, 4, 5, 8, 12, 17, 25, 33, 48, 63, 96, 128))
        elif scenario == "urgent":
            rows = rng.choice((4, 8, 64, 96))
        else:
            rows = rng.choice((4, 8, 16))
        is_urgent = scenario == "urgent" and index % 9 == 0
        budget = (25, 125, 350)[index // 9 % 3] if is_urgent else None
        tenant = "urgent" if is_urgent else f"tenant-{index % 7}"
        values = tuple(rng.randrange(-8, 9) / 4 for _ in range(rows * columns))
        yield Request(RequestKey(tenant, f"request-{index}"), model, Tensor((rows, columns), values), budget)


def latency(values):
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(fraction):
        index = min(len(ordered) - 1, int(len(ordered) * fraction))
        return ordered[index] / 1_000_000

    return {f"p{key}_ms": percentile(key / 100) for key in (50, 95, 99)}


def trial(workspace, scenario, requests, slots):
    config = RuntimeConfig(
        backend="process",
        batching=BatchingPolicy(max_slots=slots, wait_ms=4 if slots > 1 else 0, max_pending=len(requests) + 4),
        devices=(DeviceSpec("cpu0", 65_536), DeviceSpec("cpu1", 65_536)),
    )
    references = {item.key: reference(item.model, item.tensor) for item in requests}
    with Runtime(workspace, config) as runtime:
        for name in ("affine", "filter"):
            revision = runtime.install(ROOT / "fixtures" / "models" / f"{name}-r1")
            runtime.activate(revision.key)
        for model in ("affine", "filter"):
            width = 3 if model == "affine" else 2
            warm = Request(RequestKey("benchmark", f"warm-{model}"), model, Tensor((1, width), (1,) * width))
            runtime.submit(warm)
            if runtime.wait(warm.key, timeout=5).status != ResultStatus.SUCCEEDED:
                raise RuntimeError("could not warm runtime")
        begin = perf_counter_ns()
        for item in requests:
            runtime.submit(item)
        admitted = perf_counter_ns()
        results = [runtime.wait(item.key, timeout=10) for item in requests]
        completed = perf_counter_ns()

        tallies = Counter()
        outputs = {}
        for item, result in zip(requests, results, strict=True):
            tallies[result.status.value] += 1
            if result.status == ResultStatus.EXPIRED and item.deadline_ms is not None:
                continue
            if (
                result.status != ResultStatus.SUCCEEDED
                or result.key != item.key
                or result.model.name != item.model
                or result.model.revision != "r1"
                or not equivalent(result.output, references[item.key])
            ):
                raise RuntimeError(f"unexpected result for {item.key}: {result.to_json()}")
            outputs[item.key.request_id] = result.output.to_json()

        total = (completed - begin) / 1e9
        batches = Counter((result.device, result.batch_id) for result in results if result.batch_id is not None)
        urgent = [result for result in results if result.key.tenant == "urgent"]
        snapshot = runtime.snapshot().to_json()
        if any(device["workspace_bytes"] for device in snapshot["devices"]):
            raise RuntimeError("benchmark left an outstanding workspace")
        return {
            "scenario": scenario,
            "max_slots": slots,
            "requests": len(requests),
            "terminal": dict(tallies),
            "checked_outputs": len(outputs),
            "output_sha256": digest(encode(outputs)),
            "seconds": total,
            "admission_ms": (admitted - begin) / 1e6,
            "requests_per_second": len(requests) / total,
            "batches_with_receipts": len(batches),
            "slot_histogram": dict(Counter(batches.values())),
            "latency": latency([result.elapsed_ns for result in results]),
            "queue": latency([result.queue_ns for result in results if result.queue_ns is not None]),
            "urgent": {
                "total": len(urgent), "expired": sum(result.status == ResultStatus.EXPIRED for result in urgent),
                "latency": latency([result.elapsed_ns for result in urgent]),
            },
            "scheduler_counters": snapshot["metrics"]["counts"],
            "worker_phases_ns": {
                name: {key: phase[key] for key in ("count", "sum_ns", "max_ns")}
                for name, phase in snapshot["metrics"]["worker_ns"].items()
            },
        }, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=360)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not 12 <= args.requests <= 2000:
        parser.error("--requests must be between 12 and 2000")
    for scenario in ("heterogeneous", "urgent", "long-queue"):
        requests = tuple(workload(scenario, args.requests, args.seed))
        with TemporaryDirectory(prefix="batchsail-bench-") as directory:
            root = Path(directory)
            serial, singleton_outputs = trial(root / "solo", scenario, requests, 1)
            grouped, batched_outputs = trial(root / "batch", scenario, requests, 16)
        common = singleton_outputs.keys() & batched_outputs.keys()
        if not common or any(singleton_outputs[key] != batched_outputs[key] for key in common):
            raise RuntimeError("singleton and grouped executions disagree on common successful outputs")
        print(encode({
            "seed": args.seed, "singleton": serial, "dynamic": grouped,
            "equivalence": {"common_successful_outputs": len(common), "all_equal": True},
        }).decode(), flush=True)


if __name__ == "__main__":
    main()
