"""Repeatable interactive, roll, memory and recovery walkthrough."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from batchsail.batcher import BatchingPolicy
from batchsail.clock import ManualClock
from batchsail.codec import decode, encode
from batchsail.config import RuntimeConfig
from batchsail.device import DeviceSpec
from batchsail.executor import Boundary, ProcessBackend
from batchsail.registry import ModelKey
from batchsail.request import Request, RequestKey
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime
from batchsail.tensor import Tensor

ROOT = Path(__file__).resolve().parents[1]


class Gates:
    def __init__(self):
        self.arrived = {name: Event() for name in ("old", "drop")}
        self.release = {name: Event() for name in self.arrived}
        self.devices = {}

    def __call__(self, event):
        if event.boundary != Boundary.ACQUIRED:
            return
        for slot in event.batch.slots:
            name = slot.key.request_id
            if name in self.arrived and not self.arrived[name].is_set():
                self.devices[name] = event.lease.device
                self.arrived[name].set()
                if not self.release[name].wait(10):
                    raise RuntimeError(f"demo did not release {name}")

    def wait_for(self, name):
        if not self.arrived[name].wait(5):
            raise RuntimeError(f"demo did not reach {name}")
        return self.devices[name]

    def unblock_all(self):
        for event in self.release.values():
            event.set()


def check(runtime, request, expected, revision):
    result = runtime.wait(request.key, timeout=8)
    if result.status != ResultStatus.SUCCEEDED or result.model.revision != revision:
        raise RuntimeError(f"unexpected result: {result.to_json()}")
    if result.output.values != expected:
        raise RuntimeError(f"unexpected values for {request.key.request_id}: {result.to_json()}")
    return result


def tick(runtime):
    runtime.clock.advance_ms(20)
    runtime.pulse()


def run(workspace):
    config = RuntimeConfig(
        batching=BatchingPolicy(max_slots=4, wait_ms=20),
        devices=(DeviceSpec("cpu0", 6200), DeviceSpec("cpu1", 6200)),
    )
    clock = ManualClock(wall_ms=1_700_000_000_000)
    gates = Gates()
    backend = ProcessBackend(("cpu0", "cpu1"))
    source = Request.from_json(decode((ROOT / "fixtures/request.json").read_bytes()))
    affine_r1 = (2.0, 7.0, -0.5, 0.0)
    affine_r2 = (8.0, -1.25, 7.0, 0.5)
    with Runtime(workspace, config, clock=clock, backend=backend, hook=gates) as runtime:
        try:
            for name in ("affine-r1", "filter-r1", "affine-r2"):
                loaded = runtime.install(ROOT / "fixtures/models" / name)
                if loaded.key.revision == "r1":
                    runtime.activate(loaded.key)

            other = Request(RequestKey("other", "short"), "affine", Tensor((1, 3), (0, 1, 2)), 500)
            for item in (source, other):
                runtime.submit(item)
            tick(runtime)
            first = check(runtime, source, affine_r1, "r1")
            second = check(runtime, other, (0.5, 3.0), "r1")
            if first.batch_id != second.batch_id or first.slot == second.slot:
                raise RuntimeError("compatible inputs were not packed into distinct slots of the same batch")

            old = replace(source, key=RequestKey("sample", "old"))
            runtime.submit(old)
            tick(runtime)
            gates.wait_for("old")
            generation = runtime.activate(ModelKey("affine", "r2"), expected_generation=2)
            during_roll = [view.model.key.label for view in runtime.snapshot().models if view.draining]
            new = replace(source, key=RequestKey("sample", "new"))
            runtime.submit(new)
            tick(runtime)
            newer = check(runtime, new, affine_r2, "r2")
            gates.release["old"].set()
            older = check(runtime, old, affine_r1, "r1")

            filter_input = Tensor((3, 2), (1, 0, 0, 1, 0, 0))
            filtered = Request(RequestKey("sample", "filter"), "filter", filter_input, 500)
            output = (2.0, -2.0, 1.75, 1.0, 1.0, -0.25)
            runtime.submit(filtered)
            tick(runtime)
            memory_result = check(runtime, filtered, output, "r1")
            if not any(view.evictions for view in runtime.snapshot().devices):
                raise RuntimeError("resident weights were not evicted under the configured cap")

            to_cancel = replace(filtered, key=RequestKey("sample", "cancel"))
            runtime.submit(to_cancel)
            cancelled = runtime.cancel(to_cancel.key)
            drop = replace(filtered, key=RequestKey("sample", "drop"))
            runtime.submit(drop)
            tick(runtime)
            target = gates.wait_for("drop")
            if not backend.interrupt(target):
                raise RuntimeError("demo did not interrupt a live worker")
            gates.release["drop"].set()
            # Give the retry its complete batching window on the manual clock.
            from time import monotonic, sleep

            end = monotonic() + 5
            while not runtime.snapshot().metrics["counts"].get("request_requeued"):
                if monotonic() >= end:
                    raise RuntimeError("worker loss did not requeue the request")
                sleep(0.005)
            tick(runtime)
            recovered = check(runtime, drop, output, "r1")
            if recovered.attempt != 2 or cancelled.status != ResultStatus.CANCELLED:
                raise RuntimeError("recovery or cancellation failed")

            snapshot = runtime.snapshot().to_json()
            if any(view["workspace_bytes"] for view in snapshot["devices"]):
                raise RuntimeError("a workspace lease was not released")
            metrics = snapshot["metrics"]
            metrics.pop("worker_ns")
            return {
                "interactive": [first.to_json(), second.to_json()],
                "roll": {"generation": generation, "draining": during_roll, "old": older.to_json(), "new": newer.to_json()},
                "memory": {"result": memory_result.to_json(), "devices": snapshot["devices"]},
                "recovery": {"cancelled": cancelled.to_json(), "retried": recovered.to_json()},
                "metrics": metrics,
            }
        finally:
            gates.unblock_all()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="batchsail-demo-") as directory:
        print(encode(run(Path(directory))).decode("utf-8"))
