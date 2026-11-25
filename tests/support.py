from pathlib import Path
from threading import Event
from time import monotonic, sleep

from batchsail.batcher import BatchingPolicy
from batchsail.config import RuntimeConfig
from batchsail.device import DeviceSpec
from batchsail.executor import Boundary
from batchsail.registry import ModelKey
from batchsail.request import Request, RequestKey
from batchsail.tensor import Tensor

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "fixtures" / "models"


def config(*, backend="inline", devices=(8192,), **batching):
    return RuntimeConfig(
        batching=BatchingPolicy(**batching),
        devices=tuple(DeviceSpec(f"cpu{index}", size) for index, size in enumerate(devices)),
        backend=backend,
    )


def install(runtime, *names):
    for name in names:
        revision = runtime.install(MODELS / name)
        if revision.key.revision == "r1":
            runtime.activate(revision.key)


def request(name, rows=1, *, model="affine", tenant="alpha", budget=5000, retry_of=None):
    width = 3 if model == "affine" else 2
    values = tuple((index % 7) - 3 for index in range(rows * width))
    return Request(RequestKey(tenant, name), model, Tensor((rows, width), values), budget, retry_of)


def until(predicate, *, seconds=5):
    stop = monotonic() + seconds
    while monotonic() < stop:
        result = predicate()
        if result:
            return result
        sleep(0.005)
    raise AssertionError("timed out waiting for the controlled runtime condition")


class Gate:
    def __init__(self, boundary=Boundary.ACQUIRED, request_id=None):
        self.boundary = boundary
        self.request_id = request_id
        self.reached = Event()
        self.release = Event()
        self.event = None

    def __call__(self, event):
        if self.reached.is_set() or event.boundary != self.boundary:
            return
        if self.request_id is not None and all(slot.key.request_id != self.request_id for slot in event.batch.slots):
            return
        self.event = event
        self.reached.set()
        if not self.release.wait(10):
            raise RuntimeError("test did not release its gate")

    def await_reached(self):
        if not self.reached.wait(5):
            raise AssertionError(f"runtime did not reach {self.boundary}")
        return self.event
