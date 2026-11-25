"""Backend boundary and the inline CPU implementation."""

from dataclasses import dataclass
from time import monotonic_ns
from typing import Protocol

from ..batcher.batch import Batch, BatchSlot, PackedBatch
from ..codec import read_bytes
from ..device import DeviceLease, Residency
from ..errors import OutputViolation
from ..registry.model import InstalledModel, MAX_MODEL_FILE, ModelKey, WeightSet
from ..tensor import Tensor
from .kernels import Kernel, compile_kernel


@dataclass(frozen=True, slots=True)
class SlotOutput:
    slot: BatchSlot
    tensor: Tensor


@dataclass(frozen=True, slots=True)
class KernelTimings:
    load_ns: int
    pack_ns: int
    compute_ns: int
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class BatchOutput:
    batch_id: str
    lease_token: int
    model: ModelKey
    slots: tuple[SlotOutput, ...]
    timings: KernelTimings | None = None


class Backend(Protocol):
    """Execute on one exclusive lease; reset must discard all device state."""

    def execute(self, batch: Batch, lease: DeviceLease) -> BatchOutput: ...
    def prune(self, device: str, keep: tuple[Residency, ...]) -> tuple[Residency, ...]: ...
    def reset(self, device: str) -> None: ...
    def close(self) -> None: ...


class ResidentKernels:
    def __init__(self):
        self._kernels: dict[Residency, Kernel] = {}

    def prune(self, keep: tuple[Residency, ...]) -> tuple[Residency, ...]:
        available = set(keep)
        for residency in tuple(self._kernels):
            if residency not in available:
                del self._kernels[residency]
        return tuple(self._kernels)

    def run(self, batch: Batch, lease: DeviceLease) -> BatchOutput:
        if lease.key != batch.model.key or lease.residency not in lease.cache:
            raise OutputViolation("batch does not match its device residency")
        self.prune(lease.cache)
        kernel = self._kernels.get(lease.residency)
        cache_hit = kernel is not None
        start = monotonic_ns()
        if kernel is None:
            kernel = self._load(batch.model)
            self._kernels[lease.residency] = kernel
        loaded = monotonic_ns()
        packed = PackedBatch.pack(batch)
        packed_at = monotonic_ns()
        outputs = kernel.run(packed)
        computed = monotonic_ns()
        if len(outputs) != len(batch.slots):
            raise OutputViolation("kernel returned an incorrect slot count")
        return BatchOutput(
            batch.id, lease.token, batch.model.key,
            tuple(SlotOutput(slot, tensor) for slot, tensor in zip(batch.slots, outputs, strict=True)),
            KernelTimings(loaded - start, packed_at - loaded, computed - packed_at, cache_hit),
        )

    @staticmethod
    def _load(model: InstalledModel) -> Kernel:
        data = read_bytes(model.weight_path, maximum=MAX_MODEL_FILE)
        weights = WeightSet.from_v1(data, model.revision)
        return compile_kernel(model.revision, weights)


class InlineBackend:
    def __init__(self, devices: tuple[str, ...]):
        self._caches = {device: ResidentKernels() for device in devices}

    def execute(self, batch: Batch, lease: DeviceLease) -> BatchOutput:
        return self._caches[lease.device].run(batch, lease)

    def prune(self, device: str, keep: tuple[Residency, ...]) -> tuple[Residency, ...]:
        return self._caches[device].prune(keep)

    def reset(self, device: str) -> None:
        self._caches[device] = ResidentKernels()

    def close(self) -> None:
        self._caches.clear()
