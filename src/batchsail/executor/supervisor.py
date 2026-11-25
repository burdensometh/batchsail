"""Worker outcomes and result protocol validation before settlement."""

from dataclasses import dataclass
import logging
from time import monotonic_ns

from ..batcher.batch import Batch
from ..device import DeviceLease
from ..errors import ErrorCode, OutputViolation, SailError
from ..results.model import Failure
from ..tensor import Tensor
from .backend import Backend, BatchOutput, KernelTimings
from .faults import Boundary, BoundaryEvent, Hook

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchExecution:
    outputs: tuple[Tensor, ...] = ()
    failure: Failure | None = None
    reset: bool = False
    device_usable: bool = True
    backend_ns: int | None = None
    kernel: KernelTimings | None = None


def match_output(batch: Batch, lease: DeviceLease, output: BatchOutput) -> tuple[Tensor, ...]:
    if (output.batch_id, output.lease_token, output.model) != (batch.id, lease.token, batch.model.key):
        raise OutputViolation("worker returned a different batch, lease or revision")
    if len(output.slots) != len(batch.slots):
        raise OutputViolation("worker returned the wrong slot count")
    slots: dict[int, Tensor] = {}
    for item in output.slots:
        index = item.slot.index
        if index in slots or not 0 <= index < len(batch.slots) or batch.slots[index] != item.slot:
            raise OutputViolation("worker returned a duplicate or mismatched slot")
        expected_shape = (item.slot.rows, batch.model.revision.output_spec.columns)
        if not isinstance(item.tensor, Tensor) or item.tensor.shape != expected_shape:
            raise OutputViolation(f"worker returned the wrong output shape for slot {index}")
        slots[index] = item.tensor
    return tuple(slots[index] for index in range(len(batch.slots)))


class Supervisor:
    def __init__(self, backend: Backend, hook: Hook):
        self.backend = backend
        self.hook = hook

    def run(self, batch: Batch, lease: DeviceLease) -> BatchExecution:
        try:
            self.hook(BoundaryEvent(Boundary.ACQUIRED, batch, lease))
            started = monotonic_ns()
            raw = self.backend.execute(batch, lease)
            backend_ns = monotonic_ns() - started
            self.hook(BoundaryEvent(Boundary.RETURNED, batch, lease))
            outputs = match_output(batch, lease, raw)
            timings = raw.timings
            if timings is not None and (
                any(type(value) is not int or value < 0 for value in (
                    timings.load_ns, timings.pack_ns, timings.compute_ns,
                ))
                or type(timings.cache_hit) is not bool
            ):
                raise OutputViolation("worker returned invalid phase timings")
            return BatchExecution(outputs=outputs, backend_ns=backend_ns, kernel=timings)
        except SailError as exc:
            failure = Failure(
                exc.code, str(exc)[:1024] or exc.code.value,
                exc.code in (ErrorCode.WORKER_LOST, ErrorCode.INTERRUPTED),
            )
            if exc.code != ErrorCode.EXECUTION:
                return self._reset(lease, failure)
            return BatchExecution(failure=failure)
        except Exception:
            _LOG.exception("unexpected backend failure on %s for %s", lease.device, batch.id)
            return self._reset(lease, Failure(ErrorCode.EXECUTION, "unexpected backend failure"))

    def _reset(self, lease: DeviceLease, failure: Failure) -> BatchExecution:
        try:
            self.backend.reset(lease.device)
        except Exception:
            _LOG.exception("could not reset device %s", lease.device)
            return BatchExecution(failure=failure, reset=True, device_usable=False)
        return BatchExecution(failure=failure, reset=True)
