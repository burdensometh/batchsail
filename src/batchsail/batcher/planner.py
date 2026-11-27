"""Select an eligible shape bucket and a size that fits idle device memory."""

from dataclasses import dataclass

from ..clock import NS_PER_MS
from ..device import DeviceLedger, MemoryPolicy, Placement, memory_demand
from ..device.types import DeviceRejection, MemoryDemand
from ..registry.model import ModelKey
from ..request.lifecycle import Ticket
from .batch import ShapeKey
from .queue import BatchQueues, BatchingPolicy, ShapeQueue


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    shape: ShapeKey
    tickets: tuple[Ticket, ...]
    placement: Placement


@dataclass(frozen=True, slots=True)
class SchedulingBlock:
    shape: ShapeKey
    queued: int
    reasons: tuple[DeviceRejection, ...]


class BatchPlanner:
    def __init__(self, queues: BatchQueues, ledger: DeviceLedger, memory: MemoryPolicy):
        self.queues = queues
        self.ledger = ledger
        self.memory = memory

    def _demand(self, group: ShapeQueue, slots: int) -> MemoryDemand:
        return memory_demand(self.memory, group.model, padded_rows=group.shape.rows, slots=slots)

    def maximum_slots(self, group: ShapeQueue) -> int:
        policy = self.queues.policy
        # Admission guarantees that a single slot fits the largest device.
        # Binary search the aligned byte contract to avoid waiting for a batch
        # size that could never run even with every device cache empty.
        low = 1
        high = policy.max_slots
        capacity = self.ledger.max_capacity
        while low < high:
            mid = (low + high + 1) // 2
            demand = self._demand(group, mid)
            if demand.weight_bytes + demand.workspace_bytes <= capacity:
                low = mid
            else:
                high = mid - 1
        return low

    def select(
        self, now_ns: int, protected: frozenset[ModelKey], *, flush: bool = False,
    ) -> tuple[DispatchPlan | None, int | None, tuple[SchedulingBlock, ...]]:
        if not self.ledger.has_idle_device():
            return None, None, ()
        policy = self.queues.policy
        ready: list[tuple[int, int, ShapeKey, int, ShapeQueue]] = []
        next_ns: int | None = None
        for group in self.queues.groups():
            maximum = self.maximum_slots(group)
            time_ns = group.readiness_ns(policy, maximum)
            if flush:
                time_ns = min(now_ns, time_ns)
            if time_ns <= now_ns:
                ready.append((time_ns, group.oldest_ns, group.shape, maximum, group))
            elif next_ns is None or time_ns < next_ns:
                next_ns = time_ns

        ordered = sorted(ready)
        if policy.urgent_priority:
            ordered = self._urgent_first(ordered, now_ns, policy)

        blocked: list[SchedulingBlock] = []
        for _, _, shape, maximum, group in ordered:
            tickets = group.choose(maximum, now_ns + policy.deadline_guard_ms * NS_PER_MS)
            for size in range(len(tickets), 0, -1):
                candidates = self.ledger.assess(self._demand(group, size), protected)
                placement = self.ledger.preferred(candidates)
                if placement is not None:
                    return DispatchPlan(shape, tickets[:size], placement), next_ns, tuple(blocked)
            blocked.append(SchedulingBlock(
                shape, len(group),
                tuple(candidate for candidate in candidates if isinstance(candidate, DeviceRejection)),
            ))
        return None, next_ns, tuple(blocked)

    def _urgent_first(
        self,
        ordered: list[tuple[int, int, ShapeKey, int, ShapeQueue]],
        now_ns: int,
        policy: BatchingPolicy,
    ) -> list[tuple[int, int, ShapeKey, int, ShapeQueue]]:
        """Float groups inside the deadline guard ahead of earlier-queued work.

        An urgent group is one whose nearest request deadline is within the
        dispatch guard. Those groups are tried by earliest deadline first, ahead
        of the usual readiness/age order. Groups that are already full but not
        urgent keep their original relative order behind the urgent candidates.
        """
        guard_ns = policy.deadline_guard_ms * NS_PER_MS
        horizon_ns = now_ns + guard_ns
        urgent: list[tuple[int, tuple[int, int, ShapeKey, int, ShapeQueue]]] = []
        normal: list[tuple[int, int, ShapeKey, int, ShapeQueue]] = []
        for entry in ordered:
            deadline_ns = entry[4].earliest_deadline_ns
            if deadline_ns is not None and deadline_ns <= horizon_ns:
                urgent.append((deadline_ns, entry))
            else:
                normal.append(entry)
        urgent.sort(key=lambda item: (item[0],) + item[1][:3])
        return [entry for _, entry in urgent] + normal
