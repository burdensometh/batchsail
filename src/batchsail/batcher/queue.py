"""Compatible-shape queues with round-robin tenant selection."""

from collections import OrderedDict, deque
from dataclasses import dataclass
import heapq
from typing import Any

from ..clock import NS_PER_MS
from ..codec import fields, integer
from ..registry.model import ModelRevision
from ..request.lifecycle import Ticket
from ..request.model import RequestKey
from .batch import ShapeKey


@dataclass(frozen=True, slots=True)
class BatchingPolicy:
    max_slots: int = 16
    wait_ms: int = 5
    deadline_guard_ms: int = 2
    row_multiple: int = 4
    max_pending: int = 4096
    max_attempts: int = 2

    def __post_init__(self) -> None:
        integer(self.max_slots, "max_slots", minimum=1, maximum=64)
        integer(self.wait_ms, "wait_ms", maximum=60_000)
        integer(self.deadline_guard_ms, "deadline_guard_ms", maximum=60_000)
        integer(self.row_multiple, "row_multiple", minimum=1, maximum=65_536)
        integer(self.max_pending, "max_pending", minimum=1, maximum=1_000_000)
        integer(self.max_attempts, "max_attempts", minimum=1, maximum=8)

    def shape(self, ticket: Ticket) -> ShapeKey:
        revision = ticket.pin.model.revision
        tensor = ticket.request.tensor
        rows = self.padded_rows(tensor.rows, revision.input_spec.max_rows)
        return ShapeKey(revision.key, rows, tensor.columns)

    def padded_rows(self, rows: int, max_rows: int) -> int:
        multiple = self.row_multiple
        return min(max_rows, (rows + multiple - 1) // multiple * multiple)

    @classmethod
    def from_config(cls, value: Any) -> "BatchingPolicy":
        data = fields(
            value, (), "batching",
            ("max_slots", "wait_ms", "deadline_guard_ms", "row_multiple", "max_pending", "max_attempts"),
        )
        return cls(**data)


class ShapeQueue:
    def __init__(self, shape: ShapeKey, model: ModelRevision):
        self.shape = shape
        self.model = model
        self._tickets: OrderedDict[RequestKey, Ticket] = OrderedDict()
        self._tenants: OrderedDict[str, OrderedDict[RequestKey, Ticket]] = OrderedDict()
        self._deadlines: list[tuple[int, int, int, RequestKey]] = []

    def __len__(self) -> int:
        return len(self._tickets)

    @property
    def oldest_ns(self) -> int:
        return next(iter(self._tickets.values())).queued_since_ns

    def add(self, ticket: Ticket) -> None:
        if ticket.key in self._tickets:
            raise RuntimeError("request already in shape queue")
        self._tickets[ticket.key] = ticket
        self._tenants.setdefault(ticket.key.tenant, OrderedDict())[ticket.key] = ticket
        bound = ticket.deadline.deadline_ns
        if bound is not None:
            heapq.heappush(self._deadlines, (bound, ticket.sequence, ticket.attempt, ticket.key))

    def _valid_deadline(self, entry: tuple[int, int, int, RequestKey]) -> bool:
        _, sequence, attempt, key = entry
        ticket = self._tickets.get(key)
        return ticket is not None and ticket.sequence == sequence and ticket.attempt == attempt

    def _earliest(self) -> Ticket | None:
        while self._deadlines and not self._valid_deadline(self._deadlines[0]):
            heapq.heappop(self._deadlines)
        return None if not self._deadlines else self._tickets[self._deadlines[0][3]]

    def remove(self, key: RequestKey, *, served: bool = False) -> None:
        del self._tickets[key]
        tenant = self._tenants[key.tenant]
        del tenant[key]
        if not tenant:
            del self._tenants[key.tenant]
        elif served:
            self._tenants.move_to_end(key.tenant)
        if len(self._deadlines) > 2 * len(self._tickets) + 64:
            self._deadlines = [entry for entry in self._deadlines if self._valid_deadline(entry)]
            heapq.heapify(self._deadlines)

    def readiness_ns(self, policy: BatchingPolicy, maximum: int) -> int:
        if len(self) >= maximum:
            return self.oldest_ns
        ready = self.oldest_ns + policy.wait_ms * NS_PER_MS
        earliest = self._earliest()
        if earliest is not None:
            bound = earliest.deadline.deadline_ns
            if bound is not None:
                ready = min(ready, bound - policy.deadline_guard_ms * NS_PER_MS)
        return ready

    def choose(self, maximum: int, urgent_before_ns: int) -> tuple[Ticket, ...]:
        chosen: list[Ticket] = []
        urgent = self._earliest()
        priority: RequestKey | None = None
        if urgent is not None:
            bound = urgent.deadline.deadline_ns
            if bound is not None and bound <= urgent_before_ns:
                priority = urgent.key
                chosen.append(urgent)
        if len(chosen) >= maximum:
            return tuple(chosen)
        turns = deque(
            iter(tickets.values()) for tenant, tickets in self._tenants.items()
            if priority is None or tenant != priority.tenant
        )
        if priority is not None:
            turns.append(iter(self._tenants[priority.tenant].values()))
        while len(chosen) < maximum and turns:
            turn = turns.popleft()
            next_ticket = next(turn, None)
            if next_ticket is not None and next_ticket.key == priority:
                next_ticket = next(turn, None)
            if next_ticket is not None:
                chosen.append(next_ticket)
                turns.append(turn)
        return tuple(chosen)


class BatchQueues:
    def __init__(self, policy: BatchingPolicy):
        self.policy = policy
        self._groups: dict[ShapeKey, ShapeQueue] = {}
        self._locations: dict[RequestKey, ShapeKey] = {}

    def __len__(self) -> int:
        return len(self._locations)

    def add(self, ticket: Ticket, now_ns: int) -> None:
        if ticket.key in self._locations:
            raise RuntimeError("request already queued")
        shape = self.policy.shape(ticket)
        bucket = self._groups.get(shape)
        if bucket is None:
            bucket = self._groups[shape] = ShapeQueue(shape, ticket.pin.model.revision)
        ticket.queued_since_ns = now_ns
        bucket.add(ticket)
        self._locations[ticket.key] = shape

    def discard(self, key: RequestKey) -> bool:
        shape = self._locations.pop(key, None)
        if shape is None:
            return False
        bucket = self._groups[shape]
        bucket.remove(key)
        if not bucket:
            del self._groups[shape]
        return True

    def commit(self, shape: ShapeKey, tickets: tuple[Ticket, ...]) -> None:
        bucket = self._groups[shape]
        if any(self._locations.get(ticket.key) != shape for ticket in tickets):
            raise RuntimeError("batch was selected from a stale queue")
        for ticket in tickets:
            bucket.remove(ticket.key, served=True)
            del self._locations[ticket.key]
        if not bucket:
            del self._groups[shape]

    def groups(self) -> tuple[ShapeQueue, ...]:
        return tuple(self._groups.values())
