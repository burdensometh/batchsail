"""Live request state and an indexed deadline alarm queue."""

from dataclasses import dataclass
import heapq

from ..registry.catalog import RevisionPin
from .model import Request, RequestDeadline, RequestKey


@dataclass(slots=True)
class Ticket:
    request: Request
    pin: RevisionPin
    deadline: RequestDeadline
    sequence: int
    attempt: int = 0
    queued_since_ns: int = 0
    batch_id: str | None = None
    terminal: bool = False

    @property
    def key(self) -> RequestKey:
        return self.request.key


class DeadlineIndex:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, RequestKey]] = []
        self._live: dict[RequestKey, int] = {}

    def add(self, ticket: Ticket) -> None:
        bound = ticket.deadline.deadline_ns
        if bound is not None:
            self._live[ticket.key] = ticket.sequence
            heapq.heappush(self._heap, (bound, ticket.sequence, ticket.key))

    def discard(self, ticket: Ticket) -> None:
        self._live.pop(ticket.key, None)
        if len(self._heap) > 2 * len(self._live) + 64:
            self._heap = [entry for entry in self._heap if self._live.get(entry[2]) == entry[1]]
            heapq.heapify(self._heap)

    def _prune(self) -> None:
        while self._heap and self._live.get(self._heap[0][2]) != self._heap[0][1]:
            heapq.heappop(self._heap)

    def next_ns(self) -> int | None:
        self._prune()
        return None if not self._heap else self._heap[0][0]

    def due(self, now_ns: int) -> list[RequestKey]:
        self._prune()
        expired: list[RequestKey] = []
        while self._heap and self._heap[0][0] <= now_ns:
            _, sequence, key = heapq.heappop(self._heap)
            if self._live.get(key) == sequence:
                del self._live[key]
                expired.append(key)
        return expired
