"""Clock interfaces used to coordinate request timers and deterministic tests."""

from threading import Lock
from time import monotonic_ns, time_ns
from typing import Protocol

NS_PER_MS = 1_000_000


class Clock(Protocol):
    def now_ns(self) -> int: ...
    def wall_ms(self) -> int: ...


class SystemClock:
    def now_ns(self) -> int:
        return monotonic_ns()

    def wall_ms(self) -> int:
        return time_ns() // NS_PER_MS


class ManualClock:
    def __init__(self, wall_ms: int = 0):
        self._now = 0
        self._wall = wall_ms
        self._lock = Lock()

    def now_ns(self) -> int:
        with self._lock:
            return self._now

    def wall_ms(self) -> int:
        with self._lock:
            return self._wall + self._now // NS_PER_MS

    def advance_ms(self, milliseconds: int) -> None:
        if type(milliseconds) is not int or milliseconds < 0:
            raise ValueError("manual time can only advance by a nonnegative integer")
        with self._lock:
            self._now += milliseconds * NS_PER_MS
