"""Deterministic observation and interruption points for tests and diagnostics."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..batcher.batch import Batch
from ..device import DeviceLease
from ..errors import ErrorCode, SailError


class Boundary(StrEnum):
    DEQUEUED = "dequeued"
    ACQUIRED = "device_acquired"
    RETURNED = "before_results"


class AbortBoundary(SailError):
    def __init__(self, message: str = "execution interrupted at a controlled boundary"):
        super().__init__(ErrorCode.INTERRUPTED, message)


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    boundary: Boundary
    batch: Batch
    lease: DeviceLease | None


class Hook(Protocol):
    def __call__(self, event: BoundaryEvent) -> None: ...


def no_hook(event: BoundaryEvent) -> None:
    pass
