"""Errors shared by the runtime and its clients."""

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID = "invalid"
    UNKNOWN_MODEL = "unknown_model"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    QUEUE_FULL = "queue_full"
    IMPOSSIBLE = "impossible"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    WORKER_LOST = "worker_lost"
    BAD_OUTPUT = "bad_output"
    EXECUTION = "execution"
    INTERRUPTED = "interrupted"
    UNAVAILABLE = "unavailable"


class SailError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code

    def to_json(self) -> dict[str, str]:
        return {"code": self.code.value, "message": str(self)}


class Invalid(SailError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.INVALID, message)


class Conflict(SailError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.CONFLICT, message)


class WorkerLost(SailError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.WORKER_LOST, message)


class OutputViolation(SailError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.BAD_OUTPUT, message)
