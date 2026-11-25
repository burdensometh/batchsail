"""Client-facing terminal envelopes and the portable version-one receipt record."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..codec import digest_text, fields, identifier, integer
from ..errors import ErrorCode, Invalid
from ..registry.model import ModelKey
from ..request.model import Admission, RequestKey
from ..tensor import Tensor


class ResultStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self != self.PENDING


@dataclass(frozen=True, slots=True)
class Failure:
    code: ErrorCode
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.message, str) or not 0 < len(self.message) <= 1024:
            raise Invalid("failure message must contain 1 to 1024 characters")
        if type(self.retryable) is not bool:
            raise Invalid("failure retryable must be a boolean")

    def to_json(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_json(cls, value: Any) -> "Failure":
        obj = fields(value, ("code", "message", "retryable"), "failure")
        try:
            code = ErrorCode(obj["code"])
        except ValueError as exc:
            raise Invalid("unknown failure code") from exc
        return cls(code, obj["message"], obj["retryable"])


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    key: RequestKey
    model: ModelKey
    status: ResultStatus
    accepted_at_ms: int
    attempt: int = 0
    batch_id: str | None = None
    slot: int | None = None
    device: str | None = None
    queue_ns: int | None = None
    elapsed_ns: int | None = None
    output: Tensor | None = None
    failure: Failure | None = None

    def __post_init__(self) -> None:
        integer(self.accepted_at_ms, "accepted_at_ms")
        integer(self.attempt, "attempt", maximum=8)
        assignment = (self.batch_id, self.slot, self.device, self.queue_ns)
        if self.attempt:
            identifier(self.batch_id, "batch_id")
            identifier(self.device, "device")
            integer(self.slot, "slot index", maximum=63)
            integer(self.queue_ns, "queue_ns")
        elif any(value is not None for value in assignment):
            raise Invalid("an unassigned result may not describe a batch slot")
        if self.status.terminal:
            integer(self.elapsed_ns, "elapsed_ns")
            if self.queue_ns is not None and self.elapsed_ns < self.queue_ns:
                raise Invalid("elapsed time is shorter than queue time")
        elif self.elapsed_ns is not None:
            raise Invalid("a pending result may not have elapsed time")
        if (self.output is not None) != (self.status == ResultStatus.SUCCEEDED):
            raise Invalid("only a successful result carries a tensor")
        if self.status == ResultStatus.SUCCEEDED and not self.attempt:
            raise Invalid("a successful result must identify its attempt")
        has_error = self.status in (ResultStatus.CANCELLED, ResultStatus.EXPIRED, ResultStatus.FAILED)
        if (self.failure is not None) != has_error:
            raise Invalid("terminal failures must carry an error")
        if self.status == ResultStatus.CANCELLED and self.failure.code != ErrorCode.CANCELLED:
            raise Invalid("cancelled results require a cancellation error")
        if self.status == ResultStatus.EXPIRED and self.failure.code != ErrorCode.DEADLINE:
            raise Invalid("expired results require a deadline error")

    def admission(self, *, reused: bool) -> Admission:
        return Admission(self.key, reused, self.model.revision, self.accepted_at_ms)

    def to_json(self) -> dict[str, Any]:
        return {
            **self.key.to_json(),
            **self.model.to_json(),
            "status": self.status.value,
            "accepted_at_ms": self.accepted_at_ms,
            "attempt": self.attempt,
            "batch_id": self.batch_id,
            "slot": self.slot,
            "device": self.device,
            "queue_ns": self.queue_ns,
            "elapsed_ns": self.elapsed_ns,
            "output": None if self.output is None else self.output.to_json(),
            "failure": None if self.failure is None else self.failure.to_json(),
        }

    @classmethod
    def from_json(cls, value: Any) -> "ResultEnvelope":
        result = fields(
            value,
            (
                "tenant", "request_id", "model", "revision", "status", "accepted_at_ms", "attempt",
                "batch_id", "slot", "device", "queue_ns", "elapsed_ns", "output", "failure",
            ),
            "result",
        )
        try:
            status = ResultStatus(result["status"])
        except ValueError as exc:
            raise Invalid("invalid result status") from exc
        return cls(
            key=RequestKey(result["tenant"], result["request_id"]),
            model=ModelKey(result["model"], result["revision"]),
            status=status,
            accepted_at_ms=result["accepted_at_ms"],
            attempt=result["attempt"],
            batch_id=result["batch_id"],
            slot=result["slot"],
            device=result["device"],
            queue_ns=result["queue_ns"],
            elapsed_ns=result["elapsed_ns"],
            output=None if result["output"] is None else Tensor.from_json(result["output"]),
            failure=None if result["failure"] is None else Failure.from_json(result["failure"]),
        )


@dataclass(frozen=True, slots=True)
class Receipt:
    fingerprint: str
    result: ResultEnvelope

    def __post_init__(self) -> None:
        digest_text(self.fingerprint, "request fingerprint")

    def to_v1(self) -> dict[str, Any]:
        return {"version": 1, "fingerprint": self.fingerprint, "result": self.result.to_json()}

    @classmethod
    def from_v1(cls, value: Any) -> "Receipt":
        receipt = fields(value, ("version", "fingerprint", "result"), "receipt")
        integer(receipt["version"], "receipt version", minimum=1, maximum=1)
        return cls(receipt["fingerprint"], ResultEnvelope.from_json(receipt["result"]))
