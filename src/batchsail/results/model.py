"""Client-facing envelopes, attempted-work evidence, and portable receipt records."""

from dataclasses import dataclass, replace
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
class AttemptEvidence:
    """One dispatch of a request and how that attempt physically ended.

    ``outcome`` is null exactly while the attempt is still in flight. A
    ``None`` attempt list on a receipt means no evidence was ever recorded;
    ``attempts_known`` distinguishes complete history from a partially known
    tail left behind by a version-one receipt.
    """

    attempt: int
    batch_id: str
    slot: int
    device: str
    queue_ns: int
    dispatched_at_ms: int | None = None
    outcome: ResultStatus | None = None
    failure: Failure | None = None
    finished_at_ms: int | None = None

    def __post_init__(self) -> None:
        integer(self.attempt, "attempt", minimum=1, maximum=8)
        identifier(self.batch_id, "batch_id")
        integer(self.slot, "slot index", maximum=63)
        identifier(self.device, "device")
        integer(self.queue_ns, "queue_ns")
        if self.dispatched_at_ms is not None:
            integer(self.dispatched_at_ms, "dispatched_at_ms")
        if self.finished_at_ms is not None:
            integer(self.finished_at_ms, "finished_at_ms")
        if (
            self.dispatched_at_ms is not None
            and self.finished_at_ms is not None
            and self.finished_at_ms < self.dispatched_at_ms
        ):
            raise Invalid("an attempt cannot finish before it was dispatched")
        if self.outcome is not None and not isinstance(self.outcome, ResultStatus):
            raise Invalid("attempt outcome must be a result status")
        if self.outcome == ResultStatus.PENDING:
            raise Invalid("an concluded attempt cannot be pending")
        if self.outcome is None:
            if self.failure is not None or self.finished_at_ms is not None:
                raise Invalid("an unfinished attempt cannot have a failure or finish time")
            return
        if self.finished_at_ms is None:
            raise Invalid("a concluded attempt must record when it finished")
        if self.outcome == ResultStatus.SUCCEEDED:
            if self.failure is not None:
                raise Invalid("a successful attempt may not carry an error")
        elif self.failure is None:
            raise Invalid("a failed, cancelled or expired attempt must carry an error")
        elif self.outcome == ResultStatus.CANCELLED and self.failure.code != ErrorCode.CANCELLED:
            raise Invalid("cancelled attempts require a cancellation error")
        elif self.outcome == ResultStatus.EXPIRED and self.failure.code != ErrorCode.DEADLINE:
            raise Invalid("expired attempts require a deadline error")

    @property
    def concluded(self) -> bool:
        return self.outcome is not None

    def same_dispatch(self, other: "AttemptEvidence") -> bool:
        """True when both describe the same physical dispatch."""
        return (
            self.attempt == other.attempt
            and self.batch_id == other.batch_id
            and self.slot == other.slot
            and self.device == other.device
            and self.queue_ns == other.queue_ns
            and self.dispatched_at_ms == other.dispatched_at_ms
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "batch_id": self.batch_id,
            "slot": self.slot,
            "device": self.device,
            "queue_ns": self.queue_ns,
            "dispatched_at_ms": self.dispatched_at_ms,
            "outcome": None if self.outcome is None else self.outcome.value,
            "failure": None if self.failure is None else self.failure.to_json(),
            "finished_at_ms": self.finished_at_ms,
        }

    @classmethod
    def from_json(cls, value: Any) -> "AttemptEvidence":
        item = fields(
            value,
            (
                "attempt", "batch_id", "slot", "device", "queue_ns", "dispatched_at_ms",
                "outcome", "failure", "finished_at_ms",
            ),
            "attempt evidence",
        )
        outcome = item["outcome"]
        if outcome is not None:
            try:
                outcome = ResultStatus(outcome)
            except ValueError as exc:
                raise Invalid("unknown attempt outcome") from exc
        return cls(
            item["attempt"], item["batch_id"], item["slot"], item["device"], item["queue_ns"],
            item["dispatched_at_ms"], outcome,
            None if item["failure"] is None else Failure.from_json(item["failure"]),
            item["finished_at_ms"],
        )


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
    """A request fingerprint, its observable result, and its attempt history.

    Version one receipts are preserved byte-for-byte on export and carry no
    attempt evidence. Version two receipts record every dispatch in
    ``attempts``; an empty tuple means no attempt was ever assigned, while
    ``None`` means the history is unknown (a legacy version one receipt).
    """

    fingerprint: str
    result: ResultEnvelope
    attempts: tuple[AttemptEvidence, ...] | None = None
    version: int = 1

    def __post_init__(self) -> None:
        digest_text(self.fingerprint, "request fingerprint")
        integer(self.version, "receipt version", minimum=1, maximum=2)
        if self.version == 1:
            if self.attempts is not None:
                raise Invalid("a version one receipt has no attempt evidence")
            return
        if self.attempts is None:
            raise Invalid("a version two receipt must state its attempt history")
        expected = 1
        for evidence in self.attempts:
            if evidence.attempt != expected:
                raise Invalid("attempt evidence must be numbered consecutively from one")
            expected += 1
        if self.result.attempt and self.result.attempt > len(self.attempts):
            raise Invalid("the result names an attempt with no evidence")
        if len(self.attempts) > 8:
            raise Invalid("a receipt may record at most eight attempts")

    @property
    def attempts_known(self) -> bool:
        return self.attempts is not None

    @property
    def final_attempt(self) -> AttemptEvidence | None:
        return None if not self.attempts else self.attempts[-1]

    @classmethod
    def tracked(cls, fingerprint: str, result: ResultEnvelope) -> "Receipt":
        """A new version-two receipt that will record attempts as they occur."""
        return cls(fingerprint, result, (), 2)

    def with_attempts(self, attempts: tuple[AttemptEvidence, ...]) -> "Receipt":
        if self.version == 1:
            return replace(self, attempts=attempts, version=2)
        return replace(self, attempts=attempts)

    def to_json(self) -> dict[str, Any]:
        if self.version == 1:
            return self.to_v1()
        return {
            "version": 2,
            "fingerprint": self.fingerprint,
            "result": self.result.to_json(),
            "attempts": [evidence.to_json() for evidence in self.attempts or ()],
        }

    def attempts_json(self) -> list[dict[str, Any]] | None:
        """Known attempts as JSON, or ``None`` when the history is unknown."""
        if self.attempts is None:
            return None
        return [evidence.to_json() for evidence in self.attempts]

    def query_json(self) -> dict[str, Any]:
        """A receipt view that states explicitly whether attempts are known."""
        document = self.to_json()
        document["attempts_known"] = self.attempts_known
        return document

    def to_v1(self) -> dict[str, Any]:
        return {"version": 1, "fingerprint": self.fingerprint, "result": self.result.to_json()}

    @classmethod
    def from_json(cls, value: Any) -> "Receipt":
        if not isinstance(value, dict) or "version" not in value:
            raise Invalid("receipt must be an object with a version")
        version = value["version"]
        if version == 1:
            return cls.from_v1(value)
        if version == 2:
            document = fields(value, ("version", "fingerprint", "result", "attempts"), "receipt")
            attempts = document["attempts"]
            if not isinstance(attempts, list):
                raise Invalid("receipt attempts must be an array")
            return cls(
                document["fingerprint"],
                ResultEnvelope.from_json(document["result"]),
                tuple(AttemptEvidence.from_json(item) for item in attempts),
                2,
            )
        raise Invalid("receipt version must be 1 or 2")

    @classmethod
    def from_v1(cls, value: Any) -> "Receipt":
        receipt = fields(value, ("version", "fingerprint", "result"), "receipt")
        integer(receipt["version"], "receipt version", minimum=1, maximum=1)
        return cls(receipt["fingerprint"], ResultEnvelope.from_json(receipt["result"]))
