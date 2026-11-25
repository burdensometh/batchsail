"""Request identities, repeatable submissions, and monotonic deadlines."""

from dataclasses import dataclass
from typing import Any

from ..clock import NS_PER_MS
from ..codec import digest, encode, fields, identifier, integer
from ..errors import Invalid
from ..tensor import Tensor

MAX_BUDGET_MS = 86_400_000


@dataclass(frozen=True, slots=True, order=True)
class RequestKey:
    tenant: str
    request_id: str

    def __post_init__(self) -> None:
        identifier(self.tenant, "tenant")
        identifier(self.request_id, "request_id")

    def to_json(self) -> dict[str, str]:
        return {"tenant": self.tenant, "request_id": self.request_id}

    @classmethod
    def from_json(cls, value: Any) -> "RequestKey":
        key = fields(value, ("tenant", "request_id"), "request key")
        return cls(key["tenant"], key["request_id"])


@dataclass(frozen=True, slots=True)
class RequestDeadline:
    accepted_ns: int
    deadline_ns: int | None

    @classmethod
    def from_budget(cls, accepted_ns: int, budget_ms: int | None) -> "RequestDeadline":
        if budget_ms is None:
            return cls(accepted_ns, None)
        milliseconds = integer(budget_ms, "deadline_ms", maximum=MAX_BUDGET_MS)
        return cls(accepted_ns, accepted_ns + milliseconds * NS_PER_MS)

    def expired(self, now_ns: int) -> bool:
        return self.deadline_ns is not None and now_ns >= self.deadline_ns


@dataclass(frozen=True, slots=True)
class Request:
    key: RequestKey
    model: str
    tensor: Tensor
    deadline_ms: int | None = None
    retry_of: str | None = None

    def __post_init__(self) -> None:
        identifier(self.model, "model")
        if self.deadline_ms is not None:
            integer(self.deadline_ms, "deadline_ms", maximum=MAX_BUDGET_MS)
        if self.retry_of is not None and identifier(self.retry_of, "retry_of") == self.key.request_id:
            raise Invalid("retry_of must identify a different request")

    @property
    def fingerprint(self) -> str:
        return digest(encode(self.to_json()))

    def to_json(self) -> dict[str, Any]:
        return {
            **self.key.to_json(),
            "model": self.model,
            "input": self.tensor.to_json(),
            "deadline_ms": self.deadline_ms,
            "retry_of": self.retry_of,
        }

    @classmethod
    def from_json(cls, value: Any) -> "Request":
        data = fields(
            value, ("tenant", "request_id", "model", "input"), "request", ("deadline_ms", "retry_of"),
        )
        return cls(
            RequestKey(data["tenant"], data["request_id"]), data["model"], Tensor.from_json(data["input"]),
            data.get("deadline_ms"), data.get("retry_of"),
        )


@dataclass(frozen=True, slots=True)
class Admission:
    key: RequestKey
    reused: bool
    revision: str
    accepted_at_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            **self.key.to_json(),
            "reused": self.reused,
            "revision": self.revision,
            "accepted_at_ms": self.accepted_at_ms,
        }
