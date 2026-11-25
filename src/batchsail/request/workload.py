"""A versioned timeline of inference requests, cancellations, and model rolls."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..codec import atomic_write, decode, encode, fields, identifier, integer, read_bytes
from ..errors import Invalid
from ..registry.model import ModelKey
from ..results.model import ResultStatus
from ..tensor import Tensor
from .model import Request, RequestKey

_MAX_WORKLOAD_BYTES = 32 * 1024 * 1024
_MAX_ITEMS = 10_000
_MAX_DURATION_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class Expectation:
    status: ResultStatus = ResultStatus.SUCCEEDED
    revision: str | None = None
    output: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.status.terminal:
            raise Invalid("workload checks require a terminal status")
        if self.revision is not None:
            identifier(self.revision, "expected revision")
        if self.output is not None and self.status != ResultStatus.SUCCEEDED:
            raise Invalid("only success can have an expected output")

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "revision": self.revision,
            "output": None if self.output is None else self.output.to_json(),
        }

    @classmethod
    def from_json(cls, value: Any) -> "Expectation":
        data = fields(value, ("status",), "workload expectation", ("revision", "output"))
        try:
            status = ResultStatus(data["status"])
        except ValueError as exc:
            raise Invalid("unknown workload status") from exc
        output = data.get("output")
        return cls(status, data.get("revision"), None if output is None else Tensor.from_json(output))


@dataclass(frozen=True, slots=True)
class Submit:
    at_ms: int
    request: Request
    expect: Expectation | None


@dataclass(frozen=True, slots=True)
class Cancel:
    at_ms: int
    key: RequestKey


@dataclass(frozen=True, slots=True)
class Activate:
    at_ms: int
    key: ModelKey
    expected_generation: int | None


Step = Submit | Cancel | Activate


def _step_from_json(value: Any) -> Step:
    if not isinstance(value, dict):
        raise Invalid("workload step must be an object")
    at_ms = integer(value.get("at_ms"), "workload time", maximum=_MAX_DURATION_MS)
    operation = value.get("op")
    if operation == "submit":
        data = fields(value, ("op", "at_ms", "request"), "submit step", ("expect",))
        expectation = None if data.get("expect") is None else Expectation.from_json(data["expect"])
        return Submit(at_ms, Request.from_json(data["request"]), expectation)
    if operation == "cancel":
        data = fields(value, ("op", "at_ms", "key"), "cancel step")
        return Cancel(at_ms, RequestKey.from_json(data["key"]))
    if operation == "activate":
        data = fields(value, ("op", "at_ms", "key"), "activate step", ("expected_generation",))
        expected = data.get("expected_generation")
        generation = None if expected is None else integer(expected, "expected_generation")
        return Activate(at_ms, ModelKey.from_json(data["key"]), generation)
    raise Invalid(f"unrecognized workload operation {operation!r}")


def _step_to_json(step: Step) -> dict[str, Any]:
    if isinstance(step, Submit):
        return {
            "op": "submit", "at_ms": step.at_ms, "request": step.request.to_json(),
            "expect": None if step.expect is None else step.expect.to_json(),
        }
    if isinstance(step, Cancel):
        return {"op": "cancel", "at_ms": step.at_ms, "key": step.key.to_json()}
    return {
        "op": "activate", "at_ms": step.at_ms, "key": step.key.to_json(),
        "expected_generation": step.expected_generation,
    }


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        identifier(self.name, "workload name")
        integer(len(self.steps), "step count", minimum=1, maximum=_MAX_ITEMS)
        admitted: set[RequestKey] = set()
        previous = 0
        for index, step in enumerate(self.steps, start=1):
            when = integer(step.at_ms, "workload time", maximum=_MAX_DURATION_MS)
            if when < previous:
                raise Invalid(f"step {index} is earlier than the preceding step")
            previous = when
            if isinstance(step, Submit):
                if step.request.key in admitted:
                    raise Invalid(f"duplicate submission in workload at step {index}")
                admitted.add(step.request.key)
            elif isinstance(step, Cancel) and step.key not in admitted:
                raise Invalid(f"step {index} cancels a key the workload has not submitted")
        if not admitted:
            raise Invalid("workload needs at least one submission")

    @property
    def submissions(self) -> tuple[Submit, ...]:
        return tuple(step for step in self.steps if isinstance(step, Submit))

    @classmethod
    def from_v1(cls, value: Any) -> "Workload":
        doc = fields(value, ("version", "name", "steps"), "workload")
        integer(doc["version"], "workload version", minimum=1, maximum=1)
        raw_steps = doc["steps"]
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= _MAX_ITEMS:
            raise Invalid(f"workload steps must contain 1 to {_MAX_ITEMS} operations")
        steps: list[Step] = []
        for position, raw in enumerate(raw_steps, start=1):
            try:
                steps.append(_step_from_json(raw))
            except Invalid as exc:
                raise Invalid(f"workload step {position}: {exc}") from exc
        return cls(doc["name"], tuple(steps))

    def to_v1(self) -> dict[str, Any]:
        return {"version": 1, "name": self.name, "steps": [_step_to_json(step) for step in self.steps]}

    @classmethod
    def read(cls, path: Path) -> "Workload":
        return cls.from_v1(decode(read_bytes(path, maximum=_MAX_WORKLOAD_BYTES)))

    def write(self, path: Path) -> None:
        payload = encode(self.to_v1()) + b"\n"
        if len(payload) > _MAX_WORKLOAD_BYTES:
            raise Invalid("encoded workload exceeds the file size limit")
        atomic_write(path, payload)
