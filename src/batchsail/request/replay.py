"""Replay scheduled workloads against a local runtime or the HTTP client."""

from dataclasses import dataclass
from time import monotonic_ns, sleep
from typing import Any, Callable, Protocol

from ..clock import NS_PER_MS
from ..codec import finite
from ..errors import ErrorCode, Invalid, SailError
from ..registry.model import ModelKey
from ..results.model import ResultEnvelope
from ..results.report import WorkloadStatistics
from ..tensor import equivalent
from .model import Admission, Request, RequestKey
from .workload import Activate, Cancel, Step, Submit, Workload


class Endpoint(Protocol):
    def submit(self, request: Request) -> Admission: ...
    def result(self, key: RequestKey) -> ResultEnvelope: ...
    def wait(self, key: RequestKey, *, timeout: float | None = None) -> ResultEnvelope: ...
    def cancel(self, key: RequestKey) -> ResultEnvelope: ...
    def activate(self, key: ModelKey, *, expected_generation: int | None = None) -> int: ...


@dataclass(frozen=True, slots=True)
class ReplayIssue:
    stage: str
    code: str
    message: str
    at_step: int
    key: RequestKey | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "step": self.at_step,
            "key": None if self.key is None else self.key.to_json(),
        }


@dataclass(frozen=True, slots=True)
class StepReport:
    index: int
    operation: str
    at_ms: int
    started_ns: int
    duration_ns: int
    details: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.index,
            "operation": self.operation,
            "at_ms": self.at_ms,
            "started_ns": self.started_ns,
            "lag_ns": max(0, self.started_ns - self.at_ms * NS_PER_MS),
            "duration_ns": self.duration_ns,
            **self.details,
        }


@dataclass(frozen=True, slots=True)
class ReplayReport:
    name: str
    scheduled_steps: int
    elapsed_ns: int
    unsettled: tuple[RequestKey, ...]
    issues: tuple[ReplayIssue, ...]
    operations: tuple[StepReport, ...]
    statistics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return not self.issues and not self.unsettled

    def to_json(self) -> dict[str, Any]:
        return {
            "workload": self.name,
            "passed": self.passed,
            "scheduled_steps": self.scheduled_steps,
            "elapsed_ns": self.elapsed_ns,
            "unsettled": [key.to_json() for key in self.unsettled],
            "issues": [issue.to_json() for issue in self.issues],
            "operations": [step.to_json() for step in self.operations],
            "statistics": self.statistics,
        }


@dataclass(frozen=True, slots=True)
class _Submission:
    index: int
    step: Submit
    admission: Admission


def _op_name(step: Step) -> str:
    if isinstance(step, Submit):
        return "submit"
    if isinstance(step, Cancel):
        return "cancel"
    return "activate"


def _op_key(step: Step) -> RequestKey | None:
    if isinstance(step, Submit):
        return step.request.key
    if isinstance(step, Cancel):
        return step.key
    return None


class ReplayRunner:
    def __init__(
        self,
        endpoint: Endpoint,
        *,
        now_ns: Callable[[], int] = monotonic_ns,
        pause: Callable[[float], None] = sleep,
    ):
        self.endpoint = endpoint
        self._now_ns = now_ns
        self._pause = pause

    def run(self, workload: Workload, *, settle_seconds: float = 30) -> ReplayReport:
        settle = finite(settle_seconds, "settle timeout")
        if not 0 < settle <= 600:
            raise Invalid("settle timeout must be positive and at most 600 seconds")
        started = self._now_ns()
        operations: list[StepReport] = []
        issues: list[ReplayIssue] = []
        admissions: list[_Submission] = []
        for index, step in enumerate(workload.steps, start=1):
            instant = started + step.at_ms * NS_PER_MS
            self._sleep_until(instant)
            since = self._now_ns()
            name = _op_name(step)
            try:
                details = self._apply(index, step, admissions)
            except SailError as exc:
                issues.append(ReplayIssue(name, exc.code.value, str(exc), index, _op_key(step)))
                details = {"error": exc.to_json()}
            operations.append(StepReport(index, name, step.at_ms, since - started, self._now_ns() - since, details))

        deadline = self._now_ns() + int(settle * 1e9)
        statistics = WorkloadStatistics()
        unsettled = self._collect(admissions, deadline, statistics, issues)
        return ReplayReport(
            workload.name, len(workload.steps), self._now_ns() - started,
            tuple(unsettled), tuple(issues), tuple(operations), statistics.snapshot(),
        )

    def _sleep_until(self, instant: int) -> None:
        while (remaining := instant - self._now_ns()) > 0:
            self._pause(remaining / 1e9)

    def _apply(self, index: int, step: Step, admissions: list[_Submission]) -> dict[str, Any]:
        if isinstance(step, Submit):
            admission = self.endpoint.submit(step.request)
            admissions.append(_Submission(index, step, admission))
            return {"admission": admission.to_json()}
        if isinstance(step, Cancel):
            result = self.endpoint.cancel(step.key)
            return {"key": result.key.to_json(), "status": result.status.value}
        generation = self.endpoint.activate(step.key, expected_generation=step.expected_generation)
        return {"revision": step.key.to_json(), "generation": generation}

    def _collect(
        self,
        admissions: list[_Submission],
        deadline: int,
        statistics: WorkloadStatistics,
        issues: list[ReplayIssue],
    ) -> list[RequestKey]:
        pending: list[RequestKey] = []
        for submission in admissions:
            key = submission.admission.key
            remaining = (deadline - self._now_ns()) / 1e9
            try:
                if remaining <= 0:
                    result = self.endpoint.result(key)
                else:
                    result = self.endpoint.wait(key, timeout=remaining)
            except SailError as exc:
                issues.append(ReplayIssue("collect", exc.code.value, str(exc), submission.index, key))
                continue
            statistics.add(result)
            if not result.status.terminal:
                pending.append(key)
                continue
            issues.extend(self._check(submission, result))
        return pending

    @staticmethod
    def _check(submission: _Submission, result: ResultEnvelope) -> list[ReplayIssue]:
        problems: list[ReplayIssue] = []
        index = submission.index
        key = submission.admission.key

        def note(message: str) -> None:
            problems.append(ReplayIssue("check", ErrorCode.BAD_OUTPUT.value, message, index, key))

        if (result.key, result.model.name) != (key, submission.step.request.model):
            note("result identity does not match request identity")
        if result.model.revision != submission.admission.revision:
            note("result revision does not match admission revision")
        expected = submission.step.expect
        if expected is not None:
            if result.status != expected.status:
                note(f"status {result.status.value} differs from expected {expected.status.value}")
            if expected.revision is not None and expected.revision != result.model.revision:
                note(f"revision {result.model.revision} differs from expected {expected.revision}")
            if expected.output is not None and (
                result.output is None or not equivalent(result.output, expected.output)
            ):
                note("result tensor differs from the expected tensor")
        return problems
