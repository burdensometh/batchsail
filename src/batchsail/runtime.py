"""The coordinator serializes admission, batching, leases, and settlement."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from threading import Condition, Lock, RLock, Thread
from time import monotonic
from typing import Any

from .batcher import Batch, BatchPlanner, BatchQueues, DispatchPlan
from .batcher.planner import SchedulingBlock
from .clock import Clock, SystemClock
from .codec import finite, identifier, integer
from .config import RuntimeConfig, WorkspaceGuard
from .device import DeviceLedger, DeviceLease, DeviceView, memory_demand
from .device.estimate import CapacityEstimate
from .device.maintenance import TrimPlan, TrimResult
from .errors import Conflict, ErrorCode, Invalid, SailError
from .executor import Backend, BatchExecution, Boundary, BoundaryEvent, Hook, InlineBackend, ProcessBackend, Supervisor
from .executor.faults import no_hook
from .registry import Catalog, ModelKey, ModelStore
from .registry.catalog import CatalogSwitch, RevisionPin, RevisionView
from .registry.model import ModelRevision
from .request import Admission, Request, RequestDeadline, RequestKey
from .request.lifecycle import DeadlineIndex, Ticket
from .results import Failure, ReceiptStore, ResultEnvelope, ResultStatus
from .results.model import Receipt
from .telemetry import Telemetry
from .tensor import Tensor
from .workspace import WorkspaceFormat

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DrainResult:
    completed: bool
    queued: int
    executing_batches: int
    faulted: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "completed": self.completed, "queued": self.queued,
            "executing_batches": self.executing_batches, "faulted": self.faulted,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    generation: int
    models: tuple[RevisionView, ...]
    devices: tuple[DeviceView, ...]
    queued: int
    running: int
    executing_batches: int
    recovered: int
    accepting: bool
    metrics: dict[str, Any]
    blocked: tuple[SchedulingBlock, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "models": [
                {
                    **view.model.key.to_json(),
                    "operator": view.model.revision.operator,
                    "active": view.active,
                    "pins": view.pins,
                    "draining": view.draining,
                }
                for view in self.models
            ],
            "devices": [
                {
                    "name": view.name,
                    "capacity_bytes": view.capacity_bytes,
                    "resident_bytes": view.resident_bytes,
                    "workspace_bytes": view.workspace_bytes,
                    "available_bytes": view.available_bytes,
                    "lease": view.lease_token,
                    "evictions": view.evictions,
                    "resets": view.resets,
                    "residencies": [
                        {**item.key.to_json(), "generation": item.generation, "weight_bytes": item.weight_bytes}
                        for item in view.residencies
                    ],
                }
                for view in self.devices
            ],
            "queued": self.queued,
            "running": self.running,
            "executing_batches": self.executing_batches,
            "recovered": self.recovered,
            "accepting": self.accepting,
            "metrics": self.metrics,
            "blocked": [
                {
                    **item.shape.revision.to_json(),
                    "padded_rows": item.shape.rows,
                    "columns": item.shape.columns,
                    "queued": item.queued,
                    "devices": [
                        {
                            "name": rejection.device,
                            "reason": rejection.reason,
                            "missing_bytes": rejection.missing_bytes,
                            "locked_revisions": [key.to_json() for key in rejection.locked_revisions],
                        }
                        for rejection in item.reasons
                    ],
                }
                for item in self.blocked
            ],
        }


class Runtime:
    def __init__(
        self,
        root: Path,
        config: RuntimeConfig | None = None,
        *,
        clock: Clock | None = None,
        backend: Backend | None = None,
        hook: Hook = no_hook,
    ):
        self.config = config or RuntimeConfig()
        self.clock = clock or SystemClock()
        self._condition = Condition(RLock())
        self._closing = Lock()
        self._workspace = WorkspaceGuard(root)
        self._receipts: ReceiptStore
        try:
            self._catalog = Catalog(ModelStore(self._workspace.root / "models"))
            self._ledger = DeviceLedger(self.config.devices)
            self._queues = BatchQueues(self.config.batching)
            self._planner = BatchPlanner(self._queues, self._ledger, self.config.memory)
            for item in self._catalog.snapshot()[1]:
                self.config.memory.weight_bytes(item.model.revision)
            WorkspaceFormat.ensure(self._workspace.root, self.config.memory.page_bytes)
            self._receipts = ReceiptStore(self._workspace.root / "receipts.sqlite")
            try:
                self._initialize_workers(backend, hook)
            except BaseException:
                self._receipts.close()
                raise
        except BaseException:
            self._workspace.close()
            raise

    def _initialize_workers(self, backend: Backend | None, hook: Hook) -> None:
        self._telemetry = Telemetry(self.config.event_capacity)
        self._recovered = self._receipts.recover(self.clock.wall_ms())
        if self._recovered:
            self._note("admissions_recovered", count=self._recovered)
        self._sequence = 0
        self._batch_sequence = 0
        self._tickets: dict[RequestKey, Ticket] = {}
        self._running: dict[str, Batch] = {}
        self._deadlines = DeadlineIndex()
        self._blocked: tuple[SchedulingBlock, ...] = ()
        self._draining = False
        self._stopping = False
        self._closed = False
        self._fault: Exception | None = None
        devices = tuple(spec.name for spec in self.config.devices)
        self.backend = backend if backend is not None else self._new_backend(devices)
        self._supervisor = Supervisor(self.backend, hook)
        self._pool = ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="batchsail-executor")
        self._scheduler = Thread(target=self._schedule, name="batchsail-scheduler", daemon=True)
        try:
            self._scheduler.start()
        except BaseException:
            self._pool.shutdown()
            self.backend.close()
            raise

    def _new_backend(self, devices: tuple[str, ...]) -> Backend:
        if self.config.backend == "inline":
            return InlineBackend(devices)
        return ProcessBackend(devices, timeout_seconds=self.config.worker_timeout_ms / 1000)

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise SailError(ErrorCode.UNAVAILABLE, "runtime is closed")

    def _require_admission(self) -> None:
        self._require_open()
        if self._fault is not None or self._stopping or self._draining:
            raise SailError(ErrorCode.UNAVAILABLE, "runtime is not accepting work")

    def _note(self, kind: str, **values: Any) -> None:
        self._telemetry.note(kind, self.clock.wall_ms(), **values)

    def install(self, bundle: Path) -> ModelRevision:
        revision, weights = ModelStore.read_bundle(bundle)
        self.config.memory.weight_bytes(revision)
        with self._condition:
            self._require_admission()
            self._catalog.install(revision, weights)
            self._note("model_installed", **revision.key.to_json())
            return revision

    def activate(self, key: ModelKey, *, expected_generation: int | None = None) -> int:
        if expected_generation is not None:
            integer(expected_generation, "expected catalog generation")
        with self._condition:
            self._require_admission()
            revision = self._catalog.get(key).revision
            self._validate_fits(revision, 1)
            before = self._catalog.snapshot()[0]
            after = self._catalog.activate(key, expected_generation=expected_generation)
            if before != after:
                self._note("model_activated", **key.to_json(), generation=after)
                self._condition.notify_all()
            return after

    def switch_many(
        self,
        targets: tuple[ModelKey, ...],
        *,
        expected_generation: int | None = None,
    ) -> CatalogSwitch:
        """Atomically switch several models after validating every target.

        Each target revision must be installed and runnable for one slot on the
        configured devices. Any failure happens before catalog state changes, so a
        rejected switch preserves every active revision, its pins and the generation.
        A successful switch commits the whole target set in one catalog write and
        advances the generation by at most one.
        """
        if expected_generation is not None:
            integer(expected_generation, "expected catalog generation")
        names = [key.name for key in targets]
        if not names:
            raise Invalid("a catalog switch requires at least one target revision")
        if len(set(names)) != len(names):
            raise Invalid("a catalog switch can name each model at most once")
        with self._condition:
            self._require_admission()
            revisions = tuple(self._catalog.get(key).revision for key in targets)
            # Verify device runnability for the complete target set before any
            # catalog write, so one unrunnable revision aborts the whole switch.
            for revision in revisions:
                self._validate_fits(revision, 1)
            before_generation, before_views = self._catalog.snapshot()
            before_active = {
                view.model.key.name: view.model.key for view in before_views if view.active
            }
            result = self._catalog.switch_many(
                targets, expected_generation=expected_generation
            )
            if before_generation != result.generation:
                changed = [
                    key.to_json() for key in targets
                    if result.active.get(key.name) == key and before_active.get(key.name) != key
                ]
                self._note(
                    "catalog_switched", generation=result.generation,
                    targets=[key.to_json() for key in targets], changed=changed,
                )
                self._condition.notify_all()
            return result

    def _validate_fits(self, revision: ModelRevision, rows: int) -> None:
        padded = self.config.batching.padded_rows(rows, revision.input_spec.max_rows)
        demand = memory_demand(self.config.memory, revision, padded_rows=padded, slots=1)
        needed = demand.weight_bytes + demand.workspace_bytes
        if needed > self._ledger.max_capacity:
            raise SailError(ErrorCode.IMPOSSIBLE, f"{revision.key.label} needs {needed} bytes for one slot")

    def estimate(
        self, model: str, *, rows: int, slots: int = 1, revision: str | None = None,
    ) -> CapacityEstimate:
        identifier(model, "model")
        integer(slots, "slots", minimum=1, maximum=self.config.batching.max_slots)
        with self._condition:
            self._require_open()
            installed = (
                self._catalog.active(model) if revision is None
                else self._catalog.get(ModelKey(model, revision))
            )
            spec = installed.revision.input_spec
            integer(rows, "rows", minimum=1, maximum=spec.max_rows)
            padded = self.config.batching.padded_rows(rows, spec.max_rows)
            demand = memory_demand(self.config.memory, installed.revision, padded_rows=padded, slots=slots)
            placement = self._ledger.assess(demand, self._catalog.draining())
            return CapacityEstimate(demand, rows, padded, slots, placement)

    def trim(
        self, *, device: str | None = None, revision: ModelKey | None = None,
    ) -> tuple[TrimResult, ...]:
        with self._condition:
            self._require_admission()
            names = self._ledger.names()
            if device is not None:
                identifier(device, "device")
                if device not in names:
                    raise SailError(ErrorCode.NOT_FOUND, f"unknown device {device}")
                names = (device,)
            if revision is not None:
                self._catalog.get(revision)
            protected = self._catalog.protected_residencies()
            outcomes: list[TrimResult] = []
            for name in names:
                plan = self._ledger.plan_trim(name, protected, target=revision)
                if plan is None:
                    outcomes.append(TrimResult(name, "busy"))
                    continue
                if not plan.victims:
                    outcomes.append(TrimResult(name, "no_eligible_residency", remaining=plan.keep))
                    continue
                outcomes.append(self._prune_idle(plan))
            self._condition.notify_all()
            return tuple(outcomes)

    def _prune_idle(self, plan: TrimPlan) -> TrimResult:
        try:
            retained = self.backend.prune(plan.device, plan.keep)
            removed = self._ledger.apply_trim(plan, retained)
        except Exception as exc:
            try:
                self.backend.reset(plan.device)
                removed = self._ledger.reset_idle(plan.device)
            except Exception as reset_error:
                self._set_fault(reset_error)
                return TrimResult(plan.device, "failed", detail="device reset failed")
            self._note("device_cache_reset", device=plan.device, reason=type(exc).__name__)
            return TrimResult(plan.device, "reset", removed, detail=type(exc).__name__)
        self._note(
            "device_cache_trimmed", device=plan.device,
            removed=[item.key.to_json() for item in removed],
            removed_bytes=sum(item.weight_bytes for item in removed),
        )
        return TrimResult(plan.device, "trimmed", removed, retained)

    def _check_retry(self, request: Request) -> None:
        if request.retry_of is None:
            return
        original = self._receipts.result(RequestKey(request.key.tenant, request.retry_of))
        if original.model.name != request.model:
            raise Invalid("retry must use the same model as the original request")
        if original.failure is None or not original.failure.retryable:
            raise Conflict("original request is not a retryable failure")

    def submit(self, request: Request) -> Admission:
        fingerprint = request.fingerprint
        with self._condition:
            self._require_open()
            existing = self._receipts.find(request.key)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise Conflict("request key has already been used with different input or options")
                return existing.result.admission(reused=True)
            self._require_admission()
            self._check_retry(request)
            if len(self._tickets) >= self.config.batching.max_pending:
                raise SailError(ErrorCode.QUEUE_FULL, "maximum pending requests reached")
            pin = self._catalog.pin(request.model)
            try:
                return self._admit(request, fingerprint, pin)
            except BaseException:
                if request.key not in self._tickets:
                    self._catalog.release(pin)
                raise

    def _admit(self, request: Request, fingerprint: str, pin: RevisionPin) -> Admission:
        revision = pin.model.revision
        revision.input_spec.check(request.tensor, "request input")
        self._validate_fits(revision, request.tensor.rows)
        now_ns = self.clock.now_ns()
        deadline = RequestDeadline.from_budget(now_ns, request.deadline_ms)
        self._sequence += 1
        ticket = Ticket(request, pin, deadline, self._sequence)
        result = ResultEnvelope(request.key, revision.key, ResultStatus.PENDING, self.clock.wall_ms())
        self._receipts.insert(Receipt(fingerprint, result))
        self._tickets[ticket.key] = ticket
        self._note("request_accepted", **request.key.to_json(), **revision.key.to_json())
        if deadline.expired(now_ns):
            self._expire(ticket, now_ns)
        else:
            self._deadlines.add(ticket)
            self._queues.add(ticket, now_ns)
        self._condition.notify_all()
        return result.admission(reused=False)

    def result(self, key: RequestKey) -> ResultEnvelope:
        with self._condition:
            self._require_open()
            return self._receipts.result(key)

    def wait(self, key: RequestKey, *, timeout: float | None = None) -> ResultEnvelope:
        end = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                self._require_open()
                result = self._receipts.result(key)
                if result.status.terminal:
                    return result
                if self._fault is not None:
                    raise SailError(ErrorCode.UNAVAILABLE, "runtime could not complete the request")
                remaining = None if end is None else end - monotonic()
                if remaining is not None and remaining <= 0:
                    return result
                self._condition.wait(remaining)

    def cancel(self, key: RequestKey) -> ResultEnvelope:
        with self._condition:
            self._require_open()
            current = self._receipts.result(key)
            if current.status.terminal:
                return current
            if self._fault is not None or self._stopping:
                raise SailError(ErrorCode.UNAVAILABLE, "runtime cannot cancel work")
            ticket = self._tickets[key]
            now_ns = self.clock.now_ns()
            if ticket.deadline.expired(now_ns):
                return self._expire(ticket, now_ns)
            return self._settle(
                ticket, ResultStatus.CANCELLED, now_ns,
                failure=Failure(ErrorCode.CANCELLED, "request cancelled"),
            )

    def _expire(self, ticket: Ticket, now_ns: int) -> ResultEnvelope:
        return self._settle(
            ticket, ResultStatus.EXPIRED, now_ns,
            failure=Failure(ErrorCode.DEADLINE, "request deadline elapsed", True),
        )

    def _settle(
        self,
        ticket: Ticket,
        status: ResultStatus,
        now_ns: int,
        *,
        failure: Failure | None = None,
        output: Tensor | None = None,
    ) -> ResultEnvelope:
        current = self._receipts.result(ticket.key)
        if current.status.terminal:
            return current
        target = replace(
            current, status=status, failure=failure, output=output,
            elapsed_ns=max(current.queue_ns or 0, now_ns - ticket.deadline.accepted_ns),
        )
        result, changed = self._receipts.settle(target, expected_attempt=ticket.attempt)
        if not changed:
            raise RuntimeError("coordinator attempted to settle a stale live ticket")
        ticket.terminal = True
        self._deadlines.discard(ticket)
        self._telemetry.settlement(result, self.clock.wall_ms())
        if ticket.batch_id is None:
            self._retire(ticket)
        self._condition.notify_all()
        return result

    def _retire(self, ticket: Ticket) -> None:
        self._queues.discard(ticket.key)
        del self._tickets[ticket.key]
        self._catalog.release(ticket.pin)

    def pulse(self) -> None:
        """Wake scheduling, for example after advancing an injected clock."""
        with self._condition:
            self._condition.notify_all()

    def _set_fault(self, exc: Exception) -> None:
        if self._fault is None:
            self._fault = exc
            _LOG.error("runtime has stopped admitting work", exc_info=(type(exc), exc, exc.__traceback__))
            self._note("runtime_fault", exception=type(exc).__name__)
        self._condition.notify_all()

    def _schedule(self) -> None:
        with self._condition:
            try:
                while not self._stopping and self._fault is None:
                    now_ns = self.clock.now_ns()
                    for key in self._deadlines.due(now_ns):
                        self._expire(self._tickets[key], now_ns)
                    protected = self._catalog.draining()
                    plan, ready_ns, self._blocked = self._planner.select(now_ns, protected, flush=self._draining)
                    if plan is not None:
                        self._dispatch(plan, protected, now_ns)
                        # Allow cancellation, admission and completed workers to interleave
                        # with a burst of dispatches on multiple devices.
                        self._condition.wait(0.000001)
                        continue
                    self._wait_for_progress(ready_ns)
            except Exception as exc:
                self._set_fault(exc)

    def _wait_for_progress(self, ready_ns: int | None) -> None:
        deadline = self._deadlines.next_ns()
        if deadline is not None:
            ready_ns = deadline if ready_ns is None else min(ready_ns, deadline)
        remaining = None if ready_ns is None else max(0, ready_ns - self.clock.now_ns()) / 1e9
        self._condition.wait(remaining)

    def _dispatch(self, plan: DispatchPlan, protected: frozenset[ModelKey], now_ns: int) -> None:
        self._batch_sequence += 1
        batch = Batch.from_tickets(f"batch-{self._batch_sequence}", plan.shape, plan.tickets)
        self._receipts.assign(
            (
                slot.key, slot.attempt, batch.id, slot.index, plan.placement.device,
                now_ns - ticket.deadline.accepted_ns,
            )
            for slot, ticket in zip(batch.slots, plan.tickets, strict=True)
        )
        self._queues.commit(plan.shape, plan.tickets)
        for ticket, slot in zip(plan.tickets, batch.slots, strict=True):
            ticket.attempt = slot.attempt

        try:
            self._supervisor.hook(BoundaryEvent(Boundary.DEQUEUED, batch, None))
        except SailError as exc:
            failure = Failure(exc.code, str(exc)[:1024] or exc.code.value, True)
            for ticket in plan.tickets:
                self._settle(ticket, ResultStatus.FAILED, self.clock.now_ns(), failure=failure)
            return

        lease = self._ledger.acquire(plan.placement, protected)
        for ticket in plan.tickets:
            ticket.batch_id = batch.id
        self._running[batch.id] = batch
        self._telemetry.dispatch(
            self.clock.wall_ms(), size=len(batch.slots), batch_id=batch.id, **lease.key.to_json(),
            device=lease.device, lease=lease.token, residency=lease.residency.generation,
            cached=plan.placement.cached, evicted=[victim.key.to_json() for victim in plan.placement.victims],
            padded_rows=plan.shape.rows, workspace_bytes=lease.workspace_bytes,
        )
        try:
            self._pool.submit(self._execute, batch, lease)
        except Exception:
            self._ledger.release(lease, reset=True)
            del self._running[batch.id]
            raise

    def _execute(self, batch: Batch, lease: DeviceLease) -> None:
        execution = self._supervisor.run(batch, lease)
        with self._condition:
            now_ns = self.clock.now_ns()
            try:
                self._ledger.release(lease, reset=execution.reset)
                self._telemetry.worker(execution.backend_ns, execution.kernel)
                self._note(
                    "lease_released", batch_id=batch.id, device=lease.device, lease=lease.token,
                    reset=execution.reset, workspace_bytes=lease.workspace_bytes,
                )
                if not execution.device_usable:
                    self._set_fault(RuntimeError(f"device {lease.device} could not be reset"))
                for slot in batch.slots:
                    self._complete_slot(batch, slot.index, execution, now_ns)
            except Exception as exc:
                self._set_fault(exc)
            finally:
                del self._running[batch.id]
                self._condition.notify_all()

    def _complete_slot(self, batch: Batch, index: int, execution: BatchExecution, now_ns: int) -> None:
        slot = batch.slots[index]
        ticket = self._tickets.get(slot.key)
        if ticket is None or (ticket.sequence, ticket.attempt, ticket.batch_id) != (
            slot.sequence, slot.attempt, batch.id,
        ):
            raise RuntimeError("completion does not match the live slot identity")
        ticket.batch_id = None
        if ticket.terminal:
            self._retire(ticket)
            return
        if ticket.deadline.expired(now_ns):
            self._expire(ticket, now_ns)
            return
        failure = execution.failure
        if failure is None:
            self._settle(ticket, ResultStatus.SUCCEEDED, now_ns, output=execution.outputs[index])
        elif self._can_retry(ticket, failure):
            self._queues.add(ticket, now_ns)
            self._note(
                "request_requeued", **ticket.key.to_json(), **batch.model.key.to_json(),
                attempt=ticket.attempt, failure=failure.code.value,
            )
        else:
            self._settle(ticket, ResultStatus.FAILED, now_ns, failure=failure)

    def _can_retry(self, ticket: Ticket, failure: Failure) -> bool:
        return (
            failure.code == ErrorCode.WORKER_LOST
            and ticket.attempt < self.config.batching.max_attempts
            and not self._stopping
            and self._fault is None
        )

    def snapshot(self) -> RuntimeSnapshot:
        with self._condition:
            self._require_open()
            generation, models = self._catalog.snapshot()
            return RuntimeSnapshot(
                generation, models, self._ledger.snapshot(), len(self._queues),
                sum(len(batch.slots) for batch in self._running.values()), len(self._running),
                self._recovered, not self._draining and not self._stopping and self._fault is None,
                self._telemetry.snapshot(), self._blocked,
            )

    def events(self, *, after: int = 0, limit: int = 100) -> dict[str, Any]:
        integer(after, "after cursor")
        integer(limit, "event limit", minimum=1, maximum=1000)
        with self._condition:
            self._require_open()
            return self._telemetry.page(after=after, limit=limit)

    def history(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        tenant: str | None = None,
        status: ResultStatus | None = None,
    ) -> dict[str, Any]:
        with self._condition:
            self._require_open()
            page = self._receipts.history(after=after, limit=limit, tenant=tenant, status=status)
        return {
            "results": [receipt.result.to_json() for _, receipt in page],
            "next": page[-1][0] if page else after,
        }

    def drain(self, *, timeout: float) -> DrainResult:
        """Stop new admissions and give already accepted batches time to finish."""
        seconds = finite(timeout, "drain timeout")
        if not 0 <= seconds <= 600:
            raise Invalid("drain timeout must be from 0 to 600 seconds")
        end = monotonic() + seconds
        with self._condition:
            self._require_open()
            if not self._draining:
                self._draining = True
                self._note("runtime_draining", queued=len(self._queues), executing=len(self._running))
                self._condition.notify_all()
            while self._tickets and self._fault is None and not self._stopping:
                remaining = end - monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return DrainResult(
                not self._tickets and not self._running and self._fault is None,
                len(self._queues), len(self._running), self._fault is not None,
            )

    def close(self) -> None:
        with self._closing:
            with self._condition:
                if self._closed:
                    return
                self._stopping = True
                try:
                    now_ns = self.clock.now_ns()
                    for ticket in tuple(self._tickets.values()):
                        if not ticket.terminal:
                            self._settle(
                                ticket, ResultStatus.FAILED, now_ns,
                                failure=Failure(ErrorCode.INTERRUPTED, "runtime is shutting down", True),
                            )
                except Exception as exc:
                    self._set_fault(exc)
                self._condition.notify_all()
            self._scheduler.join()
            self._pool.shutdown(wait=True)
            try:
                self.backend.close()
            finally:
                with self._condition:
                    self._closed = True
                    try:
                        self._receipts.close()
                    finally:
                        self._workspace.close()
                        self._condition.notify_all()
