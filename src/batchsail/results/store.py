"""Transactional SQLite storage for admissions, results, and attempt evidence.

The database keeps one row per request: the observable result is unique, while
the attempts that produced it are recorded beside it. Version one databases
and version one receipts remain readable; they simply have no attempt history,
which is reported as unknown rather than invented.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any

from ..clock import NS_PER_MS
from ..codec import decode, encode, identifier, integer
from ..errors import Conflict, ErrorCode, Invalid, SailError
from ..request.model import RequestKey
from .model import AttemptEvidence, Failure, Receipt, ResultEnvelope, ResultStatus

_SCHEMA_VERSION = 2
_SCHEMA = """
CREATE TABLE receipts (
    position INTEGER PRIMARY KEY,
    tenant TEXT NOT NULL,
    request_id TEXT NOT NULL,
    model TEXT NOT NULL,
    revision TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'expired', 'cancelled', 'failed')),
    fingerprint TEXT NOT NULL,
    envelope BLOB NOT NULL,
    version INTEGER NOT NULL DEFAULT 2,
    attempts BLOB,
    UNIQUE (tenant, request_id)
)
"""
_COLUMNS = "fingerprint, envelope, version, attempts"
_UPDATE_PENDING = (
    "UPDATE receipts SET status=?, envelope=?, version=?, attempts=? "
    "WHERE tenant=? AND request_id=? AND status='pending'"
)
_REPLACE_EXISTING = (
    "UPDATE receipts SET envelope=?, version=?, attempts=? WHERE tenant=? AND request_id=?"
)


def _interrupted() -> Failure:
    return Failure(ErrorCode.INTERRUPTED, "runtime stopped before settlement", True)


class ReceiptStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        try:
            self._open()
        except BaseException:
            self._db.close()
            raise

    def _open(self) -> None:
        with self._transaction():
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self._db.execute(_SCHEMA)
                self._db.execute("CREATE INDEX receipts_tenant_position ON receipts (tenant, position)")
                self._db.execute("CREATE INDEX receipts_status_position ON receipts (status, position)")
                self._db.execute("PRAGMA user_version = 2")
            elif version == 1:
                # Existing version one rows have no attempt evidence. Mark them
                # version one so queries and exports report an unknown history.
                self._db.execute("ALTER TABLE receipts ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
                self._db.execute("ALTER TABLE receipts ADD COLUMN attempts BLOB")
                self._db.execute("PRAGMA user_version = 2")
            elif version != _SCHEMA_VERSION:
                raise Invalid(f"receipt database version {version} is unsupported")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = FULL")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @staticmethod
    def _serialize_attempts(attempts: tuple[AttemptEvidence, ...] | None) -> bytes | None:
        if attempts is None:
            return None
        return encode([evidence.to_json() for evidence in attempts])

    @staticmethod
    def _deserialize_attempts(version: int, payload: bytes | None) -> tuple[AttemptEvidence, ...] | None:
        if version == 1:
            return None
        if payload is None:
            raise Invalid("a version two receipt is missing its attempt evidence")
        document = decode(payload)
        if not isinstance(document, list):
            raise Invalid("receipt attempt evidence must be an array")
        return tuple(AttemptEvidence.from_json(item) for item in document)

    @classmethod
    def _record(cls, row: tuple[Any, ...]) -> Receipt:
        fingerprint, payload, version, attempts_payload = row
        return Receipt(
            fingerprint,
            ResultEnvelope.from_json(decode(payload)),
            cls._deserialize_attempts(version, attempts_payload),
            version,
        )

    def _get(self, key: RequestKey) -> Receipt | None:
        row = self._db.execute(
            f"SELECT {_COLUMNS} FROM receipts WHERE tenant=? AND request_id=?",
            (key.tenant, key.request_id),
        ).fetchone()
        return None if row is None else self._record(row)

    def find(self, key: RequestKey) -> Receipt | None:
        with self._lock:
            return self._get(key)

    def receipt(self, key: RequestKey) -> Receipt:
        stored = self.find(key)
        if stored is None:
            raise SailError(ErrorCode.NOT_FOUND, f"unknown request {key.tenant}/{key.request_id}")
        return stored

    def result(self, key: RequestKey) -> ResultEnvelope:
        return self.receipt(key).result

    def query(self, key: RequestKey) -> dict[str, Any]:
        """The stored receipt with an explicit unknown-history marker."""
        return self.receipt(key).query_json()

    def _insert(self, receipt: Receipt) -> None:
        result = receipt.result
        self._db.execute(
            """
            INSERT INTO receipts
                (tenant, request_id, model, revision, status, fingerprint, envelope, version, attempts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.key.tenant, result.key.request_id, result.model.name, result.model.revision,
                result.status.value, receipt.fingerprint, encode(result.to_json()),
                receipt.version, self._serialize_attempts(receipt.attempts),
            ),
        )

    def insert(self, receipt: Receipt) -> None:
        with self._transaction():
            if self._get(receipt.result.key) is not None:
                raise Conflict("request already has an admission")
            self._insert(receipt)

    def _update_pending(
        self,
        result: ResultEnvelope,
        version: int,
        attempts: tuple[AttemptEvidence, ...] | None,
    ) -> bool:
        response = self._db.execute(
            _UPDATE_PENDING,
            (
                result.status.value, encode(result.to_json()), version,
                self._serialize_attempts(attempts), result.key.tenant, result.key.request_id,
            ),
        )
        return response.rowcount == 1

    def assign(
        self,
        assignments: Iterable[tuple[RequestKey, int, str, int, str, int]],
        *,
        wall_ms: int | None = None,
    ) -> None:
        """Record new dispatches, appending one in-flight attempt per request.

        The receipt envelope and its evidence are written in one transaction so
        a restart never sees an assigned attempt without its dispatch record.
        """
        with self._transaction():
            for key, attempt, batch_id, slot, device, queue_ns in assignments:
                current = self._get(key)
                if current is None or current.result.status.terminal or current.result.attempt + 1 != attempt:
                    raise Conflict(f"cannot assign stale attempt {attempt} for {key.request_id}")
                if current.attempts is None:
                    # A legacy pending row may already name an earlier attempt.
                    # Its earlier dispatches are unknowable, so remain unknown.
                    if current.result.attempt:
                        evidence = None
                        version = 1
                    else:
                        evidence = (AttemptEvidence(
                            attempt, batch_id, slot, device, queue_ns, wall_ms,
                        ),)
                        version = 2
                else:
                    if len(current.attempts) != attempt - 1:
                        raise Conflict(f"attempt {attempt} for {key.request_id} is out of sequence")
                    if current.attempts and not current.attempts[-1].concluded:
                        raise Conflict(f"attempt {attempt - 1} for {key.request_id} has not concluded")
                    evidence = (*current.attempts, AttemptEvidence(
                        attempt, batch_id, slot, device, queue_ns, wall_ms,
                    ))
                    version = 2
                result = replace(
                    current.result, attempt=attempt, batch_id=batch_id, slot=slot, device=device,
                    queue_ns=queue_ns,
                )
                if not self._update_pending(result, version, evidence):
                    raise Conflict(f"failed to assign {key.request_id}")

    def conclude_attempt(
        self,
        key: RequestKey,
        attempt: int,
        status: ResultStatus,
        failure: Failure | None,
        *,
        wall_ms: int,
    ) -> bool:
        """Close an in-flight attempt while the request continues (for a retry)."""
        if status == ResultStatus.PENDING:
            raise Invalid("a concluded attempt cannot be pending")
        with self._transaction():
            current = self._get(key)
            if current is None:
                raise SailError(ErrorCode.NOT_FOUND, "cannot conclude a missing admission")
            if current.result.status.terminal or current.result.attempt != attempt:
                return False
            if current.attempts is None or len(current.attempts) != attempt:
                raise Conflict(f"attempt {attempt} for {key.request_id} has no recorded evidence")
            last = current.attempts[-1]
            if last.concluded:
                return False
            finished = replace(
                last, outcome=status, failure=failure,
                finished_at_ms=max(wall_ms, last.dispatched_at_ms or 0),
            )
            attempts = (*current.attempts[:-1], finished)
            if not self._update_pending(current.result, current.version, attempts):
                raise Conflict(f"pending result disappeared while concluding {key.request_id}")
            return True

    def settle(
        self,
        result: ResultEnvelope,
        *,
        expected_attempt: int,
        wall_ms: int | None = None,
    ) -> tuple[ResultEnvelope, bool]:
        if not result.status.terminal:
            raise Invalid("settlement requires a terminal result")
        with self._transaction():
            record = self._get(result.key)
            if record is None:
                raise SailError(ErrorCode.NOT_FOUND, "cannot settle a missing admission")
            current = record.result
            if current.status.terminal or current.attempt != expected_attempt:
                return current, False
            if (
                result.model != current.model
                or result.accepted_at_ms != current.accepted_at_ms
                or result.attempt != current.attempt
                or result.batch_id != current.batch_id
                or result.slot != current.slot
                or result.device != current.device
                or result.queue_ns != current.queue_ns
            ):
                raise Conflict("terminal result does not identify its admission and assignment")
            attempts = record.attempts
            if attempts is not None and result.attempt:
                index = result.attempt - 1
                if index >= len(attempts):
                    raise Conflict("terminal result names an attempt without evidence")
                in_flight = attempts[index]
                if not in_flight.concluded:
                    finish = result.accepted_at_ms if wall_ms is None else wall_ms
                    finish = max(finish, in_flight.dispatched_at_ms or 0)
                    attempts = (
                        *attempts[:index],
                        replace(
                            in_flight, outcome=result.status, failure=result.failure, finished_at_ms=finish,
                        ),
                        *attempts[index + 1:],
                    )
            if not self._update_pending(result, record.version, attempts):
                raise Conflict("pending result disappeared during settlement")
            return result, True

    def recover(self, wall_ms: int) -> int:
        """Pending rows belong to the previous process and have no runnable input.

        Attempts that already concluded keep their known failure; the attempt
        that was in flight when the process stopped is concluded as interrupted.
        Legacy rows without evidence stay explicitly unknown.
        """
        with self._transaction():
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM receipts WHERE status='pending' ORDER BY position",
            ).fetchall()
            for row in rows:
                record = self._record(row)
                current = record.result
                attempts = record.attempts
                version = record.version
                if attempts is not None:
                    concluded = tuple(
                        replace(
                            evidence, outcome=ResultStatus.FAILED, failure=_interrupted(),
                            finished_at_ms=max(wall_ms, evidence.dispatched_at_ms or 0),
                        )
                        if not evidence.concluded else evidence
                        for evidence in attempts
                    )
                    attempts = concluded
                age_ns = max(0, wall_ms - current.accepted_at_ms) * NS_PER_MS
                elapsed = max(current.queue_ns or 0, age_ns)
                recovered = replace(
                    current,
                    status=ResultStatus.FAILED,
                    elapsed_ns=elapsed,
                    failure=_interrupted(),
                )
                if not self._update_pending(recovered, version, attempts):
                    raise RuntimeError("admission disappeared while recovering the result database")
            return len(rows)

    def history(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        tenant: str | None = None,
        status: ResultStatus | None = None,
    ) -> tuple[tuple[int, Receipt], ...]:
        integer(after, "after cursor")
        integer(limit, "history limit", minimum=1, maximum=1000)
        query = f"SELECT position, {_COLUMNS} FROM receipts WHERE position > ?"
        args: list[str | int] = [after]
        if tenant is not None:
            query += " AND tenant = ?"
            args.append(identifier(tenant, "tenant"))
        if status is not None:
            query += " AND status = ?"
            args.append(status.value)
        args.append(limit)
        with self._lock:
            rows = self._db.execute(query + " ORDER BY position LIMIT ?", args).fetchall()
        return tuple(
            (position, self._record((fingerprint, payload, version, attempts)))
            for position, fingerprint, payload, version, attempts in rows
        )

    @staticmethod
    def _merge(existing: Receipt, incoming: Receipt) -> Receipt | None:
        """Combine duplicate terminal receipts without losing known attempts."""
        if existing.fingerprint != incoming.fingerprint:
            raise Conflict("different receipt already exists for this request")
        if existing.result != incoming.result:
            raise Conflict("different receipt already exists for this request")
        if incoming.attempts is None:
            return None
        if existing.attempts is None:
            return existing.with_attempts(incoming.attempts)
        merged: list[AttemptEvidence] = []
        changed = False
        shared = min(len(existing.attempts), len(incoming.attempts))
        for index in range(shared):
            current, candidate = existing.attempts[index], incoming.attempts[index]
            if current == candidate:
                merged.append(current)
                continue
            # The stored copy may hold the in-flight attempt that was later
            # concluded; the incoming copy only adds the exit reason.
            if current.same_dispatch(candidate):
                if candidate.concluded and not current.concluded:
                    merged.append(candidate)
                    changed = True
                else:
                    merged.append(current)
                continue
            raise Conflict("duplicate receipts disagree about a recorded attempt")
        for evidence in existing.attempts[shared:]:
            merged.append(evidence)
        for evidence in incoming.attempts[shared:]:
            merged.append(evidence)
            changed = True
        if not changed:
            return None
        return replace(existing, attempts=tuple(merged))

    def _replace_row(self, receipt: Receipt) -> None:
        result = receipt.result
        self._db.execute(
            _REPLACE_EXISTING,
            (
                encode(result.to_json()), receipt.version,
                self._serialize_attempts(receipt.attempts),
                result.key.tenant, result.key.request_id,
            ),
        )

    def import_terminal(self, receipts: Iterable[Receipt]) -> int:
        with self._transaction():
            written = 0
            for receipt in receipts:
                if not receipt.result.status.terminal:
                    raise Invalid("receipt archives may only import terminal results")
                current = self._get(receipt.result.key)
                if current is None:
                    self._insert(receipt)
                    written += 1
                    continue
                merged = self._merge(current, receipt)
                if merged is not None:
                    self._replace_row(merged)
                    written += 1
            return written

    def close(self) -> None:
        with self._lock:
            self._db.close()
