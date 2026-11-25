"""Transactional SQLite v1 storage for admissions and their observable results."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sqlite3
from threading import RLock

from ..clock import NS_PER_MS
from ..codec import decode, encode, identifier, integer
from ..errors import Conflict, ErrorCode, Invalid, SailError
from ..request.model import RequestKey
from .model import Failure, Receipt, ResultEnvelope, ResultStatus

_SCHEMA_VERSION = 1
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
    UNIQUE (tenant, request_id)
)
"""
_COLUMNS = "fingerprint, envelope"
_UPDATE = "UPDATE receipts SET status=?, envelope=? WHERE tenant=? AND request_id=? AND status='pending'"


class ReceiptStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        try:
            self._open_v1()
        except BaseException:
            self._db.close()
            raise

    def _open_v1(self) -> None:
        with self._transaction():
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self._db.execute(_SCHEMA)
                self._db.execute("CREATE INDEX receipts_tenant_position ON receipts (tenant, position)")
                self._db.execute("CREATE INDEX receipts_status_position ON receipts (status, position)")
                self._db.execute("PRAGMA user_version = 1")
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
    def _record(row: tuple[str, bytes]) -> Receipt:
        fingerprint, payload = row
        return Receipt(fingerprint, ResultEnvelope.from_json(decode(payload)))

    def _get(self, key: RequestKey) -> Receipt | None:
        row = self._db.execute(
            f"SELECT {_COLUMNS} FROM receipts WHERE tenant=? AND request_id=?",
            (key.tenant, key.request_id),
        ).fetchone()
        return None if row is None else self._record(row)

    def find(self, key: RequestKey) -> Receipt | None:
        with self._lock:
            return self._get(key)

    def result(self, key: RequestKey) -> ResultEnvelope:
        stored = self.find(key)
        if stored is None:
            raise SailError(ErrorCode.NOT_FOUND, f"unknown request {key.tenant}/{key.request_id}")
        return stored.result

    def _insert(self, receipt: Receipt) -> None:
        result = receipt.result
        self._db.execute(
            """
            INSERT INTO receipts (tenant, request_id, model, revision, status, fingerprint, envelope)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.key.tenant, result.key.request_id, result.model.name, result.model.revision,
                result.status.value, receipt.fingerprint, encode(result.to_json()),
            ),
        )

    def insert(self, receipt: Receipt) -> None:
        with self._transaction():
            if self._get(receipt.result.key) is not None:
                raise Conflict("request already has an admission")
            self._insert(receipt)

    def _update_pending(self, result: ResultEnvelope) -> bool:
        response = self._db.execute(
            _UPDATE,
            (result.status.value, encode(result.to_json()), result.key.tenant, result.key.request_id),
        )
        return response.rowcount == 1

    def assign(
        self,
        assignments: Iterable[tuple[RequestKey, int, str, int, str, int]],
    ) -> None:
        with self._transaction():
            for key, attempt, batch_id, slot, device, queue_ns in assignments:
                current = self._get(key)
                if current is None or current.result.status.terminal or current.result.attempt + 1 != attempt:
                    raise Conflict(f"cannot assign stale attempt {attempt} for {key.request_id}")
                result = replace(
                    current.result, attempt=attempt, batch_id=batch_id, slot=slot, device=device,
                    queue_ns=queue_ns,
                )
                if not self._update_pending(result):
                    raise Conflict(f"failed to assign {key.request_id}")

    def settle(
        self,
        result: ResultEnvelope,
        *,
        expected_attempt: int,
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
            if not self._update_pending(result):
                raise Conflict("pending result disappeared during settlement")
            return result, True

    def recover(self, wall_ms: int) -> int:
        """Pending rows belong to the previous process and have no runnable input."""
        with self._transaction():
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM receipts WHERE status='pending' ORDER BY position",
            ).fetchall()
            for row in rows:
                current = self._record(row).result
                age_ns = max(0, wall_ms - current.accepted_at_ms) * NS_PER_MS
                elapsed = max(current.queue_ns or 0, age_ns)
                recovered = replace(
                    current,
                    status=ResultStatus.FAILED,
                    elapsed_ns=elapsed,
                    failure=Failure(ErrorCode.INTERRUPTED, "runtime stopped before settlement", True),
                )
                if not self._update_pending(recovered):
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
        query = "SELECT position, fingerprint, envelope FROM receipts WHERE position > ?"
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
        return tuple((position, self._record((fingerprint, payload))) for position, fingerprint, payload in rows)

    def import_terminal(self, receipts: Iterable[Receipt]) -> int:
        with self._transaction():
            written = 0
            for receipt in receipts:
                if not receipt.result.status.terminal:
                    raise Invalid("receipt archives may only import terminal results")
                current = self._get(receipt.result.key)
                if current is not None:
                    if current != receipt:
                        raise Conflict(f"different receipt already exists for {receipt.result.key.request_id}")
                    continue
                self._insert(receipt)
                written += 1
            return written

    def close(self) -> None:
        with self._lock:
            self._db.close()
