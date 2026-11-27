"""Transactional SQLite v1 storage for admissions and their observable results."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any

from ..clock import NS_PER_MS
from ..codec import decode, encode, identifier, integer
from ..errors import Conflict, ErrorCode, Invalid, SailError
from ..request.model import RequestKey
from .model import Failure, Receipt, ResultEnvelope, ResultStatus

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
    UNIQUE (tenant, request_id)
)
"""
_TERMINAL_SCHEMA = """
CREATE TABLE terminal_log (
    seq INTEGER PRIMARY KEY,
    tenant TEXT NOT NULL,
    request_id TEXT NOT NULL,
    UNIQUE (tenant, request_id)
)
"""
_COLUMNS = "fingerprint, envelope"
_UPDATE = "UPDATE receipts SET status=?, envelope=? WHERE tenant=? AND request_id=? AND status='pending'"
# The durable resumable cursor: one gap-free position per terminal settlement,
# assigned inside the same transaction that persists the terminal envelope.
_APPEND_TERMINAL = """
INSERT INTO terminal_log (seq, tenant, request_id)
SELECT COALESCE(MAX(seq), 0) + 1, ?, ? FROM terminal_log
"""
_TERMINAL_PAGE = """
SELECT terminal_log.seq, receipts.fingerprint, receipts.envelope
FROM terminal_log JOIN receipts
  ON receipts.tenant = terminal_log.tenant AND receipts.request_id = terminal_log.request_id
WHERE terminal_log.seq > ?
ORDER BY terminal_log.seq
LIMIT ?
"""


@dataclass(frozen=True, slots=True)
class TerminalPage:
    """One page of the durable terminal stream, ordered by settlement position."""

    entries: tuple[tuple[int, ResultEnvelope], ...]
    next: int
    latest: int

    def to_json(self) -> dict[str, Any]:
        return {
            "results": [
                {**result.to_json(), "terminal_position": seq} for seq, result in self.entries
            ],
            "next": self.next,
            "latest": self.latest,
        }


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
                self._db.execute(_TERMINAL_SCHEMA)
                self._db.execute("PRAGMA user_version = 2")
            elif version == 1:
                self._migrate_v1()
            elif version != _SCHEMA_VERSION:
                raise Invalid(f"receipt database version {version} is unsupported")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = FULL")

    def _migrate_v1(self) -> None:
        """Backfill the terminal log from receipts already persisted by v1.

        Admission ``position`` is the only durable order available for rows
        terminal before the upgrade; the two sequences agree for every row
        settled by version two and later.
        """
        self._db.execute(_TERMINAL_SCHEMA)
        self._db.execute(
            """
            INSERT INTO terminal_log (seq, tenant, request_id)
            SELECT ROW_NUMBER() OVER (ORDER BY position), tenant, request_id
            FROM receipts WHERE status != 'pending'
            """,
        )
        self._db.execute("PRAGMA user_version = 2")

    def _append_terminal(self, result: ResultEnvelope) -> None:
        # The unique (tenant, request_id) constraint makes the append idempotent
        # for a given admission; settlement only reaches here on the single
        # pending-to-terminal transition, so each terminal result takes one seq.
        self._db.execute(_APPEND_TERMINAL, (result.key.tenant, result.key.request_id))

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
            self._append_terminal(result)
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
                self._append_terminal(recovered)
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
                self._append_terminal(receipt.result)
                written += 1
            return written

    def terminal_latest(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT MAX(seq) FROM terminal_log").fetchone()
        return 0 if row is None or row[0] is None else row[0]

    def terminal_page(self, *, after: int = 0, limit: int = 100) -> TerminalPage:
        integer(after, "after cursor")
        integer(limit, "terminal limit", minimum=1, maximum=1000)
        with self._lock:
            rows = self._db.execute(_TERMINAL_PAGE, (after, limit)).fetchall()
            latest = self.terminal_latest()
        entries = tuple((seq, self._record((fingerprint, payload)).result) for seq, fingerprint, payload in rows)
        return TerminalPage(entries, entries[-1][0] if entries else after, latest)

    def close(self) -> None:
        with self._lock:
            self._db.close()
