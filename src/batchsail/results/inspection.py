"""Read-only checks of v1 SQLite indexes and the envelopes they describe."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any

from ..codec import decode
from ..errors import Invalid
from ..registry.model import InstalledModel, ModelKey
from .model import Receipt, ResultStatus

_ISSUE_LIMIT = 50
_SCHEMA_VERSION = 2
_SCAN_QUERY = """
SELECT position, tenant, request_id, model, revision, status, fingerprint, envelope
FROM receipts ORDER BY position
"""


@dataclass(slots=True)
class ReceiptAudit:
    present: bool
    sqlite_version: int | None = None
    records: int = 0
    decoded: int = 0
    pending: int = 0
    terminal_logged: int = 0
    unlinked: Counter[str] = field(default_factory=Counter)
    by_status: Counter[str] = field(default_factory=Counter)
    by_model: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    issues: list[dict[str, Any]] = field(default_factory=list)
    issue_count: int = 0

    def issue(self, code: str, message: str, *, position: int | None = None) -> None:
        self.issue_count += 1
        if len(self.issues) < _ISSUE_LIMIT:
            self.issues.append({"code": code, "message": message, "position": position})

    def to_json(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "sqlite_version": self.sqlite_version,
            "records": self.records,
            "decoded": self.decoded,
            "pending": self.pending,
            "terminal_logged": self.terminal_logged,
            "unlinked": dict(sorted(self.unlinked.items())),
            "by_status": dict(sorted(self.by_status.items())),
            "by_model": {
                label: dict(sorted(counts.items())) for label, counts in sorted(self.by_model.items())
            },
            "issue_count": self.issue_count,
            "issues": self.issues,
        }


def _audit_row(
    audit: ReceiptAudit,
    row: tuple[int, str, str, str, str, str, str, bytes],
    models: dict[ModelKey, InstalledModel],
) -> None:
    position, tenant, request_id, model, revision, status, fingerprint, raw = row
    audit.records += 1
    try:
        receipt = Receipt.from_v1({"version": 1, "fingerprint": fingerprint, "result": decode(raw)})
    except (Invalid, TypeError) as exc:
        audit.issue("invalid_envelope", str(exc), position=position)
        return
    audit.decoded += 1
    result = receipt.result
    identity = (result.key.tenant, result.key.request_id, result.model.name, result.model.revision, result.status.value)
    if identity != (tenant, request_id, model, revision, status):
        audit.issue("index_mismatch", "indexed identity or status differs from envelope", position=position)
    audit.by_status[result.status.value] += 1
    audit.by_model[result.model.label][result.status.value] += 1
    if result.status == ResultStatus.PENDING:
        audit.pending += 1
    installed = models.get(result.model)
    if installed is None:
        # Imported terminal history may legitimately name a revision installed
        # on a different server. Only live admissions rely on a local revision.
        audit.unlinked[result.model.label] += 1
        if result.status == ResultStatus.PENDING:
            audit.issue("pending_revision_missing", "pending admission names a missing model", position=position)
    elif result.output is not None:
        try:
            installed.revision.output_spec.check(result.output, "historical output")
        except Invalid as exc:
            audit.issue("output_spec_mismatch", str(exc), position=position)


def _audit_terminal_log(connection: sqlite3.Connection, audit: ReceiptAudit) -> None:
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "terminal_log" not in tables:
        audit.issue("terminal_log_missing", "durable terminal stream table is absent")
        return
    rows = connection.execute(
        "SELECT seq, tenant, request_id FROM terminal_log ORDER BY seq",
    ).fetchall()
    audit.terminal_logged = len(rows)
    for expected, (seq, _tenant, _request_id) in enumerate(rows, start=1):
        if seq != expected:
            audit.issue("terminal_log_gap", "terminal stream positions must be gap-free", position=seq)
            break
    mismatches = connection.execute(
        """
        SELECT COUNT(*) FROM terminal_log
        JOIN receipts
          ON receipts.tenant = terminal_log.tenant AND receipts.request_id = terminal_log.request_id
        WHERE receipts.status = 'pending'
        """,
    ).fetchone()[0]
    if mismatches:
        audit.issue("terminal_log_pending", "terminal stream references a pending receipt")
    unlinked = connection.execute(
        """
        SELECT COUNT(*) FROM terminal_log
        LEFT JOIN receipts
          ON receipts.tenant = terminal_log.tenant AND receipts.request_id = terminal_log.request_id
        WHERE receipts.position IS NULL
        """,
    ).fetchone()[0]
    if unlinked:
        audit.issue("terminal_log_orphan", "terminal stream references a missing receipt")
    terminal_rows = connection.execute(
        "SELECT COUNT(*) FROM receipts WHERE status != 'pending'",
    ).fetchone()[0]
    if terminal_rows != len(rows):
        audit.issue(
            "terminal_log_drift",
            f"terminal receipts ({terminal_rows}) do not match stream positions ({len(rows)})",
        )


def inspect_receipts(path: Path, models: dict[ModelKey, InstalledModel]) -> ReceiptAudit:
    audit = ReceiptAudit(path.exists())
    if not audit.present:
        return audit
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        audit.sqlite_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if audit.sqlite_version != _SCHEMA_VERSION:
            audit.issue(
                "unsupported_schema",
                f"expected SQLite user_version {_SCHEMA_VERSION}, got {audit.sqlite_version}",
            )
            return audit
        for (result,) in connection.execute("PRAGMA quick_check(50)"):
            if result != "ok":
                audit.issue("sqlite_integrity", result)
        for row in connection.execute(_SCAN_QUERY):
            _audit_row(audit, row, models)
        _audit_terminal_log(connection, audit)
    except sqlite3.Error as exc:
        audit.issue("sqlite_error", str(exc))
    finally:
        if connection is not None:
            connection.close()
    return audit
