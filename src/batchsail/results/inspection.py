"""Read-only checks of SQLite receipt indexes and the envelopes they describe."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any

from ..codec import decode
from ..errors import Invalid
from ..registry.model import InstalledModel, ModelKey
from .model import AttemptEvidence, Receipt, ResultEnvelope, ResultStatus

_ISSUE_LIMIT = 50
_SCAN_QUERY = """
SELECT position, tenant, request_id, model, revision, status, fingerprint, envelope
FROM receipts ORDER BY position
"""
_SUPPORTED_VERSIONS = (1, 2)


@dataclass(slots=True)
class ReceiptAudit:
    present: bool
    sqlite_version: int | None = None
    records: int = 0
    decoded: int = 0
    pending: int = 0
    legacy_unknown_attempts: int = 0
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
            "legacy_unknown_attempts": self.legacy_unknown_attempts,
            "unlinked": dict(sorted(self.unlinked.items())),
            "by_status": dict(sorted(self.by_status.items())),
            "by_model": {
                label: dict(sorted(counts.items())) for label, counts in sorted(self.by_model.items())
            },
            "issue_count": self.issue_count,
            "issues": self.issues,
        }


def _read_attempts(
    audit: ReceiptAudit, raw: bytes | None, position: int,
) -> tuple[AttemptEvidence, ...] | None:
    if raw is None:
        return None
    try:
        document = decode(raw)
        if not isinstance(document, list):
            raise Invalid("attempt evidence must be an array")
        return tuple(AttemptEvidence.from_json(item) for item in document)
    except (Invalid, TypeError) as exc:
        audit.issue("invalid_attempts", str(exc), position=position)
        return None


def _audit_row(
    audit: ReceiptAudit,
    row: tuple[int, str, str, str, str, str, str, bytes, int, bytes | None],
    models: dict[ModelKey, InstalledModel],
) -> None:
    position, tenant, request_id, model, revision, status, fingerprint, raw, version, attempts_raw = row
    audit.records += 1
    try:
        result = ResultEnvelope.from_json(decode(raw))
    except (Invalid, TypeError) as exc:
        audit.issue("invalid_envelope", str(exc), position=position)
        return
    attempts = _read_attempts(audit, attempts_raw, position)
    if version not in _SUPPORTED_VERSIONS:
        audit.issue("invalid_version", f"receipt version {version} is unsupported", position=position)
        return
    if version == 1:
        if attempts_raw is not None:
            audit.issue("legacy_with_attempts", "a version one row carries attempt evidence", position=position)
        audit.legacy_unknown_attempts += 1
    elif attempts_raw is None:
        audit.issue("missing_attempts", "a version two row has no attempt evidence", position=position)
    elif attempts is not None:
        try:
            Receipt(fingerprint, result, attempts, 2)
        except Invalid as exc:
            audit.issue("invalid_attempts", str(exc), position=position)
    audit.decoded += 1
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


def inspect_receipts(path: Path, models: dict[ModelKey, InstalledModel]) -> ReceiptAudit:
    audit = ReceiptAudit(path.exists())
    if not audit.present:
        return audit
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        audit.sqlite_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if audit.sqlite_version not in _SUPPORTED_VERSIONS:
            audit.issue(
                "unsupported_schema",
                f"expected SQLite user_version 1 or 2, got {audit.sqlite_version}",
            )
            return audit
        for (result,) in connection.execute("PRAGMA quick_check(50)"):
            if result != "ok":
                audit.issue("sqlite_integrity", result)
        if audit.sqlite_version == 1:
            for row in connection.execute(_SCAN_QUERY):
                _audit_row(audit, (*row, 1, None), models)
        else:
            for row in connection.execute(
                """
                SELECT position, tenant, request_id, model, revision, status, fingerprint,
                       envelope, version, attempts
                FROM receipts ORDER BY position
                """
            ):
                _audit_row(audit, row, models)
    except sqlite3.Error as exc:
        audit.issue("sqlite_error", str(exc))
    finally:
        if connection is not None:
            connection.close()
    return audit

