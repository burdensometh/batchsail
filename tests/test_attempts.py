"""Attempt evidence, version-one compatibility, and duplicate consolidation."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from batchsail.clock import ManualClock
from batchsail.errors import Conflict, ErrorCode
from batchsail.executor import Boundary, ProcessBackend
from batchsail.registry import ModelKey
from batchsail.request import RequestKey
from batchsail.results import (
    AttemptEvidence,
    Failure,
    Receipt,
    ReceiptStore,
    ResultEnvelope,
    ResultStatus,
)
from batchsail.results.archive import export_archive, import_archive
from batchsail.runtime import Runtime

from support import Gate, ROOT, config, install, request, until


def _evidence(attempt, batch_id, *, outcome=None, failure=None, dispatched=1, finished=None):
    if outcome is not None and finished is None:
        finished = dispatched + 1
    return AttemptEvidence(attempt, batch_id, 0, "cpu0", 0, dispatched, outcome, failure, finished)


class AttemptHistoryTests(unittest.TestCase):
    def test_every_dispatch_records_batch_and_exit_reason(self):
        clock = ManualClock(wall_ms=1000)
        gate = Gate(request_id="drop")
        backend = ProcessBackend(("cpu0",))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(
                root, config(backend="process", wait_ms=0), clock=clock, backend=backend, hook=gate,
            ) as runtime:
                install(runtime, "affine-r1")
                warm = request("warm")
                runtime.submit(warm)
                self.assertEqual(runtime.wait(warm.key, timeout=5).status, ResultStatus.SUCCEEDED)
                drop = request("drop")
                runtime.submit(drop)
                try:
                    event = gate.await_reached()
                    self.assertTrue(backend.interrupt(event.lease.device))
                finally:
                    gate.release.set()
                result = runtime.wait(drop.key, timeout=8)
                self.assertEqual(result.status, ResultStatus.SUCCEEDED)
                self.assertEqual(result.attempt, 2)
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                receipt = store.receipt(drop.key)
                self.assertEqual(receipt.version, 2)
                self.assertTrue(receipt.attempts_known)
                self.assertEqual(len(receipt.attempts), 2)
                first, second = receipt.attempts
                self.assertEqual(first.outcome, ResultStatus.FAILED)
                self.assertEqual(first.failure.code, ErrorCode.WORKER_LOST)
                self.assertNotEqual(first.batch_id, second.batch_id)
                self.assertEqual(second.outcome, ResultStatus.SUCCEEDED)
                self.assertIsNone(second.failure)
                self.assertGreaterEqual(second.finished_at_ms, second.dispatched_at_ms)
            finally:
                store.close()

    def test_attempt_evidence_survives_settlement_and_expiry(self):
        clock = ManualClock(wall_ms=1000)
        gate = Gate(Boundary.RETURNED)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(
                root, config(max_slots=2, wait_ms=100, deadline_guard_ms=0), clock=clock, hook=gate,
            ) as runtime:
                try:
                    install(runtime, "affine-r1")
                    expires = request("expires", budget=200)
                    runtime.submit(expires)
                    clock.advance_ms(100)
                    runtime.pulse()
                    gate.await_reached()
                    clock.advance_ms(100)
                    runtime.pulse()
                    self.assertEqual(runtime.wait(expires.key, timeout=2).status, ResultStatus.EXPIRED)
                finally:
                    gate.release.set()
                until(lambda: runtime.snapshot().executing_batches == 0)
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                evidence = store.receipt(expires.key).attempts
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0].outcome, ResultStatus.EXPIRED)
                self.assertEqual(evidence[0].failure.code, ErrorCode.DEADLINE)
            finally:
                store.close()

    def test_cancel_before_dispatch_has_known_empty_history(self):
        gate = Gate(request_id="held")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config(max_slots=1, wait_ms=0), hook=gate) as runtime:
                try:
                    install(runtime, "affine-r1")
                    held = request("held")
                    runtime.submit(held)
                    gate.await_reached()
                    # The only device is occupied, so this admission stays queued.
                    queued = request("queued")
                    runtime.submit(queued)
                    cancelled = runtime.cancel(queued.key)
                    self.assertEqual(cancelled.status, ResultStatus.CANCELLED)
                    self.assertEqual(cancelled.attempt, 0)
                finally:
                    gate.release.set()
                until(lambda: runtime.result(held.key).status.terminal)
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                receipt = store.receipt(queued.key)
                self.assertTrue(receipt.attempts_known)
                self.assertEqual(receipt.attempts, ())
            finally:
                store.close()


class LegacyReceiptTests(unittest.TestCase):
    def _legacy_database(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE receipts (
                    position INTEGER PRIMARY KEY,
                    tenant TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    envelope BLOB NOT NULL,
                    UNIQUE (tenant, request_id)
                );
                """
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        finally:
            connection.close()

    def test_v1_receipt_queries_and_round_trips_with_unknown_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._legacy_database(root / "receipts.sqlite")
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                self.assertEqual(store._db.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertEqual(import_archive(store, ROOT / "fixtures" / "receipts.v1.jsonl"), 1)
                view = store.query(RequestKey("sample", "first"))
                self.assertEqual(view["version"], 1)
                self.assertFalse(view["attempts_known"])
                self.assertNotIn("attempts", view)
                self.assertEqual(view["result"]["output"]["values"], [2.0, 7.0, -0.5, 0.0])
                exported = root / "exported.jsonl"
                export_archive(store, exported)
                self.assertEqual(exported.read_bytes(), (ROOT / "fixtures" / "receipts.v1.jsonl").read_bytes())
            finally:
                store.close()

    def test_v1_pending_row_recovers_without_inventing_attempts(self):
        clock = ManualClock(wall_ms=1000)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config(), clock=clock) as runtime:
                install(runtime, "affine-r1")
            pending = request("interrupted")
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                store.insert(Receipt(pending.fingerprint, ResultEnvelope(
                    pending.key, ModelKey("affine", "r1"), ResultStatus.PENDING, 1000, 1,
                    "legacy-batch", 0, "cpu0", 2_000_000,
                )))
            finally:
                store.close()
            with Runtime(root, config(wait_ms=0), clock=clock) as runtime:
                recovered = runtime.result(pending.key)
                self.assertEqual(recovered.failure.code, ErrorCode.INTERRUPTED)
                self.assertEqual(runtime.snapshot().recovered, 1)
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                view = store.query(pending.key)
                self.assertEqual(view["version"], 1)
                self.assertFalse(view["attempts_known"])
            finally:
                store.close()


class RestartAndDuplicateTests(unittest.TestCase):
    def test_restart_keeps_known_failures_and_closes_the_live_attempt(self):
        clock = ManualClock(wall_ms=1000)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config(), clock=clock) as runtime:
                install(runtime, "affine-r1")
            pending = request("restart")
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                result = ResultEnvelope(
                    pending.key, ModelKey("affine", "r1"), ResultStatus.PENDING, 1000,
                    attempt=2, batch_id="batch-2", slot=0, device="cpu0", queue_ns=3_000_000,
                )
                store.insert(Receipt(pending.fingerprint, result, (
                    _evidence(1, "batch-1", outcome=ResultStatus.FAILED,
                              failure=Failure(ErrorCode.WORKER_LOST, "worker lost", True),
                              finished=1002),
                    _evidence(2, "batch-2", dispatched=1003),
                ), 2))
            finally:
                store.close()
            with Runtime(root, config(wait_ms=0), clock=clock) as runtime:
                final = runtime.result(pending.key)
                self.assertEqual(final.status, ResultStatus.FAILED)
                self.assertEqual(final.failure.code, ErrorCode.INTERRUPTED)
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                evidence = store.receipt(pending.key).attempts
                self.assertEqual(evidence[0].failure.code, ErrorCode.WORKER_LOST)
                self.assertEqual(evidence[0].outcome, ResultStatus.FAILED)
                self.assertEqual(evidence[1].outcome, ResultStatus.FAILED)
                self.assertEqual(evidence[1].failure.code, ErrorCode.INTERRUPTED)
            finally:
                store.close()

    def test_duplicate_import_only_adds_correct_new_attempts(self):
        clock = ManualClock(wall_ms=1000)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config(wait_ms=0), clock=clock) as runtime:
                install(runtime, "affine-r1")
                item = request("dup")
                runtime.submit(item)
                settled = runtime.wait(item.key, timeout=3)
                self.assertEqual(settled.status, ResultStatus.SUCCEEDED)
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                original = store.receipt(item.key)
                self.assertEqual(store.import_terminal((original,)), 0)
                archive = root / "dup.jsonl"
                export_archive(store, archive)
                self.assertEqual(import_archive(store, archive), 0)
                self.assertEqual(len(store.receipt(item.key).attempts), 1)

                # A richer copy may add only the attempts that agree with what
                # is already known; the final result stays unique.
                extended = replace(original, attempts=(
                    original.attempts[0],
                    _evidence(2, "batch-9", outcome=ResultStatus.SUCCEEDED),
                ))
                self.assertEqual(store.import_terminal((extended,)), 1)
                self.assertEqual(len(store.receipt(item.key).attempts), 2)
                self.assertEqual(store.receipt(item.key).result, original.result)

                # A copy that disagrees about a recorded attempt is rejected.
                conflicting = replace(original, attempts=(
                    _evidence(1, "batch-other", outcome=ResultStatus.FAILED,
                              failure=Failure(ErrorCode.EXECUTION, "different", False)),
                ))
                with self.assertRaises(Conflict):
                    store.import_terminal((conflicting,))
                self.assertEqual(len(store.receipt(item.key).attempts), 2)
                self.assertEqual(store.receipt(item.key).result, original.result)
            finally:
                store.close()

