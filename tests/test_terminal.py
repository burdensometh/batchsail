"""Durable, resumable terminal-result stream tests."""

from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic
import unittest

from batchsail.clock import ManualClock
from batchsail.codec import decode, encode
from batchsail.errors import ErrorCode
from batchsail.executor import AbortBoundary, Boundary
from batchsail.http import SailServer
from batchsail.registry import ModelKey
from batchsail.request.client import HttpClient
from batchsail.request.model import RequestKey
from batchsail.results import ResultStatus, TerminalPage
from batchsail.results.inspection import inspect_receipts
from batchsail.results.model import Receipt, ResultEnvelope
from batchsail.results.store import ReceiptStore
from batchsail.runtime import Runtime

from support import Gate, MODELS, config, install, request, until


class TerminalStreamTests(unittest.TestCase):
    def test_positions_follow_settlement_order_for_every_terminal_kind(self):
        clock = ManualClock()
        gate = Gate(Boundary.ACQUIRED)
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(max_slots=2, wait_ms=100, deadline_guard_ms=0),
            clock=clock, hook=gate,
        ) as runtime:
            install(runtime, "affine-r1")
            success = request("success")
            cancelled = request("cancelled", budget=None)
            expired = request("expired", budget=20)
            for item in (success, cancelled, expired):
                runtime.submit(item)
            gate.await_reached()
            try:
                # success is parked in the worker; the other two are still live.
                self.assertEqual(runtime.cancel(cancelled.key).status, ResultStatus.CANCELLED)
                clock.advance_ms(20)
                runtime.pulse()
                self.assertEqual(runtime.wait(expired.key, timeout=2).status, ResultStatus.EXPIRED)
            finally:
                gate.release.set()
            self.assertEqual(runtime.wait(success.key, timeout=3).status, ResultStatus.SUCCEEDED)

            page = runtime.terminal_stream(after=0, timeout=0)
            self.assertIsInstance(page, TerminalPage)
            by_id = {result.key.request_id: seq for seq, result in page.entries}
            self.assertEqual(set(by_id), {"success", "cancelled", "expired"})
            self.assertEqual(sorted(seq for seq, _ in page.entries), [1, 2, 3])
            self.assertEqual(page.next, 3)
            self.assertEqual(page.latest, 3)
            statuses = {result.key.request_id: result.status for _, result in page.entries}
            self.assertEqual(statuses["cancelled"], ResultStatus.CANCELLED)
            self.assertEqual(statuses["expired"], ResultStatus.EXPIRED)
            self.assertEqual(statuses["success"], ResultStatus.SUCCEEDED)

            # Resuming from a cursor only returns strictly later terminals.
            resumed = runtime.terminal_stream(after=by_id["cancelled"], timeout=0)
            self.assertEqual(
                sorted(result.key.request_id for _, result in resumed.entries),
                ["expired", "success"],
            )
            self.assertEqual(runtime.terminal_stream(after=page.latest, timeout=0).entries, ())

    def test_worker_failures_and_recovery_failures_are_visible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def hook(event):
                if event.boundary == Boundary.DEQUEUED and event.batch.slots[0].key.request_id == "broken":
                    raise AbortBoundary()

            with Runtime(root, config(wait_ms=0), hook=hook) as runtime:
                install(runtime, "affine-r1")
                broken = request("broken")
                runtime.submit(broken)
                broken_result = runtime.wait(broken.key, timeout=3)
                self.assertEqual(broken_result.status, ResultStatus.FAILED)
                self.assertEqual(broken_result.failure.code, ErrorCode.INTERRUPTED)

            # Simulate a process that died with work in flight: insert a live
            # pending receipt directly, then reopen so recovery marks it failed.
            store_path = root / "receipts.sqlite"
            pending = request("pending", budget=None)
            store = ReceiptStore(store_path)
            try:
                store.insert(
                    Receipt(
                        pending.fingerprint,
                        ResultEnvelope(
                            pending.key, ModelKey("affine", "r1"), ResultStatus.PENDING, 0,
                        ),
                    ),
                )
            finally:
                store.close()

            with Runtime(root, config(wait_ms=0)) as runtime:
                page = runtime.terminal_stream(after=0, limit=100, timeout=0)
                by_id = {result.key.request_id: (seq, result) for seq, result in page.entries}
                self.assertEqual(set(by_id), {"broken", "pending"})
                _seq_broken, broken_again = by_id["broken"]
                seq_pending, recovered = by_id["pending"]
                self.assertEqual(broken_again.failure.code, ErrorCode.INTERRUPTED)
                self.assertEqual(recovered.status, ResultStatus.FAILED)
                self.assertEqual(recovered.failure.code, ErrorCode.INTERRUPTED)
                self.assertEqual(runtime.snapshot().recovered, 1)
                # One position per terminal, gap-free, ordered by settlement.
                self.assertEqual(sorted(seq for seq, _ in page.entries), [1, 2])
                self.assertEqual(seq_pending, 2)
                # The recovered failure is resumable from its position.
                tail = runtime.terminal_stream(after=_seq_broken, timeout=0)
                self.assertEqual([result.key.request_id for _, result in tail.entries], ["pending"])

    def test_repeat_settlement_and_replay_keep_a_single_position(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1")
            item = request("once")
            runtime.submit(item)
            settled = runtime.wait(item.key, timeout=3)
            # Cancelling or re-reading an already terminal result must not add a seq.
            self.assertEqual(runtime.cancel(item.key), settled)
            self.assertEqual(runtime.result(item.key), settled)
            self.assertTrue(runtime.submit(item).reused)
            page = runtime.terminal_stream(after=0, timeout=0)
            self.assertEqual(len(page.entries), 1)
            self.assertEqual(page.latest, 1)

    def test_long_poll_waits_briefly_and_returns_without_blocking_settlement(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1")
            # An immediate read at the head returns an empty page without delay.
            start = monotonic()
            empty = runtime.terminal_stream(after=0, timeout=0.05)
            self.assertEqual(empty.entries, ())
            self.assertLess(monotonic() - start, 0.3)

            holder = {}

            def read_late():
                # No terminal result exists yet, so this parks on the condition.
                holder["page"] = runtime.terminal_stream(after=0, timeout=3)

            thread = Thread(target=read_late)
            thread.start()
            until(thread.is_alive)

            # The reader is parked. Settlements still proceed and must wake it
            # rather than queue behind the waiting reader.
            later = request("later")
            extra = request("extra")
            runtime.submit(later)
            runtime.submit(extra)
            self.assertEqual(runtime.wait(later.key, timeout=3).status, ResultStatus.SUCCEEDED)
            self.assertEqual(runtime.wait(extra.key, timeout=3).status, ResultStatus.SUCCEEDED)
            thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            page = holder["page"]
            self.assertEqual(
                [result.key.request_id for _, result in page.entries], ["later", "extra"],
            )
            self.assertEqual(page.latest, 2)

            # A timed wait at the live head returns an empty page, not forever.
            start = monotonic()
            idle = runtime.terminal_stream(after=page.latest, timeout=0.05)
            self.assertEqual(idle.entries, ())
            self.assertGreaterEqual(monotonic() - start, 0.04)

    def test_stream_survives_restart_and_buffered_event_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            options = config(wait_ms=0)
            small_ring = replace(options, event_capacity=16)
            with Runtime(root, small_ring) as runtime:
                install(runtime, "affine-r1")
                for index in range(25):
                    runtime.submit(request(f"item-{index:02d}"))
                until(lambda: runtime.terminal_stream(after=0, limit=100, timeout=0).latest == 25)
                # The volatile event ring has dropped history; the durable
                # terminal stream still exposes every settlement.
                events = runtime.events(after=0, limit=100)
                self.assertGreater(events["dropped_before"], 1)
            with Runtime(root, config(wait_ms=0)) as restarted:
                page = restarted.terminal_stream(after=0, limit=100, timeout=0)
                self.assertEqual(page.latest, 25)
                self.assertEqual(len(page.entries), 25)
                self.assertEqual([seq for seq, _ in page.entries], list(range(1, 26)))
                # Cursor resume across the restart.
                tail = restarted.terminal_stream(after=20, timeout=0)
                self.assertEqual(
                    [result.key.request_id for _, result in tail.entries],
                    [f"item-{index:02d}" for index in range(20, 25)],
                )

    def test_v1_database_is_backfilled_in_terminal_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "receipts.sqlite"
            fixture = MODELS.parent / "receipts.v1.jsonl"
            # Build a genuine v1 database with the historical terminal receipt
            # plus a pending row, using the v1 schema directly.
            receipt = Receipt.from_v1(decode(fixture.read_bytes().strip()))
            pending_key = RequestKey("sample", "pending")
            pending = ResultEnvelope(
                pending_key, receipt.result.model, ResultStatus.PENDING, receipt.result.accepted_at_ms,
            )
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    CREATE TABLE receipts (
                        position INTEGER PRIMARY KEY,
                        tenant TEXT NOT NULL, request_id TEXT NOT NULL,
                        model TEXT NOT NULL, revision TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN
                          ('pending', 'succeeded', 'expired', 'cancelled', 'failed')),
                        fingerprint TEXT NOT NULL, envelope BLOB NOT NULL,
                        UNIQUE (tenant, request_id))
                    """,
                )
                connection.execute("CREATE INDEX receipts_tenant_position ON receipts (tenant, position)")
                connection.execute("CREATE INDEX receipts_status_position ON receipts (status, position)")
                for position, record in ((10, receipt), (11, Receipt(receipt.fingerprint, pending))):
                    result = record.result
                    connection.execute(
                        "INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            position, result.key.tenant, result.key.request_id,
                            result.model.name, result.model.revision, result.status.value,
                            record.fingerprint, encode(result.to_json()),
                        ),
                    )
                connection.execute("PRAGMA user_version = 1")
                connection.commit()
            finally:
                connection.close()

            with Runtime(root, config(wait_ms=0)) as runtime:
                # The pre-existing terminal row is backfilled; the pending row
                # is recovered to failed and appended after it.
                page = runtime.terminal_stream(after=0, timeout=0)
                self.assertEqual(
                    [(seq, result.key.request_id, result.status) for seq, result in page.entries],
                    [
                        (1, "first", ResultStatus.SUCCEEDED),
                        (2, "pending", ResultStatus.FAILED),
                    ],
                )
                self.assertEqual(runtime.snapshot().recovered, 1)
            audit = inspect_receipts(path, {})
            self.assertEqual(audit.issue_count, 0)
            self.assertEqual(audit.terminal_logged, 2)
            self.assertEqual(audit.sqlite_version, 2)


class TerminalTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runtime = Runtime(Path(self.directory.name), config(wait_ms=0))
        self.addCleanup(self.runtime.close)
        self.server = SailServer(("127.0.0.1", 0), self.runtime, bundle_root=MODELS)
        self.addCleanup(self.server.server_close)
        self.thread = Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)

    def send(self, path):
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, decode(response.read())
        finally:
            connection.close()

    def test_http_endpoint_paginates_and_long_polls_by_cursor(self):
        install(self.runtime, "affine-r1")
        for name in ("one", "two", "three"):
            self.runtime.submit(request(name))
        until(lambda: self.runtime.terminal_stream(after=0, limit=100, timeout=0).latest == 3)

        status, first = self.send("/v1/terminal-results?after=0&limit=2")
        self.assertEqual(status, 200)
        self.assertEqual([item["terminal_position"] for item in first["results"]], [1, 2])
        self.assertEqual(first["next"], 2)
        self.assertEqual(first["latest"], 3)
        status, second = self.send(f"/v1/terminal-results?after={first['next']}")
        self.assertEqual(status, 200)
        self.assertEqual([item["request_id"] for item in second["results"]], ["three"])
        self.assertEqual(second["next"], 3)

        # A wait at the live head returns an empty page after the bounded wait.
        start = monotonic()
        status, idle = self.send("/v1/terminal-results?after=3&wait_ms=50")
        self.assertEqual(status, 200)
        self.assertEqual(idle["results"], [])
        self.assertEqual(idle["next"], 3)
        self.assertGreaterEqual(monotonic() - start, 0.04)

        # Invalid cursors and waits are rejected.
        for bad in ("/v1/terminal-results?after=-1", "/v1/terminal-results?wait_ms=10001",
                    "/v1/terminal-results?limit=0", "/v1/terminal-results?bogus=1"):
            with self.subTest(bad=bad):
                self.assertEqual(self.send(bad)[0], 400)

    def test_typed_client_resumes_without_gaps_or_duplicates(self):
        install(self.runtime, "affine-r1")
        host, port = self.server.server_address
        client = HttpClient(f"http://{host}:{port}")
        for index in range(6):
            self.runtime.submit(request(f"job-{index}"))
        until(lambda: self.runtime.terminal_stream(after=0, limit=100, timeout=0).latest == 6)

        seen: list[int] = []
        cursor = 0
        while cursor < 6:
            entries, cursor, latest = client.terminal_page(after=cursor, limit=2)
            self.assertEqual(latest, 6)
            seen.extend(position for position, _ in entries)
        self.assertEqual(seen, [1, 2, 3, 4, 5, 6])

        # The follow generator drains the same stream and then terminates.
        followed = list(client.terminal_follow(after=0, limit=3, wait_ms=10))
        self.assertEqual([position for position, _ in followed], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(envelope.status.terminal for _, envelope in followed))

        # Resuming from the last position duplicates nothing.
        entries, next_cursor, latest = client.terminal_page(after=6)
        self.assertEqual((entries, next_cursor, latest), ((), 6, 6))


if __name__ == "__main__":
    unittest.main()
