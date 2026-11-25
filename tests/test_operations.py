from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from batchsail.clock import ManualClock
from batchsail.device import MemoryPolicy
from batchsail.errors import Conflict, ErrorCode, SailError
from batchsail.executor import ProcessBackend
from batchsail.inspection import inspect_workspace
from batchsail.registry import ModelKey
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime

from support import config, install, request


class OperationalTests(unittest.TestCase):
    def test_idle_process_trim_releases_only_retired_residency(self):
        backend = ProcessBackend(("cpu0",))
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(devices=(16_384,), wait_ms=0), backend=backend,
        ) as runtime:
            install(runtime, "affine-r1", "affine-r2")
            old = request("old")
            runtime.submit(old)
            self.assertEqual(runtime.wait(old.key, timeout=5).model.revision, "r1")
            self.assertEqual(runtime.trim()[0].outcome, "no_eligible_residency")

            runtime.activate(ModelKey("affine", "r2"))
            new = request("new")
            runtime.submit(new)
            self.assertEqual(runtime.wait(new.key, timeout=5).model.revision, "r2")
            self.assertEqual(runtime.snapshot().devices[0].resident_bytes, 8192)
            result, = runtime.trim(revision=ModelKey("affine", "r1"))
            self.assertEqual(result.outcome, "trimmed")
            self.assertEqual([item.key.revision for item in result.removed], ["r1"])
            self.assertEqual([item.key.revision for item in result.remaining], ["r2"])
            self.assertEqual(runtime.snapshot().devices[0].resident_bytes, 4096)

    def test_drain_flushes_existing_work_without_requiring_manual_clock_advance(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=1000), clock=ManualClock(),
        ) as runtime:
            install(runtime, "affine-r1")
            first = request("first", budget=None)
            runtime.submit(first)
            self.assertEqual(runtime.result(first.key).status, ResultStatus.PENDING)
            report = runtime.drain(timeout=3)
            self.assertTrue(report.completed)
            self.assertFalse(runtime.snapshot().accepting)
            self.assertEqual(runtime.result(first.key).status, ResultStatus.SUCCEEDED)
            self.assertTrue(runtime.submit(first).reused)
            with self.assertRaises(SailError) as blocked:
                runtime.submit(request("second"))
            self.assertEqual(blocked.exception.code, ErrorCode.UNAVAILABLE)

    def test_audit_distinguishes_page_size_mismatch_from_receipt_index_drift(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            options = config(wait_ms=0)
            with Runtime(root, options) as runtime:
                install(runtime, "affine-r1")
                source = request("sample")
                runtime.submit(source)
                self.assertEqual(runtime.wait(source.key, timeout=3).status, ResultStatus.SUCCEEDED)
            self.assertTrue(inspect_workspace(root, options).clean)
            changed = replace(options, memory=MemoryPolicy(page_bytes=8192))
            self.assertFalse(inspect_workspace(root, changed).clean)
            with self.assertRaisesRegex(Conflict, "workspace uses"):
                Runtime(root, changed)
            connection = sqlite3.connect(root / "receipts.sqlite")
            try:
                with connection:
                    connection.execute("UPDATE receipts SET revision = 'other'")
            finally:
                connection.close()
            report = inspect_workspace(root, options)
            self.assertFalse(report.clean)
            self.assertEqual(report.receipts.issues[0]["code"], "index_mismatch")
