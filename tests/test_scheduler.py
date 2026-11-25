from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from batchsail.clock import ManualClock
from batchsail.device import DeviceLedger, DeviceSpec
from batchsail.device.types import MemoryDemand
from batchsail.errors import Conflict
from batchsail.registry import ModelKey
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime

from support import Gate, config, install, request


class SchedulingTests(unittest.TestCase):
    def test_estimate_agrees_with_real_lease_bytes_without_reserving(self):
        gate = Gate()
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0), hook=gate,
        ) as runtime:
            try:
                install(runtime, "affine-r1")
                expected = runtime.estimate("affine", rows=5)
                self.assertEqual(expected.padded_rows, 8)
                self.assertEqual(runtime.snapshot().devices[0].workspace_bytes, 0)
                item = request("matching", rows=5)
                runtime.submit(item)
                gate.await_reached()
                self.assertEqual(runtime.snapshot().devices[0].workspace_bytes, expected.demand.workspace_bytes)
                self.assertEqual(runtime.estimate("affine", rows=5).decisions[0].reason, "busy")
            finally:
                gate.release.set()

    def test_queue_rotates_tenants_and_expedites_urgent_shape(self):
        gate = Gate(request_id="hold")
        clock = ManualClock()
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(max_slots=2, wait_ms=40, deadline_guard_ms=5), clock=clock, hook=gate,
        ) as runtime:
            try:
                install(runtime, "affine-r1")
                hold = request("hold", budget=None)
                runtime.submit(hold)
                clock.advance_ms(40)
                runtime.pulse()
                gate.await_reached()

                a = [request(f"heavy-{index}", rows=2, tenant="heavy", budget=None) for index in range(5)]
                monitor = request("monitor", rows=2, tenant="monitor", budget=None)
                urgent = request("urgent", rows=12, tenant="urgent", budget=4)
                for item in (*a, monitor, urgent):
                    runtime.submit(item)
                gate.release.set()
                urgent_result = runtime.wait(urgent.key, timeout=3)
                heavy = runtime.wait(a[0].key, timeout=3)
                light = runtime.wait(monitor.key, timeout=3)
                self.assertEqual(urgent_result.status, ResultStatus.SUCCEEDED)
                self.assertEqual(heavy.batch_id, light.batch_id)
                self.assertLess(int(urgent_result.batch_id.split("-")[1]), int(heavy.batch_id.split("-")[1]))
            finally:
                gate.release.set()

    def test_ledger_rejects_stale_placement_and_protects_revision(self):
        ledger = DeviceLedger((DeviceSpec("cpu0", 7000),))
        one = ModelKey("model", "r1")
        two = ModelKey("model", "r2")
        protected = frozenset()
        plan = ledger.preferred(ledger.assess(MemoryDemand(one, 4096, 128), protected))
        lease = ledger.acquire(plan, protected)
        with self.assertRaises(Conflict):
            ledger.acquire(plan, protected)
        ledger.release(lease)
        rejects = ledger.assess(MemoryDemand(two, 4096, 128), frozenset({one}))
        self.assertEqual(rejects[0].reason, "protected_residency")
        self.assertEqual(rejects[0].locked_revisions, (one,))
        plan = ledger.preferred(ledger.assess(MemoryDemand(two, 4096, 128), protected))
        other = ledger.acquire(plan, protected)
        self.assertGreater(other.residency.generation, lease.residency.generation)
        self.assertEqual(plan.eviction_bytes, 4096)
        ledger.release(other)
        view = ledger.snapshot()[0]
        self.assertEqual(view.resident_bytes, 4096)
        self.assertEqual(view.workspace_bytes, 0)
        self.assertEqual(view.evictions, 1)
