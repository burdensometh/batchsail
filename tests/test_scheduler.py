from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from batchsail.batcher import BatchPlanner, BatchQueues, BatchingPolicy
from batchsail.clock import NS_PER_MS, ManualClock
from batchsail.config import RuntimeConfig
from batchsail.device import DeviceLedger, DeviceSpec, MemoryPolicy
from batchsail.device.types import MemoryDemand
from batchsail.errors import Conflict, Invalid
from batchsail.registry import ModelKey, ModelStore
from batchsail.registry.catalog import RevisionPin
from batchsail.registry.model import InstalledModel
from batchsail.request import Request, RequestDeadline, RequestKey
from batchsail.request.lifecycle import Ticket
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime
from batchsail.tensor import Tensor

from support import MODELS, Gate, config, install, request

MS = NS_PER_MS

_CACHE: dict[str, InstalledModel] = {}


def installed(name):
    if name not in _CACHE:
        revision, _ = ModelStore.read_bundle(MODELS / name)
        _CACHE[name] = InstalledModel(revision, MODELS / name / "weights.v1.json")
    return _CACHE[name]


def ticket(model, name, rows, *, tenant="alpha", accepted_ns=0, budget_ms=None, sequence=1):
    columns = 3 if model == "affine" else 2
    item = Request(
        RequestKey(tenant, name), model, Tensor((rows, columns), (0.0,) * (rows * columns)), budget_ms,
    )
    pin = RevisionPin(installed(f"{model}-r1"), sequence)
    return Ticket(item, pin, RequestDeadline.from_budget(accepted_ns, budget_ms), sequence)


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

    def test_urgent_priority_dispatch_order_reacts_to_config(self):
        def dispatch_ids(*, urgent_priority):
            gate = Gate(request_id="hold")
            clock = ManualClock()
            with TemporaryDirectory() as directory, Runtime(
                Path(directory),
                config(max_slots=4, wait_ms=0, deadline_guard_ms=5, urgent_priority=urgent_priority),
                clock=clock, hook=gate,
            ) as runtime:
                try:
                    install(runtime, "affine-r1", "filter-r1")
                    # Block the only device so both groups queue before selection.
                    hold = request("hold", model="affine", budget=None)
                    runtime.submit(hold)
                    gate.await_reached()
                    normal = [request(f"normal-{index}", rows=4, model="affine", tenant="normal")
                              for index in range(2)]
                    urgent = request("urgent", rows=4, model="filter", tenant="urgent", budget=100)
                    for item in (*normal, urgent):
                        runtime.submit(item)
                    # Move into the urgent window (deadline 100ms, guard 5ms).
                    clock.advance_ms(97)
                    gate.release.set()
                    results = [runtime.wait(item.key, timeout=3) for item in (*normal, urgent)]
                    return [int(result.batch_id.split("-")[1]) for result in results]
                finally:
                    gate.release.set()

        plain_normal, plain_last, plain_urgent = dispatch_ids(urgent_priority=False)
        self.assertEqual(plain_normal, plain_last)
        self.assertLess(plain_normal, plain_urgent)

        fast_normal, fast_last, fast_urgent = dispatch_ids(urgent_priority=True)
        self.assertEqual(fast_normal, fast_last)
        self.assertLess(fast_urgent, fast_normal)
        self.assertLess(fast_urgent, fast_last)

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


class CrossShapeUrgentPriorityTests(unittest.TestCase):
    """Unit-level coverage of urgent-first selection across shape groups."""

    def _planner(self, policy, capacity=65_536):
        ledger = DeviceLedger((DeviceSpec("cpu0", capacity),))
        return BatchPlanner(BatchQueues(policy), ledger, MemoryPolicy())

    def _normal_full_group(self, queues):
        first = ticket("affine", "first", 4, budget_ms=None, sequence=1)
        second = ticket("affine", "second", 4, budget_ms=None, sequence=2)
        for item in (first, second):
            queues.add(item, 0)
        return first, second

    def _urgent_shape(self, queues):
        urgent = ticket("filter", "urgent", 4, tenant="urgent", budget_ms=2, accepted_ns=10 * MS, sequence=3)
        queues.add(urgent, 10 * MS)
        return urgent

    def test_disabled_policy_keeps_original_group_order(self):
        policy = BatchingPolicy(max_slots=2, wait_ms=10, deadline_guard_ms=5, urgent_priority=False)
        planner = self._planner(policy)
        queues = planner.queues
        self._normal_full_group(queues)
        self._urgent_shape(queues)
        plan, _, _ = planner.select(10 * MS, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.shape.revision.name, "affine")
        self.assertEqual(len(plan.tickets), 2)

    def test_enabled_policy_dispatches_urgent_group_before_older_full_group(self):
        policy = BatchingPolicy(max_slots=2, wait_ms=10, deadline_guard_ms=5, urgent_priority=True)
        planner = self._planner(policy)
        queues = planner.queues
        self._normal_full_group(queues)
        urgent = self._urgent_shape(queues)
        plan, _, _ = planner.select(10 * MS, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.shape.revision.name, "filter")
        self.assertEqual(plan.tickets, (urgent,))

    def test_urgent_groups_are_ordered_by_earliest_deadline(self):
        policy = BatchingPolicy(max_slots=1, wait_ms=10, deadline_guard_ms=5, urgent_priority=True)
        planner = self._planner(policy)
        queues = planner.queues
        # Older group with the later deadline; newer group with the earlier one.
        older = ticket("filter", "older", 4, budget_ms=14, accepted_ns=0, sequence=1)
        newer = ticket("affine", "newer", 4, budget_ms=12, accepted_ns=0, sequence=2)
        queues.add(older, 0)
        queues.add(newer, 5 * MS)
        plan, _, _ = planner.select(10 * MS, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.tickets[0].key, newer.key)

        disabled = BatchingPolicy(max_slots=1, wait_ms=10, deadline_guard_ms=5)
        plain = self._planner(disabled)
        for item in (older, newer):
            plain.queues.add(item, 0 if item is older else 5 * MS)
        plan, _, _ = plain.select(10 * MS, frozenset())
        self.assertEqual(plan.tickets[0].key, older.key)

    def test_unplaceable_urgent_group_falls_through_to_next_placeable_shape(self):
        policy = BatchingPolicy(max_slots=1, wait_ms=0, deadline_guard_ms=5, urgent_priority=True)
        planner = self._planner(policy, capacity=5000)
        queues = planner.queues
        # Filter's padded workspace does not fit this device; affine does.
        urgent = ticket("filter", "urgent", 128, tenant="urgent", budget_ms=2, sequence=1)
        fallback = ticket("affine", "fallback", 4, budget_ms=None, sequence=2)
        queues.add(urgent, 0)
        queues.add(fallback, 0)
        plan, _, blocked = planner.select(0, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.shape.revision.name, "affine")
        self.assertEqual(plan.tickets[0].key, fallback.key)
        self.assertEqual([item.shape.revision.name for item in blocked], ["filter"])

    def test_urgent_group_jumps_ahead_even_when_it_cannot_fill_all_slots(self):
        policy = BatchingPolicy(max_slots=4, wait_ms=0, deadline_guard_ms=5, urgent_priority=True)
        planner = self._planner(policy)
        queues = planner.queues
        first = ticket("affine", "first", 4, budget_ms=None, sequence=1)
        second = ticket("affine", "second", 4, budget_ms=None, sequence=2)
        for item in (first, second):
            queues.add(item, 0)
        urgent = ticket("filter", "urgent", 4, tenant="urgent", budget_ms=2, sequence=3)
        queues.add(urgent, 0)
        plan, _, _ = planner.select(0, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.shape.revision.name, "filter")
        self.assertEqual(plan.tickets, (urgent,))

    def test_groups_merge_only_compatible_shapes_of_one_generation(self):
        policy = BatchingPolicy(max_slots=4, wait_ms=0, deadline_guard_ms=5, urgent_priority=True)
        planner = self._planner(policy)
        queues = planner.queues
        affine = [ticket("affine", f"affine-{index}", 4, tenant=f"t{index}", budget_ms=2, sequence=index)
                  for index in range(2)]
        other = ticket("filter", "other", 4, tenant="other", budget_ms=2, sequence=10)
        for item in (*affine, other):
            queues.add(item, 0)
        plan, _, _ = planner.select(0, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.shape.revision.name, "affine")
        self.assertEqual(len(plan.tickets), 2)
        for item in plan.tickets:
            self.assertEqual(item.pin.model.key, plan.shape.revision)

    def test_request_occupies_one_slot_even_when_urgent_in_a_mixed_group(self):
        policy = BatchingPolicy(max_slots=4, wait_ms=0, deadline_guard_ms=5, urgent_priority=True)
        planner = self._planner(policy)
        queues = planner.queues
        plain = ticket("affine", "plain", 4, budget_ms=None, sequence=1)
        urgent = ticket("affine", "urgent", 4, budget_ms=2, sequence=2)
        last = ticket("affine", "last", 4, budget_ms=None, sequence=3)
        for item in (plain, urgent, last):
            queues.add(item, 0)
        plan, _, _ = planner.select(0, frozenset())
        self.assertIsNotNone(plan)
        self.assertEqual(plan.tickets[0].key, urgent.key)
        keys = [item.key for item in plan.tickets]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys.count(urgent.key), 1)


class BatchingPolicyConfigTests(unittest.TestCase):
    def test_urgent_priority_is_configurable_and_validated(self):
        self.assertFalse(BatchingPolicy().urgent_priority)
        self.assertTrue(BatchingPolicy(urgent_priority=True).urgent_priority)
        for alias in ("urgent_across_shapes", "cross_shape_urgent", "cross_shape_urgent_priority"):
            with self.subTest(alias=alias):
                policy = BatchingPolicy.from_config({alias: True})
                self.assertTrue(policy.urgent_priority)
        with self.assertRaises(Invalid):
            BatchingPolicy(urgent_priority=1)
        with self.assertRaises(Invalid):
            BatchingPolicy.from_config({"urgent_priority": True, "cross_shape_urgent": False})
        with self.assertRaises(Invalid):
            BatchingPolicy.from_config({"unknown": True})

    def test_toml_runtime_config_reads_urgent_priority(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "batch.toml"
            path.write_text(
                "[batching]\nmax_slots = 4\nwait_ms = 0\nurgent_priority = true\n",
                encoding="utf-8",
            )
            config = RuntimeConfig.load(path)
            self.assertTrue(config.batching.urgent_priority)
            self.assertTrue(config.batching.urgent_across_shapes)
