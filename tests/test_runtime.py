from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from batchsail.clock import ManualClock
from batchsail.errors import Conflict, ErrorCode, SailError
from batchsail.executor import AbortBoundary, Boundary, InlineBackend, ProcessBackend
from batchsail.registry import ModelKey
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime
from batchsail.tensor import Tensor

from support import Gate, config, install, request, until


class RuntimeTests(unittest.TestCase):
    def test_dispatch_packs_different_lengths_without_crossing_filter_slots(self):
        clock = ManualClock()
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(max_slots=4, wait_ms=100, row_multiple=4), clock=clock,
        ) as runtime:
            install(runtime, "filter-r1")
            one = replace(request("one", model="filter"), tensor=Tensor((1, 2), (2, 4)))
            two = replace(request("two", 3, model="filter"), tensor=Tensor((3, 2), (1, 0, 0, 1, 0, 0)))
            for item in (one, two):
                runtime.submit(item)
            clock.advance_ms(100)
            runtime.pulse()
            left = runtime.wait(one.key, timeout=3)
            right = runtime.wait(two.key, timeout=3)
            self.assertEqual(left.batch_id, right.batch_id)
            self.assertNotEqual(left.slot, right.slot)
            self.assertEqual(left.output.values, (5.0, 5.0))
            self.assertEqual(right.output.values, (2.0, -2.0, 1.75, 1.0, 1.0, -0.25))

    def test_idempotence_after_revision_roll_while_old_generation_is_running(self):
        clock = ManualClock()
        gate = Gate(request_id="old")
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(devices=(6200, 6200), wait_ms=0), clock=clock, hook=gate,
        ) as runtime:
            try:
                install(runtime, "affine-r1", "affine-r2")
                old = request("old")
                runtime.submit(old)
                gate.await_reached()
                runtime.activate(ModelKey("affine", "r2"))
                draining = next(view for view in runtime.snapshot().models if view.model.key.revision == "r1")
                self.assertTrue(draining.draining)
                self.assertEqual(draining.pins, 1)
                new = request("new")
                runtime.submit(new)
                self.assertEqual(runtime.wait(new.key, timeout=3).model.revision, "r2")
                self.assertEqual(runtime.submit(old).revision, "r1")
                self.assertTrue(runtime.submit(old).reused)
                with self.assertRaises(Conflict):
                    runtime.submit(replace(old, deadline_ms=1000))
            finally:
                gate.release.set()
            self.assertEqual(runtime.wait(old.key, timeout=3).model.revision, "r1")
            until(lambda: all(view.pins == 0 for view in runtime.snapshot().models))

    def test_cancel_and_deadline_resist_late_success_and_release_all_pins(self):
        clock = ManualClock()
        gate = Gate(Boundary.RETURNED)
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(max_slots=2, wait_ms=100, deadline_guard_ms=0), clock=clock, hook=gate,
        ) as runtime:
            try:
                install(runtime, "affine-r1")
                cancel = request("cancel", budget=200)
                expires = request("expires", budget=20)
                for item in (cancel, expires):
                    runtime.submit(item)
                gate.await_reached()
                cancelled = runtime.cancel(cancel.key)
                self.assertEqual(cancelled.status, ResultStatus.CANCELLED)
                self.assertEqual(runtime.cancel(cancel.key), cancelled)
                clock.advance_ms(20)
                runtime.pulse()
                self.assertEqual(runtime.wait(expires.key, timeout=2).status, ResultStatus.EXPIRED)
                self.assertGreater(runtime.snapshot().devices[0].workspace_bytes, 0)
            finally:
                gate.release.set()
            until(lambda: runtime.snapshot().executing_batches == 0)
            self.assertEqual(runtime.result(cancel.key), cancelled)
            self.assertEqual(runtime.result(expires.key).status, ResultStatus.EXPIRED)
            self.assertEqual(runtime.snapshot().devices[0].workspace_bytes, 0)
            self.assertEqual(runtime.snapshot().models[0].pins, 0)

    def test_controlled_abort_across_every_boundary(self):
        for boundary in Boundary:
            with self.subTest(boundary=boundary), TemporaryDirectory() as directory:
                def hook(event):
                    if event.boundary == boundary and event.batch.slots[0].key.request_id == "abort":
                        raise AbortBoundary()

                with Runtime(Path(directory), config(wait_ms=0), hook=hook) as runtime:
                    install(runtime, "affine-r1")
                    aborted = request("abort")
                    runtime.submit(aborted)
                    result = runtime.wait(aborted.key, timeout=3)
                    self.assertEqual(result.failure.code, ErrorCode.INTERRUPTED)
                    self.assertEqual(runtime.snapshot().devices[0].workspace_bytes, 0)
                    retry = request("retry", retry_of=aborted.key.request_id)
                    runtime.submit(retry)
                    self.assertEqual(runtime.wait(retry.key, timeout=3).status, ResultStatus.SUCCEEDED)

    def test_real_process_loss_retries_and_resets_residency(self):
        gate = Gate(request_id="drop")
        backend = ProcessBackend(("cpu0",))
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(backend="process", wait_ms=0), backend=backend, hook=gate,
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
            result = runtime.wait(drop.key, timeout=5)
            self.assertEqual(result.status, ResultStatus.SUCCEEDED)
            self.assertEqual(result.attempt, 2)
            device = runtime.snapshot().devices[0]
            self.assertEqual(device.resets, 1)
            self.assertEqual(device.workspace_bytes, 0)
            self.assertGreater(device.residencies[0].generation, event.lease.residency.generation)

    def test_validation_and_impossible_single_slot(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(devices=(5000,), wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1")
            with self.assertRaises(SailError) as limitation:
                runtime.submit(request("large", rows=100))
            self.assertEqual(limitation.exception.code, ErrorCode.IMPOSSIBLE)
            expired = request("expired", budget=0)
            runtime.submit(expired)
            result = runtime.result(expired.key)
            self.assertEqual(result.status, ResultStatus.EXPIRED)
            self.assertEqual(result.attempt, 0)
            self.assertEqual(runtime.snapshot().queued, 0)

    def test_valid_reordering_is_correlated_but_stale_identity_is_rejected(self):
        class Reverse(InlineBackend):
            def execute(self, batch, lease):
                result = super().execute(batch, lease)
                if batch.slots[0].key.request_id == "stale":
                    return replace(result, lease_token=lease.token + 1)
                return replace(result, slots=tuple(reversed(result.slots)))

        clock = ManualClock()
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(max_slots=2, wait_ms=200), clock=clock, backend=Reverse(("cpu0",)),
        ) as runtime:
            install(runtime, "affine-r1")
            one = replace(request("one"), tensor=Tensor((1, 3), (1, 0, 0)))
            two = replace(request("two"), tensor=Tensor((1, 3), (2, 0, 0)))
            for item in (one, two):
                runtime.submit(item)
            self.assertEqual(runtime.wait(one.key, timeout=3).output.values, (2.5, -2.0))
            self.assertEqual(runtime.wait(two.key, timeout=3).output.values, (4.5, -2.0))
            stale = request("stale")
            runtime.submit(stale)
            clock.advance_ms(200)
            runtime.pulse()
            self.assertEqual(runtime.wait(stale.key, timeout=3).failure.code, ErrorCode.BAD_OUTPUT)
