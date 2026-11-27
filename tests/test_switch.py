from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import unittest

from batchsail.codec import decode, encode
from batchsail.errors import ErrorCode, Invalid, SailError
from batchsail.http import SailServer
from batchsail.registry import ModelKey
from batchsail.registry.authoring import pack_definition
from batchsail.request.client import HttpClient
from batchsail.runtime import Runtime

from support import MODELS, ROOT, config, install, request


class AtomicSwitchTests(unittest.TestCase):
    def test_switch_moves_every_model_in_a_single_generation(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1", "affine-r2", "filter-r1")
            # support.install activates each r1: affine -> 1, filter -> 2.
            self.assertEqual(runtime.snapshot().generation, 2)
            affine_r1 = ModelKey("affine", "r1")
            filter_r1 = ModelKey("filter", "r1")
            result = runtime.switch_many((affine_r1, filter_r1))
            # Both targets are already active, so this is a no-op.
            self.assertEqual(result.generation, 2)
            self.assertEqual(runtime.snapshot().generation, 2)
            self.assertEqual(
                result.active,
                {"affine": affine_r1, "filter": filter_r1},
            )
            views = {view.model.key: view for view in runtime.snapshot().models}
            self.assertTrue(views[affine_r1].active)
            self.assertTrue(views[filter_r1].active)

            result = runtime.switch_many(
                (ModelKey("affine", "r2"), filter_r1), expected_generation=2
            )
            # Two target entries, but only affine changed: one generation step.
            self.assertEqual(result.generation, 3)
            admitted = request("switched", model="affine")
            runtime.submit(admitted)
            self.assertEqual(runtime.wait(admitted.key, timeout=3).model.revision, "r2")
            filtered = request("filtered", model="filter", tenant="beta")
            runtime.submit(filtered)
            self.assertEqual(runtime.wait(filtered.key, timeout=3).model.revision, "r1")

    def test_any_unknown_or_unrunnable_target_preserves_the_whole_catalog(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config(wait_ms=0)) as runtime:
                install(runtime, "affine-r1", "affine-r2", "filter-r1")
                # A second filter revision with two weight pages: one slot needs
                # 8192 + 128 bytes, which cannot run on a 6144-byte device.
                big_filter = root / "big-filter"
                definition = decode((ROOT / "fixtures/definitions/filter-r1.json").read_bytes())
                definition["revision"] = "r2"
                definition["weight_pages"] = 2
                definition_path = root / "filter-r2.json"
                definition_path.write_bytes(encode(definition))
                packed = pack_definition(definition_path, big_filter)
                runtime.install(big_filter)
                before_generation = runtime.snapshot().generation
                before_active = {
                    view.model.key.name: view.model.key
                    for view in runtime.snapshot().models if view.active
                }

                missing = (ModelKey("affine", "r2"), ModelKey("filter", "r9"))
                with self.assertRaises(SailError) as unknown:
                    runtime.switch_many(missing)
                self.assertEqual(unknown.exception.code, ErrorCode.UNKNOWN_MODEL)
                self._catalog_unchanged(runtime, before_generation, before_active)
                self.assertEqual(packed.key, ModelKey("filter", "r2"))

            # Reopen against devices that fit affine (4224) but not the big
            # filter (8320), while still allowing the installed one-page filter.
            with Runtime(root, config(devices=(6144,), wait_ms=0)) as cramped:
                generation = cramped.snapshot().generation
                active = {
                    view.model.key.name: view.model.key
                    for view in cramped.snapshot().models if view.active
                }
                with self.assertRaises(SailError) as impossible:
                    cramped.switch_many(
                        (ModelKey("affine", "r2"), ModelKey("filter", "r2"))
                    )
                self.assertEqual(impossible.exception.code, ErrorCode.IMPOSSIBLE)
                self._catalog_unchanged(cramped, generation, active)

                # The runnable subset still commits afterwards exactly once.
                result = cramped.switch_many((ModelKey("affine", "r2"),))
                self.assertEqual(result.generation, generation + 1)
                self.assertEqual(cramped.snapshot().generation, generation + 1)

    def _catalog_unchanged(self, runtime, generation, active):
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot.generation, generation)
        current = {
            view.model.key.name: view.model.key
            for view in snapshot.models if view.active
        }
        self.assertEqual(current, active)
        self.assertFalse(any(view.draining for view in snapshot.models))

    def test_expected_generation_conflict_changes_nothing(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1", "affine-r2", "filter-r1")
            generation = runtime.snapshot().generation
            with self.assertRaises(SailError) as conflict:
                runtime.switch_many(
                    (ModelKey("affine", "r2"), ModelKey("filter", "r1")),
                    expected_generation=generation + 1,
                )
            self.assertEqual(conflict.exception.code, ErrorCode.CONFLICT)
            self.assertEqual(runtime.snapshot().generation, generation)
            admitted = request("still-r1")
            runtime.submit(admitted)
            self.assertEqual(runtime.wait(admitted.key, timeout=3).model.revision, "r1")

    def test_noop_switch_keeps_the_generation_stable(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1", "filter-r1")
            generation = runtime.snapshot().generation
            result = runtime.switch_many(
                (ModelKey("affine", "r1"), ModelKey("filter", "r1"))
            )
            self.assertEqual(result.generation, generation)
            self.assertEqual(runtime.snapshot().generation, generation)

    def test_empty_or_duplicate_targets_are_invalid_without_touching_generation(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1")
            generation = runtime.snapshot().generation
            with self.assertRaises(Invalid):
                runtime.switch_many(())
            with self.assertRaises(Invalid):
                runtime.switch_many(
                    (ModelKey("affine", "r1"), ModelKey("affine", "r1"))
                )
            self.assertEqual(runtime.snapshot().generation, generation)

    def test_switch_persists_and_reopens_with_the_switched_directory_view(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config(wait_ms=0)) as runtime:
                install(runtime, "affine-r1", "affine-r2", "filter-r1")
                generation = runtime.switch_many(
                    (ModelKey("affine", "r2"), ModelKey("filter", "r1"))
                ).generation
            with Runtime(root, config(wait_ms=0)) as runtime:
                self.assertEqual(runtime.snapshot().generation, generation)
                affine = request("after-restart")
                runtime.submit(affine)
                self.assertEqual(runtime.wait(affine.key, timeout=3).model.revision, "r2")
                filtered = request("after-restart-filter", model="filter", tenant="beta")
                runtime.submit(filtered)
                self.assertEqual(runtime.wait(filtered.key, timeout=3).model.revision, "r1")

    def test_concurrent_switch_requests_serialize_on_expected_generation(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1", "affine-r2", "filter-r1")
            start = runtime.snapshot().generation
            barrier = Barrier(2)
            outcomes = []

            def drive(expected):
                barrier.wait()
                try:
                    outcomes.append(runtime.switch_many(
                        (ModelKey("affine", "r2"), ModelKey("filter", "r1")),
                        expected_generation=expected,
                    ).generation)
                except SailError as exc:
                    outcomes.append(exc.code)

            threads = [
                Thread(target=drive, args=(start,)),
                Thread(target=drive, args=(start,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(outcomes), 2)
            self.assertIn(start + 1, outcomes)
            self.assertIn(ErrorCode.CONFLICT, outcomes)
            self.assertEqual(runtime.snapshot().generation, start + 1)

    def test_admitted_work_keeps_its_revision_after_an_atomic_switch(self):
        with TemporaryDirectory() as directory, Runtime(
            Path(directory), config(devices=(8192, 8192), wait_ms=0),
        ) as runtime:
            install(runtime, "affine-r1", "affine-r2", "filter-r1")
            old = request("pinned-old")
            admission = runtime.submit(old)
            self.assertEqual(admission.revision, "r1")
            runtime.switch_many(
                (ModelKey("affine", "r2"), ModelKey("filter", "r1"))
            )
            # The old revision is protected by the accepted request until it settles.
            draining = next(
                view for view in runtime.snapshot().models
                if view.model.key == ModelKey("affine", "r1")
            )
            self.assertTrue(draining.draining)
            # The already accepted request still completes against r1.
            self.assertEqual(runtime.wait(old.key, timeout=3).model.revision, "r1")
            # New admissions observe the post-switch directory.
            new = request("pinned-new")
            runtime.submit(new)
            self.assertEqual(runtime.wait(new.key, timeout=3).model.revision, "r2")


class SwitchTransportTests(unittest.TestCase):
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
        self.client = HttpClient(
            f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        )

    def test_http_switch_endpoint_roundtrip_and_failure(self):
        for name in ("affine-r1", "affine-r2", "filter-r1"):
            revision = self.client.install(name)
            if revision.key.revision == "r1":
                self.client.activate(revision.key)
        generation = self.runtime.snapshot().generation
        switched, active = self.client.switch_many(
            (ModelKey("affine", "r2"), ModelKey("filter", "r1")),
            expected_generation=generation,
        )
        self.assertEqual(switched, generation + 1)
        self.assertEqual(active["affine"], ModelKey("affine", "r2"))
        self.assertEqual(active["filter"], ModelKey("filter", "r1"))

        from http.client import HTTPConnection
        connection = HTTPConnection(*self.server.server_address, timeout=3)
        try:
            connection.request(
                "POST", "/v1/catalog/switch",
                body=encode({"activations": [{"model": "filter", "revision": "r9"}]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
        finally:
            connection.close()
        self.assertEqual(self.runtime.snapshot().generation, generation + 1)
