from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from batchsail.clock import ManualClock
from batchsail.codec import decode, encode
from batchsail.config import WorkspaceGuard
from batchsail.errors import Conflict, ErrorCode, Invalid
from batchsail.registry import ModelKey, ModelStore
from batchsail.registry.authoring import pack_definition
from batchsail.request import RequestKey
from batchsail.results import ReceiptStore, ResultEnvelope, ResultStatus
from batchsail.results.archive import export_archive, import_archive
from batchsail.results.model import Receipt
from batchsail.runtime import Runtime
from batchsail.tensor import Tensor

from support import MODELS, ROOT, config, install, request


class ModelV1Tests(unittest.TestCase):
    def test_fixture_bundle_roundtrip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("affine-r1", "affine-r2", "filter-r1"):
                with self.subTest(name=name):
                    expected, expected_raw = ModelStore.read_bundle(MODELS / name)
                    written = pack_definition(ROOT / "fixtures" / "definitions" / f"{name}.json", root / name)
                    actual, raw = ModelStore.read_bundle(root / name)
                    self.assertEqual(expected, written)
                    self.assertEqual(expected, actual)
                    self.assertEqual(expected_raw, raw)

    def test_bundle_detects_modified_weights(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            pack_definition(ROOT / "fixtures/definitions/affine-r1.json", bundle)
            weight_file = bundle / "weights.v1.json"
            weight_file.write_bytes(weight_file.read_bytes() + b" ")
            with self.assertRaisesRegex(Invalid, "SHA-256"):
                ModelStore.read_bundle(bundle)

    def test_immutable_install_and_revision_generation_survive_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Runtime(root, config()) as runtime:
                install(runtime, "affine-r1", "affine-r2")
                self.assertEqual(runtime.snapshot().generation, 1)
                self.assertEqual(runtime.activate(ModelKey("affine", "r2"), expected_generation=1), 2)
                with self.assertRaises(Conflict):
                    runtime.activate(ModelKey("affine", "r1"), expected_generation=1)
                self.assertEqual(runtime.activate(ModelKey("affine", "r2")), 2)
                with self.assertRaises(Conflict):
                    WorkspaceGuard(root)
            with Runtime(root, config(wait_ms=0)) as runtime:
                item = request("new")
                self.assertEqual(runtime.submit(item).revision, "r2")
                self.assertEqual(runtime.wait(item.key, timeout=3).model.revision, "r2")
                self.assertEqual(runtime.snapshot().generation, 2)

    def test_rejects_duplicate_fields_and_boolean_version(self):
        with self.assertRaisesRegex(Invalid, "duplicate"):
            decode('{"version": 1, "version": 2}')
        revision, _ = ModelStore.read_bundle(MODELS / "affine-r1")
        forged = revision.to_v1()
        forged["version"] = True
        with self.assertRaises(Invalid):
            type(revision).from_v1(forged)


class ReceiptV1Tests(unittest.TestCase):
    def test_historical_fixture_idempotent_import_and_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReceiptStore(root / "original.sqlite")
            replica = ReceiptStore(root / "replica.sqlite")
            try:
                sample = ROOT / "fixtures" / "receipts.v1.jsonl"
                self.assertEqual(import_archive(store, sample), 1)
                self.assertEqual(import_archive(store, sample), 0)
                result = store.result(RequestKey("sample", "first"))
                self.assertEqual(result.output.values, (2.0, 7.0, -0.5, 0.0))
                path = root / "copy.jsonl"
                self.assertEqual(export_archive(store, path, tenant="sample"), 1)
                self.assertEqual(path.read_bytes(), sample.read_bytes())
                self.assertEqual(import_archive(replica, path), 1)
                self.assertEqual(replica.result(result.key), result)
            finally:
                replica.close()
                store.close()

    def test_archive_import_conflict_rolls_back_previous_rows(self):
        with TemporaryDirectory() as directory:
            store = ReceiptStore(Path(directory) / "results.sqlite")
            try:
                import_archive(store, ROOT / "fixtures" / "receipts.v1.jsonl")
                existing = store.find(RequestKey("sample", "first"))
                other = replace(existing, result=replace(existing.result, key=RequestKey("sample", "other")))
                with self.assertRaises(Conflict):
                    store.import_terminal((other, replace(existing, fingerprint="0" * 64)))
                self.assertIsNone(store.find(other.result.key))
            finally:
                store.close()

    def _succeeded(self, name, *, tenant="alpha", fingerprint="a" * 64):
        key = RequestKey(tenant, name)
        return Receipt(
            fingerprint,
            ResultEnvelope(
                key, ModelKey("affine", "r1"), ResultStatus.SUCCEEDED, 1000,
                attempt=1, batch_id="b", slot=0, device="cpu0", queue_ns=0, elapsed_ns=0,
                output=Tensor((1, 2), (2.0, 7.0)),
            ),
        )

    def test_export_snapshot_is_fixed_when_the_export_starts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReceiptStore(root / "results.sqlite")
            try:
                store.insert(self._succeeded("before"))
                store.insert(Receipt(
                    self._succeeded("pending").fingerprint,
                    ResultEnvelope(
                        RequestKey("alpha", "pending"), ModelKey("affine", "r1"),
                        ResultStatus.PENDING, 1000,
                    ),
                ))
                snapshot = store.export_snapshot()
                # A row that becomes terminal and a brand-new row appear after the
                # snapshot opened; neither belongs to this export.
                store.settle(
                    ResultEnvelope(
                        RequestKey("alpha", "pending"), ModelKey("affine", "r1"),
                        ResultStatus.SUCCEEDED, 1000, attempt=1, batch_id="b", slot=0,
                        device="cpu0", queue_ns=0, elapsed_ns=0,
                        output=Tensor((1, 2), (1.0, 1.0)),
                    ),
                    expected_attempt=1,
                )
                store.insert(self._succeeded("later"))
                names = [decode(b"".join(chunks))["request_id"] for _, chunks in snapshot]
                self.assertEqual(names, ["before"])
            finally:
                store.close()
            count = export_archive(store, root / "next.jsonl")
            self.assertEqual(count, 2)

    def test_export_byte_roundtrip_matches_canonical_encoding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReceiptStore(root / "results.sqlite")
            try:
                receipt = self._succeeded("one")
                store.insert(receipt)
                path = root / "one.jsonl"
                self.assertEqual(export_archive(store, path), 1)
                self.assertEqual(path.read_bytes(), encode(receipt.to_v1()) + b"\n")
            finally:
                store.close()

    def test_export_streams_large_tensors_within_bounded_memory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReceiptStore(root / "results.sqlite")
            try:
                values = Tensor((512, 512), tuple(0.0 for _ in range(512 * 512)))
                for index in range(8):
                    key = RequestKey("alpha", f"big{index}")
                    store.insert(Receipt(
                        f"{index:064x}",
                        ResultEnvelope(
                            key, ModelKey("affine", "r1"), ResultStatus.SUCCEEDED, 1000,
                            attempt=1, batch_id="b", slot=0, device="cpu0", queue_ns=0,
                            elapsed_ns=0, output=values,
                        ),
                    ))
                path = root / "big.jsonl"
                self.assertEqual(export_archive(store, path), 8)
                self.assertEqual(import_archive(ReceiptStore(root / "replica.sqlite"), path), 8)
            finally:
                store.close()

    def test_export_write_failure_keeps_previous_archive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReceiptStore(root / "results.sqlite")
            try:
                store.insert(self._succeeded("one"))
                target = root / "archive.jsonl"
                self.assertEqual(export_archive(store, target), 1)
                original = target.read_bytes()

                def broken(chunks):
                    yield from chunks
                    raise OSError("disk full")

                original_snapshot = store.export_snapshot

                def failing_snapshot(*, tenant=None):
                    for fingerprint, chunks in original_snapshot(tenant=tenant):
                        yield fingerprint, broken(chunks)

                store.export_snapshot = failing_snapshot
                with self.assertRaises(OSError):
                    export_archive(store, target)
                store.export_snapshot = original_snapshot
                self.assertEqual(target.read_bytes(), original)
                leftovers = [p.name for p in root.iterdir() if p.name.startswith(f".{target.name}.")]
                self.assertEqual(leftovers, [])
            finally:
                store.close()

    def test_export_does_not_block_settlement_until_it_finishes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReceiptStore(root / "results.sqlite")
            try:
                store.insert(Receipt(
                    self._succeeded("running").fingerprint,
                    ResultEnvelope(
                        RequestKey("alpha", "running"), ModelKey("affine", "r1"),
                        ResultStatus.PENDING, 1000,
                    ),
                ))
                settled = threading.Event()

                def settle():
                    store.settle(
                        ResultEnvelope(
                            RequestKey("alpha", "running"), ModelKey("affine", "r1"),
                            ResultStatus.SUCCEEDED, 1000, attempt=1, batch_id="b", slot=0,
                            device="cpu0", queue_ns=0, elapsed_ns=0,
                            output=Tensor((1, 2), (3.0, 4.0)),
                        ),
                        expected_attempt=1,
                    )
                    settled.set()

                snapshot = store.export_snapshot()
                worker = threading.Thread(target=settle)
                worker.start()
                self.assertTrue(settled.wait(2), "settlement blocked behind the export snapshot")
                worker.join()
                self.assertEqual(list(snapshot), [])
            finally:
                store.close()

    def test_recovery_does_not_resume_old_pending_and_allows_fresh_retry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            clock = ManualClock(wall_ms=1000)
            with Runtime(root, config(), clock=clock) as runtime:
                install(runtime, "affine-r1")
            pending = request("interrupted")
            store = ReceiptStore(root / "receipts.sqlite")
            try:
                envelope = ResultEnvelope(pending.key, ModelKey("affine", "r1"), ResultStatus.PENDING, 1000)
                store.insert(Receipt(pending.fingerprint, envelope))
                store.assign(((pending.key, 1, "interrupted-batch", 0, "cpu0", 2_000_000),))
            finally:
                store.close()
            with Runtime(root, config(wait_ms=0), clock=clock) as runtime:
                recovered = runtime.result(pending.key)
                self.assertEqual(recovered.failure.code, ErrorCode.INTERRUPTED)
                self.assertEqual(recovered.elapsed_ns, 2_000_000)
                self.assertTrue(runtime.submit(pending).reused)
                self.assertEqual(runtime.snapshot().recovered, 1)
                fresh = request("fresh", retry_of=pending.key.request_id)
                runtime.submit(fresh)
                self.assertEqual(runtime.wait(fresh.key, timeout=3).status, ResultStatus.SUCCEEDED)
