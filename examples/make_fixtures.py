"""Rebuild the v1 fixtures from the checked-in deterministic definitions."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from batchsail.batcher import BatchingPolicy
from batchsail.clock import ManualClock
from batchsail.codec import atomic_write, decode, encode
from batchsail.config import RuntimeConfig
from batchsail.registry.authoring import pack_definition
from batchsail.request import Request
from batchsail.results import ReceiptStore, ResultStatus
from batchsail.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    models = args.output / "models"
    names = ("affine-r1", "affine-r2", "filter-r1")
    for name in names:
        definition = ROOT / "fixtures" / "definitions" / f"{name}.json"
        pack_definition(definition, models / name)

    source = Request.from_json(decode((ROOT / "fixtures/request.json").read_bytes()))
    config = RuntimeConfig(backend="inline", batching=BatchingPolicy(wait_ms=0))
    clock = ManualClock(wall_ms=1_700_000_000_000)
    with TemporaryDirectory() as directory:
        workspace = Path(directory)
        with Runtime(workspace, config, clock=clock) as runtime:
            revision = runtime.install(models / "affine-r1")
            runtime.activate(revision.key)
            runtime.submit(source)
            result = runtime.wait(source.key, timeout=5)
            if result.status != ResultStatus.SUCCEEDED or result.output.values != (2.0, 7.0, -0.5, 0.0):
                raise RuntimeError(f"fixture inference failed: {result.to_json()}")
        store = ReceiptStore(workspace / "receipts.sqlite")
        try:
            # The checked-in archive is the historical version-one fixture. Emit
            # it in its original shape so rebuilding the fixtures is byte-stable.
            receipt = store.receipt(source.key)
            archive = args.output / "receipts.v1.jsonl"
            payload = encode(receipt.to_v1()) + b"\n"
            atomic_write(archive, payload)
        finally:
            store.close()


if __name__ == "__main__":
    main()
