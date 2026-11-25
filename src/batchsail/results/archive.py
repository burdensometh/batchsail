"""Streaming v1 JSON Lines archives for offline transfer of terminal results."""

from collections.abc import Iterator
import os
from pathlib import Path
import tempfile

from ..codec import decode, encode, sync_directory
from ..errors import Invalid
from .model import Receipt
from .store import ReceiptStore

MAX_LINE_BYTES = 32 * 1024 * 1024


def read_archive(path: Path) -> Iterator[Receipt]:
    with path.open("rb") as file:
        index = 1
        while line := file.readline(MAX_LINE_BYTES + 1):
            if len(line) > MAX_LINE_BYTES:
                raise Invalid(f"{path}:{index}: receipt exceeds {MAX_LINE_BYTES} bytes")
            try:
                yield Receipt.from_v1(decode(line))
            except Invalid as exc:
                raise Invalid(f"{path}:{index}: {exc}") from exc
            index += 1


def import_archive(store: ReceiptStore, path: Path) -> int:
    return store.import_terminal(read_archive(path))


def export_archive(store: ReceiptStore, path: Path, *, tenant: str | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as file:
            cursor = 0
            while page := store.history(after=cursor, limit=25, tenant=tenant):
                for cursor, receipt in page:
                    if receipt.result.status.terminal:
                        file.write(encode(receipt.to_v1()) + b"\n")
                        count += 1
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return count
