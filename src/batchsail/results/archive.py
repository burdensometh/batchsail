"""Streaming v1 JSON Lines archives for offline transfer of terminal results."""

from collections.abc import Iterator
from contextlib import closing
import os
from pathlib import Path
import tempfile

from ..codec import decode, encode, sync_directory
from ..errors import Invalid
from .model import Receipt
from .store import ReceiptStore

MAX_LINE_BYTES = 32 * 1024 * 1024
_PREFIX = b'{"fingerprint":'
_MIDDLE = b',"result":'
_SUFFIX = b',"version":1}\n'


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
    """Write a v1 archive of terminal receipts visible when the export starts.

    The snapshot is fixed before the first byte is written: receipts that settle
    or arrive during the export are left for the next one. Envelopes are copied
    from SQLite in bounded chunks and never decoded, so peak extra memory tracks
    the largest single receipt rather than the whole archive. The previous file
    is replaced only after a complete flush, so an interrupted write leaves the
    earlier archive intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as file, closing(store.export_snapshot(tenant=tenant)) as snapshot:
            for fingerprint, envelope in snapshot:
                file.write(_PREFIX + encode(fingerprint) + _MIDDLE)
                for chunk in envelope:
                    file.write(chunk)
                file.write(_SUFFIX)
                count += 1
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return count
