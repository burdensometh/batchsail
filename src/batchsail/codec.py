"""Strict JSON and filesystem primitives for version-one data."""

from collections.abc import Collection, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .errors import Invalid

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Invalid(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise Invalid(f"invalid JSON constant {value}")


def decode(data: str | bytes) -> Any:
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise Invalid(f"cannot decode JSON: {exc}") from exc


def encode(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


def fields(
    value: Any,
    required: Collection[str],
    label: str,
    optional: Collection[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Invalid(f"{label} must be an object")
    missing = set(required).difference(value)
    unexpected = set(value).difference(required, optional)
    if missing or unexpected:
        extras = sorted(map(str, unexpected))
        raise Invalid(f"{label} has missing fields {sorted(missing)} and extra fields {extras}")
    return value


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise Invalid(f"{label} must be an identifier of 1 to 64 characters")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise Invalid(f"{label} must be a lowercase SHA-256")
    return value


def integer(value: Any, label: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise Invalid(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise Invalid(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise Invalid(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise Invalid(f"{label} must be a finite number")
    return result


def read_bytes(path: Path, *, maximum: int) -> bytes:
    try:
        with path.open("rb") as file:
            data = file.read(maximum + 1)
    except OSError as exc:
        raise Invalid(f"cannot read {path}: {exc.strerror}") from exc
    if len(data) > maximum:
        raise Invalid(f"{path} exceeds the {maximum} byte limit")
    return data


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
