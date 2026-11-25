"""Runtime configuration and exclusive writable workspace ownership."""

from dataclasses import dataclass
import errno
import fcntl
from pathlib import Path
import tomllib
from typing import IO

from .batcher import BatchingPolicy
from .codec import fields, integer, read_bytes
from .device import DeviceSpec, MemoryPolicy
from .errors import Conflict, Invalid


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    batching: BatchingPolicy = BatchingPolicy()
    memory: MemoryPolicy = MemoryPolicy()
    devices: tuple[DeviceSpec, ...] = (DeviceSpec("cpu0", 131_072),)
    backend: str = "process"
    worker_timeout_ms: int = 20_000
    event_capacity: int = 1024

    def __post_init__(self) -> None:
        if not 1 <= len(self.devices) <= 64:
            raise Invalid("configure one to sixty-four devices")
        if len({device.name for device in self.devices}) != len(self.devices):
            raise Invalid("device names must be unique")
        if self.backend not in ("inline", "process"):
            raise Invalid("backend must be inline or process")
        integer(self.worker_timeout_ms, "worker_timeout_ms", minimum=10, maximum=600_000)
        integer(self.event_capacity, "event_capacity", minimum=16, maximum=65_536)

    @classmethod
    def load(cls, path: Path) -> "RuntimeConfig":
        raw = read_bytes(path, maximum=65_536)
        try:
            document = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise Invalid(f"invalid TOML configuration: {exc}") from exc
        data = fields(document, (), "configuration", ("runtime", "memory", "batching", "devices"))
        defaults = cls()
        runtime = fields(data.get("runtime", {}), (), "runtime", ("backend", "worker_timeout_ms", "event_capacity"))
        memory = defaults.memory if "memory" not in data else MemoryPolicy.from_config(data["memory"])
        devices = defaults.devices
        if "devices" in data:
            if not isinstance(data["devices"], list):
                raise Invalid("devices must be an array of tables")
            devices = tuple(DeviceSpec.from_config(item) for item in data["devices"])
        return cls(
            batching=BatchingPolicy.from_config(data.get("batching", {})),
            memory=memory,
            devices=devices,
            **runtime,
        )


class WorkspaceGuard:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._file: IO[bytes] = (self.root / ".batchsail.lock").open("a+b")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise Conflict(f"workspace is already open: {self.root}") from exc
            raise

    def __enter__(self) -> "WorkspaceGuard":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()
