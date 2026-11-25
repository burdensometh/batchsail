"""Spawned CPU workers with stable transport, reset, and crash detection."""

from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
import socket
from threading import Lock
from time import monotonic

from ..batcher.batch import Batch
from ..device import DeviceLease, Residency
from ..errors import ErrorCode, SailError, WorkerLost
from .backend import BatchOutput, ResidentKernels
from .wire import FrameChannel, WireError


@dataclass(frozen=True, slots=True)
class RunCommand:
    batch: Batch
    lease: DeviceLease


@dataclass(frozen=True, slots=True)
class RunFailure:
    code: ErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class PruneCommand:
    keep: tuple[Residency, ...]


@dataclass(frozen=True, slots=True)
class Pruned:
    remaining: tuple[Residency, ...]


def _worker(connection: socket.socket) -> None:
    channel = FrameChannel(connection)
    cache = ResidentKernels()
    try:
        while True:
            command = channel.receive()
            if isinstance(command, PruneCommand):
                channel.send(Pruned(cache.prune(command.keep)))
            elif not isinstance(command, RunCommand):
                return
            else:
                try:
                    output = cache.run(command.batch, command.lease)
                except SailError as exc:
                    channel.send(RunFailure(exc.code, str(exc)))
                else:
                    channel.send(output)
    except (EOFError, OSError, WireError):
        return
    finally:
        channel.close()


class _Worker:
    def __init__(self, context: BaseContext, device: str):
        self._context = context
        self._device = device
        self._lock = Lock()
        self._process: BaseProcess | None = None
        self._channel: FrameChannel | None = None

    def _start(self) -> FrameChannel:
        if self._process is not None:
            if self._channel is None or not self._process.is_alive():
                raise WorkerLost(f"worker for {self._device} is unavailable")
            return self._channel

        parent, child = socket.socketpair()
        process = self._context.Process(
            target=_worker, args=(child,), name=f"batchsail-{self._device}", daemon=True,
        )
        try:
            process.start()
        except (OSError, RuntimeError) as exc:
            parent.close()
            raise WorkerLost(f"worker for {self._device} could not start") from exc
        finally:
            child.close()
        self._process = process
        self._channel = FrameChannel(parent)
        return self._channel

    def _request(self, command: RunCommand | PruneCommand, timeout_seconds: float) -> object:
        deadline = monotonic() + timeout_seconds
        with self._lock:
            channel = self._start()
        try:
            channel.send(command, deadline=deadline)
            return channel.receive(deadline=deadline)
        except TimeoutError as exc:
            raise WorkerLost(f"worker on {self._device} exceeded its transport timeout") from exc
        except (EOFError, OSError, WireError) as exc:
            raise WorkerLost(f"worker on {self._device} lost its transport") from exc

    def execute(self, command: RunCommand, timeout_seconds: float) -> BatchOutput:
        response = self._request(command, timeout_seconds)
        if isinstance(response, RunFailure):
            raise SailError(response.code, response.message)
        if not isinstance(response, BatchOutput):
            raise WorkerLost(f"worker on {self._device} returned an invalid transport message")
        return response

    def prune(self, keep: tuple[Residency, ...], *, timeout_seconds: float) -> tuple[Residency, ...]:
        response = self._request(PruneCommand(keep), timeout_seconds)
        if not isinstance(response, Pruned):
            raise WorkerLost(f"worker on {self._device} did not acknowledge cache maintenance")
        return response.remaining

    def interrupt(self) -> bool:
        """Terminate the current process without racing with connection ownership."""
        with self._lock:
            process = self._process
            if process is None or not process.is_alive():
                return False
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            return True

    def close(self) -> None:
        with self._lock:
            process, channel = self._process, self._channel
            if channel is not None:
                channel.close()
            if process is not None:
                process.join(0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            self._process = None
            self._channel = None


class ProcessBackend:
    def __init__(self, devices: tuple[str, ...], *, timeout_seconds: float = 20):
        self._timeout = timeout_seconds
        context = get_context("spawn")
        self._workers = {device: _Worker(context, device) for device in devices}

    def execute(self, batch: Batch, lease: DeviceLease) -> BatchOutput:
        return self._workers[lease.device].execute(RunCommand(batch, lease), self._timeout)

    def prune(self, device: str, keep: tuple[Residency, ...]) -> tuple[Residency, ...]:
        return self._workers[device].prune(keep, timeout_seconds=min(self._timeout, 2))

    def reset(self, device: str) -> None:
        self._workers[device].close()

    def interrupt(self, device: str) -> bool:
        return self._workers[device].interrupt()

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()
