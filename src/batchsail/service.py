"""Signal-driven service lifecycle for bounded shutdown and HTTP draining."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from signal import SIGINT, SIGTERM, Signals, getsignal, signal
from threading import Event
from types import FrameType
from typing import Any

from .http import SailServer
from .runtime import Runtime


class ServiceSignals(AbstractContextManager):
    def __init__(self) -> None:
        self.requested = Event()
        self._previous: dict[Signals, Any] = {}

    def _stop(self, signum: int, frame: FrameType | None) -> None:
        self.requested.set()

    def __enter__(self) -> "ServiceSignals":
        try:
            for signum in (SIGINT, SIGTERM):
                self._previous[signum] = getsignal(signum)
                signal(signum, self._stop)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for signum, previous in self._previous.items():
            signal(signum, previous)
        self._previous.clear()

    def __exit__(self, *exc: object) -> None:
        self._restore()


def serve(
    server: SailServer,
    runtime: Runtime,
    *,
    drain_seconds: float,
    announce: Callable[[dict[str, Any]], None],
) -> None:
    server.timeout = 0.2
    with ServiceSignals() as signals:
        host, port = server.server_address
        announce({"listening": {"host": host, "port": port}})
        while not signals.requested.is_set():
            server.handle_request()
        result = runtime.drain(timeout=drain_seconds)
        announce({"shutdown": result.to_json()})
