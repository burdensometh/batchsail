"""Length-delimited transport for one locally owned worker process."""

import pickle
import socket
from struct import Struct
from time import monotonic

_HEADER = Struct("!Q")
MAX_FRAME_BYTES = 768 * 1024 * 1024


class WireError(Exception):
    pass


class FrameChannel:
    """The worker and parent each own one end of a private socketpair."""

    def __init__(self, channel: socket.socket):
        self._channel = channel

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        duration = deadline - monotonic()
        if duration <= 0:
            raise TimeoutError("worker transport deadline elapsed")
        return duration

    def send(self, payload: object, *, deadline: float | None = None) -> None:
        try:
            message = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OverflowError) as exc:
            raise WireError("cannot serialize worker message") from exc
        if not 0 < len(message) <= MAX_FRAME_BYTES:
            raise WireError("worker message exceeds the transport size limit")
        self._channel.settimeout(self._remaining(deadline))
        self._channel.sendall(_HEADER.pack(len(message)))
        self._channel.settimeout(self._remaining(deadline))
        self._channel.sendall(message)

    def _read(self, length: int, deadline: float | None) -> bytearray:
        data = bytearray(length)
        view = memoryview(data)
        received = 0
        while received < length:
            self._channel.settimeout(self._remaining(deadline))
            count = self._channel.recv_into(view[received:])
            if count == 0:
                if received:
                    raise WireError("worker closed a partial message")
                raise EOFError("worker closed its channel")
            received += count
        return data

    def receive(self, *, deadline: float | None = None) -> object:
        size, = _HEADER.unpack(self._read(_HEADER.size, deadline))
        if not 0 < size <= MAX_FRAME_BYTES:
            raise WireError("worker declared an invalid transport size")
        try:
            return pickle.loads(self._read(size, deadline))
        except (pickle.PickleError, EOFError, ImportError, AttributeError, ValueError) as exc:
            raise WireError("cannot decode worker message") from exc

    def close(self) -> None:
        self._channel.close()
