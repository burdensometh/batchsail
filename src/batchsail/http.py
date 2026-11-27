"""Small HTTP v1 transport for trusted networks and local demonstrations."""

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
from pathlib import Path
import socket
from threading import BoundedSemaphore
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .codec import decode, encode, fields, identifier, integer
from .errors import ErrorCode, Invalid, SailError
from .registry.model import ModelKey
from .request.model import Request, RequestKey
from .results.model import ResultStatus
from .runtime import Runtime

_LOG = logging.getLogger(__name__)
_MAX_BODY = 32 * 1024 * 1024

_ERROR_STATUS = {
    ErrorCode.INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.UNKNOWN_MODEL: HTTPStatus.NOT_FOUND,
    ErrorCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorCode.CONFLICT: HTTPStatus.CONFLICT,
    ErrorCode.QUEUE_FULL: HTTPStatus.TOO_MANY_REQUESTS,
    ErrorCode.IMPOSSIBLE: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
}


@dataclass(frozen=True, slots=True)
class Reply:
    status: HTTPStatus
    body: Any
    location: str | None = None


def _natural(value: str, label: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if not value or len(value) > 19 or not value.isascii() or not value.isdecimal():
        raise Invalid(f"{label} must be a decimal integer")
    return integer(int(value), label, minimum=minimum, maximum=maximum)


def _options(query: str, allowed: set[str]) -> dict[str, str]:
    try:
        values = parse_qs(query, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
    except ValueError as exc:
        raise Invalid("invalid query parameters") from exc
    if set(values).difference(allowed):
        raise Invalid("unexpected query parameter")
    if any(len(entries) != 1 for entries in values.values()):
        raise Invalid("each query parameter can only occur once")
    return {key: entries[0] for key, entries in values.items()}


def _pagination(options: dict[str, str]) -> dict[str, int]:
    return {
        "after": _natural(options.get("after", "0"), "after"),
        "limit": _natural(options.get("limit", "100"), "limit", minimum=1, maximum=1000),
    }


class SailServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        runtime: Runtime,
        *,
        bundle_root: Path | None = None,
        max_connections: int = 64,
    ):
        self.runtime = runtime
        self.bundle_root = None if bundle_root is None else bundle_root.resolve()
        self._slots = BoundedSemaphore(max_connections)
        super().__init__(address, SailHandler)

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        connection.settimeout(15)
        return connection, address

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(0.2)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class SailHandler(BaseHTTPRequestHandler):
    server: SailServer
    server_version = "BatchSail/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Request payloads and URLs can contain user data; runtime events
        # contain only the routing identity needed to correlate work.
        _LOG.debug("HTTP connection from %s", self.client_address[0])

    def _body(self) -> Any:
        if self.headers.get("Transfer-Encoding") is not None:
            raise Invalid("chunked request bodies are unsupported")
        if self.headers.get_content_type() != "application/json":
            raise Invalid("request body must be application/json")
        lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(lengths) != 1:
            raise Invalid("a single Content-Length header is required")
        size = _natural(lengths[0], "Content-Length", minimum=1, maximum=_MAX_BODY)
        data = self.rfile.read(size)
        if len(data) != size:
            raise Invalid("incomplete request body")
        return decode(data)

    def _reply(self, response: Reply) -> None:
        payload = encode(response.body)
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if response.location is not None:
            self.send_header("Location", response.location)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _handle(self) -> None:
        try:
            response = self._route()
        except SailError as exc:
            status = _ERROR_STATUS.get(exc.code, HTTPStatus.INTERNAL_SERVER_ERROR)
            response = Reply(status, {"error": exc.to_json()})
        except TimeoutError:
            response = Reply(HTTPStatus.REQUEST_TIMEOUT, {"error": {"code": "timeout", "message": "read timed out"}})
        except ConnectionError:
            return
        except Exception:
            _LOG.exception("HTTP request failed")
            response = Reply(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"code": "execution", "message": "request could not be processed"}},
            )
        try:
            self._reply(response)
        except (ConnectionError, TimeoutError):
            return

    def _route(self) -> Reply:
        try:
            target = urlsplit(self.path)
            segments = tuple(unquote(part, errors="strict") for part in target.path.split("/")[1:])
        except (UnicodeError, ValueError) as exc:
            raise Invalid("invalid request path") from exc
        if target.fragment or target.netloc or target.scheme:
            raise Invalid("request target must be a local path")
        runtime = self.server.runtime
        if self.command == "GET":
            if segments == ("v1", "status"):
                _options(target.query, set())
                return Reply(HTTPStatus.OK, runtime.snapshot().to_json())
            if segments == ("v1", "events"):
                options = _options(target.query, {"after", "limit"})
                return Reply(HTTPStatus.OK, runtime.events(**_pagination(options)))
            if segments == ("v1", "terminal-results"):
                return self._terminal_stream(target.query)
            if segments == ("v1", "requests"):
                return self._history(target.query)
        if segments == ("v1", "requests") and self.command == "POST":
            _options(target.query, set())
            admission = runtime.submit(Request.from_json(self._body()))
            location = f"/v1/requests/{admission.key.tenant}/{admission.key.request_id}"
            status = HTTPStatus.OK if admission.reused else HTTPStatus.ACCEPTED
            return Reply(status, admission.to_json(), location)
        if len(segments) == 4 and segments[:2] == ("v1", "requests"):
            return self._request(RequestKey(*segments[2:]), target.query)
        if segments[:2] == ("v1", "catalog") and self.command == "POST":
            _options(target.query, set())
            if segments == ("v1", "catalog", "activate"):
                return self._activate()
            if segments == ("v1", "catalog", "install"):
                return self._install()
        if segments == ("v1", "devices", "trim") and self.command == "POST":
            _options(target.query, set())
            body = fields(self._body(), (), "trim request", ("device", "revision"))
            revision = None if body.get("revision") is None else ModelKey.from_json(body["revision"])
            result = runtime.trim(device=body.get("device"), revision=revision)
            return Reply(HTTPStatus.OK, {"devices": [entry.to_json() for entry in result]})
        if segments == ("v1", "devices", "estimate") and self.command == "POST":
            _options(target.query, set())
            body = fields(self._body(), ("model", "rows"), "capacity estimate", ("slots", "revision"))
            estimate = runtime.estimate(
                body["model"], rows=body["rows"], slots=body.get("slots", 1), revision=body.get("revision"),
            )
            return Reply(HTTPStatus.OK, estimate.to_json())
        raise SailError(ErrorCode.NOT_FOUND, "unknown endpoint")

    def _request(self, key: RequestKey, query: str) -> Reply:
        runtime = self.server.runtime
        if self.command == "DELETE":
            _options(query, set())
            return Reply(HTTPStatus.OK, runtime.cancel(key).to_json())
        if self.command == "GET":
            options = _options(query, {"wait_ms"})
            wait = _natural(options.get("wait_ms", "0"), "wait_ms", maximum=10_000)
            result = runtime.result(key) if wait == 0 else runtime.wait(key, timeout=wait / 1000)
            status = HTTPStatus.OK if result.status.terminal else HTTPStatus.ACCEPTED
            return Reply(status, result.to_json())
        raise SailError(ErrorCode.NOT_FOUND, "unknown endpoint")

    def _history(self, query: str) -> Reply:
        options = _options(query, {"tenant", "status", "after", "limit"})
        try:
            status = None if "status" not in options else ResultStatus(options["status"])
        except ValueError as exc:
            raise Invalid("unknown status filter") from exc
        return Reply(
            HTTPStatus.OK,
            self.server.runtime.history(status=status, tenant=options.get("tenant"), **_pagination(options)),
        )

    def _terminal_stream(self, query: str) -> Reply:
        options = _options(query, {"after", "limit", "wait_ms"})
        wait = _natural(options.get("wait_ms", "0"), "wait_ms", maximum=10_000)
        page = self.server.runtime.terminal_stream(
            **_pagination(options), timeout=0 if wait == 0 else wait / 1000,
        )
        return Reply(HTTPStatus.OK, page.to_json())

    def _activate(self) -> Reply:
        body = fields(self._body(), ("model", "revision"), "activation", ("expected_generation",))
        key = ModelKey(body["model"], body["revision"])
        generation = self.server.runtime.activate(key, expected_generation=body.get("expected_generation"))
        return Reply(HTTPStatus.OK, {**key.to_json(), "generation": generation})

    def _install(self) -> Reply:
        root = self.server.bundle_root
        if root is None:
            raise SailError(ErrorCode.NOT_FOUND, "online bundle installation is disabled")
        body = fields(self._body(), ("bundle",), "installation")
        name = identifier(body["bundle"], "bundle")
        bundle = (root / name).resolve()
        if bundle.parent != root:
            raise Invalid("bundle directory must be an immediate child of the configured root")
        revision = self.server.runtime.install(bundle)
        return Reply(HTTPStatus.OK, revision.to_v1())
