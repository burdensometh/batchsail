"""Typed client for the BatchSail HTTP result and administration contracts."""

from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from time import monotonic, sleep
from typing import Any
from collections.abc import Iterator
from urllib.parse import urlencode, urlsplit

from ..codec import decode, encode, fields, identifier, integer
from ..errors import ErrorCode, Invalid, SailError
from ..registry.model import ModelKey, ModelRevision
from ..results.model import ResultEnvelope, ResultStatus
from .model import Admission, Request, RequestKey

MAX_JSON_RESPONSE = 64 * 1024 * 1024
MAX_LONG_POLL_MS = 10_000


class TransportFailure(SailError):
    """The remote outcome is unknown because no reliable response was received."""

    def __init__(self, message: str):
        super().__init__(ErrorCode.UNAVAILABLE, message)


@dataclass(frozen=True, slots=True)
class RemoteAddress:
    host: str
    port: int | None
    tls: bool

    @classmethod
    def parse(cls, value: str) -> "RemoteAddress":
        try:
            url = urlsplit(value)
            port = url.port
        except ValueError as exc:
            raise Invalid("invalid BatchSail endpoint URL") from exc
        if (
            url.scheme not in ("http", "https")
            or url.hostname is None
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.fragment
            or url.query
        ):
            raise Invalid("endpoint URL requires an http or https origin without path or credentials")
        return cls(url.hostname, port, url.scheme == "https")


class HttpClient:
    def __init__(self, address: str, *, timeout_seconds: float = 12, transport_retries: int = 1):
        if not 0 < timeout_seconds <= 120:
            raise Invalid("client timeout must be positive and at most 120 seconds")
        self.address = RemoteAddress.parse(address)
        self.timeout_seconds = timeout_seconds
        self._retries = integer(transport_retries, "transport retries", maximum=5)

    def _exchange(
        self, method: str, route: str, body: dict[str, Any] | None, timeout: float,
    ) -> tuple[int, Any]:
        connector = HTTPSConnection if self.address.tls else HTTPConnection
        connection = connector(self.address.host, self.address.port, timeout=timeout)
        try:
            headers = {"Accept": "application/json"}
            payload = None
            if body is not None:
                headers["Content-Type"] = "application/json"
                payload = encode(body)
            connection.request(method, route, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_JSON_RESPONSE + 1)
            if len(raw) > MAX_JSON_RESPONSE:
                raise TransportFailure("server response exceeds the JSON size limit")
            if not raw and response.status == 503:
                raise SailError(ErrorCode.UNAVAILABLE, "server connections are at capacity")
            if response.headers.get_content_type() != "application/json":
                raise TransportFailure("server did not return a JSON response")
            try:
                document = decode(raw)
            except Invalid as exc:
                raise TransportFailure("server returned malformed JSON") from exc
            return response.status, document
        except (OSError, HTTPException) as exc:
            raise TransportFailure(f"could not complete {method} request: {type(exc).__name__}") from exc
        finally:
            connection.close()

    def _send(
        self,
        method: str,
        route: str,
        expected: tuple[int, ...],
        *,
        body: dict[str, Any] | None = None,
        replayable: bool = False,
        limit: float | None = None,
    ) -> tuple[int, Any]:
        timeout = self.timeout_seconds if limit is None else min(self.timeout_seconds, limit)
        end = monotonic() + timeout
        tries = 1 + (self._retries if replayable else 0)
        for attempt in range(tries):
            remaining = end - monotonic()
            if remaining <= 0:
                raise TransportFailure(f"client timeout for {method} request")
            try:
                status, document = self._exchange(method, route, body, remaining)
            except TransportFailure:
                if attempt + 1 == tries or end <= monotonic():
                    raise
                sleep(min(0.02 * 2**attempt, max(0, end - monotonic())))
                continue
            if status not in expected:
                self._raise_status(status, document)
            return status, document
        raise AssertionError("client transport loop did not return or raise")

    @staticmethod
    def _raise_status(status: int, document: Any) -> None:
        try:
            obj = fields(document, ("error",), "HTTP error")
            error = fields(obj["error"], ("code", "message"), "HTTP error details")
            code = ErrorCode(error["code"])
            if not isinstance(error["message"], str):
                raise Invalid("HTTP error message must be a string")
        except (Invalid, ValueError):
            raise TransportFailure(f"server returned unexpected HTTP status {status}") from None
        raise SailError(code, error["message"])

    @staticmethod
    def _path(key: RequestKey) -> str:
        return f"/v1/requests/{key.tenant}/{key.request_id}"

    @staticmethod
    def _result(key: RequestKey, status: int, payload: Any) -> ResultEnvelope:
        try:
            result = ResultEnvelope.from_json(payload)
        except Invalid as exc:
            raise TransportFailure("server returned an invalid result envelope") from exc
        if result.key != key or (status == 200) != result.status.terminal:
            raise TransportFailure("server result identity or HTTP status is inconsistent")
        return result

    def submit(self, request: Request) -> Admission:
        status, payload = self._send(
            "POST", "/v1/requests", (200, 202), body=request.to_json(), replayable=True,
        )
        record = fields(payload, ("tenant", "request_id", "reused", "revision", "accepted_at_ms"), "admission")
        admitted = RequestKey(record["tenant"], record["request_id"])
        if admitted != request.key or type(record["reused"]) is not bool or record["reused"] != (status == 200):
            raise TransportFailure("server admission identity or reuse receipt is inconsistent")
        return Admission(
            admitted, record["reused"], identifier(record["revision"], "revision"),
            integer(record["accepted_at_ms"], "accepted_at_ms"),
        )

    def result(self, key: RequestKey) -> ResultEnvelope:
        status, payload = self._send("GET", self._path(key), (200, 202), replayable=True)
        return self._result(key, status, payload)

    def wait(self, key: RequestKey, *, timeout: float | None = None) -> ResultEnvelope:
        if timeout is not None and timeout <= 0:
            return self.result(key)
        stop = None if timeout is None else monotonic() + timeout
        while True:
            remaining = None if stop is None else stop - monotonic()
            if remaining is not None and remaining <= 0:
                return self.result(key)
            # Leave a margin for the response after the server's timed wait.
            window = max(0, self.timeout_seconds - 0.25)
            if remaining is not None:
                window = min(window, max(0, remaining - 0.1))
            wait_ms = min(MAX_LONG_POLL_MS, int(window * 1000))
            status, payload = self._send(
                "GET", self._path(key) + "?" + urlencode({"wait_ms": wait_ms}),
                (200, 202), replayable=True,
                limit=None if remaining is None else max(remaining, 0.05),
            )
            result = self._result(key, status, payload)
            if result.status.terminal or (stop is not None and monotonic() >= stop):
                return result
            if wait_ms == 0:
                delay = 0.02 if stop is None else min(0.02, max(0, stop - monotonic()))
                if delay == 0:
                    return result
                sleep(delay)

    def cancel(self, key: RequestKey) -> ResultEnvelope:
        status, payload = self._send("DELETE", self._path(key), (200,), replayable=True)
        return self._result(key, status, payload)

    def install(self, bundle: str) -> ModelRevision:
        _, payload = self._send(
            "POST", "/v1/catalog/install", (200,), body={"bundle": identifier(bundle, "bundle")},
        )
        return ModelRevision.from_v1(payload)

    def activate(self, key: ModelKey, *, expected_generation: int | None = None) -> int:
        body: dict[str, Any] = key.to_json()
        if expected_generation is not None:
            body["expected_generation"] = integer(expected_generation, "catalog generation")
        _, payload = self._send("POST", "/v1/catalog/activate", (200,), body=body)
        obj = fields(payload, ("model", "revision", "generation"), "activation")
        if ModelKey(obj["model"], obj["revision"]) != key:
            raise TransportFailure("server activation refers to a different revision")
        return integer(obj["generation"], "catalog generation")

    def history(
        self, *, after: int = 0, limit: int = 100, tenant: str | None = None, status: ResultStatus | None = None,
    ) -> tuple[tuple[ResultEnvelope, ...], int]:
        query: dict[str, str | int] = {
            "after": integer(after, "after cursor"),
            "limit": integer(limit, "history limit", minimum=1, maximum=1000),
        }
        if tenant is not None:
            query["tenant"] = identifier(tenant, "tenant")
        if status is not None:
            query["status"] = status.value
        _, payload = self._send("GET", "/v1/requests?" + urlencode(query), (200,), replayable=True)
        page = fields(payload, ("results", "next"), "history page")
        if not isinstance(page["results"], list) or len(page["results"]) > limit:
            raise TransportFailure("server returned an invalid history page")
        receipts = tuple(ResultEnvelope.from_json(item) for item in page["results"])
        marker = integer(page["next"], "history next cursor", minimum=after)
        if bool(receipts) != (marker > after):
            raise TransportFailure("server did not advance the history cursor consistently")
        return receipts, marker

    def terminal_page(
        self, *, after: int = 0, limit: int = 100, wait_ms: int = 0,
    ) -> tuple[tuple[tuple[int, ResultEnvelope], ...], int, int]:
        """Fetch one durable terminal-stream page after cursor ``after``.

        Returns ``(entries, next, latest)`` where each entry is
        ``(terminal_position, envelope)``. A positive ``wait_ms`` long-polls for
        new terminal results, bounded by the server's ten-second wait ceiling.
        """
        query = {
            "after": integer(after, "after cursor"),
            "limit": integer(limit, "terminal limit", minimum=1, maximum=1000),
            "wait_ms": integer(wait_ms, "wait_ms", maximum=MAX_LONG_POLL_MS),
        }
        _, payload = self._send(
            "GET", "/v1/terminal-results?" + urlencode(query), (200,), replayable=True,
        )
        page = fields(payload, ("results", "next", "latest"), "terminal stream page")
        if not isinstance(page["results"], list) or len(page["results"]) > limit:
            raise TransportFailure("server returned an invalid terminal stream page")
        entries: list[tuple[int, ResultEnvelope]] = []
        previous = after
        for item in page["results"]:
            position = integer(item["terminal_position"], "terminal_position", minimum=after + 1)
            if position <= previous:
                raise TransportFailure("terminal stream positions must be strictly increasing")
            envelope = ResultEnvelope.from_json(
                {key: value for key, value in item.items() if key != "terminal_position"},
            )
            if not envelope.status.terminal:
                raise TransportFailure("terminal stream entry is not terminal")
            entries.append((position, envelope))
            previous = position
        marker = integer(page["next"], "terminal next cursor", minimum=after)
        latest = integer(page["latest"], "terminal latest cursor")
        if latest < marker or bool(entries) != (marker > after):
            raise TransportFailure("server did not advance the terminal cursor consistently")
        return tuple(entries), marker, latest

    def terminal_follow(
        self, *, after: int = 0, limit: int = 100, wait_ms: int = MAX_LONG_POLL_MS,
    ) -> Iterator[tuple[int, ResultEnvelope]]:
        """Yield terminal results in durable order, resuming from ``after``.

        The generator keeps the server-side cursor itself, so a transport retry
        or a fresh ``terminal_follow`` with the last yielded position never
        duplicates or skips a terminal result.
        """
        cursor = after
        while True:
            entries, cursor, _latest = self.terminal_page(after=cursor, limit=limit, wait_ms=wait_ms)
            if not entries:
                return
            for entry in entries:
                yield entry
