"""Command-line entry points for runtime, model preparation, and v1 archives."""

import argparse
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable

from .codec import decode, encode, read_bytes
from .config import RuntimeConfig, WorkspaceGuard
from .errors import ErrorCode, Invalid, SailError
from .http import SailServer
from .inspection import inspect_workspace
from .registry.authoring import pack_definition
from .registry.model import ModelKey
from .registry.store import ModelStore
from .request.client import HttpClient
from .request.model import Request, RequestKey
from .request.replay import ReplayRunner
from .request.workload import Workload
from .results.archive import export_archive, import_archive
from .results.model import ResultStatus
from .results.store import ReceiptStore
from .runtime import Runtime
from .service import serve


def _display(value: Any) -> None:
    print(encode(value).decode("utf-8"), flush=True)


def _runtime(args: argparse.Namespace) -> Runtime:
    configuration = None if args.config is None else RuntimeConfig.load(args.config)
    return Runtime(args.workspace, configuration)


def _serve(args: argparse.Namespace) -> int:
    with _runtime(args) as runtime, SailServer(
        (args.bind, args.port), runtime, bundle_root=args.bundle_root,
    ) as server:
        serve(server, runtime, drain_seconds=args.shutdown_grace_ms / 1000, announce=_display)
    return 0


def _install(args: argparse.Namespace) -> int:
    with _runtime(args) as runtime:
        revision = runtime.install(args.bundle)
        _display(revision.to_v1())
    return 0


def _activate(args: argparse.Namespace) -> int:
    key = ModelKey(args.model, args.revision)
    with _runtime(args) as runtime:
        generation = runtime.activate(key, expected_generation=args.expected_generation)
        _display({**key.to_json(), "generation": generation})
    return 0


def _switch(args: argparse.Namespace) -> int:
    if len(args.target) % 2:
        raise Invalid("switch targets must be given as model/revision pairs")
    targets = tuple(
        ModelKey(args.target[index], args.target[index + 1])
        for index in range(0, len(args.target), 2)
    )
    with _runtime(args) as runtime:
        result = runtime.switch_many(targets, expected_generation=args.expected_generation)
        _display(result.to_json())
    return 0


def _pack(args: argparse.Namespace) -> int:
    _display(pack_definition(args.definition, args.bundle).to_v1())
    return 0


def _validate(args: argparse.Namespace) -> int:
    revision, _ = ModelStore.read_bundle(args.bundle)
    _display(revision.to_v1())
    return 0


def _infer(args: argparse.Namespace) -> int:
    request = Request.from_json(decode(read_bytes(args.input, maximum=32 * 1024 * 1024)))
    with _runtime(args) as runtime:
        runtime.submit(request)
        result = runtime.wait(request.key, timeout=args.wait_ms / 1000)
        if not result.status.terminal:
            result = runtime.cancel(request.key)
        _display(result.to_json())
    return 0 if result.status == ResultStatus.SUCCEEDED else 1


def _replay(args: argparse.Namespace) -> int:
    workload = Workload.read(args.workload)
    report = ReplayRunner(HttpClient(args.url)).run(workload, settle_seconds=args.settle_seconds)
    _display(report.to_json())
    return 0 if report.passed else 1


def _inspect(args: argparse.Namespace) -> int:
    config = RuntimeConfig() if args.config is None else RuntimeConfig.load(args.config)
    report = inspect_workspace(args.workspace, config)
    _display(report.to_json())
    return 0 if report.clean else 1


def _receipt_command(args: argparse.Namespace) -> int:
    with WorkspaceGuard(args.workspace) as workspace:
        store = ReceiptStore(workspace.root / "receipts.sqlite")
        try:
            if args.action == "query":
                _display(store.result(RequestKey(args.tenant, args.request_id)).to_json())
            elif args.action == "import":
                _display({"imported": import_archive(store, args.archive)})
            else:
                if args.archive.resolve().is_relative_to(workspace.root):
                    raise Invalid("receipt exports require a path outside the writable workspace")
                _display({"exported": export_archive(store, args.archive, tenant=args.tenant)})
        finally:
            store.close()
    return 0


def _set_action(
    parser: argparse.ArgumentParser,
    function: Callable[[argparse.Namespace], int],
) -> argparse.ArgumentParser:
    parser.set_defaults(function=function)
    return parser


def _runtime_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, metavar="TOML")
    parser.add_argument("--workspace", type=Path, required=True)
    return parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batchsail")
    parser.add_argument("--verbose", action="store_true", help="enable runtime debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = _runtime_flags(_set_action(sub.add_parser("serve", help="run the HTTP service"), _serve))
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--bundle-root", type=Path, help="directory of named bundles for online installation")
    serve.add_argument("--shutdown-grace-ms", type=int, default=5000, help="time to drain on SIGINT/SIGTERM")

    install = _runtime_flags(_set_action(sub.add_parser("install", help="install a v1 bundle"), _install))
    install.add_argument("bundle", type=Path)

    activate = _runtime_flags(_set_action(sub.add_parser("activate", help="activate a model revision"), _activate))
    activate.add_argument("model")
    activate.add_argument("revision")
    activate.add_argument("--expected-generation", type=int)

    switch = _runtime_flags(_set_action(
        sub.add_parser("switch", help="atomically switch one or more model revisions"), _switch,
    ))
    switch.add_argument(
        "target", nargs="+", metavar="model revision",
        help="alternating model and revision identifiers, e.g. affine r2 filter r1",
    )
    switch.add_argument("--expected-generation", type=int)

    infer = _runtime_flags(_set_action(sub.add_parser("infer", help="submit one request from JSON"), _infer))
    infer.add_argument("input", type=Path)
    infer.add_argument("--wait-ms", type=int, default=30_000)

    pack = _set_action(sub.add_parser("pack", help="write a v1 bundle from a numeric definition"), _pack)
    pack.add_argument("definition", type=Path)
    pack.add_argument("bundle", type=Path)

    validate = _set_action(sub.add_parser("validate", help="verify a v1 bundle and its weights"), _validate)
    validate.add_argument("bundle", type=Path)

    replay = _set_action(sub.add_parser("replay", help="replay a v1 timeline against HTTP"), _replay)
    replay.add_argument("workload", type=Path)
    replay.add_argument("--url", default="http://127.0.0.1:8080")
    replay.add_argument("--settle-seconds", type=float, default=30)

    _runtime_flags(_set_action(
        sub.add_parser("inspect", help="inspect the model catalog and v1 receipts while offline"),
        _inspect,
    ))

    receipts = _set_action(sub.add_parser("receipts", help="operate on offline v1 receipts"), _receipt_command)
    receipts.add_argument("--workspace", type=Path, required=True)
    actions = receipts.add_subparsers(dest="action", required=True)
    query = actions.add_parser("query")
    query.add_argument("tenant")
    query.add_argument("request_id")
    importer = actions.add_parser("import")
    importer.add_argument("archive", type=Path)
    exporter = actions.add_parser("export")
    exporter.add_argument("archive", type=Path)
    exporter.add_argument("--tenant")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    try:
        if args.command == "infer" and not 1 <= args.wait_ms <= 600_000:
            raise Invalid("wait_ms must be from 1 to 600000")
        if args.command == "serve" and not 0 <= args.port <= 65535:
            raise Invalid("port must be from 0 to 65535")
        if args.command == "serve" and not 0 <= args.shutdown_grace_ms <= 600_000:
            raise Invalid("shutdown grace must be from 0 to 600000 ms")
        return args.function(args)
    except SailError as exc:
        print(encode({"error": exc.to_json()}).decode("utf-8"), file=sys.stderr)
    except (OSError, sqlite3.Error) as exc:
        print(f"{ErrorCode.UNAVAILABLE.value}: {exc}", file=sys.stderr)
    return 2
