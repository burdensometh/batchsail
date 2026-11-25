"""Start the packaged service, replay a v1 workload, and drain it via SIGTERM."""

import json
from pathlib import Path
from signal import SIGTERM
from subprocess import PIPE, Popen
import sys
from tempfile import TemporaryDirectory

from batchsail.config import RuntimeConfig
from batchsail.registry import ModelKey

ROOT = Path(__file__).resolve().parents[1]


def run(directory: Path) -> None:
    from batchsail.request.client import HttpClient
    from batchsail.request.replay import ReplayRunner
    from batchsail.request.workload import Workload

    config = ROOT / "configs/demo.toml"
    assert RuntimeConfig.load(config).backend == "process"
    process = Popen(
        (
            sys.executable, "-m", "batchsail", "serve",
            "--workspace", str(directory), "--port", "0",
            "--bundle-root", str(ROOT / "fixtures/models"), "--config", str(config),
        ),
        stdout=PIPE,
        text=True,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("could not open service stdout")
        listening = json.loads(process.stdout.readline())["listening"]
        client = HttpClient(f"http://{listening['host']}:{listening['port']}")
        for name in ("affine-r1", "affine-r2"):
            client.install(name)
        client.activate(ModelKey("affine", "r1"))
        report = ReplayRunner(client).run(Workload.read(ROOT / "fixtures/timeline.v1.json"))
        print(json.dumps(report.to_json(), allow_nan=False, sort_keys=True), flush=True)
        if not report.passed or report.statistics["output_fingerprints"] != 2:
            raise RuntimeError("HTTP replay failed")
        process.send_signal(SIGTERM)
        stdout, _ = process.communicate(timeout=10)
        shutdown = json.loads(stdout)
        if process.returncode or not shutdown["shutdown"]["completed"]:
            raise RuntimeError(f"service did not drain: rc={process.returncode}, result={shutdown}")
        print(json.dumps(shutdown, sort_keys=True), flush=True)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="batchsail-http-") as directory:
        run(Path(directory))
