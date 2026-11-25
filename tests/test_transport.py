from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from batchsail.codec import decode, encode
from batchsail.http import SailServer
from batchsail.request.client import HttpClient
from batchsail.request.replay import ReplayRunner
from batchsail.request.workload import Workload
from batchsail.results import ResultStatus
from batchsail.runtime import Runtime

from support import MODELS, ROOT, config, request


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runtime = Runtime(Path(self.directory.name), config(wait_ms=0))
        self.addCleanup(self.runtime.close)
        self.server = SailServer(("127.0.0.1", 0), self.runtime, bundle_root=MODELS)
        self.addCleanup(self.server.server_close)
        self.thread = Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)

    def send(self, verb, path, payload=None):
        connection = HTTPConnection(*self.server.server_address, timeout=3)
        try:
            headers = {} if payload is None else {"Content-Type": "application/json"}
            connection.request(verb, path, body=None if payload is None else encode(payload), headers=headers)
            response = connection.getresponse()
            return response.status, decode(response.read()), response.getheader("Location")
        finally:
            connection.close()

    def test_submit_wait_lookup_history_and_roll(self):
        self.assertEqual(self.send("POST", "/v1/catalog/install", {"bundle": "affine-r1"})[0], 200)
        self.assertEqual(self.send("POST", "/v1/catalog/activate", {"model": "affine", "revision": "r1"})[0], 200)
        item = request("net")
        status, admission, location = self.send("POST", "/v1/requests", item.to_json())
        self.assertEqual(status, 202)
        self.assertFalse(admission["reused"])
        self.assertEqual(location, "/v1/requests/alpha/net")
        status, result, _ = self.send("GET", location + "?wait_ms=1000")
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["output"]["shape"], [1, 2])
        self.assertEqual(self.send("DELETE", location)[1], result)
        self.assertEqual(self.send("POST", "/v1/requests", item.to_json())[:2], (200, {**admission, "reused": True}))
        changed = {**item.to_json(), "deadline_ms": 500}
        self.assertEqual(self.send("POST", "/v1/requests", changed)[0], 409)
        history = self.send("GET", "/v1/requests?tenant=alpha&limit=1&status=succeeded")[1]
        self.assertEqual(history["results"], [result])
        next_page = self.send("GET", f"/v1/requests?after={history['next']}")[1]
        self.assertEqual(next_page["results"], [])
        self.assertGreater(self.send("GET", "/v1/events?limit=8")[1]["latest"], 0)
        self.assertEqual(self.send("GET", "/v1/status")[1]["generation"], 1)

    def test_invalid_transport_inputs(self):
        invalid_paths = (
            "/v1/requests/alpha/missing?wait_ms=10001",
            "/v1/requests/alpha/missing?wait_ms=1&wait_ms=2",
            "/v1/requests/alpha/%2F",
            "/v1/events?limit=-1",
        )
        for path in invalid_paths:
            with self.subTest(path=path):
                self.assertEqual(self.send("GET", path)[0], 400)
        self.assertEqual(self.send("GET", "/v1/requests/alpha/unknown")[0], 404)
        self.assertEqual(self.send("POST", "/v1/catalog/install", {"bundle": "../bad"})[0], 400)
        self.assertEqual(self.send("POST", "/v1/requests", {"tenant": "invalid"})[0], 400)

    def test_typed_client_replays_real_v1_timeline(self):
        host, port = self.server.server_address
        client = HttpClient(f"http://{host}:{port}")
        for name in ("affine-r1", "affine-r2"):
            revision = client.install(name)
            if name == "affine-r1":
                client.activate(revision.key)
        plan = Workload.read(ROOT / "fixtures" / "timeline.v1.json")
        plan.write(Path(self.directory.name) / "replay.json")
        self.assertEqual(Workload.read(Path(self.directory.name) / "replay.json"), plan)
        report = ReplayRunner(client).run(plan, settle_seconds=3)
        self.assertTrue(report.passed, report.to_json())
        self.assertEqual(report.statistics["output_fingerprints"], 2)
        self.assertEqual(set(report.statistics["by_model"]), {"affine@r1", "affine@r2"})
        results, cursor = client.history(tenant="timeline", status=ResultStatus.SUCCEEDED)
        self.assertEqual(len(results), 2)
        self.assertEqual(client.history(after=cursor, tenant="timeline"), ((), cursor))
