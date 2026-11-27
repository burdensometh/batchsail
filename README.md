# BatchSail

BatchSail is a multi-model inference runtime for Python 3.12 on Linux. It admits requests against an immutable model revision, groups compatible shapes, and executes packed float32 batches in CPU worker processes. The two sample models are an affine projection and a causal filter with independent history for each slot.

## Run it

```sh
python -m pip install .
batchsail install --workspace var/service fixtures/models/affine-r1
batchsail activate --workspace var/service affine r1
batchsail infer --workspace var/service fixtures/request.json
batchsail receipts --workspace var/service query sample first
```

The example returns a `[2, 2]` tensor with row-major values `[2.0, 7.0, -0.5, 0.0]`. Use a new request ID for different input. Replaying the same tenant, request ID and content returns the existing admission and result.

```sh
batchsail serve --workspace var/service --config configs/demo.toml --bundle-root fixtures/models
curl -fsS http://127.0.0.1:8080/v1/status
curl -fsS -H 'Content-Type: application/json' --data @fixtures/request.json http://127.0.0.1:8080/v1/requests
curl -fsS 'http://127.0.0.1:8080/v1/requests/sample/first?wait_ms=1000'
```

The HTTP listener defaults to loopback. It is intended for trusted clients and operators in a controlled environment. There is no authentication or TLS. Run one process per writable workspace. Different workspaces can run independently.

`docker build -t batchsail:local .` builds the packaged service on Python 3.12. Set `BASE_IMAGE` to use another Python 3.12 image. For an offline image build, preload the base with setuptools and wheel, then use `--build-arg OFFLINE_BUILD=1 --network none`.

## Runtime contract

- Submissions identify a tenant and request ID. The fingerprint includes model, tensor, deadline and optional retry source. Identical submissions reuse the original admission even across restart or activation.
- A deadline is a duration from admission measured against a monotonic clock. Omission means unlimited. At the bound the request expires. Queue wait has its own batch timeout and dispatch guard. Batch queues rotate tenants within each compatible shape.
- A catalog activation affects subsequent admissions. The coordinator pins accepted work until its queue or worker obligation ends, including executions whose client result has already become terminal.
- The per-device ledger reserves weight pages and aligned transient workspace for the complete padded batch. Each device has at most one running batch. Idle cached kernels are evicted by pressure. Pending drain pins protect old residency. Allocation counts refer to logical float32 buffers and reservations. Python object memory and IPC copies have separate overhead.
- The supervisor verifies batch, lease, revision, attempt, slot and output shape before settlement. It can repeat a lost worker attempt within the original request deadline and configured attempt count. A cancellation or timeout becomes immutable immediately. Durable results survive restart. Interrupted durable pending receipts are recovered as failed because request tensors are held in memory.
- A caller can explicitly retry a retryable receipt with a new request ID and the `retry_of` field. This is a fresh admission to the current revision of the same model.
- CPU execution supports an inline mode for controlled interleavings and spawned processes for isolated execution. A device backend implements `executor.Backend`; its buffers must obey the lease and slot contracts. A GPU backend would implement weight residency, transfer, execution, synchronization and device reset behind that interface.

| Method | Path | Behavior |
| --- | --- | --- |
| POST | `/v1/requests` | Admit a JSON request and return its location. 202 for new work, 200 for repeat. |
| GET | `/v1/requests/{tenant}/{id}?wait_ms=1000` | Read the result, optionally long poll up to 10 seconds. 202 pending or 200 terminal. |
| DELETE | `/v1/requests/{tenant}/{id}` | Cancel pending work or return the existing terminal result. |
| GET | `/v1/requests?tenant=sample&after=0&limit=10` | Cursor-paginated receipt history. Optional status filter. |
| GET | `/v1/status` | Catalog generation, revision pins, allocator state, scheduling blocks, and latency histograms. |
| GET | `/v1/events?after=0&limit=100` | Bounded event stream with cursor and dropped range. |
| POST | `/v1/catalog/install` | Install the named immediate child of `--bundle-root`. Body contains `bundle`. |
| POST | `/v1/catalog/activate` | Activate a `model` and `revision`. Optional `expected_generation` provides compare-and-swap. |
| POST | `/v1/devices/estimate` | Preview capacity for `model` and `rows`. Optional `slots` and `revision`. |
| POST | `/v1/devices/trim` | Release idle retired weights. Accepts a `device` and a `revision` model key as optional filters. |

For example, send `{"bundle":"affine-r2"}` to install the second revision, then `{"model":"affine","revision":"r2","expected_generation":1}` to activate it.

`batchsail replay fixtures/timeline.v1.json --url http://127.0.0.1:8080` plays the submitted operations against a running service and compares terminal results to the timeline. Start with both affine bundles installed, r1 active, and fresh request IDs. `python examples/http_replay.py` provisions these conditions in a temporary workspace. On SIGTERM and SIGINT, the service stops taking new connections and waits up to `--shutdown-grace-ms` for admitted work to complete.

## Version-one formats

- A bundle contains `manifest.v1.json` and `weights.v1.json`. The manifest names a float32 input/output width, maximum rows, an operator, tap count, resident pages, and the SHA-256 of the exact weight file. Coefficients are output-channel-major for affine and tap-then-output-channel-major for the causal filter. Within a channel they follow input-column order.
- `batchsail pack fixtures/definitions/affine-r1.json /tmp/affine-r1` writes a bundle. `batchsail validate /tmp/affine-r1` verifies it. Installation copies checked weights and materializes a canonical manifest. `catalog.v1.json` persists active revisions and catalog generation.
- `receipts.sqlite` uses SQLite user version 2. Each row still holds one unique result per request, and a version-two row also records the attempt evidence that produced it: every dispatch's attempt number, batch, slot, device and queue time, plus the exit reason and finish time once known. A pending row from an earlier process recovers as failed while keeping already known failures intact. A version-one database migrates in place; its rows keep no evidence and report their attempt history as unknown. `GET /v1/requests` and `batchsail receipts --workspace var/archive query` state `attempts_known` alongside the receipt.
- For transfer, use `batchsail receipts --workspace var/archive import fixtures/receipts.v1.jsonl` and `batchsail receipts --workspace var/archive export /tmp/receipts.v1.jsonl`. Archives may hold version-one receipts (round-tripped byte-for-byte) or version-two receipts carrying attempt evidence. Import is atomic and idempotent; duplicate terminal receipts merge only the attempts both copies agree on, so the request keeps one final result.
- `configs/demo.toml` shows both logical devices, page units, workspace alignment, batch limits, and worker settings. A model identity is immutable inside one workspace. `workspace.v1.json` records its page size. `batchsail inspect --workspace var/service --config configs/demo.toml` checks the catalog, bundle contents, capacity fit and receipt consistency while the service is offline.

## Verification

```sh
python -m unittest discover -s tests -v
python examples/demo.py
python benchmarks/hotspots.py --requests 360 --seed 17
sh scripts/verify.sh
```

The demo exercises interactive batching, revision roll, memory pressure, cancel and recovery. The benchmark drives heterogeneous, urgent and long-queue workloads and verifies every successful result against independent tensor arithmetic. It reports admission and latency distributions, batch occupancy, expiration counts, and measured worker round-trip, load, packing and kernel phases.

`ManualClock`, `Runtime.pulse()`, `Boundary`, and a replaceable backend enable fault tests. The dequeue hook runs under the coordinator and must return immediately. The acquisition and return hooks run on executor threads and can wait on an explicit barrier. `ProcessBackend.interrupt(device)` supports tests that terminate a real worker.
