#!/bin/sh
# Use an isolated Linux environment with the package already installed.
set -eu
python_command="${PYTHON:-python}"
requests="${BATCHSAIL_BENCH_REQUESTS:-360}"
seed="${BATCHSAIL_BENCH_SEED:-17}"
scratch=$(mktemp -d -t batchsail-verify.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
export PYTHONPYCACHEPREFIX="$scratch/cache"

run() {
    printf '\n%s\n' "$*"
    set +e
    "$@"
    status=$?
    set -e
    printf 'exit=%s\n' "$status"
    return "$status"
}

run "$python_command" -VV
run "$python_command" -m compileall -q src
run "$python_command" -m unittest discover -s tests -v
run "$python_command" examples/make_fixtures.py "$scratch/v1"
run diff -r fixtures/models "$scratch/v1/models"
run cmp fixtures/receipts.v1.jsonl "$scratch/v1/receipts.v1.jsonl"
run batchsail receipts --workspace "$scratch/restore" import fixtures/receipts.v1.jsonl
run batchsail receipts --workspace "$scratch/restore" query sample first
run batchsail receipts --workspace "$scratch/restore" export "$scratch/exported.v1.jsonl"
run cmp fixtures/receipts.v1.jsonl "$scratch/exported.v1.jsonl"
run batchsail inspect --workspace "$scratch/restore"
run "$python_command" examples/demo.py
run "$python_command" examples/http_replay.py
run "$python_command" benchmarks/hotspots.py --requests "$requests" --seed "$seed"
