#!/usr/bin/env bash
# The test battery: rules, search and network.  Run from the repository root after `make` and `make featlib`.
# The search tests need a network: NET=file, else nets/best.nnue, else the first network in nets/, else the one
# `make net` downloaded.
# The engine-against-numpy check of the network also needs data: DATA=file, else the first data file in data/.
set -euo pipefail
cd "$(dirname "$0")/.."
first() { find "$1" -maxdepth 2 -name "$2" 2>/dev/null | sort | head -n 1 || true; }
net=${NET:-}
[[ -n "$net" || ! -f nets/best.nnue ]] || net=nets/best.nnue
[[ -n "$net" ]] || net=$(first nets '*.nnue')
[[ -n "$net" ]] || net=$(find . -maxdepth 1 -name 'tsumi-*.nnue' | sort | head -n 1 || true)
[[ -n "$net" ]] || { echo "no network found (NET=file, a .nnue in nets/, or make net): the search tests need one" >&2; exit 1; }
data=${DATA:-$(first data '*.bin')}
fail() { echo "FAILED: $1" >&2; exit 1; }

echo "== network: $net"
echo "== perft suite"
out=$(./tsumi test | tail -n 1); echo "$out"
[[ "$out" == "all perft tests passed" ]] || fail "perft"
echo "== move generation against python-chess"
python3 tests/crosscheck_movegen.py 500 1
echo "== selfmate puzzles"
python3 tools/puzzles.py --net "$net" --epd tests/puzzles.epd --movetime 500
echo "== incremental accumulator updates"
out=$(./tsumi "setoption name EvalFile value $net" "nnuetest 5000" | tail -n 1); echo "$out"
[[ "$out" == *" 0 errors" ]] || fail "nnuetest"
if [[ -n "$data" ]]; then
    echo "== the engine's evaluations against numpy ($data)"
    python3 train/verify_net.py --net "$net" --data "$data" --count 100
else
    echo "== (no data: the engine-against-numpy check is skipped; DATA=file to run it)"
fi
echo "== bench"
./tsumi "setoption name EvalFile value $net" "bench 9" | tail -n 2
echo "all tests passed"
