#!/usr/bin/env python3
"""Check that the engine evaluates a network exactly as an independent numpy implementation does.

    python3 train/verify_net.py --net nets/gen1.nnue --data 'data/gen0/*.bin' [--engine ./tsumi] [--count 200]

Picks positions at random from the data, evaluates them with the network in numpy (model.evaluate_quantised,
the engine's integer arithmetic) and compares the results with the engine's `eval` command.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES_PER_BUCKET, RECORD_DTYPE, THREAT_FEATURES, batch_features, record_fen
from model import evaluate_quantised, read_net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="the network (.nnue)")
    ap.add_argument("--data", nargs="+", required=True, help="data files or globs to take positions from")
    ap.add_argument("--engine", default="./tsumi")
    ap.add_argument("--count", type=int, default=200, help="positions to compare (default 200)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    files = sorted({f for p in args.data for f in glob.glob(p)})
    if not files:
        sys.exit(f"no data files match {' '.join(args.data)}")
    data = np.concatenate([np.fromfile(f, dtype=RECORD_DTYPE) for f in files])
    if len(data) == 0:
        sys.exit("the data files are empty")
    recs = data[np.random.default_rng(args.seed).choice(len(data), size=min(args.count, len(data)), replace=False)]
    try:
        net = read_net(args.net)
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    us, them, tus, tthem, extras, bucket, truncated = batch_features(recs, net["king_buckets"], max_threats=256)
    if truncated:
        sys.exit(f"{truncated} positions have more than 256 threats")

    fens = [record_fen(r) for r in recs]
    cmds = ["uci", f"setoption name EvalFile value {args.net}"] + [c for fen in fens for c in (f"position fen {fen}", "eval")] + ["quit"]
    # check=False: a failing engine is reported below with the end of its output.
    out = subprocess.run([args.engine], input="\n".join(cmds) + "\n", capture_output=True, text=True, check=False).stdout
    engine_vals = [int(line.split(":")[1]) for line in out.splitlines() if line.startswith("NNUE (side to move):")]
    if len(engine_vals) != len(recs):
        sys.exit(f"the engine returned {len(engine_vals)} evaluations for {len(recs)} positions (can it load the net?)\n{out[-2000:]}")

    n_psq = int(net["NB"]) * FEATURES_PER_BUCKET
    bad = 0
    for i in range(len(recs)):
        ref = evaluate_quantised(net, us[i][us[i] < n_psq], them[i][them[i] < n_psq], tus[i][tus[i] < THREAT_FEATURES],
                                 tthem[i][tthem[i] < THREAT_FEATURES], extras[i], int(bucket[i]))
        if ref != engine_vals[i]:
            bad += 1
            if bad <= 10:
                print(f"MISMATCH {fens[i]}: numpy {ref}, engine {engine_vals[i]}")
    print(f"  {'verify':<16}{args.net}: {len(recs) - bad}/{len(recs)} positions evaluate identically in the engine and in numpy"
          + ("" if bad == 0 else "  <- MISMATCH"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
