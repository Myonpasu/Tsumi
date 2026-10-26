#!/usr/bin/env python3
"""How well do the search scores in a data set predict the game results?

    python3 train/calibrate.py data/gen3/data.bin [more files or globs] [--anchor-p P]

The trainer's target is  lambda * model(score) + (1 - lambda) * result  (train/train.py): the score model, fitted
to the data's own results (train/wdl.py), turns a score into win/draw/loss probabilities and an expected result.
This prints the actual win/draw/loss frequencies over the range of scores next to that model, how well it fits
(log-loss per position, against the best any function of the score can do and against knowing nothing), and
where the weight of the training loss goes.  Selfmate self-play is very drawish, so look here before lowering
lambda.

It also checks the output units: a network trained on such data learns, for each position, the output whose
expected result on the anchored curve (wdl.anchored_curve, --anchor-p as in train.py) matches the position's, and
the engine reports its output x 1000 in internal units.  So the positions that the search of such a network scored
near +-1000 (+-1.00 as reported, a 50% win and a 50% loss on the curve) should have the curve's expected results.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import RECORD_DTYPE, SCALE
from wdl import ANCHOR_P, SCORE_CLIP, anchor_probability, anchored_curve, fit_wdl


def kv(key, value):
    print(f"  {key:<16}{value}")


def binned_nll(s, res):
    """Log-loss of the empirical win/draw/loss frequencies in quantile bins of the score (up to 400, of at least 200
    positions): about the best any function of the score can do (a shade optimistic, as the frequencies are fitted
    to the same data)."""
    bins = min(400, max(20, len(s) // 200))
    q = np.unique(np.quantile(s, np.linspace(0.0, 1.0, bins + 1)))
    b = np.clip(np.searchsorted(q, s, side="right") - 1, 0, len(q) - 2)
    total = 0.0
    tot = np.bincount(b, minlength=len(q) - 1).astype(np.float64)
    for v in (1, 0, -1):
        n = np.bincount(b[res == v], minlength=len(q) - 1).astype(np.float64)
        m = n > 0
        total -= (n[m] * np.log(n[m] / tot[m])).sum()
    return total / len(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="data files or globs (tsumi datagen output)")
    ap.add_argument("--anchor-p", type=anchor_probability, default=ANCHOR_P,
                    help=f"the anchored curve's win probability at 0, as in train.py (default {ANCHOR_P:g})")
    args = ap.parse_args()
    files = [f for p in args.files for f in sorted(glob.glob(p))]
    if not files:
        sys.exit(f"no data files match {' '.join(args.files)}")
    recs = np.concatenate([np.fromfile(f, dtype=RECORD_DTYPE) for f in files])
    if len(recs) == 0:
        sys.exit("the data files are empty")
    print("calibration: do the search scores predict the game results?")
    kv("data", f"{len(recs):,} positions from {len(files)} file{'s' if len(files) != 1 else ''}: " + ", ".join(files))
    mate = np.abs(recs["score"].astype(np.int64)) >= 30000
    # How many distinct positions (placement and side to move) the data holds: self-play with too little variety
    # repeats games.
    fields = recs[["occ", "pcs", "stm"]]
    key = np.ascontiguousarray(fields).view(np.dtype((np.void, fields.dtype.itemsize)))
    unique = len(np.unique(key))
    a = np.abs(recs["score"][~mate].astype(np.float64))
    if len(a):
        kv("scores", f"|score| median {np.median(a):.0f}, 90th percentile {np.percentile(a, 90):.0f}, 99th {np.percentile(a, 99):.0f}; "
                     f"{mate.sum():,} mate scores ({mate.mean():.2%})")
    kv("distinct", f"{unique:,} of {len(recs):,} positions ({unique / len(recs):.1%})"
                   + ("" if unique / len(recs) >= 0.9 else "  <- repetitive: raise the random-move settings"))
    # Positions without a result label (flags bit 1) have only their score.  Everything below is about the others.
    labelled = (recs["flags"] & 2) == 0
    if not labelled.all():
        kv("result labels", f"{labelled.mean():.1%} of the positions have one; the others are left out below")
    recs = recs[labelled & ~mate]
    if len(recs) < 1000:
        sys.exit("too few positions with a result label to calibrate")
    s = recs["score"].astype(np.float64)
    res = np.where(recs["stm"] == 1, -recs["result"], recs["result"]).astype(np.int64)
    kv("results", f"for the side to move: win {np.mean(res == 1):.1%}  draw {np.mean(res == 0):.1%}  loss {np.mean(res == -1):.1%}"
                  f"   ({np.mean(res != 0):.1%} of the positions are from decisive games)")

    model, nll_model = fit_wdl(s, res)
    base = np.array([np.mean(res == v) for v in (1, 0, -1)])
    nll_none = max(0.0, float(-(base[base > 0] * np.log(base[base > 0])).sum()))
    nll_best = binned_nll(s, res)
    pw, pd, pl = model.wdl_np(s)

    # Per score range: what the games did, and what the fitted model says.
    edges = [-10000, -6000, -3000, -1500, -800, -400, -200, -100, -50, -20, 20, 50, 100, 200, 400, 800, 1500, 3000, 6000, 10000]
    idx = np.digitize(s, edges)
    print(f"\n  {'score range':>16}  {'positions':>10}  {'actual W/D/L %':>17}  {'result':>6}  |  {'model W/D/L %':>17}  {'result':>6}")
    for b in range(len(edges) + 1):
        sel = idx == b
        if sel.sum() < 100:
            continue
        label = f"< {edges[0]}" if b == 0 else f">= {edges[-1]}" if b == len(edges) else f"[{edges[b-1]}, {edges[b]})"
        aw, ad, al = (np.mean(res[sel] == v) for v in (1, 0, -1))
        mw, md, ml = (float(np.mean(p[sel])) for p in (pw, pd, pl))
        print(f"  {label:>16}  {sel.sum():10,}  {100*aw:5.1f}/{100*ad:5.1f}/{100*al:5.1f}  {aw + ad / 2:6.3f}  |  "
              f"{100*mw:5.1f}/{100*md:5.1f}/{100*ml:5.1f}  {mw + md / 2:6.3f}")
    print()
    win50, loss50 = model.even_scores()
    kv("model", model.describe())
    kv("log-loss", f"per position over win/draw/loss: the model {nll_model:.4f}, the best any function of the score does "
                   f"{nll_best:.4f}, no score at all {nll_none:.4f}")
    # Where the training loss's weight goes at lambda 1: a position counts in proportion to the square of the
    # model's slope at its score.
    ranges = ((0, 200), (200, 1000), (1000, 3000), (3000, np.inf))
    wt = model.slope_np(s) ** 2

    def span(lo, hi):
        return f"{lo}+" if hi == np.inf else f"{lo}-{hi}"

    shares = ", ".join(f"{span(lo, hi)} {wt[(np.abs(s) >= lo) & (np.abs(s) < hi)].sum() / wt.sum():.0%}" for lo, hi in ranges)
    counts = ", ".join(f"{span(lo, hi)} {np.mean((np.abs(s) >= lo) & (np.abs(s) < hi)):.0%}" for lo, hi in ranges)
    kv("loss weight", f"share of the training loss's weight by |score| at lambda 1: {shares}")
    kv("", f"(the positions themselves: {counts})")
    # The output units, model-free: the actual results of the positions within 25% of +-SCALE.
    A, curve = float(SCALE), anchored_curve(args.anchor_p)

    def near(lo, hi, expect):
        sel = (s >= lo) & (s < hi)
        if sel.sum() < 100:
            return f"{sel.sum():,} positions (too few)"
        win, draw, loss = (np.mean(res[sel] == v) for v in (1, 0, -1))
        return (f"{sel.sum():,} positions, W/D/L {100 * win:.1f}/{100 * draw:.1f}/{100 * loss:.1f}, expected result "
                f"{win + draw / 2:.3f} (curve {expect:.3f})")
    kv("anchor", f"anchored curve (p = {args.anchor_p:.4g}): +-{A:.0f} (+-1.00 as reported) is a 50% win and a 50% loss, "
                 f"expected results {curve.prob_np(1.0):.3f} and {curve.prob_np(-1.0):.3f}")
    kv("", f"scored within 25% of +{A:.0f}: {near(0.75 * A, 1.25 * A, float(curve.prob_np(1.0)))}")
    kv("", f"scored within 25% of -{A:.0f}: {near(-1.25 * A, -0.75 * A, float(curve.prob_np(-1.0)))}")
    explained = (nll_none - nll_best) / nll_none if nll_none > 0 else 0.0
    if not (res != 0).any():
        kv("=>", "every game was drawn: the results say nothing about the scores")
    elif explained < 0.1:
        kv("=>", f"the search scores barely predict the results ({explained:.0%} of the outcome's uncertainty at best): the "
                 "result part of the target is mostly noise")
    else:
        kv("=>", f"the scores account for {(nll_none - nll_model) / nll_none:.0%} of the outcome's uncertainty through the model "
                 f"({explained:.0%} at best)")
    if max(abs(win50), abs(loss50)) <= SCORE_CLIP and win50 > 1.3 * -loss50:
        kv("", f"the scores are optimistic for the side to move: it wins half its games at {win50:+.0f} but loses half already "
               f"at {loss50:+.0f}")


if __name__ == "__main__":
    main()
