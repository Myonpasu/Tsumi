#!/usr/bin/env python3
"""Train a Tsumi network on self-play data from `tsumi datagen`.

    python3 train/train.py --data 'data/gen0/*.bin' --out nets/gen1.nnue [--resume nets/best.nnue.pt]

Run `make featlib` first: the trainer computes the network's inputs with the engine's own code.

Target.  Each position's target blends its search score and its game result, both as expected results for the
side to move (1 a win, 0.5 a draw, 0 a loss):
    target = lambda * model(score) + (1 - lambda) * result
where model is a win/draw/loss model of the search scores fitted to this run's own game results (train/wdl.py),
so that the two parts agree in expectation.  lambda moves linearly from --lambda at the first epoch to
--end-lambda at the last, towards the results by default: the scores are smooth labels that guide the early
epochs, and the results, which hold what the shallow searches behind the scores could not see, weigh more at the
end.  A position without a result label (flags bit 1 in src/datagen.h) uses its score alone.

Result filter.  A position is kept with the probability the score model gives the result its game had (Stockfish's
wld filter), so a result the score makes unlikely, a won position lost to a later mistake say, mostly drops out
rather than teach the network the mistake.  --no-result-filter keeps every position.

Loss.  The network's output y goes through a fixed curve: the side to move wins with probability
win(y) = 1 / (1 + (1/p - 1)^(1 - y)) and loses with win(-y), p being --anchor-p, and the loss is the squared
difference between the expected result on that curve and the target, weighted 1 + --loss-asymmetry where the
network is more optimistic than the target (an evaluation too high misleads a search more than one too low).  An
output of +1 is a 50% chance of winning, whatever the scale of the search scores the network learnt from; the
engine multiplies y by SCALE (1000) into its internal units and shows it as +1.00.

Validation.  Whole games holding --val-fraction of the positions are held out (the positions of a game are alike
and share its result, so a split by position would reward memorising games), and filtered like the training
positions.  The table's train and val losses are both measured against the target at the final lambda, which
keeps them comparable from epoch to epoch; "vs score" and "vs result" are the held-out positions against the
search score and the game result alone.  --keep-best writes the epoch with the lowest val loss rather than the
last.

--resume continues from a .pt checkpoint (as the pipeline does whenever nets/best.nnue.pt exists).  The learning
rate is multiplied by --lr-decay every --lr-step epochs.  The .nnue is written with its units reordered for the
engine's sparse first layer (train/ftperm.py: the same evaluations, faster); the .pt keeps the trained order.
"""
import argparse
import glob
import os
import queue
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (
    HEAD_EXTRAS,
    MAX_THREATS,
    NUM_KING_BUCKETS,
    OUTPUT_BUCKETS,
    RECORD_DTYPE,
    batch_features,
    load_featlib,
)
from ftperm import DEFAULT_POSITIONS as FTPERM_POSITIONS
from ftperm import permute_file
from model import L2, L3, NNUE, check_checkpoint
from wdl import ANCHOR_P, anchor_probability, anchored_curve, fit_wdl

LR_RESUME = 1e-5    # default --lr when resuming a checkpoint
LR_SCRATCH = 1e-3   # ... and when training from scratch


def kv(key, value):
    """One line of the settings block: a left-aligned key and its value."""
    print(f"  {key:<16}{value}", flush=True)


def fmt_duration(seconds):
    """45s, 1m05s, 1h05m, as the pipeline prints them."""
    s = round(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def load_data(patterns, max_positions=0):
    """All the records of the files in one array, sized from the files first so that the data is held only once."""
    files = []
    for p in patterns:
        files += sorted(glob.glob(p))
    if not files:
        sys.exit(f"no data files match {' '.join(patterns)}")
    sizes = [os.path.getsize(f) // RECORD_DTYPE.itemsize for f in files]
    if len(files) == 1 and not max_positions:
        # One whole file: memory-mapped, so its pages are file-backed and the system can drop and reread them under
        # memory pressure rather than hold an allocation of its size.
        data = np.memmap(files[0], dtype=RECORD_DTYPE, mode="r", shape=(sizes[0],))
        kv("data", f"{files[0]}: {len(data):,} positions")
        return data
    capacity = sum(sizes) if not max_positions else min(sum(sizes), max_positions)
    data = np.empty(capacity, dtype=RECORD_DTYPE)
    total = 0
    for f, n in zip(files, sizes):
        a = np.fromfile(f, dtype=RECORD_DTYPE, count=min(n, capacity - total))
        data[total:total + len(a)] = a
        total += len(a)
        kv("data" if f == files[0] else "", f"{f}: {len(a):,} positions")
        if total >= capacity:
            break
    if len(files) > 1:
        kv("", f"{total:,} positions in all")
    if max_positions and total == max_positions:
        kv("", f"capped at {max_positions:,} positions (--max-positions)")
    return data[:total]


def game_ids(recs):
    """The game each position belongs to (0, 1, ...).  Datagen writes the positions of a game together in increasing
    ply, so a new game starts wherever the ply goes down, or stays the same below 255 (where the record's ply
    saturates), or the result, one per game, changes."""
    if len(recs) == 0:
        return np.zeros(0, dtype=np.int32)
    ply, res = recs["ply"], recs["result"]
    start = np.ones(len(recs), dtype=bool)
    start[1:] = (ply[1:] < ply[:-1]) | ((ply[1:] == ply[:-1]) & (ply[1:] < 255)) | (res[1:] != res[:-1])
    return np.cumsum(start, dtype=np.int32) - 1


def split_by_game(recs, fraction, rng):
    """(validation indices, training indices, validation games, games): whole games, drawn at random until they hold
    `fraction` of the positions, are held out.  Indices (int32) rather than copies: the records are held once."""
    gid = game_ids(recs)
    n_games = int(gid[-1]) + 1 if len(gid) else 0
    want = int(len(recs) * fraction)
    order = rng.permutation(n_games)
    filled = np.cumsum(np.bincount(gid, minlength=n_games)[order])
    k = min(n_games, int(np.searchsorted(filled, want)) + 1) if want > 0 else 0
    held = np.zeros(n_games, dtype=bool)
    held[order[:k]] = True
    is_val = held[gid]
    all_idx = np.arange(len(recs), dtype=np.int32)
    return all_idx[is_val], all_idx[~is_val], k, n_games


def has_result(recs):
    """Positions whose game result is a label for them (flags bit 1 clear, see src/datagen.h)."""
    return (recs["flags"] & 2) == 0


def results_stm(recs):
    """Game results (+1, 0, -1) from the side to move's point of view."""
    return np.where(recs["stm"] == 1, -recs["result"], recs["result"])


def make_targets(recs, lam, model):
    """lambda * model(score) + (1 - lambda) * result, as expected results; lambda 1 without a result label."""
    p_score = model.prob_np(recs["score"].astype(np.float32))
    result = (results_stm(recs).astype(np.float32) + 1.0) / 2.0
    lam_eff = np.where(has_result(recs), lam, 1.0).astype(np.float32)
    return (lam_eff * p_score + (1.0 - lam_eff) * result).astype(np.float32)


def result_filter(data, idx, model, rng, chunk=1 << 20):
    """The indices in idx (sorted) of the positions kept by the result filter: each with the probability the score model
    gives the result its game had, and all of those without a result label."""
    keep = np.ones(len(idx), dtype=bool)
    for a in range(0, len(idx), chunk):
        recs = data[idx[a:a + chunk]]
        win, draw, loss = model.wdl_np(recs["score"].astype(np.float64))
        res = results_stm(recs)
        p = np.where(res > 0, win, np.where(res < 0, loss, draw))
        keep[a:a + chunk] = ~has_result(recs) | (rng.random(len(recs)) < p)
    return idx[keep]


def extract(recs, stats):
    """The model's inputs for a batch of records (numpy arrays, in the order of NNUE.forward)."""
    psq_us, psq_them, thr_us, thr_them, extras, bucket, truncated = batch_features(recs)
    stats["truncated"] += truncated
    return psq_us, psq_them, thr_us, thr_them, extras.astype(np.float32) / 127.0, bucket


def batch_iter(data, train_idx, batch_size, lam, shown_lam, model, stats, rng, prefetch=4):
    """(inputs, target, shown target) for each batch of a shuffled pass over the training positions, prepared in a
    background thread.  The shown target is the one at shown_lam, which the loss shown is measured against, or None
    when that is lam itself."""
    perm = np.arange(len(train_idx), dtype=np.int32)
    rng.shuffle(perm)
    n_batches = len(train_idx) // batch_size
    q = queue.Queue(maxsize=prefetch)

    def producer():
        for b in range(n_batches):
            recs = data[np.sort(train_idx[perm[b * batch_size:(b + 1) * batch_size]])]
            shown = None if shown_lam == lam else make_targets(recs, shown_lam, model)
            q.put((extract(recs, stats), make_targets(recs, lam, model), shown))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="data files or globs written by tsumi datagen")
    ap.add_argument("--out", required=True, help="the network to write (.nnue; the checkpoint goes to OUT.pt)")
    ap.add_argument("--resume", help="continue training from this .pt checkpoint")
    ap.add_argument("--hidden", type=int, default=1024, help="units per perspective, a multiple of 64 (default 1024)")
    ap.add_argument("--epochs", type=int, default=6, help="epochs to train (default 6)")
    ap.add_argument("--batch", type=int, default=16384, help="positions per batch (default 16384)")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"learning rate (default {LR_RESUME:g} with --resume, {LR_SCRATCH:g} from scratch)")
    ap.add_argument("--lr-decay", type=float, default=0.5, help="multiply the learning rate by this every --lr-step epochs (default 0.5)")
    ap.add_argument("--lr-step", type=int, default=3, help="epochs between those decays (default 3)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.8,
                    help="weight of the search score against the game result in the target at the first epoch (default 0.8; 1: the score alone)")
    ap.add_argument("--end-lambda", dest="end_lam", type=float, default=0.5, help="... and at the last epoch (default 0.5)")
    ap.add_argument("--result-filter", action=argparse.BooleanOptionalAction, default=True,
                    help="keep a position with the probability the score model gives its game's result (default on)")
    ap.add_argument("--loss-asymmetry", type=float, default=0.2,
                    help="weight the loss 1 + this where the network is more optimistic than the target (default 0.2; 0: none)")
    ap.add_argument("--anchor-p", type=anchor_probability, default=ANCHOR_P,
                    help=f"the win probability at an output of 0 on the curve the output goes through, a decimal or a fraction "
                         f"(default {ANCHOR_P:g}); smaller is steeper and gives balanced positions less weight")
    ap.add_argument("--keep-best", action="store_true", help="write the epoch with the lowest val loss rather than the last")
    ap.add_argument("--val-fraction", type=float, default=0.02, help="share of the positions held out, as whole games (default 0.02)")
    ap.add_argument("--max-positions", type=int, default=0,
                    help="train on at most this many positions, taken from the --data files in the order given (0 = all)")
    ap.add_argument("--progress-every", type=int, default=2_000_000,
                    help="a progress row every this many positions within an epoch (default 2,000,000; 0 = only at its end)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch's default)")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = LR_RESUME if args.resume else LR_SCRATCH
    if args.progress_every < 0 or args.epochs < 0 or args.batch <= 0 or args.lr_step <= 0:
        sys.exit("--progress-every and --epochs must not be negative, --batch and --lr-step must be positive")
    if not 0.0 <= args.val_fraction < 1.0:
        sys.exit("--val-fraction must be at least 0 and below 1")
    if not 0.0 <= args.lam <= 1.0 or not 0.0 <= args.end_lam <= 1.0 or args.loss_asymmetry < 0.0:
        sys.exit("--lambda and --end-lambda must be between 0 and 1, --loss-asymmetry at least 0")
    if args.hidden % 64 or not 0 < args.hidden <= 4096:
        sys.exit("--hidden must be a multiple of 64, at most 4096")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    load_featlib()   # exits with a message if `make featlib` has not been run

    data = load_data(args.data, args.max_positions)
    val_idx, train_idx, val_games, n_games = split_by_game(data, args.val_fraction, rng)
    # The score model is fitted, and the data described, on a sample of the training positions when there are more
    # than FIT_SAMPLE: the fit needs no more, and a copy of all of them would cost as much memory as the data.
    FIT_SAMPLE = 5_000_000
    fit_idx = train_idx if len(train_idx) <= FIT_SAMPLE else np.sort(rng.choice(train_idx, FIT_SAMPLE, replace=False))
    sample = data[fit_idx]
    sampled = "" if len(fit_idx) == len(train_idx) else f" (a sample of {len(fit_idx):,})"
    del fit_idx
    labelled = has_result(sample)
    res = sample["result"][labelled]
    kv("split", f"{len(train_idx):,} positions to train on, {len(val_idx):,} to validate on ({val_games:,} of the "
                f"{n_games:,} games, held out whole)")
    kv("results", f"white {np.mean(res == 1):.1%}, black {np.mean(res == -1):.1%}, draw {np.mean(res == 0):.1%} of the "
                  f"training positions{sampled}")
    if not labelled.all():
        kv("", f"{labelled.mean():.1%} of the positions have a result label; the rest are trained on their search score alone")

    model = NNUE(args.hidden).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        why = check_checkpoint(state, args.hidden)
        if why:
            sys.exit(f"cannot resume {args.resume}: {why}")
        model.load_state_dict(state)
        kv("start", f"resumed from {args.resume}")
    else:
        kv("start", "from scratch (random initialisation)")
    kv("network", f"{args.hidden} units per perspective -> {args.hidden + HEAD_EXTRAS} (pairwise products + legal-move inputs) "
                  f"-> {L2} -> {L3} -> 1; {NUM_KING_BUCKETS} king buckets, {OUTPUT_BUCKETS} output buckets")
    if args.lam == args.end_lam == 1.0:
        kv("target", "the search score as an expected result (lambda 1: no game results)")
    else:
        kv("target", "lambda x search score + (1 - lambda) x game result, as expected results; lambda "
                     + (f"{args.lam:.3f} -> {args.end_lam:.3f} over the epochs" if args.lam != args.end_lam else f"{args.lam:.3f}"))
    # The score model, fitted to the positions with a result label (mate scores left out).
    fit_sel = labelled & (np.abs(sample["score"]) < 30000)
    if fit_sel.sum() < 1000:
        sys.exit("fewer than 1000 positions with a result label: too few to fit the score model to")
    score_model, _ = fit_wdl(sample["score"][fit_sel].astype(np.float64), results_stm(sample[fit_sel]))
    kv("score model", f"win/draw/loss, fitted to {fit_sel.sum():,} positions with a result label: {score_model.describe()}")
    del sample
    if args.result_filter:
        before = len(train_idx), len(val_idx)
        train_idx = result_filter(data, train_idx, score_model, rng)
        val_idx = result_filter(data, val_idx, score_model, rng)
        kv("result filter", f"{len(train_idx):,} of the {before[0]:,} training positions kept ({len(train_idx) / max(before[0], 1):.1%}) "
                            f"and {len(val_idx):,} of the {before[1]:,} held out: each with the probability the score model gives "
                            "the result its game had")
    val_recs = data[val_idx]
    del val_idx
    if args.loss_asymmetry:
        kv("loss", f"squared error of the expected result, weighted {1.0 + args.loss_asymmetry:g} where the network is more "
                   "optimistic than the target")
    else:
        kv("loss", "squared error of the expected result")
    # The curve the output goes through.
    curve = anchored_curve(args.anchor_p)
    w0, d0, _ = (float(v) for v in curve.wdl_np(0.0))
    kv("output units", f"anchored (p = {args.anchor_p:.4g}): an output of +1.00 is a 50% win, 0 a {w0:.0%} win and {d0:.0%} draw")
    schedule = f"x{args.lr_decay:g} every {args.lr_step} epochs" if args.lr_decay != 1.0 else "constant"
    kv("learning rate", f"{args.lr:.1e}, {schedule}")
    n_batches = len(train_idx) // args.batch
    per_epoch = n_batches * args.batch   # positions trained on per epoch: the last, partial batch is left out
    if args.epochs > 0 and n_batches == 0:
        sys.exit(f"fewer training positions ({len(train_idx):,}) than one batch ({args.batch:,}): lower --batch")
    kv("run", f"{args.epochs} epoch{'s' if args.epochs != 1 else ''}, batch {args.batch:,}, {device}"
       + (f", {args.threads} thread{'s' if args.threads != 1 else ''}" if args.threads else "")
       + (f"; a row every {args.progress_every:,} positions" if args.progress_every and args.epochs else ""))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stats = {"truncated": 0}

    def to_device(arrays, non_blocking=False):
        return [torch.from_numpy(a).to(device, non_blocking=non_blocking) for a in arrays]

    final_lam = args.end_lam

    def loss_of(p, target):
        """The loss of each position: squared error, weighted where the network is more optimistic than the target."""
        err = (p - target) ** 2
        if args.loss_asymmetry:
            err = err * (1.0 + args.loss_asymmetry * (p > target).to(err.dtype))
        return err

    def validate(recs):
        """The held-out positions' losses against the target at the final lambda, against the search score alone and
        against the game result alone (the last on the positions with a result label)."""
        model.eval()
        lambdas = (final_lam, 1.0, 0.0)
        sums, counts = np.zeros(len(lambdas)), np.zeros(len(lambdas))
        with torch.no_grad():
            for b in range(0, len(recs), args.batch):
                chunk = recs[b:b + args.batch]
                p = curve.prob_torch(model(*to_device(extract(chunk, stats))))
                labelled = torch.from_numpy(has_result(chunk)).to(device)
                for k, lam in enumerate(lambdas):
                    err = loss_of(p, torch.from_numpy(make_targets(chunk, lam, score_model)).to(device))
                    if k == 2:
                        err = err[labelled]
                    sums[k] += err.sum().item()
                    counts[k] += len(err)
        model.train()
        return sums / np.maximum(counts, 1)

    # The units are reordered for the engine's sparse first layer on a fixed sample of the training positions.
    perm_recs = data[np.sort(np.random.default_rng(args.seed + 1).choice(train_idx, min(len(train_idx), FTPERM_POSITIONS), replace=False))]
    perm_rates = None

    def write_net():
        """The checkpoint and the .nnue, each written to a temporary file and moved into place, so that a run stopped
        at any moment leaves the last complete pair."""
        nonlocal perm_rates
        torch.save(model.state_dict(), args.out + ".pt.tmp")
        os.replace(args.out + ".pt.tmp", args.out + ".pt")
        model.export(args.out + ".tmp")
        os.replace(args.out + ".tmp", args.out)
        perm_rates = permute_file(args.out, perm_recs, device=device)   # writes a .tmp and replaces too

    def report_written(note=""):
        kv("wrote", f"{args.out} and {args.out}.pt{note}")
        if perm_rates is not None:
            kv("unit order", f"reordered for the engine's sparse first layer (same evaluations): {perm_rates[0]:.1%} -> "
                             f"{perm_rates[1]:.1%} of the input chunks all zero")

    if args.epochs == 0:   # no training: write the network of the checkpoint (or of the random initialisation)
        write_net()
        report_written(" (--epochs 0: no training)")
        return

    # The table: a row every --progress-every positions while an epoch runs, and one at its end with the validation
    # losses.  Columns that would only repeat another are left out: vs score at a final lambda of 1 (it is the val
    # loss), vs result at 0, and lambda when it does not change.
    validating = len(val_recs) >= 64
    columns = [("epoch", len(f"{args.epochs}/{args.epochs}")), ("done", 4), ("train loss", 10)]
    if validating:
        columns.append(("val loss", 8))
        if final_lam != 1.0:
            columns.append(("vs score", 8))
        if final_lam != 0.0:
            columns.append(("vs result", 9))
    if args.lam != args.end_lam:
        columns.append(("lambda", 6))
    columns += [("lr", 7), ("pos/s", 5), ("elapsed", 7), ("ETA", 6)]
    columns = [(name, max(w, len(name))) for name, w in columns]
    names = [c for c, _ in columns]
    print()
    target_desc = {1.0: "the search score alone (lambda 1)", 0.0: "the game result alone (lambda 0)"}.get(
        final_lam, f"the target at the final lambda, {final_lam:.3f}" if args.lam != args.end_lam else f"the target (lambda {final_lam:.3f})")
    if validating:
        kv("columns", f"train loss, val loss: the training and the held-out positions against {target_desc}")
        fixed = [(n, d) for n, d in (("vs score", "the search score"), ("vs result", "the game result")) if n in names]
        if fixed:
            kv("", f"{' / '.join(n for n, _ in fixed)}: the held-out positions against {' / '.join(d for _, d in fixed)} alone")
        kv("", "(losses compare with another run's only on the same data)")
    else:
        kv("columns", f"train loss: against {target_desc} (too few held-out positions for validation losses)")
    print()
    print("  " + "  ".join(f"{name:>{w}}" for name, w in columns), flush=True)

    t_run = time.time()
    total = args.epochs * per_epoch

    def row(epoch, seen, train_loss, rate, val=None, lam=None, lr=None):
        """A row of the table: a progress row while the epoch runs (no validation losses, lambda or lr), or its last."""
        elapsed = time.time() - t_run
        done = (epoch - 1) * per_epoch + seen
        cells = {"epoch": f"{epoch}/{args.epochs}", "done": f"{100 * seen // per_epoch}%", "train loss": f"{train_loss:.5f}",
                 "pos/s": f"{rate / 1000:.0f}k", "elapsed": fmt_duration(elapsed),
                 "ETA": "-" if done >= total else fmt_duration(elapsed / max(done, 1) * (total - done))}
        if val is not None:
            cells.update({"val loss": f"{val[0]:.5f}", "vs score": f"{val[1]:.5f}", "vs result": f"{val[2]:.5f}"})
        if lam is not None:
            cells.update({"lambda": f"{lam:.3f}", "lr": f"{lr:.1e}"})
        print("  " + "  ".join(f"{cells.get(name, ''):>{w}}" for name, w in columns), flush=True)

    best_val, written_epoch = float("inf"), 0
    history = {}   # epoch -> its validation losses
    interrupted = None
    epoch = 0
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            seen, loss_sum = 0, 0.0
            lam = args.lam + (args.end_lam - args.lam) * (epoch - 1) / max(args.epochs - 1, 1)
            epoch_lr = args.lr * args.lr_decay ** ((epoch - 1) // args.lr_step)
            for g in opt.param_groups:
                g["lr"] = epoch_lr
            next_row = args.progress_every
            for inputs, target, shown_target in batch_iter(data, train_idx, args.batch, lam, final_lam, score_model, stats, rng):
                target_t = torch.from_numpy(target).to(device, non_blocking=True)
                p = curve.prob_torch(model(*to_device(inputs, non_blocking=True)))
                loss = loss_of(p, target_t).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.clamp_weights()
                if shown_target is None:   # this epoch's lambda is the final one: the loss minimised is the one shown
                    shown = loss.item()
                else:
                    with torch.no_grad():
                        shown = loss_of(p.detach(), torch.from_numpy(shown_target).to(device)).mean().item()
                seen += len(target)
                loss_sum += shown * len(target)
                if args.progress_every and seen >= next_row:
                    next_row += args.progress_every * ((seen - next_row) // args.progress_every + 1)
                    if per_epoch - seen >= args.progress_every // 2:   # not just before the epoch's own row
                        row(epoch, seen, loss_sum / seen, seen / (time.time() - t0))
            rate = seen / max(time.time() - t0, 1e-9)
            val = validate(val_recs) if validating else None
            if val is not None:
                history[epoch] = val
            row(epoch, seen, loss_sum / max(seen, 1), rate, val, lam, epoch_lr)
            if stats["truncated"]:
                print(f"  note: {stats['truncated']:,} positions so far had more than {MAX_THREATS} threats and lost the excess "
                      "(raise MAX_THREATS in features.py if this is more than a handful)", flush=True)
            if args.keep_best and val is not None:
                if val[0] >= best_val:
                    continue   # not the best epoch so far: the network written stays
                best_val = val[0]
            write_net()
            written_epoch = epoch
    except KeyboardInterrupt:
        interrupted = epoch
    print()
    if interrupted:
        kv("stopped", f"in epoch {interrupted} (Ctrl-C): that epoch is discarded")
        if not written_epoch:
            kv("wrote", "nothing: no epoch of this run was complete")
            sys.exit(1)
    if history and written_epoch:
        last = max(history)
        kept = f"epoch {written_epoch} of {args.epochs}: val loss {history[written_epoch][0]:.5f}"
        if not args.keep_best:
            kept += ", the last epoch's (--keep-best keeps the lowest)"
        elif written_epoch != last:
            kept += f", the lowest (epoch {last}: {history[last][0]:.5f})"
        elif not interrupted:
            kept += ", the lowest - still falling at the end, so more epochs or a slower decay might help"
        else:
            kept += ", the lowest so far"
        kv("kept", kept)
    elif written_epoch:
        kv("kept", f"epoch {written_epoch} of {args.epochs}, the last (too few held-out positions to choose by)")
    report_written()


if __name__ == "__main__":
    main()
