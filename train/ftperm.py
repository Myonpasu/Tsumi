#!/usr/bin/env python3
"""Reorder the units of a network so that the engine's sparse first layer skips more work.

The first layer of the head (src/nnue.cpp, fc0_sparse_*) reads the pairwise products in chunks of four
consecutive inputs and skips every chunk whose four products are all zero.  Most products are zero in a typical
position, but a chunk is skipped only when all four of its products are.  The units can be listed in any order
without changing a single evaluation - each unit's weights move with it, and unit j and unit j + H/2 move
together, as their product is one input - so this tool lists them so that products which are zero in the same
positions share chunks.  Many more chunks are then all zero, and the engine searches a few percent more nodes per
second with identical results.  The idea is Stockfish's (nnue-pytorch's ftperm.py).

    python3 train/ftperm.py nets/best.nnue --data 'data/gen*/data.bin' [--out nets/best.nnue]

train.py reorders every network it writes (its .pt checkpoint keeps the trained order, which is the same
network), so this is only needed for a network written otherwise.

The order is chosen on a sample of positions: which products are zero in each (both perspectives), from the
quantised weights exactly as the engine computes them.  Units are paired by how often their products are zero
together, the pairs paired again into chunks of four, and then units are swapped between chunks while any swap
makes more chunks all zero.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import QA, RECORD_DTYPE, batch_features
from model import read_net, write_net

CHUNK = 4                 # inputs per chunk of the engine's sparse first layer (one 32-bit lane of int8 inputs)
DEFAULT_POSITIONS = 16384


def _wrap16(x):
    """int16 wrap-around of exact integer sums held in float32 (the engine's accumulators are int16)."""
    return torch.remainder(x + 32768.0, 65536.0) - 32768.0


def product_zero_masks(net, recs, device: "torch.device | str" = "cpu", batch=4096):
    """[2N, H/2] bool: which pairwise products are zero in each perspective of N positions, computed from the
    quantised weights as the engine does (src/nnue.cpp, pair_inputs): unit = piece-square sum + threat sum,
    clipped to 0..QA, product (a * b + 256) >> 9 - zero exactly when a * b < 256."""
    H = int(net["H"])
    half = H // 2
    # Float32 holds the integer sums exactly (|sum| < 2^24).  An extra zero row for the padding indices.
    psq_w = torch.cat([torch.from_numpy(net["psq_w"].astype(np.float32)), torch.zeros(1, H)]).to(device)
    threat_w = torch.cat([torch.from_numpy(net["threat_w"].astype(np.float32)), torch.zeros(1, H)]).to(device)
    bias = torch.from_numpy(net["bias"].astype(np.float32)).to(device)
    masks = []
    for i in range(0, len(recs), batch):
        us, them, tus, tthem = batch_features(recs[i:i + batch], net["king_buckets"], max_threats=256)[:4]
        for idx, tidx in ((us, tus), (them, tthem)):   # padding indices: the extra zero rows
            psq = _wrap16(bias + F.embedding_bag(torch.from_numpy(idx).to(device), psq_w, mode="sum"))
            thr = _wrap16(F.embedding_bag(torch.from_numpy(tidx).to(device), threat_w, mode="sum"))
            v = torch.clamp(psq + thr, 0, QA)
            masks.append((v[:, :half] * v[:, half:] < 256).cpu())
    return torch.cat(masks).numpy()


def chunk_zero_rate(Z, perm):
    """Fraction of (perspective, chunk) pairs whose products are all zero when the products are in `perm` order."""
    return float(Z[:, perm].reshape(len(Z), -1, CHUNK).all(axis=2).mean())


def _greedy_matching(C):
    """Pair up the rows of a symmetric score matrix, highest score first (a greedy maximum-weight matching)."""
    m = C.shape[0]
    C = C.copy()
    np.fill_diagonal(C, -1)
    used = np.zeros(m, dtype=bool)
    pairs = []
    for f in np.argsort(-C, axis=None, kind="stable"):
        i, j = divmod(int(f), m)
        if i < j and not used[i] and not used[j]:
            used[i] = used[j] = True
            pairs.append((i, j))
            if len(pairs) == m // 2:
                break
    return pairs


def find_permutation(Z, device: "torch.device | str" = "cpu", max_swap_rounds=50):
    """An order of the products (a permutation of 0..H/2-1) under which chunks of CHUNK consecutive products
    are often all zero, from a [rows, H/2] bool matrix of zero products."""
    half = Z.shape[1]
    assert half % CHUNK == 0 and CHUNK == 4
    Zt = torch.from_numpy(Z.astype(np.float32)).to(device)
    # 1. Pairs of products that are often zero together, then pairs of those pairs: chunks of four.
    pairs = _greedy_matching((Zt.T @ Zt).cpu().numpy())
    both = torch.stack([Zt[:, i] * Zt[:, j] for i, j in pairs], dim=1)
    quads = _greedy_matching((both.T @ both).cpu().numpy())
    perm = np.array([u for a, b in quads for u in (*pairs[a], *pairs[b])], dtype=np.int64)
    # 2. Swaps between chunks.  For product a in chunk A and b in chunk B, swapping them changes the number of
    # all-zero chunks by rest(A without a).Z_b + rest(B without b).Z_a - zero(A) - zero(B), where rest is the
    # product of the chunk's other three masks; each round applies the best swaps that share no chunk.
    chunk_of = np.arange(half) // CHUNK
    same = torch.from_numpy(chunk_of[:, None] == chunk_of[None, :]).to(device)
    others = [[j for j in range(CHUNK) if j != k] for k in range(CHUNK)]
    for _ in range(max_swap_rounds):
        P = Zt[:, torch.from_numpy(perm).to(device)]
        Bk = P.reshape(len(P), half // CHUNK, CHUNK)
        zero = Bk.prod(dim=2).sum(dim=0)[torch.from_numpy(chunk_of).to(device)]
        rest = torch.stack([Bk[:, :, o].prod(dim=2) for o in others], dim=2).reshape(len(P), half)
        M = rest.T @ P
        gain = (M + M.T - zero[:, None] - zero[None, :]).masked_fill(same, -1.0).cpu().numpy()
        if gain.max() <= 0:
            break
        busy = np.zeros(half // CHUNK, dtype=bool)
        for f in np.argsort(-gain, axis=None, kind="stable"):
            a, b = divmod(int(f), half)
            if gain[a, b] <= 0:
                break
            if busy[chunk_of[a]] or busy[chunk_of[b]]:
                continue
            busy[chunk_of[a]] = busy[chunk_of[b]] = True
            perm[a], perm[b] = perm[b], perm[a]
    return perm


def permute_units(net, perm):
    """The same network with its units reordered: product k of the new order is product perm[k] of the old,
    i.e. unit k is old unit perm[k] and unit k + H/2 is old unit perm[k] + H/2."""
    half = int(net["H"]) // 2
    P = np.concatenate([perm, perm + half])
    out = dict(net)
    out["psq_w"] = np.ascontiguousarray(net["psq_w"][:, P])
    out["bias"] = np.ascontiguousarray(net["bias"][P])
    out["threat_w"] = np.ascontiguousarray(net["threat_w"][:, P])
    H = int(net["H"])
    P0 = np.concatenate([P, np.arange(H, net["fc0_w"].shape[2])])   # the legal-move inputs stay where they are
    out["fc0_w"] = np.ascontiguousarray(net["fc0_w"][:, :, P0])  # inputs: the side to move's products, then the opponent's
    return out


def permute_file(src, recs, dst=None, device: "torch.device | str" = "cpu", held_out=0.25, seed=1):
    """Reorder the units of the network in `src` (written to `dst`, default in place), choosing the order on the
    positions `recs`.  Returns the all-zero chunk rates (before, after) on a held-out part of them."""
    net = read_net(src)
    Z = product_zero_masks(net, recs, device)
    rows = np.random.default_rng(seed).permutation(len(Z))
    n_test = int(len(Z) * held_out) if len(Z) >= 64 else 0
    test, fit = Z[rows[:n_test]], Z[rows[n_test:]]
    perm = find_permutation(fit, device)
    if n_test == 0:
        test = fit
    before = chunk_zero_rate(test, np.arange(Z.shape[1]))
    after = chunk_zero_rate(test, perm)
    tmp = (dst or src) + ".tmp"
    write_net(permute_units(net, perm), tmp)
    os.replace(tmp, dst or src)
    return before, after


def sample_records(patterns, count, seed=1):
    """`count` positions drawn at random from the data files (memory-mapped: only the sample is read)."""
    files = sorted({f for p in patterns for f in glob.glob(p)})
    maps = [np.memmap(f, dtype=RECORD_DTYPE, mode="r") for f in files if os.path.getsize(f) >= RECORD_DTYPE.itemsize]
    if not maps:
        raise SystemExit(f"no data in {' '.join(patterns)}")
    starts = np.cumsum([0] + [len(m) for m in maps])
    pick = np.sort(np.random.default_rng(seed).choice(starts[-1], size=min(count, int(starts[-1])), replace=False))
    parts = []
    for m, lo, hi in zip(maps, starts[:-1], starts[1:]):
        sel = pick[(pick >= lo) & (pick < hi)] - lo
        if len(sel):
            parts.append(np.asarray(m[sel]))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", help="the network (.nnue)")
    ap.add_argument("--data", nargs="+", required=True, help="data files or globs (tsumi datagen output) to sample positions from")
    ap.add_argument("--out", help="output network (default: overwrite the input)")
    ap.add_argument("--positions", type=int, default=DEFAULT_POSITIONS, help=f"positions to sample (default {DEFAULT_POSITIONS})")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    recs = sample_records(args.data, args.positions, args.seed)
    try:
        before, after = permute_file(args.net, recs, args.out, torch.device(args.device), seed=args.seed)
    except ValueError as e:
        sys.exit(str(e))
    print(f"wrote {args.out or args.net}: first-layer input chunks all zero {before:.1%} -> {after:.1%} "
          f"(positions not used to choose the order), same evaluations; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
