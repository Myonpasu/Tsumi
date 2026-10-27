The evaluation is an NNUE network, trained on self-play games ([Training](training.md)):

```
piece-square features ─┐
                       ├─> H units per side ─> clip, pairwise products ─> H/2 per side ─┐
threat features ───────┘                                                                ├─> 32 ─> 32 ─> 1
                       legal-move inputs (the side to move's move count, in check) ─────┘
```

- **Piece-square features** (HalfKAv2_hm). One for every piece, both kings included, by its square and by the
  perspective's king square (32 king buckets, mirrored horizontally).
- **Threat features.** One per (attacking piece, target square, what the target holds). A piece's targets are
  the pieces it attacks, including the enemy king (a check) and its own pieces (a defence) but not its own
  king, and the empty squares next to either king (the flight squares). Kings are not attackers here (the king
  buckets already say what stands next to a king). There are 8,320 threat features, of which a position from
  self-play has about 17 active on average, and rarely more than 60. `src/nnue_features.h` defines both sets.
- **Accumulators.** Each perspective (the side to move's and the opponent's) has `H` units (1024 in the
  released network), the sums of the active features' rows (int8 weights), updated incrementally move by
  move, with a per-king-square cache for king moves. The threat changes of a move are worked out once and
  applied to both perspectives.
- **Head.** Each unit is clipped to [0, 1] and multiplied with its partner half the layer away, giving `H/2`
  products per perspective. The side to move's products, the opponent's, and the **legal-move inputs** feed
  two layers of 32 with clipped and squared clipped activations, and a skip connection from the first layer
  carries large evaluations past the clipped ones. The legal-move inputs are what selfmate turns on and what
  the board features show only indirectly: the side to move's number of legal moves (one-hot over 14 count
  classes, and the count itself scaled to 0–1) and whether it is in check. The search has the moves anyway,
  so they cost nothing.
- **Output buckets.** The head has 16 sets of weights, chosen by 4 piece-count classes times 4 classes of the
  side to move's legal-move count (0–1, 2–4, 5–11, 12 or more), so a side that is forced and a side that is
  free are evaluated by different weights.
- **Integer inference.** The engine computes in integers (int8 and int16 weights, int16 accumulators), with
  vector kernels for SSE4.2, AVX2, AVX-512, and AVX-512 VNNI, and a sparse first layer that skips the groups of
  products that are all zero. `train/verify_net.py` checks that the engine's evaluations equal a numpy
  implementation's bit for bit.

## Output units

The network's output goes through a fixed curve chosen so that an output of +1 is a 50% chance of winning
for the side to move, whatever the scale of the search scores it was trained on, and the engine shows it as
+1.00. *Output units* under [Training](training.md) gives the curve.

## Network file

`src/nnue.cpp` describes the format: an 8-byte magic (`TSUMINET`), a header of 13 uint32 values (format
version 4, the dimensions, and the scale), the king-bucket table, and the weights. The engine reads the
dimensions from the file, so networks of other widths load without recompiling. The units of a network
written by the trainer are ordered for the sparse first layer (`train/ftperm.py`), which changes no
evaluation.
