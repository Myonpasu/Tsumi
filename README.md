# Tsumi

Tsumi is a chess engine for **selfmate chess**: ordinary chess rules, except that the player who gets
**checkmated wins**. (*Tsumi*, 詰み, is the shogi word for checkmate.)

Both players try to get themselves mated, and neither will ever mate the other voluntarily, so the only way to
win is to *force* it: leave the opponent no legal move but mating ones. That makes the game a contest of
compulsion — checks, pins, single-reply positions, a king boxed in by its own pieces — and very drawish
between strong players. A normal engine playing its worst moves gets nowhere: it assumes an opponent who
wants to win the ordinary way. Tsumi is built for the variant: its search is tuned for forcing lines, and its
NNUE evaluation is trained from its own self-play games with the training pipeline in this repository.

```
src/      the engine (C++17, UCI)
train/    the trainer (PyTorch): network, export, calibration, verification against the engine
tools/    training loop, match runner, terminal play, puzzle tools, opening books
tests/    test scripts, verified selfmate puzzles
```

## Contents

- [Getting started](#getting-started)
- [Rules and scores](#rules-and-scores)
- [Using the engine](#using-the-engine)
- [How it works](#how-it-works)
- [Training a network](#training-a-network)
- [Reference](#reference): data format, network file, tools, releasing
- [Acknowledgements](#acknowledgements) and [license](#license)

## Getting started

**Build** (x86-64, g++ or clang with C++17):

```bash
make -j                 # ./tsumi, optimised for this machine (add PEXT=1 on Intel, or AMD Zen 3 and later)
make -j universal       # ./tsumi-universal: runs on any x86-64 CPU with SSE4.2 and POPCNT (see CPU support)
make -j windows         # tsumi.exe, the universal build for Windows: cross-compiled from Linux with mingw-w64
```

On Windows itself, build in an MSYS2 MINGW64 shell with `make` or `make universal`. `make windows-avx512`
cross-compiles a dedicated AVX-512 Windows executable, and `make debug` builds with sanitizers. MSVC is not
supported (the code uses GCC built-ins).

**Get a network.** The network is Tsumi's only evaluation, and it is not part of the repository:

```bash
make net                # downloads the network this version is built for (and checks its hash)
```

The release archives include it. Networks are named after their contents, `tsumi-<the first 12 hex digits
of the file's SHA-256>.nnue`, and every published one is an asset of the [networks release](../../releases/tag/networks),
with its training checkpoint beside it (`make checkpoint`). Tsumi loads its own network (`NNUE::DEFAULT_NET`
in `src/nnue.h`) from its working directory or from the executable's directory; the `EvalFile` option names
another file. Without a network the engine still answers `uci`, `perft` and the like, but exits when asked to
search.

**Check it**:

```bash
./tsumi bench           # searches a set of positions: prints the nodes and the speed
./tsumi test            # perft suite: the rules
```

**Play it.** Any UCI GUI can run Tsumi, but GUIs score checkmate for the side that delivers it, which is the
wrong way round here, so use the tools in this repository to play and to run matches:

```bash
pip install chess                               # python-chess (training also needs numpy and torch: requirements.txt)
python3 tools/play.py                           # play against Tsumi in the terminal
python3 tools/match.py --e1 ./tsumi --e2 random --games 20 --nodes 5000     # Tsumi against a random mover
```

## Rules and scores

| Situation | Result |
|---|---|
| The side to move is checkmated | the side to move **wins** |
| Stalemate, the third occurrence of a position, the fifty-move rule | draw |
| Dead position: no sequence of moves can mate (K v K, KN v K, KB v K, only bishops left and all on one square colour) | draw |
| One side has only its king (a *bare king*) | drawn in practice: a bare king gives no check, so its opponent can never be mated; the bare king wins only if its opponent is left with nothing but mating moves |

The fifty-move rule draws once 100 half-moves have passed without a capture or a pawn move, unless the
100th half-move mates.

**Scores** are from the side to move's point of view, and positive means winning, that is, heading for being
mated:

- `score cp 150`: +1.50, good for the engine. The evaluation is calibrated so that **+1.00 is a 50% chance of
  winning**, the rest being mostly draws (see *Output units* under [Training](#training)).
- `score mate 3`: the engine can force its own mate in three moves (it wins).
- `score mate -2`: the engine will be forced to mate its opponent within two moves (it loses).

Internally the engine counts in millipawns and reports centipawns; the training data holds millipawns.

## Using the engine

### UCI options

| Option | Default | |
|---|---|---|
| `Hash` | 64 | hash table (MB), in large pages where the system allows (below) |
| `Threads` | 1 | search threads |
| `EvalFile` | the engine's own network | the network file; a relative path is also looked for next to the executable |
| `MultiPV` | 1 | lines to report |
| `MoveOverhead` | 30 | milliseconds kept in reserve on every move, for the GUI's and the system's delays |

Rather than play in a degraded way, Tsumi says why in an `info string` and exits: when asked to search without
a network, or when the system cannot provide the hash table asked for (`info string out of memory: could not
allocate 8192 MB for the hash table`). When a GUI plays Tsumi against itself, each engine has its own table,
and both must fit in memory at once.

**Large pages.** On Linux the hash table and the network's weights use transparent huge pages. On Windows
they need the *Lock pages in memory* right (Local Security Policy > Local Policies > User Rights Assignment,
then log on again); the engine says at `setoption name Hash` whether it got them, and works without them, a
little slower.

**Extra commands**: `d` (show the board), `eval` (the network's evaluation of the position), `legal` (the
legal moves, `+` marking checks), `perft N`, `bench [depth]`, `test` (perft suite), `nnuetest [plies]`
(incremental accumulator updates compared with full recomputation), `wait` (block until the search ends, for
scripts). `go mate N` stops at a forced selfmate in N; `go searchmoves` is supported.

**Command line**: `tsumi bench 13`, `tsumi test`, or any UCI commands as arguments, run in order before the
engine exits: `tsumi "setoption name EvalFile value my.nnue" "position startpos" "go depth 12"`.
`tsumi datagen` and `tsumi rescore` generate and relabel training data ([below](#training-a-network)).

**CPU support.** Only the network code benefits measurably from newer instruction sets, so the universal
build compiles just that code four times — for SSE4.2, AVX2, AVX-512 and AVX-512 VNNI — and picks the best set
the CPU supports at startup (the banner says which). All four give identical results. `TSUMI_ISA=avx2` (or
`sse4.2`, `avx512`) forces a lower set for testing. A build for a specific CPU (`make`, `make windows-avx512`)
prints what is missing and exits on a CPU that lacks its instructions.

## How it works

### Search

Iterative deepening with aspiration windows and principal variation search, a hash table, killer,
counter-move, history and continuation-history move ordering, late move reductions and pruning, futility and
reverse futility pruning, internal iterative reductions, and lazy SMP for several threads. The parts specific
to selfmate:

- **Every searched node generates its legal moves**, so mates and stalemates are exact everywhere and the
  number of legal moves is known: a side with one or two legal moves is about to be forced, which the network
  is told (below). The hash table is probed first, so a hit that ends a node costs no move generation.
- **No null-move pruning.** Zugzwang is the whole game, and null-move pruning assumes there is none.
- **Forcing moves first.** Checks are searched right after captures and reduced less than quiet moves, and a
  position with a single legal move is extended.
- **Quiescence search** covers captures, promotions, quiet checks at its first ply and every evasion when in
  check. Capture and check sequences can run very deep in this game, so it is bounded: losing captures are
  skipped, after its fifth ply only recaptures are searched, and after its tenth a node not in check stands
  pat.
- **A bare king means a draw.** The side facing a bare king can never win, and the bare king wins only if
  its opponent runs out of non-mating moves, which the search has to prove move by move. Such positions are
  evaluated as a draw, shaded a little against the side with the material while the game goes on, and more
  the more squares the bare king has, so that the search drives the king into a stalemate and ends the game.
  A repetition does not count as a draw against a bare king until it can be claimed (the bare side need not
  repeat), and at the root a stalemate on the board is preferred to a draw further ahead.

### Evaluation

The evaluation is an NNUE network, trained on self-play games ([below](#training-a-network)):

```
piece-square features ─┐
                       ├─> H units per side ─> clip, pairwise products ─> H/2 per side ─┐
threat features ───────┘                                                                ├─> 32 ─> 32 ─> 1
                       legal-move inputs (the side to move's move count, in check) ─────┘
```

- **Inputs.** Piece-square features (HalfKAv2_hm) — every piece, both kings included, by square and by the
  perspective's king square (32 king buckets, mirrored horizontally) — and **threat features**: one per
  (attacking piece, target square, what the target holds). A piece's
  targets are the pieces it attacks — the enemy king included (a check), its own pieces included (a defence),
  its own king not — and the empty squares next to either king (the flight squares). Kings are not attackers
  here (the king buckets already say what stands next to a king). There are 8,320 threat features; a position
  from self-play has about 17 of them active on average, and rarely more than 60. `src/nnue_features.h`
  defines both sets.
- **Accumulators.** Each perspective (the side to move's and the opponent's) has `H` units (1024 in the
  released network), the sums of the active features' rows (int8 weights), updated incrementally move by
  move, with a per-king-square cache for king moves. The threat changes of a move are worked out once and
  applied to both perspectives.
- **Head.** Each unit is clipped to [0, 1] and multiplied with its partner half the layer away, giving `H/2`
  products per perspective. The side to move's products, the opponent's, and the
  **legal-move inputs** feed two layers of 32 with clipped and squared clipped activations, and a skip
  connection from the first layer carries large evaluations past the clipped ones. The legal-move inputs
  are what selfmate turns on and what the board features show only indirectly: the side to move's number of
  legal moves (one-hot over 14 count classes, and the count itself scaled to 0–1) and whether it is in check.
  The search has the moves anyway, so they cost nothing.
- **Output buckets.** The head has 16 sets of weights, chosen by 4 piece-count classes times 4 classes of the
  side to move's legal-move count (0–1, 2–4, 5–11, 12 or more), so a side that is forced and a side that is
  free are evaluated by different weights.
- **Integer inference.** The engine computes in integers (int8 and int16 weights, int16 accumulators), with
  vector kernels for SSE4.2, AVX2, AVX-512 and AVX-512 VNNI, and a sparse first layer that skips the groups of
  products that are all zero. `train/verify_net.py` checks that the engine's evaluations equal a numpy
  implementation's bit for bit.

## Training a network

Training is a loop: the best network plays games against itself, a candidate is trained on them, starting
from the best network, and a match decides whether the candidate replaces it. `tools/pipeline.sh` runs the
loop; the pieces can also be run by hand.

### Setup

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # for a GPU, a CUDA build of torch
make -j && make featlib       # the engine, and train/libfeat.so: the engine's feature code for the trainer
make net checkpoint           # the released network and its checkpoint, to start from
mkdir -p nets && cp tsumi-*.nnue nets/best.nnue && cp tsumi-*.nnue.pt nets/best.nnue.pt
```

The loop starts from `nets/best.nnue`, and its candidates are warm-started from the checkpoint beside it,
`nets/best.nnue.pt` (the float weights, published with the network). Without a checkpoint each candidate is
trained from scratch.

### The loop

```bash
tools/pipeline.sh 0 9999                 # generations 0, 1, 2, ... until stopped
THREADS=32 GAMES=50000 tools/pipeline.sh 0 9999
```

Each generation *N*:

1. **Data**: `tsumi datagen` plays `GAMES` games with the best network → `data/genN/data.bin`.
2. **Calibration**: `train/calibrate.py` reports how well the search scores predict the results.
3. **Training**: `train/train.py` trains `nets/gen(N+1).nnue` on this generation's data and on earlier
   generations labelled by the same network (up to `HISTORY` generations). Data labelled by older networks is
   left out: it does not help.
4. **Verification**: `train/verify_net.py` checks the engine against numpy on the new network.
5. **Match**: `tools/match.py` plays the candidate against the best network. A candidate that scores at least
   even becomes `nets/best.nnue` (with its `.pt`).

The loop can be interrupted with Ctrl-C and run again with the same arguments at any time: completed
generations and phases are skipped, an interrupted datagen carries on from the games it has, and an
interrupted training or match starts again. Deleting a phase's product redoes it on the next run: delete `nets/gen(N+1).nnue` to train
generation *N*'s candidate again (and play its match), or `logs/match_genN.log` to play the match again. If the
deleted candidate had been accepted, the loop stops and says which network to put back as the best first. Logs
go to `logs/` (`datagen_genN.log`, `calibrate_genN.log`, `train_genN.log`, `match_genN.log` and its PGN), and
every generation ends with a summary.

The settings are environment variables:

| Variable | Default | |
|---|---|---|
| `THREADS` | all cores | datagen threads |
| `GAMES` | 20000 | games per generation |
| `NODES` | 10000 | search nodes per move in datagen |
| `BOOK_RANDOM_SHARE` | 50 | % of each generation's games that start from a random-play position (8–12 random plies) |
| `BOOK` | | the other games start from positions drawn from this book ([opening books](#opening-books)); without one, from the start position |
| `RANDOM_MOVE_PCT`, `RANDOM_MOVE_MINPLY`, `RANDOM_MOVE_MAXPLY` | 40, 0, 20 | random moves on 40% of plies 0–19 |
| `RANDOM_MULTI_PV`, `RANDOM_MULTI_PV_DIFF` | 4, 250 | a random move is one of the best 4 within 0.25 pawns of the best |
| `WRITE_MIN_PLY` | 5 | positions before this ply are not recorded |
| `HIDDEN` | 1024 | network width (units per side), a multiple of 64; must match the checkpoint it warm-starts from |
| `EPOCHS` | 6 | epochs per generation (the learning rate halves every 3) |
| `LR`, `LR_SCRATCH` | 1e-5, 1e-3 | learning rate when warm-starting, and from scratch |
| `LAMBDA`, `END_LAMBDA` | 0.8, 0.8 | weight of the search score against the game result, at the first and last epoch (moved linearly between them) |
| `ANCHOR_P` | 0.01 | the output curve's win probability at 0 (see [Training](#training)) |
| `KEEP_BEST` | 1 | keep the epoch with the lowest validation loss (0: the last) |
| `HISTORY`, `MAX_POSITIONS` | 10, 0 | at most this many generations of same-network data (0: all), and positions (0: no cap) |
| `RESUME` | 1 | warm-start from the best network's checkpoint (0: train from scratch) |
| `MATCH_GAMES`, `MATCH_NODES`, `MATCH_TC` | 1000, 50000, | the match: games, and nodes per move or a time control such as `10+0.1` |
| `MATCH_BOOK`, `MATCH_RANDOM_SHARE` | , 50 | the match's openings: 8 random plies from the start position each, or with a book this % of them random and the rest drawn from it |
| `CONCURRENCY` | half of `THREADS` | match games played at once |
| `DEVICE` | `cuda` if available | the trainer's device |
| `PROGRESS_EVERY` | 2000000 | a row of the training table every this many positions |

A generation of 20,000 games gives about 2.5 million positions. More games per generation and more nodes per
move both make better networks; raise `NODES` when the gains per generation flatten, keeping `MATCH_NODES`
well above it.

### Data generation

```bash
mkdir -p data/gen0
python3 tools/gen_book.py --count 20000 --random-share 50 --out data/gen0/book.epd
./tsumi datagen --net nets/best.nnue --threads 32 --games 20000 --nodes 10000 --book data/gen0/book.epd \
    --out data/gen0/data.bin --log data/gen0/games.log
```

Each game starts from the start position or, with `--book`, from the next position of the book (taken in a
random order, each once; see [opening books](#opening-books)). The engine plays at about `--nodes` nodes per
move (each search gets 80–120% of it). Every position the search scores is a candidate for recording, except
positions in check or whose best move is a capture (their scores are tactical), positions before
`--write-min-ply`, and positions already recorded in this run. A record holds the position, the search score
and, once the game ends, its result.

**Random moves.** On `--random-move-pct` % of the plies from `--random-move-minply` to
`--random-move-maxply` − 1, a second search with `--random-multi-pv` lines picks one of the moves within
`--random-multi-pv-diff` millipawns of the best (the best included): the games vary, and they go on at engine
strength, so their results still say something about the earlier positions.
A ply with no other move in the window keeps the best move. When the next search shows that a random move
cost more than the window allowed, the positions before it lose their result label (the game was steered by
a mistake) and are trained on their score alone. Datagen reports how often that happens.

**Adjudication**: a game is drawn after `--adj-draw-plies` (16) searches in a row within
±`--adj-draw-score` (10) millipawns from ply `--adj-draw-minply` (120) on, or from any ply when a side has a
bare king; `--max-plies` (400) ends a game as a draw.

`--resume` continues an interrupted run (it needs the `--log`, where the games are counted); without it, the
output is appended to. `tsumi datagen --help` lists every option; the defaults are in `src/datagen.h`.

### Opening books

A book is a file with a FEN or an EPD line per position (EPD operations after the four position fields are
ignored), or a PGN file of opening lines, each game played to its end. The pipeline's `BOOK` and
`MATCH_BOOK`, `tools/gen_book.py --base` and `tools/match.py --book` read both, plain or in a .zip, as the
books of the [Stockfish books repository](https://github.com/official-stockfish/books) come;
`tsumi datagen --book` reads a plain FEN/EPD file, which is what the pipeline hands it. Entries without a
position a game can start from are left out, and the tools say how many. Datagen takes a book's positions in
a random order, each once, and `match.py` plays each opening once with both colours.

`UHO_Lichess_4852_v1.epd.zip` there (unbalanced human openings) holds 2.6 million positions from Lichess
games, 2–16 plies into the game. With it as `BOOK`, the games that do not start from a random-play position
start from one of its positions instead of from the start position:

```bash
mkdir -p books && curl -L -o books/UHO_Lichess_4852_v1.epd.zip \
    https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip
BOOK=books/UHO_Lichess_4852_v1.epd.zip tools/pipeline.sh 0 9999
```

`BOOK_RANDOM_SHARE=0` starts every game from the book. In selfmate these openings produce more decisive games
than the start position but fewer than random-play positions (of 600 games each at 5,000 nodes per move: UHO
17%, start position 11%, random-play positions 32%), so the random-play share is worth keeping. Stefan Pohl's
PGN books in the same repository, such as `UHO_XXL_2022_+120_+149.pgn.zip` (226,629 lines of 8 moves after
which White stands 1.20–1.49 pawns better), work the same way.

The acceptance matches open with 8 random plies from the start position, never with the start position
itself (a deterministic engine would play the same game over and over). `MATCH_BOOK=books/UHO_Lichess_4852_v1.epd.zip`
draws half of their openings from a book instead (`MATCH_RANDOM_SHARE` sets the other half's share), so that
the match also covers the kind of opening the engine meets in real games, while the random openings keep
the decisive games coming. The match book need not be the training book: a different one tests the candidate
on openings its training data never started from. `tools/match.py --book FILE --book-share P` does the same
by hand.

### Training

```bash
python3 train/train.py --data 'data/gen*/data.bin' --out nets/gen1.nnue --resume nets/best.nnue.pt --keep-best
```

**Target.** Each position is trained towards a blend of its search score and its game result, as expected
results for the side to move (1 a win, ½ a draw, 0 a loss):

```
target = λ · model(score) + (1 − λ) · result
```

`model` is a win/draw/loss model of the search scores fitted to the run's own data (`train/wdl.py`), so that
the two parts agree on average. λ is `--lambda` (0.8); with a different `--end-lambda` it moves linearly from the
one to the other over the epochs.

**Output units.** The network's output *y* goes through a fixed curve: the side to move wins with probability
`win(y) = 1 / (1 + (1/p − 1)^(1 − y))` and loses with probability `win(−y)`, *p* being `--anchor-p` (0.01).
The loss is the squared difference between the expected result on that curve and the target. An output of
+1 is therefore a 50% chance of winning, whatever the scale of the search scores the network learnt from,
and the engine shows it as +1.00. The smaller *p*, the steeper the curve and the less weight balanced
positions get.

**The table.** The trainer prints its settings, then a row every `--progress-every` positions and one at the
end of each epoch:

```
  epoch  done  train loss  val loss  vs score  vs result       lr  pos/s  elapsed     ETA
    1/2  100%     0.03742   0.02179   0.01960    0.04038  1.0e-03     5k      13s     13s
    2/2  100%     0.02896   0.02811   0.02592    0.04690  1.0e-03     6k      29s       -
```

`val loss` is measured on whole games held out from training (`--val-fraction`, 2%), and `train loss` on the
training positions, both against the target at the final λ (a column shows λ when it changes), so they
compare from epoch to epoch; a widening gap between them is the network memorising its training games.
`vs score` and `vs result` are the held-out positions against the search score and against the game result
alone. With `--keep-best` the network written is the epoch with the lowest `val loss`, and a run can be
stopped with Ctrl-C at any time without losing it.

The trainer writes the network (`OUT.nnue`, with its units reordered for the engine's sparse first layer) and
its float checkpoint (`OUT.nnue.pt`). Training-only *feature factorisation* lets rare features share weights
with their king-independent and square-independent counterparts; it is folded into the network at export. `python3 train/train.py --help` lists the options.

### Relabelling data and changing the network's size

`tsumi rescore` searches the positions of existing data again with a network and writes them with the new
scores. Its default `--nodes 1` is a depth-1 search with quiescence: the network's own judgement with the
tactics resolved, thousands of positions per second per thread.

This is how to move to a network of another size (`HIDDEN`), which cannot warm-start from the current one.
A network trained from scratch on a few generations starts far behind one that has absorbed every
generation since the beginning. So relabel all of the data with the best network, train the new network
from scratch on all of it, and continue the loop from there:

```bash
./tsumi rescore --net nets/best.nnue --threads 16 --out data/rescored.bin --in data/gen*/data.bin
python3 train/train.py --data data/rescored.bin --out nets/wide.nnue --hidden 1536 --epochs 16 --lr-step 4 --keep-best
```

Then make the new network `nets/best.nnue` (with its `.pt`) and run the loop with `HIDDEN=1536`.

The trainer holds all the training data in memory (32 bytes per position); a single file is memory-mapped.

### Checking a network

```bash
python3 train/verify_net.py --net nets/gen1.nnue --data data/gen0/data.bin      # the engine against numpy
./tsumi "setoption name EvalFile value nets/gen1.nnue" "nnuetest 20000"         # incremental vs full accumulator updates
python3 tools/match.py --e1 ./tsumi --e2 ./tsumi --opt1 EvalFile=nets/gen1.nnue --opt2 EvalFile=nets/best.nnue \
    --games 1000 --nodes 50000 --concurrency 16
```

The match plays every opening with both colours. Selfmate has a strong first-move advantage, which the two
games of a pair largely cancel, so the Elo error bar is computed over the pairs. Most games between equal
networks are draws: 1000 games give about ±10 Elo.

**Sparring partners.** Matches at equal strength are mostly draws, so it helps to measure against other
players too: the random mover (`--e2 random`, which Tsumi should never lose to), and
[Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish), playing selfmate chess with its own
search and evaluation:

```bash
git clone --depth 1 https://github.com/fairy-stockfish/Fairy-Stockfish
(cd Fairy-Stockfish/src && make -j build ARCH=x86-64-avx2)
python3 tools/match.py --e1 ./tsumi --e2 Fairy-Stockfish/src/stockfish --games 100 --tc 10+0.1 --concurrency 8 \
    --opt2 VariantPath=tools/fairy-stockfish-selfmate.ini --opt2 UCI_Variant=selfmate
```

Flounder, a separate project, plays the move Stockfish rates worst, which makes a reasonable selfmate player:
`--e2 "python3 ../flounder/flounder.py" --opt2 Stockfish=/path/to/stockfish`.

## Reference

### Training data

Records of 32 bytes, little-endian (`src/datagen.h`):

| Field | Type | |
|---|---|---|
| `occ` | uint64 | occupied squares (a1 = bit 0) |
| `pcs` | 16 bytes | the pieces on the occupied squares in square order, 4 bits each, low half first (colour × 6 + type; white 0, black 1; pawn to king 0–5) |
| `score` | int16 | the search score, millipawns, side to move's view |
| `result` | int8 | the game result from White's view: 1 White won (was mated), 0 draw, −1 Black won |
| `stm` | uint8 | side to move: 0 White, 1 Black |
| `ply` | uint8 | game ply (at most 255) |
| `flags` | uint8 | bit 1 (value 2): the result is not a label for this position (see *Random moves*) |
| `next` | int16 | unused (0) |

A record holds neither castling rights nor an en-passant square, so the network's legal-move inputs count
neither castling nor en-passant moves. A data file is a plain sequence of records, so files can be
concatenated.

### Network file

`src/nnue.cpp` describes the format: an 8-byte magic (`TSUMINET`), a header of 13 uint32 values (format
version 4, the dimensions and the scale), the king-bucket table, and the weights. The engine reads the
dimensions from the file, so networks of other widths load without recompiling.

### Tools

| | |
|---|---|
| `tools/pipeline.sh` | the training loop |
| `tools/match.py` | engine against engine under selfmate rules: random, book or mixed openings played with both colours, fixed nodes, depth or a time control, concurrency, PGN, Elo with its error bar; `--e2 random` plays a random mover |
| `tools/play.py` | play against the engine in the terminal (SAN or UCI moves, `hint`, `undo`) |
| `tools/make_puzzles.py`, `tools/puzzles.py` | make selfmate puzzles with an independent python-chess solver, and check that the engine solves them |
| `tools/gen_book.py` | opening books for datagen: random-play positions, with the start position or positions drawn from another book |
| `tools/release_net.py` | name a network for publication after its SHA-256, and make it the engine's default |
| `tools/fairy-stockfish-selfmate.ini` | selfmate for [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish), as a sparring partner |
| `train/train.py` | the trainer |
| `train/calibrate.py` | how well a data set's search scores predict its results |
| `train/verify_net.py` | the engine's evaluations against numpy, bit for bit |
| `train/ftperm.py` | reorder a network's units for the engine's sparse first layer (same evaluations, faster); `train.py` does it for every network it writes |
| `tests/run_all.sh` | the tests: perft, move generation against python-chess, puzzles, network updates, engine against numpy, bench |
| `tests/universal.sh` | the universal build: every kernel set gives the same results, and with `qemu-user` the binary runs on emulated older CPUs |

`tests/run_all.sh` runs after `make` and `make featlib`. It needs a network (`NET=file`, else `nets/best.nnue`,
the first `.nnue` in `nets/`, or the one `make net` downloaded) and, for the check against numpy, data
(`DATA=file`, else the first file in `data/`).

### Releasing

`.github/workflows/build.yml` builds and tests every push, and on a tag `v<version>` packages the Windows and
Linux executables with the network and drafts the release. To publish a network, name it and make it the
engine's default, then upload the two files to the `networks` release (a pre-release, so that it never
counts as the latest):

```bash
python3 tools/release_net.py nets/best.nnue --set-default      # tsumi-<hash>.nnue and .nnue.pt; sets src/nnue.h
gh release create networks --prerelease --title Networks --notes "Tsumi's networks" tsumi-*.nnue*   # the first time
gh release upload networks tsumi-*.nnue*                       # afterwards
```

Then commit, and for a release set `VERSION` in `src/uci.cpp`, tag `v<version>` and push the tag; when the
workflow has drafted the release, write its notes and publish it.

## Acknowledgements

Tsumi owes much of its design to [Stockfish](https://github.com/official-stockfish/Stockfish): the board
representation and move generation (magic bitboards), the network architecture and its quantisation, the
threat features, the training practices, time management and many search techniques follow Stockfish's, and
some of the code is adapted from it. Stockfish is copyright the Stockfish developers and licensed under the
GPL-3.0. The Python tools use [python-chess](https://github.com/niklasf/python-chess) for the rules of chess.

## License

Tsumi is free software under the [GNU General Public License v3.0](LICENSE).
