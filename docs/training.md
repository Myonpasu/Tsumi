Training is a loop in which the best network plays games against itself, a candidate is trained on them,
starting from the best network, and a match decides whether the candidate replaces it. `tools/pipeline.sh`
runs the loop, and the pieces can also be run manually.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # for a GPU, a CUDA build of torch
make -j && make featlib       # the engine, and train/libfeat.so: the engine's feature code for the trainer
make net checkpoint           # the released network and its checkpoint, to start from
mkdir -p nets && cp tsumi-*.nnue nets/best.nnue && cp tsumi-*.nnue.pt nets/best.nnue.pt
```

The loop starts from `nets/best.nnue`, and its candidates are warm-started from the checkpoint beside it,
`nets/best.nnue.pt` (the float weights, published with the network). Without a checkpoint each candidate is
trained from scratch.

## The loop

```bash
tools/pipeline.sh 0 9999                 # generations 0, 1, 2, ... until stopped
THREADS=32 GAMES=50000 tools/pipeline.sh 0 9999
```

Each generation *N*:

1. **Data**: `tsumi datagen` plays `GAMES` games with the best network → `data/genN/data.bin`.
2. **Calibration**: `train/calibrate.py` reports how well the search scores predict the results.
3. **Training**: `train/train.py` trains `nets/gen(N+1).nnue` on this generation's data and on earlier
   generations labelled by the same network (up to `HISTORY` generations). Data labelled by older networks
   does not help and is left out.
4. **Verification**: `train/verify_net.py` checks the engine against numpy on the new network.
5. **Match**: `tools/match.py` plays the candidate against the best network. A candidate that scores at least
   even becomes `nets/best.nnue` (with its `.pt`).

The loop can be interrupted with Ctrl-C and run again with the same arguments at any time. Completed
generations and phases are skipped, an interrupted datagen carries on from the games it has, and an
interrupted training or match starts again. Deleting a phase's product redoes it on the next run: delete
`nets/gen(N+1).nnue` to train generation *N*'s candidate again (and play its match), or `logs/match_genN.log`
to play the match again. If the deleted candidate had been accepted, the loop stops and says which network
to put back as the best first. Logs go to `logs/` (`datagen_genN.log`, `calibrate_genN.log`,
`train_genN.log`, and `match_genN.log` and its PGN), and every generation ends with a summary.

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
| `ANCHOR_P` | 0.01 | the output curve's win probability at 0 (see [output units](#output-units)) |
| `KEEP_BEST` | 1 | keep the epoch with the lowest validation loss (0: the last) |
| `HISTORY`, `MAX_POSITIONS` | 10, 0 | at most this many generations of same-network data (0: all), and positions (0: no cap) |
| `RESUME` | 1 | warm-start from the best network's checkpoint (0: train from scratch) |
| `MATCH_GAMES`, `MATCH_NODES`, `MATCH_TC` | 1000, 50000, | the match: games, and nodes per move or a time control such as `10+0.1` |
| `MATCH_BOOK`, `MATCH_RANDOM_SHARE` | , 50 | the match's openings: 8 random plies from the start position each, or with a book this % of them random and the rest drawn from it |
| `CONCURRENCY` | half of `THREADS` | match games played at once |
| `DEVICE` | `cuda` if available | the trainer's device |
| `PROGRESS_EVERY` | 2000000 | a row of the training table every this many positions |

A generation of 20,000 games gives about 2.5 million positions. More games per generation and more nodes per
move both make better networks. Raise `NODES` when the gains per generation flatten, keeping `MATCH_NODES`
well above it.

## Data generation

```bash
mkdir -p data/gen0
python3 tools/gen_book.py --count 20000 --random-share 50 --out data/gen0/book.epd
./tsumi datagen --net nets/best.nnue --threads 32 --games 20000 --nodes 10000 --book data/gen0/book.epd \
    --out data/gen0/data.bin --log data/gen0/games.log
```

Each game starts from the start position or, with `--book`, from the next position of the book (taken in a
random order, each once, as described under [opening books](#opening-books)). The engine plays at about
`--nodes` nodes per move (each search gets 80–120% of it). Every position the search scores is a candidate for
recording, except positions in check or whose best move is a capture (their scores are tactical), positions
before `--write-min-ply`, and positions already recorded in this run. A record holds the position, the search
score, and, once the game ends, its result.

**Random moves.** On `--random-move-pct` % of the plies from `--random-move-minply` to
`--random-move-maxply` − 1, a second search with `--random-multi-pv` lines picks one of the moves within
`--random-multi-pv-diff` millipawns of the best (the best included), so the games vary but go on at engine
strength, and their results still say something about the earlier positions. A ply with no other move in the
window keeps the best move. When the next search shows that a random move cost more than the window allowed,
the positions before it lose their result label (the game was steered by a mistake) and are trained on their
score alone. Datagen reports how often that happens.

**Adjudication.** A game is drawn after `--adj-draw-plies` (16) searches in a row within
±`--adj-draw-score` (10) millipawns from ply `--adj-draw-minply` (120) on, or from any ply when a side has a
bare king. `--max-plies` (400) ends a game as a draw.

`--resume` continues an interrupted run (it needs the `--log`, where the games are counted). Without it, the
output is appended to. `tsumi datagen --help` lists every option, and the defaults are in `src/datagen.h`.

## Opening books

A book is a file with a FEN or an EPD line per position (EPD operations after the four position fields are
ignored), or a PGN file of opening lines, each game played to its end. The pipeline's `BOOK` and
`MATCH_BOOK`, `tools/gen_book.py --base`, and `tools/match.py --book` read both, plain or in a .zip, as the
books of the [Stockfish books repository](https://github.com/official-stockfish/books) come.
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
than the start position but fewer than random-play positions (17%, 11%, and 32% respectively, over 600 games
each at 5,000 nodes per move), so the random-play share is worth keeping. Stefan Pohl's
PGN books in the same repository, such as `UHO_XXL_2022_+120_+149.pgn.zip` (226,629 lines of 8 moves after
which White stands 1.20–1.49 pawns better), work the same way.

The acceptance matches open with 8 random plies from the start position, never with the start position
itself (a deterministic engine would play the same game over and over).
`MATCH_BOOK=books/UHO_Lichess_4852_v1.epd.zip` draws half of their openings from a book instead
(`MATCH_RANDOM_SHARE` sets the other half's share), so that the match also covers the kind of opening the
engine meets in real games, while the random openings keep the decisive games coming. The match book need
not be the training book, and a different one tests the candidate on openings its training data never
started from. `tools/match.py --book FILE --book-share P` does the same manually.

## The trainer

```bash
python3 train/train.py --data 'data/gen*/data.bin' --out nets/gen1.nnue --resume nets/best.nnue.pt --keep-best
```

**Target.** Each position is trained towards a blend of its search score and its game result, as expected
results for the side to move (1 a win, ½ a draw, 0 a loss):

```
target = λ · model(score) + (1 − λ) · result
```

`model` is a win/draw/loss model of the search scores fitted to the run's own data (`train/wdl.py`), so that
the two parts agree on average. λ is `--lambda` (0.8). With a different `--end-lambda` it moves linearly from
the one to the other over the epochs.

### Output units

The network's output *y* goes through a fixed curve, on which the side to move wins with probability
`win(y) = 1 / (1 + (1/p − 1)^(1 − y))` and loses with probability `win(−y)`, *p* being `--anchor-p` (0.01).
The loss is the squared difference between the expected result on that curve and the target. An output of
+1 is therefore a 50% chance of winning, whatever the scale of the search scores the network learnt from,
and the engine shows it as +1.00. The smaller *p*, the steeper the curve and the less weight balanced
positions get.

### The table

The trainer prints its settings, then a row every `--progress-every` positions and one at the end of each
epoch:

```
  epoch  done  train loss  val loss  vs score  vs result       lr  pos/s  elapsed     ETA
    1/2  100%     0.03742   0.02179   0.01960    0.04038  1.0e-03     5k      13s     13s
    2/2  100%     0.02896   0.02811   0.02592    0.04690  1.0e-03     6k      29s       -
```

`val loss` is measured on whole games held out from training (`--val-fraction`, 2%), and `train loss` on the
training positions, both against the target at the final λ (a column shows λ when it changes), so they
compare from epoch to epoch. A widening gap between them is the network memorising its training games.
`vs score` and `vs result` are the held-out positions against the search score and against the game result
alone. With `--keep-best` the network written is the epoch with the lowest `val loss`, and a run can be
stopped with Ctrl-C at any time without losing it.

The trainer writes the network (`OUT.nnue`, with its units reordered for the engine's sparse first layer) and
its float checkpoint (`OUT.nnue.pt`). Training-only *feature factorisation* lets rare features share weights
with their king-independent and square-independent counterparts, and it is folded into the network at export.
The trainer holds all the training data in memory (32 bytes per position), and a single file is memory-mapped.
`python3 train/train.py --help` lists the options.

## Relabelling data and changing the network's size

`tsumi rescore` searches the positions of existing data again with a network and writes them with the new
scores. Its default `--nodes 1` is a depth-1 search with quiescence, which is the network's own judgement
with the tactics resolved, at thousands of positions per second per thread.

This is how to move to a network of another size (`HIDDEN`), which cannot warm-start from the current one.
A network trained from scratch on a few generations starts far behind one that has absorbed every
generation since the beginning. So relabel all of the data with the best network, train the new network
from scratch on all of it, and continue the loop from there:

```bash
./tsumi rescore --net nets/best.nnue --threads 16 --out data/rescored.bin --in data/gen*/data.bin
python3 train/train.py --data data/rescored.bin --out nets/wide.nnue --hidden 1536 --epochs 16 --lr-step 4 --keep-best
```

Then make the new network `nets/best.nnue` (with its `.pt`) and run the loop with `HIDDEN=1536`.

## Checking a network

```bash
python3 train/verify_net.py --net nets/gen1.nnue --data data/gen0/data.bin      # the engine against numpy
./tsumi "setoption name EvalFile value nets/gen1.nnue" "nnuetest 20000"         # incremental vs full accumulator updates
python3 tools/match.py --e1 ./tsumi --e2 ./tsumi --opt1 EvalFile=nets/gen1.nnue --opt2 EvalFile=nets/best.nnue \
    --games 1000 --nodes 50000 --concurrency 16
```

The match plays every opening with both colours. Selfmate has a strong first-move advantage, which the two
games of a pair largely cancel, so the Elo error bar is computed over the pairs. Most games between equal
networks are draws, and 1000 games give about ±10 Elo.

**Sparring partners.** Matches at equal strength are mostly draws, so it helps to measure against other
players too. One is the random mover (`--e2 random`), which Tsumi should never lose to. Another is
[Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish), which has had the game built in as
`misere` (misère chess) since August 2024, after its last binary release, so it is built from source:

```bash
git clone --depth 1 https://github.com/fairy-stockfish/Fairy-Stockfish
(cd Fairy-Stockfish/src && make -j build ARCH=x86-64-avx2)
python3 tools/match.py --e1 ./tsumi --e2 Fairy-Stockfish/src/stockfish --games 100 --tc 10+0.1 --concurrency 8 \
    --opt2 UCI_Variant=misere
```

Its evaluation is its handcrafted chess one with a selfmate endgame rule. The
[Fairy-Stockfish NNUE page](https://fairy-stockfish.github.io/nnue/) also has a network for the variant,
`misere-c488fb592f7b.nnue`, a few Elo stronger, which `--opt2 EvalFile=path/to/misere-c488fb592f7b.nnue`
loads (Fairy-Stockfish uses a network only when its file name starts with the variant's name).

Flounder, a separate project, plays the move Stockfish rates worst, which makes a reasonable selfmate player.
`tools/match.py` runs it with `--e2 "python3 ../flounder/flounder.py" --opt2 Stockfish=/path/to/stockfish`.

## Training data

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
