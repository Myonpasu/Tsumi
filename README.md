# Tsumi

Tsumi is a chess engine for **selfmate chess**: ordinary chess rules, except that the player who gets
**checkmated wins**. (*Tsumi*, 詰み, is the shogi word for checkmate.)

Neither player will ever mate the other voluntarily, so the only way to win is to force it: leave the
opponent no legal move but mating ones. The game is a contest of compulsion — checks, pins, single-reply
positions, a king boxed in by its own pieces — and a normal engine playing its worst moves is no good at it,
because it assumes an opponent who wants to win the ordinary way. Tsumi's search is built for forcing lines,
and its NNUE evaluation is trained on its own self-play games with the training pipeline in this repository.
The [documentation](docs) describes both in detail.

## Getting started

The [releases](../../releases) have Windows and Linux executables with the network. To build from source
(x86-64, g++ or clang with C++17):

```bash
make -j              # ./tsumi, optimised for this machine
make -j universal    # ./tsumi-universal: any x86-64 CPU with SSE4.2 and POPCNT; picks AVX2 or AVX-512 kernels at startup
make -j windows      # tsumi.exe, cross-compiled from Linux with mingw-w64
make net             # downloads the network this version is built for
```

The network is Tsumi's only evaluation and is not part of the repository; the engine looks for it in its
working directory or next to the executable. [Using the engine](docs/using-the-engine.md) covers
Windows builds, CPU support, large pages and the network files.

Try it:

```bash
./tsumi bench                     # searches a set of positions: nodes and speed
./tsumi test                      # perft suite
pip install chess                 # python-chess, for the tools
python3 tools/play.py             # play against Tsumi in the terminal
python3 tools/match.py --e1 ./tsumi --e2 random --games 20 --nodes 5000     # against a random mover
```

Any UCI GUI can run Tsumi, but GUIs score checkmate for the side that delivers it, which is the wrong way
round here, so play and run matches with the tools in this repository.

## Rules and scores

| Situation | Result |
|---|---|
| The side to move is checkmated | the side to move **wins** |
| Stalemate, the third occurrence of a position, the fifty-move rule | draw |
| Dead position: no sequence of moves can mate | draw |
| One side has only its king | drawn in practice: a bare king gives no check, so its opponent can never be mated |

Scores are from the side to move's point of view, and positive means winning, that is, heading for being
mated. `score cp 150` is +1.50 and good for the engine; +1.00 is calibrated to a 50% chance of winning, the
rest being mostly draws. `score mate 3` means the engine can force its own mate in three moves, and
`score mate -2` that it will be forced to mate its opponent within two.

## UCI options and commands

| Option | Default | |
|---|---|---|
| `Hash` | 64 | hash table in MB (in large pages where the system allows) |
| `Threads` | 1 | search threads |
| `EvalFile` | the engine's own network | the network file; a relative path is also looked for next to the executable |
| `MultiPV` | 1 | lines to report |
| `MoveOverhead` | 30 | milliseconds kept in reserve on every move |

Extra commands: `d` (the board), `eval`, `legal`, `perft N`, `bench [depth]`, `test` (perft suite),
`nnuetest [plies]` and `wait`; `go mate N` and `go searchmoves` are supported. Commands on the command line
run in order before the engine exits: `tsumi bench 13`, or
`tsumi "setoption name EvalFile value my.nnue" "position startpos" "go depth 12"`. `tsumi datagen` and
`tsumi rescore` generate and relabel training data.

## How it works

**Search.** Iterative deepening with aspiration windows and principal variation search, a hash table,
killer, counter-move, history and continuation-history move ordering, late move reductions and pruning,
futility and reverse futility pruning, internal iterative reductions, and lazy SMP. For selfmate: every node
generates its legal moves, so mates and stalemates are exact everywhere and the move count is known; there is
no null-move pruning, since zugzwang is the whole game; checks are searched early and reduced less,
single-reply positions are extended, and the quiescence search of captures, checks and evasions is bounded
in depth; a position with a bare king is a draw, which the search steers towards a stalemate.
[Search](docs/search.md) has the details.

**Evaluation.** An NNUE network: HalfKAv2_hm piece-square features and threat features, 1024 units per
perspective, pairwise products, two layers of 32 and 16 output buckets, with inputs the board alone does not
give — the side to move's number of legal moves and whether it is in check. Inference is in integers, with
SSE4.2, AVX2 and AVX-512 kernels that give identical results. [Evaluation](docs/evaluation.md) has the
architecture and the network file format.

## Training

The best network plays games against itself, a candidate is trained on them, starting from the best network,
and a match decides whether the candidate replaces it. `tools/pipeline.sh` runs the loop; its pieces
(`tsumi datagen`, `train/train.py`, `tools/match.py`) can also be run by hand.

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
make -j && make featlib && make net checkpoint
mkdir -p nets && cp tsumi-*.nnue nets/best.nnue && cp tsumi-*.nnue.pt nets/best.nnue.pt
tools/pipeline.sh 0 9999          # generations 0, 1, 2, ... until stopped
```

[Training](docs/training.md) explains the pipeline and its settings, data generation, opening books, the
trainer and the data format.

## Acknowledgements

Tsumi owes much of its design to [Stockfish](https://github.com/official-stockfish/Stockfish): the board
representation and move generation (magic bitboards), the network architecture and its quantisation, the
threat features, the training practices, time management and many search techniques follow Stockfish's, and
some of the code is adapted from it. Stockfish is copyright the Stockfish developers and licensed under the
GPL-3.0. The Python tools use [python-chess](https://github.com/niklasf/python-chess) for the rules of chess.

## License

Tsumi is free software under the [GNU General Public License v3.0](LICENSE).
