## Building

The engine builds on x86-64 with g++ or clang (C++17):

```bash
make -j                 # ./tsumi, optimised for this machine (add PEXT=1 on Intel, or AMD Zen 3 and later)
make -j universal       # ./tsumi-universal: runs on any x86-64 CPU with SSE4.2 and POPCNT
make -j windows         # tsumi.exe, the universal build for Windows: cross-compiled from Linux with mingw-w64
make -j windows-avx512  # tsumi-avx512.exe, a dedicated AVX-512 Windows build
make debug              # tsumi-debug, with sanitizers
make featlib            # train/libfeat.so, the trainer's feature extraction (the engine's own code)
```

On Windows itself, build in an MSYS2 MINGW64 shell with `make` or `make universal`. MSVC is not supported
(the code uses GCC built-ins).

**CPU support.** Only the network code benefits measurably from newer instruction sets, so the universal
build compiles just that code four times — for SSE4.2, AVX2, AVX-512 and AVX-512 VNNI — and picks the best set
the CPU supports at startup (the banner says which). All four give identical results. `TSUMI_ISA=avx2` (or
`sse4.2`, `avx512`) forces a lower set for testing. A build for a specific CPU (`make`, `make windows-avx512`)
prints what is missing and exits on a CPU that lacks its instructions.

## The network

The network is Tsumi's only evaluation, and it is not part of the repository:

```bash
make net                # downloads the network this version is built for, and checks its hash
make checkpoint         # ... and its training checkpoint (.pt), to train from
```

Networks are named after their contents, `tsumi-<the first 12 hex digits of the file's SHA-256>.nnue`, and
every published one is an asset of the
[networks release](https://github.com/Myonpasu/tsumi/releases/tag/networks), with its training checkpoint
beside it. The engine's own network is `NNUE::DEFAULT_NET` in `src/nnue.h`; the release archives include
it. Tsumi loads it from its working directory or from the executable's directory, and the `EvalFile` option
names another file. Without a network the engine still answers `uci`, `perft` and the like, but exits when
asked to search.

## Rules

| Situation | Result |
|---|---|
| The side to move is checkmated | the side to move **wins** |
| Stalemate, the third occurrence of a position, the fifty-move rule | draw |
| Dead position: no sequence of moves can mate (K v K, KN v K, KB v K, only bishops left and all on one square colour) | draw |
| One side has only its king (a *bare king*) | drawn in practice: a bare king gives no check, so its opponent can never be mated; the bare king wins only if its opponent is left with nothing but mating moves |

The fifty-move rule draws once 100 half-moves have passed without a capture or a pawn move, unless the
100th half-move mates.

## Scores

Scores are from the side to move's point of view, and positive means winning, that is, heading for being
mated:

- `score cp 150`: +1.50, good for the engine. The evaluation is calibrated so that **+1.00 is a 50% chance of
  winning**, the rest being mostly draws (see *Output units* under [Training](training.md)).
- `score mate 3`: the engine can force its own mate in three moves (it wins).
- `score mate -2`: the engine will be forced to mate its opponent within two moves (it loses).

Internally the engine counts in millipawns and reports centipawns; the training data holds millipawns.

## Options and commands

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
`tsumi datagen` and `tsumi rescore` generate and relabel training data ([Training](training.md)).

**Playing and matches.** Any UCI GUI can run Tsumi, but GUIs score checkmate for the side that delivers it,
which is the wrong way round here. `tools/play.py` plays against the engine in the terminal, and
`tools/match.py` runs engine-against-engine matches under the selfmate rules
([Tools and tests](tools-and-tests.md)).
