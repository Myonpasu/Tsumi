## UCI options

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

## Scores

Scores are from the side to move's point of view, and positive means winning, that is, heading for being
mated:

- `score cp 150`: +1.50, good for the engine. The evaluation is calibrated so that +1.00 is a 50% chance of
  winning, the rest being mostly draws (see *Output units* under [Training](training.md)).
- `score mate 3`: the engine can force its own mate in three moves (it wins).
- `score mate -2`: the engine will be forced to mate its opponent within two moves (it loses).

Internally the engine counts in millipawns and reports centipawns; the training data holds millipawns.

## Commands

Besides the UCI commands: `d` (show the board), `eval` (the network's evaluation of the position), `legal`
(the legal moves, `+` marking checks), `perft N`, `bench [depth]`, `test` (perft suite), `nnuetest [plies]`
(incremental accumulator updates compared with full recomputation), `wait` (block until the search ends, for
scripts). `go mate N` stops at a forced selfmate in N; `go searchmoves` is supported.

**Command line**: `tsumi bench 13`, `tsumi test`, or any UCI commands as arguments, run in order before the
engine exits: `tsumi "setoption name EvalFile value my.nnue" "position startpos" "go depth 12"`.
`tsumi datagen` and `tsumi rescore` generate and relabel training data ([Training](training.md)).
