Tsumi is a UCI chess engine with no interface of its own: a chess GUI shows the board, and Tsumi plays and
analyses behind it. This page goes from a download to a game.

## Download

The [releases](https://github.com/Myonpasu/Tsumi/releases) have an archive for each system:

| | |
|---|---|
| `tsumi-<version>-windows-x86-64.zip` | Windows, 64-bit |
| `tsumi-<version>-linux-x86-64.tar.gz` | Linux, 64-bit, statically linked, so it runs on any distribution |

Each unpacks to a folder `tsumi` holding the executable and the network file (`tsumi-<hash>.nnue`), which the
engine loads from its own folder, so keep the two together. The executables run on any x86-64 CPU with
SSE4.2 and POPCNT (Intel since 2008, AMD since 2011) and use AVX2 or AVX-512 instructions when the CPU has
them, so there is one download per system. There is no macOS build. [Building](building.md) covers building
from source.

## Chess GUIs

Any UCI chess GUI can run Tsumi. Instructions for particular GUIs follow below.

Whatever the GUI, it plays ordinary chess, and so scores a checkmate as a win for the side that delivered it,
which in selfmate is the side that lost. The board, the legal moves, and the draws are the same as in chess, so
everything else works as usual, and only the announced winner is reversed. A GUI also does not adjudicate a
bare king as a draw. Tsumi plays such positions towards a stalemate.

Tsumi starts with a 64 MB hash table and one thread. The GUI's engine settings set `Hash` and `Threads`, and
[Options and commands](options-and-commands.md) lists every option.

### En Croissant

1. Open **Engines** in the sidebar and click **Add New**.
2. On the **Local** tab, click **Binary file**, select the Tsumi executable (`tsumi-windows-x86-64.exe` or
   `tsumi-linux-x86-64`), and click **Add**. The name is taken from the engine.
3. To play, choose **Play Chess** on the home page, pick Tsumi as the opponent, set the time control, and start
   the game. For analysis, enable Tsumi in the analysis board. Its settings are under **Engine Settings**, in
   the Engines tab or in the analysis board.

## Playing against Tsumi

You are trying to get your own king checkmated, and so is Tsumi. The game ends as in chess, and when it ends
in checkmate the GUI congratulates the loser.

## Reading Tsumi's scores

Tsumi's score is for the side to move, and positive is good for it, that is, heading for being mated. +1.00 is
calibrated to a 50% chance of winning, the rest being mostly draws. A mate score means the side to move can
force its own mate in that many moves, and a negative one means it will be forced to mate its opponent.

## The rules

| Situation | Result |
|---|---|
| The side to move is checkmated | the side to move **wins** |
| Stalemate, the third occurrence of a position, the fifty-move rule | draw |
| Dead position: no sequence of moves can mate (K v K, KN v K, KB v K, only bishops left and all on one square colour) | draw |
| One side has only its king (a *bare king*) | drawn in practice: a bare king gives no check, so its opponent can never be mated; the bare king wins only if its opponent is left with nothing but mating moves |

The fifty-move rule draws once 100 half-moves have passed without a capture or a pawn move, unless the
100th half-move mates.

## The tools in the repository

The repository's Python tools apply the selfmate rules themselves. `tools/play.py` plays against the engine in
the terminal, and `tools/match.py` runs engine-against-engine matches with the right winner and an Elo
estimate. They need Python 3 and python-chess (`pip install chess`), and [Tools and tests](tools-and-tests.md)
describes them.
