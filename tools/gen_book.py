#!/usr/bin/env python3
"""Write a book of starting positions for `tsumi datagen --book`, one FEN per line.

    python3 tools/gen_book.py --count 20000 --random-share 50 --out data/gen0/book.epd
    python3 tools/gen_book.py --count 20000 --random-share 50 --base UHO_Lichess_4852_v1.epd.zip --out data/gen0/book.epd

--random-share P makes P% of the lines positions reached by --plies-min to --plies-max random legal moves from the
start position.  The other lines are the start position itself, or with --base positions drawn at random from
another book: a file with a FEN or an EPD line per position, or a PGN file of opening lines (each game played to its
end), either one plain or in a .zip, as the Stockfish opening books come (https://github.com/official-stockfish/books).
Games from the start position or from book openings cover the real opening (datagen's random moves branch them);
games from random-play positions are unbalanced and cover the decisive phase, which games from the start position
mostly lack.
"""
import argparse
import io
import logging
import os
import random
import zipfile

import chess
import chess.pgn

logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)   # a game that does not parse is counted, not reported


def random_play_position(rng: random.Random, plies_min: int, plies_max: int) -> chess.Board:
    while True:
        b = chess.Board()
        for _ in range(rng.randint(plies_min, plies_max)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            return b


def book_name(path):
    """A book's file name: the path's own, or that of the one file inside a .zip (the Stockfish books come zipped)."""
    if not zipfile.is_zipfile(path):
        return os.path.basename(path)
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise SystemExit(f"{path}: a zipped book must hold one file, not {len(names)}")
        return names[0]


def book_text(path):
    """The lines of a book's text, from the file itself or from the one file inside a .zip."""
    if not zipfile.is_zipfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f
        return
    with zipfile.ZipFile(path) as archive, archive.open(book_name(path)) as raw:
        yield from io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def pgn_games(lines):
    """The games of a PGN file, each as its text.  A game ends where the next one's tag pairs begin after its moves,
    or, without tag pairs, at a blank line after its moves."""
    game, in_moves = [], False
    for line in lines:
        if in_moves and (line.startswith("[") or (not game[-1].strip() and line.strip())):
            yield "".join(game)
            game, in_moves = [], False
        if line.strip() and not line.startswith("["):
            in_moves = True
        game.append(line)
    if in_moves:
        yield "".join(game)


def book_entries(path):
    """A book's entries, one position each: the lines of a FEN or EPD file that are neither blank nor '#' comments,
    or the games of a PGN file (by its name, .pgn) as text."""
    lines = book_text(path)
    if is_pgn(path):
        yield from pgn_games(lines)
    else:
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def is_pgn(path):
    return book_name(path).lower().endswith(".pgn")


def parse_position(line):
    """The board of a FEN or an EPD line (EPD operations after the four position fields are ignored), or None if the
    line holds no position a game can start from."""
    fields = line.split()
    if len(fields) < 4:
        return None
    counters = fields[4:6] if len(fields) >= 6 and fields[4].isdigit() and fields[5].isdigit() else []
    try:
        board = chess.Board(" ".join(fields[:4] + counters))
    except ValueError:
        return None
    return board if board.is_valid() and not board.is_game_over() else None


def parse_game(text):
    """The board at the end of a PGN game's main line (an opening line played out), or None if the game does not
    parse or ends in a position no game can start from."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.errors:
        return None
    board = game.end().board()
    return board if board.is_valid() and not board.is_game_over() else None


def draw_positions(path, k, rng):
    """Up to k positions drawn at random, without replacement, from a book (see book_entries): (boards, a description
    of the book's size such as "2,632,036 lines" or "226,629 games", how many of the entries drawn held no usable
    position and were left out).  Fewer than k boards only if the book has too few.  One pass over the book, keeping
    only the entries drawn in memory."""
    want = k + k // 10 + 10   # a margin for entries that hold no usable position
    sample, n = [], 0
    for entry in book_entries(path):   # reservoir sampling
        if n < want:
            sample.append(entry)
        else:
            j = rng.randrange(n + 1)
            if j < want:
                sample[j] = entry
        n += 1
    rng.shuffle(sample)
    pgn = is_pgn(path)
    boards = [b for b in map(parse_game if pgn else parse_position, sample) if b is not None]
    return boards[:k], f"{n:,} {'games' if pgn else 'lines'}", len(sample) - len(boards)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=10000, help="lines to write (default 10000)")
    ap.add_argument("--random-share", type=int, default=100, help="percentage of the lines that are random-play positions (default 100)")
    ap.add_argument("--plies-min", type=int, default=8, help="random plies from the start position: uniform from this ... (default 8)")
    ap.add_argument("--plies-max", type=int, default=12, help="... to this (default 12)")
    ap.add_argument("--base", metavar="BOOK", help="draw the other lines from this book (FEN/EPD lines or PGN games, plain or zipped) "
                                                   "instead of using the start position")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 0 <= args.random_share <= 100 or not 0 <= args.plies_min <= args.plies_max:
        ap.error("--random-share must be 0-100, and 0 <= --plies-min <= --plies-max")
    if args.base and not os.path.isfile(args.base):
        ap.error(f"--base: {args.base} not found")
    rng = random.Random(args.seed)
    lines = []
    for _ in range(args.count):
        if rng.randrange(100) < args.random_share:
            lines.append(random_play_position(rng, args.plies_min, args.plies_max).fen())
        else:
            lines.append(None)   # the start position, or a position of the base book
    n_other = lines.count(None)
    n_random = args.count - n_other
    if n_other == 0:
        other, others = [], "no others"
    elif not args.base:
        other, others = [chess.Board().fen()] * n_other, f"{n_other:,} times the start position"
    else:
        boards, size, bad = draw_positions(args.base, n_other, rng)
        if not boards:
            raise SystemExit(f"{args.base} holds no position to start from (a FEN or an EPD line per position, or PGN games)")
        other = [b.fen() for b in boards]
        others = f"{n_other:,} positions drawn from {os.path.basename(args.base)} ({size})"
        if bad:
            others += f"; {bad:,} of the entries drawn held no position to start from and were left out"
        if len(other) < n_other:   # a book with too few positions: they repeat
            others += f"; it has only {len(other):,} position{'s' if len(other) != 1 else ''}, so they repeat"
            other += [rng.choice(other) for _ in range(n_other - len(other))]
            rng.shuffle(other)
    it = iter(other)
    lines = [next(it) if line is None else line for line in lines]
    with open(args.out + ".tmp", "w") as f:   # moved into place when complete
        f.writelines(line + "\n" for line in lines)
    os.replace(args.out + ".tmp", args.out)
    print(f"  {'book':<16}{args.out}: {args.count:,} lines, {n_random:,} random-play positions ({args.plies_min}-{args.plies_max} "
          f"plies) and {others}")


if __name__ == "__main__":
    main()
