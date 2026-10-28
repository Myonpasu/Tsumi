#!/usr/bin/env python3
"""Write a book of starting positions for `tsumi datagen --book`, one FEN per line, drawn at random from another book.

    python3 tools/gen_book.py --count 10000 --base UHO_Lichess_4852_v1.epd.zip --out data/gen0/book.epd

The base book is a file with a FEN or an EPD line per position, or a PGN file of opening lines (each game played to
its end), either one plain or in a .zip, as the Stockfish opening books come
(https://github.com/official-stockfish/books).  Datagen makes its own random-play openings; the book is for the
games that cover the real opening.
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
    ap.add_argument("--count", type=int, default=10000, help="positions to write (default 10000)")
    ap.add_argument("--base", metavar="BOOK", required=True, help="the book to draw them from (FEN/EPD lines or PGN games, plain or zipped)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.count < 1:
        ap.error("--count must be at least 1")
    if not os.path.isfile(args.base):
        ap.error(f"--base: {args.base} not found")
    rng = random.Random(args.seed)
    boards, size, bad = draw_positions(args.base, args.count, rng)
    if not boards:
        raise SystemExit(f"{args.base} holds no position to start from (a FEN or an EPD line per position, or PGN games)")
    lines = [b.fen() for b in boards]
    drawn = f"{len(lines):,} positions drawn from {os.path.basename(args.base)} ({size})"
    if bad:
        drawn += f"; {bad:,} of the entries drawn held no position to start from and were left out"
    if len(lines) < args.count:   # a book with too few positions: they repeat
        drawn += f"; it has only {len(lines):,} position{'s' if len(lines) != 1 else ''}, so they repeat"
        lines += [rng.choice(lines) for _ in range(args.count - len(lines))]
        rng.shuffle(lines)
    with open(args.out + ".tmp", "w") as f:   # moved into place when complete
        f.writelines(line + "\n" for line in lines)
    os.replace(args.out + ".tmp", args.out)
    print(f"  {'book':<16}{args.out}: {drawn}")


if __name__ == "__main__":
    main()
