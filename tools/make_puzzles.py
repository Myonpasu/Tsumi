#!/usr/bin/env python3
"""Generate verified selfmate puzzles with a brute-force solver built on python-chess.

A "selfmate in N" position is one where the side to move can force the opponent to checkmate it
within N of its own moves, whatever the opponent does.  The solver is independent from the engine,
so the puzzles double as an end-to-end test of the engine's rules and search.

Usage: python3 tools/make_puzzles.py [--count N] [--max-n 2] [--seed S] [--out tests/puzzles.epd]
"""
import argparse
import random
import time

import chess


def selfmate(board: chess.Board, n: int):
    """Return a move that forces selfmate in <= n moves for the side to move, or None."""
    for m in board.legal_moves:
        board.push(m)
        ok = _forced(board, n)
        board.pop()
        if ok:
            return m
    return None


def _forced(board: chess.Board, n: int) -> bool:
    """Opponent to move: every legal reply must checkmate us, or (if n > 1) leave us a selfmate in n-1."""
    if board.is_checkmate():      # we just mated the opponent: that is a loss for us
        return False
    replies = list(board.legal_moves)
    if not replies:               # stalemate
        return False
    for r in replies:
        board.push(r)
        if board.is_checkmate():  # we are checkmated: success along this branch
            board.pop()
            continue
        ok = n > 1 and not board.is_stalemate() and selfmate(board, n - 1) is not None
        board.pop()
        if not ok:
            return False
    return True


def random_position(rng: random.Random) -> chess.Board:
    """Random sparse position with both kings boxed in by their own pieces (where selfmates are plausible)."""
    corners = [chess.A1, chess.H1, chess.A8, chess.H8, chess.A2, chess.H2, chess.B1, chess.G1,
               chess.A7, chess.H7, chess.B8, chess.G8, chess.A4, chess.H5, chess.D1, chess.E8]
    while True:
        board = chess.Board(None)
        wk = rng.choice(corners)
        bk = rng.choice(corners)
        if chess.square_distance(wk, bk) < 3:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        for ksq, color in ((wk, chess.WHITE), (bk, chess.BLACK)):
            other = bk if color == chess.WHITE else wk
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]):
                if rng.random() < 0.6 and board.piece_at(sq) is None and chess.square_distance(sq, other) >= 2:
                    pt = rng.choice([chess.PAWN, chess.PAWN, chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                    if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                        pt = chess.KNIGHT
                    board.set_piece_at(sq, chess.Piece(pt, color))
        for _ in range(rng.randint(1, 4)):
            sq = rng.choice(list(chess.SQUARES))
            if board.piece_at(sq) is not None or chess.square_distance(sq, bk) < 1 or chess.square_distance(sq, wk) < 1:
                continue
            color = rng.random() < 0.5
            pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            board.set_piece_at(sq, chess.Piece(pt, color))
        board.turn = chess.WHITE
        if not board.is_valid() or board.is_game_over():
            continue
        return board


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--max-n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="tests/puzzles.epd")
    ap.add_argument("--time-limit", type=float, default=600)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    found = {n: [] for n in range(1, args.max_n + 1)}
    # The brute-force search for selfmates in 2 and 3 is only affordable when the side to move has few
    # legal moves; longer puzzles are skipped in busy positions.
    move_limit = {1: 256, 2: 40, 3: 25}
    t0 = time.time()
    tried = 0
    while sum(len(v) for v in found.values()) < args.count and time.time() - t0 < args.time_limit:
        board = random_position(rng)
        tried += 1
        n_moves = board.legal_moves.count()
        for n in range(1, args.max_n + 1):          # shortest selfmate first, so each puzzle is classified exactly
            if n_moves > move_limit.get(n, 25):
                break
            if selfmate(board, n) is not None:
                found[n].append(board.fen())
                break
    with open(args.out, "w") as f:
        f.writelines(f"{fen}; sm{n}\n" for n in sorted(found) for fen in found[n])
    print(f"tried {tried} positions in {time.time() - t0:.0f}s: " + ", ".join(f"sm{n}: {len(v)}" for n, v in found.items()))


if __name__ == "__main__":
    main()
