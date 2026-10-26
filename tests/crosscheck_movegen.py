#!/usr/bin/env python3
"""Cross-check Tsumi's move generator, check detection, FEN output and dead-position rule against python-chess
on random positions.   Usage: python3 tests/crosscheck_movegen.py [count] [seed] [engine]"""
import random
import subprocess
import sys

import chess

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
exe = sys.argv[3] if len(sys.argv) > 3 else "./tsumi"

boards = []
for _ in range(N):
    b = chess.Board()
    for _ in range(random.randint(0, 120)):
        moves = list(b.legal_moves)
        if not moves:
            break
        weights = [3 if (b.is_capture(m) or m.promotion or b.gives_check(m)) else 1 for m in moves]
        b.push(random.choices(moves, weights)[0])
        if b.is_game_over():
            break
    boards.append(b)

inp = "crosscheck\n" + "\n".join(b.fen() for b in boards) + "\n"
stdout = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout
out = [line for line in stdout.splitlines() if "|" in line or line == "badfen"]
assert len(out) == len(boards), (len(out), len(boards))
bad = 0
for b, line in zip(boards, out):
    if "MISMATCH" in line or "INCONSISTENT" in line or line == "badfen":
        print("ERR", b.fen(), line)
        bad += 1
        continue
    mv_s, fen_s, dead_s = [x.strip() for x in line.split("|")]
    mine = set(mv_s.split()) if mv_s else set()
    ref = {m.uci() + ("+" if b.gives_check(m) else "") for m in b.legal_moves}
    if mine != ref:
        print("MOVES MISMATCH", b.fen(), "only engine:", mine - ref, "only python-chess:", ref - mine)
        bad += 1
    if fen_s.split()[:3] != b.fen().split()[:3]:
        print("FEN MISMATCH", b.fen(), "->", fen_s)
        bad += 1
    if bool(int(dead_s)) != b.is_insufficient_material():
        print("DEAD MISMATCH", b.fen(), dead_s)
        bad += 1
print(f"checked {len(boards)} positions, {bad} problems")
sys.exit(1 if bad else 0)
