#!/usr/bin/env python3
"""Run the engine on selfmate puzzles and verify its answers with the independent python-chess solver.

Each line of the EPD file is "<fen>; smN" meaning the side to move has a forced selfmate in N.
The engine must report a mate score of at most N and its move must be a verified key move.

Usage: python3 tools/puzzles.py [--engine ./tsumi] [--net nets/best.nnue] [--epd tests/puzzles.epd] [--movetime 2000]
"""
import argparse
import os
import re
import subprocess
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_puzzles import _forced


def engine_solve(engine, net, fen, movetime, depth):
    cmds = ["uci", "setoption name Hash value 64"] + ([f"setoption name EvalFile value {net}"] if net else [])
    cmds += ["ucinewgame", f"position fen {fen}"]
    cmds.append(f"go depth {depth}" if depth else f"go movetime {movetime}")
    cmds += ["wait", "quit"]
    # check=False: an engine that fails shows as an unsolved puzzle.
    out = subprocess.run([engine], input="\n".join(cmds) + "\n", capture_output=True, text=True, check=False).stdout
    if "no network loaded" in out:   # the engine searches with a network or not at all
        sys.exit(f"{engine}: no network loaded - pass --net FILE (or get the default one with make net)")
    mate = None
    for line in out.splitlines():
        if line.startswith("info depth") and (m := re.search(r"score mate (-?\d+)", line)):
            mate = int(m.group(1))
        elif line.startswith("info depth"):
            mate = None
    m = re.search(r"bestmove (\S+)", out)
    return (m.group(1) if m else None), mate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./tsumi")
    ap.add_argument("--net", help="network file (EvalFile option); default: the engine's own network")
    ap.add_argument("--epd", default="tests/puzzles.epd")
    ap.add_argument("--movetime", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=0)
    args = ap.parse_args()

    with open(args.epd) as f:
        lines = [line.strip() for line in f]
    total = ok = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fen, tag = [x.strip() for x in line.split(";")[:2]]
        n = int(tag[2:])
        total += 1
        move, mate = engine_solve(args.engine, args.net, fen, args.movetime, args.depth)
        board = chess.Board(fen)
        verdict = "FAIL"
        try:
            mv = chess.Move.from_uci(move) if move else None
        except ValueError:
            mv = None
        if mv is not None and mv in board.legal_moves and mate is not None and 0 < mate <= n:
            board.push(mv)
            if _forced(board, n):
                verdict = "ok"
        ok += verdict == "ok"
        print(f"{verdict:4s} {tag:4s} engine: {move} mate {mate}   {fen}")
    print(f"{ok}/{total} puzzles solved")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
