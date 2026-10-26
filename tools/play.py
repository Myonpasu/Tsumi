#!/usr/bin/env python3
"""Play selfmate chess against Tsumi in the terminal.

Remember: you are trying to get your own king checkmated, and so is the engine.  Whoever is checkmated wins;
stalemate and the usual draw rules are draws.

Usage: python3 tools/play.py [--engine ./tsumi] [--net nets/x.nnue] [--color white|black] [--movetime 1000]
Enter moves in UCI (e2e4, e7e8q) or SAN (Nf3, O-O).  Commands: 'undo', 'hint', 'fen', 'quit'.
"""
import argparse
import subprocess
import sys

import chess


class Engine:
    def __init__(self, path, options):
        try:
            self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            sys.exit(f"cannot run {path}: {e}")
        assert self.p.stdin is not None and self.p.stdout is not None
        self.stdin, self.stdout = self.p.stdin, self.p.stdout
        self.cmd("uci")
        lines = self.wait("uciok")
        for k, v in options.items():
            self.cmd(f"setoption name {k} value {v}")
        self.cmd("isready")
        lines += self.wait("readyok")
        # The engine says whether it could load its network, at start-up and again for an EvalFile option.
        net = [line for line in lines if line.startswith("info string") and ("network" in line or "EvalFile" in line)]
        if not net or "loaded network" not in net[-1]:
            self.quit()
            sys.exit(f"{path}: {net[-1][len('info string '):] if net else 'no network loaded'} (--net FILE)")

    def cmd(self, s):
        self.stdin.write(s + "\n")
        self.stdin.flush()

    def wait(self, token):
        lines = []
        while True:
            line = self.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated" + (f" after: {lines[-1]}" if lines else ""))
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    def think(self, board, movetime):
        """(best move, score) for the side to move; the score is ("cp" or "mate", value) or None."""
        moves = " ".join(m.uci() for m in board.move_stack)
        self.cmd(f"position fen {board.root().fen()}" + (f" moves {moves}" if moves else ""))
        self.cmd(f"go movetime {movetime}")
        lines = self.wait("bestmove")
        score = None
        for line in lines:
            toks = line.split()
            if toks[:2] == ["info", "depth"] and "score" in toks[:-2]:
                i = toks.index("score")
                score = (toks[i + 1], int(toks[i + 2]))
        return lines[-1].split()[1], score

    def quit(self):
        try:
            self.cmd("quit")
            self.p.wait(timeout=5)
        except (OSError, ValueError, subprocess.TimeoutExpired):   # the engine is gone already, or does not stop
            self.p.kill()


def describe(score, who):
    """A score for the side that searched, `who` ("the engine" or "you")."""
    if score is None:
        return ""
    kind, val = score
    if kind == "mate":
        return f"({who} can force your own mate in {val})" if val > 0 and who == "you" else \
               f"({who} expects to get mated in {val})" if val > 0 else \
               f"({who} will be forced to mate the opponent in {-val})"
    return f"(eval {val / 100:+.2f} for {who})"


def status(board):
    if board.is_checkmate():
        return f"Checkmate! {'White' if board.turn else 'Black'} is mated and therefore WINS."
    if board.is_stalemate():
        return "Stalemate: draw."
    if board.is_insufficient_material():
        return "Dead position: draw."
    if board.is_fifty_moves():
        return "Fifty-move rule: draw."
    if board.is_repetition(3):
        return "Threefold repetition: draw."
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./tsumi")
    ap.add_argument("--net", help="network file (the EvalFile option; default: the engine's tsumi.nnue)")
    ap.add_argument("--color", choices=["white", "black"], default="white")
    ap.add_argument("--movetime", type=int, default=1000, help="the engine's thinking time per move in ms (default 1000)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fen", help="start from this position")
    args = ap.parse_args()

    opts = {"Threads": args.threads, "Hash": 128}
    if args.net:
        opts["EvalFile"] = args.net
    eng = Engine(args.engine, opts)
    board = chess.Board(args.fen) if args.fen else chess.Board()
    human = chess.WHITE if args.color == "white" else chess.BLACK
    print(__doc__)
    try:
        while True:
            print(board.unicode(invert_color=True, borders=True) if sys.stdout.encoding and "UTF" in sys.stdout.encoding.upper() else board)
            print(f"FEN: {board.fen()}")
            end = status(board)
            if end:
                print(end)
                break
            if board.turn == human:
                try:
                    s = input("your move> ").strip()
                except EOFError:   # Ctrl-D
                    break
                if s in ("quit", "exit"):
                    break
                if s == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                    continue
                if s == "fen":
                    continue
                if s == "hint":
                    mv, sc = eng.think(board, args.movetime)
                    print(f"hint: {board.san(chess.Move.from_uci(mv))} {describe(sc, 'you')}")
                    continue
                try:
                    move = board.parse_san(s)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(s)
                    except ValueError:
                        move = None
                if move is None or move not in board.legal_moves:
                    print("illegal move")
                    continue
                board.push(move)
            else:
                mv, sc = eng.think(board, args.movetime)
                move = chess.Move.from_uci(mv)
                print(f"engine plays {board.san(move)} {describe(sc, 'the engine')}")
                board.push(move)
    except KeyboardInterrupt:   # Ctrl-C: leave quietly
        print()
    eng.quit()


if __name__ == "__main__":
    main()
